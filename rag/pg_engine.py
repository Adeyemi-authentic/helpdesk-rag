"""Week-4 Day 3: the same retrieval engine, with pgvector (Neon) as the store.

The point of this file is what it does NOT contain. The Week-3 engine's
hybrid fusion, BM25, RRF, and reranking are inherited UNCHANGED. We override
only the three methods that touch storage:

    __init__  -- open a Postgres/pgvector connection instead of Qdrant
    build     -- create the table + ingest chunks into pgvector
    _load     -- read all rows back (for the BM25 cache + id->text/source maps)
    _dense    -- nearest-neighbour search with the `<=>` cosine operator
    close     -- close the connection

Everything above the store (hybrid(), search(), rerank) calls these methods
through the same interface, so it neither knows nor cares that Qdrant became
Postgres. That is the whole lesson: a clean boundary lets you replace the
database without touching the system built on top of it.

    python pg_engine.py build          # create table + ingest into Neon (once)
    python pg_engine.py "my question"  # search via pgvector and print top passages
"""

import os
import pathlib
import sys

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from rank_bm25 import BM25Okapi

HERE = pathlib.Path(__file__).resolve().parent                 # ...\rag
ROOT = HERE.parent                                             # repo root
sys.path.insert(0, str(HERE))
load_dotenv(ROOT / ".env")

# Reuse the EXACT retrieval engine + its ingest helpers. We subclass the engine
# and reuse its chunking, tokenizer, dimensions, and docs folder verbatim, so
# the chunks here are byte-identical to the ones Qdrant holds -> fair parity.
from engine import (                                            # noqa: E402
    RetrievalEngine, smart_chunks, tokenize,
    DOCS_DIR, VECTOR_SIZE,
)
from voyage_client import embed                                 # noqa: E402

TABLE = "chunks"


class PgRetrievalEngine(RetrievalEngine):
    """RetrievalEngine backed by pgvector instead of Qdrant.

    Only storage methods are overridden; hybrid()/search()/rerank are inherited.
    """

    def __init__(self, dsn=None):
        # connect_timeout so a DNS/network blip fails fast instead of hanging
        # (Neon is remote; the free tier occasionally drops a lookup).
        self.conn = psycopg.connect(dsn or os.environ["DATABASE_URL"], connect_timeout=15)
        # The `vector` type must exist before we can register its adapter.
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.commit()
        register_vector(self.conn)          # lets psycopg bind numpy arrays as vectors
        self.collection = TABLE
        # same corpus-cache slots the parent uses (filled lazily by _load)
        self._ids = self._sources = self._texts = self._bm25 = None

    def close(self):
        self.conn.close()

    # ingest --------------------------------------------------------------
    def build(self, docs_dir=DOCS_DIR):
        """Create the chunks table and ingest every .txt in docs_dir. Idempotent.

        Chunks are produced by the SAME smart_chunks() Qdrant used and inserted
        with explicit 0-based ids in the same order, so ids line up 1:1 across
        both stores -- which makes the parity check apples-to-apples.
        """
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    id          INT PRIMARY KEY,
                    source      TEXT NOT NULL,
                    chunk_text  TEXT NOT NULL,
                    embedding   vector({VECTOR_SIZE})
                )
            """)
            cur.execute(f"SELECT count(*) FROM {TABLE}")
            existing = cur.fetchone()[0]
            if existing:
                print(f"Index already built: '{TABLE}' has {existing} chunks.")
                self.conn.commit()
                return

            records = []   # (source, chunk_text), same as the Qdrant build
            for path in sorted(docs_dir.glob("*.txt")):
                for chunk in smart_chunks(path.read_text(encoding="utf-8")):
                    records.append((path.name, chunk))

            texts = [c for _s, c in records]
            print(f"Embedding {len(texts)} chunks from {len(list(docs_dir.glob('*.txt')))} docs...")
            vectors, tokens = embed(texts, input_type="document")
            print(f"  embedded ({tokens} tokens).")

            for i, ((source, chunk), vec) in enumerate(zip(records, vectors)):
                cur.execute(
                    f"INSERT INTO {TABLE} (id, source, chunk_text, embedding) VALUES (%s, %s, %s, %s)",
                    (i, source, chunk, np.array(vec, dtype=np.float32)),
                )
        self.conn.commit()
        print(f"Stored {len(records)} chunks in pgvector table '{TABLE}'.")
        # NOTE: no ANN index (e.g. HNSW) on purpose -- with ~50 chunks an exact
        # scan is instant AND gives EXACT nearest neighbours, which is what the
        # parity test needs. A production corpus would add an HNSW index here.

    # corpus cache --------------------------------------------------------
    def _load(self):
        """Pull all rows once: id->source, id->text, and the BM25 index.

        Identical role to the parent's Qdrant scroll; only the source changes.
        """
        if self._ids is not None:
            return
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id, source, chunk_text FROM {TABLE} ORDER BY id")
            rows = cur.fetchall()
        self._ids = [r[0] for r in rows]
        self._sources = {r[0]: r[1] for r in rows}
        self._texts = {r[0]: r[2] for r in rows}
        self._bm25 = BM25Okapi([tokenize(self._texts[i]) for i in self._ids])

    # dense retrieval -----------------------------------------------------
    def _dense(self, query):
        """Embed the query and rank ALL chunks by cosine distance via `<=>`.

        Returns every id best-first (the parent's hybrid() needs the full
        ranking to fuse with BM25 via RRF). `<=>` = cosine distance, so the
        smallest distance = most similar, matching Qdrant's COSINE metric.
        """
        qv, _ = embed([query], input_type="query")
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id FROM {TABLE} ORDER BY embedding <=> %s",
                (np.array(qv[0], dtype=np.float32),),
            )
            return [r[0] for r in cur.fetchall()]


def _flat(text, n=110):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


def main():
    args = sys.argv[1:]
    engine = PgRetrievalEngine()
    try:
        if not args or args[0] == "build":
            engine.build()
            if not args:
                print('\nUsage: python pg_engine.py "your question"')
            return
        query = " ".join(args)
        print(f'Query: "{query}"  (via pgvector)\n')
        for rank, (pid, score, source, text) in enumerate(engine.search(query), 1):
            print(f"#{rank}  rel={score:0.4f}  [{source}]")
            print(f"    {_flat(text)}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
