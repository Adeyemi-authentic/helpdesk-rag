# SKILL: Building a grounded RAG chatbot you can deploy

A complete, reusable playbook for building a retrieval-augmented generation (RAG)
assistant that answers **only** from a document corpus, **cites** every claim with
a verifiable span, **refuses** what it cannot support, **streams** its answer, and
**ships** behind a hardened API to a public URL.

It is written as a transferable method: Part A-C document how the reference system
(an IT-helpdesk assistant) is built; Part D is the step-by-step guide to rebuild it
for a different niche (legal, medical, finance, product docs, internal wiki, etc.).

- Reference implementation: `helpdesk-rag/` (repo: Adeyemi-authentic/helpdesk-rag)
- Live example: https://helpdesk-rag.onrender.com
- Built framework-free (Anthropic SDK + hand-wired pipeline) so every mechanism is
  explicit and tunable. No LangChain/LlamaIndex in the hot path.

---

## 0. Mental model

A RAG chatbot is two separable things:

- **The brain** = retrieval + grounded generation. Given a question, find the right
  passages and write an answer that stays inside them. This is domain logic.
- **The body** = the product around the brain. An API, streaming, auth, a database,
  a container, a UI. This is infrastructure and is almost entirely niche-agnostic.

**The single most important design rule: the brain talks to its vector store through
one small interface, so the store is swappable.** You develop against a local store
and deploy against a managed one without touching anything above the storage layer.
Prove they are interchangeable with a parity test, then swap freely.

When you move to a new niche, the **body is ~90% reusable**; the work is in the
brain's inputs (the corpus), two or three tuned numbers, and the evaluation set.

---

## 1. Architecture (the full pipeline)

```
                     INGEST  (one-time, offline)
   docs/*  --> smart chunk --> embed (input_type=document) --> vector store
                                                              (+ BM25 keyword index)

                     QUERY  (per request)
   user turn (+ history)
      |
      v  CONTEXTUALIZE      rewrite an elliptical follow-up ("what about that?")
      |                     into a standalone query using the chat history
   standalone query
      |
      v  RETRIEVE           dense (embed query, cosine kNN) + BM25 keyword,
      |                     fused by Reciprocal Rank Fusion (RRF) -> top-N candidates
      v  RERANK             cross-encoder re-scores candidates -> top-k, best first
      |
      v  CONFIDENCE GATE    if top-1 rerank score < THRESHOLD, REFUSE here --
      |                     no generation call, no cost, no hallucination risk
      |
      +--(below threshold)--> grounded refusal ("I don't know ...")
      |
      v  GENERATE w/ CITATIONS   each passage is a citable `document` block; the
      |                          model API attaches the exact source span per claim
      v  COVERAGE CHECK          fraction of answer backed by a citation; flag a
      |                          confident answer that cites little (possible drift)
      v
   answer + verified sources  --stream (SSE)-->  UI renders tokens live + badges + sources
```

**Defense in depth.** No single guard is trusted alone:
1. the **gate** is a cheap, hard pre-filter for obviously out-of-scope questions;
2. the **grounded prompt** ("answer only from the documents, else say I don't know")
   is the soft backstop for anything the gate lets through;
3. **citations** make whatever survives auditable;
4. the **coverage check** flags a confident-but-thinly-grounded answer for review.

A single rerank-score threshold is blunt and uncalibrated, which is exactly why it
is one layer of four, not the whole defense.

---

## 2. Part A — The brain: retrieval + generation techniques

Each technique below lists WHAT, WHY, the KNOBS you tune per niche, and the code
pointer in the reference repo.

### A1. Smart chunking  (`rag/engine.py: smart_chunks`)
- WHAT: split each document into ~`CHUNK_SIZE`-char pieces, **paragraph-aware**
  (never mid-paragraph), carrying ~`OVERLAP` of the previous chunk into the next.
- WHY: embeddings represent a *span* of text; too big dilutes the signal, too small
  loses context. Paragraph boundaries keep ideas intact; overlap prevents a fact
  from being severed at a chunk edge.
- KNOBS: `CHUNK_SIZE=350`, `OVERLAP=0.15`. Dense/technical corpora (legal, medical)
  often want smaller chunks; narrative corpora tolerate larger.

### A2. Embeddings  (`rag/voyage_client.py: embed`)
- WHAT: turn text into a 1024-dim vector. Voyage `voyage-3.5-lite`.
- WHY: nearest vectors = semantically similar text, so you can retrieve by meaning,
  not keywords.
- CRITICAL DETAIL: pass `input_type="document"` when embedding the corpus and
  `input_type="query"` when embedding the user's question. The model tunes the
  vector differently for each side; mixing them silently degrades retrieval.
- KNOBS: model choice (bigger = better recall, more cost), vector dimension.

### A3. Vector store behind an interface  (`rag/engine.py` Qdrant, `rag/pg_engine.py` pgvector)
- WHAT: store `{id, source, chunk_text, embedding}` and support "give me the ids
  nearest this query vector." Two backends: **Qdrant** (local on-disk, zero-setup
  dev) and **pgvector** (Postgres extension, production).
- WHY THE INTERFACE: the engine calls `_dense()`, `_load()`, `build()`, `close()`.
  `PgRetrievalEngine(RetrievalEngine)` overrides only those; `hybrid()`, `search()`,
  and rerank are inherited unchanged. Swapping stores changes nothing downstream.
- pgvector specifics: `CREATE EXTENSION vector;`, an `embedding vector(1024)` column,
  cosine distance via the `<=>` operator (`similarity = 1 - distance`). Bind numpy
  float32 with `pgvector.psycopg.register_vector`.
- WHY TWO STORES: a container's filesystem is ephemeral, so local files reset on
  restart and cannot be shared across instances. The production store lives outside
  the container. `ENGINE` env var picks the backend at boot.

### A4. Hybrid retrieval + RRF  (`rag/engine.py: hybrid`)
- WHAT: run dense (vector) retrieval AND BM25 keyword retrieval, then fuse the two
  rankings with **Reciprocal Rank Fusion**: `score(id) = sum over lists of 1/(RRF_K + rank)`.
- WHY: dense catches paraphrase and meaning; BM25 catches exact tokens, codes, IDs,
  error strings (e.g. `0x0000011b`) that embeddings blur. RRF combines them without
  needing to calibrate incomparable score scales.
- KNOBS: `RRF_K=60` (higher = flatter fusion), candidate `top_n=10`.

### A5. Reranking  (`rag/voyage_client.py: rerank`)
- WHAT: a cross-encoder (`rerank-2-lite`) re-scores the query against each candidate
  jointly and returns the top-k, best first.
- WHY: first-stage retrieval is recall-oriented and cheap; the reranker is a precise
  second pass that reads query+doc together, sharply improving the top of the list.
  Its top-1 score is also the signal the gate keys on.
- KNOBS: `TOP_K=5` passages sent to generation.

### A6. Confidence gate  (`rag/chat.py: answer`, `THRESHOLD`)
- WHAT: if the top rerank score `< THRESHOLD`, refuse immediately, before any
  generation call.
- WHY: it is the cheapest possible refusal (no tokens spent) and removes the
  hallucination opportunity entirely for clearly out-of-scope questions.
- KNOBS: `THRESHOLD=0.55`. **This is the single most niche-specific number.** Tune it
  by measuring where in-scope and out-of-scope scores actually land on YOUR corpus
  (see D3). It is blunt and uncalibrated on purpose; the prompt is the backstop.

### A7. Grounded generation with native citations  (`rag/chat.py: build_documents`, `_parse`)
- WHAT: send each retrieved passage as a `{"type":"document", ...,
  "citations":{"enabled":True}}` block plus the question. The model API attaches, to
  each answer span, the **exact substring** of the source document it rests on.
- WHY: the model does not type `[1]` and hope you trust it. The citation is copied
  mechanically from the source, so it cannot be fabricated. Every claim is auditable.
- SYSTEM PROMPT: "Answer using ONLY the provided documents. Cite the document(s)
  behind each claim. If the documents do not contain the answer, reply with exactly:
  <DONT_KNOW sentinel>." The exact sentinel lets you detect a refusal by string match.
- GOTCHA: enabling citations is all-or-none across blocks and is **incompatible with
  structured-output `format`** (returns HTTP 400). Choose citations OR JSON mode.

### A8. Coverage / hallucination check  (`rag/chat.py: _parse`, `COVERAGE_FLOOR`)
- WHAT: coverage = fraction of answer characters that live in a text block carrying
  at least one citation. Flag when the model **claimed to answer** yet coverage is
  below `COVERAGE_FLOOR` (a refusal is allowed to be uncited).
- WHY: a confident answer with little grounding is the drift you most want to catch.
- KNOBS: `COVERAGE_FLOOR=0.50`.

### A9. Query contextualization (multi-turn)  (`rag/contextualize.py`)
- WHAT: before retrieval, rewrite the latest user turn into a standalone query using
  the chat history ("what about printers?" -> "how do I fix printer connection
  problems?").
- WHY: retrieval is single-shot and cannot see history; an elliptical follow-up
  embeds to nothing useful. Rewrite first, then retrieve on the standalone query.

---

## 3. Part B — The body: production infrastructure

All of this is niche-agnostic. Reuse it verbatim; only the brain it wraps changes.

### B1. FastAPI service  (`app/app.py`)
- Pydantic models validate the HTTP body (`ChatRequest(question: str, min_length=1)`)
  and shape the response. Bad input -> automatic HTTP 422.
- **Lifespan pattern**: build the engine and the model client ONCE at startup on
  `app.state`, close on shutdown. Never rebuild per request (retrieval caches + DB
  connections are expensive).
- Routes: `GET /` (UI), `GET /health` (liveness probe, unauthenticated),
  `POST /chat` (one-shot JSON), `POST /chat/stream` (SSE).

### B2. Streaming over Server-Sent Events  (`app/app.py: stream_pipeline`)
- A generator yields `data: <json>\n\n` frames. Token events stream text live; a
  final `done` event carries citations + scores (citations attach to the FINISHED
  message, so text streams first, sources land at the end).
- `StreamingResponse(media_type="text/event-stream")` with `Cache-Control: no-cache`
  and `X-Accel-Buffering: no` (defeats proxy buffering).
- Gate FIRST inside the generator: an out-of-scope question streams a single refusal
  token + done, with no model call.
- BROWSER SIDE: use `fetch()` + `response.body.getReader()` (ReadableStream), NOT
  `EventSource`. EventSource is GET-only and cannot set headers, so it can send
  neither a JSON body nor the `X-API-Key` header. Buffer bytes, split on the `\n\n`
  frame boundary (a frame can split across reads), parse each `data:` line.

### B3. Auth  (`app/app.py: require_api_key`)
- `APIKeyHeader(name="X-API-Key", auto_error=False)` -> a dependency that compares
  with `secrets.compare_digest` (constant-time). Missing/wrong -> 401 JSON.

### B4. Rate limiting  (`app/app.py: enforce_rate_limit`)
- In-memory `dict[key] -> deque[timestamps]`; drop timestamps outside the window;
  reject when the bucket is full. `RATE_LIMIT`/`RATE_WINDOW` from env.
- LIMIT: correct for a single process only. Multi-instance deploys must move this to
  a shared store (Redis) so the counter is shared. Note it in the code.

### B5. Structured errors  (`app/app.py` exception handlers)
- One `@app.exception_handler` per custom error -> always `{"error": "..."}` with the
  right status (401/429), and a catch-all `Exception` handler that logs the real
  error to stderr and returns a generic 500. **Never leak a stack trace to a client.**

### B6. CORS  (`CORSMiddleware`)
- Allow the frontend origin(s) from env. If the UI is served by the same app
  (same-origin), CORS never triggers -- keep the middleware anyway for a future
  split frontend.

### B7. Config via environment (12-factor)
- Every secret and every environment-specific value comes from an env var:
  `API_KEY, ENGINE, DATABASE_URL, RATE_LIMIT, RATE_WINDOW, ALLOWED_ORIGINS,
  ANTHROPIC_API_KEY, VOYAGE_API_KEY`. Same image, different behavior per environment.
  Secrets are NEVER baked into the code or the image. Ship a `.env.example`.

### B8. Docker  (`Dockerfile`)
- `python:3.14-slim` base. Install deps in their OWN layer BEFORE copying code, so a
  code edit reuses the cached pip layer (builds drop from minutes to seconds).
  Order the Dockerfile least-frequently-changed to most.
- `pip` from a **pinned** `requirements.txt` (`==`, not `>=`) for reproducible builds.
- Bind `--host 0.0.0.0` (reachable outside the container); read `--port ${PORT}`
  (the platform assigns it). `EXPOSE` the default. Set `ENV ENGINE=pg` for prod.
- `.dockerignore` keeps `.env`, `.venv`, `.git`, `__pycache__` out of the image.

### B9. docker-compose  (`docker-compose.yml`)
- Two services: `app` + `db` (pgvector image), wired by env vars, with a DB
  healthcheck so the app waits until Postgres accepts connections, a named volume for
  persistence, and a one-time idempotent ingest on boot. `docker compose up` = whole
  stack, no local Python.

### B10. Deploy (Render + managed Postgres/Neon)
- Push repo to GitHub. Render Web Service, Language = **Docker** (auto-detected from
  the Dockerfile). Set env secrets in the dashboard. Health Check Path `/health`.
- Managed pgvector (Neon free tier): run the one-time ingest against it locally
  (`ENGINE=pg python rag/pg_engine.py build`) before/at deploy; the container does not
  re-ingest.
- Free tiers sleep when idle -> first request after a nap takes ~30-60s (cold start).
  Fine for a portfolio demo; mention it.

---

## 4. Part C — Evaluation (do not skip)

Building a RAG is easy; knowing whether it works is the part most portfolios skip.
See `rag/evaluate.py`.

- Build a **labelled set**: ANSWERABLE questions (each with a `gold` substring only
  the correct answer would contain) and UNANSWERABLE questions the corpus cannot
  answer. Phrase them WITHOUT copying the document wording.
- Three metrics:
  1. **Answer correctness** (answerable) = answered AND `gold in answer`.
  2. **Citation accuracy** = every cited span is a genuine substring of the document
     it is attributed to. ~100% by construction (the point: auditable, not trusted).
  3. **Refusal accuracy** (unanswerable) = correctly declined.
  Also report **false refusals** (answerable questions wrongly declined).
- **Cache** each question's result (`eval_cache.json`) so a re-run resumes instead of
  re-spending API calls; delete the cache to re-evaluate.
- This is a deterministic gold-substring + structural eval -- cheap, repeatable,
  honest about limits. LLM-as-judge scoring of answer *quality* is a later add.
- Reference numbers: answer 6/6, citation 18/18, refusal 5/5, false refusals 0.

---

## 5. Part D — Reuse playbook: rebuild for a new niche

This is the transfer. Assume you keep the entire `app/` body and the `rag/` engine
mechanics; you change the corpus, a few numbers, the prompt framing, and the eval set.

### D1. What stays vs what changes

| Component | New niche? | Notes |
|-----------|-----------|-------|
| FastAPI service, streaming, auth, rate limit, errors, CORS | STAYS | verbatim |
| Docker, compose, deploy, env config | STAYS | verbatim |
| Chunking, hybrid retrieval, RRF, reranking, gate, citations, coverage | STAYS | same mechanics |
| The **corpus** (`rag/docs/`) | CHANGES | your niche's documents |
| `CHUNK_SIZE` / `OVERLAP` | MAYBE | retune for dense vs narrative text |
| `THRESHOLD` (gate) | CHANGES | re-measure on the new corpus (D3) |
| `SYSTEM` prompt (domain framing + refusal sentinel) | CHANGES | reword for the domain |
| The **eval set** (answerable/unanswerable + gold substrings) | CHANGES | rewrite for the domain |
| Embedding / rerank model choice | MAYBE | upgrade for hard domains |

### D2. Step by step
1. **Swap the corpus.** Drop the new documents into `rag/docs/` (convert PDFs/HTML to
   clean text first; garbage in, garbage out). Keep files reasonably topical per file.
2. **Rebuild the index.** `python rag/engine.py build` (local) and, for prod,
   `ENGINE=pg python rag/pg_engine.py build` against your managed Postgres.
3. **Retune the gate (D3).** Measure in-scope vs out-of-scope score bands; set
   `THRESHOLD` in the gap.
4. **Reword the system prompt.** State the domain and role ("You are a <domain>
   assistant. Answer ONLY from the provided documents..."), keep the exact refusal
   sentinel so refusal detection by string-match still works.
5. **Rewrite the eval set.** 5-10 answerable questions with gold substrings, 5+
   unanswerable ones. Delete `eval_cache.json`. Run `python rag/evaluate.py`.
6. **Adjust chunking if needed.** If retrieval misses, try smaller chunks for dense
   text or larger for narrative; rebuild and re-measure.
7. **Deploy** exactly as in B10 (new repo, new env secrets, same Dockerfile).

### D3. Tuning the gate empirically
Run a batch of clearly in-scope and clearly out-of-scope questions through
`engine.search()` and record the top rerank score for each. In-scope scores cluster
high, out-of-scope cluster low; set `THRESHOLD` in the gap between the bands. In the
reference build, out-of-scope topped ~0.53 and worst in-scope was ~0.58, so 0.55
sits in the gap -- but note the margin is thin, which is why the prompt refusal stays
as a backstop. Expect to re-tune per corpus; do not carry a threshold across domains.

### D4. Domain examples and what to watch

| Niche | Corpus | Watch out for |
|-------|--------|---------------|
| Legal / contracts | clauses, policies, statutes | small chunks (dense text); citations are a compliance feature; refusal must be strict -- a wrong legal answer is worse than none |
| Medical / clinical guidelines | care protocols, drug info | refusal + coverage are safety-critical; add human-in-the-loop; never present as advice |
| Customer support (product X) | help center, manuals, release notes | keep the index fresh (docs change); BM25 helps with error codes/SKUs |
| Internal wiki / onboarding | Confluence/Notion exports | access control matters -- who can see which docs; auth the USER, not just the machine |
| Financial filings / research | 10-Ks, reports | numbers and tables need careful chunking; cite exact figures; watch stale data |
| Developer docs / API reference | markdown docs, code | larger chunks for prose, exact-match (BM25) for symbol names |

### D5. Reuse checklist
- [ ] New corpus cleaned to text and dropped in `rag/docs/`
- [ ] Local index built; prod pgvector index built
- [ ] `THRESHOLD` re-measured and set for the new corpus
- [ ] System prompt reworded for the domain (refusal sentinel unchanged)
- [ ] Eval set rewritten; three metrics measured and recorded in the README
- [ ] Env secrets set; deployed; `/health` green on prod store
- [ ] README updated: architecture diagram, live URL, eval numbers

---

## 6. Part E — Pitfalls and lessons (from the real build)

- **Native Citations are incompatible with structured-output `format`** (HTTP 400).
  Pick one. For a grounded assistant, citations win.
- **EventSource cannot do auth or POST a body** -- authenticated SSE must use
  `fetch()` + ReadableStream.
- **The in-memory rate limiter has a check-then-append race** under truly concurrent
  fire (threads all read the bucket below the limit). Correct for sequential traffic;
  use Redis for multi-instance or high concurrency.
- **Rerank scores vary run to run**, so a question sitting near `THRESHOLD` can flip
  gated/answered between runs. This is why the gate is one layer, not the verdict.
- **Free-tier embedding/rerank APIs rate-limit (429) and drop connections** -- wrap
  every call in exponential backoff that retries both 429 and connection errors
  (respect `Retry-After`). See `voyage_client._post`.
- **Managed DBs have occasional DNS blips and cold starts** -- set a `connect_timeout`
  so a blip fails fast instead of hanging; retry once.
- **Ephemeral container filesystems** mean a file-based vector store resets on restart
  -- use an external DB in production (this is why the store is swappable).
- **Load `.env` explicitly from a known path**, not by walking up from the current
  working directory -- the CWD-walk is fragile in Docker/deploy.
- **Never commit `.env`** -- gitignore it, ship `.env.example`, verify no secret is
  staged before the first push.
- **Windows console is cp1252** -- avoid em-dashes and non-ASCII in stdout prints.

---

## 7. Part F — Stack, models, cost

- Generation: **Claude `claude-haiku-4-5`** with native Citations (cheap, fast, good
  enough for grounded extraction; upgrade to a larger Claude model for harder
  reasoning). Use the latest available model IDs at build time.
- Embeddings: **Voyage `voyage-3.5-lite`** (1024 dims), `input_type` document/query.
- Reranker: **Voyage `rerank-2-lite`** cross-encoder.
- Vector store: **pgvector** (prod) / **Qdrant** (local); keyword index: **rank-bm25**.
- Serving: **FastAPI + SSE**, **Docker**, **Render** (or any Docker host), **Neon**
  (managed Postgres) or any pgvector-capable Postgres.
- Cost: Haiku generation + cheap Voyage models over a small corpus stays in cents.
  Managed free tiers keep hosting near zero while learning.

---

## 8. Part G — File map and quickstart

```
app/
  app.py            FastAPI service: routes, auth, rate limit, errors, CORS, SSE
  static/index.html single-file chat UI (fetch + ReadableStream, no framework)
rag/
  engine.py         RetrievalEngine (Qdrant): chunk, embed, hybrid, RRF, rerank, search
  pg_engine.py      PgRetrievalEngine(RetrievalEngine): same, pgvector backend
  chat.py           answer(): gate -> generate w/ citations -> coverage; system prompt
  contextualize.py  rewrite a follow-up into a standalone query (multi-turn)
  voyage_client.py  embed() + rerank() over HTTP, with 429/connection backoff
  evaluate.py       labelled eval set + the three accuracy metrics
  docs/             the corpus (swap this per niche)
Dockerfile          image: slim base, cached deps layer, ENGINE=pg, uvicorn on $PORT
docker-compose.yml  app + pgvector db, healthcheck, one-command full stack
requirements.txt    pinned deps (== for reproducible builds)
.env.example        every required env var, documented
```

Quickstart (local, Qdrant):
```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY, VOYAGE_API_KEY, API_KEY
python rag/engine.py build    # chunk + embed + store docs/ (once)
uvicorn app:app --reload --app-dir app
python rag/evaluate.py        # reproduce the eval numbers
```

Full stack in Docker:
```bash
docker compose up --build     # app + pgvector, serves at http://localhost:8000
```

---

## 9. One-paragraph summary to reuse

Build the brain framework-free so every step is visible: paragraph-aware chunking,
Voyage embeddings, a vector store behind a small swappable interface, hybrid dense +
BM25 retrieval fused by RRF, a cross-encoder reranker, a cheap confidence gate that
refuses before spending a generation call, grounded generation with native citations
that copy the exact source span, and a coverage check. Wrap it in a niche-agnostic
body: a FastAPI service with Pydantic validation, SSE streaming, API-key auth,
rate limiting, clean error JSON, CORS, env-based config, Docker, and a managed
pgvector store, deployed to a public URL. Measure it on a labelled answerable and
unanswerable set (answer correctness, citation accuracy, refusal accuracy). To move
to a new niche, swap the corpus, retune the gate threshold, reword the system prompt,
rewrite the eval set, and redeploy -- the body and the retrieval mechanics carry over
unchanged.
