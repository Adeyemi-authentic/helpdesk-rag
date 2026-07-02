# Helpdesk RAG -- a deployed, grounded document-chat assistant

A web app that answers questions over a document corpus (an IT helpdesk
knowledge base) and does the three things most demo RAGs skip:

1. **Answers only from the documents**, with **verifiable citations** -- every
   claim links to the exact span of the source it rests on, copied mechanically
   by the model API, not typed by the model, so it cannot be faked.
2. **Refuses cleanly** when the documents do not cover the question, instead of
   inventing a plausible-sounding answer.
3. **Streams** the answer token-by-token over Server-Sent Events, then renders
   the sources.

The retrieval and generation engine is **framework-free** (Anthropic SDK + a
hand-wired pipeline) so every mechanism is explicit and tunable. It is wrapped
in a production FastAPI service (auth, rate limiting, structured errors, CORS),
backed by **pgvector**, packaged with **Docker**, and deployed to a public URL.

- **Live demo:** https://helpdesk-rag.onrender.com
- **Demo video (2 min):** <ADD_VIDEO_LINK>

> The live demo runs on a free tier that sleeps after ~15 minutes idle, so the
> first request may take ~30-60s to wake the service. Ask a question in the UI
> with the API key to see a cited answer stream in.

## Architecture

```
                        INGEST (one-time)
   docs/*.txt --> smart chunk --> embed (Voyage) --> pgvector store
                                                     (+ BM25 keyword index)

                        QUERY (per request)
   Browser UI
      |  POST /chat/stream  (X-API-Key)
      v
   FastAPI service ......... auth -> rate limit -> validate (Pydantic)
      |
      v  RETRIEVE + RERANK   hybrid (dense + BM25, fused by RRF) -> top-10,
      |                      then a cross-encoder reranker -> top-5
      v  CONFIDENCE GATE     if the #1 passage scores below threshold,
      |                      REFUSE here -- no generation call, no cost
      |
      +--(below threshold)--> grounded refusal streamed back ("I don't know ...")
      |
      v  GENERATE w/ CITATIONS  each passage is a citable document block; the
      |                         API attaches the exact source span behind each claim
      v  STREAM (SSE)           tokens streamed live; a final event carries
      |                         top_score, coverage, and the numbered sources
      v
   Browser UI  renders streamed answer + grounded/refused/low-coverage badges
               + a numbered Sources list (title + verified quote)
```

**Defense in depth.** No single guard is trusted alone: the **gate** is a cheap
hard pre-filter for out-of-scope questions; the **prompt** ("answer only from the
documents, else say I don't know") is the soft backstop; **citations** make
whatever survives auditable. A single rerank-score threshold is blunt, which is
exactly why it is one layer of three.

## Results

`rag/evaluate.py` scores the three behaviours that matter, on a labelled set of
**answerable** and **unanswerable** questions (phrased without copying the
document wording):

| metric                              | result           |
|-------------------------------------|------------------|
| answer correctness (answerable)     | **6/6 = 100%**   |
| citation accuracy (verified spans)  | **18/18 = 100%** |
| refusal accuracy (unanswerable)     | **5/5 = 100%**   |
| false refusals (answerable refused) | 0/6              |

**Reading the numbers honestly.** Answer correctness = answered *and* the answer
contains the known-correct fact (a gold substring only the right answer would
mention). Citation accuracy is ~100% *by construction* -- native Citations copies
each span from the source, so a cited quote is always a real substring of its
document; that is the point, the answer is auditable, not taken on trust.
Refusal accuracy = the unanswerable questions were declined; all five were caught
by the gate alone (top score 0.385-0.531, below the 0.55 threshold) so no
generation call was spent. "What is the capital of France" scored 0.531 -- only
0.02 under the gate, which is why the prompt-based refusal is kept as a backstop.
This is a deterministic gold-substring + structural eval; full LLM-as-judge
scoring of answer *quality* is a later step.

## Vector store: swappable behind one interface

The retrieval engine calls its store through a small interface, so the backend
is a drop-in choice made by the `ENGINE` env var:

- `ENGINE=qdrant` -- local on-disk Qdrant files (`rag/index_data/`), for dev.
- `ENGINE=pg` -- pgvector over `DATABASE_URL`, for production (a container's
  filesystem is ephemeral, so the store must live outside it).

The two backends were checked for **retrieval parity** (identical top passages
and scores on the same queries), so switching stores changes nothing above the
store.

## Run it

### Local (Qdrant, no database needed)

```bash
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY, VOYAGE_API_KEY, API_KEY
python rag/engine.py build  # chunk + embed + store docs/ into a local index (once)
uvicorn app:app --reload --app-dir app
```

Open http://localhost:8000, enter your `API_KEY`, and ask a helpdesk question.

Reproduce the eval numbers (cached results included; delete `rag/eval_cache.json`
to re-run against the APIs):

```bash
python rag/evaluate.py
```

### Full stack in Docker (app + pgvector)

```bash
cp .env.example .env        # fill in the model keys and API_KEY
docker compose up --build
```

Ingests the corpus into pgvector on first boot, then serves at
http://localhost:8000.

## Deploy (Render)

1. Push this repo to GitHub.
2. Create a managed **pgvector**-capable Postgres (this project uses Neon) and
   run the one-time ingest against it: set `DATABASE_URL` + `ENGINE=pg` locally
   and run `python rag/pg_engine.py build`.
3. On Render, create a **Web Service** from the repo (Docker runtime; the
   `Dockerfile` sets `ENGINE=pg`). Add environment secrets: `ANTHROPIC_API_KEY`,
   `VOYAGE_API_KEY`, `DATABASE_URL`, `API_KEY`, and `ALLOWED_ORIGINS` (your
   Render URL). Render injects `PORT`.
4. Deploy. Health check path: `/health`.

## API

| method | path           | auth | description                                   |
|--------|----------------|------|-----------------------------------------------|
| GET    | `/`            | no   | chat UI                                        |
| GET    | `/health`      | no   | liveness probe; reports the active store       |
| POST   | `/chat`        | yes  | one-shot cited answer (JSON)                    |
| POST   | `/chat/stream` | yes  | token-by-token SSE stream + a final sources event |

Protected routes require the `X-API-Key` header. Bodies are validated by
Pydantic (`{"question": "..."}`, non-empty). Errors return clean JSON
(`{"error": "..."}`) with the right status code (401 / 422 / 429 / 500) -- never
a stack trace.

## Stack

Claude (`claude-haiku-4-5`) for generation with native Citations · Voyage AI
(`voyage-3.5-lite` embeddings + `rerank-2-lite` cross-encoder) · pgvector /
Qdrant vector store · rank-bm25 keyword index · FastAPI + Server-Sent Events ·
Docker · Render. No LLM framework -- the RAG mechanics are wired by hand.

## Layout

```
app/   FastAPI service (app.py) + static chat UI (static/index.html)
rag/   the engine: chunk, embed, hybrid retrieve, rerank, gate, generate, cite
       engine.py (Qdrant) · pg_engine.py (pgvector) · chat.py · contextualize.py
       · voyage_client.py · evaluate.py · docs/ (the corpus)
Dockerfile · docker-compose.yml · requirements.txt
```
