"""Helpdesk RAG -- FastAPI service: hardened, streaming, cited answers.

Wraps the framework-free retrieval+generation engine (../rag) in a web product:
  GET  /                UI (static chat page)
  GET  /health          liveness probe (reports the active store)
  POST /chat            one-shot cited answer         (JSON, auth required)
  POST /chat/stream     token-by-token SSE + citations (auth required)

The RAG brain is unchanged; this is the body around it -- auth, rate limiting,
clean error JSON, CORS, streaming, and a UI.

The retrieval store is chosen by an env var, not hardcoded:
  ENGINE=qdrant  -> local Qdrant files in ../rag/index_data  (dev default)
  ENGINE=pg      -> pgvector over DATABASE_URL               (production)
A container's filesystem is ephemeral, so production points at an external
pgvector database; retrieval parity between the two stores is proven, so the
swap is invisible to everything above the store.

All config comes from env vars (secrets never baked into the image):
  API_KEY, ENGINE, DATABASE_URL, RATE_LIMIT, RATE_WINDOW, ALLOWED_ORIGINS,
  ANTHROPIC_API_KEY, VOYAGE_API_KEY.

Run:  uvicorn app:app --reload          (from the app/ folder)
"""

import json
import os
import pathlib
import sys
import time
from collections import deque
from contextlib import asynccontextmanager

import anthropic
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Wire in the retrieval+generation engine (../rag) ---------------------
HERE = pathlib.Path(__file__).resolve().parent                 # .../app
ROOT = HERE.parent                                             # repo root
RAG = ROOT / "rag"
sys.path.insert(0, str(RAG))
load_dotenv(ROOT / ".env")                                     # no-op in the container

from chat import (                                             # noqa: E402
    answer, build_documents, _parse,
    SYSTEM, MODEL, DONT_KNOW, THRESHOLD, TOP_K, COVERAGE_FLOOR,
)

# --- Pick the retrieval store from the environment ------------------------
ENGINE = os.environ.get("ENGINE", "qdrant").lower()
if ENGINE == "pg":
    from pg_engine import PgRetrievalEngine as Engine          # noqa: E402
else:
    from engine import RetrievalEngine as Engine               # noqa: E402


# --- Config (secrets from env) --------------------------------------------
API_KEY = os.environ["API_KEY"]
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "10"))
RATE_WINDOW = float(os.environ.get("RATE_WINDOW", "60"))
# Comma-separated in prod (e.g. your public URL); localhost is the dev default.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",") if o.strip()
]


# --- Pydantic schemas -----------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="The user's helpdesk question.")


class Citation(BaseModel):
    n: int
    title: str
    quote: str


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    gated: bool
    top_score: float
    coverage: float
    flagged: bool
    citations: list[Citation]


# --- Lock 1: API-key auth -------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthError(Exception):
    """Raised when a request has no valid API key."""


class RateLimitError(Exception):
    """Raised when a key exceeds its per-window quota."""


def require_api_key(key: str | None = Depends(_api_key_header)) -> str:
    import secrets
    if not key or not secrets.compare_digest(key, API_KEY):
        raise AuthError("Missing or invalid API key.")
    return key


# --- Lock 2: in-memory per-key rate limiter -------------------------------
# NOTE: correct for a single process. A multi-instance deploy would move this
# to a shared store (e.g. Redis) so the counter is shared across instances.
_hits: dict[str, deque[float]] = {}


def enforce_rate_limit(key: str) -> None:
    now = time.monotonic()
    bucket = _hits.setdefault(key, deque())
    while bucket and now - bucket[0] > RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT:
        raise RateLimitError(
            f"Rate limit exceeded: max {RATE_LIMIT} requests per "
            f"{int(RATE_WINDOW)}s."
        )
    bucket.append(now)


def guard(key: str = Depends(require_api_key)) -> str:
    """Combined gate: auth THEN rate limit."""
    enforce_rate_limit(key)
    return key


# --- App lifecycle --------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] retrieval store: {ENGINE}", file=sys.stderr)
    app.state.engine = Engine()
    app.state.client = anthropic.Anthropic()
    yield
    app.state.engine.close()


app = FastAPI(title="Helpdesk RAG", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# --- Lock 3: structured error handlers ------------------------------------
@app.exception_handler(AuthError)
async def _auth_handler(request: Request, exc: AuthError):
    return JSONResponse(status_code=401, content={"error": str(exc)})


@app.exception_handler(RateLimitError)
async def _rate_handler(request: Request, exc: RateLimitError):
    return JSONResponse(status_code=429, content={"error": str(exc)})


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
    return JSONResponse(status_code=500, content={"error": "Internal server error."})


# --- UI routes (unauthenticated: just the page; the API it calls is not) --
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "engine": ENGINE}


# --- Protected API --------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _key: str = Depends(guard)):
    result = answer(app.state.engine, app.state.client, req.question)
    return ChatResponse(
        answer=result["text"], refused=result["refused"], gated=result["gated"],
        top_score=result["top_score"], coverage=result["coverage"],
        flagged=result["flagged"],
        citations=[Citation(**c) for c in result["citations"]],
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def stream_pipeline(engine, client, query):
    passages = engine.search(query, k=TOP_K)
    top_score = passages[0][1] if passages else 0.0

    if top_score < THRESHOLD:
        yield _sse({"type": "token", "text": DONT_KNOW})
        yield _sse({"type": "done", "refused": True, "gated": True,
                    "top_score": top_score, "coverage": 0.0,
                    "flagged": False, "citations": []})
        return

    with client.messages.stream(
        model=MODEL, max_tokens=1024, system=SYSTEM,
        messages=[{
            "role": "user",
            "content": build_documents(passages) + [
                {"type": "text", "text": f"Question: {query}"}
            ],
        }],
    ) as stream:
        for text in stream.text_stream:
            yield _sse({"type": "token", "text": text})
        final = stream.get_final_message()

    text, citations, cited_chars, total_chars = _parse(final)
    refused = text.strip() == DONT_KNOW
    coverage = (cited_chars / total_chars) if total_chars else 0.0
    flagged = (not refused) and coverage < COVERAGE_FLOOR
    yield _sse({"type": "done", "refused": refused, "gated": False,
                "top_score": top_score, "coverage": coverage,
                "flagged": flagged, "citations": citations})


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, _key: str = Depends(guard)):
    gen = stream_pipeline(app.state.engine, app.state.client, req.question)
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
