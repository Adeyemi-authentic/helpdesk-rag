"""Tiny Voyage AI client over plain HTTP (no SDK).

Why HTTP instead of the `voyageai` package: the SDK has no working release for
Python 3.14 (this venv), so we talk to Voyage's REST endpoint directly with
`requests`. An "embedding API" is just: send text -> get back a list of numbers.

Reused by every Week-2 day. Import it from a day folder like this:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from voyage import embed, rerank

Reads VOYAGE_API_KEY from the project .env.
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

_BASE = "https://api.voyageai.com/v1"


def _key() -> str:
    key = os.getenv("VOYAGE_API_KEY")
    if not key:
        raise SystemExit("VOYAGE_API_KEY not found. Check your .env file.")
    return key


def _post(path, payload, timeout, max_retries=8):
    """POST with automatic backoff on 429 AND transient connection drops.

    The free tier is both rate-limited (429) and occasionally drops the
    connection mid-request (RemoteDisconnected / read timeout). Both are
    retryable with the same exponential backoff -- a flaky upstream must not
    crash the pipeline.
    """
    headers = {
        "Authorization": f"Bearer {_key()}",
        "Content-Type": "application/json",
    }
    for attempt in range(max_retries):
        last = attempt == max_retries - 1
        try:
            resp = requests.post(f"{_BASE}/{path}", headers=headers, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            if last:
                raise
            wait = min(2 ** attempt + 1, 60.0)
            print(f"  (connection error: {type(e).__name__}; waiting {wait:.0f}s then retrying...)")
            time.sleep(wait)
            continue
        if resp.status_code == 429 and not last:
            # Respect Retry-After if given, else exponential backoff capped at 60s.
            wait = min(float(resp.headers.get("Retry-After", 2 ** attempt + 1)), 60.0)
            print(f"  (rate limited; waiting {wait:.0f}s then retrying...)")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # exhausted retries


def embed(texts, input_type, model="voyage-3.5-lite", timeout=30):
    """Turn a list of strings into a list of vectors (each a list of floats).

    texts:      list[str] to embed.
    input_type: "document" for things you store/search, "query" for a user's
                question at search time. Voyage tunes the vector based on this.
    model:      embedding model id. Default voyage-3.5-lite (cheapest, 1024 dims).

    Returns (vectors, total_tokens).
    """
    if isinstance(texts, str):
        texts = [texts]

    body = _post(
        "embeddings",
        {"input": texts, "model": model, "input_type": input_type},
        timeout,
    )
    # data comes back in the same order we sent it; sort by index to be safe.
    rows = sorted(body["data"], key=lambda d: d["index"])
    vectors = [row["embedding"] for row in rows]
    total_tokens = body.get("usage", {}).get("total_tokens", 0)
    return vectors, total_tokens


def rerank(query, documents, model="rerank-2-lite", top_k=None, timeout=30):
    """Re-score a query against candidate documents (used on Day 6).

    Returns a list of dicts: {"index": i, "document": str, "relevance_score": float},
    best first. `index` points back into the original `documents` list.
    """
    payload = {"query": query, "documents": documents, "model": model}
    if top_k is not None:
        payload["top_k"] = top_k

    body = _post("rerank", payload, timeout)
    return sorted(body["data"], key=lambda d: -d["relevance_score"])
