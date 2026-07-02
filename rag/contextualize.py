"""Day 6 capstone kickoff: query contextualization (the query-rewriter).

A conversational RAG breaks single-shot retrieval. A follow-up like "what about
printers?" or "the error is 0x0000011b" is meaningless to embed on its own -- the
retriever can't see the chat history. So BEFORE retrieval we rewrite the latest
user turn into a standalone query using the prior turns, then retrieve on that.

    history + "what about printers?"  --rewrite-->  "how do I fix printer
                                                     connection problems?"

This stub is the first piece of the Day-7 capstone. We prove it matters by
comparing what gets retrieved for the RAW follow-up vs the REWRITTEN query.
"""

import pathlib
import sys

import anthropic
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
sys.path.insert(0, str(HERE))
from engine import RetrievalEngine                            # noqa: E402

MODEL = "claude-haiku-4-5"

REWRITE_SYSTEM = (
    "You rewrite a user's latest message into a standalone search query for a "
    "document-retrieval system. Resolve references (pronouns, 'that', 'what about X', "
    "error codes mentioned earlier) using the chat history. Capture the user's CURRENT "
    "intent: if the latest message switches topic, base the query on the NEW topic and "
    "drop the previous one. If the latest message is already a complete standalone "
    "question, return it unchanged. Output ONLY the rewritten query text -- no preamble, "
    "no quotes, no explanation."
)


def contextualize(client, history, latest):
    """history = [(role, text), ...] prior turns; latest = newest user message."""
    if not history:
        return latest                                         # nothing to resolve
    convo = "\n".join(f"{role}: {text}" for role, text in history)
    user = f"Chat history:\n{convo}\n\nLatest user message: {latest}\n\nStandalone query:"
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=128, system=REWRITE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        rewritten = "".join(b.text for b in resp.content if b.type == "text").strip()
        return rewritten or latest                            # never return empty
    except anthropic.APIError as e:
        # Degrade gracefully: a failed rewrite must not break the chat. Fall back
        # to the raw query -- still retrieves fine for content-bearing follow-ups.
        print(f"  (rewrite failed: {type(e).__name__}; using raw query)")
        return latest


def top_hit(engine, query):
    hits = engine.search(query, k=1)
    if not hits:
        return ("(none)", 0.0)
    _pid, score, source, _text = hits[0]
    return (source, score)


def main():
    engine = RetrievalEngine()
    client = anthropic.Anthropic()
    # A simulated multi-turn conversation (assistant turns kept short for context).
    history = [
        ("user", "the shared office printer won't connect, it shows an error"),
        ("assistant", "That's error 0x0000011b, caused by a recent Windows security "
                      "update. Install the latest Windows updates and restart."),
    ]
    followups = [
        "what if that doesn't fix it?",          # CONTENTLESS -- 'that' = the update fix
        "what about my vpn dropping at a hotel?", # topic SWITCH (has content)
        "how do I unlock my account after too many wrong passwords",  # already standalone
    ]
    try:
        for latest in followups:
            standalone = contextualize(client, history, latest)
            raw_src, raw_sc = top_hit(engine, latest)          # retrieve on raw follow-up
            new_src, new_sc = top_hit(engine, standalone)      # retrieve on rewritten query
            print("=" * 70)
            print(f"follow-up : {latest!r}")
            print(f"rewritten : {standalone!r}")
            print(f"  retrieval on RAW       -> {raw_src} ({raw_sc:0.3f})")
            print(f"  retrieval on REWRITTEN -> {new_src} ({new_sc:0.3f})")
            # extend history so later follow-ups can resolve against earlier ones
            history.append(("user", latest))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
