"""Week-3 capstone: a conversational "chat with your documents" RAG.

This wires together every piece built across Week 3 into one pipeline that
holds a multi-turn conversation and answers ONLY from a document corpus,
with verifiable citations -- or refuses cleanly when the docs don't cover it.

    user turn (+ history)
        |
        v  CONTEXTUALIZE (Day 6) -- rewrite an elliptical follow-up
        |                           into a standalone search query
    standalone query
        |
        v  RETRIEVE + RERANK (Week-2 engine) -- hybrid (dense+BM25+RRF)
        |                                       then cross-encoder rerank
    top-k passages (+ rerank scores)
        |
        v  CONFIDENCE GATE (Day 3) -- if the #1 score is below threshold,
        |                            REFUSE before spending a generation call
        |
        +--(below threshold)--> grounded refusal, no model call
        |
        v  GENERATE WITH CITATIONS (Day 2) -- each passage is a citable
        |                                     `document` block; the API attaches
        |                                     the exact span each claim rests on
    answer with inline [n] markers + verified-span footnotes
        |
        v  HALLUCINATION CHECK (Day 4) -- citation-coverage %, flagged only
                                          when the model CLAIMS to answer

Defense in depth: the gate is a cheap hard pre-filter, the prompt is the soft
backstop, and citations make whatever survives auditable. No single layer is
trusted alone.

Run an interactive chat:   python chat.py
Run the scripted demo:     python chat.py --demo
"""

import pathlib
import sys

import anthropic
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
sys.path.insert(0, str(HERE))
from engine import RetrievalEngine                              # noqa: E402
from contextualize import contextualize                         # noqa: E402

MODEL = "claude-haiku-4-5"
DONT_KNOW = "I don't know based on the available documentation."

# Tuned on Day 3 (out-of-scope topped 0.539, worst in-scope 0.578). 0.55 sits in
# the gap. It is a blunt pre-filter, not the only guard -- the prompt refusal
# below is the backstop for anything the gate lets through.
THRESHOLD = 0.55
TOP_K = 5

SYSTEM = (
    "You are an IT helpdesk assistant. Answer the user's question using ONLY the "
    "provided documents. Cite the document(s) that support each claim. If the "
    f"documents do not contain the answer, reply with exactly: \"{DONT_KNOW}\""
)

# How short an answer's citation coverage can get before we flag it as possibly
# ungrounded -- but ONLY when the model claimed to answer (a refusal is allowed
# to be uncited). See Day 4: coverage is a signal, not a verdict.
COVERAGE_FLOOR = 0.50


def build_documents(passages):
    """Each retrieved passage becomes a citable plain-text `document` block.

    Enabling citations is all-or-none across the blocks, and is incompatible
    with structured outputs (output_config.format -> 400). See Day 2.
    """
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": text},
            "title": source,
            "citations": {"enabled": True},
        }
        for _pid, _score, source, text in passages
    ]


def generate(client, query, passages):
    """Single grounded, citation-enabled generation call over the passages."""
    return client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": build_documents(passages) + [
                {"type": "text", "text": f"Question: {query}"}
            ],
        }],
    )


def answer(engine, client, query):
    """The full pipeline for one already-standalone query.

    Returns a dict the caller can render or evaluate:
        refused      -- bool, did we decline to answer?
        gated        -- bool, did the GATE (not the model) cause the refusal?
        top_score    -- float, reranker score of the best passage
        text         -- the answer (or refusal) string
        citations    -- list of {n, title, quote} verified spans
        coverage     -- fraction of answer chars backed by a citation
        flagged      -- bool, claimed-to-answer but coverage below floor
        passages     -- the retrieved passages (id, score, source, text)
    """
    passages = engine.search(query, k=TOP_K)
    top_score = passages[0][1] if passages else 0.0

    # Layer 1: hard confidence gate -- refuse before paying for generation.
    if top_score < THRESHOLD:
        return {
            "refused": True, "gated": True, "top_score": top_score,
            "text": DONT_KNOW, "citations": [], "coverage": 0.0,
            "flagged": False, "passages": passages,
        }

    response = generate(client, query, passages)
    text, citations, cited_chars, total_chars = _parse(response)

    refused = text.strip() == DONT_KNOW
    coverage = (cited_chars / total_chars) if total_chars else 0.0
    # Flag only a CONFIDENT answer with thin grounding -- a refusal is allowed
    # to be uncited (it is honest, not a hallucination).
    flagged = (not refused) and coverage < COVERAGE_FLOOR

    return {
        "refused": refused, "gated": False, "top_score": top_score,
        "text": text, "citations": citations, "coverage": coverage,
        "flagged": flagged, "passages": passages,
    }


def _parse(response):
    """Pull plain answer text + verified citation spans out of the response.

    Returns (answer_text, citations, cited_chars, total_chars) where each
    citation is {n, title, quote}; n = document_index + 1 (retrieval order).
    cited_chars counts characters in text blocks that carried >=1 citation.
    """
    parts, citations = [], []
    cited_chars = total_chars = 0
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        total_chars += len(block.text)
        cites = getattr(block, "citations", None)
        if cites:
            cited_chars += len(block.text)
            for c in cites:
                # keep the FULL span here; render() truncates for display, the
                # eval verifies the full span against the source document.
                citations.append({
                    "n": c.document_index + 1,
                    "title": c.document_title,
                    "quote": " ".join((c.cited_text or "").split()),
                })
    return "".join(parts), citations, cited_chars, total_chars


def _clip(text, n=90):
    return text if len(text) <= n else text[:n] + "..."


def render(result):
    """Print one answer the way a user would see it."""
    gate = "REFUSE (gated, no model call)" if result["gated"] else \
           ("REFUSE" if result["refused"] else "ANSWER")
    print(f"[{gate}]  top score={result['top_score']:0.3f}")
    print(f"assistant: {result['text']}")
    if result["citations"]:
        print(f"  coverage: {result['coverage']:0.0%} of the answer is cited")
        print("  sources (verified spans):")
        for c in dict.fromkeys(
            f"    [{c['n']}] ({c['title']}) \"{_clip(c['quote'])}\""
            for c in result["citations"]
        ):
            print(c)
    elif not result["refused"]:
        print("  (no citations attached -- ungrounded)")
    if result["flagged"]:
        print("  !! FLAG: answered with low citation coverage -- verify before trusting.")


DEMO = [
    # A normal in-scope question.
    [("user", "my vpn keeps dropping when I work from a hotel")],
    # A multi-turn conversation: the 2nd turn is a contentless follow-up that is
    # meaningless to retrieve on by itself -- contextualization rescues it.
    [
        ("user", "the shared office printer won't connect, it shows an error"),
        ("user", "what if installing the updates doesn't fix it?"),
    ],
    # Out of scope: the gate should refuse without ever calling the model.
    [("user", "what is the company holiday schedule for next year")],
]


def run_conversation(engine, client, turns):
    """Replay a list of (role, text) user turns as one conversation."""
    history = []
    for _role, latest in turns:
        print("=" * 72)
        standalone = contextualize(client, history, latest)
        print(f"user: {latest!r}")
        if standalone != latest:
            print(f"  -> rewritten for retrieval: {standalone!r}")
        result = answer(engine, client, standalone)
        render(result)
        history.append(("user", latest))
        # keep a short assistant turn in history so later follow-ups resolve
        history.append(("assistant", result["text"][:200]))


def interactive(engine, client):
    print("Chat with the IT helpdesk docs. Type 'exit' to quit.\n")
    history = []
    while True:
        try:
            latest = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not latest:
            continue
        if latest.lower() in {"exit", "quit"}:
            break
        standalone = contextualize(client, history, latest)
        if standalone != latest:
            print(f"  (searching for: {standalone!r})")
        result = answer(engine, client, standalone)
        render(result)
        history.append(("user", latest))
        history.append(("assistant", result["text"][:200]))


def main():
    engine = RetrievalEngine()
    client = anthropic.Anthropic()
    try:
        if "--demo" in sys.argv[1:]:
            for turns in DEMO:
                run_conversation(engine, client, turns)
        else:
            interactive(engine, client)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
