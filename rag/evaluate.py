"""Capstone eval: does the RAG answer correctly, cite honestly, and refuse safely?

Building a RAG is easy; knowing whether it WORKS is the part most portfolios
skip. This harness scores the three behaviours a grounded assistant must get
right, on a labelled set of ANSWERABLE and UNANSWERABLE questions:

  1. ANSWER CORRECTNESS  (answerable set)
     Did it answer (not refuse) AND does the answer contain the known-correct
     fact? Each answerable question carries a `gold` substring that only the
     right answer would mention. correctness = answered AND gold in answer.

  2. CITATION ACCURACY  (every answered question)
     Is each cited span a GENUINE substring of the document it is attributed
     to? Native Citations copies the span mechanically from the source, so this
     should be ~100% -- the eval PROVES the citations are real, not fabricated.
     (Day 1's hand-typed [n] markers could not pass this check.)

  3. REFUSAL ACCURACY  (unanswerable set)
     Did it REFUSE the questions the docs cannot answer, instead of inventing
     something? A RAG that answers everything is worse than useless.

We also report FALSE REFUSALS: answerable questions the system wrongly declined
(over-cautious gate/prompt) -- the cost of being strict.

This is a gold-substring + structural eval (deterministic, cheap, honest about
its limits). Full LLM-as-judge scoring of answer quality comes in Week 6.

Run (after `python engine.py build`):  python evaluate.py
"""

import json
import pathlib
import sys

import anthropic
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
sys.path.insert(0, str(HERE))
from engine import RetrievalEngine                              # noqa: E402
from chat import answer, DONT_KNOW                              # noqa: E402

DOCS_DIR = HERE / "docs"
# Anthropic + Voyage calls cost money / hit rate limits; cache each question's
# result so a re-run resumes instead of re-spending. Delete to re-evaluate.
CACHE_PATH = HERE / "eval_cache.json"

# Answerable: real questions phrased WITHOUT copying the doc wording, each with
# a gold substring that only the correct answer would contain.
ANSWERABLE = [
    ("my vpn keeps dropping when I work from a hotel",      "fully quit GlobalProtect"),
    ("I'm locked out after too many wrong password tries",  "fifteen minutes"),
    ("the shared printer throws an error when I connect",    "0x0000011b"),
    ("outlook won't send, it says my mailbox is full",       "Deleted Items"),
    ("I need software that isn't in the company portal",     "business justification"),
    ("my laptop dies after an hour off the charger",         "different charger"),
]

# Unanswerable: plausible-sounding questions the helpdesk docs do NOT cover.
# The correct behaviour for every one of these is to refuse.
UNANSWERABLE = [
    "what is the company holiday schedule for next year",
    "how do I change my 401k contribution rate",
    "can I expense a taxi from the airport",
    "who is the CEO of the company",
    "what is the capital of France",
]


def _norm(text):
    return " ".join(text.lower().split())


def load_docs():
    """Full normalised text of each source doc, for verifying cited spans."""
    return {p.name: _norm(p.read_text(encoding="utf-8")) for p in DOCS_DIR.glob("*.txt")}


def run_pipeline(engine, client, question, cache):
    """Run the full chat pipeline for one question, cached. Single-turn (no
    history) -- eval questions are standalone, so contextualization is a no-op."""
    if question in cache:
        return cache[question]
    result = answer(engine, client, question)
    record = {
        "refused": result["refused"],
        "gated": result["gated"],
        "top_score": result["top_score"],
        "text": result["text"],
        "coverage": result["coverage"],
        "citations": result["citations"],   # full spans (already plain dicts)
    }
    cache[question] = record
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    return record


def citations_verify(record, docs):
    """(verified, total) cited spans whose quote really occurs in its source."""
    verified = total = 0
    for c in record["citations"]:
        total += 1
        doc = docs.get(c["title"], "")
        if _norm(c["quote"]) in doc:
            verified += 1
    return verified, total


def main():
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    docs = load_docs()
    engine = RetrievalEngine()
    client = anthropic.Anthropic()
    try:
        # --- 1. ANSWERABLE: correctness + citation accuracy ------------------
        print("ANSWERABLE QUESTIONS (should answer with the right fact)\n")
        correct = false_refusals = 0
        cite_ok = cite_total = 0
        for q, gold in ANSWERABLE:
            r = run_pipeline(engine, client, q, cache)
            answered = not r["refused"]
            has_gold = gold.lower() in r["text"].lower()
            is_correct = answered and has_gold
            correct += is_correct
            false_refusals += (not answered)
            v, t = citations_verify(r, docs)
            cite_ok += v
            cite_total += t
            mark = "OK " if is_correct else "XX "
            detail = (f"cites {v}/{t} verified, coverage {r['coverage']:0.0%}"
                      if answered else "REFUSED (false refusal)")
            print(f"  {mark} {q[:46]:<46} gold={'y' if has_gold else 'n'}  {detail}")

        n_ans = len(ANSWERABLE)
        print(f"\n  answer correctness : {correct}/{n_ans} = {correct / n_ans:0.0%}")
        print(f"  false refusals     : {false_refusals}/{n_ans}")
        cite_rate = (cite_ok / cite_total) if cite_total else 0.0
        print(f"  citation accuracy  : {cite_ok}/{cite_total} spans verified = {cite_rate:0.0%}")

        # --- 2. UNANSWERABLE: refusal accuracy -------------------------------
        print("\nUNANSWERABLE QUESTIONS (should refuse)\n")
        refused = 0
        for q in UNANSWERABLE:
            r = run_pipeline(engine, client, q, cache)
            refused += r["refused"]
            how = "gated" if r["gated"] else ("prompt" if r["refused"] else "ANSWERED")
            mark = "OK " if r["refused"] else "XX "
            print(f"  {mark} {q[:46]:<46} top={r['top_score']:0.3f}  {how}")

        n_un = len(UNANSWERABLE)
        print(f"\n  refusal accuracy   : {refused}/{n_un} = {refused / n_un:0.0%}")

        # --- summary ---------------------------------------------------------
        print("\n" + "-" * 60)
        print(f"{'answer correctness (answerable)':<34}{correct}/{n_ans}  {correct / n_ans:>5.0%}")
        print(f"{'citation accuracy (verified spans)':<34}{cite_ok}/{cite_total}  {cite_rate:>5.0%}")
        print(f"{'refusal accuracy (unanswerable)':<34}{refused}/{n_un}  {refused / n_un:>5.0%}")
        print("\nCitation accuracy is ~100% by construction: native Citations copies")
        print("each span from the source, so a cited quote is always a real substring")
        print("of its document. That is the point -- the answer is auditable, not")
        print("taken on trust. Answer correctness and refusal accuracy are the")
        print("behaviours that actually vary and are worth tuning.")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
