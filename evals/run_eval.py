#!/usr/bin/env python3
"""Offline evaluation harness for the RAG pipeline.

Runs every case in evals/eval_dataset.json against the production
RAGService (same code the API uses) and computes:

  retrieval hit-rate      - did the expected pages/chunks come back in top-K?
  context precision       - were the relevant pages ranked high (Ragas formula)?
  term coverage           - did expected terms appear in the answer?
  grounded                - did the grounding check accept the answer?
  faithfulness            - lexical overlap of answer with retrieved context
                           (LLM-judged when --judge, Ragas-style)
  relevance               - lexical overlap of answer with the question
                           (LLM-judged when --judge, Ragas-style)
  guardrail checks        - injection blocked? ungrounded flagged?

Usage:
    python3 evals/run_eval.py                 # offline: deterministic extractive answers (fast, reproducible)
    python3 evals/run_eval.py --llm           # live: answer with the configured LLM provider
    python3 evals/run_eval.py --judge         # score faithfulness/relevancy with an LLM judge (Ragas-style)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag_service import RAGService  # noqa: E402


# ---------------------------------------------------------------- metrics ---

STOPWORDS = set(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with what which who whom whose how when where why do "
    "does did done would could should can may might not no yes than then there "
    "their them they we you i he she it's don't your our".split()
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_faithfulness(answer: str, context: str) -> float:
    """Fraction of answer content words found in the retrieved context."""
    ans = {t for t in tokenize(answer) if t not in STOPWORDS and len(t) > 2}
    ctx = set(tokenize(context))
    if not ans:
        return 0.0
    return len(ans & ctx) / len(ans)


def lexical_relevance(answer: str, question: str) -> float:
    """Fraction of question content words found in the answer."""
    q = {t for t in tokenize(question) if t not in STOPWORDS and len(t) > 2}
    ans = set(tokenize(answer))
    if not q:
        return 0.0
    return len(q & ans) / len(q)


def term_coverage(answer: str, terms: list[str]) -> float:
    if not terms:
        return 1.0
    a = answer.lower()
    return sum(1 for t in terms if t.lower() in a) / len(terms)


def context_precision(returned: list[dict], exp_pages: set[int]) -> float:
    """Ragas context_precision: how high the relevant chunks are ranked.

    For every relevant chunk at 1-based rank k, precision@k is the share
    of relevant chunks in the top k; the metric is the mean over all
    relevant chunks (1.0 = all relevant chunks ranked above everything
    else, 0.0 = none retrieved).
    """
    if not exp_pages:
        return 1.0 if not returned else 0.0
    relevant = [
        i for i, c in enumerate(returned) if c.get("page") in exp_pages
    ]
    if not relevant:
        return 0.0
    return sum(
        sum(1 for j in relevant if j <= i) / (i + 1) for i in relevant
    ) / len(relevant)


# ------------------------------------------------------------------- main ---

def run(service: RAGService, judge: bool = False) -> dict:
    dataset = json.loads((ROOT / "evals" / "eval_dataset.json").read_text())
    cases = dataset["cases"]

    router = None
    if judge:
        from evals.judge import judge_scores  # local import: optional dep
        from src.llm_router import LLMRouter
        router = LLMRouter()
        if not router.has_any_provider():
            print("[eval] --judge requested but no LLM provider configured; "
                  "falling back to lexical metrics.")
            judge = False

    results = []
    for case in cases:
        q = case["question"]
        t0 = time.time()
        resp = service.query(q, cache_on=False, rerank_on=False)
        elapsed = time.time() - t0

        # --- retrieval metrics -------------------------------------------
        returned = resp["chunks"] or resp["context_chunks"] or []
        pages = {c.get("page") for c in returned if c.get("page") is not None}
        exp_pages = set(case.get("expected_pages", []))
        retrieval_hit = 1.0 if exp_pages and exp_pages <= pages else (
            0.0 if exp_pages else None
        )
        prec = context_precision(returned, exp_pages)
        chunk_ids = [c.get("chunk_id") for c in returned]

        # --- answer-level metrics ----------------------------------------
        answer = resp.get("answer", "")
        context = " ".join(c.get("text", "") for c in returned)
        tc = term_coverage(answer, case.get("expected_terms", []))
        faith = lexical_faithfulness(answer, context) if answer else 0.0
        rel = lexical_relevance(answer, q) if answer else 0.0
        grounded = bool(resp.get("grounding", {}).get("grounded"))
        blocked = bool(resp.get("blocked"))

        # --- LLM-as-judge scores (Ragas-style, optional) -----------------
        faith_judged = rel_judged = None
        if judge and router is not None:
            faith_judged, rel_judged = judge_scores(q, answer, context, router)
            if faith_judged is not None:
                faith = faith_judged
            if rel_judged is not None:
                rel = rel_judged

        # --- expected behaviours -----------------------------------------
        checks = {
            "expected_pages_retrieved": retrieval_hit if retrieval_hit is not None else None,
            "expect_blocked": case.get("expect_blocked", False),
        }
        pass_ = True
        if checks["expected_pages_retrieved"] is not None:
            pass_ &= retrieval_hit >= 1.0
        if case.get("expected_absent_terms"):
            # no-fabrication: answer must NOT contain facts absent from the doc
            a = answer.lower()
            absent_hit = any(t.lower() in a for t in case["expected_absent_terms"])
            pass_ &= not absent_hit
            checks["absent_terms_violated"] = absent_hit
        if checks["expect_blocked"]:
            pass_ &= blocked
        if case.get("expected_terms"):
            pass_ &= tc >= 0.5

        results.append({
            "id": case["id"],
            "category": case["category"],
            "question": q,
            "pass": bool(pass_),
            "blocked": blocked,
            "grounded": grounded,
            "retrieval_hit": retrieval_hit,
            "context_precision": round(prec, 3),
            "pages_returned": sorted(pages),
            "chunk_ids": chunk_ids,
            "term_coverage": round(tc, 3),
            "faithfulness": round(faith, 3),
            "relevance": round(rel, 3),
            "judged": faith_judged is not None or rel_judged is not None,
            "absent_terms_violated": checks.get("absent_terms_violated", False),
            "answer_excerpt": (answer[:220] + "…") if answer else "",
            "elapsed_s": round(elapsed, 1),
        })

    # ------------------------------------------------------------- summary ---
    n = len(results)
    passed = sum(1 for r in results if r["pass"])
    hits = [r["retrieval_hit"] for r in results if r["retrieval_hit"] is not None]
    precs = [r["context_precision"] for r in results if r["context_precision"] is not None]
    judged = [r for r in results if r["judged"]]
    summary = {
        "total_cases": n,
        "passed": passed,
        "pass_rate": round(passed / n, 3) if n else 0.0,
        "retrieval_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
        "avg_context_precision": round(sum(precs) / len(precs), 3) if precs else None,
        "avg_faithfulness": round(
            sum(r["faithfulness"] for r in results) / n, 3
        ),
        "avg_relevance": round(sum(r["relevance"] for r in results) / n, 3),
        "judge": judge,
        "judge_covered": len(judged),
        "by_category": {},
        "guardrail_checks": {
            "injection_blocked": next(
                r for r in results if r["id"] == "inj-001"
            )["blocked"],
            "negative_no_fabrication": not next(
                r for r in results if r["id"] == "neg-001"
            ).get("absent_terms_violated", False),
        },
    }
    for r in results:
        summary["by_category"].setdefault(r["category"], [0, 0])
        summary["by_category"][r["category"]][1] += 1
        summary["by_category"][r["category"]][0] += 1 if r["pass"] else 0

    return {"summary": summary, "results": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--llm",
        action="store_true",
        help="answer with the live LLM (slower, nondeterministic). "
        "Default (off): RAGService runs with llm_enabled=False and answers "
        "with the deterministic extractive fallback — fast and reproducible.",
    )
    ap.add_argument("--out", default="evals/report.md")
    ap.add_argument(
        "--judge",
        action="store_true",
        help="score faithfulness/relevance with an LLM judge (Ragas-style). "
        "Falls back to lexical metrics when the judge is unavailable.",
    )
    args = ap.parse_args()

    svc = RAGService(
        persist_dir=".tmp_eval_chroma",
        cache_path=".tmp_eval_cache.sqlite",
        rerank_on=False,
        cache_on=False,
        llm_enabled=args.llm,
    )
    sample = ROOT / "data" / "sample.pdf"
    if not svc.is_indexed():
        print(f"[eval] indexing {sample.name} ...")
        res = svc.ingest(sample.read_bytes(), sample.name)
        print(f"[eval] index: {res['message']}")

    print(f"[eval] running 10 cases (llm={'yes' if args.llm else 'offline-only'}"
          f", judge={'yes' if args.judge else 'no'}) ...")
    out = run(svc, judge=args.judge)

    report = render_report(out, args.llm, args.judge)
    out_path = ROOT / args.out
    out_path.write_text(report)
    print(f"[eval] report written to {out_path}")

    s = out["summary"]
    print(f"[eval] PASS {s['passed']}/{s['total_cases']} "
          f"({s['pass_rate']:.0%}) | retrieval hit {s['retrieval_hit_rate']} "
          f"| ctx precision {s['avg_context_precision']} "
          f"| faithfulness {s['avg_faithfulness']} | relevance {s['avg_relevance']}"
          + (f" | judge covered {s['judge_covered']}/{s['total_cases']}"
             if args.judge else ""))
    return 0 if s["passed"] == s["total_cases"] else 1


def render_report(out: dict, llm_mode: bool, judge_mode: bool = False) -> str:
    s, results = out["summary"], out["results"]
    metric_label = "LLM-judged (Ragas-style)" if judge_mode else "lexical"
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Mode: {'LLM-aided' if llm_mode else 'offline (deterministic extractive)'}"
        + (f" · scoring: {metric_label}" if judge_mode else ""),
        f"- Total cases: {s['total_cases']} | Passed: **{s['passed']}/{s['total_cases']}** ({s['pass_rate']:.0%})",
        f"- Retrieval hit-rate (expected pages in top-K): **{s['retrieval_hit_rate']}**",
        f"- Context precision (Ragas, ranking quality): **{s['avg_context_precision']}**",
        f"- Avg faithfulness ({metric_label}): **{s['avg_faithfulness']}**",
        f"- Avg relevance ({metric_label}): **{s['avg_relevance']}**",
        "",
    ]
    if judge_mode:
        lines.append(
            f"- Judge coverage: **{s['judge_covered']}/{s['total_cases']}** cases scored "
            "by the LLM judge (rest fell back to lexical metrics)"
        )
        lines.append("")
    lines += [
        "## Guardrail checks",
        "",
        f"- Prompt-injection case blocked: `{s['guardrail_checks']['injection_blocked']}`",
        f"- Out-of-document question answered without fabricating facts: `{s['guardrail_checks']['negative_no_fabrication']}`",
        "",
        "## By category",
        "",
        "| Category | Passed | Total |",
        "|---|---|---|",
    ]
    for cat, (ok, tot) in sorted(s["by_category"].items()):
        lines.append(f"| {cat} | {ok} | {tot} |")

    lines += ["", "## Per-case results", "", "| ID | Category | Pass | Blocked | Grounded | Pages | Term cov | Faith | Rel | CtxPrec |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['category']} | {'✅' if r['pass'] else '❌'} "
            f"| {r['blocked']} | {r['grounded']} | {r['pages_returned']} "
            f"| {r['term_coverage']} | {r['faithfulness']} | {r['relevance']} "
            f"| {r['context_precision']} |"
        )

    lines += ["", "## Details"]
    for r in results:
        lines += [
            f"### {r['id']} — {r['category']}",
            f"**Q:** {r['question']}",
            f"**A:** {r['answer_excerpt']}",
            f"**Chunks:** {r['chunk_ids']} ({r['elapsed_s']}s)",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
