#!/usr/bin/env python3
"""LLM-as-judge scoring (Ragas-style) for the evaluation harness.

The deterministic lexical metrics in run_eval.py are proxies for the
industry-standard Ragas faithfulness / answer-relevancy metrics, which
are computed by an LLM judge. This module implements that judge with a
single strict-JSON call per case:

  faithfulness   - is every claim in the answer entailed by the context?
  relevancy      - does the answer actually address the question?

The judge is deliberately *optional*: when the provider is unavailable,
flaky, or refuses, it returns None and the harness falls back to the
lexical metric (the run stays fully reproducible offline).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_router import LLMRouter  # noqa: E402

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluation judge for retrieval-augmented "
    "generation systems. You score two metrics on a 0.0-1.0 scale.\n"
    "faithfulness: the fraction of claims in the ANSWER that are directly "
    "supported by the CONTEXT. Any invented or unsupported detail lowers it.\n"
    "relevancy: how well the ANSWER addresses the QUESTION (is it on-topic "
    "and complete enough to answer the user?).\n"
    "Respond with ONLY a JSON object, no prose, no markdown fences:\n"
    '{"faithfulness": 0.0, "relevancy": 0.0}'
)

JUDGE_PROMPT = """QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Score the ANSWER with the two metrics defined in the system prompt.
Output only the JSON object."""


def _parse_json_score(text: str) -> Optional[Tuple[float, float]]:
    """Extract {faithfulness, relevancy} from a (possibly chatty) judge reply."""
    if not text:
        return None
    # strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE)
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        f = float(data.get("faithfulness", -1))
        r = float(data.get("relevancy", -1))
        if 0.0 <= f <= 1.0 and 0.0 <= r <= 1.0:
            return f, r
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    # last resort: bare numbers already in range
    nums = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0)?)\b", cleaned)
    if len(nums) >= 2:
        try:
            return float(nums[0]), float(nums[1])
        except ValueError:
            pass
    return None


def judge_scores(
    question: str,
    answer: str,
    context: str,
    router: Optional[LLMRouter] = None,
    max_tokens: int = 64,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (faithfulness, relevancy) judged by the LLM, or (None, None).

    One router call per case (both metrics in a single JSON reply).
    """
    if not answer:
        return None, None
    router = router or LLMRouter()
    if not router.has_any_provider():
        return None, None

    prompt = JUDGE_PROMPT.format(
        question=question, context=context[:6000], answer=answer[:1500]
    )
    try:
        result, provider, _model = router.generate(
            prompt, temperature=0.0, max_tokens=max_tokens, system=JUDGE_SYSTEM
        )
        text = result if isinstance(result, str) else "".join(result)
    except Exception:  # noqa: BLE001 - judge is best-effort
        return None, None
    return _parse_json_score(text) or (None, None)
