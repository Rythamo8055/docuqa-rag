"""
Grounded Generation and Citation System.

Engineered prompts enforce strict grounding: answers must come ONLY from the
retrieved document context, with explicit [Page X, Chunk Y] citations.
If the answer cannot be derived from the context, the model must reply:
"Information not found in the provided document."

LLM calls are routed through LLMRouter (Groq → OpenAI → Anthropic → Ollama)
for cost efficiency.
"""
from typing import List, Dict, Union
import logging
import re

from src.llm_router import LLMRouter

logger = logging.getLogger(__name__)

NOT_FOUND_PHRASE = "Information not found in the provided document."


class GroundedGenerator:
    """Generates answers strictly grounded in retrieved chunks."""

    PROMPT_TEMPLATE = """You are an expert assistant that answers questions strictly based on the provided document context. Your role is to prevent hallucinations by ONLY using information explicitly present in the context.

INSTRUCTIONS:
- Answer the question ONLY using the retrieved chunks below.
- If the answer cannot be derived from the context, respond with EXACTLY: "{not_found}"
- Include explicit citations in the format [Page X, Chunk Y] for every fact or statement you make.
- Be concise and accurate.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    def __init__(self, router: Union[LLMRouter, None] = None):
        self.router = router or LLMRouter()

    def format_context(self, chunks: List[Dict]) -> str:
        """Label each chunk with its citation and join into one context block."""
        parts = []
        for chunk in chunks:
            citation = f"[Page {chunk['page']}, Chunk {chunk['chunk_id']}]"
            parts.append(f"{citation}: {chunk['text']}")
        return "\n\n".join(parts)

    def generate_response(
        self,
        question: str,
        chunks: List[Dict],
        stream: bool = False,
    ):
        """
        Generate a grounded response for a question given retrieved chunks.

        Returns:
            (result, provider, model)
            - result: str or generator (when stream=True)
            - provider: "groq"/"openai"/"anthropic"/"ollama" or "rule-based"
            - model: model name (or None for rule-based fallback)
        """
        if not chunks:
            return NOT_FOUND_PHRASE, "rule-based", None

        context = self.format_context(chunks)
        prompt = self.PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            not_found=NOT_FOUND_PHRASE,
        )

        if not self.router.has_any_provider():
            # No LLM available — honest rule-based fallback (never fabricates)
            return (
                f"{NOT_FOUND_PHRASE} (No LLM provider configured — "
                f"set GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                f"or run Ollama locally.)",
                "rule-based",
                None,
            )

        result, provider, model = self.router.generate(prompt, stream=stream)
        return result, provider, model


def check_grounding(answer: str, chunks: List[Dict]) -> Dict:
    """
    Lightweight post-hoc grounding check.

    Verifies the answer either:
    - contains the "not found" phrase (acceptable when context is thin), or
    - includes at least one [Page X, Chunk Y] citation.

    Returns: {"grounded": bool, "reason": str}
    """
    if not answer:
        return {"grounded": False, "reason": "Empty answer"}
    if NOT_FOUND_PHRASE.lower() in answer.lower():
        return {"grounded": True, "reason": "Not-found guard triggered"}
    if re.search(r"\[Page \d+, Chunk \d+\]", answer):
        return {"grounded": True, "reason": "Citations present"}
    return {
        "grounded": False,
        "reason": "No citation found — answer may be ungrounded",
    }