"""
Grounded Generation and Citation System.

Connects to language models via API (OpenAI/Anthropic/Groq) or locally (Ollama),
with prompts engineered to prevent hallucinations and enforce strict grounding
from retrieved document context.
"""
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Default model settings
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0  # Deterministic for grounded responses
MAX_TOKENS = 1024


class GroundedGenerator:
    """
    Generates answers grounded in retrieved document chunks, with explicit
    citations to prevent hallucinations.
    """

    # Prompt template that enforces strict grounding
    PROMPT_TEMPLATE = """You are an expert assistant that answers questions strictly based on the provided document context. Your role is to prevent hallucinations by only using information explicitly present in the context.

INSTRUCTIONS:
- Answer the question ONLY using the retrieved chunks below.
- If the answer cannot be derived from the context, respond with EXACTLY: "Information not found in the provided document."
- Include explicit citations in the format [Page X, Chunk Y] for every fact or statement you make.
- Be concise and accurate.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature

    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into a labeled context string.

        Each chunk is labeled with its page and chunk ID for citation.
        """
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
        Generate a grounded response using the retrieved chunks.

        Args:
            question: The user's question
            chunks: List of retrieved chunk dictionaries
            stream: Whether to stream the response token-by-token

        Returns:
            Generated response string (or generator if stream=True)
        """
        if not chunks:
            return "Information not found in the provided document."

        context = self.format_context(chunks)
        prompt = self.PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        if self.api_key:
            return self._call_openai(prompt, stream=stream)
        else:
            # Fallback to local generation (Ollama)
            return self._call_local_llm(prompt, stream=stream)

    def _call_openai(self, prompt: str, stream: bool = False):
        """Call OpenAI API for generation."""
        import openai

        client = openai.OpenAI(api_key=self.api_key)

        messages = [{"role": "user", "content": prompt}]

        if stream:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
            return response  # Generator for streaming
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()

    def _call_local_llm(self, prompt: str, stream: bool = False):
        """Call a local LLM via Ollama."""
        import requests

        # Default to a lightweight local model
        local_model = os.getenv("OLLAMA_MODEL", "llama3")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

        payload = {
            "model": local_model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": MAX_TOKENS,
            },
        }

        response = requests.post(ollama_url, json=payload)
        if stream:
            return response.iter_lines()
        else:
            result = response.json()
            return result.get("response", "Information not found in the provided document.")
