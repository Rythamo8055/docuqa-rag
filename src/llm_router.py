"""
LLM Provider Router (LiteLLM-style routing, zero heavy dependencies).

Routes each request to the cheapest/fastest available provider:
    Groq (free tier) → Gemini (free tier) → OpenAI → Anthropic → Ollama (local)

- Groq/OpenAI use the OpenAI-compatible SDK (base_url switch)
- Gemini uses the Google Generative Language REST API (no SDK needed)
- Anthropic uses its Messages API via requests
- Ollama runs fully local at $0 cost
"""
from typing import List, Tuple, Union
import logging
import os

import requests
import openai

from src.env_loader import load_env_file

logger = logging.getLogger(__name__)

load_env_file()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",   # free tier, very fast
    "gemini": "gemma-4-31b-it",          # free tier (Google AI Studio key)
    "openai": "gpt-4o-mini",             # cheap, high quality
    "anthropic": "claude-3-5-haiku-latest",  # cheapest Claude
    "ollama": None,                      # from env OLLAMA_MODEL
}


class LLMRouter:
    """Routes LLM calls to the best available provider."""

    def __init__(self):
        self._keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        }
        # Allow overriding the default model per provider via env
        self._models = {
            "gemini": os.getenv("GEMINI_MODEL") or DEFAULT_MODELS["gemini"],
            "ollama": os.getenv("OLLAMA_MODEL"),
        }

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def available_providers(self) -> List[str]:
        """Providers that can be used right now (cheapest first)."""
        providers = [p for p, key in self._keys.items() if key]
        if self._ollama_available():
            providers.append("ollama")
        return providers

    def has_any_provider(self) -> bool:
        return bool(self.available_providers())

    def _ollama_available(self) -> bool:
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        try:
            return requests.get(f"{url}/api/tags", timeout=2).ok
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system: str = "",
    ) -> Tuple[Union[str, object], str, str]:
        """
        Generate a response using the best available provider.

        Args:
            prompt: user-turn text
            system: system-level instructions (sent via the provider's
                    native system channel where supported)

        Returns:
            (result, provider, model)
            - result: str, or a generator of str pieces when stream=True
            - provider: name of the provider used ("groq"/"openai"/...)
            - model: model identifier used

        Raises:
            RuntimeError if no provider is configured/available.
        """
        # Groq first (free tier, fastest)
        if self._keys["groq"]:
            gen = self._openai_compatible(
                api_key=self._keys["groq"],
                base_url=GROQ_BASE_URL,
                model=DEFAULT_MODELS["groq"],
                prompt=prompt, stream=stream,
                temperature=temperature, max_tokens=max_tokens,
                system=system,
            )
            return gen, "groq", DEFAULT_MODELS["groq"]

        # Gemini second (free tier with Google AI Studio key)
        if self._keys["gemini"]:
            model = self._models["gemini"]
            result = self._gemini(
                api_key=self._keys["gemini"],
                model=model,
                prompt=prompt, stream=stream,
                temperature=temperature, max_tokens=max_tokens,
                system=system,
            )
            return result, "gemini", model

        # OpenAI third
        if self._keys["openai"]:
            gen = self._openai_compatible(
                api_key=self._keys["openai"],
                base_url=None,
                model=DEFAULT_MODELS["openai"],
                prompt=prompt, stream=stream,
                temperature=temperature, max_tokens=max_tokens,
                system=system,
            )
            return gen, "openai", DEFAULT_MODELS["openai"]

        # Anthropic fourth
        if self._keys["anthropic"]:
            result = self._anthropic(
                api_key=self._keys["anthropic"],
                model=DEFAULT_MODELS["anthropic"],
                prompt=prompt, temperature=temperature, max_tokens=max_tokens,
                system=system,
            )
            return result, "anthropic", DEFAULT_MODELS["anthropic"]

        # Local Ollama last (free, always available if installed)
        if self._ollama_available():
            model = os.getenv("OLLAMA_MODEL", "llama3")
            result = self._ollama(
                model=model, prompt=prompt,
                stream=stream, temperature=temperature, max_tokens=max_tokens,
            )
            return result, "ollama", model

        raise RuntimeError(
            "No LLM provider configured. Set GROQ_API_KEY, GEMINI_API_KEY, "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, or run Ollama locally."
        )

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------
    def _openai_compatible(self, api_key, base_url, model, prompt, stream,
                           temperature, max_tokens, system=""):
        """Groq & OpenAI (both OpenAI-compatible)."""
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if stream:
            resp = client.chat.completions.create(**kwargs, stream=True)

            def gen():
                for chunk in resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            return gen()
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    def _anthropic(self, api_key, model, prompt, temperature, max_tokens, system=""):
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        resp = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

    def _gemini(self, api_key, model, prompt, stream, temperature, max_tokens, system=""):
        """Google Gemini API (REST, no SDK needed)."""
        url = (
            f"{GEMINI_BASE_URL}/models/{model}:"
            + ("streamGenerateContent" if stream else "generateContent")
        )
        params = {"key": api_key}
        if stream:
            params["alt"] = "sse"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            # Native system-instruction channel (prevents prompt echoing)
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        # Retry on transient failures (5xx, 429 rate limit) — 3 attempts
        # with exponential backoff. Powered by the resilience module.
        from src.resilience import retry, RetryConfig
        import requests as _requests

        _cfg = RetryConfig(
            max_attempts=3, base_delay=1.0, max_delay=8.0,
            retryable_exceptions=(_requests.exceptions.RequestException,),
        )

        @retry(_cfg)
        def _post():
            r = requests.post(url, params=params, json=payload, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise _requests.exceptions.HTTPError(
                    f"Gemini {r.status_code}: {r.text[:200]}", response=r
                )
            return r

        resp = _post()
        resp.raise_for_status()

        def _extract(data):
            candidates = data.get("candidates") or []
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()

        if stream:
            def gen():
                import json as _json
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        data = _json.loads(line[5:].strip())
                        text = _extract(data)
                        if text:
                            yield text
                    except _json.JSONDecodeError:
                        continue
            return gen()

        return self._clean_gemma_output(_extract(resp.json()))

    @staticmethod
    def _clean_gemma_output(text: str) -> str:
        """
        Gemma 4 models (served via Gemini API) emit a chain-of-thought
        trace before the final answer. Keep only the final answer block.

        Heuristics, in order:
          1. If the response is a "not found" refusal, normalize it to the
             exact mandated phrase (drops any leaked reasoning that might
             mention the question terms, e.g. "the capital of France").
          2. If it looks like a reasoning trace (multiple paragraphs where
             only later ones carry a citation), keep everything from the
             first cited paragraph onward.
          3. Dedupe a sentence that immediately repeats itself (a common
             Gemma quirk).
        """
        if not text:
            return text

        import re as _re
        low = text.lower()

        # 1) Refusal normalization — the only acceptable out-of-scope answer
        #    is the exact phrase; strip any reasoning that leaked around it.
        if "not found in the provided document" in low:
            try:
                from src.llm import NOT_FOUND_PHRASE  # deferred: avoid cycle
                return NOT_FOUND_PHRASE
            except ImportError:
                return "Information not found in the provided document."

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            # 2) Drop reasoning paragraphs before the first cited block.
            first_cited = next(
                (
                    i for i, p in enumerate(paragraphs)
                    if _re.search(r"\[Page \d+, Chunk \d+\]", p)
                ),
                None,
            )
            if first_cited is not None and first_cited > 0:
                text = "\n\n".join(paragraphs[first_cited:])
            # Fallback for trace-looking output without any citation block.
            elif (
                "constraint" in low
                or "context provided" in low
                or "the user is asking" in low
                or "i need to check" in low
                or "looking at the provided" in low
            ):
                text = paragraphs[-1]

        # Dedupe a leading sentence that repeats itself, e.g.
        # "*   X is Y.X is Y and Z." → "X is Y and Z."
        text = text.strip()
        if text:
            # strip leading bullet markers
            text = _re.sub(r"^\s*\*\s*", "", text)
            m = _re.match(r"^(.*?[.!?])(.*)$", text, _re.S)
            if m:
                first = m.group(1).rstrip(".!?").strip()
                rest = m.group(2).lstrip()
                if rest.startswith(first):
                    text = rest
            # sentence-level dedupe: keep first occurrence of repeated sentences
            # (Gemma sometimes repeats the answer sentence before continuing)
            sentences = _re.split(r"(?<=[.!?])\s*(?=[A-Z*])", text)
            if len(sentences) > 2:
                seen, kept = set(), []
                for sent in sentences:
                    key = _re.sub(r"[\s\W]+", "", sent.lower())[:80]
                    if key and key not in seen:
                        seen.add(key)
                        kept.append(sent)
                text = " ".join(kept)
        return text

    def _ollama(self, model, prompt, stream, temperature, max_tokens):
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if stream:
            return self._ollama_stream(url, payload)
        resp = requests.post(f"{url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _ollama_stream(self, url, payload):
        with requests.post(f"{url}/api/generate", json=payload, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                import json as _json
                try:
                    data = _json.loads(line)
                    piece = data.get("response", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        break
                except _json.JSONDecodeError:
                    continue