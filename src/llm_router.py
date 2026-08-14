"""
LLM Provider Router (LiteLLM-style routing, zero heavy dependencies).

Routes each request to the cheapest/fastest available provider:
    Groq (free tier) → OpenAI → Anthropic → Ollama (local)

- Groq/OpenAI use the OpenAI-compatible SDK (base_url switch)
- Anthropic uses its Messages API via requests
- Ollama runs fully local at $0 cost
"""
from typing import List, Tuple, Union
import logging
import os

import requests
import openai

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",   # free tier, very fast
    "openai": "gpt-4o-mini",             # cheap, high quality
    "anthropic": "claude-3-5-haiku-latest",  # cheapest Claude
    "ollama": None,                      # from env OLLAMA_MODEL
}


class LLMRouter:
    """Routes LLM calls to the best available provider."""

    def __init__(self):
        self._keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
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
    ) -> Tuple[Union[str, object], str, str]:
        """
        Generate a response using the best available provider.

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
            )
            return gen, "groq", DEFAULT_MODELS["groq"]

        # OpenAI second
        if self._keys["openai"]:
            gen = self._openai_compatible(
                api_key=self._keys["openai"],
                base_url=None,
                model=DEFAULT_MODELS["openai"],
                prompt=prompt, stream=stream,
                temperature=temperature, max_tokens=max_tokens,
            )
            return gen, "openai", DEFAULT_MODELS["openai"]

        # Anthropic third
        if self._keys["anthropic"]:
            result = self._anthropic(
                api_key=self._keys["anthropic"],
                model=DEFAULT_MODELS["anthropic"],
                prompt=prompt, temperature=temperature, max_tokens=max_tokens,
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
            "No LLM provider configured. Set GROQ_API_KEY, OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, or run Ollama locally."
        )

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------
    def _openai_compatible(self, api_key, base_url, model, prompt, stream,
                           temperature, max_tokens):
        """Groq & OpenAI (both OpenAI-compatible)."""
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
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

    def _anthropic(self, api_key, model, prompt, temperature, max_tokens):
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

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