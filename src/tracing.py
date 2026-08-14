"""
Observability / Tracing (optional).

Sends retrieval + generation traces to Langfuse when configured
(LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY + LANGFUSE_HOST).
Without config, gracefully degrades to a no-op logger — zero impact.

Langfuse free tier is generous and ideal for debugging grounding failures:
you can see exactly which chunks were retrieved and what the LLM did with them.
"""
from typing import Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class Tracer:
    """Minimal trace wrapper. No-op unless Langfuse env vars are set."""

    def __init__(self):
        self.enabled = False
        self._client = None
        self._try_init_langfuse()

    def _try_init_langfuse(self) -> None:
        pub = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if not (pub and secret):
            logger.debug("Langfuse not configured — tracing disabled")
            return
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=pub,
                secret_key=secret,
                host=host,
            )
            self.enabled = True
            logger.info("Langfuse tracing enabled")
        except Exception as e:
            logger.warning(f"Could not init Langfuse: {e}")

    def trace(self, name: str, **data: Any) -> None:
        """Record a trace event (retrieval, generation, cache hit, ...)."""
        logger.debug(f"[trace:{name}] {data}")
        if self.enabled and self._client is not None:
            try:
                span = self._client.span(name=name, input=data)
                span.end(output=data)
            except Exception as e:
                logger.warning(f"Langfuse trace failed: {e}")

    def close(self) -> None:
        """Flush & close the Langfuse client if active."""
        if self.enabled and self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass