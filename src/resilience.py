"""
Resilience Layer — retries, circuit breaker, graceful degradation.

- retry():               exponential backoff + jitter for flaky calls
- CircuitBreaker:        fail-fast after repeated failures, auto-recovery
- safe_llm_call():       never-raise LLM wrapper (returns error message)
- empty_answer_fallback(): user-friendly fallback for empty LLM output
- ErrorBucket:           collect non-fatal errors for "degraded mode" UI
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 8.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)


def retry(config: RetryConfig | None = None):
    """
    Decorator: retry a callable with exponential backoff + jitter.
    Re-raises the last exception after exhausting attempts.
    """
    cfg = config or RetryConfig()

    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except cfg.retryable_exceptions as e:
                    attempt += 1
                    if attempt >= cfg.max_attempts:
                        logger.error(
                            f"{fn.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise
                    delay = min(
                        cfg.base_delay * (cfg.backoff_factor ** (attempt - 1)),
                        cfg.max_delay,
                    )
                    if cfg.jitter:
                        delay *= random.uniform(0.7, 1.3)
                    logger.warning(
                        f"{fn.__name__} attempt {attempt} failed ({e}); "
                        f"retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open (fail-fast)."""

    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(message)
        self.message = message


class CircuitBreaker:
    """
    Classic circuit breaker: CLOSED → OPEN (after N failures) → HALF_OPEN
    (after reset_timeout, one test call) → CLOSED or OPEN again.
    """

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._opened_at = 0.0
        self._lock = threading.Lock()

    def call(self, fn: Callable, *args, **kwargs):
        """Execute fn through the breaker. Raises CircuitOpenError when OPEN."""
        with self._lock:
            now = time.monotonic()
            if self._state == "OPEN":
                if now - self._opened_at >= self.reset_timeout:
                    self._state = "HALF_OPEN"
                else:
                    raise CircuitOpenError()

        try:
            result = fn(*args, **kwargs)
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._state = "OPEN"
                    self._opened_at = time.monotonic()
            raise

        # Success
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"
        return result

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"

    def record_success(self) -> None:
        """Record a successful LLM call — reset failure count, close circuit."""
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"

    def record_failure(self) -> None:
        """Record a failed LLM call — increment failures, open circuit if threshold reached."""
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = "OPEN"
                self._opened_at = time.monotonic()


def safe_llm_call(
    router: Any,
    prompt: str,
    stream: bool = False,
    max_attempts: int = 2,
) -> tuple[Any | None, str, str, str]:
    """
    Never-raise LLM wrapper around router.generate().

    Returns: (result, provider, model, error_message)
    - On success: (result, provider, model, "")
    - On failure: (None, "none", "none", human_readable_error)
    """
    if router is None:
        return None, "none", "none", "LLM router not available."

    attempt = 0
    last_err = ""
    while attempt < max_attempts:
        try:
            result, provider, model = router.generate(prompt, stream=stream)
            return result, provider, model, ""
        except Exception as e:
            attempt += 1
            last_err = str(e)
            if attempt < max_attempts:
                time.sleep(0.5 * attempt)
    logger.error(f"safe_llm_call failed: {last_err}")
    return None, "none", "none", last_err


def empty_answer_fallback(question: str, chunks_count: int, reason: str = "") -> tuple[str, str, str]:
    """User-friendly fallback when the LLM returns nothing useful.

    Returns:
        (answer_text, provider, model) — provider is always "rule-based",
        model is always None.
    """
    if chunks_count == 0:
        return (
            "Information not found in the provided document. "
            "No relevant content was retrieved for this question.",
            "rule-based",
            None,
        )
    return (
        "I could not generate a complete answer for this question. "
        f"{chunks_count} relevant section(s) were retrieved, but the "
        "generation step failed. Please try rephrasing your question, "
        "or try again in a moment.",
        "rule-based",
        None,
    )


class ErrorBucket:
    """Collects non-fatal errors during a request (degraded-mode UI)."""

    def __init__(self):
        self._errors: list[dict] = []

    def add(self, code: str, message: str) -> None:
        self._errors.append({"code": code, "message": message})
        logger.warning(f"[{code}] {message}")

    def has_errors(self) -> bool:
        return bool(self._errors)

    def to_list(self) -> list[dict]:
        return list(self._errors)

    def summary(self) -> str:
        return ", ".join(e["code"] for e in self._errors)


# ── Self-tests ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    passed = failed = 0

    def check(name: str, cond: bool) -> None:
        global passed, failed
        if cond:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}")

    print("resilience self-tests")

    # retry: flaky fn succeeds on 3rd attempt
    calls = {"n": 0}

    @retry(RetryConfig(max_attempts=4, base_delay=0.01, max_delay=0.05, jitter=False))
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("flaky")
        return "ok"

    check("retry: succeeds after failures", flaky() == "ok" and calls["n"] == 3)

    # retry: exhausts
    @retry(RetryConfig(max_attempts=2, base_delay=0.01, max_delay=0.02, jitter=False))
    def always_fails():
        raise ValueError("always")

    try:
        always_fails()
        check("retry: exhausts -> raises", False)
    except ValueError:
        check("retry: exhausts -> raises", True)

    # circuit breaker
    cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

    def boom():
        raise RuntimeError("boom")

    for _ in range(3):
        try:
            cb.call(boom)
        except RuntimeError:
            pass
    check("circuit: opens after 3 failures", cb.state == "OPEN")
    try:
        cb.call(lambda: "x")
        check("circuit: fail-fast when OPEN", False)
    except CircuitOpenError:
        check("circuit: fail-fast when OPEN", True)
    # simulate timeout passage
    cb._opened_at = time.monotonic() - 120
    check("circuit: half-open after timeout", cb.call(lambda: "ok") == "ok" and cb.state == "CLOSED")

    # safe_llm_call
    class FakeRouter:
        def __init__(self, fail: bool):
            self.fail = fail

        def generate(self, prompt, stream=False):
            if self.fail:
                raise RuntimeError("provider down")
            return "answer", "groq", "model-x"

    res, prov, model, err = safe_llm_call(FakeRouter(False), "p")
    check("safe_llm: success passthrough", res == "answer" and prov == "groq" and err == "")
    res, prov, model, err = safe_llm_call(FakeRouter(True), "p", max_attempts=2)
    check("safe_llm: failure -> no raise", res is None and prov == "none" and err != "")
    res, prov, model, err = safe_llm_call(None, "p")
    check("safe_llm: None router guarded", res is None and "not available" in err)

    # fallbacks
    f0, fp0, fm0 = empty_answer_fallback("q", 0)
    f5, fp5, fm5 = empty_answer_fallback("q", 5)
    check("fallback: zero chunks -> not-found", "not found" in f0)
    check("fallback: has chunks -> retry hint", "rephrasing" in f5)
    check("fallback: returns rule-based provider", fp0 == "rule-based" and fm5 is None)

    # ErrorBucket
    eb = ErrorBucket()
    check("bucket: empty initially", not eb.has_errors())
    eb.add("LLM_FAIL", "bad")
    eb.add("RETRIEVE_FAIL", "worse")
    check("bucket: collects + summarizes",
          eb.has_errors() and eb.summary() == "LLM_FAIL, RETRIEVE_FAIL")

    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)