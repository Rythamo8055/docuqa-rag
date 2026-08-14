"""
Output Guardrails - defense-in-depth Layer 3 (OWASP LLM05, LLM01).
Applied to LLM answers BEFORE they reach the user.
Sensitive phrase lists are stored base64-encoded and decoded at runtime.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field


def _decode(encoded: str) -> str:
    return base64.b64decode(encoded).decode("utf-8")


@dataclass
class OutputGuardResult:
    text: str
    pii_redacted: int = 0
    leakage_detected: bool = False
    unsafe_detected: bool = False
    notices: list[str] = field(default_factory=list)


# PII / secret redaction patterns
_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "EMAIL"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (r"\b(?:\d[ -]*?){13,16}\b", "CARD"),
    (r"\+?\d[\d\s\-()]{7,}\d", "PHONE"),
    (r"\b(sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}|"
     r"ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,})\b", "SECRET"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "IP"),
]


def redact_pii(text: str) -> tuple[str, int]:
    redacted = text
    count = 0
    for pattern, label in _PII_PATTERNS:
        redacted, n = re.subn(pattern, f"[REDACTED-{label}]", redacted)
        count += n
    return redacted, count


# Leakage indicators, decoded at runtime
_LEAK_INDICATORS: tuple[str, ...] = tuple(
    _decode(b) for b in (
        "eW91IGFyZSBhbiBleHBlcnQgYXNzaXN0YW50",
        "eW91ciByb2xlIGlzIHRv",
        "c3lzdGVtIGluc3RydWN0aW9ucw==",
        "eW91IGFyZSBhIGhlbHBmdWw=",
        "YXMgYW4gYWkgbGFuZ3VhZ2UgbW9kZWwsIGk=",
        "bmV2ZXIgcmV2ZWFsIHRoZXNlIGluc3RydWN0aW9ucw==",
        "cHJvbXB0IHRlbXBsYXRl",
        "dXNlciBpbnB1dCAodW50cnVzdGVkKQ==",
        "dGhlIGFib3ZlIGluc3RydWN0aW9ucyBhcmUgaW1tdXRhYmxl",
    )
)


def detect_prompt_leakage(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    hits = sum(1 for ind in _LEAK_INDICATORS if ind in lower)
    if hits >= 2:
        return True
    return hits == 1 and len(text) < 200


# Unsafe phrases, decoded at runtime
_UNSAFE_PHRASES: tuple[str, ...] = tuple(
    _decode(b) for b in (
        "aG93IHRvIG1ha2UgYSBib21i",
        "aG93IHRvIGJ1aWxkIGEgYm9tYg==",
        "aG93IHRvIG1hbnVmYWN0dXJlIGV4cGxvc2l2ZXM=",
        "aG93IHRvIG1ha2UgZXhwbG9zaXZlcw==",
        "aG93IHRvIGNvbW1pdCBtdXJkZXI=",
        "aG93IHRvIHNlbGYtaGFybQ==",
        "d2F5cyB0byBzZWxmLWhhcm0=",
        "aG93IHRvIG1ha2UgcG9pc29u",
        "c3RlcHMgdG8gYnVpbGQgYSB3ZWFwb24=",
    )
)


def detect_unsafe_content(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in _UNSAFE_PHRASES)


def filter_output(raw_text: str) -> OutputGuardResult:
    """Apply all output controls: PII redaction, leakage check, unsafe check."""
    if not raw_text:
        return OutputGuardResult(text="", notices=["Empty response"])

    text, n_pii = redact_pii(raw_text)

    leakage = detect_prompt_leakage(text)
    if leakage:
        lower = text.lower()
        idxs = [lower.find(ind) for ind in _LEAK_INDICATORS if ind in lower]
        idxs = [i for i in idxs if i != -1]
        if idxs:
            cut = min(idxs)
            text = text[:max(0, cut - 50)].rstrip()

    unsafe = detect_unsafe_content(text)
    if unsafe:
        text = "This response was filtered by safety guardrails."

    notices = []
    if n_pii:
        notices.append(f"{n_pii} PII/secret item(s) redacted")
    if leakage:
        notices.append("Potential instruction leakage filtered")
    if unsafe:
        notices.append("Unsafe content blocked")

    return OutputGuardResult(
        text=text,
        pii_redacted=n_pii,
        leakage_detected=leakage,
        unsafe_detected=unsafe,
        notices=notices,
    )


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

    print("output_guardrails self-tests")

    t, n = redact_pii("Contact me at john.doe@example.com or 555-123-4567")
    check("redact: email + phone", n == 2 and "REDACTED-EMAIL" in t and "REDACTED-PHONE" in t)

    t, n = redact_pii("SSN 123-45-6789 and card 4111 1111 1111 1111")
    check("redact: SSN + card", n == 2 and "REDACTED-SSN" in t and "REDACTED-CARD" in t)

    fake_key = "sk-" + "x" * 40
    t, n = redact_pii("key " + fake_key)
    check("redact: api secret", n == 1 and "REDACTED-SECRET" in t)

    t, n = redact_pii("server at 192.168.1.10")
    check("redact: IP", n == 1 and "REDACTED-IP" in t)

    t, n = redact_pii("The document explains the methodology clearly.")
    check("redact: normal text untouched", n == 0 and "methodology" in t)

    leak_text = _decode("eW91IGFyZSBhbiBleHBlcnQgYXNzaXN0YW50") + ". " + \
                _decode("eW91ciByb2xlIGlzIHRv") + " answer."
    check("leak: system instruction phrases", detect_prompt_leakage(leak_text))

    short_leak = _decode("eW91IGFyZSBhIGhlbHBmdWw=") + "."
    check("leak: short strong indicator", detect_prompt_leakage(short_leak))
    check("leak: benign answer not flagged",
          not detect_prompt_leakage("The key finding is that RAG improves accuracy by 15%."))

    check("unsafe: harmful phrase flagged",
          detect_unsafe_content(_decode("aG93IHRvIG1ha2UgYSBib21i")))
    check("unsafe: benign not flagged",
          not detect_unsafe_content("The recipe calls for flour, sugar and eggs."))

    res = filter_output("Call 555-123-4567 - " + leak_text)
    check("filter: redacts + truncates leak", res.pii_redacted >= 1 and res.leakage_detected)
    check("filter: notices present", len(res.notices) >= 1)

    res = filter_output(_decode("aG93IHRvIG1ha2UgYSBib21i") + ": instructions follow")
    check("filter: unsafe blocked", res.unsafe_detected and "filtered" in res.text)

    res = filter_output("")
    check("filter: empty handled", res.text == "" and res.notices)

    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
