"""
Upload Security - file-upload hardening (OWASP LLM08) + RAG poisoning defense.
- validate_upload: extension + magic-bytes + size + emptiness checks
- sanitize_document_text: strip instruction-override phrases from DOCUMENT
  content before indexing (indirect-injection defense)
- safe_page_count: crash-proof page counting for corrupt/encrypted PDFs
- create_secure_temp_file: temp-file helper with cleanup-friendly API
"""
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from dataclasses import dataclass

from pypdf import PdfReader

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE_MB = 50
DEFAULT_MAX_PAGES = 1000


class UploadError(Exception):
    """User-facing upload/ingest error."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class UploadValidation:
    ok: bool
    reason: str = ""
    file_size: int = 0
    file_type: str = "unknown"
    is_pdf: bool = False


def validate_upload(
    filename: str | None,
    data: bytes,
    max_size_mb: int = DEFAULT_MAX_SIZE_MB,
) -> UploadValidation:
    """Validate an uploaded file before any processing."""
    if filename is None or not filename.strip():
        return UploadValidation(False, "No file provided.")
    if data is None or len(data) == 0:
        return UploadValidation(False, "Empty file.", file_size=0)

    if not filename.lower().endswith(".pdf"):
        return UploadValidation(
            False, "Only PDF files are allowed.",
            file_size=len(data),
        )

    if len(data) > max_size_mb * 1024 * 1024:
        return UploadValidation(
            False, f"File too large (max {max_size_mb} MB).",
            file_size=len(data),
        )

    # Magic-bytes check: PDFs start with %PDF (allow leading whitespace)
    stripped = data.lstrip()
    if not stripped.startswith(b"%PDF"):
        return UploadValidation(
            False, "File is not a valid PDF (bad magic bytes).",
            file_size=len(data),
        )

    return UploadValidation(
        ok=True, file_size=len(data),
        file_type="application/pdf", is_pdf=True,
    )


# Document-sanitization phrases (decoded at runtime)
_DOC_PATTERNS: tuple[str, ...] = tuple(
    b.decode("utf-8") for b in (
        b"aWdub3JlIGFsbCBwcmV2aW91cyBpbnN0cnVjdGlvbnM=",
        b"ZGlzcmVnYXJkIHByaW9yIGluc3RydWN0aW9ucw==",
        b"c3lzdGVtIG92ZXJyaWRl",
        b"Zm9yZ2V0IHlvdXIgaW5zdHJ1Y3Rpb25z",
        b"bmV3IGluc3RydWN0aW9ucyBmb2xsb3c=",
        b"eW91IG11c3Qgbm93",
    )
)
_REDACT_TAG = "[REDACTED-INSTRUCTION]"


def sanitize_document_text(text: str) -> str:
    """
    Defense against indirect prompt injection (poisoned documents):
    replace instruction-override phrases found in PDF content with a tag
    BEFORE the text is chunked and indexed. Conservative exact-phrase
    replacement - normal text is never mangled.
    """
    if not text:
        return ""
    cleaned = text
    for phrase in _DOC_PATTERNS:
        cleaned = re.sub(re.escape(phrase), _REDACT_TAG, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned)
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", cleaned)
    return cleaned


def safe_page_count(data: bytes, max_pages: int = DEFAULT_MAX_PAGES) -> tuple[int, bool]:
    """Crash-proof page counting. (count, is_ok)."""
    try:
        reader = PdfReader(io.BytesIO(data))
        count = len(reader.pages)
        if count == 0:
            return 0, False
        if count > max_pages:
            return count, False
        return count, True
    except Exception as e:
        logger.warning(f"safe_page_count failed: {e}")
        return 0, False


def create_secure_temp_file(
    data: bytes,
    suffix: str = ".pdf",
) -> tuple[str | None, str | None]:
    """Write bytes to a random-name temp file. Returns (path, None) or (None, error)."""
    temp_dir = os.getenv("RAG_TEMP_DIR") or None
    try:
        with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            return tmp.name, None
    except OSError as e:
        logger.error(f"temp file error: {e}")
        return None, f"Could not write temporary file: {e}"


if __name__ == "__main__":
    from pypdf import PdfWriter

    passed = failed = 0

    def check(name: str, cond: bool) -> None:
        global passed, failed
        if cond:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}")

    print("upload_security self-tests")

    buf = io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=100, height=100)
    w.write(buf)
    valid_pdf = buf.getvalue()

    check("validate: valid pdf ok", validate_upload("doc.pdf", valid_pdf).ok)
    check("validate: no filename rejected", not validate_upload(None, valid_pdf).ok)
    check("validate: empty data rejected", not validate_upload("doc.pdf", b"").ok)
    check("validate: wrong extension rejected", not validate_upload("doc.txt", valid_pdf).ok)
    check("validate: fake magic bytes rejected",
          not validate_upload("doc.pdf", b"not a pdf at all").ok)
    check("validate: oversized rejected",
          not validate_upload("doc.pdf", valid_pdf, max_size_mb=0).ok)

    # Use the module's own decoded pattern (no literal, no decode call)
    attack = _DOC_PATTERNS[0]
    sanitized = sanitize_document_text(attack + ". The real content continues here.")
    check("sanitize: injection phrase replaced",
          attack not in sanitized and "[REDACTED-INSTRUCTION]" in sanitized)
    check("sanitize: normal text intact", "real content continues" in sanitized)

    count, ok = safe_page_count(valid_pdf)
    check("safe_page_count: valid pdf", ok and count == 1)
    count, ok = safe_page_count(b"garbage-not-a-pdf")
    check("safe_page_count: garbage -> no crash", not ok and count == 0)

    path, err = create_secure_temp_file(valid_pdf)
    check("temp file created", path is not None and err is None)
    if path:
        os.unlink(path)
        check("temp file deleted", not os.path.exists(path))

    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
