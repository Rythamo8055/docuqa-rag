"""
PDF Processing and Chunking Module.

Handles extraction of text from PDF files and splitting into semantic chunks
for vector indexing.

Features:
- Token-aware recursive character splitting (zero external splitter deps)
- Parent-Child chunking: index small chunks (precise retrieval),
  feed full parent sections to the LLM (better answers)
"""
from typing import Dict, List, Any
from pathlib import Path
import logging

from pypdf import PdfReader
import tiktoken

logger = logging.getLogger(__name__)

# Chunking parameters (assessment: ~500-1000 tokens/parent, 10-15% overlap)
DEFAULT_CHILD_SIZE = 300      # small chunks for retrieval (indexed)
DEFAULT_CHILD_OVERLAP = 45    # ~15% of child
DEFAULT_PARENT_SIZE = 800     # large chunks for LLM context (generation)
DEFAULT_PARENT_OVERLAP = 120  # ~15% of parent
SEPARATORS = ("\n\n", "\n", ". ", " ")

_ENCODING = "cl100k_base"


def count_tokens(text: str, encoding_name: str = _ENCODING) -> int:
    """Count tokens in a text string using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def _tail(text: str, n_tokens: int) -> str:
    """Return the last n_tokens tokens of text (for chunk overlap)."""
    if n_tokens <= 0:
        return ""
    enc = tiktoken.get_encoding(_ENCODING)
    tokens = enc.encode(text)
    return enc.decode(tokens[-n_tokens:]) if tokens else ""


def recursive_split(
    text: str,
    chunk_size: int = DEFAULT_PARENT_SIZE,
    chunk_overlap: int = DEFAULT_PARENT_OVERLAP,
    separators: tuple = SEPARATORS,
) -> List[str]:
    """
    Token-aware recursive character splitting.

    1. Splits text by separators (paragraph → line → sentence → word)
    2. Merges small pieces back up to chunk_size
    3. Applies token-based overlap between chunks
    """
    # Phase 1: break oversized pieces down by separator
    pieces = [text]
    for sep in separators:
        expanded = []
        for piece in pieces:
            if count_tokens(piece) <= chunk_size or sep == " ":
                expanded.append(piece)
            else:
                expanded.extend(
                    x.strip() for x in piece.split(sep) if x.strip()
                )
        pieces = expanded

    # Phase 2: merge pieces into ~chunk_size chunks with overlap
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif count_tokens(current + "\n" + piece) <= chunk_size:
            current += "\n" + piece
        else:
            chunks.append(current)
            current = _tail(current, chunk_overlap) + "\n" + piece
    if current:
        chunks.append(current)
    return chunks


def extract_pdf_text(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text from a PDF file, preserving page numbers.

    Returns: {"total_pages": int, "pages": {1: "...", 2: "...", ...}}
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = {}
    for i, page in enumerate(reader.pages, start=1):
        try:
            pages[i] = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Failed to extract text from page {i}: {e}")
            pages[i] = ""

    logger.info(f"Extracted {len(reader.pages)} pages from {pdf_path.name}")
    return {"total_pages": len(reader.pages), "pages": pages}


def parent_child_chunking(
    pages: Dict[int, str],
    child_size: int = DEFAULT_CHILD_SIZE,
    child_overlap: int = DEFAULT_CHILD_OVERLAP,
    parent_size: int = DEFAULT_PARENT_SIZE,
    parent_overlap: int = DEFAULT_PARENT_OVERLAP,
) -> Dict[str, Any]:
    """
    Parent-Child (small-to-big) chunking.

    - Parents: ~800-token sections → fed to the LLM for generation
    - Children: ~300-token sub-chunks → embedded & indexed for retrieval

    Returns: {
        "children": [{"chunk_id", "parent_id", "page", "text", "tokens"}],
        "parents":  [{"parent_id", "page", "text", "tokens"}],
        "total_pages": int
    }
    """
    parents: List[Dict] = []
    children: List[Dict] = []
    parent_id = 1
    child_id = 1

    for page_num in sorted(pages.keys()):
        text = pages[page_num]
        if not text.strip():
            continue

        for parent_text in recursive_split(text, parent_size, parent_overlap):
            parents.append({
                "parent_id": parent_id,
                "page": page_num,
                "text": parent_text,
                "tokens": count_tokens(parent_text),
            })
            for child_text in recursive_split(parent_text, child_size, child_overlap):
                children.append({
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "page": page_num,
                    "text": child_text,
                    "tokens": count_tokens(child_text),
                })
                child_id += 1
            parent_id += 1

    logger.info(
        f"Parent-Child chunking: {len(parents)} parents, "
        f"{len(children)} children from {len(pages)} pages"
    )
    return {
        "children": children,
        "parents": parents,
        "total_pages": len(pages),
    }


def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline: extract text → parent-child chunking.

    Returns the parent_child_chunking result dict.
    """
    extraction = extract_pdf_text(pdf_path)
    return parent_child_chunking(extraction["pages"])