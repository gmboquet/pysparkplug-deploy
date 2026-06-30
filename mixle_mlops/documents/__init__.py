"""Document ingestion: extract text from uploaded files and chunk it for embedding/retrieval."""
from __future__ import annotations

from .parse import (
    DocumentParseError,
    chunk_text,
    extract_text,
    parse_and_chunk,
)

__all__ = [
    "DocumentParseError",
    "extract_text",
    "chunk_text",
    "parse_and_chunk",
]
