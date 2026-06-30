"""Persistence for the RAG layer: stored vectors and uploaded-document records.

Follows the ``accounts/models.py`` pattern (uuid pk, utc timestamps, JSON-string ``meta``). The vector itself is
stored as a JSON-encoded list of floats in :class:`VectorItem.vector_json` — portable across SQLite (local) and
Postgres (cloud) without a pgvector dependency; the cosine ranking happens in numpy in :mod:`vectorstore`. A
cloud deployment can swap :class:`~mixle_mlops.rag.vectorstore.PgVectorStore` in for server-side ANN.

These tables are created idempotently by ``_ensure_tables()`` in :mod:`vectorstore` (so the RAG layer works
without editing ``storage.db.init_db``), and also registered for ``init_db`` once the integrator adds the import.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class VectorItem(SQLModel, table=True):
    """One embedded text chunk in a user's vector store.

    Scoped by ``(user_id, namespace)`` — ``namespace`` separates e.g. ``conversation`` memory from ``document``
    chunks while sharing one retriever. ``source_id`` ties chunks back to their origin (a conversation id or a
    document id) so a source can be re-indexed or deleted as a unit.
    """

    __tablename__ = "rag_vector_item"

    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: str = Field(index=True)
    namespace: str = Field(default="default", index=True)   # "conversation" | "document" | custom
    source_id: str | None = Field(default=None, index=True)  # conversation_id or document_id
    text: str = ""
    vector_json: str = ""                                    # JSON list[float]
    meta_json: str | None = None                             # JSON dict of extra context
    created_at: datetime = Field(default_factory=_now)

    def meta(self) -> dict[str, Any]:
        if not self.meta_json:
            return {}
        try:
            obj = json.loads(self.meta_json)
            return obj if isinstance(obj, dict) else {}
        except (ValueError, TypeError):
            return {}


class Document(SQLModel, table=True):
    """An uploaded document the user ingested for retrieval (the blob lives in the :class:`BlobStore`)."""

    __tablename__ = "rag_document"

    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: str = Field(index=True)
    filename: str = ""
    content_type: str = ""
    blob_id: str | None = Field(default=None, index=True)    # BlobStore id of the raw bytes
    n_chunks: int = 0
    n_chars: int = 0
    created_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "rag.document",
            "filename": self.filename,
            "content_type": self.content_type,
            "blob_id": self.blob_id,
            "n_chunks": self.n_chunks,
            "n_chars": self.n_chars,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
