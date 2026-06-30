"""Persistence for generated dataset artifacts.

One :class:`DatasetArtifact` row records a materialised dataset: where its bytes live (a blob id + URL in
the :class:`~mixle_mlops.multimodal.store.BlobStore`), how it was produced (source/model/seed/prompt), its
row count, and the inferred column schema. Follows the ``accounts/models.py`` pattern (uuid pk, utc
timestamps, JSON-encoded blob columns).

The table is created defensively at first use via :func:`ensure_dataset_table` (see this package's route),
so it works without editing ``storage/db.init_db`` — though the integrator is asked to add the import there
too so it is created up-front alongside the other tables.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return "ds-" + uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DatasetArtifact(SQLModel, table=True):
    """Metadata for one generated dataset (the bytes live in the blob store)."""

    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: str | None = Field(default=None, index=True)        # who generated it (nullable for anon)
    source: str = Field(index=True)                              # "mixle" | "llm"
    model: str | None = Field(default=None, index=True)          # registry model id used
    fmt: str = Field(default="jsonl")                            # "jsonl" | "csv" | "parquet"
    n_rows: int = 0
    seed: int | None = None
    prompt: str | None = None                                    # llm-source generation prompt
    blob_id: str | None = Field(default=None, index=True)        # blob store id of the materialised file
    blob_url: str | None = None                                  # gateway path serving the bytes back
    schema_def: str | None = None                              # JSON-encoded {column: type} schema
    created_at: datetime = Field(default_factory=_now)

    # --- convenience helpers (not columns) ---
    def schema_dict(self) -> dict[str, Any]:
        if not self.schema_def:
            return {}
        try:
            obj = json.loads(self.schema_def)
            return obj if isinstance(obj, dict) else {}
        except (ValueError, TypeError):
            return {}

    @staticmethod
    def encode_schema(schema: dict[str, Any] | None) -> str | None:
        return None if schema is None else json.dumps(schema, default=str)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "dataset",
            "source": self.source,
            "model": self.model,
            "format": self.fmt,
            "n_rows": self.n_rows,
            "seed": self.seed,
            "prompt": self.prompt,
            "blob_id": self.blob_id,
            "url": self.blob_url,
            "schema": self.schema_dict(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
