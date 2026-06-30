"""Dataset-generation routes — mixle's home turf: a generative library makes labeled data with verifiable
labels.

``POST /v1/datasets/generate`` ({source, model, n, schema?, prompt?, format, seed?, columns?}) pulls the
model from the registry, samples/drives it, exports the bytes into the blob store, records a
:class:`~mixle_mlops.datasets.models.DatasetArtifact`, and returns the artifact ref. ``GET /v1/datasets/{id}``
fetches a previously-generated artifact's metadata. Both require an authenticated user.

Wiring (integrator):
  * ``from .routes import datasets`` then ``app.include_router(datasets.router, prefix="/v1", tags=["datasets"])``
    in ``gateway/app.py``.
  * optional: ``from ..datasets import models as _datasets  # noqa: F401`` inside
    ``storage/db.init_db`` so the table is created up-front (the route also creates it defensively).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Session, select

from ...accounts.models import User
from ...datasets.export import export_dataset
from ...datasets.generate import DatasetSpec, GeneratedDataset, generate_dataset
from ...datasets.models import DatasetArtifact
from ...multimodal.store import get_blob_store
from ...storage.db import get_engine, get_session
from ..auth import require_user

router = APIRouter(prefix="/datasets", tags=["datasets"])

_table_ready = False


def _ensure_table() -> None:
    """Idempotently create the DatasetArtifact table (so this works without editing init_db)."""
    global _table_ready
    if _table_ready:
        return
    SQLModel.metadata.create_all(get_engine(), tables=[DatasetArtifact.__table__])
    _table_ready = True


class GenerateBody(BaseModel):
    source: str = "mixle"                          # "mixle" | "llm"
    model: str
    n: int = Field(default=100, ge=1, le=100_000)
    seed: int = 0
    schema_: dict[str, str] | None = Field(default=None, alias="schema")
    prompt: str | None = None
    format: str = "jsonl"                          # "jsonl" | "csv" | "parquet"
    columns: list[str] | None = None

    model_config = {"populate_by_name": True}


def _persist(
    session: Session, user: User, dataset: GeneratedDataset, artifact_ref: dict[str, Any], fmt: str
) -> DatasetArtifact:
    row = DatasetArtifact(
        user_id=getattr(user, "id", None),
        source=dataset.source,
        model=dataset.model,
        fmt=fmt,
        n_rows=dataset.n_rows,
        seed=dataset.seed,
        prompt=dataset.prompt,
        blob_id=artifact_ref.get("id"),
        blob_url=artifact_ref.get("url"),
        schema_def=DatasetArtifact.encode_schema(dataset.schema),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


@router.post("/generate")
async def generate(
    body: GenerateBody,
    request: Request,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    """Generate a labeled dataset, export it to the blob store, and return the artifact ref."""
    _ensure_table()
    registry = request.app.state.registry
    spec = DatasetSpec(
        source=body.source,
        model=body.model,
        n=body.n,
        seed=body.seed,
        schema=body.schema_,
        prompt=body.prompt,
        fmt=body.format,
        columns=body.columns,
    )
    try:
        dataset = await generate_dataset(spec, registry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        artifact_ref = export_dataset(dataset, body.format, store=get_blob_store())
    except ValueError as exc:                      # unknown format
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:                     # missing optional dep (parquet)
        raise HTTPException(status_code=501, detail=str(exc))

    row = _persist(session, user, dataset, artifact_ref, body.format)
    result = row.to_dict()
    result["artifact"] = artifact_ref
    return result


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    """Metadata for a previously-generated dataset artifact."""
    _ensure_table()
    row = session.get(DatasetArtifact, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id!r} not found")
    return row.to_dict()


@router.get("")
async def list_datasets(
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    """List the authenticated user's generated datasets (most recent first)."""
    _ensure_table()
    stmt = select(DatasetArtifact).where(
        DatasetArtifact.user_id == getattr(user, "id", None)
    ).order_by(DatasetArtifact.created_at.desc())
    rows = session.exec(stmt).all()
    return {"object": "list", "data": [r.to_dict() for r in rows]}
