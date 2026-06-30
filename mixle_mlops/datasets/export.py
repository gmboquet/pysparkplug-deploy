"""Materialise a :class:`~mixle_mlops.datasets.generate.GeneratedDataset` into the blob store.

Three formats, one return shape (``{id, url, n_rows, schema, format}``): :func:`to_jsonl` and :func:`to_csv`
use only the stdlib; :func:`to_parquet` lazily imports ``pandas`` + ``pyarrow`` (report the ``datasets``
extra). Each writes the bytes through the pluggable :class:`~mixle_mlops.multimodal.store.BlobStore`, so the
file is addressable by an opaque id and served back over ``/v1/files/{id}/content`` — local fs now, S3 in
cloud, same code.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any

from ..multimodal.store import BlobStore, get_blob_store
from .generate import GeneratedDataset


def _store(store: BlobStore | None) -> BlobStore:
    return store if store is not None else get_blob_store()


def _result(record: Any, dataset: GeneratedDataset, fmt: str) -> dict[str, Any]:
    return {
        "id": record.id,
        "url": record.url,
        "n_rows": dataset.n_rows,
        "schema": dict(dataset.schema),
        "format": fmt,
        "blob_id": record.id,
    }


def _filename(dataset: GeneratedDataset, ext: str) -> str:
    base = (dataset.model or dataset.source or "dataset").replace("/", "_")
    return f"{base}.{ext}"


def to_jsonl(dataset: GeneratedDataset, *, store: BlobStore | None = None) -> dict[str, Any]:
    """One JSON object per line (the canonical training-data interchange format)."""
    buf = io.StringIO()
    for row in dataset.rows:
        buf.write(json.dumps(row, default=str))
        buf.write("\n")
    data = buf.getvalue().encode("utf-8")
    record = _store(store).put(
        data, filename=_filename(dataset, "jsonl"), content_type="application/x-ndjson"
    )
    return _result(record, dataset, "jsonl")


def to_csv(dataset: GeneratedDataset, *, store: BlobStore | None = None) -> dict[str, Any]:
    """CSV with a header row of the union of all column names (schema order first, then any extras)."""
    columns = list(dataset.schema.keys())
    for row in dataset.rows:                       # include any columns not captured in the schema
        for k in row:
            if k not in columns:
                columns.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in dataset.rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    data = buf.getvalue().encode("utf-8")
    record = _store(store).put(data, filename=_filename(dataset, "csv"), content_type="text/csv")
    return _result(record, dataset, "csv")


def to_parquet(dataset: GeneratedDataset, *, store: BlobStore | None = None) -> dict[str, Any]:
    """Columnar Parquet via lazy ``pandas`` + ``pyarrow`` (the ``datasets`` extra).

    Raises :class:`RuntimeError` with an actionable message if the optional deps are absent.
    """
    try:
        import pandas as pd  # noqa: F401  (lazy optional dep)
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only when the extra is missing
        raise RuntimeError(
            "parquet export needs pandas + pyarrow; install the 'datasets' extra "
            "(pip install 'mixle-mlops[datasets]')"
        ) from exc
    import pandas as pd

    columns = list(dataset.schema.keys())
    frame = pd.DataFrame(dataset.rows, columns=columns or None)
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    record = _store(store).put(
        buf.getvalue(), filename=_filename(dataset, "parquet"),
        content_type="application/vnd.apache.parquet",
    )
    return _result(record, dataset, "parquet")


_EXPORTERS = {"jsonl": to_jsonl, "csv": to_csv, "parquet": to_parquet}


def export_dataset(
    dataset: GeneratedDataset, fmt: str = "jsonl", *, store: BlobStore | None = None
) -> dict[str, Any]:
    """Dispatch to the right exporter by ``fmt`` (``jsonl`` | ``csv`` | ``parquet``)."""
    try:
        exporter = _EXPORTERS[fmt]
    except KeyError:
        raise ValueError(f"unknown export format {fmt!r} (expected one of {sorted(_EXPORTERS)})")
    return exporter(dataset, store=store)
