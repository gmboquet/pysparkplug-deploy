"""File upload/retrieve routes for the chat UI's multimodal flow: upload an image, get back an id + URL, then
reference that id in a chat message's ``image_url`` part. Backed by the pluggable :class:`BlobStore` (local fs
now, S3 in cloud). OpenAI-files-shaped so existing clients/UIs work.

Wiring (integrator): ``app.include_router(files.router, prefix="/v1", tags=["files"])`` in ``gateway/app.py``."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import Response

from ...accounts.models import User
from ...multimodal.content import MultimodalError, guard_image
from ...multimodal.store import get_blob_store
from ..auth import require_user

router = APIRouter()


@router.post("/files")
async def upload_file(file: UploadFile, user: User = Depends(require_user)):
    """Multipart upload → blob store. Returns ``{id, url, ...}`` (OpenAI-files-compatible)."""
    data = await file.read()
    content_type = file.content_type or "application/octet-stream"
    # Guard images by mime/size; non-image files (e.g. future doc support) pass the image guard by being
    # skipped — only enforce the guard when the upload claims to be an image.
    if content_type.split(";", 1)[0].strip().lower().startswith("image/"):
        try:
            guard_image(content_type=content_type, size=len(data))
        except MultimodalError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    record = get_blob_store().put(data, filename=file.filename or "upload", content_type=content_type)
    return record.to_dict()


@router.get("/files/{file_id}")
async def get_file(file_id: str, user: User = Depends(require_user)):
    """Metadata for an uploaded file."""
    store = get_blob_store()
    try:
        record, _ = store.get(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"file {file_id!r} not found")
    return record.to_dict()


@router.get("/files/{file_id}/content")
async def get_file_content(file_id: str, user: User = Depends(require_user)):
    """Raw bytes of an uploaded file (what the ``url`` in the upload response points at)."""
    store = get_blob_store()
    try:
        record, data = store.get(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"file {file_id!r} not found")
    return Response(content=data, media_type=record.content_type)
