"""Normalize ``ChatMessage`` content parts before they reach a backend. Image parts can reference a blob three
ways, and we resolve them all to the ``data:``/``https`` ``image_url`` form the OpenAI-compatible vision backends
expect:

  * ``{"url": "data:image/png;base64,..."}``   — already inline, passed through (size/mime guarded)
  * ``{"url": "https://..."}``                  — remote, passed through
  * ``{"url": "/v1/files/file-abc/content"}`` or ``{"file_id": "file-abc"}`` — an uploaded blob; resolved to a
    ``data:`` URL by reading it from the :class:`BlobStore`.

This keeps the gateway backend-agnostic: by the time a request leaves ``normalize_messages`` every image is a
self-contained ``image_url`` part, so :class:`OpenAICompatAdapter` just forwards it to the vision LLM."""
from __future__ import annotations

import base64
import binascii
import re

from ..core.adapters import ChatMessage, ContentPart, ImagePart, TextPart
from .store import BlobStore, get_blob_store

# Reasonable defaults; a vision request with a 30 MB image is almost always a mistake.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<b64>;base64)?,(?P<payload>.*)$", re.DOTALL)
_FILE_PATH_RE = re.compile(r"^/v1/files/(?P<id>[\w.-]+?)(?:/content)?$")


class MultimodalError(Exception):
    """Bad image content (oversize, unsupported mime, unknown file id). → HTTP 400 at the gateway."""


def _blob_id_from_url(url: str) -> str | None:
    """Extract a blob id from a gateway file URL/path, else ``None``."""
    m = _FILE_PATH_RE.match(url.strip())
    return m.group("id") if m else None


def guard_image(*, content_type: str, size: int) -> None:
    """Reject oversize or unsupported-mime images. Raises :class:`MultimodalError`."""
    if size > MAX_IMAGE_BYTES:
        raise MultimodalError(f"image is {size} bytes; max is {MAX_IMAGE_BYTES}")
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime not in ALLOWED_IMAGE_TYPES:
        raise MultimodalError(f"unsupported image type {mime!r}; allowed: {sorted(ALLOWED_IMAGE_TYPES)}")


def _guard_data_url(url: str) -> None:
    """Validate an inline ``data:`` image URL's mime + decoded size."""
    m = _DATA_URL_RE.match(url)
    if not m:
        return  # not a data URL (e.g. https://) — nothing to guard here
    mime = (m.group("mime") or "").lower()
    payload = m.group("payload") or ""
    if m.group("b64"):
        try:
            size = len(base64.b64decode(payload, validate=False))
        except (binascii.Error, ValueError) as exc:
            raise MultimodalError(f"invalid base64 image payload: {exc}")
    else:
        size = len(payload)
    guard_image(content_type=mime or "image/png", size=size)


def _resolve_image_part(part: ImagePart, store: BlobStore) -> ImagePart:
    """Turn any blob reference into an inline ``data:`` URL; pass data:/https through (guarded)."""
    image_url = dict(part.image_url)
    file_id = image_url.pop("file_id", None)
    url = image_url.get("url", "")

    if file_id is None and isinstance(url, str):
        file_id = _blob_id_from_url(url)

    if file_id is not None:
        if not store.has(file_id):
            raise MultimodalError(f"referenced file {file_id!r} not found")
        record, data = store.get(file_id)
        guard_image(content_type=record.content_type, size=record.size)
        image_url["url"] = store.data_url(file_id)
        return ImagePart(image_url=image_url)

    if isinstance(url, str) and url:
        _guard_data_url(url)
        image_url["url"] = url
        return ImagePart(image_url=image_url)

    raise MultimodalError("image part has neither a file id nor a url")


def resolve_content(
    content: str | list[ContentPart], store: BlobStore | None = None
) -> str | list[ContentPart]:
    """Resolve every image part of one message's content into a self-contained ``image_url`` part."""
    if isinstance(content, str):
        return content
    store = store or get_blob_store()
    out: list[ContentPart] = []
    for part in content:
        if isinstance(part, ImagePart):
            out.append(_resolve_image_part(part, store))
        elif isinstance(part, TextPart):
            out.append(part)
        else:  # pragma: no cover - exhaustive over ContentPart union
            out.append(part)
    return out


def normalize_messages(
    messages: list[ChatMessage], store: BlobStore | None = None
) -> list[ChatMessage]:
    """Return new messages with all image parts resolved to backend-ready ``image_url`` parts."""
    store = store or get_blob_store()
    return [
        m.model_copy(update={"content": resolve_content(m.content, store)})
        for m in messages
    ]
