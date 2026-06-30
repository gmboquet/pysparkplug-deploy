"""Multimodal support: upload an image, retrieve it, reference it from a chat message, and check the content
helpers (text()/images()) + normalization to backend-ready image_url parts. Self-contained: builds the app via
create_app(), includes the files router, signs up for a key, and uses a fresh blob store under tmp_path."""
from __future__ import annotations

import base64

import mixle_mlops.multimodal.store as store_mod
import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.config import get_settings
from mixle_mlops.core.adapters import ChatMessage, ImagePart, TextPart
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import files as files_route
from mixle_mlops.multimodal.content import (
    MultimodalError,
    guard_image,
    normalize_messages,
    resolve_content,
)
from mixle_mlops.multimodal.store import LocalBlobStore

# a 1x1 transparent PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    store_mod.reset_blob_store()
    app = create_app()
    app.include_router(files_route.router, prefix="/v1", tags=["files"])
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None
    store_mod.reset_blob_store()


def _signup(client) -> dict:
    raw = client.post("/auth/signup", json={"email": "m@t.com", "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {raw}"}


def test_upload_and_retrieve(client):
    headers = _signup(client)

    # auth required
    assert client.post("/v1/files", files={"file": ("x.png", _PNG, "image/png")}).status_code == 401

    up = client.post("/v1/files", headers=headers,
                     files={"file": ("pixel.png", _PNG, "image/png")})
    assert up.status_code == 200, up.text
    body = up.json()
    fid = body["id"]
    assert body["url"] == f"/v1/files/{fid}/content"
    assert body["content_type"] == "image/png"
    assert body["size"] == len(_PNG)

    meta = client.get(f"/v1/files/{fid}", headers=headers).json()
    assert meta["id"] == fid and meta["filename"] == "pixel.png"

    content = client.get(f"/v1/files/{fid}/content", headers=headers)
    assert content.status_code == 200
    assert content.content == _PNG
    assert content.headers["content-type"].startswith("image/png")

    assert client.get("/v1/files/file-nope", headers=headers).status_code == 404


def test_chat_message_helpers():
    msg = ChatMessage(
        role="user",
        content=[
            TextPart(text="what is in this image?"),
            ImagePart(image_url={"url": "data:image/png;base64,abc"}),
        ],
    )
    assert msg.text() == "what is in this image?"
    assert msg.images() == ["data:image/png;base64,abc"]
    assert ChatMessage(role="user", content="plain").text() == "plain"
    assert ChatMessage(role="user", content="plain").images() == []


def test_resolve_file_reference_to_data_url(tmp_path):
    store = LocalBlobStore(tmp_path / "blobs")
    record = store.put(_PNG, filename="pixel.png", content_type="image/png")

    # reference by file_id
    msg_by_id = ChatMessage(role="user", content=[ImagePart(image_url={"file_id": record.id})])
    # reference by gateway url path
    msg_by_url = ChatMessage(role="user", content=[ImagePart(image_url={"url": record.url})])

    out = normalize_messages([msg_by_id, msg_by_url], store=store)
    expected = "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii")
    for m in out:
        assert m.images() == [expected]

    # an already-inline data URL passes through untouched
    inline = ChatMessage(role="user", content=[ImagePart(image_url={"url": expected})])
    assert normalize_messages([inline], store=store)[0].images() == [expected]

    # plain-text content survives normalization
    assert normalize_messages([ChatMessage(role="user", content="hi")], store=store)[0].text() == "hi"


def test_resolve_unknown_file_raises(tmp_path):
    store = LocalBlobStore(tmp_path / "blobs")
    bad = ChatMessage(role="user", content=[ImagePart(image_url={"file_id": "file-missing"})])
    with pytest.raises(MultimodalError):
        resolve_content(bad.content, store=store)


def test_guards():
    with pytest.raises(MultimodalError):
        guard_image(content_type="image/png", size=10 ** 9)
    with pytest.raises(MultimodalError):
        guard_image(content_type="application/zip", size=10)
    # oversize inline data URL is rejected during normalization
    big = "data:image/png;base64," + base64.b64encode(b"\x00" * (21 * 1024 * 1024)).decode("ascii")
    with pytest.raises(MultimodalError):
        resolve_content([ImagePart(image_url={"url": big})], store=LocalBlobStore())


def test_upload_rejects_oversize_image(client, monkeypatch):
    headers = _signup(client)
    import mixle_mlops.multimodal.content as content_mod
    monkeypatch.setattr(content_mod, "MAX_IMAGE_BYTES", 8)
    # files route imports guard_image by reference; patch there too
    import mixle_mlops.gateway.routes.files as files_mod
    monkeypatch.setattr(files_mod, "guard_image", content_mod.guard_image)
    r = client.post("/v1/files", headers=headers,
                    files={"file": ("big.png", _PNG, "image/png")})
    assert r.status_code == 400
