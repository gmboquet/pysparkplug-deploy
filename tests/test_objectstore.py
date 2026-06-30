"""Multi-cloud object store: the local fsspec path round-trips a blob; URL parsing for every cloud scheme is
exercised without needing the cloud drivers (those are skipped if not installed). Also covers the cloud_init
scaffolder and the /v1/cloud/objectstore router. Self-contained: builds the app via create_app(), includes the
cloud router, signs up for a key."""
from __future__ import annotations

import importlib.util

import pytest
from fastapi.testclient import TestClient

import mixle_mlops.storage.db as db
import mixle_mlops.storage.objectstore as oss
from mixle_mlops.cloud_init import PROVIDERS, init_cloud, next_steps
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import cloud as cloud_route
from mixle_mlops.storage.objectstore import ObjectStore, ObjectStoreSettings, _parse_url


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


# ---------------------------------------------------------------- local round-trip (fsspec builtin) ---

def test_local_roundtrip(tmp_path):
    settings = ObjectStoreSettings(url=f"file://{tmp_path}/objects")
    store = ObjectStore(settings)
    assert store.protocol == "file"

    info = store.put("models/v1/artifact.bin", b"hello-cloud")
    assert info.size == len("hello-cloud")
    assert info.uri.startswith("file://")
    assert store.exists("models/v1/artifact.bin")
    assert store.get("models/v1/artifact.bin") == b"hello-cloud"

    # url() falls back to the canonical file URI for local
    assert store.url("models/v1/artifact.bin").startswith("file://")

    store.delete("models/v1/artifact.bin")
    assert not store.exists("models/v1/artifact.bin")

    with pytest.raises(KeyError):
        store.get("models/v1/artifact.bin")


def test_default_url_is_local(monkeypatch, tmp_path):
    monkeypatch.delenv("MIXLE_OBJECT_STORE_URL", raising=False)
    s = ObjectStoreSettings.from_env()
    assert s.url.startswith("file://")


def test_from_env(monkeypatch):
    monkeypatch.setenv("MIXLE_OBJECT_STORE_URL", "s3://bucket/prefix")
    monkeypatch.setenv("MIXLE_OBJECT_STORE_ENDPOINT", "https://minio.local:9000")
    monkeypatch.setenv("MIXLE_OBJECT_STORE_REGION", "us-west-2")
    monkeypatch.setenv("MIXLE_OBJECT_STORE_ANON", "true")
    s = ObjectStoreSettings.from_env()
    assert s.url == "s3://bucket/prefix"
    assert s.endpoint == "https://minio.local:9000"
    assert s.region == "us-west-2"
    assert s.anon is True


# ---------------------------------------------------------------- URL parsing for every cloud ---------

@pytest.mark.parametrize(
    "url,scheme,protocol,bucket,prefix",
    [
        ("s3://my-bucket/a/b", "s3", "s3", "my-bucket", "a/b"),
        ("s3a://my-bucket", "s3a", "s3", "my-bucket", ""),
        ("gs://my-objects/models", "gs", "gcs", "my-objects", "models"),
        ("az://container/p", "az", "az", "container", "p"),
        ("abfs://container/p", "abfs", "abfs", "container", "p"),
        ("oss://my-objects/x", "oss", "oss", "my-objects", "x"),
        ("file:///var/mixle", "file", "file", "/var/mixle", ""),
    ],
)
def test_parse_url_all_clouds(url, scheme, protocol, bucket, prefix):
    sch, proto, bkt, pre = _parse_url(url)
    assert (sch, proto, bkt, pre) == (scheme, protocol, bucket, prefix)


def test_parse_url_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        _parse_url("ftp://nope/x")


def test_full_path_and_uri_cloud():
    store = ObjectStore(ObjectStoreSettings(url="s3://bucket/prefix"))
    assert store._full_path("k/obj") == "bucket/prefix/k/obj"
    assert store.uri("k/obj") == "s3://bucket/prefix/k/obj"


def test_missing_driver_raises_helpful(monkeypatch):
    # If a cloud driver isn't importable, _make_fs should give a clear 'install cloud extra' error.
    store = ObjectStore(ObjectStoreSettings(url="gs://bucket/x"))
    if _have("gcsfs"):
        pytest.skip("gcsfs installed; the missing-driver branch can't be exercised")
    with pytest.raises(RuntimeError, match="cloud"):
        _ = store.fs


# ---------------------------------------------------------------- cloud_init scaffolder ----------------

@pytest.mark.parametrize("provider", PROVIDERS)
def test_init_cloud_writes_env(provider, tmp_path):
    dest = tmp_path / ".env"
    path = init_cloud(provider, dest=dest)
    text = path.read_text()
    assert "MIXLE_OBJECT_STORE_URL=" in text
    assert "MIXLE_DEPLOYMENT=" in text
    steps = next_steps(provider)
    assert isinstance(steps, str) and steps


def test_init_cloud_no_overwrite(tmp_path):
    dest = tmp_path / ".env"
    init_cloud("aws", dest=dest)
    with pytest.raises(FileExistsError):
        init_cloud("aws", dest=dest)
    init_cloud("gcp", dest=dest, overwrite=True)  # ok
    assert "gs://" in dest.read_text()


def test_init_cloud_unknown_provider(tmp_path):
    with pytest.raises(ValueError):
        init_cloud("ibm", dest=tmp_path / ".env")


# ---------------------------------------------------------------- router (self-contained) --------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MIXLE_OBJECT_STORE_URL", f"file://{tmp_path}/objects")
    get_settings.cache_clear()
    db._engine = None
    oss.reset_object_store()
    app = create_app()
    app.include_router(cloud_route.router, prefix="/v1", tags=["cloud"])
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None
    oss.reset_object_store()


def _signup(client) -> dict:
    raw = client.post("/auth/signup", json={"email": "c@t.com", "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {raw}"}


def test_objectstore_status_requires_auth(client):
    assert client.get("/v1/cloud/objectstore").status_code == 401


def test_objectstore_status_and_check(client):
    headers = _signup(client)
    st = client.get("/v1/cloud/objectstore", headers=headers)
    assert st.status_code == 200, st.text
    body = st.json()
    assert body["protocol"] == "file"
    assert body["scheme"] == "file"

    chk = client.post("/v1/cloud/objectstore/check", headers=headers,
                      json={"key": "ping.txt", "text": "pong"})
    assert chk.status_code == 200, chk.text
    out = chk.json()
    assert out["ok"] is True
    assert out["size"] == len("pong")
    assert out["uri"].startswith("file://")
