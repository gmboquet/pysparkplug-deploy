"""Unified multi-cloud object store over `fsspec`.

One small abstraction — :class:`ObjectStore` — that puts/gets/exists/url against *any* of the major clouds
(AWS S3, GCP GCS, Azure Blob, Alibaba Cloud OSS) **and** any S3-compatible endpoint (MinIO, Cloudflare R2,
Wasabi, …), with a local-filesystem default so everything runs on a laptop with zero cloud setup.

Selection is by a single URL setting ``MIXLE_OBJECT_STORE_URL`` (env-driven):

  - ``file:///var/mixle/objects``  → local filesystem (the default; no extra deps)
  - ``s3://bucket/prefix``         → AWS S3 / any S3-compatible (needs ``s3fs``; endpoint via ``MIXLE_OBJECT_STORE_ENDPOINT``)
  - ``gs://bucket/prefix``         → Google Cloud Storage (needs ``gcsfs``)
  - ``az://container/prefix``      → Azure Blob Storage (needs ``adlfs``; account via ``MIXLE_OBJECT_STORE_ENDPOINT`` or env)
  - ``oss://bucket/prefix``        → Alibaba Cloud OSS (needs ``ossfs``; endpoint via ``MIXLE_OBJECT_STORE_ENDPOINT``)

This is standalone: it generalizes the cloud path of ``multimodal/store.py`` without depending on or editing it.
Credentials follow each driver's standard chain (instance/pod identity, env vars, shared config) so nothing
secret is ever read here — the Helm/Terraform layer wires identity in. ``fsspec`` and the per-backend driver are
lazy-imported so importing this module costs nothing and the base install stays cloud-free.

The setting :class:`ObjectStoreSettings` is read from the ``MIXLE_OBJECT_STORE_*`` env vars; it is intentionally
*separate* from the platform ``Settings`` (config.py) so this module can be reported as a drop-in without editing
shared config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

# Map a URL scheme → (fsspec protocol, the pip driver that provides it, the 'cloud' extra it lives in).
_SCHEME_DRIVER: dict[str, tuple[str, str]] = {
    "file": ("file", ""),          # builtin (LocalFileSystem) — no driver needed
    "": ("file", ""),              # bare/relative path → local
    "s3": ("s3", "s3fs"),
    "s3a": ("s3", "s3fs"),
    "gs": ("gcs", "gcsfs"),
    "gcs": ("gcs", "gcsfs"),
    "az": ("az", "adlfs"),
    "abfs": ("abfs", "adlfs"),
    "abfss": ("abfs", "adlfs"),
    "oss": ("oss", "ossfs"),
}


@dataclass
class ObjectStoreSettings:
    """Env-driven config for the object store (prefix ``MIXLE_OBJECT_STORE_``).

    ``url`` is the only required knob; ``endpoint`` covers S3-compatible/OSS custom endpoints and (optionally) the
    Azure account url. ``region`` and ``anon`` are passed through to drivers that accept them.
    """

    url: str = "file://./mixle_data/objects"
    endpoint: str | None = None
    region: str | None = None
    anon: bool = False

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "ObjectStoreSettings":
        env = environ if environ is not None else os.environ
        return cls(
            url=env.get("MIXLE_OBJECT_STORE_URL", cls.url),
            endpoint=env.get("MIXLE_OBJECT_STORE_ENDPOINT") or None,
            region=env.get("MIXLE_OBJECT_STORE_REGION") or None,
            anon=env.get("MIXLE_OBJECT_STORE_ANON", "").lower() in ("1", "true", "yes"),
        )


def _parse_url(url: str) -> tuple[str, str, str, str]:
    """Return ``(scheme, driver_protocol, bucket_or_root, prefix)`` for an object-store URL.

    For ``s3://bucket/a/b`` → ``("s3", "s3", "bucket", "a/b")``.
    For ``file:///var/x``   → ``("file", "file", "/var/x", "")`` (the whole path is the root).
    For a bare ``./p``      → treated as a local path.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _SCHEME_DRIVER:
        raise ValueError(
            f"unsupported object-store URL scheme {scheme!r} in {url!r}; "
            f"supported: {sorted(s for s in _SCHEME_DRIVER if s)}"
        )
    protocol, _driver = _SCHEME_DRIVER[scheme]
    if protocol == "file":
        # local: everything after the scheme is the root directory
        root = parts.path or parts.netloc or url
        if scheme == "":
            root = url
        return "file", "file", root, ""
    # cloud: netloc is the bucket/container, path is the key prefix
    bucket = parts.netloc
    prefix = parts.path.lstrip("/")
    return scheme, protocol, bucket, prefix


def _storage_options(settings: ObjectStoreSettings, protocol: str) -> dict:
    """Build the ``storage_options`` for the chosen driver from settings (and the driver's own cred chain)."""
    opts: dict = {}
    if protocol == "s3":
        if settings.endpoint:
            opts["client_kwargs"] = {"endpoint_url": settings.endpoint}
        if settings.region:
            opts.setdefault("client_kwargs", {})["region_name"] = settings.region
        if settings.anon:
            opts["anon"] = True
    elif protocol == "oss":
        if settings.endpoint:
            opts["endpoint"] = settings.endpoint
    elif protocol in ("az", "abfs"):
        # adlfs reads AZURE_STORAGE_* from env; allow an explicit account url via endpoint.
        if settings.endpoint:
            opts["account_url"] = settings.endpoint
    elif protocol == "gcs":
        if settings.anon:
            opts["token"] = "anon"
    return opts


@dataclass
class ObjectInfo:
    """Lightweight result of a put: where the object lives + how big it is."""

    key: str
    uri: str
    size: int


class ObjectStore:
    """fsspec-backed object store. Same API on local fs and every cloud.

    Keys are store-relative (e.g. ``models/v3/artifact.bin``); they are joined onto the configured root/prefix.
    """

    def __init__(self, settings: ObjectStoreSettings | None = None):
        self.settings = settings or ObjectStoreSettings.from_env()
        self.scheme, self.protocol, self.bucket, self.prefix = _parse_url(self.settings.url)
        self._fs = None  # lazy

    # -- fsspec wiring (lazy) ------------------------------------------------
    @property
    def fs(self):
        if self._fs is None:
            self._fs = self._make_fs()
        return self._fs

    def _make_fs(self):
        try:
            import fsspec
        except ImportError as exc:  # pragma: no cover - base install always has it via deps? no — lazy
            raise RuntimeError(
                "fsspec is required for the object store; install the 'cloud' extra: pip install mixle-mlops[cloud]"
            ) from exc
        if self.protocol != "file":
            driver = _SCHEME_DRIVER[self.scheme][1]
            try:
                __import__(driver)
            except ImportError as exc:
                raise RuntimeError(
                    f"object store {self.settings.url!r} needs the {driver!r} driver; "
                    f"install the 'cloud' extra: pip install mixle-mlops[cloud]"
                ) from exc
        opts = _storage_options(self.settings, self.protocol)
        if self.protocol == "file":
            os.makedirs(self.bucket, exist_ok=True)
        return fsspec.filesystem(self.protocol, **opts)

    # -- path helpers --------------------------------------------------------
    def _full_path(self, key: str) -> str:
        key = key.lstrip("/")
        if self.protocol == "file":
            return os.path.join(self.bucket, key)
        base = f"{self.bucket}/{self.prefix}".rstrip("/") if self.prefix else self.bucket
        return f"{base}/{key}"

    def uri(self, key: str) -> str:
        """The fully-qualified URI of a key (``s3://bucket/prefix/key`` or ``file:///abs/path``)."""
        full = self._full_path(key)
        if self.protocol == "file":
            return "file://" + os.path.abspath(full)
        return f"{self.scheme}://{full}"

    # -- the small CRUD surface ---------------------------------------------
    def put(self, key: str, data: bytes) -> ObjectInfo:
        """Write ``data`` at ``key``; creates parent dirs/prefixes as needed."""
        path = self._full_path(key)
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if parent:
            try:
                self.fs.makedirs(parent, exist_ok=True)
            except (FileExistsError, NotImplementedError):
                pass
        with self.fs.open(path, "wb") as fh:
            fh.write(data)
        return ObjectInfo(key=key, uri=self.uri(key), size=len(data))

    def get(self, key: str) -> bytes:
        """Read the bytes at ``key``; raise ``KeyError`` if missing."""
        path = self._full_path(key)
        if not self.fs.exists(path):
            raise KeyError(key)
        with self.fs.open(path, "rb") as fh:
            return fh.read()

    def exists(self, key: str) -> bool:
        return bool(self.fs.exists(self._full_path(key)))

    def delete(self, key: str) -> None:
        path = self._full_path(key)
        if self.fs.exists(path):
            self.fs.rm_file(path) if hasattr(self.fs, "rm_file") else self.fs.rm(path)

    def url(self, key: str, *, expires: int = 3600) -> str:
        """A retrievable URL for ``key``.

        On clouds that support presigned URLs (``s3``) we mint a short-lived signed URL so large objects aren't
        inlined into requests. Where presigning isn't available we return the canonical URI (the caller can proxy
        the bytes via :meth:`get`, mirroring the local-first multimodal flow).
        """
        path = self._full_path(key)
        fs = self.fs
        if hasattr(fs, "url"):
            try:
                return fs.url(path, expires=expires)
            except (TypeError, NotImplementedError):
                try:
                    return fs.url(path)
                except (TypeError, NotImplementedError):
                    pass
        if hasattr(fs, "sign"):
            try:
                return fs.sign(path, expiration=expires)
            except (TypeError, NotImplementedError, ValueError):
                pass
        return self.uri(key)


_store: ObjectStore | None = None


def get_object_store() -> ObjectStore:
    """Process-wide object store selected by ``MIXLE_OBJECT_STORE_URL`` (cached after first use)."""
    global _store
    if _store is None:
        _store = ObjectStore()
    return _store


def reset_object_store() -> None:
    """Test hook: drop the cached store so a fresh ``MIXLE_OBJECT_STORE_URL`` is picked up."""
    global _store
    _store = None
