"""Indexing + retrieval: chunk → embed → add into a user's vector store, and rank snippets for a query.

One retriever serves two sources that share the user's store via distinct namespaces:

  * ``conversation`` — :func:`index_conversation` turns a chat transcript into retrievable memory of past turns.
  * ``document``     — :func:`index_document_chunks` adds the chunks of an uploaded document.

:func:`retrieve` embeds the query once and cosine-ranks across both (or a filtered subset), returning snippet
dicts ready to drop into a context block.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..documents.parse import chunk_text
from .embeddings import Embedder, get_embedder
from .vectorstore import VectorStore, get_vector_store

NS_CONVERSATION = "conversation"
NS_DOCUMENT = "document"


def _stores(
    embedder: Embedder | None, store: VectorStore | None
) -> tuple[Embedder, VectorStore]:
    return embedder or get_embedder(), store or get_vector_store()


def _message_text(m: Any) -> tuple[str, str]:
    """Return ``(role, text)`` from a ChatMessage-like object or a plain ``{role, content}`` dict."""
    if isinstance(m, Mapping):
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        if isinstance(content, str):
            return role, content
        # list-of-parts: concatenate text parts
        parts = []
        for p in content if isinstance(content, list) else []:
            if isinstance(p, Mapping) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return role, " ".join(parts)
    role = str(getattr(m, "role", "user"))
    text_fn = getattr(m, "text", None)
    if callable(text_fn):
        return role, text_fn()
    return role, str(getattr(m, "content", ""))


def index_conversation(
    user_id: str,
    conversation_id: str,
    messages: Sequence[Any],
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    chunk_tokens: int = 256,
    overlap_tokens: int = 32,
    roles: Iterable[str] = ("user", "assistant"),
    replace: bool = True,
) -> list[str]:
    """Index a conversation's messages as retrievable memory.

    Each ``role: text`` turn is chunked (long turns split), embedded, and stored under the ``conversation``
    namespace with ``source_id = conversation_id``. ``replace=True`` first drops any prior chunks for this
    conversation so re-indexing after new turns is idempotent.
    """
    emb, vs = _stores(embedder, store)
    roleset = set(roles)
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        role, text = _message_text(m)
        if role not in roleset:
            continue
        for j, ch in enumerate(chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)):
            labelled = f"{role}: {ch}"
            texts.append(labelled)
            metas.append({"source": "conversation", "conversation_id": conversation_id,
                          "role": role, "message_index": i, "chunk_index": j})
    if replace:
        vs.delete_source(user_id, conversation_id)
    if not texts:
        return []
    vectors = emb.embed(texts)
    items = [
        {"text": texts[i], "vector": vectors[i], "meta": metas[i],
         "namespace": NS_CONVERSATION, "source_id": conversation_id}
        for i in range(len(texts))
    ]
    return vs.add(user_id, items)


def index_document_chunks(
    user_id: str,
    document_id: str,
    chunks: Sequence[str],
    *,
    filename: str = "",
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    extra_meta: Mapping[str, Any] | None = None,
    replace: bool = True,
) -> list[str]:
    """Embed and add pre-chunked document text under the ``document`` namespace (``source_id = document_id``)."""
    emb, vs = _stores(embedder, store)
    chunks = [c for c in chunks if c and c.strip()]
    if replace:
        vs.delete_source(user_id, document_id)
    if not chunks:
        return []
    vectors = emb.embed(chunks)
    base = {"source": "document", "document_id": document_id, "filename": filename}
    if extra_meta:
        base.update(dict(extra_meta))
    items = [
        {"text": chunks[i], "vector": vectors[i],
         "meta": {**base, "chunk_index": i},
         "namespace": NS_DOCUMENT, "source_id": document_id}
        for i in range(len(chunks))
    ]
    return vs.add(user_id, items)


def retrieve(
    user_id: str,
    query: str,
    k: int = 5,
    *,
    namespace: str | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Embed ``query`` and return the top-``k`` ranked snippets across the user's store.

    ``namespace`` restricts to ``conversation`` or ``document`` (default: both). ``min_score`` drops weak hits.
    Returns plain dicts (``id, text, score, namespace, source_id, meta``) ready for a context block.
    """
    emb, vs = _stores(embedder, store)
    if not query or not query.strip():
        return []
    qvec = emb.embed_one(query)
    hits = vs.query(user_id, qvec, k=k, filter={"namespace": namespace} if namespace else None)
    out = []
    for h in hits:
        if min_score is not None and h.score < min_score:
            continue
        out.append(h.to_dict())
    return out
