"""The chat composition hook: prepend a retrieved-context system block to a chat.

``build_rag_messages(user_id, messages)`` takes the request's messages, retrieves snippets relevant to the latest
user turn (from the user's conversation memory *and* uploaded documents — one retriever), and returns a new
message list with a system block of that context prepended. The integrator calls this in the chat pipeline
(``gateway/routes/chat.py``) just before handing the messages to the adapter, gated on a per-request/user flag.

It is defensive: if retrieval returns nothing (or errors — e.g. no store yet), the original messages are returned
unchanged so RAG can never break a chat.
"""
from __future__ import annotations

from typing import Any, Sequence

from .embeddings import Embedder
from .index import _message_text, retrieve
from .vectorstore import VectorStore

CONTEXT_HEADER = (
    "You have access to the following retrieved context from the user's past conversations and uploaded "
    "documents. Use it when relevant to answer; if it does not contain the answer, rely on your own knowledge "
    "and do not fabricate citations.\n\n"
)


def _latest_user_query(messages: Sequence[Any]) -> str:
    """The most recent user-authored text — what we retrieve against."""
    for m in reversed(list(messages)):
        role, text = _message_text(m)
        if role == "user" and text.strip():
            return text
    # fall back to the last message's text
    if messages:
        return _message_text(messages[-1])[1]
    return ""


def format_context_block(snippets: Sequence[dict[str, Any]]) -> str:
    """Render retrieved snippets into a single system-message string."""
    lines = [CONTEXT_HEADER]
    for i, s in enumerate(snippets, 1):
        meta = s.get("meta", {}) or {}
        src = meta.get("filename") or meta.get("conversation_id") or s.get("source_id") or s.get("namespace")
        tag = f"[{i}] ({s.get('namespace', 'context')}"
        if src:
            tag += f": {src}"
        tag += ")"
        lines.append(f"{tag}\n{s.get('text', '').strip()}")
    return "\n\n".join(lines)


def build_rag_messages(
    user_id: str | None,
    messages: Sequence[Any],
    *,
    k: int = 5,
    namespace: str | None = None,
    min_score: float | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    as_dict: bool = True,
) -> list[Any]:
    """Return ``messages`` with a retrieved-context system block prepended (unchanged if nothing relevant).

    ``messages`` items may be ``ChatMessage`` objects or ``{role, content}`` dicts; the prepended block matches
    the input style (a dict when ``as_dict`` and inputs are dicts, otherwise a ``ChatMessage``). Returns the
    original list object's contents untouched on any failure.
    """
    out = list(messages)
    if not user_id:
        return out
    query = _latest_user_query(messages)
    if not query.strip():
        return out
    try:
        snippets = retrieve(
            user_id, query, k=k, namespace=namespace,
            min_score=min_score, embedder=embedder, store=store,
        )
    except Exception:
        return out
    if not snippets:
        return out

    block = format_context_block(snippets)
    inputs_are_dicts = bool(messages) and isinstance(messages[0], dict)
    if as_dict and inputs_are_dicts:
        sys_msg: Any = {"role": "system", "content": block}
    else:
        from ..core.adapters import ChatMessage  # lazy: avoid import cycle at module load

        sys_msg = ChatMessage(role="system", content=block)
    return [sys_msg, *out]
