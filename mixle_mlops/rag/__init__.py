"""Retrieval-augmented generation: one retriever over **past conversations** *and* **uploaded documents**.

The pieces compose top-down:

  * :mod:`embeddings` — turn text into vectors (OpenAI-compatible ``/v1/embeddings`` backend with a
    deterministic local hashing fallback, so tests/offline need no server).
  * :mod:`vectorstore` — store vectors per ``(user, namespace)`` and cosine-rank them.
  * :mod:`index` — chunk + embed + add conversations and document chunks; ``retrieve`` ranks snippets.
  * :mod:`augment` — ``build_rag_messages`` prepends a retrieved-context system block to a chat (the chat
    composition hook the integrator wires into the chat pipeline).
"""
from __future__ import annotations

from .embeddings import Embedder, get_embedder
from .index import index_conversation, index_document_chunks, retrieve
from .vectorstore import LocalVectorStore, VectorStore, get_vector_store

__all__ = [
    "Embedder",
    "get_embedder",
    "VectorStore",
    "LocalVectorStore",
    "get_vector_store",
    "index_conversation",
    "index_document_chunks",
    "retrieve",
]
