"""RAG / document routes: upload a document (parse → chunk → index), list documents, and search the retriever.

  * ``POST /v1/documents``       — multipart upload → :class:`BlobStore` → extract text → chunk → embed → index
    into the user's vector store (``document`` namespace). Returns the document record.
  * ``GET  /v1/documents``       — list the caller's ingested documents.
  * ``POST /v1/rag/search``      — retrieve ranked snippets for a query across conversation memory + documents.

All routes require an authenticated user (``Depends(require_user)``); the vector store and documents are scoped to
``user.id``. Wiring (integrator): ``app.include_router(rag.router, prefix="/v1", tags=["rag"])`` in
``gateway/app.py``.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from ...accounts.models import User
from ...documents.parse import DocumentParseError, chunk_text, extract_text
from ...multimodal.store import get_blob_store
from ...rag.index import index_document_chunks, retrieve
from ...rag.models import Document
from ...rag.vectorstore import _ensure_tables, get_vector_store
from ...storage.db import get_engine
from ..auth import require_user

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    namespace: str | None = None        # "conversation" | "document" | None (both)
    min_score: float | None = None


@router.post("/documents")
async def upload_document(
    file: UploadFile,
    chunk_tokens: int = 256,
    overlap_tokens: int = 32,
    user: User = Depends(require_user),
):
    """Upload a document, store its bytes, parse + chunk + index its text for retrieval."""
    data = await file.read()
    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"
    try:
        text = extract_text(data, filename=filename, content_type=content_type)
    except DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    chunks = chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        raise HTTPException(status_code=400, detail="no extractable text in document")

    record = get_blob_store().put(data, filename=filename, content_type=content_type)
    _ensure_tables()
    doc = Document(
        user_id=user.id,
        filename=filename,
        content_type=content_type,
        blob_id=record.id,
        n_chunks=len(chunks),
        n_chars=len(text),
    )
    with Session(get_engine()) as session:
        session.add(doc)
        session.commit()
        session.refresh(doc)
        doc_dict = doc.to_dict()

    index_document_chunks(
        user.id, doc_dict["id"], chunks, filename=filename, store=get_vector_store()
    )
    return doc_dict


@router.get("/documents")
async def list_documents(user: User = Depends(require_user)):
    """List the caller's ingested documents (most recent first)."""
    _ensure_tables()
    with Session(get_engine()) as session:
        rows = list(
            session.exec(select(Document).where(Document.user_id == user.id))
        )
    rows.sort(key=lambda d: d.created_at or 0, reverse=True)
    return {"object": "list", "data": [d.to_dict() for d in rows]}


@router.post("/rag/search")
async def rag_search(body: SearchRequest = Body(...), user: User = Depends(require_user)):
    """Retrieve ranked context snippets for a query across conversation memory + uploaded documents."""
    hits = retrieve(
        user.id, body.query, k=body.k, namespace=body.namespace,
        min_score=body.min_score, store=get_vector_store(),
    )
    return {"object": "list", "query": body.query, "data": hits}
