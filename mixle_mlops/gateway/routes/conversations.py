"""Conversation history routes — list / read / export / delete persisted chat threads.

  * ``GET    /v1/conversations``               — the caller's conversations (most recent first).
  * ``GET    /v1/conversations/{id}``          — one conversation with its messages.
  * ``GET    /v1/conversations/{id}/export``   — download as ``?format=json|markdown|pdf``.
  * ``DELETE /v1/conversations/{id}``          — delete a conversation the caller owns.

All routes require an authenticated user; a caller may only see/touch their own conversations.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlmodel import Session, SQLModel

from ...accounts.models import User
from ...conversations import service
from ...conversations.export import ExportError, export_conversation
from ...conversations.models import Conversation, Message  # noqa: F401  (registers tables)
from ...storage.db import get_engine, get_session
from ..auth import require_user

router = APIRouter()


def _ensure_table() -> None:
    """Create the conversation tables on demand (idempotent) in case init_db ran before this import."""
    SQLModel.metadata.create_all(
        get_engine(), tables=[Conversation.__table__, Message.__table__]
    )


def _conv_summary(conv: Conversation) -> dict:
    return {
        "id": conv.id,
        "title": conv.title,
        "model": conv.model,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
    }


def _msg_dict(msg: Message) -> dict:
    return {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "created_at": msg.created_at.isoformat(),
    }


def _owned(session: Session, conversation_id: str, user: User) -> tuple[Conversation, list[Message]]:
    result = service.get_conversation(session, conversation_id)
    if result is None or result[0].user_id != user.id:
        raise HTTPException(status_code=404, detail="conversation not found")
    return result


@router.get("/conversations")
def list_conversations_route(
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    _ensure_table()
    convs = service.list_conversations(session, user.id)
    return {"object": "list", "data": [_conv_summary(c) for c in convs]}


@router.get("/conversations/{conversation_id}")
def get_conversation_route(
    conversation_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    _ensure_table()
    conv, messages = _owned(session, conversation_id, user)
    out = _conv_summary(conv)
    out["messages"] = [_msg_dict(m) for m in messages]
    return out


@router.get("/conversations/{conversation_id}/export")
def export_conversation_route(
    conversation_id: str,
    format: str = Query(default="json"),
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    _ensure_table()
    conv, messages = _owned(session, conversation_id, user)
    try:
        data, media_type, suffix = export_conversation(conv, messages, format)
    except ExportError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    filename = f"conversation-{conv.id}.{suffix}"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation_route(
    conversation_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    _ensure_table()
    if not service.delete_conversation(session, conversation_id=conversation_id, user_id=user.id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"deleted": True, "id": conversation_id}
