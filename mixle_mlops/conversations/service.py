"""Create / append / read conversation history through a SQLModel ``Session``.

``persist_turn`` is the integration seam: the chat route calls it after a completion to record the
(user message, assistant message) pair, lazily creating the conversation if one was not supplied.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Session, select

from .models import Conversation, Message


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _title_from(text: str, limit: int = 60) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "New conversation"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def create_conversation(
    session: Session,
    *,
    user_id: str,
    title: str | None = None,
    model: str | None = None,
) -> Conversation:
    """Create and persist a new (empty) conversation owned by ``user_id``."""
    conv = Conversation(
        user_id=user_id,
        title=title or "New conversation",
        model=model,
    )
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


def append_message(
    session: Session,
    *,
    conversation_id: str,
    role: str,
    content: str,
) -> Message:
    """Append one message to a conversation and bump its ``updated_at``."""
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    session.add(msg)
    conv = session.get(Conversation, conversation_id)
    if conv is not None:
        conv.updated_at = _now()
        session.add(conv)
    session.commit()
    session.refresh(msg)
    return msg


def get_messages(session: Session, conversation_id: str) -> list[Message]:
    """All messages of a conversation, oldest-first."""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
    )
    return list(session.exec(stmt).all())


def get_conversation(
    session: Session, conversation_id: str
) -> tuple[Conversation, list[Message]] | None:
    """Return ``(conversation, messages)`` or ``None`` if it does not exist."""
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        return None
    return conv, get_messages(session, conversation_id)


def list_conversations(session: Session, user_id: str) -> list[Conversation]:
    """All conversations owned by ``user_id``, most-recently-updated first."""
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id)
    )
    return list(session.exec(stmt).all())


def delete_conversation(session: Session, *, conversation_id: str, user_id: str) -> bool:
    """Delete a conversation (and its messages) if owned by ``user_id``. Returns whether it deleted."""
    conv = session.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user_id:
        return False
    for msg in get_messages(session, conversation_id):
        session.delete(msg)
    session.delete(conv)
    session.commit()
    return True


def persist_turn(
    session: Session,
    user_id: str,
    conversation_id: str | None,
    user_msg: str,
    assistant_msg: str,
    *,
    model: str | None = None,
) -> Conversation:
    """Record one chat turn, creating the conversation on first use.

    The integrator calls this from the chat route after a completion. When ``conversation_id`` is
    ``None`` (or names a missing/foreign conversation), a fresh conversation is created — titled from
    the first user message — so the caller can persist a turn without pre-creating the thread.

    Returns the conversation the turn was written to (read ``.id`` to thread subsequent turns).
    """
    conv: Conversation | None = None
    if conversation_id:
        existing = session.get(Conversation, conversation_id)
        if existing is not None and existing.user_id == user_id:
            conv = existing
    if conv is None:
        conv = create_conversation(
            session, user_id=user_id, title=_title_from(user_msg), model=model
        )
    elif model is not None and conv.model is None:
        conv.model = model
        session.add(conv)
        session.commit()

    append_message(session, conversation_id=conv.id, role="user", content=user_msg)
    append_message(session, conversation_id=conv.id, role="assistant", content=assistant_msg)
    return conv
