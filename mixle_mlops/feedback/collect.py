"""Ingest + persist human feedback through a SQLModel ``Session``.

Three entry points mirror the three kinds of signal the chat UI emits (👍/👎, pairwise choose, edit),
each normalised into a :class:`~mixle_mlops.feedback.models.Feedback` row. ``list_preferences`` and
``preference_pairs`` read back the comparisons the reward model trains on.
"""

from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from .models import Feedback


def record_rating(
    session: Session,
    *,
    value: float,
    user_id: str | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    model: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Feedback:
    """Persist a 👍/👎 (or scalar) rating of a single message. ``value`` is +1 / -1 / any scalar."""
    fb = Feedback(
        kind="rating",
        value=float(value),
        user_id=user_id,
        conversation_id=conversation_id,
        message_id=message_id,
        model=model,
        payload=Feedback.encode_payload(payload),
    )
    return _save(session, fb)


def record_preference(
    session: Session,
    *,
    chosen_id: str,
    rejected_id: str,
    user_id: str | None = None,
    conversation_id: str | None = None,
    model: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Feedback:
    """Persist a pairwise preference: ``chosen_id`` was preferred over ``rejected_id``.

    Item ids are opaque labels — response/message ids, candidate ids, or model ids. The reward model
    (``reward.fit_reward``) maps the distinct ids to Bradley-Terry items.
    """
    if chosen_id == rejected_id:
        raise ValueError("a preference must have chosen_id != rejected_id.")
    fb = Feedback(
        kind="preference",
        chosen_id=str(chosen_id),
        rejected_id=str(rejected_id),
        user_id=user_id,
        conversation_id=conversation_id,
        model=model,
        payload=Feedback.encode_payload(payload),
    )
    return _save(session, fb)


def record_edit(
    session: Session,
    *,
    edited_text: str,
    original_text: str | None = None,
    user_id: str | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    model: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Feedback:
    """Persist a user edit of a response (a strong implicit negative on the original + a gold target)."""
    blob = dict(payload or {})
    blob["edited_text"] = edited_text
    if original_text is not None:
        blob["original_text"] = original_text
    fb = Feedback(
        kind="edit",
        user_id=user_id,
        conversation_id=conversation_id,
        message_id=message_id,
        model=model,
        payload=Feedback.encode_payload(blob),
    )
    return _save(session, fb)


def ingest(session: Session, body: dict[str, Any], *, user_id: str | None = None) -> Feedback:
    """Dispatch a raw feedback dict (as posted to ``POST /feedback``) to the right recorder."""
    kind = body.get("kind")
    if kind == "rating":
        return record_rating(
            session,
            value=body.get("value", 1.0),
            user_id=user_id,
            conversation_id=body.get("conversation_id"),
            message_id=body.get("message_id"),
            model=body.get("model"),
            payload=body.get("payload"),
        )
    if kind == "preference":
        chosen = body.get("chosen_id")
        rejected = body.get("rejected_id")
        if chosen is None or rejected is None:
            raise ValueError("a preference requires chosen_id and rejected_id.")
        return record_preference(
            session,
            chosen_id=chosen,
            rejected_id=rejected,
            user_id=user_id,
            conversation_id=body.get("conversation_id"),
            model=body.get("model"),
            payload=body.get("payload"),
        )
    if kind == "edit":
        return record_edit(
            session,
            edited_text=body.get("edited_text", body.get("payload", {}).get("edited_text", "")),
            original_text=body.get("original_text"),
            user_id=user_id,
            conversation_id=body.get("conversation_id"),
            message_id=body.get("message_id"),
            model=body.get("model"),
            payload=body.get("payload"),
        )
    raise ValueError(f"unknown feedback kind {kind!r}; expected 'rating' | 'preference' | 'edit'.")


def list_preferences(session: Session, *, model: str | None = None) -> list[Feedback]:
    """All stored pairwise-preference rows (optionally for a single producing model)."""
    stmt = select(Feedback).where(Feedback.kind == "preference")
    if model is not None:
        stmt = stmt.where(Feedback.model == model)
    return list(session.exec(stmt))


def preference_pairs(session: Session, *, model: str | None = None) -> list[tuple[str, str]]:
    """The stored preferences as ``(chosen_id, rejected_id)`` tuples — the reward model's training data."""
    return [
        (str(fb.chosen_id), str(fb.rejected_id))
        for fb in list_preferences(session, model=model)
        if fb.chosen_id is not None and fb.rejected_id is not None
    ]


def _save(session: Session, fb: Feedback) -> Feedback:
    session.add(fb)
    session.commit()
    session.refresh(fb)
    return fb
