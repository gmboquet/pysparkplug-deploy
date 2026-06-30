"""Render a conversation to a downloadable artifact: ``json`` | ``markdown`` | ``pdf``.

JSON and Markdown are stdlib-only. PDF uses ``reportlab`` (lazy-imported; the 'export' extra) so the
core install stays slim and PDF export degrades to a clear error if the dependency is missing.

``export_conversation`` returns ``(data: bytes, media_type, filename_suffix)`` so a route can stream it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import Conversation, Message

_ROLE_LABEL = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
}


class ExportError(Exception):
    """Raised on an unknown format or a missing optional dependency."""


def _conv_dict(conv: "Conversation", messages: list["Message"]) -> dict:
    return {
        "id": conv.id,
        "title": conv.title,
        "model": conv.model,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }


def to_json(conv: "Conversation", messages: list["Message"]) -> bytes:
    return json.dumps(_conv_dict(conv, messages), indent=2, default=str).encode("utf-8")


def to_markdown(conv: "Conversation", messages: list["Message"]) -> bytes:
    lines: list[str] = [f"# {conv.title or 'Conversation'}", ""]
    meta = []
    if conv.model:
        meta.append(f"**Model:** {conv.model}")
    meta.append(f"**Created:** {conv.created_at.isoformat()}")
    lines.append("  \n".join(meta))
    lines.append("")
    for m in messages:
        label = _ROLE_LABEL.get(m.role, m.role.capitalize())
        lines.append(f"## {label}")
        lines.append("")
        lines.append(m.content or "")
        lines.append("")
    return ("\n".join(lines)).encode("utf-8")


def to_pdf(conv: "Conversation", messages: list["Message"]) -> bytes:
    """Render to a PDF via reportlab (lazy import — the 'export' extra)."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:  # pragma: no cover - exercised only when reportlab absent
        raise ExportError(
            "PDF export requires the 'export' extra (pip install reportlab)."
        ) from exc

    import io

    def _esc(text: str) -> str:
        return (
            (text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title=conv.title or "Conversation")
    styles = getSampleStyleSheet()
    flow = [Paragraph(_esc(conv.title or "Conversation"), styles["Title"])]
    if conv.model:
        flow.append(Paragraph(f"Model: {_esc(conv.model)}", styles["Normal"]))
    flow.append(Spacer(1, 12))
    for m in messages:
        label = _ROLE_LABEL.get(m.role, m.role.capitalize())
        flow.append(Paragraph(_esc(label), styles["Heading2"]))
        flow.append(Paragraph(_esc(m.content), styles["BodyText"]))
        flow.append(Spacer(1, 8))
    doc.build(flow)
    return buf.getvalue()


_FORMATS = {
    "json": ("application/json", "json"),
    "markdown": ("text/markdown", "md"),
    "md": ("text/markdown", "md"),
    "pdf": ("application/pdf", "pdf"),
}


def export_conversation(
    conv: "Conversation", messages: list["Message"], fmt: str = "json"
) -> tuple[bytes, str, str]:
    """Render ``conv`` + ``messages`` to ``fmt``.

    Returns ``(data, media_type, suffix)``. Raises :class:`ExportError` on an unknown format or a
    missing optional dependency (PDF).
    """
    key = (fmt or "json").lower()
    if key not in _FORMATS:
        raise ExportError(f"unknown export format {fmt!r} (use json | markdown | pdf).")
    media_type, suffix = _FORMATS[key]
    if key == "pdf":
        data = to_pdf(conv, messages)
    elif key in ("markdown", "md"):
        data = to_markdown(conv, messages)
    else:
        data = to_json(conv, messages)
    return data, media_type, suffix
