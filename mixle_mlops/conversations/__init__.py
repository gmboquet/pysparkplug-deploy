"""Conversation persistence + export.

Stores chat history (``Conversation`` + its ``Message`` rows) per user, and exports a conversation
to JSON / Markdown / PDF for download or archival.

Submodules:
  * :mod:`models`  — SQLModel ``Conversation`` and ``Message`` tables.
  * :mod:`service` — create / append / get / list, plus a ``persist_turn`` helper the chat route calls.
  * :mod:`export`  — render a conversation to ``json`` | ``markdown`` | ``pdf``.
"""

from __future__ import annotations
