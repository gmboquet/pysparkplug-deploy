"""The principled RLHF / human-feedback loop.

Human feedback (👍/👎 ratings, pairwise preferences, edits) becomes a *calibrated* mixle preference
(reward) model with uncertainty, and active elicitation picks the most informative next comparison.

Submodules:
  * :mod:`models`  — SQLModel ``Feedback`` table (ratings / preferences / edits).
  * :mod:`collect` — ingest + persist feedback through a SQLModel ``Session``.
  * :mod:`reward`  — fit a mixle Bradley-Terry reward model with bootstrap uncertainty.
  * :mod:`elicit`  — active preference elicitation (most-informative next comparison).
  * :mod:`loop`    — close-the-loop ranking + DPO-style preference export.
"""

from __future__ import annotations
