"""Close the loop: rank hosted models / responses by the calibrated reward, and export the collected
preferences as DPO-style training pairs.

  * :func:`rank_by_reward` — items (models or responses) best-first by calibrated reward, with the
    uncertainty carried through so promotion decisions are uncertainty-aware.
  * :func:`promote` — pick the champion, but only commit to it when it is *significantly* better than
    the runner-up (their reward credible intervals don't overlap); otherwise abstain and recommend the
    most-informative next comparison to elicit (active-learning hand-off to ``elicit.py``).
  * :func:`export_dpo_jsonl` — write the stored pairwise preferences as DPO ``{prompt, chosen,
    rejected}`` JSONL, the standard preference-tuning format.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Iterable

from sqlmodel import Session

from . import collect, elicit
from .models import Feedback
from .reward import RewardModel, RewardItem


def rank_by_reward(model: RewardModel) -> list[RewardItem]:
    """Items best-first by calibrated reward (carries std + CI for uncertainty-aware decisions)."""
    return model.ranking()


def promote(model: RewardModel) -> dict[str, Any]:
    """Decide whether to promote a champion item under the calibrated reward.

    Promotes only when the top item's reward CI does not overlap the runner-up's (a significant,
    not-just-noise win); otherwise abstains and hands back the most-informative next comparison.
    """
    ranking = model.ranking()
    if len(ranking) < 2:
        top = ranking[0]
        return {"promote": True, "champion": top.item_id, "reason": "only one candidate", "item": asdict(top)}

    top, runner = ranking[0], ranking[1]
    significant = top.ci_low > runner.ci_high            # credible intervals disjoint → real separation
    decision: dict[str, Any] = {
        "promote": bool(significant),
        "champion": top.item_id if significant else None,
        "top": asdict(top),
        "runner_up": asdict(runner),
        "reward_margin": float(top.reward - runner.reward),
        "prob_top_beats_runner": model.prob_prefer(top.item_id, runner.item_id),
    }
    if not significant:
        nxt = elicit.next_comparison(model)
        decision["reason"] = "top two not separated at the CI level; elicit more before promoting"
        decision["next_comparison"] = {"item_a": nxt.item_a, "item_b": nxt.item_b, "score": nxt.score}
    else:
        decision["reason"] = "champion reward CI clears the runner-up"
    return decision


def export_dpo_records(session: Session, *, model: str | None = None) -> list[dict[str, Any]]:
    """Stored preferences as DPO ``{prompt, chosen, rejected}`` dicts.

    Uses the preference's ``payload`` for the prompt + candidate texts when present (the chat UI stores
    them there); falls back to the opaque chosen/rejected item ids so an export is always producible.
    """
    records: list[dict[str, Any]] = []
    for fb in collect.list_preferences(session, model=model):
        if fb.chosen_id is None or fb.rejected_id is None:
            continue
        payload = fb.payload_dict()
        prompt = payload.get("prompt", "")
        chosen = payload.get("chosen_text", payload.get("chosen", str(fb.chosen_id)))
        rejected = payload.get("rejected_text", payload.get("rejected", str(fb.rejected_id)))
        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_id": str(fb.chosen_id),
            "rejected_id": str(fb.rejected_id),
            "model": fb.model,
        })
    return records


def export_dpo_jsonl(session: Session, *, model: str | None = None) -> str:
    """The DPO preference dataset as a JSONL string (one ``{prompt, chosen, rejected}`` per line)."""
    return _to_jsonl(export_dpo_records(session, model=model))


def _to_jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, default=str) for r in records)
