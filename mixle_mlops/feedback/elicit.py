"""Active preference elicitation — choose the most informative next comparison.

Given the current calibrated reward posterior (a :class:`~mixle_mlops.feedback.reward.RewardModel`, whose
bootstrap covariance ``cov`` IS the posterior over the per-item rewards) and a set of candidate items,
score every unordered pair ``(i, j)`` by the expected information a fresh comparison would carry, and
return the best pair.

Scoring (a Bayesian-D-optimal / BALD-style criterion for a Bradley-Terry observation):

    score(i, j) = Var[reward_i - reward_j] * p_ij * (1 - p_ij)

  * ``Var[reward_i - reward_j] = cov_ii + cov_jj - 2 cov_ij`` is our *uncertainty* about the contest
    (large when the two rewards' credible intervals overlap and we don't yet know who wins) — this is
    exactly the reward-uncertainty-overlap the brief asks for, read off the bootstrap covariance.
  * ``p_ij (1 - p_ij)`` is the Fisher information of one Bernoulli (Bradley-Terry) outcome, maximal at
    ``p = 0.5`` — a close, contested matchup. It is the per-observation term in
    ``mixle.doe.expected_information_gain_linear``'s ``log det(I + sigma^-2 F^T F)`` for this design.

This couples the model uncertainty (bootstrap posterior) with the observation informativeness, the same
information-gain logic mixle.doe applies to GP/linear designs, specialised to pairwise comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .reward import RewardModel

try:  # mixle.doe is the canonical home of the EIG machinery; we mirror its linear-design score here.
    from mixle.doe import expected_information_gain_linear as _eig_linear
except Exception:  # pragma: no cover - doe is part of mixle; guard only for very trimmed installs
    _eig_linear = None


@dataclass
class Comparison:
    item_a: str
    item_b: str
    score: float                  # expected information gain of asking this comparison
    prob_a_beats_b: float         # current calibrated P(a preferred over b)
    reward_gap_std: float         # posterior std of (reward_a - reward_b)


def _pair_score(model: RewardModel, i: int, j: int) -> tuple[float, float, float]:
    var_gap = float(model.cov[i, i] + model.cov[j, j] - 2.0 * model.cov[i, j])
    var_gap = max(var_gap, 0.0)
    gap = float(model.rewards[i] - model.rewards[j])
    p = float(1.0 / (1.0 + np.exp(-gap)))
    fisher = p * (1.0 - p)                                  # Bradley-Terry/Bernoulli information
    score = var_gap * fisher
    return score, p, float(np.sqrt(var_gap))


def rank_comparisons(
    model: RewardModel,
    candidates: list[str] | None = None,
) -> list[Comparison]:
    """Score every candidate unordered pair by expected information gain, most-informative first."""
    items = candidates if candidates is not None else list(model.items)
    items = [it for it in items if it in model.items]
    if len(items) < 2:
        raise ValueError("need at least two known candidate items to propose a comparison.")
    out: list[Comparison] = []
    for a, b in combinations(items, 2):
        i, j = model.index_of(a), model.index_of(b)
        score, p, gap_std = _pair_score(model, i, j)
        out.append(Comparison(a, b, score, p, gap_std))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def next_comparison(
    model: RewardModel,
    candidates: list[str] | None = None,
) -> Comparison:
    """The single most informative next comparison to ask a human to judge."""
    return rank_comparisons(model, candidates)[0]


def eig_of_pair(model: RewardModel, a: str, b: str, *, noise: float = 1.0) -> float:
    """Expected information gain of comparing ``a`` vs ``b`` via mixle.doe's linear-Gaussian EIG.

    Models the single comparison as a one-row contrast design ``F = e_a - e_b`` over the reward vector
    with prior covariance = the current bootstrap posterior ``cov``; this is the exact pairwise
    specialisation of ``mixle.doe.expected_information_gain_linear``. Returned in nats.
    """
    if _eig_linear is None:  # pragma: no cover
        score, _, _ = _pair_score(model, model.index_of(a), model.index_of(b))
        return float(score)
    k = len(model.items)
    i, j = model.index_of(a), model.index_of(b)
    f = np.zeros((1, k))
    f[0, i], f[0, j] = 1.0, -1.0
    return float(_eig_linear(f, noise=noise, prior_cov=model.cov))
