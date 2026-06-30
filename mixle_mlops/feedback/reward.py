"""Fit a *calibrated* mixle preference / reward model over collected pairwise comparisons.

The reward family is mixle's :class:`mixle.stats.BradleyTerryDistribution` — a proper distribution over
ordered ``(winner, loser)`` pairs whose centered log-worths ``log_w[i]`` ARE the per-item rewards
(``P(i beats j) = sigmoid(reward_i - reward_j)``). We fit it with mixle's Zermelo / MM estimator
(``BradleyTerryEstimator``), the closed-form MLE.

Bradley-Terry has no closed-form parameter covariance under the MM fit, so the **uncertainty is
bootstrapped** with :func:`mixle.inference.bootstrap`: we resample the comparison pairs, refit the
worths each time, and report per-item std + a percentile CI. (``pseudo_count`` smoothing keeps
never-winners finite so the bootstrap statistic is always well defined.)

Output is a :class:`RewardModel`: item ids → reward (log-worth) WITH uncertainty, plus the fitted
``BradleyTerryDistribution`` and its bootstrap covariance (used by ``elicit.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.inference import bootstrap
from mixle.stats import BradleyTerryDistribution


@dataclass
class RewardItem:
    item_id: str
    reward: float                 # Bradley-Terry log-worth (centered): higher = preferred
    std: float                    # bootstrap standard error of the reward
    ci_low: float
    ci_high: float


@dataclass
class RewardModel:
    """A fitted Bradley-Terry reward model with bootstrap uncertainty over item rewards."""

    items: list[str]                                      # item id per Bradley-Terry index
    rewards: np.ndarray                                   # (K,) centered log-worths (the rewards)
    std: np.ndarray                                       # (K,) bootstrap std error
    ci_low: np.ndarray
    ci_high: np.ndarray
    cov: np.ndarray                                       # (K, K) bootstrap covariance of the rewards
    n_comparisons: int
    n_boot: int
    family: str = "BradleyTerry"
    distribution: BradleyTerryDistribution | None = field(default=None, repr=False)

    def index_of(self, item_id: str) -> int:
        return self.items.index(item_id)

    def reward_of(self, item_id: str) -> RewardItem:
        i = self.index_of(item_id)
        return RewardItem(item_id, float(self.rewards[i]), float(self.std[i]),
                          float(self.ci_low[i]), float(self.ci_high[i]))

    def ranking(self) -> list[RewardItem]:
        """Items sorted best-first by reward."""
        order = np.argsort(-self.rewards)
        return [
            RewardItem(self.items[i], float(self.rewards[i]), float(self.std[i]),
                       float(self.ci_low[i]), float(self.ci_high[i]))
            for i in order
        ]

    def prob_prefer(self, a: str, b: str) -> float:
        """Calibrated P(item a is preferred over item b) under the fitted Bradley-Terry model."""
        ra, rb = self.rewards[self.index_of(a)], self.rewards[self.index_of(b)]
        return float(1.0 / (1.0 + np.exp(-(ra - rb))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_comparisons": self.n_comparisons,
            "n_boot": self.n_boot,
            "items": [
                {
                    "item_id": self.items[i],
                    "reward": float(self.rewards[i]),
                    "std": float(self.std[i]),
                    "ci_low": float(self.ci_low[i]),
                    "ci_high": float(self.ci_high[i]),
                }
                for i in np.argsort(-self.rewards)
            ],
        }


def _fit_log_worths(encoded: np.ndarray, dim: int, pseudo_count: float) -> np.ndarray:
    """MLE Bradley-Terry centered log-worths from an ``(n, 2)`` array of ``(winner, loser)`` indices."""
    est = BradleyTerryDistribution(np.zeros(dim)).estimator(pseudo_count=pseudo_count)
    acc = est.accumulator_factory().make()
    acc.seq_update(np.ascontiguousarray(encoded.astype(np.int64)), np.ones(encoded.shape[0]), None)
    return est.estimate(encoded.shape[0], acc.value()).log_w


def fit_reward(
    pairs: list[tuple[str, str]],
    *,
    pseudo_count: float = 0.5,
    n_boot: int = 400,
    ci_level: float = 0.9,
    seed: int = 0,
) -> RewardModel:
    """Fit the calibrated Bradley-Terry reward model over ``(chosen_id, rejected_id)`` comparisons.

    Args:
        pairs: pairwise preferences as ``(chosen, rejected)`` opaque item-id tuples.
        pseudo_count: symmetric smoothing so never-winners keep a finite reward (and the bootstrap
            statistic is always defined).
        n_boot: number of bootstrap resamples used to quantify reward uncertainty.
        ci_level: central probability of the reported per-item credible/confidence interval.
        seed: bootstrap RNG seed.

    Returns:
        A :class:`RewardModel` with per-item reward + bootstrap std / CI and the bootstrap covariance.
    """
    if len(pairs) < 1:
        raise ValueError("need at least one pairwise comparison to fit a reward model.")

    items = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    if len(items) < 2:
        raise ValueError("need at least two distinct items to fit a Bradley-Terry reward model.")
    index = {item: i for i, item in enumerate(items)}
    dim = len(items)

    encoded = np.array([[index[c], index[r]] for c, r in pairs], dtype=np.int64)

    point = _fit_log_worths(encoded, dim, pseudo_count)

    # Bootstrap the worths: resample comparison rows, refit. mixle.inference.bootstrap drives the
    # resampling, returns the per-item CI, and exposes every replicate so we can form std + covariance.
    def statistic(rows: np.ndarray) -> np.ndarray:
        return _fit_log_worths(rows, dim, pseudo_count)

    n_boot = max(int(n_boot), 2)
    result = bootstrap(encoded, statistic, n_boot=n_boot, method="percentile",
                       ci_level=ci_level, seed=seed)
    reps = np.asarray(result.distribution)                # (n_boot, K) bootstrap replicates
    std = reps.std(axis=0, ddof=1)
    cov = np.cov(reps, rowvar=False)
    cov = np.atleast_2d(cov)
    ci_low = np.asarray(result.ci_low, dtype=float)
    ci_high = np.asarray(result.ci_high, dtype=float)

    return RewardModel(
        items=items,
        rewards=np.asarray(point, dtype=float),
        std=np.asarray(std, dtype=float),
        ci_low=ci_low,
        ci_high=ci_high,
        cov=cov,
        n_comparisons=int(encoded.shape[0]),
        n_boot=n_boot,
        distribution=BradleyTerryDistribution(point),
    )
