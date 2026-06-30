"""Focal-diversity proposer selection: pick decorrelated proposers, not error-correlated near-duplicates.

Uses an injected *toy* embedder that maps known texts to known unit vectors, so the geometry is exact and the
assertions are about the selection logic — not about any real embedding model. Two answers are near-duplicates
(almost identical vectors) and two are distinct (orthogonal); a good focal-diversity selection of ``k=2`` must
span the distinct directions rather than collapse onto the duplicate pair.
"""
from __future__ import annotations

import numpy as np
import pytest

from mixle_mlops.gateway.moa_select import (
    focal_diversity_select,
    mean_diversity,
    pairwise_cosine,
)


class ToyEmbedder:
    """Maps each known text to a fixed L2-normalised vector; unknown text -> zero vector."""

    def __init__(self, table: dict[str, np.ndarray]):
        self._table = {}
        for text, vec in table.items():
            v = np.asarray(vec, dtype=np.float64)
            norm = np.linalg.norm(v)
            self._table[text] = v / norm if norm > 0 else v

    def embed(self, texts) -> np.ndarray:
        return np.vstack([self._table.get(t, np.zeros(self._dim)) for t in texts])

    @property
    def _dim(self) -> int:
        return len(next(iter(self._table.values())))


# Two near-duplicate answers (proposers 0 and 1) and two distinct answers (proposers 2 and 3).
#   dup_a, dup_b  ->  nearly the same direction  (cosine ~ 1, distance ~ 0)
#   distinct_x    ->  orthogonal to the duplicates and to distinct_y
#   distinct_y    ->  orthogonal to everything else
_TABLE = {
    "dup_a": np.array([1.0, 0.0, 0.0, 0.0]),
    "dup_b": np.array([0.999, 0.044, 0.0, 0.0]),   # cosine to dup_a ~ 0.999 (near-duplicate)
    "distinct_x": np.array([0.0, 0.0, 1.0, 0.0]),  # orthogonal -> cosine 0
    "distinct_y": np.array([0.0, 0.0, 0.0, 1.0]),  # orthogonal -> cosine 0
}
_ANSWERS = ["dup_a", "dup_b", "distinct_x", "distinct_y"]
_DUP_PAIR = [0, 1]


@pytest.fixture
def embedder() -> ToyEmbedder:
    return ToyEmbedder(_TABLE)


def test_selects_distinct_not_duplicate_pair(embedder):
    """k=2 must return two *decorrelated* proposers — never the near-duplicate pair {0, 1}."""
    selected = focal_diversity_select(_ANSWERS, k=2, embedder=embedder)

    assert len(selected) == 2
    assert len(set(selected)) == 2, "selected indices must be distinct"
    assert set(selected) != set(_DUP_PAIR), "must not pick the error-correlated near-duplicate pair"
    # At most one member of the duplicate pair may survive (they are redundant with each other).
    assert len(set(selected) & set(_DUP_PAIR)) <= 1


def test_selected_set_more_diverse_than_duplicate_pair(embedder):
    """The chosen set's mean diversity must exceed the (near-zero) diversity of the duplicate pair."""
    emb = embedder.embed(_ANSWERS)
    selected = focal_diversity_select(_ANSWERS, k=2, embedder=embedder)

    div_selected = mean_diversity(selected, emb)
    div_dup_pair = mean_diversity(_DUP_PAIR, emb)

    assert div_selected > div_dup_pair
    # The duplicates are nearly identical, so their pairwise diversity is essentially zero.
    assert div_dup_pair < 0.05
    # An orthogonal pick pair has cosine ~0 -> diversity ~1.
    assert div_selected > 0.9


def test_quality_tradeoff_seeds_on_quality(embedder):
    """With quality + alpha=1 (quality only), the top-quality proposer is always selected."""
    quality = [0.1, 0.2, 0.9, 0.3]  # proposer 2 is best
    selected = focal_diversity_select(_ANSWERS, k=2, embedder=embedder, quality=quality, alpha=1.0)
    assert 2 in selected


def test_k_clamped_and_indices_valid(embedder):
    """k is clamped to [1, n]; returned indices are valid and unique."""
    selected = focal_diversity_select(_ANSWERS, k=99, embedder=embedder)
    assert len(selected) == len(_ANSWERS)
    assert sorted(selected) == [0, 1, 2, 3]

    one = focal_diversity_select(_ANSWERS, k=0, embedder=embedder)
    assert len(one) == 1


def test_pairwise_cosine_shape_and_diagonal(embedder):
    """Observability helper: square similarity matrix, unit diagonal, symmetric, near-1 for the duplicates."""
    emb = embedder.embed(_ANSWERS)
    sim = pairwise_cosine(emb)
    assert sim.shape == (4, 4)
    assert np.allclose(np.diag(sim), 1.0)
    assert np.allclose(sim, sim.T)
    assert sim[0, 1] > 0.99          # near-duplicates
    assert abs(sim[2, 3]) < 1e-9     # orthogonal distinct answers


def test_empty_answers_returns_empty(embedder):
    assert focal_diversity_select([], k=2, embedder=embedder) == []
