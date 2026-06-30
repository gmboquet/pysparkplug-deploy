"""Feature-conditioned reward model: it must recover the quality ordering on HELD-OUT (unseen) text.

We build a deterministic toy embedder whose first coordinate IS the (hidden) quality of a text, plus some
orthogonal nuisance coordinates. Preference pairs are generated from a *training* set of texts so the winner
is always the higher-quality one. After fitting on those pairs, the model must rank a DISJOINT set of texts
(never seen during fit) by their true quality — that generalization is the whole point of conditioning the
reward on features instead of item identity.
"""

from __future__ import annotations

import itertools
import zlib

import numpy as np

from mixle_mlops.feedback.feature_reward import FeatureRewardModel, fit_feature_reward
from mixle_mlops.rag.embeddings import Embedder


class ToyEmbedder:
    """Maps each text to a fixed vector whose coord 0 = quality, rest = deterministic nuisance.

    L2-normalised like the real :class:`Embedder`. Quality and nuisance are derived deterministically from
    the text, so identical text → identical vector (and held-out text still gets a meaningful embedding).
    """

    def __init__(self, qualities: dict[str, float], dim: int = 8, nuisance_radius: float = 2.0):
        self.qualities = qualities
        self.dim = dim
        self.nuisance_radius = nuisance_radius

    def _nuisance(self, text: str) -> np.ndarray:
        # Deterministic per-text nuisance in coords 1..dim-1: a quality-orthogonal distractor on a sphere of
        # FIXED radius. Fixed-radius (not free Gaussian) keeps every embedding's pre-normalisation norm
        # roughly constant, so the L2 normalisation the real Embedder applies stays approximately linear in
        # the quality coordinate — i.e. the embedding geometry is genuinely (near-)linear in quality, which
        # is exactly the regime in which a linear reward is expected to generalise.
        rng = np.random.default_rng(zlib.crc32(text.encode()))   # deterministic across processes (not hash())
        nz = rng.normal(size=self.dim - 1)
        nz /= np.linalg.norm(nz)
        return self.nuisance_radius * nz

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        # Center quality so coord 0 carries a signed signal symmetric about 0 (otherwise L2 normalisation
        # of an all-positive coordinate couples magnitude to the nuisance norm).
        qs = np.array(list(self.qualities.values()), dtype=np.float64)
        vec[0] = self.qualities[text] - qs.mean()
        vec[1:] = self._nuisance(text)
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.vstack([self.embed_one(t) for t in texts])


def _make_quality_map(prefix: str, n: int) -> dict[str, float]:
    # Distinct, monotone qualities so the ground-truth ordering is unambiguous.
    return {f"{prefix}_{i}": float(i) for i in range(n)}


def _spearman(a, b) -> float:
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ar, br)[0, 1])


def test_recovers_ordering_on_held_out_text():
    train = _make_quality_map("train", 8)
    held_out = _make_quality_map("heldout", 8)
    all_q = {**train, **held_out}
    emb = ToyEmbedder(all_q, dim=8, nuisance_radius=1.0)

    # Pairs from TRAIN ONLY; winner is the higher-quality text. (Held-out texts never appear here.)
    train_texts = list(train)
    pairs = [
        (a, b) if train[a] > train[b] else (b, a)
        for a, b in itertools.combinations(train_texts, 2)
    ]

    # Adequate regularization so the reward locks onto the *consistent* quality direction rather than overfitting
    # train-specific nuisance — the regime in which a feature reward generalizes.
    model = FeatureRewardModel(embedder=emb, l2=1.0).fit(pairs)

    # The model has never seen any held-out text. Its scores must rank them by true quality (strong correlation,
    # not a knife-edge perfect order which would depend on the exact nuisance geometry).
    held_texts = list(held_out)
    scores = model.scores(held_texts)
    true = np.array([held_out[t] for t in held_texts], dtype=float)
    rho = _spearman(scores, true)
    assert rho >= 0.85, rho                                    # generalizes to unseen text
    # the extremes are unambiguous: the best held-out text out-scores the worst
    assert model.score(held_texts[int(np.argmax(true))]) > model.score(held_texts[int(np.argmin(true))])


def test_prob_prefer_is_calibrated_direction():
    q = _make_quality_map("t", 5)
    emb = ToyEmbedder(q, dim=6)
    texts = list(q)
    pairs = [
        (a, b) if q[a] > q[b] else (b, a)
        for a, b in itertools.combinations(texts, 2)
    ]
    model = fit_feature_reward(pairs, embedder=emb, l2=1e-3)

    hi, lo = "t_4", "t_0"
    assert model.prob_prefer(hi, lo) > 0.5
    assert model.prob_prefer(lo, hi) < 0.5
    # Symmetric: P(a≻b) + P(b≻a) == 1 (up to float rounding of sigmoid)
    assert abs(model.prob_prefer(hi, lo) + model.prob_prefer(lo, hi) - 1.0) < 1e-9
    # bias cancels: prob_prefer is a function of the reward gap only
    assert abs(model.prob_prefer(hi, hi) - 0.5) < 1e-9


def test_local_fallback_embedder_no_server():
    # Real platform embedder forced to the deterministic local fallback (no server). Texts that literally
    # contain "excellent" should out-score "terrible" ones if we train that association.
    emb = Embedder(allow_remote=False)
    good = ["this is excellent and great", "excellent wonderful superb output", "an excellent fine answer"]
    bad = ["this is terrible and bad", "terrible awful poor output", "a terrible weak answer"]
    pairs = [(g, b) for g, b in zip(good, bad)]

    model = fit_feature_reward(pairs, embedder=emb, l2=1.0)

    # Held-out phrasing it never saw, but built from the same quality words.
    assert model.score("an excellent and superb result") > model.score("a terrible and awful result")
    assert model.prob_prefer("excellent great work", "terrible bad work") > 0.5


def test_rank_and_unfitted_guard():
    q = _make_quality_map("r", 4)
    emb = ToyEmbedder(q, dim=5)

    unfitted = FeatureRewardModel(embedder=emb)
    try:
        unfitted.score("r_0")
        raise AssertionError("expected RuntimeError on unfitted model")
    except RuntimeError:
        pass

    texts = list(q)
    pairs = [(a, b) if q[a] > q[b] else (b, a) for a, b in itertools.combinations(texts, 2)]
    model = FeatureRewardModel(embedder=emb, l2=1e-3).fit(pairs)
    order = model.rank(texts)
    assert order[0] == texts.index("r_3")  # highest quality first
    assert len(order) == len(texts)
