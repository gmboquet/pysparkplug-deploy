"""Feature-conditioned reward model — a reward that generalizes to UNSEEN text.

The Bradley-Terry reward in :mod:`reward` scores a *fixed candidate pool* by item identity: each item has
its own learned log-worth, so a brand-new piece of text has no reward at all. This module instead learns a
reward as a function of the text's *features* (an embedding), so it scores any string — including text the
fitter never saw.

Model
-----
The reward is **linear over the embedding**::

    r(x) = w · phi(x) + b

where ``phi(x)`` is the L2-normalised embedding from :class:`mixle_mlops.rag.embeddings.Embedder`.

Fit (Bradley-Terry / logistic on the embedding difference)
----------------------------------------------------------
Each preference ``(winner, loser)`` says ``P(winner ≻ loser) = sigmoid(r(winner) - r(loser))``. Because the
reward is linear, the bias cancels in the difference::

    r(winner) - r(loser) = w · (phi(winner) - phi(loser))

so with ``d = phi(winner) - phi(loser)`` every pair is a logistic-regression example ``(d, y=1)`` and the
fit minimises the **L2-regularised Bradley-Terry / logistic loss**::

    L(w) = sum_pairs  -log sigmoid(w · d)  +  (lambda / 2) ||w||^2

This is convex. We solve it with **Newton / IRLS** (a handful of steps): with ``p = sigmoid(Xw)``, gradient
``g = X^T (p - 1) + lambda w`` and Hessian ``H = X^T diag(p(1-p)) X + lambda I``, step ``w <- w - H^{-1} g``.
Falls back to plain gradient descent if a Newton step is ever non-finite.

The bias ``b`` is *unidentifiable from pairwise preferences* (it cancels in every difference), so it only
shifts all rewards by a constant and changes neither the ranking nor any ``P(a ≻ b)``. We set it so the mean
reward over the training texts is 0 (a harmless, interpretable gauge).

Honesty / limits
----------------
This is a **linear (1-layer) reward over embeddings** — deliberately dependency-light (numpy only) and convex,
so the fit is deterministic and has no local optima. It generalizes to unseen text, but its ceiling is the
embedding: quality directions the embedder cannot represent cannot be learned. A small MLP would lift that
ceiling at the cost of convexity and extra knobs; we keep the linear model and document the trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from ..rag.embeddings import Embedder, get_embedder


class _EmbedderLike(Protocol):
    """Minimal embedder contract this model depends on (so a toy embedder can be injected in tests)."""

    def embed(self, texts: Sequence[str] | str) -> np.ndarray: ...

    def embed_one(self, text: str) -> np.ndarray: ...


@dataclass
class FeatureRewardModel:
    """A linear, feature-conditioned reward ``r(x) = w · phi(x) + b`` fit from preference pairs.

    Unlike the identity-based Bradley-Terry reward, this scores *any* text via its embedding, so it
    generalizes to inputs that were never part of the training comparisons.

    Args:
        embedder: any object with ``embed`` / ``embed_one`` returning L2-normalised vectors. Defaults to the
            platform embedder (:func:`mixle_mlops.rag.embeddings.get_embedder`). Inject a toy embedder in tests.
        l2: L2 (ridge) regularisation strength ``lambda`` on ``w``. Keeps the fit well-posed when pairs are
            few or separable.
        max_iter: maximum Newton/IRLS iterations.
        tol: convergence tolerance on the gradient infinity-norm.
    """

    embedder: _EmbedderLike = field(default_factory=get_embedder)
    l2: float = 1.0
    max_iter: int = 50
    tol: float = 1e-8

    w: np.ndarray | None = field(default=None, init=False, repr=False)
    b: float = field(default=0.0, init=False)
    n_pairs: int = field(default=0, init=False)
    n_iter: int = field(default=0, init=False)
    family: str = field(default="FeatureBradleyTerry", init=False)

    # -- fitting -----------------------------------------------------------------------------------

    def _embed_many(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.asarray(self.embedder.embed(list(texts)), dtype=np.float64)
        return np.atleast_2d(mat)

    def fit(self, pairs: Sequence[tuple[str, str]]) -> "FeatureRewardModel":
        """Fit ``w`` (and the gauge bias ``b``) from ``(winner, loser)`` preference pairs.

        Minimises the L2-regularised Bradley-Terry / logistic loss on the embedding differences
        ``phi(winner) - phi(loser)`` (all labels 1) via Newton/IRLS.
        """
        pairs = list(pairs)
        if not pairs:
            raise ValueError("need at least one (winner, loser) preference pair to fit.")

        winners = [w for w, _ in pairs]
        losers = [lo for _, lo in pairs]
        phi_w = self._embed_many(winners)
        phi_l = self._embed_many(losers)
        x = phi_w - phi_l                                  # (n, dim): winner ≻ loser  =>  w·x > 0 desired
        n, dim = x.shape
        self.n_pairs = n

        lam = float(self.l2)
        w = np.zeros(dim, dtype=np.float64)
        last_iter = 0
        for it in range(self.max_iter):
            last_iter = it + 1
            z = x @ w
            p = _sigmoid(z)                                # p = P(winner ≻ loser) under current w
            grad = x.T @ (p - 1.0) + lam * w               # all labels are 1
            if np.max(np.abs(grad)) < self.tol:
                last_iter = it
                break
            s = p * (1.0 - p)                              # IRLS weights
            hess = (x.T * s) @ x + lam * np.eye(dim)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = grad / (np.max(s) + lam)            # GD fallback
            if not np.all(np.isfinite(step)):
                step = grad / (np.max(s) + lam)            # GD fallback on non-finite Newton step
            w = w - step
        self.n_iter = last_iter
        self.w = w

        # Gauge: b only shifts all rewards by a constant (cancels in every pairwise difference and in the
        # ranking). Pick b so mean reward over the training texts is 0 — interpretable, ranking-neutral.
        all_phi = np.vstack([phi_w, phi_l])
        self.b = float(-np.mean(all_phi @ w))
        return self

    # -- scoring -----------------------------------------------------------------------------------

    def _check_fitted(self) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("FeatureRewardModel is not fitted; call fit(pairs) first.")
        return self.w

    def score(self, text: str) -> float:
        """Reward ``r(text) = w · phi(text) + b``. Higher = preferred. Works on unseen text."""
        w = self._check_fitted()
        phi = np.asarray(self.embedder.embed_one(text), dtype=np.float64)
        return float(phi @ w + self.b)

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        """Vectorised :meth:`score` over many texts → ``(len(texts),)`` rewards."""
        w = self._check_fitted()
        texts = list(texts)
        if not texts:
            return np.zeros(0)
        phi = self._embed_many(texts)
        return phi @ w + self.b

    def rank(self, texts: Sequence[str]) -> list[int]:
        """Indices of ``texts`` ordered best-first (descending reward)."""
        return list(np.argsort(-self.scores(texts)))

    def prob_prefer(self, a: str, b: str) -> float:
        """Calibrated ``P(a ≻ b) = sigmoid(r(a) - r(b))`` (the bias cancels)."""
        return float(_sigmoid(self.score(a) - self.score(b)))


def _sigmoid(z):
    """Numerically stable logistic sigmoid (scalar or array). Returns a float for scalar input."""
    scalar = np.isscalar(z) or (isinstance(z, np.ndarray) and z.ndim == 0)
    arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    out = np.empty_like(arr)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    ez = np.exp(arr[~pos])
    out[~pos] = ez / (1.0 + ez)
    return float(out[0]) if scalar else out


def fit_feature_reward(
    pairs: Sequence[tuple[str, str]],
    *,
    embedder: _EmbedderLike | None = None,
    l2: float = 1.0,
    allow_remote: bool = True,
) -> FeatureRewardModel:
    """Convenience constructor: build a :class:`FeatureRewardModel` and fit it on ``pairs``.

    Args:
        pairs: ``(winner, loser)`` text preference pairs.
        embedder: optional injected embedder; defaults to the platform embedder (local fallback when no
            embedding server is reachable). Pass ``allow_remote=False`` to force the deterministic local
            embedder without a server.
        l2: ridge regularisation strength.
        allow_remote: when no ``embedder`` is given, whether the default :class:`Embedder` may call a server.
    """
    if embedder is None:
        embedder = Embedder(allow_remote=allow_remote)
    model = FeatureRewardModel(embedder=embedder, l2=l2)
    return model.fit(pairs)
