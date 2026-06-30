"""Focal-diversity proposer selection for Mixture-of-Agents.

Mixture-of-Agents (``moa.py``) only pays off when the proposers are *individually competent* AND *decorrelated* —
averaging error-correlated proposers reinforces shared mistakes rather than cancelling them. This module decides
*which* proposers enter the mix: given more candidate answers than the aggregator should consume, it greedily picks
a ``k``-subset that is both high quality and mutually diverse (a facility-location / greedy-MI style objective).

**Honest caveat (load-bearing).** True focal diversity is about *error* decorrelation — whether the proposers make
*different mistakes* — which you can only measure with labels or a judge. Here we have neither at selection time, so
we approximate it by the **embedding decorrelation of the answers themselves**: proposers whose answer embeddings are
far apart in cosine space are *taken as a proxy* for proposers whose errors are uncorrelated. This is a real but
coarse signal — two proposers can phrase the same wrong answer differently (false diversity), or phrase the same
right answer differently (false diversity that happens to be harmless). It is the best always-available, label-free
proxy; replace it with a verifier/judge-based error-correlation matrix where one exists (same greedy machinery).

API: :func:`focal_diversity_select` returns the selected proposer *indices*. :func:`pairwise_cosine` and
:func:`mean_diversity` are observability helpers for logging/inspecting the chosen set.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class _EmbedderLike(Protocol):
    """The minimal embedder contract this module needs: ``embed(texts) -> (n, dim)`` L2-normalised rows."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def pairwise_cosine(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine *similarity* matrix of the embedding rows.

    Assumes rows are (approximately) L2-normalised — which the platform embedder guarantees — so the dot product is
    the cosine similarity. We renormalise defensively (a zero row -> zero similarity, never a divide-by-zero) and
    clip into ``[-1, 1]`` to absorb floating-point overshoot. Shape ``(n, n)``; the diagonal is 1 for non-zero rows.
    """
    emb = np.asarray(emb, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D (n, dim); got shape {emb.shape}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    sim = unit @ unit.T
    return np.clip(sim, -1.0, 1.0)


def mean_diversity(indices: Sequence[int], emb: np.ndarray) -> float:
    """Mean pairwise cosine *distance* (``1 - similarity``) over the selected set — higher = more decorrelated.

    A single index (or empty) has no pairs, so diversity is ``0.0`` by convention. This is the scalar an observer
    logs to see *how* diverse the chosen proposers ended up being (the realised value of the selection objective).
    """
    idx = list(indices)
    if len(idx) < 2:
        return 0.0
    sim = pairwise_cosine(emb)
    sub = sim[np.ix_(idx, idx)]
    n = len(idx)
    # average over the off-diagonal pairs only
    off_sum = float(sub.sum() - np.trace(sub))
    mean_sim = off_sum / (n * (n - 1))
    return 1.0 - mean_sim


def focal_diversity_select(
    answers: Sequence[str],
    *,
    k: int,
    embedder: _EmbedderLike | None = None,
    quality: Sequence[float] | None = None,
    alpha: float = 0.5,
) -> list[int]:
    """Greedily select ``k`` proposer indices maximising a focal-*diversity* objective over their answer embeddings.

    Parameters
    ----------
    answers
        Candidate texts, one per proposer, for the current query.
    k
        Number of proposers to keep. Clamped to ``[1, len(answers)]``.
    embedder
        Anything with ``.embed(list[str]) -> np.ndarray`` (L2-normalised rows). Defaults to the platform embedder
        (``mixle_mlops.rag.embeddings.get_embedder()``), which itself falls back to a deterministic local hashing
        embedder when no embeddings server is reachable — so this works offline / in tests with no setup.
    quality
        Optional per-proposer competence scores (e.g. self-consistency confidence, a reward-model score, historical
        win-rate). When given, the seed is the highest-quality proposer and each subsequent pick maximises
        ``alpha * quality + (1 - alpha) * diversity`` (both terms min-max normalised to ``[0, 1]`` so ``alpha`` is a
        meaningful mixing weight regardless of the raw scales). When absent, selection is pure max-min diversity and
        the seed is proposer 0.
    alpha
        Quality/diversity trade-off in ``[0, 1]`` (only used when ``quality`` is given). ``1.0`` = quality only,
        ``0.0`` = diversity only. Default ``0.5``.

    Returns
    -------
    list[int]
        The selected proposer indices (the seed first, then in greedy pick order).

    Notes
    -----
    The diversity step is the classic max-min facility-location greedy: at each step add the *un*-selected proposer
    whose *worst-case* (maximum) cosine similarity to the already-selected set is smallest — i.e. the one that is
    least redundant with everything chosen so far. Diversity here is an embedding-decorrelation proxy for error
    decorrelation; see the module docstring for the honest limits of that proxy.
    """
    n = len(answers)
    if n == 0:
        return []
    k = max(1, min(int(k), n))

    if embedder is None:
        from ..rag.embeddings import get_embedder

        embedder = get_embedder()

    emb = np.asarray(embedder.embed(list(answers)), dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] != n:
        raise ValueError(f"embedder returned shape {emb.shape}; expected ({n}, dim) for {n} answers")

    sim = pairwise_cosine(emb)
    dist = 1.0 - sim  # cosine distance: higher = more diverse

    q = _normalize_quality(quality, n)

    # Seed: best quality if provided (ties -> lowest index), else proposer 0.
    if q is not None:
        seed = int(np.argmax(q))
    else:
        seed = 0
    selected = [seed]
    remaining = set(range(n)) - {seed}

    while len(selected) < k and remaining:
        cand = sorted(remaining)
        # Diversity gain of adding j = its distance to the NEAREST already-selected proposer (max-min).
        div_gain = np.array([min(dist[j, s] for s in selected) for j in cand], dtype=np.float64)

        if q is not None:
            div_score = _minmax(div_gain)
            qual_score = np.array([q[j] for j in cand], dtype=np.float64)  # already min-max normalised
            score = alpha * qual_score + (1.0 - alpha) * div_score
        else:
            score = div_gain

        pick = cand[int(np.argmax(score))]
        selected.append(pick)
        remaining.discard(pick)

    return selected


def _normalize_quality(quality: Sequence[float] | None, n: int) -> np.ndarray | None:
    """Validate + min-max normalise the quality vector to ``[0, 1]``; ``None`` passes through."""
    if quality is None:
        return None
    q = np.asarray(list(quality), dtype=np.float64)
    if q.shape != (n,):
        raise ValueError(f"quality must have one score per answer (len {n}); got shape {q.shape}")
    return _minmax(q)


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max scale to ``[0, 1]``; a constant vector maps to all-zeros (no information to exploit)."""
    x = np.asarray(x, dtype=np.float64)
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
