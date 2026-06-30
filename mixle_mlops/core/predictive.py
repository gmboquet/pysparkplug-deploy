"""Serve a fitted mixle ``Distribution`` as a *predictive distribution*, not a point.

This is the platform's differentiator over an LLM proxy: given a fitted mixle model and a batch of
records, return per-record means, quantiles, central credible intervals, an ensemble of draws, and a
``cdf-at-y`` callable -- whatever the fitted distribution actually supports, chosen by *capability*
rather than class name.

Dispatch (via :mod:`mixle.capability`):

* **Closed-form path** -- when the model is :class:`~mixle.capability.HasCDF` (exact ``cdf`` +
  ``quantile``), quantiles/intervals/CDF come straight from the inverse-CDF and the density is exact.
* **Ensemble path** -- otherwise we draw a Monte-Carlo ensemble with ``model.sampler(seed).sample(n)``
  and read empirical quantiles/mean/CDF off the draws; the density semantics are then a Monte-Carlo
  *estimate* (or whatever the model declares).

If the model supports neither closed-form CDF nor sampling, a :class:`~mixle_mlops.core.adapters.CapabilityError`
is raised so the gateway can answer 422.

This module is written to upstream cleanly into mixle-core later (it only uses the public
``Distribution`` contract + ``mixle.capability``); the package keeps only the HTTP/serving opinions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from mixle import capability as cap
from mixle.stats.compute.pdist import DensitySemantics

from .adapters import CapabilityError

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass
class RecordPrediction:
    """The predictive summary for a single record/conditioning context."""

    mean: float | None
    quantiles: dict[float, float]
    interval: tuple[float, float]            # central credible interval at ``Predictions.interval_level``
    ensemble: list[float]                    # Monte-Carlo draws (empty on the pure closed-form path)
    cdf_at_y: float | None = None            # F(y) when an observed/target y was supplied per record

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "quantiles": {str(q): v for q, v in self.quantiles.items()},
            "interval": list(self.interval),
            "ensemble": self.ensemble,
            "cdf_at_y": self.cdf_at_y,
        }


@dataclass
class Predictions:
    """A batch of predictive summaries plus the metadata a calibration/decision layer needs.

    ``density_semantics`` tags whether the predictive numbers are *exact* (closed-form inverse-CDF) or a
    Monte-Carlo *estimate*; ``path`` records which dispatch branch produced them. ``cdf`` is a callable
    ``cdf(record_index, y) -> F(y)`` so a caller can ask tail probabilities after the fact.
    """

    records: list[RecordPrediction]
    quantile_levels: tuple[float, ...]
    interval_level: float
    path: str                                # "closed_form" | "ensemble"
    density_semantics: str                   # DensitySemantics value: "exact" | "estimate" | ...
    n_ensemble: int = 0
    _cdf_fns: list[Callable[[float], float]] = field(default_factory=list, repr=False)

    def cdf(self, record_index: int, y: float) -> float:
        """F_i(y) for record ``i`` -- exact on the closed-form path, empirical on the ensemble path."""
        return float(self._cdf_fns[record_index](y))

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "density_semantics": self.density_semantics,
            "quantile_levels": list(self.quantile_levels),
            "interval_level": self.interval_level,
            "n_ensemble": self.n_ensemble,
            "records": [r.as_dict() for r in self.records],
        }


def _conditioned(model: Any, record: Any) -> Any:
    """Return the per-record predictive distribution.

    A record may be a conditioning context for a :class:`~mixle.capability.Conditionable` model
    (``{coord: value}``) -- then we condition on it. Otherwise the record is treated as an *observed
    target* (used only for ``cdf_at_y``) and the unconditional fitted model is the predictive law.
    """
    if isinstance(record, dict) and cap.supports(model, cap.Conditionable):
        observed = {int(k): float(v) for k, v in record.items()}
        try:
            return model.condition(observed)
        except Exception:
            return model
    return model


def _target_y(record: Any) -> float | None:
    """Pull an observed target value ``y`` from a record for the cdf-at-y query, if one is present."""
    if record is None:
        return None
    if isinstance(record, dict):
        for key in ("y", "target", "value"):
            if key in record:
                try:
                    return float(record[key])
                except (TypeError, ValueError):
                    return None
        return None
    if isinstance(record, (int, float, np.integer, np.floating)):
        return float(record)
    return None


def _draw_ensemble(dist: Any, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` iid samples, tolerating leaf samplers that do not accept ``batched=``."""
    sampler = dist.sampler(seed)
    try:
        draws = sampler.sample(n, batched=True)
    except TypeError:
        draws = sampler.sample(n)
    arr = np.asarray(draws, dtype=float).reshape(-1)
    return arr


def _empirical_cdf(sorted_draws: np.ndarray) -> Callable[[float], float]:
    n = sorted_draws.size

    def _cdf(y: float) -> float:
        # right-continuous empirical CDF: fraction of draws <= y
        return float(np.searchsorted(sorted_draws, y, side="right") / n)

    return _cdf


def predictive_batch(
    model: Any,
    records: Sequence[Any],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    interval_level: float = 0.9,
    n_ensemble: int = 2000,
    seed: int = 0,
) -> Predictions:
    """Serve ``model``'s predictive distribution over ``records``.

    Dispatch is by capability: closed-form via :class:`~mixle.capability.HasCDF`, else a Monte-Carlo
    ensemble via the model's sampler. Raises :class:`CapabilityError` when the model can neither give a
    CDF nor be sampled.

    Args:
        model: a fitted mixle ``Distribution``.
        records: one entry per prediction. Each entry may be a conditioning context (``dict`` for a
            conditionable model) and/or carry an observed target ``y`` for the cdf-at-y query; pass
            ``None``/scalars for the unconditional predictive.
        quantiles: predictive quantile levels to report.
        interval_level: central credible-interval mass (0.9 -> the 5th/95th percentiles).
        n_ensemble: ensemble size on the sampling path (and for a closed-form model's ``ensemble`` field).
        seed: base RNG seed; record ``i`` uses ``seed + i`` for reproducible, independent streams.
    """
    quantiles = tuple(float(q) for q in quantiles)
    lo_q = (1.0 - interval_level) / 2.0
    hi_q = 1.0 - lo_q

    has_cdf = cap.supports(model, cap.HasCDF)
    samplable = callable(getattr(model, "sampler", None))
    if not has_cdf and not samplable:
        raise CapabilityError(getattr(model, "name", type(model).__name__), "predict")

    path = "closed_form" if has_cdf else "ensemble"
    out: list[RecordPrediction] = []
    cdf_fns: list[Callable[[float], float]] = []

    for i, record in enumerate(records):
        dist = _conditioned(model, record)
        y = _target_y(record)

        # the conditioned slice may have lost the closed-form CDF; re-check per record.
        use_cdf = cap.supports(dist, cap.HasCDF)
        if use_cdf:
            qmap = {q: float(dist.quantile(q)) for q in quantiles}
            interval = (float(dist.quantile(lo_q)), float(dist.quantile(hi_q)))
            mean = float(dist.mean()) if cap.supports(dist, cap.HasMoments) else float(dist.quantile(0.5))
            ensemble: list[float] = []
            if samplable:
                ensemble = _draw_ensemble(dist, min(n_ensemble, 256), seed + i).tolist()
            cdf_fn: Callable[[float], float] = (lambda d: (lambda yy: float(d.cdf(yy))))(dist)
            cdf_at_y = float(dist.cdf(y)) if y is not None else None
        else:
            if not callable(getattr(dist, "sampler", None)):
                raise CapabilityError(getattr(model, "name", type(model).__name__), "predict")
            draws = _draw_ensemble(dist, n_ensemble, seed + i)
            sorted_draws = np.sort(draws)
            qmap = {q: float(np.quantile(sorted_draws, q)) for q in quantiles}
            interval = (float(np.quantile(sorted_draws, lo_q)), float(np.quantile(sorted_draws, hi_q)))
            mean = float(sorted_draws.mean())
            ensemble = draws.tolist()
            cdf_fn = _empirical_cdf(sorted_draws)
            cdf_at_y = cdf_fn(y) if y is not None else None

        cdf_fns.append(cdf_fn)
        out.append(
            RecordPrediction(
                mean=mean, quantiles=qmap, interval=interval, ensemble=ensemble, cdf_at_y=cdf_at_y
            )
        )

    semantics = _predictive_semantics(model, path)
    return Predictions(
        records=out,
        quantile_levels=quantiles,
        interval_level=interval_level,
        path=path,
        density_semantics=semantics,
        n_ensemble=n_ensemble if path == "ensemble" else min(n_ensemble, 256),
        _cdf_fns=cdf_fns,
    )


def _predictive_semantics(model: Any, path: str) -> str:
    """Tag the predictive numbers' fidelity.

    Closed-form inverse-CDF quantiles are *exact*. Ensemble quantiles are a Monte-Carlo *estimate*,
    unless the model already declares a coarser density semantics we should not over-promise past.
    """
    declared = None
    fn = getattr(model, "density_semantics", None)
    if callable(fn):
        try:
            declared = fn()
        except Exception:
            declared = None
    if path == "closed_form":
        return (declared or DensitySemantics.EXACT).value
    # ensemble path: Monte-Carlo estimate (never tighter than the model's own declaration)
    if declared is not None and declared is not DensitySemantics.EXACT:
        return declared.value
    return DensitySemantics.ESTIMATE.value
