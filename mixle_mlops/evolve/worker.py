"""The ``EvolutionWorker`` — one run of the autonomous self-improvement loop for a hosted model.

It reads the champion (a fitted mixle model wrapped in a :class:`MixleAdapter`), runs ``mixle.evolve.improve``
on the supplied data under the task policy, and — only if the challenger *verifiably and non-regressively*
beats the champion — promotes it (serving the improved model immediately, keeping the previous one for rollback).

The ``improve`` driver carries the anti-regression guarantee: a run can never serve a worse model. The worker
adds the serving concerns the library deliberately omits: which model to read, when to swap, and rollback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..core.registry import ModelRegistry
from .policy import EvolutionPolicy, build_objective, build_operators


@dataclass
class EvolutionRun:
    model_id: str
    verified: bool
    operator: str | None
    delta: float
    objective: str
    promoted: bool
    n_data: int
    verdict: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id, "verified": self.verified, "operator": self.operator,
            "delta": self.delta, "objective": self.objective, "promoted": self.promoted,
            "n_data": self.n_data, "verdict": self.verdict, "error": self.error,
        }


class EvolutionWorker:
    """Run the measure → propose → verify → promote loop for one hosted mixle model."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def run(self, model_id: str, data: Sequence[Any], policy: EvolutionPolicy, *, promote: bool = True) -> EvolutionRun:
        if not self.registry.has(model_id):
            return EvolutionRun(model_id, False, None, 0.0, policy.objective, False, 0, error="model not found")
        adapter = self.registry.get(model_id)
        champion = getattr(adapter, "_model", None)
        if champion is None:                                 # only mixle-backed models can self-evolve
            return EvolutionRun(model_id, False, None, 0.0, policy.objective, False, 0,
                                error="model is not a mixle model (no fitted distribution to improve)")

        rows = list(data or [])
        retained = list(getattr(adapter, "_fit_data", None) or [])
        if len(rows) < 4:                                    # improve needs >=4 to split; fall back to retained data
            rows = retained + rows
        if len(rows) < 4:
            return EvolutionRun(model_id, False, None, 0.0, policy.objective, False, len(rows),
                                error="need >=4 observations to improve (supply records or retain fit_data)")

        try:
            from mixle.evolve import EvolutionLedger, improve

            objective = build_objective(policy)
            operators = build_operators(policy)
            ledger = EvolutionLedger()
            result = improve(champion, rows, objective=objective, operators=operators,
                             alpha=policy.alpha, min_effect=policy.min_effect, holdout=policy.holdout,
                             seed=policy.seed, ledger=ledger)
        except Exception as exc:
            return EvolutionRun(model_id, False, None, 0.0, policy.objective, False, len(rows), error=str(exc))

        promoted = False
        if result.verified and promote and policy.approval == "none":
            promoted = self._promote(adapter, model_id, result, retained, rows)
        return EvolutionRun(
            model_id, result.verified, result.operator, float(result.delta), policy.objective,
            promoted, len(rows), result.verdict.as_dict() if result.verdict else None,
        )

    def _promote(self, adapter: Any, model_id: str, result: Any, retained: list, rows: list) -> bool:
        """Serve the improved model. Keep the previous one for rollback; version it in a mixle Registry if backed."""
        previous = adapter._model
        registry = getattr(adapter, "_registry", None)
        if registry is not None:                             # best-effort durable versioning when registry-backed
            try:
                registry.register(result.model, model_id, metadata={"evolved": True, "operator": result.operator,
                                                                     "delta": float(result.delta)})
            except Exception:
                pass
        adapter._previous_model = previous                   # enable rollback
        adapter._model = result.model                        # serve the improved model immediately
        # online-update operators consumed `rows`; remember them so future improve() splits stay meaningful
        if hasattr(adapter, "_fit_data") and adapter._fit_data is not None:
            adapter._fit_data = (retained + rows)[-10000:]
        return True

    def rollback(self, model_id: str) -> bool:
        """Restore the previous champion for a model that was promoted. Returns whether anything was rolled back."""
        if not self.registry.has(model_id):
            return False
        adapter = self.registry.get(model_id)
        previous = getattr(adapter, "_previous_model", None)
        if previous is None:
            return False
        adapter._model = previous
        adapter._previous_model = None
        return True
