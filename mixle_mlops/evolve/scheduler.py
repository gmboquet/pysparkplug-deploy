"""Autonomous evolution scheduler — run the improve loop across all hosted mixle models in one pass.

A ``tick`` is the unit of autonomous self-improvement: for each mixle-backed model it runs the verify-gated
``EvolutionWorker`` (on supplied data, else the model's retained fit_data) and records the lineage. It can be
driven on a timer, on a drift trigger, or on demand via the ``/v1/evolve/tick`` route. The anti-regression
guarantee in the worker means a tick can only ever improve or no-op a model — never degrade one."""
from __future__ import annotations

from sqlmodel import Session

from ..core.registry import ModelRegistry
from .lineage import record_run
from .policy import EvolutionPolicy
from .worker import EvolutionRun, EvolutionWorker


class EvolutionScheduler:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def mixle_models(self) -> list[str]:
        return [info.id for info in self.registry.list() if info.kind == "mixle"]

    def tick(self, session: Session, policy: EvolutionPolicy | None = None, *,
             data_by_model: dict | None = None) -> list[EvolutionRun]:
        """Run one autonomous improvement pass over every mixle model; persist and return the runs."""
        policy = policy or EvolutionPolicy()
        worker = EvolutionWorker(self.registry)
        runs: list[EvolutionRun] = []
        for model_id in self.mixle_models():
            data = (data_by_model or {}).get(model_id, [])    # worker falls back to retained fit_data when empty
            run = worker.run(model_id, data, policy)
            record_run(session, run, user_id=None)
            runs.append(run)
        return runs
