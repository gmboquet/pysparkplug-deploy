"""Per-task self-evolution policy + the mapping from policy names to ``mixle.evolve`` objects.

The policy is the serving-side knob set; ``build_objective`` / ``build_operators`` translate it into the
library's ``Objective`` and ``ImprovementOperator`` instances. Keeping the translation here means the route
and the worker stay free of ``mixle.evolve`` import details."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

OBJECTIVES = ("nll", "log_score", "crps", "interval", "calibration")
OPERATORS = {"refit", "online", "autoselect", "recalibrate"}


class EvolutionPolicy(BaseModel):
    """How to improve a hosted model: what to measure, which moves to try, and the promotion gate."""
    objective: str = "nll"                                   # nll|log_score|crps|interval|calibration
    objective_params: dict[str, Any] = Field(default_factory=dict)
    operators: list[str] | None = None                       # subset of OPERATORS; None = library defaults
    alpha: float = 0.05                                      # significance level for the verify gate
    min_effect: float = 0.0                                  # practical effect-size floor
    holdout: float = 0.25                                    # train/verify split
    approval: str = "none"                                   # none -> auto-promote on a verified win; required -> hold
    seed: int = 0


def build_objective(policy: EvolutionPolicy):
    """Map ``policy.objective`` (+ params) to a ``mixle.evolve`` Objective."""
    from mixle.evolve import (
        calibration_objective,
        crps_objective,
        interval_objective,
        log_score_objective,
        nll_objective,
    )

    name = policy.objective
    p = dict(policy.objective_params)
    if name == "nll":
        return nll_objective()
    if name == "log_score":
        return log_score_objective()
    if name == "crps":
        return crps_objective(ensemble=int(p.get("ensemble", 256)), seed=policy.seed)
    if name == "interval":
        return interval_objective(level=float(p.get("level", 0.9)),
                                  ensemble=int(p.get("ensemble", 256)), seed=policy.seed)
    if name == "calibration":
        return calibration_objective(ensemble=int(p.get("ensemble", 256)), seed=policy.seed,
                                     bins=int(p.get("bins", 10)))
    raise ValueError(f"unknown objective {name!r}; choose from {OBJECTIVES}")


def build_operators(policy: EvolutionPolicy):
    """Map ``policy.operators`` (names) to operator instances, or ``None`` for the library defaults."""
    if policy.operators is None:
        return None
    from mixle.evolve import AutoSelect, OnlineUpdate, Recalibrate, Refit

    classes = {"refit": Refit, "online": OnlineUpdate, "autoselect": AutoSelect, "recalibrate": Recalibrate}
    ops = [classes[n]() for n in policy.operators if n in classes]
    return ops or None
