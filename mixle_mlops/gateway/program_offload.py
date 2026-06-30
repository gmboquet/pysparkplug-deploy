"""Program-offload — the PAL / Program-aided-LM lever (Gao 2022; Schick 2023 Toolformer).

The model does not *compute*; it emits a structured spec and a deterministic solver computes the exact answer.
A 7B model that offloads arithmetic/probability/statistics to an exact solver matches far larger models on the
computational core of a problem — the reasoning correctness comes from the solver, not the parameters.

Security: the model's output is **never** ``eval``-ed. ``safe_eval`` is a locked-down AST walker that admits only
numeric literals, the arithmetic operators, a whitelist of math functions, and caller-provided variables — no
attribute access, no imports, no calls outside the whitelist. The other ops are exact closed-form / numpy."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Any] = {
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "min": min, "max": max,
    "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial, "comb": math.comb,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def safe_eval(expr: str, variables: dict[str, float] | None = None) -> float:
    """Evaluate an arithmetic expression with no access to Python internals. Raises ``ValueError`` on anything
    outside numbers/operators/whitelisted-functions/provided-variables."""
    variables = variables or {}
    tree = ast.parse(str(expr), mode="eval")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise ValueError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ValueError("only whitelisted functions may be called")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return float(ev(tree))


def _normal_prob(mean: float, std: float, x: float, side: str = "upper") -> float:
    """Exact Gaussian tail probability via the error function (stdlib, no dependency)."""
    if std <= 0:
        raise ValueError("std must be positive")
    z = (x - mean) / (std * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + math.erf(z))
    return cdf if side == "lower" else 1.0 - cdf


def _describe(data: list[float]) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(data, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty data")
    return {
        "n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "min": float(arr.min()), "max": float(arr.max()),
        "q05": float(np.quantile(arr, 0.05)), "median": float(np.median(arr)), "q95": float(np.quantile(arr, 0.95)),
    }


def _fit_predict(data: list[float], query: str = "mean", q: float = 0.5) -> dict[str, Any]:
    """Auto-fit the best mixle distribution to ``data`` and read off a statistic. Falls back to empirical."""
    try:
        import numpy as np

        from mixle.inference import optimize
        from mixle.utils.automatic import get_estimator

        rows = list(np.asarray(data, dtype=float).reshape(-1))
        est = get_estimator(rows)
        model = optimize(rows, est)
        sampler = getattr(model, "sampler", None)
        if callable(sampler):
            draws = np.asarray(sampler(0).sample(4000), dtype=float).reshape(-1)
            value = float(draws.mean()) if query == "mean" else float(np.quantile(draws, float(q)))
            return {"op": "fit_predict", "model": type(model).__name__, "query": query, "value": value}
    except Exception:
        pass
    desc = _describe(data)                                    # honest fallback: exact empirical statistic
    value = desc["mean"] if query == "mean" else float(__import__("numpy").quantile(data, float(q)))
    return {"op": "fit_predict", "model": "empirical", "query": query, "value": value}


def solve_program(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one offloaded computation spec exactly. Returns a JSON-serializable result (or ``{"error": ...}``)."""
    op = spec.get("op")
    try:
        if op == "eval":
            return {"op": "eval", "value": safe_eval(spec["expr"], spec.get("vars"))}
        if op == "normal_prob":
            return {"op": "normal_prob",
                    "probability": _normal_prob(float(spec["mean"]), float(spec["std"]), float(spec["x"]),
                                                str(spec.get("side", "upper")))}
        if op == "describe":
            return {"op": "describe", **_describe(spec["data"])}
        if op == "fit_predict":
            return _fit_predict(spec["data"], str(spec.get("query", "mean")), float(spec.get("q", 0.5)))
        return {"error": f"unknown op {op!r}; choose eval|normal_prob|describe|fit_predict"}
    except Exception as exc:
        return {"error": str(exc)}
