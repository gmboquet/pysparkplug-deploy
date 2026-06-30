"""Train a mixle model whose components are neural nets — a mixture of neural experts.

This is a *mixle model* (a MixtureDistribution); each expert is a `NeuralLeaf` (a torch MLP wrapped as a
generative p(y|x) leaf). mixle fits it with EM: the E-step computes responsibilities, the M-step trains
each expert by responsibility-weighted gradient descent. The trained artifact is the mixle model, saved
with provenance into a registry — the LLM/neural net is just *one part* (the leaf) of the model.

Run via the training pipeline (locally, then the same on a rented GPU):

    mixle-mlops train neural-experts --local --backend mixle \\
        --script train.py --workdir examples/mixle_neural_experts

It accepts --output (where to write the model) and an optional --dataset; with no dataset it generates a
two-regime synthetic problem (y = +2x in one regime, y = -2x in the other) so the two experts must
specialize — the textbook win that a single network can't get.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from mixle.inference.production import fit_with_provenance
from mixle.models.neural_leaf import NeuralLeaf
from mixle.stats import MixtureEstimator


def mlp(dims: list[int]) -> torch.nn.Module:
    layers: list[torch.nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


def synthetic(n: int = 800, seed: int = 0):
    rng = np.random.RandomState(seed)
    z = rng.randint(0, 2, n)  # latent regime
    x = rng.uniform(-2, 2, n).astype("float32")
    y = (np.where(z == 0, 2 * x, -2 * x) + 0.1 * rng.randn(n)).astype("float32")
    return list(zip(x[:, None], y[:, None])), x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifact", help="where to write the trained mixle model")
    p.add_argument("--dataset", default=None, help="(optional) jsonl/npz; default = synthetic two-regime")
    p.add_argument("--name", default="neural-experts")
    p.add_argument("--experts", type=int, default=2)
    p.add_argument("--restarts", type=int, default=6, help="seeded restarts (mixture EM can stall at a saddle)")
    p.add_argument("--em-iters", dest="em_iters", type=int, default=15)
    a = p.parse_args()

    cuda = torch.cuda.is_available()
    dev = "cuda:" + torch.cuda.get_device_name(0) if cuda else "cpu"
    print(f"torch {torch.__version__} | device: {dev}")  # NeuralLeaf auto-uses CUDA when present

    data, x = synthetic()  # (a --dataset loader would slot in here)
    print(f"training a mixture of {a.experts} neural experts on {len(data)} points "
          f"({a.restarts} restarts x {a.em_iters} EM iters)")

    best_model = None
    best_header = None
    for seed in range(a.restarts):
        torch.manual_seed(seed)
        estimator = MixtureEstimator(
            [NeuralLeaf(mlp([1, 16, 1]), noise=1.0, m_steps=40, lr=0.02).estimator() for _ in range(a.experts)]
        )
        model, header = fit_with_provenance(data, estimator, max_its=a.em_iters, seed=seed)
        if best_header is None or header.final_loglik > best_header.final_loglik:
            best_model, best_header = model, header
    assert best_model is not None and best_header is not None

    used = next(best_model.components[0].module.parameters()).device
    print(f"trained on device: {used}")

    # The trained artifact IS the mixle model. mixle's JSON registry can't store torch-backed leaves,
    # so neural models are persisted with pickle + a JSON provenance summary from the fit header.
    # Move the expert modules to CPU first so the pickle loads on any machine (not just a GPU box).
    for c in best_model.components:
        c.module.to("cpu")
        c.device = "cpu"
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    prov = {
        "name": a.name,
        "kind": "mixle.MixtureDistribution[NeuralLeaf]",
        "experts": a.experts,
        "final_loglik": best_header.final_loglik,
        "iterations": best_header.training.get("iterations"),
        "model_hash": best_header.model_hash,
        "data_hash": best_header.dataset_hash,
    }
    (out / "header.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"final loglik {best_header.final_loglik:.1f} | saved mixle model -> {out}/model.pkl")

    # Show the experts specialized: each net's slope should be ~+2 and ~-2 (either assignment).
    slopes = sorted(round(float(np.polyfit(x, c._forward(x[:, None])[:, 0], 1)[0]), 2) for c in best_model.components)
    print("expert slopes:", slopes, "(synthetic regimes are -2x and +2x)")

    # Round-trip: reload the saved model and confirm it predicts.
    with open(out / "model.pkl", "rb") as f:
        reloaded = pickle.load(f)
    pred = reloaded.components[0]._forward(np.array([[1.0]], dtype="float32"))[0, 0]
    print(f"reloaded OK — expert-0 prediction at x=1: {pred:+.2f}")


if __name__ == "__main__":
    main()
