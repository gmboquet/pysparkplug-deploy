"""Train a mixle model with the *torch compute engine* on a GPU — validates the engine E-step on CUDA.

Unlike the neural-experts example (where only the leaf's gradient M-step is on the GPU), here the whole
EM runs through `engine=TorchEngine(device=...)`: the forward/score/accumulate of a Gaussian mixture
execute on the device. Recovers a two-component mixture and saves the (pure-stats) model.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from mixle.engines import TorchEngine
from mixle.inference import optimize
from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifact")
    p.add_argument("--dataset", default=None)  # accepted/ignored (pipeline convention)
    p.add_argument("--device", default=None, help="torch engine device; default auto (cuda>mps>cpu)")
    a = p.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"torch {torch.__version__} | compute-engine device: {dev} | cuda={torch.cuda.is_available()}")

    rng = np.random.RandomState(0)
    data = np.concatenate([rng.normal(-4, 1, 5000), rng.normal(4, 1, 5000)]).tolist()
    init = MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
    model = optimize(
        data,
        MixtureEstimator([GaussianEstimator(), GaussianEstimator()]),
        max_its=40,
        engine=TorchEngine(device=dev),
        prev_estimate=init,
    )
    mus = sorted(round(c.mu, 3) for c in model.components)
    print(f"recovered mus={mus} on the torch engine ({dev}) — expect ~ -4, +4")

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    (out / "report.json").write_text(json.dumps({"engine_device": dev, "mus": mus, "cuda": torch.cuda.is_available()}))
    print(f"saved mixle model -> {out / 'model.pkl'}")


if __name__ == "__main__":
    main()
