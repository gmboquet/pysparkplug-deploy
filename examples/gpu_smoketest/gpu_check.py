"""Minimal GPU smoke test: confirm CUDA is usable on a rented box and run a real timed workload.

Run via the training pipeline (it rents the box, ships this, runs it, fetches the report, destroys the box):

    mixle-mlops train gpu-smoketest --backend mixle --script gpu_check.py \\
        --workdir examples/gpu_smoketest --gpu RTX_4090 --max-price 0.4 --max-runtime 20 --no-dry-run

Needs only torch (already in the GPU image). Writes a JSON report to --output so the pipeline has something
to download as the "artifact".
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifact")
    p.add_argument("--dataset", default=None)  # accepted/ignored (pipeline convention)
    a = p.parse_args()

    info = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    print(f"torch {info['torch']} | cuda_available={info['cuda_available']} | devices={info['device_count']}")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        print(f"GPU: {info['device_name']} (sm_{info['capability'][0]}{info['capability'][1]})")
        # real GPU workload: a few large matmuls, timed -> effective TFLOP/s
        n = 4096
        x = torch.randn(n, n, device=dev)
        y = torch.randn(n, n, device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            z = x @ y  # noqa: F841
        torch.cuda.synchronize()
        dt = time.time() - t0
        info["matmul_tflops"] = round(20 * 2 * n**3 / dt / 1e12, 2)
        print(f"20x {n}x{n} matmul: {dt:.3f}s  ~{info['matmul_tflops']} TFLOP/s")
    else:
        print("!! NO CUDA — GPU not usable on this box")

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gpu_report.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {out / 'gpu_report.json'}")


if __name__ == "__main__":
    main()
