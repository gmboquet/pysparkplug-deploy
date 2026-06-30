"""Train on a rented vast.ai GPU: pick an offer, run the job (ssh or onstart), fetch + register the model.

`plan()` and `launch(dry_run=True)` are fully offline (no spend) — they show the offer query, the image,
the pip set, the training command, and (for onstart) the generated startup script. The live path provisions
a real instance and is gated behind dry_run=False (it needs your vast.ai key and costs money).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import jobspec
from .jobspec import TrainingJob
from .vast import VastClient, VastError

Log = Callable[[str], None]

_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]


def plan(job: TrainingJob, *, s3_dest: str | None = None) -> dict:
    job.validate()
    return {
        "job": asdict(job),
        "image": jobspec.resolve_image(job),
        "pip_install": jobspec.pip_packages(job),
        "offer_query": VastClient.build_query(
            gpu_name=job.gpu, num_gpus=job.num_gpus, max_price=job.max_price, limit=20
        ),
        "training_command": jobspec.training_command(job),
        "onstart": jobspec.build_onstart(job, s3_dest) if job.mode == "onstart" else None,
    }


def launch(
    job: TrainingJob,
    *,
    api_key: str = "",
    dry_run: bool = True,
    on_log: Log = print,
    registry_root: str | None = None,
    s3_dest: str | None = None,
    artifact_dir: str = "./trained",
) -> dict:
    job.validate()
    p = plan(job, s3_dest=s3_dest)

    if dry_run:
        on_log("DRY RUN — nothing is provisioned and no money is spent.\n")
        on_log(f"GPU offer search: {json.dumps(p['offer_query'])}")
        on_log(f"image:           {p['image']}")
        on_log(f"pip install:     {' '.join(p['pip_install'])}")
        on_log(f"train command:   {p['training_command']}")
        on_log(f"mode:            {job.mode}")
        if p["onstart"]:
            on_log("\n--- onstart script ---\n" + p["onstart"])
        on_log("\nRe-run with --no-dry-run (and MIXLE_VAST_API_KEY set) to actually rent a GPU and train.")
        return {"dry_run": True, "plan": p}

    client = VastClient(api_key)
    on_log(f"searching vast.ai for {job.num_gpus}x {job.gpu} <= ${job.max_price}/hr …")
    offers = client.search_offers(gpu_name=job.gpu, num_gpus=job.num_gpus, max_price=job.max_price)
    if not offers:
        raise VastError("no matching vast.ai offers — relax --gpu / --max-price")
    offer = offers[0]
    on_log(f"renting offer {offer.id}: {offer.gpu_name} x{offer.num_gpus} @ ${offer.price:.3f}/hr")

    if job.mode == "onstart":
        instance_id = client.create_instance(
            offer.id, image=p["image"], disk=job.disk, onstart=p["onstart"], runtype="ssh_direct", label=job.name
        )
        on_log(f"instance {instance_id} launched; it trains unattended and self-reports.")
        if not s3_dest:
            on_log("note: no object store configured — set s3_dest so the artifact is uploaded for retrieval.")
        return {"instance_id": instance_id, "mode": "onstart", "offer": offer.id, "s3_dest": s3_dest}

    # --- ssh mode ---
    instance_id = client.create_instance(
        offer.id, image=p["image"], disk=job.disk, runtype="ssh_direct", label=job.name
    )
    try:
        host, port = _wait_for_ssh(client, instance_id, on_log)
        _run_over_ssh(host, port, job, on_log)
        local = _fetch_artifact(host, port, job, artifact_dir, on_log)
        result: dict = {"instance_id": instance_id, "offer": offer.id, "artifact": local}
        if job.register and registry_root:
            result["registered"] = _register(job, local, registry_root, offer, on_log)
        return result
    finally:
        on_log(f"destroying instance {instance_id} …")
        try:
            client.destroy(instance_id)
        except Exception as e:  # noqa: BLE001
            on_log(f"warning: could not destroy instance {instance_id}: {e} — destroy it manually!")


def _wait_for_ssh(client: VastClient, instance_id: int, on_log: Log, timeout: float = 600) -> tuple[str, int]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        inst = client.instance(instance_id)
        status = inst.get("actual_status")
        host, port = inst.get("ssh_host"), inst.get("ssh_port")
        if status == "running" and host and port:
            on_log(f"instance running at {host}:{port}")
            time.sleep(5)  # let sshd settle
            return str(host), int(port)
        on_log(f"  waiting (status={status}) …")
        time.sleep(10)
    raise VastError("instance did not become reachable in time")


def _ssh(host: str, port: int, command: str, on_log: Log) -> None:
    proc = subprocess.Popen(
        ["ssh", *_SSH_OPTS, "-p", str(port), f"root@{host}", command],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        on_log(line.rstrip())
    if proc.wait() != 0:
        raise VastError("remote command failed")


def _run_over_ssh(host: str, port: int, job: TrainingJob, on_log: Log) -> None:
    on_log("uploading code …")
    if job.backend == "mixle" and job.workdir:
        subprocess.run(
            ["rsync", "-az", "-e", f"ssh {' '.join(_SSH_OPTS)} -p {port}",
             f"{job.workdir.rstrip('/')}/", f"root@{host}:/workspace/work/"],
            check=True,
        )
        remote_dir = "/workspace/work"
    else:
        remote_dir = "/workspace/work"
        _ssh(host, port, f"mkdir -p {remote_dir}", on_log)
        # ship the generated LoRA script
        b64 = __import__("base64").b64encode(jobspec.LLM_LORA_SCRIPT.encode()).decode()
        _ssh(host, port, f"echo {b64} | base64 -d > {remote_dir}/llm_lora_train.py", on_log)

    setup = " && ".join([
        f"cd {remote_dir}",
        "pip install -q " + " ".join(shlex.quote(p) for p in jobspec.pip_packages(job)),
        f"({jobspec.training_command(job)}) 2>&1 | tee train.log",
    ])
    on_log("training …")
    _ssh(host, port, setup, on_log)


def _fetch_artifact(host: str, port: int, job: TrainingJob, artifact_dir: str, on_log: Log) -> str:
    dest = Path(artifact_dir) / job.name
    dest.mkdir(parents=True, exist_ok=True)
    on_log(f"downloading artifact → {dest}")
    subprocess.run(
        ["scp", *_SSH_OPTS, "-P", str(port), "-r",
         f"root@{host}:/workspace/work/{job.output}", str(dest)],
        check=True,
    )
    return str(dest)


def _register(job: TrainingJob, local: str, registry_root: str, offer, on_log: Log) -> str:
    """Record the trained artifact + provenance under the registry root (works for both backends)."""
    root = Path(registry_root) / job.name
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": job.name,
        "backend": job.backend,
        "base_model": job.base_model,
        "artifact": local,
        "offer": {"id": offer.id, "gpu": offer.gpu_name, "price": offer.price},
        "created_at": time.time(),
    }
    (root / "metadata.json").write_text(json.dumps(meta, indent=2))
    on_log(f"registered '{job.name}' at {root}")
    return str(root)
