"""Offline tests for the vast.ai training launcher (no provisioning, no spend)."""
import pytest

from mixle_mlops.compute import TrainingJob, plan
from mixle_mlops.compute.jobspec import build_onstart, pip_packages, training_command
from mixle_mlops.compute.launcher import launch
from mixle_mlops.compute.vast import VastClient


def test_offer_query_shape():
    q = VastClient.build_query(gpu_name="RTX_4090", num_gpus=2, max_price=1.2, limit=10)
    assert q["gpu_name"] == {"in": ["RTX 4090"]}  # underscore -> space
    assert q["num_gpus"] == {"gte": 2}
    assert q["dph_total"] == {"lte": 1.2}
    assert q["rentable"] == {"eq": True}


def test_validate():
    with pytest.raises(ValueError):
        TrainingJob(name="x", backend="llm").validate()  # llm needs base_model
    with pytest.raises(ValueError):
        TrainingJob(name="x", backend="mixle").validate()  # mixle needs a script
    # valid ones
    TrainingJob(name="x", backend="llm", base_model="Qwen/Qwen2.5-0.5B", mode="onstart").validate()
    TrainingJob(name="x", backend="mixle", script="train.py", workdir=".").validate()


def test_training_command_mixle_and_llm():
    m = training_command(TrainingJob(name="m", backend="mixle", script="t.py", workdir=".", dataset="d.jsonl"))
    assert m.startswith("python") and "t.py" in m and "--dataset" in m and "--output" in m
    llm = training_command(
        TrainingJob(name="l", backend="llm", base_model="Qwen/Qwen2.5-0.5B", dataset="d.jsonl", qlora=True)
    )
    assert "llm_lora_train.py" in llm and "--base_model" in llm and "--qlora" in llm


def test_pip_packages():
    mixle_pkgs = pip_packages(TrainingJob(name="m", backend="mixle", script="t.py", workdir="."))
    assert any("mixle" in p for p in mixle_pkgs)  # installed from git by default
    assert ["pomegranate"] == pip_packages(
        TrainingJob(name="m", backend="mixle", script="t.py", workdir=".", mixle_spec="pomegranate")
    )
    llm = pip_packages(TrainingJob(name="l", backend="llm", base_model="x"))
    assert any("peft" in p for p in llm) and any("transformers" in p for p in llm)


def test_onstart_script():
    # mixle onstart clones the repo and runs the script
    s = build_onstart(TrainingJob(name="m", backend="mixle", mode="onstart", repo="https://git/x.git", script="t.py"))
    assert "git clone" in s and "https://git/x.git" in s and "python" in s and "touch /workspace/DONE" in s
    # llm onstart writes the LoRA script + can upload to s3
    s2 = build_onstart(
        TrainingJob(name="l", backend="llm", mode="onstart", base_model="Qwen/Qwen2.5-0.5B"),
        s3_dest="s3://bucket/runs/l/",
    )
    assert "llm_lora_train.py" in s2 and "aws s3 cp" in s2 and "s3://bucket/runs/l/" in s2


def test_plan_and_dry_run():
    job = TrainingJob(name="demo", backend="llm", base_model="Qwen/Qwen2.5-0.5B", dataset="d.jsonl", gpu="RTX_4090")
    p = plan(job)
    assert p["image"] and p["offer_query"]["gpu_name"] == {"in": ["RTX 4090"]}
    assert "llm_lora_train.py" in p["training_command"]

    logs: list[str] = []
    out = launch(job, dry_run=True, on_log=logs.append)
    assert out["dry_run"] is True
    joined = "\n".join(logs)
    assert "DRY RUN" in joined and "offer search" in joined and "train command" in joined
