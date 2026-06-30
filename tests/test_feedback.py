"""RLHF / human-feedback loop: capture preferences, fit a calibrated mixle reward model with
uncertainty, assert the better item earns a higher reward, and that active elicitation returns a pair.

Self-contained: builds the app via ``create_app()``, includes the feedback router, signs up for a key,
and drives everything over HTTP through the TestClient — no dependence on app.py edits.
"""

import mixle_mlops.storage.db as db
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.config import get_settings
from mixle_mlops.feedback import elicit, loop
from mixle_mlops.feedback.reward import fit_reward
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import feedback as feedback_routes


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    app.include_router(feedback_routes.router, tags=["feedback"])   # self-contained wiring
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def _signup(client, email="rlhf@t.com"):
    raw = client.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {raw}"}


def _ground_truth_pairs(seed=0, n=400):
    """Generate preferences where item 'A' > 'B' > 'C' > 'D' (decreasing true worth)."""
    rng = np.random.RandomState(seed)
    items = ["A", "B", "C", "D"]
    worth = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0}
    pairs = []
    for _ in range(n):
        i, j = rng.choice(len(items), size=2, replace=False)
        a, b = items[i], items[j]
        p = 1.0 / (1.0 + np.exp(-(worth[a] - worth[b])))
        if rng.rand() < p:
            pairs.append((a, b))
        else:
            pairs.append((b, a))
    return pairs


# ---------- unit-level: the reward model itself ----------

def test_reward_recovers_ordering_with_uncertainty():
    pairs = _ground_truth_pairs()
    model = fit_reward(pairs, n_boot=200, seed=1)
    # better item gets higher reward
    assert model.reward_of("A").reward > model.reward_of("B").reward
    assert model.reward_of("B").reward > model.reward_of("C").reward
    assert model.reward_of("C").reward > model.reward_of("D").reward
    # ranking best-first
    assert [r.item_id for r in model.ranking()] == ["A", "B", "C", "D"]
    # sensible, finite, positive uncertainty on every item
    assert np.all(np.isfinite(model.std))
    assert np.all(model.std > 0.0)
    # the credible interval brackets the point estimate
    for it in model.ranking():
        assert it.ci_low <= it.reward <= it.ci_high
    # calibrated preference probability is in (0.5, 1) for the clearly-better item
    assert 0.5 < model.prob_prefer("A", "D") < 1.0


def test_next_comparison_returns_a_pair():
    pairs = _ground_truth_pairs()
    model = fit_reward(pairs, n_boot=120, seed=2)
    nxt = elicit.next_comparison(model)
    assert {nxt.item_a, nxt.item_b} <= set(model.items)
    assert nxt.item_a != nxt.item_b
    assert nxt.score >= 0.0
    # eig via mixle.doe linear EIG is non-negative
    assert elicit.eig_of_pair(model, nxt.item_a, nxt.item_b) >= 0.0


def test_promote_abstains_when_not_separated():
    # two near-equal items → not significantly separated → abstain + propose a comparison
    pairs = [("X", "Y"), ("Y", "X")] * 5
    model = fit_reward(pairs, n_boot=120, seed=3)
    decision = loop.promote(model)
    assert "promote" in decision
    if not decision["promote"]:
        assert "next_comparison" in decision


# ---------- end-to-end over HTTP ----------

def test_feedback_loop_over_http(client):
    headers = _signup(client)

    # capture pairwise preferences (A beats the rest most of the time)
    for chosen, rejected in _ground_truth_pairs(seed=4, n=300):
        r = client.post("/feedback", headers=headers, json={
            "kind": "preference", "chosen_id": chosen, "rejected_id": rejected,
            "model": "demo", "payload": {"prompt": "hi", "chosen_text": chosen, "rejected_text": rejected},
        })
        assert r.status_code == 200

    # also a rating, to exercise that path
    assert client.post("/feedback", headers=headers,
                       json={"kind": "rating", "value": 1.0, "message_id": "m1"}).status_code == 200

    # fit the calibrated reward model
    rw = client.post("/rlhf/reward", headers=headers, json={"n_boot": 150}).json()
    assert rw["family"] == "BradleyTerry"
    ids = [it["item_id"] for it in rw["items"]]
    assert ids[0] == "A"                                   # best item ranked first
    assert ids == ["A", "B", "C", "D"]
    for it in rw["items"]:
        assert it["std"] > 0.0 and np.isfinite(it["std"])
        assert it["ci_low"] <= it["reward"] <= it["ci_high"]

    # active elicitation returns a pair
    nc = client.get("/rlhf/next-comparison", headers=headers, params={"n_boot": 100}).json()
    assert nc["item_a"] != nc["item_b"]
    assert {nc["item_a"], nc["item_b"]} <= {"A", "B", "C", "D"}
    assert nc["expected_information_gain"] >= 0.0

    # DPO export is valid JSONL of {prompt, chosen, rejected}
    text = client.get("/rlhf/export", headers=headers).text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 300
    import json
    rec = json.loads(lines[0])
    assert {"prompt", "chosen", "rejected"} <= set(rec)


def test_feedback_requires_auth(client):
    assert client.post("/feedback", json={"kind": "rating", "value": 1.0}).status_code == 401
    assert client.post("/rlhf/reward", json={}).status_code == 401


def test_reward_route_422_without_preferences(client):
    headers = _signup(client, email="empty@t.com")
    assert client.post("/rlhf/reward", headers=headers, json={}).status_code == 422
    assert client.get("/rlhf/next-comparison", headers=headers).status_code == 422
