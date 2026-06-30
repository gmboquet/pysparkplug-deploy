"""Self-evolution: the EvolutionWorker improves a deliberately-bad champion (anti-regression gate verified),
and the /v1/evolve routes trigger it, record lineage, gate on admin, and roll back — end-to-end."""
import mixle_mlops.storage.db as db
import numpy as np
import pytest
from fastapi.testclient import TestClient
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

from mixle_mlops.config import get_settings
from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.evolve.policy import EvolutionPolicy
from mixle_mlops.evolve.worker import EvolutionWorker
from mixle_mlops.gateway.app import create_app
from mixle_mlops.models.mixle_model import MixleAdapter


def _improvable_adapter():
    rng = np.random.RandomState(0)
    data = list(rng.normal(5.0, 1.0, size=80))               # truth ~ N(5,1)
    champion = GaussianDistribution(0.0, 1.0)                 # deliberately wrong mean -> very improvable
    return champion, data


def test_worker_improves_bad_champion_and_rolls_back():
    champion, data = _improvable_adapter()
    reg = ModelRegistry()
    reg.register(MixleAdapter("g", model=champion, fit_data=data))
    worker = EvolutionWorker(reg)

    run = worker.run("g", data, EvolutionPolicy(objective="nll"))
    assert run.verified and run.promoted and run.delta > 0    # the gate confirmed a real, non-regressive win
    assert reg.get("g")._model is not champion                # the improved model is now served
    assert worker.rollback("g") and reg.get("g")._model is champion   # rollback restores the original


def test_worker_declines_non_mixle_model():
    reg = ModelRegistry()
    from mixle_mlops.models import EchoAdapter

    reg.register(EchoAdapter("echo"))
    run = EvolutionWorker(reg).run("echo", [1, 2, 3, 4], EvolutionPolicy())
    assert not run.verified and run.error and "mixle" in run.error


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    with TestClient(app) as c:
        champion, data = _improvable_adapter()
        app.state.registry.register(MixleAdapter("g", model=champion, fit_data=data))
        yield c
    get_settings.cache_clear()
    db._engine = None


def _admin_headers(c):
    from sqlmodel import Session

    from mixle_mlops.accounts import service as acct
    from mixle_mlops.storage.db import get_engine

    with Session(get_engine()) as s:
        user = acct.create_user(s, "admin@t.com", "pw12345", is_admin=True)
        _key, raw = acct.create_api_key(s, user)
    return {"Authorization": f"Bearer {raw}"}


def test_evolve_endpoint_runs_records_and_rolls_back(client):
    admin = _admin_headers(client)
    r = client.post("/v1/evolve/g", headers=admin, json={"policy": {"objective": "nll"}})
    assert r.status_code == 200
    run = r.json()
    assert run["verified"] and run["promoted"] and run["delta"] > 0 and run["verdict"] is not None
    # lineage recorded
    runs = client.get("/v1/evolve/g/runs", headers=admin).json()["data"]
    assert len(runs) == 1 and runs[0]["id"] == run["id"]
    assert client.get(f"/v1/evolve/runs/{run['id']}", headers=admin).json()["verified"]
    # rollback works, and a second rollback has nothing to restore
    assert client.post("/v1/evolve/g/rollback", headers=admin).json()["rolled_back"]
    assert client.post("/v1/evolve/g/rollback", headers=admin).status_code == 409


def test_evolve_requires_admin(client):
    raw = client.post("/auth/signup", json={"email": "u@t.com", "password": "pw12345"}).json()["api_key"]
    r = client.post("/v1/evolve/g", headers={"Authorization": f"Bearer {raw}"}, json={})
    assert r.status_code == 403


def test_evolve_unknown_model_404(client):
    admin = _admin_headers(client)
    assert client.post("/v1/evolve/nope", headers=admin, json={}).status_code == 404
