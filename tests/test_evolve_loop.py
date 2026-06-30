"""The closed self-improvement loop: usage signals -> router self-calibration, and the autonomous scheduler tick."""
import mixle_mlops.storage.db as db
import numpy as np
import pytest
from fastapi.testclient import TestClient
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from sqlmodel import Session

from mixle_mlops.config import get_settings
from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.evolve.policy import EvolutionPolicy
from mixle_mlops.evolve.scheduler import EvolutionScheduler
from mixle_mlops.evolve.signals import recommend_threshold, record_signal, router_stats
from mixle_mlops.gateway.app import create_app
from mixle_mlops.models.mixle_model import MixleAdapter


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    from mixle_mlops.storage.db import get_engine, init_db

    init_db()
    with Session(get_engine()) as s:
        yield s
    get_settings.cache_clear()
    db._engine = None


def test_router_self_calibration(session):
    for conf, escalated in [(0.2, True), (0.3, True), (0.5, False), (0.6, False),
                            (0.9, False), (0.95, False), (0.1, True), (0.4, True)]:
        record_signal(session, "local", kind="cascade", confidence=conf, escalated=escalated)
    stats = router_stats(session, "local")
    assert stats["n"] == 8 and 0.0 < stats["escalation_rate"] < 1.0 and stats["mean_confidence"] is not None
    rec = recommend_threshold(session, "local", target_escalation_rate=0.25)
    assert rec["recommended_threshold"] is not None and 0.1 <= rec["recommended_threshold"] <= 0.95


def test_scheduler_tick_improves_mixle_models(session):
    rng = np.random.RandomState(0)
    data = list(rng.normal(5.0, 1.0, size=80))
    reg = ModelRegistry()
    reg.register(MixleAdapter("g", model=GaussianDistribution(0.0, 1.0), fit_data=data))
    runs = EvolutionScheduler(reg).tick(session, EvolutionPolicy(objective="nll"))
    assert len(runs) == 1 and runs[0].model_id == "g" and runs[0].verified


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    with TestClient(app) as c:
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 1.0, size=80))
        app.state.registry.register(MixleAdapter("g", model=GaussianDistribution(0.0, 1.0), fit_data=data))
        yield c
    get_settings.cache_clear()
    db._engine = None


def _admin_headers(c):
    from mixle_mlops.accounts import service as acct
    from mixle_mlops.storage.db import get_engine

    with Session(get_engine()) as s:
        user = acct.create_user(s, "admin@t.com", "pw12345", is_admin=True)
        _key, raw = acct.create_api_key(s, user)
    return {"Authorization": f"Bearer {raw}"}


def test_autonomous_tick_and_signals_over_http(client):
    admin = _admin_headers(client)
    runs = client.post("/v1/evolve/tick", headers=admin).json()["data"]
    assert any(r["model_id"] == "g" and r["verified"] for r in runs)        # autonomous pass improved the model
    sig = client.get("/v1/evolve/g/signals", headers=admin).json()
    assert "recommended_threshold" in sig and sig["model_id"] == "g"


def test_background_evolution_loop_runs_a_pass(tmp_path, monkeypatch):
    """The lifespan timer loop autonomously improves models with no human trigger."""
    import asyncio
    from types import SimpleNamespace

    from sqlmodel import select

    from mixle_mlops.evolve.models import EvolutionRecord
    from mixle_mlops.gateway.app import _evolution_loop
    from mixle_mlops.storage.db import get_engine, init_db

    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    init_db()
    rng = np.random.RandomState(0)
    reg = ModelRegistry()
    reg.register(MixleAdapter("g", model=GaussianDistribution(0.0, 1.0), fit_data=list(rng.normal(5.0, 1.0, size=80))))
    app = SimpleNamespace(state=SimpleNamespace(registry=reg))

    async def drive():
        task = asyncio.create_task(_evolution_loop(app, 0))    # interval 0 -> ticks immediately, repeatedly
        await asyncio.sleep(0.25)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    with Session(get_engine()) as s:
        rows = list(s.exec(select(EvolutionRecord)).all())
    assert any(r.model_id == "g" and r.verified for r in rows)
    get_settings.cache_clear()
    db._engine = None
