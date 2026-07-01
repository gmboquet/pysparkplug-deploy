"""A narrated, runnable tour of the mixle-mlops platform — every headline capability, end to end, with no
external models, GPUs, or network. It builds the gateway in-process (``TestClient``), registers a few toy
adapters, and walks the bridge stack + self-evolution, printing what each step demonstrates.

    python examples/platform_tour.py

It doubles as a smoke test: if the tour prints "TOUR COMPLETE" the whole stack wired together correctly."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root (not pip-installed)

import numpy as np
from fastapi.testclient import TestClient

from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle_mlops.config import get_settings
from mixle_mlops.core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChoiceDelta,
    ModelAdapter,
)
from mixle_mlops.models.mixle_model import MixleAdapter


class _Seq(ModelAdapter):
    """A deterministic stand-in for a stochastic sampler: cycles through fixed replies."""
    kind = "llm"

    def __init__(self, name, texts):
        self._name, self._texts, self._i = name, list(texts), 0

    @property
    def name(self):
        return self._name

    async def chat(self, req):
        text = self._texts[self._i % len(self._texts)]
        self._i += 1
        return ChatCompletion(model=req.model, choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=text), finish_reason="stop")])

    async def stream(self, req):
        c = await self.chat(req)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(content=c.choices[0].message.text()), finish_reason="stop")])


class _Reflect(ModelAdapter):
    """Returns the last user message it received — an aggregator that provably sees the proposals."""
    kind = "llm"

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    async def chat(self, req):
        last = req.messages[-1].text() if req.messages else ""
        return ChatCompletion(model=req.model, choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=f"synthesis of: {last}"), finish_reason="stop")])

    async def stream(self, req):
        c = await self.chat(req)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(content=c.choices[0].message.text()), finish_reason="stop")])


def _step(title):
    print(f"\n--- {title} ---")


def main() -> None:
    import mixle_mlops.storage.db as db

    with tempfile.TemporaryDirectory() as tmp:
        import os

        os.environ["MIXLE_DATA_DIR"] = tmp
        get_settings.cache_clear()
        db._engine = None
        from mixle_mlops.gateway.app import create_app

        app = create_app()
        with TestClient(app) as c:
            reg = app.state.registry
            reg.register(_Seq("voter", ["answer: 42", "answer: 42", "answer: 7"]))          # best-of-N
            reg.register(_Seq("local", ["answer: 1", "answer: 2", "answer: 3", "answer: 4"]))  # low self-consistency
            reg.register(_Seq("frontier", ["answer: 42"]))                                   # cascade target
            for n, t in [("p1", "cats"), ("p2", "dogs"), ("p3", "birds")]:
                reg.register(_Seq(n, [t]))                                                   # MoA proposers
            reg.register(_Reflect("agg"))                                                    # MoA aggregator
            rng = np.random.RandomState(0)
            reg.register(MixleAdapter("gauss", model=GaussianDistribution(0.0, 1.0),
                                      fit_data=list(rng.normal(5.0, 1.0, 80))))              # a self-evolvable model

            raw = c.post("/auth/signup", json={"email": "tour@x.com", "password": "pw12345"}).json()["api_key"]
            h = {"Authorization": f"Bearer {raw}"}

            def chat(model, content, **extra):
                body = {"model": model, "messages": [{"role": "user", "content": content}]}
                if extra:
                    body["extra"] = extra
                return c.post("/v1/chat/completions", headers=h, json=body)

            _step("1. OpenAI-compatible chat")
            print("models:", [m["id"] for m in c.get("/v1/models", headers=h).json()["data"]][:6], "...")

            _step("2. Best-of-N self-consistency (test-time compute)")
            r = chat("voter", "2*21?", best_of_n=3)
            print("answer:", r.json()["choices"][0]["message"]["content"],
                  "| confidence:", r.headers.get("X-Self-Consistency"))

            _step("3. Cascade router (answer locally, else escalate)")
            r = chat("local", "hard", cascade={"frontier": "frontier", "threshold": 0.6, "n": 4})
            print("escalated:", r.headers.get("X-Cascade-Escalated"),
                  "| answer:", r.json()["choices"][0]["message"]["content"])

            _step("4. Mixture-of-Agents (several propose, one synthesizes)")
            r = chat("agg", "name an animal", moa={"proposers": ["p1", "p2", "p3"], "aggregator": "agg"})
            print("proposals reached aggregator:",
                  all(a in r.json()["choices"][0]["message"]["content"] for a in ("cats", "dogs", "birds")))

            _step("5. mixle probabilistic surface (calibrated prediction + Bayes decision)")
            pred = c.post("/v1/mixle/predict", headers=h, json={"model": "gauss", "records": [None]}).json()
            print("predictive:", {k: pred.get(k) for k in ("path", "density_semantics")})

            _step("6. Self-evolution (verify-gated, anti-regression) — admin only")
            from mixle_mlops.accounts import service as acct
            from mixle_mlops.storage.db import get_engine
            from sqlmodel import Session
            with Session(get_engine()) as s:
                admin = acct.create_user(s, "admin@x.com", "pw12345", is_admin=True)
                _k, araw = acct.create_api_key(s, admin)
            ah = {"Authorization": f"Bearer {araw}"}
            run = c.post("/v1/evolve/gauss", headers=ah, json={"policy": {"objective": "nll"}}).json()
            print("evolved: verified =", run["verified"], "| promoted =", run["promoted"],
                  "| operator =", run["operator"], "| delta =", round(run["delta"], 3))

            print("\nTOUR COMPLETE — the full bridge stack + self-evolution wired together end to end.")


if __name__ == "__main__":
    main()
