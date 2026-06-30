"""Best-of-N with real verifiers (beyond self-consistency): exact-match, computed-reference, LLM-judge."""
import asyncio

from mixle_mlops.core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    ModelAdapter,
)
import numpy as np

from mixle_mlops.gateway.verifiers import (
    best_of_n_verified,
    build_verifier,
    exact_match_verifier,
    llm_judge_verifier,
    numeric_verifier,
)


class SeqAdapter(ModelAdapter):
    kind = "llm"

    def __init__(self, name, texts):
        self._name = name
        self._texts = list(texts)
        self._i = 0

    @property
    def name(self):
        return self._name

    async def chat(self, req):
        text = self._texts[self._i % len(self._texts)]
        self._i += 1
        return ChatCompletion(model=req.model, choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=text), finish_reason="stop")])

    async def stream(self, req):
        completion = await self.chat(req)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(content=completion.choices[0].message.text()), finish_reason="stop")])


def test_exact_match_verifier():
    v = exact_match_verifier("the answer is 42")
    assert asyncio.run(v("clearly 42")) == 1.0
    assert asyncio.run(v("it is 7")) == 0.0


def test_numeric_verifier_uses_exact_solver():
    v = numeric_verifier({"op": "eval", "expr": "6*7"})
    assert asyncio.run(v("I computed 42")) == 1.0
    assert asyncio.run(v("about 41")) == 0.0


def test_llm_judge_verifier_parses_rating():
    judge = SeqAdapter("judge", ["I rate this 8 out of 10"])
    v = llm_judge_verifier(judge)
    assert asyncio.run(v("some answer")) == 8.0


class _ToyEmbedder:
    """coord 0 = (count 'good' − count 'bad'); deterministic, so the learned reward is testable."""

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            q = float(t.count("good") - t.count("bad"))
            v = np.array([q, 1.0])
            out.append(v / np.linalg.norm(v))
        return np.array(out)

    def embed_one(self, text):
        return self.embed([text])[0]


def test_feature_reward_verifier_selects_preferred():
    from mixle_mlops.feedback.feature_reward import FeatureRewardModel

    reward = FeatureRewardModel(embedder=_ToyEmbedder(), l2=1e-3).fit([("good good", "bad bad"), ("good", "bad")])

    async def verifier(text):
        return float(reward.score(text))

    adapter = SeqAdapter("gen", ["bad answer", "good good answer", "neutral"])
    req = ChatRequest(model="gen", messages=[ChatMessage(role="user", content="?")])
    completion, _info = asyncio.run(best_of_n_verified(adapter, req, n=3, verifier=verifier))
    assert "good" in completion.choices[0].message.text()    # the learned reward picks the preferred candidate


def test_build_feature_reward_verifier_from_spec():
    from mixle_mlops.core.registry import ModelRegistry

    spec = {"type": "feature_reward", "pairs": [["great work", "awful work"], ["good", "bad"]]}
    verifier = build_verifier(spec, ModelRegistry())
    assert verifier is not None
    assert isinstance(asyncio.run(verifier("great work")), float)


def test_best_of_n_verified_selects_correct():
    # three candidates; only the middle one matches the reference -> verifier must pick it
    adapter = SeqAdapter("gen", ["answer: 7", "answer: 42", "answer: 13"])
    req = ChatRequest(model="gen", messages=[ChatMessage(role="user", content="6*7?")])
    completion, info = asyncio.run(
        best_of_n_verified(adapter, req, n=3, verifier=numeric_verifier({"op": "eval", "expr": "6*7"})))
    assert "42" in completion.choices[0].message.text()
    assert info["best_score"] == 1.0 and info["scores"].count(1.0) == 1
