"""Token-level Product-of-Experts fusion (exact, via mixle.ops.product_of_experts) + the sequence reranker."""
import asyncio
import math

from mixle_mlops.gateway.poe import fuse_next_token, poe_rerank


def test_fuse_next_token_exact_product():
    # two models' top-token logprobs over a shared vocab {a, b, c}
    m1 = {"a": math.log(0.5), "b": math.log(0.3), "c": math.log(0.2)}
    m2 = {"a": math.log(0.1), "b": math.log(0.6), "c": math.log(0.3)}
    fused = fuse_next_token([m1, m2])
    # PoE ∝ p1*p2: a=.05, b=.18, c=.06 -> normalize by .29
    z = 0.05 + 0.18 + 0.06
    assert abs(fused["a"] - 0.05 / z) < 1e-9
    assert abs(fused["b"] - 0.18 / z) < 1e-9
    assert abs(fused["c"] - 0.06 / z) < 1e-9
    assert abs(sum(fused.values()) - 1.0) < 1e-9
    # the fused mode is b (both models' agreement sharpens it)
    assert max(fused, key=fused.get) == "b"


def test_fuse_next_token_weighted():
    m1 = {"x": math.log(0.8), "y": math.log(0.2)}
    m2 = {"x": math.log(0.2), "y": math.log(0.8)}
    # weight model 1 heavily -> fused leans toward x
    fused = fuse_next_token([m1, m2], weights=[3.0, 1.0])
    assert fused["x"] > fused["y"]


def test_poe_rerank_picks_highest_joint_logprob():
    # candidate logprobs under two models; cand 1 has the best weighted sum
    table = {
        ("m1", "cand0"): -2.0, ("m2", "cand0"): -3.0,
        ("m1", "cand1"): -1.0, ("m2", "cand1"): -0.5,
        ("m1", "cand2"): -0.5, ("m2", "cand2"): -5.0,
    }

    async def logprob_fn(model, candidate):
        return table[(model, candidate)]

    best, info = asyncio.run(poe_rerank(["cand0", "cand1", "cand2"], logprob_fn, ["m1", "m2"]))
    assert best == "cand1" and info["best_index"] == 1
