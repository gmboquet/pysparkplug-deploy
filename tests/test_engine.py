"""Logit-level decode engine: PoE fusion == mixle.ops.product_of_experts, grammar masking guarantees well-formed
output, and the real transformers backend works (a tiny in-process GPT-2, no download)."""
import asyncio

import numpy as np
import pytest

from mixle_mlops.core.adapters import ChatMessage, ChatRequest
from mixle_mlops.engines import HFLogitProvider, NgramProvider, TokenFSA, decode, fuse_logprobs
from mixle_mlops.engines.decode import _softmax
from mixle_mlops.engines.providers import CharProvider
from mixle_mlops.models.local_engine import LocalEngineAdapter


def test_fuse_logprobs_matches_mixle_product_of_experts():
    from mixle.ops import product_of_experts
    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

    rng = np.random.default_rng(0)
    l1, l2 = rng.normal(size=5), rng.normal(size=5)
    fused = _softmax(fuse_logprobs([l1, l2]))                 # token-level PoE in the engine

    p1, p2 = _softmax(l1), _softmax(l2)
    d1 = CategoricalDistribution(pmap={i: float(p1[i]) for i in range(5)})
    d2 = CategoricalDistribution(pmap={i: float(p2[i]) for i in range(5)})
    poe = product_of_experts([d1, d2]).pmap                   # the mixle-core primitive
    for i in range(5):
        assert abs(fused[i] - poe[i]) < 1e-9


def test_poe_decode_picks_consensus_token():
    # provider A prefers token 1 (3 best), token 3 second; B prefers token 2, token 3 second -> fused picks 3
    v = 4
    a = np.zeros((v, v))
    a[:, 1], a[:, 3] = 3.0, 2.0
    b = np.zeros((v, v))
    b[:, 2], b[:, 3] = 3.0, 2.0
    out = decode([NgramProvider(a), NgramProvider(b)], prompt_ids=[0], max_new_tokens=1, greedy=True)
    assert out == [3]                                         # the product-of-experts consensus


def test_grammar_mask_guarantees_well_formed_output():
    # provider always wants token 0; the grammar forces alternation between {0,1} and {2,3} for 4 tokens
    v = 4
    table = np.zeros((v, v))
    table[:, 0] = 10.0
    fsa = TokenFSA.from_token_sequence_alternation(class_a=[0, 1], class_b=[2, 3], length=4)
    out = decode(NgramProvider(table), prompt_ids=[], max_new_tokens=10, grammar=fsa, greedy=True)
    assert len(out) == 4
    assert out[0] in (0, 1) and out[1] in (2, 3) and out[2] in (0, 1) and out[3] in (2, 3)
    assert out[0] == 0 and out[2] == 0                        # provider's preference honored within the mask


def test_hf_provider_real_transformers_no_download():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=2, n_head=2))
    provider = HFLogitProvider(model=model)                   # real transformers, random weights, no download
    assert provider.vocab_size == 32
    assert provider.next_logits([1, 2, 3]).shape == (32,)

    out = decode(provider, prompt_ids=[1], max_new_tokens=5, greedy=True)
    assert len(out) == 5 and all(0 <= t < 32 for t in out)
    # PoE of a model with itself reproduces its own greedy decode (the fusion is correct)
    assert decode([provider, provider], prompt_ids=[1], max_new_tokens=5, greedy=True) == out
    # grammar masking works on the real model too
    fsa = TokenFSA.from_token_sequence_alternation(class_a=[5, 6], class_b=[7, 8], length=4)
    masked = decode(provider, prompt_ids=[1], max_new_tokens=10, grammar=fsa, greedy=True)
    assert len(masked) == 4 and masked[0] in (5, 6) and masked[1] in (7, 8)


def test_local_engine_adapter_generates():
    table = np.full((3, 3), -10.0)
    table[0, 1], table[1, 2], table[2, 0] = 10.0, 10.0, 10.0       # a->b->c->a cycle
    adapter = LocalEngineAdapter("toy", CharProvider("abc", table=table), max_new_tokens=5)
    req = ChatRequest(model="toy", messages=[ChatMessage(role="user", content="a")])
    text = asyncio.run(adapter.chat(req)).choices[0].message.text()
    assert len(text) == 5 and all(c in "abc" for c in text)
    succ = {"a": "b", "b": "c", "c": "a"}
    assert all(text[i + 1] == succ[text[i]] for i in range(len(text) - 1))


def test_local_engine_poe_ensemble_picks_consensus():
    ta = np.full((3, 3), -10.0)
    ta[:, 0], ta[:, 1] = 5.0, 4.0                                  # model A prefers a, then b
    tb = np.full((3, 3), -10.0)
    tb[:, 2], tb[:, 1] = 5.0, 4.0                                  # model B prefers c, then b
    adapter = LocalEngineAdapter("poe", [CharProvider("abc", table=ta), CharProvider("abc", table=tb)],
                                 max_new_tokens=4)
    req = ChatRequest(model="poe", messages=[ChatMessage(role="user", content="a")])
    text = asyncio.run(adapter.chat(req)).choices[0].message.text()
    assert len(text) == 4 and set(text) == {"b"}                   # the PoE consensus token dominates every step
