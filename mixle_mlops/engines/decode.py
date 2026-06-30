"""Logit-level incremental decoding — the serving integration the OpenAI chat API can't provide.

This is where the bridge levers that need per-step logit access become real instead of deferred:

  * **True token-level Product-of-Experts**: at every step, fuse several models' next-token log-probabilities,
    ``log p(t) = Σ_k w_k·log p_k(t) − log Z`` — the exact per-token form of ``mixle.ops.product_of_experts``,
    then sample from the fused distribution and feed the token back. (The chat API exposes no forced-token
    continuation, so this only works with logit access — which a local engine has.)
  * **Grammar-masked sampling**: restrict the logits at each step to the tokens a grammar/FSA permits, so the
    output is *guaranteed* well-formed — masking inside the decode loop, not validate-and-retry after the fact.

It runs against any :class:`LogitProvider` (a model exposing next-token logits): a real transformers model, or a
toy n-gram provider for tests. No assumption about which."""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class LogitProvider(Protocol):
    vocab_size: int

    def next_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        """Next-token logits ``(vocab_size,)`` given the tokens so far."""
        ...


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    z = m + np.log(np.exp(x - m).sum())
    return x - z


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def fuse_logprobs(logits_list: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Token-level Product-of-Experts in log space: ``Σ_k w_k·log softmax(logits_k)``, renormalized to a proper
    log-distribution. Equivalent to ``mixle.ops.product_of_experts`` over the per-step categoricals."""
    weights = weights or [1.0] * len(logits_list)
    fused = None
    for w, logits in zip(weights, logits_list):
        lp = _log_softmax(logits)
        fused = w * lp if fused is None else fused + w * lp
    return fused - (fused.max() + np.log(np.exp(fused - fused.max()).sum()))


def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return probs
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    keep = cum <= top_p
    keep[0] = True                                            # always keep the top token
    out = np.zeros_like(probs)
    out[order[keep]] = probs[order[keep]]
    s = out.sum()
    return out / s if s > 0 else probs


def decode(providers: LogitProvider | list[LogitProvider], *, prompt_ids: Sequence[int] = (),
           max_new_tokens: int = 32, eos_id: int | None = None, weights: list[float] | None = None,
           grammar: Any = None, greedy: bool = True, temperature: float = 1.0, top_p: float = 1.0,
           seed: int = 0) -> list[int]:
    """Decode token-by-token with optional Product-of-Experts fusion (multiple providers) and grammar masking.

    Args:
        providers: one provider, or several to fuse via token-level PoE.
        grammar: optional object with ``start``, ``allowed(state) -> ids``, ``advance(state, token) -> state``,
            ``is_accepting(state) -> bool``; masks each step to grammar-allowed tokens.
    Returns the generated token ids (excluding the prompt).
    """
    provs = list(providers) if isinstance(providers, (list, tuple)) else [providers]
    rng = np.random.default_rng(seed)
    tokens = list(prompt_ids)
    state = getattr(grammar, "start", None) if grammar is not None else None
    out: list[int] = []

    for _ in range(max_new_tokens):
        fused = fuse_logprobs([p.next_logits(tokens) for p in provs], weights)

        if grammar is not None:                              # restrict to grammar-allowed tokens
            allowed = list(grammar.allowed(state))
            if not allowed:
                break                                        # dead end (no valid continuation)
            mask = np.full(fused.shape, -np.inf)
            mask[allowed] = 0.0
            fused = fused + mask

        if greedy:
            nxt = int(np.argmax(fused))
        else:
            probs = _softmax(fused / max(temperature, 1e-6))
            probs = _top_p_filter(probs, top_p)
            nxt = int(rng.choice(len(probs), p=probs))

        out.append(nxt)
        tokens.append(nxt)
        if grammar is not None:
            state = grammar.advance(state, nxt)
        if eos_id is not None and nxt == eos_id:
            break
        if grammar is not None and grammar.is_accepting(state) and not list(grammar.allowed(state)):
            break                                            # grammar reached a terminal accepting state
    return out
