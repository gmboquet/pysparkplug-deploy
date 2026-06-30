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


def decode_iter(providers: LogitProvider | list[LogitProvider], *, prompt_ids: Sequence[int] = (),
                max_new_tokens: int = 32, eos_id: int | None = None, weights: list[float] | None = None,
                grammar: Any = None, greedy: bool = True, temperature: float = 1.0, top_p: float = 1.0,
                seed: int = 0):
    """Stream token-by-token with optional Product-of-Experts fusion (multiple providers) and grammar masking.
    Yields each generated token id as it is produced — the basis for true streaming generation."""
    provs = list(providers) if isinstance(providers, (list, tuple)) else [providers]
    rng = np.random.default_rng(seed)
    tokens = list(prompt_ids)
    state = getattr(grammar, "start", None) if grammar is not None else None

    for _ in range(max_new_tokens):
        fused = fuse_logprobs([p.next_logits(tokens) for p in provs], weights)

        if grammar is not None:                              # restrict to grammar-allowed tokens
            allowed = list(grammar.allowed(state))
            if not allowed:
                return                                       # dead end (no valid continuation)
            mask = np.full(fused.shape, -np.inf)
            mask[allowed] = 0.0
            fused = fused + mask

        if greedy:
            nxt = int(np.argmax(fused))
        else:
            probs = _softmax(fused / max(temperature, 1e-6))
            probs = _top_p_filter(probs, top_p)
            nxt = int(rng.choice(len(probs), p=probs))

        yield nxt
        tokens.append(nxt)
        if grammar is not None:
            state = grammar.advance(state, nxt)
        if eos_id is not None and nxt == eos_id:
            return
        if grammar is not None and grammar.is_accepting(state) and not list(grammar.allowed(state)):
            return                                           # grammar reached a terminal accepting state


def decode(providers: LogitProvider | list[LogitProvider], **kwargs) -> list[int]:
    """Collect :func:`decode_iter` into a list of generated token ids (token-level PoE + grammar masking)."""
    return list(decode_iter(providers, **kwargs))


def _seq_logits(provider: LogitProvider, token_ids: Sequence[int]) -> np.ndarray:
    """All-position logits for a sequence in one shot when the provider supports it (``seq_logits``), else by
    looping ``next_logits`` (correct, slower). Row ``i`` predicts the token after ``token_ids[:i+1]``."""
    fn = getattr(provider, "seq_logits", None)
    if callable(fn):
        return np.asarray(fn(token_ids), dtype=np.float64)
    ids = list(token_ids)
    return np.asarray([provider.next_logits(ids[:i + 1]) for i in range(len(ids))], dtype=np.float64)


def speculative_decode(draft: LogitProvider, target: LogitProvider, *, prompt_ids: Sequence[int] = (),
                       max_new_tokens: int = 32, k: int = 4, eos_id: int | None = None, greedy: bool = True,
                       temperature: float = 1.0, seed: int = 0) -> list[int]:
    """Speculative sampling (Leviathan/Chen 2023): the cheap ``draft`` proposes ``k`` tokens, the ``target``
    verifies them in ONE forward pass; accepted tokens are kept, the first rejection is corrected by resampling.
    The output distribution is identical to sampling from ``target`` alone — a lossless speedup. Returns tokens."""
    rng = np.random.default_rng(seed)
    tokens = list(prompt_ids)
    out: list[int] = []

    def pick(logits: np.ndarray) -> int:
        if greedy:
            return int(np.argmax(logits))
        probs = _softmax(np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6))
        return int(rng.choice(len(probs), p=probs))

    while len(out) < max_new_tokens:
        n = len(tokens)
        # draft proposes k tokens autoregressively
        draft_toks: list[int] = []
        draft_logits: list[np.ndarray] = []
        cur = tokens[:]
        for _ in range(k):
            dl = np.asarray(draft.next_logits(cur), dtype=np.float64)
            draft_logits.append(dl)
            t = pick(dl)
            draft_toks.append(t)
            cur.append(t)

        seq = _seq_logits(target, tokens + draft_toks)        # one target pass over the whole drafted sequence
        accepted: list[int] = []
        rejected = False
        for j in range(k):
            tl = seq[n - 1 + j]
            if greedy:
                tgt = int(np.argmax(tl))
                accepted.append(tgt)
                if draft_toks[j] != tgt:
                    rejected = True
                    break
            else:
                p = _softmax(tl / max(temperature, 1e-6))
                q = _softmax(draft_logits[j] / max(temperature, 1e-6))
                t = draft_toks[j]
                if rng.random() < min(1.0, p[t] / max(q[t], 1e-12)):
                    accepted.append(t)
                else:
                    resid = np.maximum(p - q, 0.0)
                    s = resid.sum()
                    accepted.append(int(rng.choice(len(p), p=resid / s)) if s > 0 else int(np.argmax(p)))
                    rejected = True
                    break
        if not rejected:                                      # all k accepted -> a free bonus token from the target
            accepted.append(pick(seq[n - 1 + k]))

        for t in accepted:
            out.append(t)
            tokens.append(t)
            if eos_id is not None and t == eos_id:
                return out[:max_new_tokens]
    return out[:max_new_tokens]
