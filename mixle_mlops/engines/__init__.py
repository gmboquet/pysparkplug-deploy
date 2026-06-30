"""Logit-level local inference: incremental decoding with token-level Product-of-Experts fusion + grammar masking
— the serving integration the OpenAI chat API can't provide (it has no forced-token continuation / logit access)."""
from .decode import LogitProvider, decode, fuse_logprobs
from .grammar import TokenFSA
from .providers import HFLogitProvider, NgramProvider

__all__ = ["decode", "fuse_logprobs", "LogitProvider", "TokenFSA", "NgramProvider", "HFLogitProvider"]
