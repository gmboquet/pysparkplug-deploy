"""Logit providers for the decode engine: a deterministic toy n-gram (tests) and a transformers backend (real)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


class NgramProvider:
    """Deterministic toy LM: next-token logits are a fixed function of the last token (a bigram logit table).

    Useful for fast, exact tests of the decode loop / PoE fusion / grammar masking without loading a real model."""

    def __init__(self, logit_table: np.ndarray, *, initial: np.ndarray | None = None):
        self.table = np.asarray(logit_table, dtype=np.float64)        # (vocab, vocab): row = last token
        self.vocab_size = int(self.table.shape[0])
        self.initial = (np.asarray(initial, dtype=np.float64) if initial is not None
                        else np.zeros(self.vocab_size, dtype=np.float64))

    def next_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        if len(token_ids) == 0:
            return self.initial.copy()
        return self.table[int(token_ids[-1])].copy()


class HFLogitProvider:
    """A real transformers ``AutoModelForCausalLM`` exposing per-step next-token logits — the genuine logit-level
    backend that makes token-level PoE + grammar masking work with actual models."""

    def __init__(self, model=None, tokenizer=None, *, model_name: str | None = None, device: str = "cpu"):
        import torch

        self._torch = torch
        if model is None:
            if model_name is None:
                raise ValueError("HFLogitProvider needs a model= or a model_name=")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = model
            self.tokenizer = tokenizer
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.vocab_size = int(self.model.config.vocab_size)
        self._bos = getattr(self.model.config, "bos_token_id", None)
        self.eos = getattr(self.model.config, "eos_token_id", None)

    def next_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        torch = self._torch
        ids = list(token_ids) or [self._bos if self._bos is not None else 0]
        with torch.no_grad():
            tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(tensor).logits[0, -1]
        return logits.float().cpu().numpy()

    def seq_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        """All-position next-token logits ``(len, vocab)`` in ONE forward pass — what makes speculative
        verification a speedup (the target checks k drafted tokens in a single call)."""
        torch = self._torch
        ids = list(token_ids) or [self._bos if self._bos is not None else 0]
        with torch.no_grad():
            tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(tensor).logits[0]
        return logits.float().cpu().numpy()

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text)) if self.tokenizer is not None else []

    def decode_text(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids)) if self.tokenizer is not None else ""

    def vocab(self) -> dict[int, str]:
        if getattr(self, "_vocab", None) is None:
            if self.tokenizer is None:
                self._vocab = {}
            else:
                self._vocab = {i: self.tokenizer.decode([i]) for i in range(self.vocab_size)}
        return self._vocab


class CharProvider:
    """Toy char-level LM over a fixed alphabet: deterministic bigram logits + char encode/decode. Lets the
    LocalEngineAdapter be exercised end-to-end without a model/tokenizer download."""

    def __init__(self, alphabet: str, *, table: np.ndarray | None = None, eos: int | None = None):
        self.alphabet = alphabet
        self.vocab_size = len(alphabet)
        self._c2i = {c: i for i, c in enumerate(alphabet)}
        self.table = (np.asarray(table, dtype=np.float64) if table is not None
                      else np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64))
        self.eos = eos

    def next_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        if len(token_ids) == 0:
            return np.zeros(self.vocab_size, dtype=np.float64)
        return self.table[int(token_ids[-1])].copy()

    def encode(self, text: str) -> list[int]:
        return [self._c2i[c] for c in text if c in self._c2i]

    def decode_text(self, token_ids: Sequence[int]) -> str:
        return "".join(self.alphabet[i] for i in token_ids if 0 <= i < self.vocab_size)

    def vocab(self) -> dict[int, str]:
        return {i: c for i, c in enumerate(self.alphabet)}
