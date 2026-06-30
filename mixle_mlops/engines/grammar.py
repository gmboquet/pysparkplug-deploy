"""Token-level grammar as a finite-state automaton — the constraint a logit mask enforces during decoding.

A ``TokenFSA`` over the vocabulary: from each state, only the tokens with an outgoing transition are permitted, so
masking the logits to ``allowed(state)`` at every step makes ungrammatical output *impossible* (not merely
repaired afterward). This is the in-decode-loop counterpart to the after-the-fact JSON-schema validation in
``gateway/constrained.py``."""
from __future__ import annotations

from typing import Hashable, Iterable


class TokenFSA:
    def __init__(self, transitions: dict[tuple[Hashable, int], Hashable], start: Hashable,
                 accepting: Iterable[Hashable]):
        """``transitions``: ``{(state, token_id): next_state}``; ``start``: initial state; ``accepting``: terminal
        states where generation may stop."""
        self.transitions = dict(transitions)
        self.start = start
        self.accepting = set(accepting)
        self._allowed: dict[Hashable, list[int]] = {}
        for (state, token), _ in self.transitions.items():
            self._allowed.setdefault(state, []).append(token)

    def allowed(self, state: Hashable) -> list[int]:
        return self._allowed.get(state, [])

    def advance(self, state: Hashable, token: int) -> Hashable | None:
        return self.transitions.get((state, token))

    def is_accepting(self, state: Hashable) -> bool:
        return state in self.accepting

    @classmethod
    def from_token_sequence_alternation(cls, class_a: Iterable[int], class_b: Iterable[int], *,
                                        length: int) -> "TokenFSA":
        """A demo grammar: emit exactly ``length`` tokens alternating between class A and class B (A first)."""
        transitions: dict[tuple[Hashable, int], Hashable] = {}
        for step in range(length):
            classes = class_a if step % 2 == 0 else class_b
            for tok in classes:
                transitions[(step, tok)] = step + 1
        return cls(transitions, start=0, accepting={length})
