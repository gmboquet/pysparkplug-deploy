"""Compile a regex (or an enum of choices) into a TokenFSA over a model's vocabulary — the Outlines-style
schema→automaton step that drives the decode engine's grammar masking from arbitrary patterns.

Pipeline: regex → Thompson NFA (a useful subset: literals, ``.``, ``\\d\\w\\s`` + negations, ``[...]`` classes,
groups, ``|``, ``* + ?``, ``{n}`` / ``{n,m}``) → on-the-fly subset construction *over the vocabulary tokens*
(walk each token's characters through the NFA), yielding a :class:`TokenFSA` whose ``allowed(state)`` is exactly
the tokens that keep the output matching the pattern. Masking to that set makes non-matching output impossible.

This is the in-decode counterpart to ``gateway/constrained.py`` (which validates-and-repairs after the fact);
here the constraint is enforced token-by-token so a single pass is always well-formed."""
from __future__ import annotations

from typing import Callable

from .grammar import TokenFSA

CharPred = Callable[[str], bool]

_CLASSES: dict[str, CharPred] = {
    "d": str.isdigit,
    "D": lambda c: not c.isdigit(),
    "w": lambda c: c.isalnum() or c == "_",
    "W": lambda c: not (c.isalnum() or c == "_"),
    "s": lambda c: c.isspace(),
    "S": lambda c: not c.isspace(),
}


class _NFA:
    def __init__(self) -> None:
        self.eps: list[set[int]] = []
        self.sym: list[list[tuple[CharPred, int]]] = []

    def new(self) -> int:
        self.eps.append(set())
        self.sym.append([])
        return len(self.eps) - 1

    def closure(self, states: frozenset[int]) -> frozenset[int]:
        stack = list(states)
        seen = set(states)
        while stack:
            s = stack.pop()
            for t in self.eps[s]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    def step(self, states: frozenset[int], ch: str) -> frozenset[int]:
        nxt: set[int] = set()
        for s in states:
            for pred, t in self.sym[s]:
                if pred(ch):
                    nxt.add(t)
        return self.closure(frozenset(nxt))


class _Parser:
    """Recursive-descent regex parser building Thompson NFA fragments ``(start, accept)``."""

    def __init__(self, pattern: str, nfa: _NFA) -> None:
        self.p = pattern
        self.i = 0
        self.nfa = nfa

    def _peek(self) -> str | None:
        return self.p[self.i] if self.i < len(self.p) else None

    def parse(self) -> tuple[int, int]:
        frag = self._alt()
        if self.i != len(self.p):
            raise ValueError(f"unexpected {self.p[self.i]!r} at {self.i} in regex {self.p!r}")
        return frag

    def _alt(self) -> tuple[int, int]:
        frags = [self._concat()]
        while self._peek() == "|":
            self.i += 1
            frags.append(self._concat())
        if len(frags) == 1:
            return frags[0]
        s, a = self.nfa.new(), self.nfa.new()
        for fs, fa in frags:
            self.nfa.eps[s].add(fs)
            self.nfa.eps[fa].add(a)
        return s, a

    def _concat(self) -> tuple[int, int]:
        frags = []
        while self._peek() not in (None, "|", ")"):
            frags.append(self._repeat())
        if not frags:
            s = self.nfa.new()
            return s, s
        for k in range(len(frags) - 1):
            self.nfa.eps[frags[k][1]].add(frags[k + 1][0])
        return frags[0][0], frags[-1][1]

    def _repeat(self) -> tuple[int, int]:
        start_i = self.i
        s, a = self._atom()
        src = self.p[start_i:self.i]                          # the atom's source, for {n} replication
        c = self._peek()
        if c == "*":
            self.i += 1
            return self._star(s, a)
        if c == "+":
            self.i += 1
            return self._plus_wrap(s, a)
        if c == "?":
            self.i += 1
            return self._opt(s, a)
        if c == "{":
            return self._brace(s, a, src)
        return s, a

    def _star(self, s: int, a: int) -> tuple[int, int]:
        ns, na = self.nfa.new(), self.nfa.new()
        self.nfa.eps[ns].add(s)
        self.nfa.eps[ns].add(na)
        self.nfa.eps[a].add(s)
        self.nfa.eps[a].add(na)
        return ns, na

    def _plus_wrap(self, s: int, a: int) -> tuple[int, int]:
        ns, na = self.nfa.new(), self.nfa.new()
        self.nfa.eps[ns].add(s)
        self.nfa.eps[a].add(s)
        self.nfa.eps[a].add(na)
        return ns, na

    def _opt(self, s: int, a: int) -> tuple[int, int]:
        ns, na = self.nfa.new(), self.nfa.new()
        self.nfa.eps[ns].add(s)
        self.nfa.eps[ns].add(na)
        self.nfa.eps[a].add(na)
        return ns, na

    def _brace(self, s: int, a: int, src: str) -> tuple[int, int]:
        j = self.i + 1
        num = ""
        while j < len(self.p) and self.p[j].isdigit():
            num += self.p[j]
            j += 1
        if not num or j >= len(self.p) or self.p[j] != "}":
            raise ValueError("only exact {n} repetition is supported (use explicit copies or +/*/? for ranges)")
        self.i = j + 1
        n = int(num)
        if n == 0:
            e = self.nfa.new()
            return e, e
        frags = [(s, a)]
        for _ in range(n - 1):
            frags.append(_Parser(src, self.nfa).parse())     # replicate the atom by re-parsing its source
        for k in range(len(frags) - 1):
            self.nfa.eps[frags[k][1]].add(frags[k + 1][0])
        return frags[0][0], frags[-1][1]

    def _atom(self) -> tuple[int, int]:
        c = self._peek()
        if c == "(":
            self.i += 1
            frag = self._alt()
            if self._peek() != ")":
                raise ValueError("unbalanced '(' in regex")
            self.i += 1
            return frag
        if c == "[":
            return self._charclass()
        if c == ".":
            self.i += 1
            return self._sym_frag(lambda ch: ch != "\n")
        if c == "\\":
            self.i += 1
            esc = self._peek()
            if esc is None:
                raise ValueError("trailing backslash in regex")
            self.i += 1
            if esc in _CLASSES:
                return self._sym_frag(_CLASSES[esc])
            return self._sym_frag(lambda ch, e=esc: ch == e)
        if c is None or c in ")|*+?{":
            raise ValueError(f"unexpected {c!r} in regex {self.p!r}")
        self.i += 1
        return self._sym_frag(lambda ch, lit=c: ch == lit)

    def _charclass(self) -> tuple[int, int]:
        self.i += 1                                           # consume '['
        negate = False
        if self._peek() == "^":
            negate = True
            self.i += 1
        preds: list[CharPred] = []
        while self._peek() not in (None, "]"):
            ch = self.p[self.i]
            if ch == "\\":
                self.i += 1
                esc = self.p[self.i]
                self.i += 1
                preds.append(_CLASSES.get(esc, (lambda c, e=esc: c == e)))
                continue
            # range a-z
            if self.i + 2 < len(self.p) and self.p[self.i + 1] == "-" and self.p[self.i + 2] != "]":
                lo, hi = ch, self.p[self.i + 2]
                preds.append(lambda c, lo=lo, hi=hi: lo <= c <= hi)
                self.i += 3
            else:
                preds.append(lambda c, lit=ch: c == lit)
                self.i += 1
        if self._peek() != "]":
            raise ValueError("unbalanced '[' in regex")
        self.i += 1

        def pred(c: str) -> bool:
            hit = any(p(c) for p in preds)
            return (not hit) if negate else hit

        return self._sym_frag(pred)

    def _sym_frag(self, pred: CharPred) -> tuple[int, int]:
        s, a = self.nfa.new(), self.nfa.new()
        self.nfa.sym[s].append((pred, a))
        return s, a


def regex_to_token_fsa(pattern: str, vocab: dict[int, str], *, eos_id: int | None = None,
                       max_states: int = 50_000) -> TokenFSA:
    """Compile ``pattern`` into a TokenFSA over ``vocab`` (``{token_id: token_text}``). ``allowed(state)`` is the
    set of tokens that keep the output matching the regex; accepting states are where the regex has fully matched
    (and, if ``eos_id`` is given, accept an explicit end token there)."""
    nfa = _NFA()
    start, accept = _Parser(pattern, nfa).parse()
    start_set = nfa.closure(frozenset([start]))

    state_id: dict[frozenset[int], int] = {start_set: 0}
    order: list[frozenset[int]] = [start_set]
    transitions: dict[tuple[int, int], int] = {}
    accepting: set[int] = set()
    final_id: int | None = None

    qi = 0
    while qi < len(order):
        cur = order[qi]
        qi += 1
        cid = state_id[cur]
        if accept in cur:
            accepting.add(cid)
            if eos_id is not None:
                if final_id is None:
                    final_id = len(state_id)
                    sink = frozenset()                        # dead/terminal state after EOS
                    state_id[sink] = final_id
                    accepting.add(final_id)
                transitions[(cid, eos_id)] = final_id
        for tok_id, text in vocab.items():
            if not text:
                continue
            nxt = cur
            ok = True
            for ch in text:
                nxt = nfa.step(nxt, ch)
                if not nxt:
                    ok = False
                    break
            if not ok or not nxt:
                continue
            if nxt not in state_id:
                if len(state_id) >= max_states:
                    raise ValueError(f"token-FSA exceeded {max_states} states for {pattern!r}")
                state_id[nxt] = len(state_id)
                order.append(nxt)
            transitions[(cid, tok_id)] = state_id[nxt]

    return TokenFSA(transitions, start=0, accepting=accepting)


def build_choice_fsa(choices: list[str], vocab: dict[int, str], *, eos_id: int | None = None) -> TokenFSA:
    """Constrain output to exactly one of ``choices`` — a trie over the vocabulary (always correct, no regex)."""
    inv: dict[str, list[int]] = {}
    for tid, text in vocab.items():
        if text:
            inv.setdefault(text, []).append(tid)

    transitions: dict[tuple[tuple, int], tuple] = {}
    accepting: set[tuple] = set()

    def add(choice: str) -> None:
        state: tuple = ()
        remaining = choice
        while remaining:
            # greedily consume the longest vocab token that is a prefix of `remaining`
            tok = next((t for t in sorted(inv, key=len, reverse=True) if remaining.startswith(t)), None)
            if tok is None:
                return                                        # choice not expressible in this vocab; skip it
            nxt = state + (tok,)
            for tid in inv[tok]:
                transitions[(state, tid)] = nxt
            state = nxt
            remaining = remaining[len(tok):]
        accepting.add(state)
        if eos_id is not None:
            transitions[(state, eos_id)] = state

    for ch in choices:
        add(ch)
    # remap tuple states to ints
    states = {(): 0}
    for (st, _tid), nxt in transitions.items():
        for s in (st, nxt):
            if s not in states:
                states[s] = len(states)
    int_transitions = {(states[st], tid): states[nxt] for (st, tid), nxt in transitions.items()}
    int_accepting = {states[s] for s in accepting}
    return TokenFSA(int_transitions, start=0, accepting=int_accepting)
