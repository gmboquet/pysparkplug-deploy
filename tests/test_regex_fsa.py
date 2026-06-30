"""Regex / enum -> TokenFSA compiler: the compiled automaton must constrain the decode engine's output to match."""
import numpy as np
import pytest

from mixle_mlops.engines import NgramProvider, decode
from mixle_mlops.engines.regex_fsa import build_choice_fsa, regex_to_token_fsa


def _char_vocab(chars: str) -> dict[int, str]:
    return {i: c for i, c in enumerate(chars)}


def test_regex_plus_class_constrains_output():
    chars = "ab012"
    vocab = _char_vocab(chars)
    eos = len(chars)
    fsa = regex_to_token_fsa("[ab]+", vocab, eos_id=eos)
    v = len(chars) + 1
    table = np.full((v, v), -10.0)
    table[:, 2] = 5.0          # provider most wants '0' (id 2) — but it's masked out of [ab]+
    table[:, 0] = 4.0          # then 'a' (id 0)
    out = decode(NgramProvider(table), prompt_ids=[0], max_new_tokens=6, grammar=fsa, eos_id=eos, greedy=True)
    generated = [t for t in out if t != eos]
    assert generated and all(vocab.get(t) in ("a", "b") for t in generated)


def test_regex_exact_count():
    chars = "01"
    vocab = _char_vocab(chars)
    fsa = regex_to_token_fsa("0{3}", vocab)                   # exactly "000"
    out = decode(NgramProvider(np.zeros((2, 2))), prompt_ids=[0], max_new_tokens=10, grammar=fsa, greedy=True)
    assert "".join(chars[t] for t in out) == "000"


def test_regex_alternation_and_groups():
    chars = "abc"
    vocab = _char_vocab(chars)
    fsa = regex_to_token_fsa("(a|b)c", vocab)                 # "ac" or "bc"
    table = np.full((3, 3), -10.0)
    table[:, 1] = 5.0                                         # prefer 'b'
    out = decode(NgramProvider(table), prompt_ids=[0], max_new_tokens=10, grammar=fsa, greedy=True)
    assert "".join(chars[t] for t in out) == "bc"


def test_choice_fsa_forces_one_choice():
    chars = "yesno "
    vocab = _char_vocab(chars)
    fsa = build_choice_fsa(["yes", "no"], vocab)
    v = len(chars)
    table = np.full((v, v), -10.0)
    table[:, 3] = 5.0                                         # prefer 'n' -> steer toward "no"
    out = decode(NgramProvider(table), prompt_ids=[0], max_new_tokens=10, grammar=fsa, greedy=True)
    assert "".join(chars[t] for t in out) == "no"


def test_invalid_regex_raises():
    with pytest.raises(ValueError):
        regex_to_token_fsa("a{2,5}", _char_vocab("a"))        # ranges are an honest unsupported subset


def test_digit_class_matches_digits_only():
    chars = "0a"
    vocab = _char_vocab(chars)
    fsa = regex_to_token_fsa(r"\d", vocab)                    # exactly one digit
    table = np.full((2, 2), -10.0)
    table[:, 1] = 5.0                                         # provider wants 'a' (id 1) — masked, not a digit
    out = decode(NgramProvider(table), prompt_ids=[0], max_new_tokens=5, grammar=fsa, greedy=True)
    assert "".join(chars[t] for t in out) == "0"
