"""Tests for ``adapters.perturb_stats`` — the §3.5 perturbation family.

Pure-stdlib unit tests: the three deterministic perturbation operators
(ws_jitter, token_sub, clause_swap), the public dispatch, set-Jaccard, and
verdict extraction (last A–D letter with gold, else the raw output).
"""

from __future__ import annotations

import pytest

from adapters.perturb_stats import (
    PERTURB_AGG_METRICS,
    PERTURB_STEP_METRICS,
    PERTURBATIONS,
    apply_perturbation,
    clause_swap,
    jaccard,
    token_sub,
    verdict,
    ws_jitter,
)


def test_metric_tables_stable():
    assert PERTURBATIONS == ("ws_jitter", "token_sub", "clause_swap")
    assert PERTURB_STEP_METRICS == ("perturb_jaccard",
                                    "perturb_verdict_changed")
    assert PERTURB_AGG_METRICS == ("perturb_mean_jaccard",
                                   "perturb_verdict_change_rate")


def test_ws_jitter_permutes_tabs_and_indents():
    assert ws_jitter("a\tb") == "a  b"
    assert ws_jitter("    x") == "\tx"
    assert ws_jitter("        x") == "\t\tx"
    assert ws_jitter("  x") == "  x"  # 2-space indent stays
    assert ws_jitter("") == ""

    src = "def f():\n\treturn 1\n"
    assert ws_jitter(src) == "def f():\n\treturn 1\n"


def test_ws_jitter_collapses_inner_runs():
    assert ws_jitter("a      b") == "a  b"
    assert ws_jitter("a  b    c") == "a  b  c"


def test_token_sub_middle_first_last_position():
    assert token_sub("one two three") == "one ___ three"
    assert token_sub("one two three", position="first") == "___ two three"
    assert token_sub("one two three", position="last") == "one two ___"
    assert token_sub("one two three", position=0) == "___ two three"
    assert token_sub("one two three", position=1) == "one ___ three"
    assert token_sub("one two three", position=-1) == "one two ___"
    assert token_sub("one two three", position=99) == "one two ___"  # clamp
    assert token_sub("single") == "___"
    assert token_sub("") == ""


def test_clause_swap_first_two_sentences():
    text = "First sentence. Second sentence. Third stays."
    assert clause_swap(text) == "Second sentence. First sentence. Third stays."
    assert clause_swap("Just one sentence.") == "Just one sentence."
    assert clause_swap("No terminal punct here") == "No terminal punct here"


def test_jaccard():
    assert jaccard("a b c", "a b d") == pytest.approx(2 / 4)
    assert jaccard("a b c", "a b c") == pytest.approx(1.0)
    assert jaccard("a b c", "x y z") == pytest.approx(0.0)
    assert jaccard("", "") == pytest.approx(1.0)
    assert jaccard("", "a") == pytest.approx(0.0)
    assert jaccard("a b", "b a") == pytest.approx(1.0)


def test_verdict_letter_and_raw():
    assert verdict("think B is right", "B") == "B"
    assert verdict("the final answer is: C", "B") == "C"
    assert verdict("no letters here", "B") == ""
    assert verdict("free-form text", None) == "free-form text"


def test_apply_perturbation_dispatch():
    assert apply_perturbation("ws_jitter", "a  b") == "a  b"
    assert apply_perturbation("token_sub", "one two") == "one ___"
    assert apply_perturbation("clause_swap", "A. B.")
    with pytest.raises(ValueError, match="unknown perturbation"):
        apply_perturbation("nope", "x")