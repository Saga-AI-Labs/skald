"""Tests for ``adapters.length_stats`` — the §3.3 failure-mode family.

Pure-stdlib unit tests: the single-label precedence (empty > refusal >
loop > truncated > ok), loop-period detection, refusal markers, and the
per-prompt curve summary (failure rate, first failure budget, max clean
length, with the documented -1 sentinels).
"""

from __future__ import annotations

import pytest

from adapters.length_stats import (
    MODES,
    STRESS_CURVE_METRICS,
    STRESS_STEP_METRICS,
    classify,
    loop_detected,
    refusal_detected,
    summarize_curve,
    tokens,
)


def test_metric_name_tables():
    assert set(STRESS_STEP_METRICS) == {
        "stress_ok", "stress_refusal", "stress_loop", "stress_truncated",
        "stress_empty", "stress_completion_chars",
    }
    assert set(STRESS_CURVE_METRICS) == {
        "stress_failure_rate", "stress_first_failure_at",
        "stress_max_clean_chars",
    }
    assert MODES == ("empty", "refusal", "loop", "truncated", "ok")


def test_tokens_whitespace_runs():
    assert tokens("  hello   world\nnew ") == ["hello", "world", "new"]
    assert tokens("") == []


def test_refusal_markers():
    assert refusal_detected("I'm sorry, I can't help with that.")
    assert refusal_detected("As an AI, I am unable to comply.")
    assert not refusal_detected("The answer is 42, derived step by step.")


def test_loop_detected_periodic_tail():
    tail = " ".join(["alpha beta"] * 12)
    text = "a clean opening sentence. " + tail
    loop, period, span = loop_detected(text)
    assert loop
    assert period == 2
    assert span == 6
    single = " ".join(["echo"] * 10)
    loop, period, span = loop_detected("start " + single)
    assert loop and period == 1 and span == 3


def test_loop_not_detected_for_varied_text():
    text = ("The quick brown fox jumps over the lazy dog near the river "
            "bank at dawn while birds sing in the tall green trees.")
    assert loop_detected(text) == (False, 0, 0)
    assert loop_detected("too short") == (False, 0, 0)


def test_classify_precedence():
    assert classify("", "stop") == "empty"
    assert classify("   ", "length") == "empty"
    assert classify("I'm sorry, I cannot do that.", "stop") == "refusal"
    looping = "intro words here. " + " ".join(["repeat after"] * 12)
    assert classify(looping, "length") == "loop"
    assert classify(looping, "stop") == "loop"  # a loop is a loop
    assert classify("a complete natural answer.", "length") == "truncated"
    assert classify("a complete natural answer.", "stop") == "ok"
    assert classify("a complete natural answer.", None) == "ok"


def test_summarize_curve():
    curve = summarize_curve(
        ["ok", "ok", "truncated", "loop"], [10, 20, 30, 40], [32, 128, 512, 2048]
    )
    assert set(curve) == set(STRESS_CURVE_METRICS)
    assert curve["stress_failure_rate"] == pytest.approx(0.5)
    assert curve["stress_first_failure_at"] == pytest.approx(512)
    assert curve["stress_max_clean_chars"] == pytest.approx(20)


def test_summarize_curve_sentinels():
    clean = summarize_curve(["ok", "ok"], [5, 9], [32, 64])
    assert clean["stress_failure_rate"] == pytest.approx(0.0)
    assert clean["stress_first_failure_at"] == pytest.approx(-1.0)
    assert clean["stress_max_clean_chars"] == pytest.approx(9)
    broken = summarize_curve(["refusal", "empty"], [12, 0], [32, 64])
    assert broken["stress_failure_rate"] == pytest.approx(1.0)
    assert broken["stress_first_failure_at"] == pytest.approx(32)
    assert broken["stress_max_clean_chars"] == pytest.approx(-1.0)


def test_summarize_curve_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        summarize_curve([], [], [])
