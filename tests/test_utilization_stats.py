"""Tests for ``adapters.utilization_stats`` — the §3.4 metric family.

Pure-stdlib unit tests: dead-slot fraction (never-selected and threshold
semantics), gate entropy bounds, histogram bucket bookkeeping, the
per-gate summary, and the cross-domain entropy spread.
"""

from __future__ import annotations

import math

import pytest

from adapters.utilization_stats import (
    UTIL_METRICS,
    dead_slot_fraction,
    entropy_spread,
    gate_entropy,
    load_histogram,
    summarize_gate,
)


def test_dead_slot_fraction_strict():
    assert dead_slot_fraction([0.0, 0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert dead_slot_fraction([0.1, 0.2]) == pytest.approx(0.0)
    assert dead_slot_fraction([0.0, 0.0]) == pytest.approx(1.0)


def test_dead_slot_fraction_threshold():
    # "selected with mass below a stated threshold" (proposal §3.4)
    loads = [0.0, 0.01, 0.05, 0.5]
    assert dead_slot_fraction(loads, threshold=0.02) == pytest.approx(0.5)
    assert dead_slot_fraction(loads, threshold=0.0) == pytest.approx(0.25)


def test_dead_slot_fraction_rejects():
    with pytest.raises(ValueError, match="empty"):
        dead_slot_fraction([])
    with pytest.raises(ValueError, match="threshold"):
        dead_slot_fraction([0.1], threshold=-0.5)


def test_gate_entropy_bounds():
    # uniform load over 4 slots -> ln(4); single slot -> 0
    assert gate_entropy([1.0, 1.0, 1.0, 1.0]) == pytest.approx(math.log(4))
    assert gate_entropy([3.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)
    # scale-invariant: doubling every load changes nothing
    assert gate_entropy([2.0, 2.0]) == pytest.approx(gate_entropy([1.0, 1.0]))
    # all-zero loads: nothing selected anywhere -> 0, not NaN
    assert gate_entropy([0.0, 0.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="empty"):
        gate_entropy([])


def test_load_histogram_bookkeeping():
    # mean = 0.5; edges at 0.125/0.25/0.375/0.5 -> buckets [2,1,1,0,4]
    loads = [0.0, 0.0, 0.2, 0.3, 1.0, 1.0, 1.0, 0.5]
    hist = load_histogram(loads)
    assert len(hist) == 5
    assert sum(hist) == pytest.approx(1.0)
    assert hist[0] == pytest.approx(2 / 8)
    assert hist[-1] == pytest.approx(4 / 8)
    # zero total load: everything dead in bucket 0
    assert load_histogram([0.0, 0.0]) == [1.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="empty"):
        load_histogram([])


def test_summarize_gate_shape():
    summary = summarize_gate([50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert set(summary) == set(UTIL_METRICS)
    assert summary["util_dead_slot_fraction"] == pytest.approx(0.75)
    assert summary["util_gate_entropy_nats"] == pytest.approx(math.log(2))
    assert summary["util_load_mean"] == pytest.approx(100 / 8)
    assert summary["util_positions"] == pytest.approx(0.0)  # caller fills in
    with pytest.raises(ValueError, match="empty"):
        summarize_gate([])


def test_entropy_spread():
    assert entropy_spread([1.0, 1.0]) == pytest.approx(0.0)
    assert entropy_spread([0.5, 1.0, 1.5]) == pytest.approx(1.0)
    assert entropy_spread([2.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="empty"):
        entropy_spread([])
