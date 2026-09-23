"""Unit tests for ``adapters.likelihood_stats`` — shared §3.2 distributions.

Stdlib-only, deterministic by construction: no sampling, no network, no
torch. Covers the exact shape the store sees (per-domain ``kl_*`` / ``kld_*``
metric families) plus the quantile interpolation contract.
"""

from __future__ import annotations

import pytest

from adapters.likelihood_stats import (
    KLD_METRICS,
    KL_METRICS,
    _summarize,
    quantile,
    summarize_kld,
    summarize_kl,
)


def test_metric_families_are_parallel_and_distinct():
    assert len(KLD_METRICS) == len(KL_METRICS) == 7
    assert KLD_METRICS == tuple(m.replace("kl_", "kld_", 1) for m in KL_METRICS)
    # Path A and Path B must never collide in the store — reporting one
    # family when meaning the other is the exact bug the proposal warns of.
    assert set(KLD_METRICS).isdisjoint(set(KL_METRICS))
    assert KLD_METRICS == (
        "kld_mean", "kld_median", "kld_p95", "kld_p99",
        "kld_p999", "kld_max", "kld_cvar95",
    )


def test_summarize_returns_metric_family_plus_n():
    summary = summarize_kl([1.0, 2.0, 3.0])
    assert set(summary) == set(KL_METRICS) | {"n"}
    summary = summarize_kld([1.0, 2.0, 3.0])
    assert set(summary) == set(KLD_METRICS) | {"n"}
    assert "kl_mean" not in summary and "kld_max" in summary


def test_summarize_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        summarize_kl([])

    with pytest.raises(ValueError, match="empty"):
        summarize_kld([])


def test_quantile_single_value():
    assert quantile([3.0], 0.5) == 3.0
    assert quantile([-2.5], 0.0) == -2.5
    assert quantile([-2.5], 1.0) == -2.5


def test_quantile_linear_interpolation():
    ordered = [1.0, 2.0, 3.0, 4.0]
    assert quantile(ordered, 0.5) == pytest.approx(2.5)  # pos 1.5
    assert quantile(ordered, 0.0) == 1.0
    assert quantile(ordered, 1.0) == 4.0
    assert quantile(ordered, 0.25) == pytest.approx(1.75)  # pos 0.75
    assert quantile(ordered, 0.95) == pytest.approx(3.85)  # pos 2.85


def test_quantile_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError, match="empty"):
        quantile([], 0.5)
    with pytest.raises(ValueError, match="q must be"):
        quantile([1.0], -0.1)
    with pytest.raises(ValueError, match="q must be"):
        quantile([1.0], 1.0000001)


def test_summarize_known_distribution():
    values = [0.0, 0.5, 1.0, 2.0, 5.0]
    s = summarize_kl(values)
    assert s["kl_mean"] == pytest.approx(1.7)
    assert s["kl_median"] == pytest.approx(1.0)
    assert s["kl_p95"] == pytest.approx(4.4)    # pos 3.8 -> 2 + 0.8*(5-2)
    assert s["kl_p99"] == pytest.approx(4.88)
    assert s["kl_p999"] == pytest.approx(4.988)
    assert s["kl_max"] == 5.0
    assert s["kl_cvar95"] == pytest.approx(5.0)  # only 5.0 at/above p95
    assert s["n"] == 5


def test_cvar95_is_mean_of_violating_tail():
    # The tail is the measurement (§3.2): a mean-only summary is a gate a
    # damaged distribution can walk through. cvar95 must reflect the bad tail.
    tail_heavy = summarize_kl([0.1, 0.1, 0.1, 5.0, 6.0])
    assert tail_heavy["kl_mean"] == pytest.approx(2.26)
    # p95 of [0.1,0.1,0.1,5,6]: pos 3.8 -> 5 + 0.8*(6-5) = 5.8; tail >= 5.8 -> [6]
    assert tail_heavy["kl_cvar95"] == pytest.approx(6.0)


def test_summarize_handles_nan_free_negative_values():
    # Realized-token KLD (Path B) is signed: ref_lp - cand_lp. Negative
    # values are valid and must not be clamped or dropped.
    s = summarize_kld([-3.0, -1.0, 0.0])
    assert s["kld_mean"] == pytest.approx(-4.0 / 3)
    assert s["kld_max"] == 0.0
    assert s["kld_median"] == pytest.approx(-1.0)
    assert s["n"] == 3


def test_summarize_does_not_mutate_input():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    before = list(values)
    summarize_kl(values)
    assert values == before


def test_stats_module_provides_bare_family():
    # Domain composition happens in the adapter ("family@domain"); the stats
    # module supplies the bare family only.
    s = summarize_kl([1.0, 2.0])
    assert "kl_mean@legal" not in s
    assert "kl_mean" in s