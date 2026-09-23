"""Shared distribution statistics for the §3.2 likelihood-parity family.

Both parity paths (Path A hidden-state capture in ``bdh_likelihood`` and
Path B serving-path logprobs in ``openai_compat``) report the same metric
family over per-token KL values: mean, median, p95, p99, p99.9, max, and
CVaR95 — because the tail is the measurement (proposal §3.2a), a mean-only
KL is a gate a damaged distribution can walk through.

Stdlib only, deterministic: linear-interpolation quantiles over sorted
values, no sampling, no ties broken by input order.
"""

from __future__ import annotations

# Metric names in stable report order.
KL_METRICS = (
    "kl_mean",
    "kl_median",
    "kl_p95",
    "kl_p99",
    "kl_p999",
    "kl_max",
    "kl_cvar95",
)

# Path B (serving-path) names: realized-token KLD, not full-vocab KL. Kept
# distinct so the two paths can never be conflated in the store — reporting
# one when meaning the other is the 2.2× bug the proposal warns about.
KLD_METRICS = tuple(m.replace("kl_", "kld_", 1) for m in KL_METRICS)


def quantile(sorted_values: list[float], q: float) -> float:
    """q-quantile (0 <= q <= 1) by linear interpolation over sorted values."""
    if not sorted_values:
        raise ValueError("quantile of empty values")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1]; got {q}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    frac = pos - lo
    if frac == 0.0:
        return float(sorted_values[lo])
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[lo + 1] * frac)


def _summarize(values: list[float], metric_names: tuple[str, ...]) -> dict[str, float]:
    """Summarize signed per-token values into the §3.2 metric family.

    Returns one entry per name in *metric_names* plus ``"n"`` (count).
    CVaR95 is the mean over values at or above the p95 threshold — the
    expected damage given you are already in the bad tail.
    """
    if not values:
        raise ValueError("summarize of empty values")
    ordered = sorted(float(v) for v in values)
    p95 = quantile(ordered, 0.95)
    tail = [v for v in ordered if v >= p95]
    return {
        metric_names[0]: sum(ordered) / len(ordered),
        metric_names[1]: quantile(ordered, 0.50),
        metric_names[2]: p95,
        metric_names[3]: quantile(ordered, 0.99),
        metric_names[4]: quantile(ordered, 0.999),
        metric_names[5]: ordered[-1],
        metric_names[6]: sum(tail) / len(tail),
        "n": len(ordered),
    }


def summarize_kl(values: list[float]) -> dict[str, float]:
    """Summarize per-token KL values (Path A, full-vocab) as ``kl_*``."""
    return _summarize(values, KL_METRICS)


def summarize_kld(values: list[float]) -> dict[str, float]:
    """Summarize realized-token KLD values (Path B, serving-path) as ``kld_*``."""
    return _summarize(values, KLD_METRICS)
