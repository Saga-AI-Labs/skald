"""Shared gate-utilisation statistics for the §3.4 utilisation family.

The §3.2 question was likelihood parity over served positions; the §3.4
question is whether every capacity slot still does anything. Over the
per-slot selection masses (load_i = selections of slot i / positions,
pooled across heads, so loads sum to n_head * k) this
module computes, per layer and gate side:

- ``dead_slot_fraction`` — slots never selected, or selected with mass
  below a stated threshold (proposal §3.4: "or selected with mass below a
  stated threshold"; the threshold always travels in the record protocol);
- ``gate_entropy`` — entropy (nats) of the load distribution over slots;
- the load histogram as bucket fractions plus mean/median/p90 of per-slot
  loads (the store holds scalars; the full histogram lives in the run
  artifact).

Stdlib only, deterministic.
"""

from __future__ import annotations

import math

# Metric names in stable report order (per layer, side, domain).
UTIL_METRICS = (
    "util_dead_slot_fraction",
    "util_gate_entropy_nats",
    "util_load_mean",
    "util_load_median",
    "util_load_p90",
    "util_positions",
)

# Default bucket edges for the load histogram, as fractions of the
# uniform-share load (mean load over slots). Bucket b0 is [0, 0.25*mean);
# the dead-slot fraction at threshold 0 is the zero-mass subset of b0.
HIST_BUCKET_EDGES = (0.25, 0.5, 0.75, 1.0)


def dead_slot_fraction(loads: list[float], threshold: float = 0.0) -> float:
    """Fraction of slots whose load is below *threshold*.

    ``threshold=0.0`` counts exactly the never-selected slots; a positive
    threshold counts those plus slots "selected with mass below a stated
    threshold" (proposal §3.4). Loads must be non-negative.
    """
    if not loads:
        raise ValueError("dead_slot_fraction of empty loads")
    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0; got {threshold}")
    dead = sum(1 for load in loads if load < threshold or
               (threshold == 0.0 and load == 0.0))
    return dead / len(loads)


def gate_entropy(loads: list[float]) -> float:
    """Entropy (nats) of the load distribution over slots.

    Loads are normalised to a probability distribution first, so this is
    the uncertainty about which slot serves a random selected position —
    0 when a single slot does everything, ln(N) when load is uniform.
    """
    if not loads:
        raise ValueError("gate_entropy of empty loads")
    total = sum(loads)
    if total <= 0.0:
        return 0.0
    return -sum((load / total) * math.log(load / total)
                for load in loads if load > 0.0)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile of empty values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    frac = pos - lo
    if frac == 0.0:
        return float(sorted_values[lo])
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[lo + 1] * frac)


def load_histogram(loads: list[float],
                   edges: tuple[float, ...] = HIST_BUCKET_EDGES
                   ) -> list[float]:
    """Fraction of slots in each load bucket, relative to the mean load.

    Edges are fractions of the mean per-slot load: bucket 0 holds slots
    below edges[0]*mean (including the never-selected), the last bucket
    holds slots at or above edges[-1]*mean. Returns len(edges)+1 fractions
    summing to 1. A zero total load puts every slot in bucket 0.
    """
    if not loads:
        raise ValueError("load_histogram of empty loads")
    mean = sum(loads) / len(loads)
    if mean <= 0.0:
        return [1.0] + [0.0] * len(edges)
    bounds = [e * mean for e in edges]
    counts = [0] * (len(edges) + 1)
    for load in loads:
        bucket = len(edges)
        for i, bound in enumerate(bounds):
            if load < bound:
                bucket = i
                break
        counts[bucket] += 1
    return [c / len(loads) for c in counts]


def summarize_gate(loads: list[float], threshold: float = 0.0
                   ) -> dict[str, float]:
    """Summarize per-slot loads of one (layer, side) gate into UTIL_METRICS.

    Returns one entry per name in UTIL_METRICS; ``util_positions`` is left
    to the caller (the capture reports it) — set to 0.0 here and overwritten
    by the adapter from the capture's position count.
    """
    if not loads:
        raise ValueError("summarize_gate of empty loads")
    ordered = sorted(float(v) for v in loads)
    return {
        "util_dead_slot_fraction": dead_slot_fraction(ordered, threshold),
        "util_gate_entropy_nats": gate_entropy(ordered),
        "util_load_mean": sum(ordered) / len(ordered),
        "util_load_median": _quantile(ordered, 0.50),
        "util_load_p90": _quantile(ordered, 0.90),
        "util_positions": 0.0,
    }


def entropy_spread(per_domain_entropy: list[float]) -> float:
    """Max-minus-min gate entropy across domains ("spread", proposal §3.4)."""
    if not per_domain_entropy:
        raise ValueError("entropy_spread of empty entropies")
    return max(per_domain_entropy) - min(per_domain_entropy)
