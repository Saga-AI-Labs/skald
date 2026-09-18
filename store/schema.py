"""Unified result-record schema (plan §4.2).

One record per (model, suite, task, metric).  The checkpoint SHA-256 is the
identity anchor and the join key (plan §4.4) — no separate model registry.
"""

from __future__ import annotations

import math
import socket
from datetime import datetime, timezone

# Canonical field order from plan §4.2.
RECORD_FIELDS: tuple[str, ...] = (
    "model_checkpoint_sha256",
    "adapter",
    "suite",
    "task",
    "metric",
    "value",
    "n",
    "ci_low",
    "ci_high",
    "protocol",
    "created_at",
    "host",
    "script_sha256",
    "seed",
    "artifacts",
)

# Fields that must be present and non-empty on every record.
REQUIRED_TEXT = frozenset(
    {
        "model_checkpoint_sha256",
        "protocol",
        "adapter",
        "suite",
        "task",
        "metric",
    }
)

# Queryable scalar index columns (plan §4.2; artifacts is a list and is only
# stored in the per-run file, never an equality-filter target).
FILTERABLE = frozenset(
    {
        "model_checkpoint_sha256",
        "adapter",
        "suite",
        "task",
        "metric",
        "value",
        "n",
        "ci_low",
        "ci_high",
        "protocol",
        "created_at",
        "host",
        "script_sha256",
        "seed",
    }
)


class ValidationError(ValueError):
    """Raised when a record does not conform to the unified schema."""


def utcnow() -> str:
    """Current UTC time as an RFC 3339 timestamp (``...Z``)."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite(value) -> bool:
    return _is_number(value) and math.isfinite(value)


def normalize(record: dict) -> dict:
    """Validate *record* and return a normalized copy with defaults applied.

    Enforces the plan §4.2 shape: no unknown fields, and non-empty
    ``model_checkpoint_sha256`` and ``protocol`` on every record.  Optional
    fields default to ``created_at`` = now (UTC), ``host`` = local hostname,
    ``artifacts`` = ``[]``, and nulls for the other optional scalars.
    """
    if not isinstance(record, dict):
        raise ValidationError("record must be a dict")

    unknown = set(record) - set(RECORD_FIELDS)
    if unknown:
        raise ValidationError(
            f"unknown fields {sorted(unknown)}; allowed: {list(RECORD_FIELDS)}"
        )

    out: dict = {}

    for field in set(RECORD_FIELDS):
        if field not in record:
            continue
        value = record[field]

        if field in REQUIRED_TEXT:
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"{field} must be a non-empty string")
            out[field] = value.strip()
        elif field == "value":
            if not _is_finite(value):
                raise ValidationError("value must be a finite number")
            out[field] = value
        elif field == "n":
            if value is None:
                out[field] = None
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError("n must be a non-negative integer or null")
            else:
                out[field] = value
        elif field in ("ci_low", "ci_high"):
            if value is None:
                out[field] = None
            elif not _is_finite(value):
                raise ValidationError(f"{field} must be a finite number or null")
            else:
                out[field] = value
        elif field == "created_at":
            if not isinstance(value, str) or not value.strip():
                raise ValidationError("created_at must be a non-empty string")
            out[field] = value.strip()
        elif field in ("host", "script_sha256"):
            if value is None:
                out[field] = None
            elif not isinstance(value, str):
                raise ValidationError(f"{field} must be a string or null")
            else:
                out[field] = value
        elif field == "seed":
            if value is None:
                pass
            elif isinstance(value, bool):
                raise ValidationError("seed must be an int, string, or null")
            elif isinstance(value, (int, str)):
                out[field] = value
            else:
                raise ValidationError("seed must be an int, string, or null")
        elif field == "artifacts":
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValidationError("artifacts must be a list of strings")
            out[field] = list(value)

    # Mandatory fields: caught above when present-but-empty, here when absent.
    for field in ("model_checkpoint_sha256", "protocol"):
        if field not in out:
            raise ValidationError(f"{field} is required on every record")

    # Optional defaults.
    out.setdefault("n", None)
    out.setdefault("ci_low", None)
    out.setdefault("ci_high", None)
    out.setdefault("created_at", utcnow())
    out.setdefault("host", socket.gethostname())
    out.setdefault("script_sha256", None)
    out.setdefault("seed", None)
    out.setdefault("artifacts", [])

    # Reorder to the canonical plan §4.2 field order.
    return {field: out[field] for field in RECORD_FIELDS}