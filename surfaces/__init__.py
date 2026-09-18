"""Shared functional-surface capability specification (plan §3.1, §4.3).

The single source from which both the ``api/`` and ``ui/`` surfaces are
generated. Neutral by construction: this package imports only the stdlib and
the ``store`` module, and declares operations without any per-surface branch.
"""

from .spec import (
    ANOMALY_RULE,
    OPERATIONS,
    SPEC_DESCRIPTION,
    SPEC_ID,
    SPEC_VERSION,
    Operation,
    Parameter,
    RecordsResponse,
    ValuesResponse,
    flag_anomalies,
    get_operation,
    get_operation_ids,
    validate_spec,
)

__all__ = [
    "ANOMALY_RULE",
    "OPERATIONS",
    "SPEC_DESCRIPTION",
    "SPEC_ID",
    "SPEC_VERSION",
    "Operation",
    "Parameter",
    "RecordsResponse",
    "ValuesResponse",
    "flag_anomalies",
    "get_operation",
    "get_operation_ids",
    "validate_spec",
]