"""Shared functional-surface capability specification (plan §3.1, §4.3).

The single declarative source from which both the ``api/`` and ``ui/``
surfaces are generated: list suites/adapters, query results, per-task
drilldown, and the anomaly view. "A capability added to one appears in both
by construction" (§4.3) holds because neither surface invents its own
capability list — they enumerate a common set, built here and validated
against ``store.schema`` at import time.

The spec is surface-neutral on purpose: it imports only the stdlib and the
``store`` module, declares request parameters and response shapes derived
from ``store.schema.FILTERABLE`` / ``RECORD_FIELDS``, and carries no per-
surface branch. No cloud LLM participates anywhere in this chain; the spec
is pure data plus validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from store.schema import FILTERABLE, RECORD_FIELDS

SPEC_ID = "skald-functional-surface"
SPEC_VERSION = "1"
SPEC_DESCRIPTION = (
    "Functional surface of the Skald result store: one declarative operation "
    "set shared by the api/ and ui/ surfaces (plan §3.1, §4.3), derived from "
    "the unified record schema (plan §4.2)."
)

# --- Anomaly rule -------------------------------------------------------

# The exact deterministic rule the anomaly view applies (plan §4.3 "anomaly
# view"). A model_checkpoint_sha256 that appears under two or more distinct
# protocol labels is flagged: plan §4.2 forbids silently comparing numbers
# from different protocols, and a checkpoint straddling protocols is exactly
# the shape that invites that mistake.
ANOMALY_RULE = "cross-protocol-checkpoint"


def flag_anomalies(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag records worth an anomaly view under :data:`ANOMALY_RULE`.

    Deterministic: entries are ordered by sorted checkpoint, protocols by
    sorted label, and input order does not influence the result. Returns one
    entry per flagged checkpoint:

    - ``model_checkpoint_sha256`` — the checkpoint appearing under multiple
      protocols,
    - ``protocols`` — the distinct protocol labels, sorted,
    - ``protocol_count`` — how many distinct protocols it spans,
    - ``reason`` — always :data:`ANOMALY_RULE`.
    """
    by_checkpoint: dict[str, set[str]] = {}
    for record in records:
        by_checkpoint.setdefault(
            record["model_checkpoint_sha256"], set()
        ).add(record["protocol"])

    anomalies: list[dict[str, Any]] = []
    for checkpoint in sorted(by_checkpoint):
        protocols = sorted(by_checkpoint[checkpoint])
        if len(protocols) > 1:
            anomalies.append(
                {
                    "model_checkpoint_sha256": checkpoint,
                    "protocols": protocols,
                    "protocol_count": len(protocols),
                    "reason": ANOMALY_RULE,
                }
            )
    return anomalies


# --- Declarative shapes ----------------------------------------------------

_KINDS = frozenset({"string", "number", "int", "scalar"})


@dataclass(frozen=True)
class Parameter:
    """One declared request parameter.

    ``filterable=True`` marks a parameter that must be a
    ``store.schema.FILTERABLE`` member — an equality filter passed straight to
    ``store.query``.
    """

    name: str
    kind: str  # "string" | "number" | "int" | "scalar"
    required: bool
    description: str
    filterable: bool = False


@dataclass(frozen=True)
class RecordsResponse:
    """A record-carrying response: canonical records inside a declared envelope.

    Enforces the §4.2 contract literally: the record key set is exactly
    ``RECORD_FIELDS`` and only the declared ``metadata_keys`` may be added
    around it.
    """

    description: str
    metadata_keys: tuple[str, ...]
    record_fields: tuple[str, ...] = RECORD_FIELDS

    def __post_init__(self) -> None:
        if tuple(self.record_fields) != tuple(RECORD_FIELDS):
            raise ValueError(
                "record-carrying responses must use the canonical "
                f"RECORD_FIELDS key set exactly; got {self.record_fields}"
            )


@dataclass(frozen=True)
class ValuesResponse:
    """A non-record response: a declared set of value lists (e.g. listings)."""

    description: str
    value_keys: tuple[str, ...]


@dataclass(frozen=True)
class Operation:
    """One capability of the shared functional surface."""

    id: str
    summary: str
    request: tuple[Parameter, ...]
    response: RecordsResponse | ValuesResponse


# --- Filter parameters derived from store.schema ---------------------------
#
# (name) -> (kind, description), one entry per FILTERABLE field. Kept in lock-
# step with store.schema: any field added to FILTERABLE must be declared here
# too, or module import fails and the surface builders notice immediately.

_FILTER_META: dict[str, tuple[str, str]] = {
    "model_checkpoint_sha256": (
        "string",
        "Identity anchor — SHA-256 of the evaluated weights (plan §4.4).",
    ),
    "adapter": ("string", "Adapter name of the benchmark family that ran it."),
    "suite": ("string", "Suite name within the family."),
    "task": ("string", "Task name within the suite."),
    "metric": ("string", "Metric name."),
    "value": ("number", "Metric value (finite number)."),
    "n": ("int", "Sample count, or null."),
    "ci_low": ("number", "Lower confidence bound, or null."),
    "ci_high": ("number", "Upper confidence bound, or null."),
    "protocol": (
        "string",
        "Evaluation protocol label (plan §4.2) — never silently compared",
    ),
    "created_at": ("string", "Record creation timestamp (RFC 3339, UTC)."),
    "host": ("string", "Host that ran the benchmark, or null."),
    "script_sha256": ("string", "SHA-256 of the generating script, or null."),
    "seed": ("scalar", "Random seed (int or string), or null."),
}

_LIMIT = Parameter(
    name="limit",
    kind="int",
    required=False,
    description="Maximum number of records to return; absent means all.",
)


def _filter_parameters() -> tuple[Parameter, ...]:
    """One optional equality-filter Parameter per FILTERABLE field.

    Declared in the canonical ``RECORD_FIELDS`` order so the surfaces (and the
    conformance test) see a stable, schema-derived parameter list.
    """
    return tuple(
        Parameter(
            name=name,
            kind=_FILTER_META[name][0],
            required=False,
            description=_FILTER_META[name][1],
            filterable=True,
        )
        for name in RECORD_FIELDS
        if name in FILTERABLE
    )


# --- The one operation set (no per-surface branch) --------------------------

OPERATIONS: tuple[Operation, ...] = (
    Operation(
        id="list_suites_adapters",
        summary=(
            "List the distinct suite and adapter values present in the store "
            "— the result-browsing entry point (plan §4.3)."
        ),
        request=(),
        response=ValuesResponse(
            description="Distinct dimension values under the declared keys.",
            value_keys=("suites", "adapters"),
        ),
    ),
    Operation(
        id="query_results",
        summary=(
            "Query result records by equality on any filterable field "
            "(plan §4.2) — the store query surface."
        ),
        request=_filter_parameters() + (_LIMIT,),
        response=RecordsResponse(
            description=(
                "Matching canonical records inside a metadata envelope; "
                "\"count\" is the number of records returned."
            ),
            metadata_keys=("count",),
        ),
    ),
    Operation(
        id="task_drilldown",
        summary=(
            "Per-task drilldown (plan §4.3): every record for one checkpoint "
            "and one task."
        ),
        request=(
            Parameter(
                name="model_checkpoint_sha256",
                kind="string",
                required=True,
                description=(
                    "Identity anchor — SHA-256 of the evaluated weights "
                    "(plan §4.4); the drilldown is per checkpoint."
                ),
                filterable=True,
            ),
            Parameter(
                name="task",
                kind="string",
                required=True,
                description="Task name within the suite; the drilldown is per task.",
                filterable=True,
            ),
            _LIMIT,
        ),
        response=RecordsResponse(
            description=(
                "The task's canonical records; metadata echoes the requested "
                "checkpoint and task and reports the record count."
            ),
            metadata_keys=("model_checkpoint_sha256", "task", "count"),
        ),
    ),
    Operation(
        id="list_anomalies",
        summary=(
            "Anomaly view (plan §4.3): records worth flagging under the "
            "declared deterministic rule ANOMALY_RULE."
        ),
        request=_filter_parameters() + (_LIMIT,),
        response=RecordsResponse(
            description=(
                "Canonical records belonging to flagged checkpoints; metadata "
                "\"anomalies\" lists one entry per flagged checkpoint from "
                "flag_anomalies, and \"count\" the number of records returned."
            ),
            metadata_keys=("count", "anomalies"),
        ),
    ),
)

_OPERATION_INDEX = {op.id: op for op in OPERATIONS}


def get_operation_ids() -> tuple[str, ...]:
    """The full, sole capability list — the ids both surfaces generate from."""
    return tuple(op.id for op in OPERATIONS)


def get_operation(operation_id: str) -> Operation:
    """Look up one operation descriptor by id."""
    try:
        return _OPERATION_INDEX[operation_id]
    except KeyError:
        raise KeyError(f"unknown surface operation {operation_id!r}") from None


# --- Validation --------------------------------------------------------------

def validate_spec() -> list[str]:
    """Return every detected spec/store-schema divergence (empty = valid).

    Mirrors the contract the acceptance tests assert: unique non-empty ids,
    every ``filterable`` parameter a ``FILTERABLE`` member, ``query_results``
    covering exactly the full ``FILTERABLE`` set, valid kinds, and
    RECORD_FIELDS-exact record responses.
    """
    errors: list[str] = []

    unknown_meta = sorted(set(_FILTER_META) - set(FILTERABLE))
    if unknown_meta:
        errors.append(f"_FILTER_META declares non-FILTERABLE fields: {unknown_meta}")
    missing_meta = sorted(set(FILTERABLE) - set(_FILTER_META))
    if missing_meta:
        errors.append(f"_FILTER_META missing FILTERABLE fields: {missing_meta}")

    ids = [op.id for op in OPERATIONS]
    if not ids:
        errors.append("operation set is empty")
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        errors.append(f"duplicate operation ids: {duplicates}")

    for op in OPERATIONS:
        if not op.id:
            errors.append(f"operation with empty id in {op!r}")
        for param in op.request:
            if param.kind not in _KINDS:
                errors.append(
                    f"{op.id}: parameter {param.name!r} has unknown kind "
                    f"{param.kind!r}"
                )
            if param.filterable and param.name not in FILTERABLE:
                errors.append(
                    f"{op.id}: filter parameter {param.name!r} is not in "
                    "store.schema.FILTERABLE"
                )
            if not param.filterable and param.name in FILTERABLE:
                errors.append(
                    f"{op.id}: parameter {param.name!r} is a FILTERABLE field "
                    "but is not marked filterable"
                )
        response = op.response
        if isinstance(response, RecordsResponse):
            if tuple(response.record_fields) != tuple(RECORD_FIELDS):
                errors.append(
                    f"{op.id}: response records are not RECORD_FIELDS-exact"
                )

    query_results = get_operation("query_results")
    qr_filters = {p.name for p in query_results.request if p.filterable}
    if qr_filters != set(FILTERABLE):
        errors.append(
            "query_results must cover exactly store.schema.FILTERABLE: "
            f"got {sorted(qr_filters)}, expected {sorted(FILTERABLE)}"
        )

    return errors


_import_errors = validate_spec()
if _import_errors:
    raise ValueError("invalid surfaces spec:\n- " + "\n- ".join(_import_errors))