"""Shared functional-surface capability spec tests.

Locks in the single-source-of-truth contract (plan §3.1, §4.3): one
declarative operation list generated for both surfaces, filters restricted to
``store.schema.FILTERABLE``, RECORD_FIELDS-exact record responses, and a
deterministic anomaly rule. The spec is pure data + validation.
"""

from __future__ import annotations

import random
import re

from store.schema import FILTERABLE, RECORD_FIELDS

from surfaces import spec


def test_operation_set_is_non_empty_and_stable():
    assert spec.get_operation_ids() == (
        "list_suites_adapters",
        "query_results",
        "task_drilldown",
        "list_anomalies",
        "run_benchmark",
        "job_status",
    )
    assert len(spec.OPERATIONS) == len(set(spec.get_operation_ids()))


def test_every_filter_parameter_is_a_filterable_field():
    for op in spec.OPERATIONS:
        for param in op.request:
            if param.filterable:
                assert param.name in FILTERABLE


def test_query_results_covers_all_store_equality_filters():
    qr = spec.get_operation("query_results")
    filter_names = {p.name for p in qr.request if p.filterable}
    assert filter_names == set(FILTERABLE)
    assert [p.name for p in qr.request if p.filterable] == [
        f for f in RECORD_FIELDS if f in FILTERABLE
    ]
    for param in qr.request:
        assert param.kind in spec._KINDS


def test_task_drilldown_requires_identity_anchor_and_task():
    td = spec.get_operation("task_drilldown")
    required = {p.name for p in td.request if p.required}
    assert required == {"model_checkpoint_sha256", "task"}
    for param in td.request:
        assert param.name in set(FILTERABLE) | {"limit"}


def test_record_carrying_responses_are_record_fields_exact():
    for op in spec.OPERATIONS:
        if isinstance(op.response, spec.RecordsResponse):
            assert tuple(op.response.record_fields) == tuple(RECORD_FIELDS)
            assert op.response.metadata_keys


def test_envelope_only_adds_declared_metadata_keys():
    for op in spec.OPERATIONS:
        if isinstance(op.response, spec.RecordsResponse):
            assert "records" not in op.response.metadata_keys
            assert "metadata" not in op.response.metadata_keys


def test_spec_validates_clean_and_has_no_surface_branch():
    assert spec.validate_spec() == []
    source = re.sub(r"\s+", " ", open(spec.__file__).read())
    assert not re.search(r"(^|\s)(import|from)\s+(api|ui)(\.|\s|$)", source)


def make_record(checkpoint: str = "c" * 64, protocol: str = "p1") -> dict:
    return {
        "model_checkpoint_sha256": checkpoint,
        "adapter": "a",
        "suite": "s",
        "task": "t",
        "metric": "m",
        "value": 1.0,
        "n": 1,
        "ci_low": None,
        "ci_high": None,
        "protocol": protocol,
        "created_at": "2026-01-01T00:00:00Z",
        "host": "h",
        "script_sha256": None,
        "seed": None,
        "artifacts": [],
    }


def test_anomaly_rule_flags_cross_protocol_checkpoint_only():
    records = [
        make_record(checkpoint="a" * 64, protocol="p1"),
        make_record(checkpoint="a" * 64, protocol="p2"),
        make_record(checkpoint="b" * 64, protocol="p1"),
    ]
    anomalies = spec.flag_anomalies(records)
    assert len(anomalies) == 1
    assert anomalies[0]["model_checkpoint_sha256"] == "a" * 64
    assert anomalies[0]["protocols"] == ["p1", "p2"]
    assert anomalies[0]["protocol_count"] == 2
    assert anomalies[0]["reason"] == spec.ANOMALY_RULE


def test_anomaly_rule_is_deterministic_under_input_reordering():
    records = [
        make_record(checkpoint="z" * 64, protocol="p1"),
        make_record(checkpoint="a" * 64, protocol="p2"),
        make_record(checkpoint="z" * 64, protocol="p3"),
        make_record(checkpoint="m" * 64, protocol="p1"),
    ]
    expected = spec.flag_anomalies(records)
    rng = random.Random(0)
    for _ in range(5):
        shuffled = records[:]
        rng.shuffle(shuffled)
        assert spec.flag_anomalies(shuffled) == expected


def test_anomaly_rule_ignores_single_protocol_and_empty_store():
    assert spec.flag_anomalies([make_record(protocol="p1")] * 3) == []
    assert spec.flag_anomalies([]) == []