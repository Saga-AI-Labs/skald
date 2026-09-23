"""Store round-trip tests — unified result schema and put/query (plan §4.2)."""

from __future__ import annotations

import json

import pytest

from store import Store, ValidationError, normalize
from store.schema import RECORD_FIELDS


def make_record(**overrides) -> dict:
    base = {
        "model_checkpoint_sha256": "a" * 64,
        "adapter": "bdh_cl",
        "suite": "continual-learning",
        "task": "territory-routing",
        "metric": "accuracy",
        "value": 0.93,
        "n": 100,
        "ci_low": 0.90,
        "ci_high": 0.96,
        "protocol": "random-crop-cold",
        "script_sha256": "b" * 64,
        "seed": 7,
        "artifacts": ["/tmp/run/out.jsonl"],
    }
    base.update(overrides)
    return base


def test_round_trip_with_filters(tmp_path):
    store = Store(tmp_path)
    sq = store.put(
        [
            make_record(model_checkpoint_sha256="a" * 64),
            make_record(model_checkpoint_sha256="c" * 64, value=0.71),
        ]
    )

    got = store.query(
        {"model_checkpoint_sha256": "a" * 64, "adapter": "bdh_cl"}
    )
    assert got == [normalize(make_record(model_checkpoint_sha256="a" * 64))]

    got_by_protocol = store.query(
        {"protocol": "random-crop-cold", "suite": "continual-learning"}
    )
    assert got_by_protocol == [normalize(r) for r in sq]

    assert store.query({"suite": "continual-learning"}) == sq


def test_query_no_filters_returns_all(tmp_path):
    store = Store(tmp_path)
    records = store.put([make_record(), make_record(value=0.71)])
    assert store.query() == records


def test_query_limit(tmp_path):
    store = Store(tmp_path)
    store.put([make_record()] * 3)
    assert len(store.query(limit=2)) == 2


def test_schema_rejects_missing_empty_identity_fields(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValidationError):
        store.put([make_record(model_checkpoint_sha256="")])
    with pytest.raises(ValidationError):
        store.put([make_record(protocol="   ")])
    with pytest.raises(ValidationError):
        store.put([make_record(model_checkpoint_sha256="") ])
    with pytest.raises(ValidationError):
        rec = make_record()
        del rec["model_checkpoint_sha256"]
        store.put([rec])
    with pytest.raises(ValidationError):
        rec = make_record()
        del rec["protocol"]
        store.put([rec])
    assert store.query() == []


def test_schema_matches_plan_fields_and_rejects_unknown(tmp_path):
    assert list(RECORD_FIELDS) == [
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
        "runtime_sha256",
        "seed",
        "artifacts",
    ]
    store = Store(tmp_path)
    with pytest.raises(ValidationError):
        store.put([make_record(bogus_field=1)])


def test_round_trip_preserves_optional_defaults(tmp_path):
    store = Store(tmp_path)
    [record] = store.put([make_record(host="bench-host")])
    got = store.query()[0]
    assert got["host"] == "bench-host"
    assert got["artifacts"] == ["/tmp/run/out.jsonl"]
    assert got["script_sha256"] == "b" * 64
    assert got["n"] == 100


def test_append_only_immutable_run_files(tmp_path):
    store = Store(tmp_path)
    store.put([make_record()])
    first_file = next((store.root / "runs").glob("RUN-*.json"))

    store.put([make_record(value=0.71)])
    run_files = sorted((store.root / "runs").glob("RUN-*.json"))
    assert len(run_files) == 2

    before = first_file.read_bytes()
    first_file.write_bytes(b"corrupted")
    with pytest.raises((json.JSONDecodeError, KeyError)):
        store.query()
    first_file.write_bytes(before)
    assert len(store.query()) == 2


def test_invalid_filter_field_rejected(tmp_path):
    store = Store(tmp_path)
    store.put([make_record()])
    with pytest.raises(ValueError, match="cannot filter on 'artifacts'"):
        store.query({"artifacts": ["x"]})