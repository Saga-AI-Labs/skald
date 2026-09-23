"""Tests for the runtime-manifest facet (``runtime_sha256``).

Spec: ``docs/plans/2026-09-23_runtime-manifest-spec.md``. Covers canonical
derivation, schema validation, store migration + filtering, and the
``cross-runtime-checkpoint`` anomaly extension.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

import store
from store.runtime import (
    canonical_json,
    collect_manifest,
    digest_manifest,
    runtime_digest,
)
from store.schema import FILTERABLE, RECORD_FIELDS, ValidationError, normalize
from surfaces.spec import (
    ANOMALY_RULE,
    RUNTIME_ANOMALY_RULE,
    flag_anomalies,
)


def test_record_fields_gains_facet_beside_script():
    assert "runtime_sha256" in RECORD_FIELDS
    assert "runtime_sha256" in FILTERABLE  # facet you filter on, not key on
    assert RECORD_FIELDS.index("runtime_sha256") == (
        RECORD_FIELDS.index("script_sha256") + 1
    )


def test_canonical_form_is_order_and_none_insensitive():
    a = {"served": {"server": "uvicorn", "models": ["b", "a"]},
         "local": {"os": "Linux", "arch": "x86_64", "python": "3.12.3"},
         "extra": None}
    b = {"local": {"python": "3.12.3", "arch": "x86_64", "os": "Linux"},
         "served": {"models": ["a", "b"], "server": "uvicorn"}}
    assert canonical_json(a) == canonical_json(b)
    assert digest_manifest(a) == digest_manifest(b)
    assert len(digest_manifest(a)) == 64


def test_empty_manifest_digests_to_none():
    assert canonical_json({}) is None
    assert digest_manifest({}) is None
    assert digest_manifest({"served": None, "extra": None}) is None


def test_collect_manifest_local_facts_exclude_hostname():
    import platform
    import socket

    manifest = collect_manifest()
    assert manifest["local"]["python"] == platform.python_version()
    assert manifest["local"]["os"] == platform.system()
    assert manifest["local"]["arch"] == platform.machine()
    assert "served" not in manifest and "extra" not in manifest
    assert socket.gethostname() not in json.dumps(manifest)


def test_collect_manifest_served_and_extra():
    manifest = collect_manifest(
        served={"models": ["m"], "server": "uvicorn"},
        extra={"vendor_pin": "abc123"},
    )
    digest = digest_manifest(manifest)
    assert digest == runtime_digest(
        served={"models": ["m"], "server": "uvicorn"},
        extra={"vendor_pin": "abc123"},
    )


def test_normalize_defaults_and_validates_digest():
    minimal = {
        "model_checkpoint_sha256": "c" * 64,
        "adapter": "a",
        "suite": "s",
        "task": "t",
        "metric": "m",
        "value": 1.0,
        "protocol": "p",
    }
    out = normalize(dict(minimal))
    assert out["runtime_sha256"] is None
    assert list(out) == list(RECORD_FIELDS)

    digest = "d" * 64
    assert normalize({**minimal, "runtime_sha256": digest})["runtime_sha256"] == digest

    for bad in ["not-a-digest", "D" * 64, "d" * 63, "d" * 65, 42, True]:
        with pytest.raises(ValidationError, match="runtime_sha256"):
            normalize({**minimal, "runtime_sha256": bad})


def _record(checkpoint, protocol, runtime):
    return {
        "model_checkpoint_sha256": checkpoint,
        "adapter": "a",
        "suite": "s",
        "task": "t",
        "metric": "m",
        "value": 1.0,
        "protocol": protocol,
        "runtime_sha256": runtime,
    }


def test_anomaly_extension_flags_runtime_divergence():
    records = [
        _record("c1", "p", "r1"),
        _record("c1", "p", "r2"),
        _record("c2", "p", "r1"),
        _record("c3", "p1", None),
        _record("c3", "p2", None),  # protocol rule still fires without runtimes
        _record("c4", "p", None),
        _record("c4", "p", "r1"),  # unknown + known is not divergence
    ]
    anomalies = flag_anomalies(records)
    by_reason = {}
    for a in anomalies:
        by_reason.setdefault(a["reason"], []).append(a)

    assert [a["model_checkpoint_sha256"] for a in by_reason[ANOMALY_RULE]] == ["c3"]
    runtime_flags = by_reason[RUNTIME_ANOMALY_RULE]
    assert len(runtime_flags) == 1
    entry = runtime_flags[0]
    assert entry["model_checkpoint_sha256"] == "c1"
    assert entry["protocol"] == "p"
    assert entry["runtimes"] == ["r1", "r2"]
    assert entry["runtime_count"] == 2


def test_anomaly_extension_is_deterministic():
    records = [_record("c1", "p", "r2"), _record("c1", "p", "r1")]
    assert flag_anomalies(records) == flag_anomalies(list(reversed(records)))


def test_store_migrates_pre_facet_database(tmp_path, monkeypatch):
    import store as store_mod

    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "oldstore"))
    root = tmp_path / "oldstore"
    root.mkdir(parents=True)
    runs = root / "runs"
    runs.mkdir()

    # A store file as created before the facet existed: no runtime column.
    db = sqlite3.connect(root / "index.sqlite3")
    db.execute(
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, file_path TEXT NOT NULL,"
        " created_at TEXT NOT NULL, record_count INTEGER NOT NULL)"
    )
    db.execute(
        "CREATE TABLE record_index (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " run_id TEXT NOT NULL, row_id INTEGER NOT NULL,"
        " model_checkpoint_sha256 TEXT NOT NULL, adapter TEXT NOT NULL,"
        " suite TEXT NOT NULL, task TEXT NOT NULL, metric TEXT NOT NULL,"
        " value REAL NOT NULL, n INTEGER, ci_low REAL, ci_high REAL,"
        " protocol TEXT NOT NULL, created_at TEXT NOT NULL, host TEXT,"
        " script_sha256 TEXT, seed TEXT,"
        " artifacts TEXT NOT NULL DEFAULT '[]', UNIQUE (run_id, row_id))"
    )
    run_id = "RUN-legacy"
    record = _record("c" * 64, "p", None)
    (runs / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "created_at": "2026-01-01T00:00:00Z",
        "records": [{**record, "created_at": "2026-01-01T00:00:00Z",
                     "host": "h", "script_sha256": None, "seed": None,
                     "n": None, "ci_low": None, "ci_high": None,
                     "artifacts": []}],
    }))
    db.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?)",
        (run_id, f"runs/{run_id}.json", "2026-01-01T00:00:00Z", 1),
    )
    db.execute(
        "INSERT INTO record_index (run_id, row_id, model_checkpoint_sha256,"
        " adapter, suite, task, metric, value, protocol, created_at)"
        " VALUES (?, 0, ?, 'a', 's', 't', 'm', 1.0, 'p',"
        " '2026-01-01T00:00:00Z')",
        (run_id, "c" * 64),
    )
    db.commit()
    db.close()

    # Opening the store migrates the index; the legacy file reads back exact.
    back = store.query({})
    assert len(back) == 1
    assert back[0]["runtime_sha256"] is None
    assert set(back[0]) == set(RECORD_FIELDS)

    digest = "e" * 64
    stored = store.put([_record("c" * 64, "p", digest)])
    assert stored[0]["runtime_sha256"] == digest
    assert store.query({"runtime_sha256": digest}) == stored


def test_openai_compat_records_carry_runtime_digest():
    from adapters.openai_compat import OpenAICompatAdapter
    from tests.test_openai_compat import MMLU_ITEMS, _Stub

    import threading
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    config = {"model": "stub-model", "mmlu_items": MMLU_ITEMS, "num_fewshot": 1}
    try:
        records = OpenAICompatAdapter().run(base, "mmlu", config)
        again_records = OpenAICompatAdapter().run(base, "mmlu", config)
    finally:
        server.shutdown()
    # mmlu now emits accuracy + answerable + budget_starved; the digest is
    # carried on every record, so check them all.
    assert [r["metric"] for r in records] == [
        "accuracy", "answerable", "budget_starved"
    ]
    for record, again in zip(records, again_records):
        digest = record["runtime_sha256"]
        assert digest is not None and len(digest) == 64
        # Served facts are folded in: the digest differs from a local-only one
        # and is stable across runs against the same endpoint.
        assert digest != runtime_digest()
        assert again["runtime_sha256"] == digest


def test_null_model_records_carry_no_runtime():
    from adapters.null_model import NullModelAdapter
    from tests.test_null_model import MMLU_ITEMS

    (record,) = NullModelAdapter().run("label", "mmlu", {"mmlu_items": MMLU_ITEMS})
    assert record["runtime_sha256"] is None
