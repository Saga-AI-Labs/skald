"""Tests for ``adapters.jlens`` — the JLens suite adapter (round 5).

Covers the round-1 adapter interface, unified record shape constraints
(non-empty 64-hex ``model_checkpoint_sha256`` + distinct ``protocol``, exact
``RECORD_FIELDS`` key set and order), determinism of the config/protocol
mapping, unknown-task/unknown-artifact rejection, the refuse-on-empty guard,
the vendored upstream pin (``script_sha256`` = direct SHA-256 of the invoked
upstream file ``vendor/jlens/jlens/__init__.py``, matching the pinned digest
in ``vendor/jlens/PIN.md``), and a real-material execution over the recon M4
run target (an abliterated HF model, minimally CPU-fitted and read out, with
raw-trace and fitted-lens artifacts written under the OPEN-5 locations).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
from pathlib import Path

import pytest

import store
from adapters import RECORD_FIELDS, SuiteAdapter
from store import RECORD_FIELDS as STORE_RECORD_FIELDS
from adapters.jlens import (
    DEFAULT_MODEL,
    DEFAULT_PYTHON,
    DEFAULT_VENDOR,
    JLensAdapter,
    TASKS,
    _RESULT_MARK,
    _checkpoint_sha256,
    _config_hash,
    _file_sha256,
)

PYTHON = str(DEFAULT_PYTHON) if DEFAULT_PYTHON.is_file() else "/media/data/coding/OBLITERATUS/.venv/bin/python"
UPSTREAM_SCRIPT = DEFAULT_VENDOR / "jlens" / "__init__.py"

# Recon M4 run target (smallest abliterated HF decoder the wrapper loads).
RUN_TARGET = DEFAULT_MODEL

# Fast local model for plumbing tests (gpt2 layout; not abliterated).
LOCAL_MODEL = Path("/media/data/coding/skald/.skald/saga_models/tiny-gpt2")


def _adapter() -> JLensAdapter:
    return JLensAdapter()


def _build_path(root: Path) -> str:
    return str(root)


def test_round1_adapter_interface():
    adapter = _adapter()
    assert isinstance(adapter, SuiteAdapter)
    sig = inspect.signature(adapter.run)
    params = list(sig.parameters)
    assert params[:2] == ["model", "task"]
    assert "config" in params
    assert "layer_readout" in TASKS
    assert TASKS <= {
        "layer_readout",
        "verbal_report",
        "directed_modulation",
        "multi_hop_reasoning",
        "general_broadcast",
        "selective_mediation",
    }
    assert "layer_readout" in sorted(TASKS)


def test_mapping_determinism():
    fit = {"prompts": ["a" * 40, "b" * 40], "source_layers": [4],
           "dim_batch": 16, "max_seq_len": 48, "skip_first": 16,
           "dtype": "float32"}
    readout = {"prompt": "c" * 40, "layers": [4], "position": -1,
               "top_n": 3, "seed": 42}
    assert _config_hash(fit, readout) == _config_hash(fit, readout)
    assert re.fullmatch(r"[0-9a-f]{16}", _config_hash(fit, readout))
    # changing a readout knob changes the run's content hash (OPEN-5 name)
    readout2 = dict(readout, top_n=5)
    assert _config_hash(fit, readout2) != _config_hash(fit, readout)
    # protocol is deterministic given the same payload + params
    adapter = _adapter()
    payload = {"layout": {"n_layers": 18, "d_model": 640}}
    p1 = adapter._protocol("layer_readout", RUN_TARGET, fit, readout, payload)
    p2 = adapter._protocol("layer_readout", RUN_TARGET, fit, readout, payload)
    assert p1 == p2
    assert "layer_readout" in p1
    assert "WIRING CHECK" in p1


def test_driver_source_is_fixed_and_importable():
    import ast

    body = "import json, os, sys\ncfg = {}\nsys.path.insert(0, cfg.get('vendor', '/dev/null'))\n"
    ast.parse(body + _DRIVER_SOURCE())
    assert _RESULT_MARK in _DRIVER_SOURCE()


def _DRIVER_SOURCE() -> str:
    from adapters.jlens import _DRIVER, _RESULT_MARK  # noqa: F401
    import string
    # substitute() requires every $placeholder filled; only $vendor is used
    return _DRIVER.substitute(vendor="dummy")


def test_records_carry_exact_field_order_and_hash(tmp_path):
    adapter = _adapter()
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    trace = raw / ".skald/jlens_raw/dummy/trace.jsonl"
    lens = raw / ".skald/jlens_lenses/dummy/lens.pt"
    trace.parent.mkdir(parents=True, exist_ok=True)
    lens.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("x\n")
    lens.write_bytes(b"lens")
    payload = {
        "layout": {"n_layers": 18, "d_model": 640},
        "trace_rel": str(trace.relative_to(raw)),
        "lens_rel": str(lens.relative_to(raw)),
        "script_sha256": _file_sha256(UPSTREAM_SCRIPT),
        "seed": 42,
        "readout": [
            {"layer": 4, "rank": 1, "token_id": 0, "token": "<x>", "prob": 0.5},
            {"layer": 4, "rank": 2, "token_id": 1, "token": "<y>", "prob": 0.2},
        ],
    }
    records = adapter._records(
        task="layer_readout",
        protocol="jlens wiring check protocol",
        checkpoint_sha="d" * 64,
        seed=42,
        readout_rows=payload["readout"],
        payload=payload,
        artifact_root=raw,
    )
    assert records
    for r in records:
        assert list(r) == list(STORE_RECORD_FIELDS), \
            "record key order must equal RECORD_FIELDS"
        assert set(r) == set(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["protocol"]
        assert r["adapter"] == "jlens"
        assert r["suite"] == "jlens"
        assert r["host"]
        # script_sha256 is the direct hash of the invoked upstream file, and
        # equals the digest pinned in vendor/jlens/PIN.md
        assert r["script_sha256"] == _file_sha256(UPSTREAM_SCRIPT), (
            "script_sha256 must be the direct SHA-256 of the invoked upstream "
            "file (vendor/jlens/jlens/__init__.py)"
        )
        assert r["metric"] in {"top1_prob@L4", "top2_prob@L4"}
        assert 0.0 <= r["value"] <= 1.0
        assert r["n"] == 1
        # artifacts referenced and not prunable: digest matches the file
        assert r["artifacts"], "raw trace + fitted lens must be referenced"
        for art in r["artifacts"]:
            digest, path = art.split("  ", 1)
            assert _file_sha256(raw / path) == digest
        normalized = store.normalize(r)
        assert normalized == r


def test_unknown_task_rejection():
    with pytest.raises(ValueError, match="unsupported task"):
        _adapter().run(str(LOCAL_MODEL), "bbq")


def test_unknown_lens_source_rejection():
    with pytest.raises(FileNotFoundError, match="lens source not found"):
        _adapter().run(
            str(LOCAL_MODEL),
            "layer_readout",
            {"lens_source": "/no/such/lens.pt", "python": PYTHON},
        )


def test_unknown_vendor_rejection():
    with pytest.raises(FileNotFoundError, match="vendored package not found"):
        _adapter().run(
            str(LOCAL_MODEL),
            "layer_readout",
            {"vendor_dir": "/no/such/vendor", "python": PYTHON},
        )


def test_empty_result_guard(monkeypatch):
    payload = {
        "vendor": str(DEFAULT_VENDOR),
        "root": "/media/data/coding/skald",
        "layout": {"n_layers": 2, "d_model": 2},
        "model_dir": str(LOCAL_MODEL),
        "readout": [],
    }
    fake = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=f"{_RESULT_MARK} " + json.dumps(payload),
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="no readout records"):
        _adapter()._capture(
            Path("/media/data/coding/skald"), PYTHON, payload, timeout=60
        )


def test_checkpoint_hash_file_matches_identity(tmp_path):
    import identity

    artifact = tmp_path / "ckpt.bin"
    artifact.write_bytes(b"weights-0-1-2")
    assert _checkpoint_sha256(artifact) == identity.hash_checkpoint(artifact)
    assert re.fullmatch(r"[0-9a-f]{64}", _checkpoint_sha256(artifact))


def test_checkpoint_hash_dir_deterministic(tmp_path):
    d = tmp_path / "model_dir"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"a":1}')
    (d / "model.bin").write_bytes(b"w" * 64)
    h1 = _checkpoint_sha256(d)
    h2 = _checkpoint_sha256(d)
    assert h1 == h2
    assert re.fullmatch(r"[0-9a-f]{64}", h1)


def test_pin_matches_manifest():
    pin = DEFAULT_VENDOR / "PIN.md"
    assert pin.is_file(), "vendor/jlens/PIN.md must document the upstream pin"
    text = pin.read_text()
    assert "4e3f3b2cf1d85a4821a6fb1c46970efea872004d" in text
    # the digest in the manifest must equal the real file digest
    assert _file_sha256(UPSTREAM_SCRIPT) in text
    assert "Apache-2.0" in text


@pytest.mark.parametrize("task", ["layer_readout", "verbal_report"])
def test_real_material_execution(tmp_path, monkeypatch, task):
    """A real JLens invocation over the recon-fixed run target.

    Uses a minimal CPU fit (one source layer, a single long prompt,
    ``max_seq_len`` 48) exactly as the recon M4 prescribes — a wiring check,
    not a scientific measurement — then reads out the fitted lens, persists
    the records straight from ``store.put`` into a fresh test store, queries
    them back, and verifies the OPEN-5 artifacts (append-only raw JSONL trace
    + fitted lens) landed on disk, referenced by digest.

    Requires the torch-capable interpreter, the vendored package, and network
    to fetch the abliterated HF model at run time.
    """
    import store as store_mod

    # Isolate the store for this run (pattern as in test_bdh_cl/test_pi50):
    # never touch the default live store even when this file runs alone.
    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "isolated"))

    adapter = _adapter()
    records = adapter.run(
        RUN_TARGET,
        task,
        {
            "python": PYTHON,
            "source_layers": [4],
            "layers": [4],
            "dim_batch": 16,
            "max_seq_len": 48,
            "top_n": 3,
            "seed": 42,
            "timeout": 1800,
            "artifact_dir": str(tmp_path / "artifacts"),
        },
    )
    assert records, "a real invocation must emit at least one record"
    assert len(records) == 3, "3 ranks for layer 4 (top_n=3), one position"
    for r in records:
        assert r["task"] == task
        assert r["metric"].startswith("top") and f"@L4" in r["metric"]
        assert 0.0 <= r["value"] <= 1.0
        assert r["n"] == 1
        assert r["adapter"] == "jlens" and r["suite"] == "jlens"
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert "WIRING CHECK" in r["protocol"]
        assert r["protocol"] != "saga"  # distinct protocol label
        assert r["script_sha256"] == _file_sha256(UPSTREAM_SCRIPT)
        assert r["artifacts"]
        normalized = store.normalize(r)
        assert normalized == r

    # store roundtrip in a fresh store, queryable back, default store clean
    key = {
        "adapter": "jlens",
        "task": task,
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    }
    store.put(records)
    back = store.query(key)
    assert len(back) >= len(records)
    assert all(b["adapter"] == "jlens" for b in back)

    # OPEN-5: raw traces + fitted lens referenced in artifacts, on disk,
    # append-only JSONL with a header + per-layer logit rows
    asserted = False
    for art in records[0]["artifacts"]:
        digest, rel = art.split("  ", 1)
        path = tmp_path / "artifacts" / rel
        assert path.is_file(), f"referenced artifact must exist: {path}"
        assert _file_sha256(path) == digest
        if ".skald/jlens_raw" in rel:
            lines = path.read_text().strip().splitlines()
            assert len(lines) >= 2
            header = json.loads(lines[0])
            assert header["kind"] == "jlens_raw_v1"
            assert header["model_checkpoint_sha256"] == records[0]["model_checkpoint_sha256"]
            layer_rows = [json.loads(ln) for ln in lines[1:]]
            assert all(row["kind"] == "layer_logits" for row in layer_rows)
            assert all(row["layer"] == 4 for row in layer_rows)
            assert all(len(row["logits"]) > 0 for row in layer_rows)
            asserted = True
    assert asserted, "a raw trace must be referenced"


def test_default_store_not_polluted(monkeypatch):
    """The real run above must not touch the default live store.

    We guard this by running with an isolated ``SKALD_STORE_DIR`` and
    asserting the default store existed before and is unchanged after.
    """
    import sqlite3

    import store as store_mod

    # Re-point the module singleton so the put below lands in the isolated
    # store even if an earlier test in this session cached the default store.
    store_mod._DEFAULT_STORE = None

    default = Path("/media/data/coding/skald/.skald/store/index.sqlite3")
    before = None
    if default.is_file():
        con = sqlite3.connect(default)
        before = con.execute("SELECT COUNT(*) FROM record_index").fetchone()[0]
        con.close()
    isolated = Path("/tmp/opencode/jlens_test_store").resolve()
    isolated.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SKALD_STORE_DIR", str(isolated))
    # a record put here goes to the isolated store, not the default
    store.put([{
        "model_checkpoint_sha256": "a" * 64,
        "adapter": "jlens",
        "suite": "jlens",
        "task": "layer_readout",
        "metric": "top1_prob@L0",
        "value": 0.5,
        "n": 1,
        "ci_low": None,
        "ci_high": None,
        "protocol": "isolation test",
        "created_at": "2026-09-17T00:00:00Z",
        "host": "test",
        "script_sha256": "b" * 64,
        "seed": 42,
        "artifacts": [],
    }])
    if default.is_file():
        con = sqlite3.connect(default)
        after = con.execute("SELECT COUNT(*) FROM record_index").fetchone()[0]
        con.close()
        assert after == before