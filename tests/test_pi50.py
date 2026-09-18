"""Tests for ``adapters.pi50`` — the Pi-50 phase-1 suite adapter (round 2).

Covers the round-1 adapter interface, unified record shape constraints
(non-empty ``model_checkpoint_sha256`` + ``protocol``), canned parse of the
manifest-check verdict, real execution of the frozen Pi-50 instrument
``phase1_manifest.py --check``, and the store end-to-end persistence gate.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import store
from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.pi50 import Pi50Adapter, PROTOCOLS

REPO = Path("/media/data/coding/bdh-cl")
MANIFEST = REPO / "docs/PHASE1-MANIFEST.md"

FRESH_OUT = "manifest up to date\n"
STALE_OUT = "MANIFEST STALE - regenerate\n"


def _adapter():
    return Pi50Adapter()


def test_round1_adapter_interface():
    adapter = _adapter()
    assert isinstance(adapter, SuiteAdapter)
    sig = inspect.signature(adapter.run)
    params = list(sig.parameters)
    assert params[:2] == ["model", "task"]
    assert "config" in params


def test_manifest_check_canned_verdicts():
    adapter = _adapter()
    assert adapter._parse_manifest(FRESH_OUT, rc=0) == [
        {"metric": "manifest_current", "value": 1.0}
    ]
    assert adapter._parse_manifest(STALE_OUT, rc=1) == [
        {"metric": "manifest_current", "value": 0.0}
    ]
    with pytest.raises(RuntimeError, match="did not carry a recognised verdict"):
        adapter._parse_manifest("unrelated output\n", rc=0)


def test_manifest_check_records_have_required_fields():
    records = _adapter()._records(
        model=str(MANIFEST),
        task="manifest_check",
        repo=REPO,
        script=REPO / "scripts" / "pi50" / "phase1_manifest.py",
        protocol=PROTOCOLS["manifest_check"],
        seed=None,
        metrics=[{"metric": "manifest_current", "value": 0.0}],
        stdout=STALE_OUT,
        n_default=1,
    )
    assert records
    for r in records:
        assert set(r) == set(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["model_checkpoint_sha256"]
        assert r["protocol"]
        assert r["adapter"] == "pi50"
        assert r["suite"] == "pi50"
        assert r["host"]
        assert r["script_sha256"]


@pytest.mark.skipif(
    not (MANIFEST.is_file() and (REPO / "scripts/pi50/phase1_manifest.py").is_file()),
    reason="BDH-CL phase-1 manifest / instrument not present",
)
def test_manifest_check_real_execution():
    records = Pi50Adapter().run(
        str(MANIFEST), "manifest_check", {"repo": str(REPO)}
    )
    assert [r["metric"] for r in records] == ["manifest_current"]
    assert records[0]["value"] in (0.0, 1.0)
    for r in records:
        assert r["model_checkpoint_sha256"]
        assert r["protocol"] == PROTOCOLS["manifest_check"]
        assert r["created_at"]
        assert r["script_sha256"]


@pytest.mark.skipif(
    not (MANIFEST.is_file() and (REPO / "scripts/pi50/phase1_manifest.py").is_file()),
    reason="BDH-CL phase-1 manifest / instrument not present",
)
def test_store_end_to_end_persist_and_readback(tmp_path, monkeypatch):
    import store as store_mod

    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "isolated"))
    records = Pi50Adapter().run(
        str(MANIFEST), "manifest_check", {"repo": str(REPO)}
    )
    stored = store.put(records)
    assert stored == records
    key = {
        "adapter": "pi50",
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    }
    back = store.query(key)
    assert len(back) == len(records)
    assert all(r["protocol"] for r in back)
    assert all(r["value"] in (0.0, 1.0) for r in back)
    saved_runs = list((tmp_path / "isolated" / "runs").glob("*.json"))
    assert saved_runs, "per-run immutable JSON files must be persisted"