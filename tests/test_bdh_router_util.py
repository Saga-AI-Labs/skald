"""Tests for ``adapters.bdh_router_util`` — the §3.4 utilisation probe.

The torch helper runs in a subprocess that is not available in the skald
venv; these tests monkeypatch ``_run_helper`` and ``hash_checkpoint`` so
the adapter's orchestration, artifacts, manifests, and records are exercised
over canned helper JSON with the stdlib interpreter. Real-capture
integration stays a documented manual step (like bdh_cl's NEEDS_SUITE
pattern).

The §3.4 contract under test: at ``k_sparse_ratio <= 0`` the adapter must
raise (not applicable, never a stored zero); otherwise every record's
protocol must carry the measured ``k_sparse_ratio`` and width.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import store
from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.bdh_router_util import BdhRouterUtilAdapter
from adapters.utilization_stats import UTIL_METRICS

_64_HEX = "a" * 64


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "vendor-bdh"
    r.mkdir()
    (r / "PIN.md").write_text(f"Pinned commit: `{_64_HEX}`")
    return r


@pytest.fixture()
def adapter():
    return BdhRouterUtilAdapter()


@pytest.fixture()
def cfg(tmp_path):
    return {"python": "/usr/bin/python3", "seed": 7,
            "artifact_dir": str(tmp_path / "artifacts")}


def _checkpoint(tmp_path, name="ckpt.pt"):
    ckpt = tmp_path / name
    ckpt.write_text("fake checkpoint bytes")
    return ckpt


def _canned(width=8, ratio=0.25):
    def layers(counts_x, counts_y, positions):
        assert len(counts_x) == len(counts_y) == width
        return [
            {"layer": 0, "side": "x", "counts": counts_x},
            {"layer": 0, "side": "y", "counts": counts_y},
        ]

    return {
        "not_applicable": False,
        "config": {
            "k_sparse_ratio": ratio,
            "k_absolute": max(1, int(ratio * width)),
            "width": width,
            "n_layer": 1,
            "n_head": 1,
            "block_size": 128,
        },
        "domains": {
            "general": {
                "positions": 100,
                "layers": layers([50] * 2 + [0] * 6, [100] + [0] * 7, 100),
            },
            "legal": {
                "positions": 40,
                "layers": layers([20] * 2 + [0] * 6, [10] * 4 + [0] * 4, 40),
            },
        },
    }


def test_adapter_interface(adapter):
    assert isinstance(adapter, SuiteAdapter)
    assert isinstance(adapter.run, object)


def test_requires_repo_dir(tmp_path, adapter):
    with pytest.raises(FileNotFoundError, match="repo not found"):
        adapter.run("x", "router_utilization",
                    {"repo": str(tmp_path / "missing")})


def test_requires_torch_python(tmp_path, adapter, repo):
    ckpt = _checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="torch python not found"):
        adapter.run(str(ckpt), "router_utilization",
                    {"repo": str(repo)})


def test_unknown_task_rejected(adapter, repo, cfg):
    with pytest.raises(ValueError, match="unsupported task"):
        adapter.run("x", "router", {"repo": str(repo), **cfg})


def test_requires_checkpoint(adapter, repo, cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        adapter.run(str(tmp_path / "no-ckpt.pt"), "router_utilization",
                    {"repo": str(repo), **cfg})


def test_rejects_negative_threshold(adapter, repo, cfg, tmp_path,
                                    monkeypatch):
    ckpt = _checkpoint(tmp_path)
    monkeypatch.setattr(
        "adapters.bdh_router_util._run_helper", lambda *a, **k: _canned()
    )
    with pytest.raises(ValueError, match="mass_threshold"):
        adapter.run(str(ckpt), "router_utilization",
                    {"repo": str(repo), **cfg, "mass_threshold": -0.1,
                     "contexts": [{"domain": "general", "text": "hi"}]})


def test_zero_gate_raises_not_applicable(adapter, repo, cfg, tmp_path,
                                         monkeypatch):
    """k_sparse_ratio == 0: no records, ever — NA is not a zero."""
    ckpt = _checkpoint(tmp_path)
    monkeypatch.setattr(
        "adapters.bdh_router_util._run_helper",
        lambda *a, **k: {
            "not_applicable": True,
            "k_sparse_ratio": 0.0,
            "reason": "plain ReLU, no gate",
        },
    )
    with pytest.raises(ValueError, match="not applicable"):
        adapter.run(str(ckpt), "router_utilization",
                    {"repo": str(repo), **cfg,
                     "contexts": [{"domain": "general", "text": "hi"}]})


def test_utilization_writes_artifacts_and_records(
    adapter, repo, cfg, tmp_path, monkeypatch
):
    ckpt = _checkpoint(tmp_path)
    canned = _canned()
    monkeypatch.setattr(
        "adapters.bdh_router_util._run_helper", lambda *a, **k: canned
    )
    monkeypatch.setattr(
        "adapters.bdh_router_util.hash_checkpoint", lambda p: _64_HEX
    )

    records = adapter.run(
        str(ckpt), "router_utilization",
        {"repo": str(repo), **cfg,
         "contexts": [{"domain": "general", "text": "hello"},
                      {"domain": "legal", "text": "party"}]},
    )
    # 2 gates x 2 domains x 6 metrics, plus per gate: 5 pooled + 5
    # histogram buckets + 1 spread = 24 + 22
    assert len(records) == 46
    for r in records:
        assert set(r) == set(RECORD_FIELDS)
        assert r["adapter"] == "bdh_router_util"
        assert r["task"] == "router_utilization"
        assert r["model_checkpoint_sha256"] == _64_HEX
        assert "k_sparse_ratio=0.25" in r["protocol"]
        assert "width 8" in r["protocol"]
        assert "mass_threshold=0.0" in r["protocol"]
    by_metric = {r["metric"]: r for r in records}

    # general/x gate: two slots share all mass -> dead 6/8, entropy ln 2
    assert by_metric["util_dead_slot_fraction:L0x@general"]["value"] == \
        pytest.approx(0.75)
    assert by_metric["util_gate_entropy_nats:L0x@general"]["value"] == \
        pytest.approx(math.log(2))
    assert by_metric["util_positions:L0x@general"]["value"] == \
        pytest.approx(100)
    assert by_metric["util_positions:L0x@general"]["n"] == 100
    # general/y gate: single live slot -> dead 7/8, entropy 0
    assert by_metric["util_dead_slot_fraction:L0y@general"]["value"] == \
        pytest.approx(7 / 8)
    assert by_metric["util_gate_entropy_nats:L0y@general"]["value"] == \
        pytest.approx(0.0)
    # spread across domains is a real difference, not a constant
    spread = by_metric["util_gate_entropy_spread:L0y"]["value"]
    assert spread == pytest.approx(
        abs(by_metric["util_gate_entropy_nats:L0y@general"]["value"]
            - by_metric["util_gate_entropy_nats:L0y@legal"]["value"]))
    assert spread > 0.0
    # histogram buckets sum to 1 per pooled gate
    pooled_buckets = sum(
        by_metric[f"util_load_hist_b{i}:L0x@pooled"]["value"]
        for i in range(5))
    assert pooled_buckets == pytest.approx(1.0)
    # pooled gate carries the pooled coverage
    assert by_metric["util_gate_entropy_spread:L0x"]["n"] == 140

    artifact_dir = Path(cfg["artifact_dir"])
    histograms = json.loads(
        (artifact_dir / "histograms.json").read_text())
    assert histograms["checkpoint_sha256"] == _64_HEX
    assert histograms["capture_config"]["k_sparse_ratio"] == 0.25
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    assert manifest["vendor_pin"] == _64_HEX
    assert manifest["domains"] == {"general": 100, "legal": 40}
    assert manifest["runtime_sha256"] is not None
    assert manifest["seed"] == 7


def test_empty_capture_refuses_to_store(adapter, repo, cfg, tmp_path,
                                        monkeypatch):
    ckpt = _checkpoint(tmp_path)
    canned = _canned()
    canned["domains"] = {}
    monkeypatch.setattr(
        "adapters.bdh_router_util._run_helper", lambda *a, **k: canned
    )
    monkeypatch.setattr(
        "adapters.bdh_router_util.hash_checkpoint", lambda p: _64_HEX
    )
    with pytest.raises(RuntimeError, match="no domains"):
        adapter.run(str(ckpt), "router_utilization",
                    {"repo": str(repo), **cfg,
                     "contexts": [{"domain": "general", "text": "hi"}]})


def test_store_roundtrip_isolated(adapter, repo, cfg, tmp_path, monkeypatch):
    """Persisted §3.4 records query back with their gate identity."""
    ckpt = _checkpoint(tmp_path)
    monkeypatch.setattr(
        "adapters.bdh_router_util._run_helper", lambda *a, **k: _canned()
    )
    monkeypatch.setattr(
        "adapters.bdh_router_util.hash_checkpoint", lambda p: _64_HEX
    )
    monkeypatch.setattr(
        "adapters.bdh_router_util.runtime_digest", lambda: "d" * 64
    )
    import store as store_mod

    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "isolated"))
    records = adapter.run(
        str(ckpt), "router_utilization",
        {"repo": str(repo), **cfg,
         "contexts": [{"domain": "general", "text": "hello"},
                      {"domain": "legal", "text": "party"}]},
    )
    stored = store.put(records)
    assert stored == records
    back = store.query({"adapter": "bdh_router_util"})
    assert len(back) == len(records)
    metrics = {r["metric"] for r in back}
    assert "util_dead_slot_fraction:L0x@general" in metrics
    assert "util_gate_entropy_spread:L0y" in metrics
