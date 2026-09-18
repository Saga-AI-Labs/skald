"""Tests for ``adapters.atlas`` — the weight-atlas diff adapter.

All tests run against an in-process stub atlas API (stdlib ``http.server``
in a thread): no network, no scans. Live verification against the real
atlas box is a documented manual step (docs/usage.md), not part of this
suite.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import store
from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.atlas import AtlasAdapter, AtlasError

SHA = "a" * 64
JOB_A = "job-aaaa"
JOB_B = "job-bbbb"

DELTA_BODY = {
    "model_a": JOB_A,
    "model_b": JOB_B,
    "tier": "statistic_diff",
    "metric": "frobenius",
    "n_compared": 1200,
    "n_changed_above_5pct": 2,
    "summary": {
        "mean_change_pct": 0.0114,
        "max_change_pct": 0.0127,
        "most_affected_type": "ssm_out",
        "most_affected_layer_range": "15-48",
    },
    "rows": [
        {"tensor_name": "blk.18.ssm_out.weight", "layer": 18,
         "type": "ssm_out", "a": 1.0, "b": 1.0001,
         "abs_change": 0.0001, "pct_change": 0.01},
        {"tensor_name": "blk.34.ssm_out.weight", "layer": 34,
         "type": "ssm_out", "a": 2.0, "b": 2.0002,
         "abs_change": 0.0002, "pct_change": 0.01},
    ],
}


class _Stub(BaseHTTPRequestHandler):
    def _json(self, body, status=200):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/api/model/missing/delta"):
            self._json({"error": {"code": 404, "type": "model_not_found"}},
                       status=404)
        elif "/delta" in self.path:
            self._json(DELTA_BODY)
        else:
            self._json({"error": "nope"}, status=404)

    def log_message(self, *a):
        pass


@pytest.fixture()
def atlas_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _adapter():
    return AtlasAdapter()


def _cfg(atlas_url, **over):
    cfg = {"atlas_url": atlas_url, "atlas_job": JOB_A, "with_job": JOB_B}
    cfg.update(over)
    return cfg


def test_adapter_interface():
    adapter = _adapter()
    assert isinstance(adapter, SuiteAdapter)
    sig = inspect.signature(adapter.run)
    params = list(sig.parameters)
    assert params[:2] == ["model", "task"]
    assert "config" in params


def test_model_must_be_a_checkpoint_hash(atlas_url):
    with pytest.raises(ValueError, match="64-hex"):
        _adapter().run("not-a-hash", "diff", _cfg(atlas_url))


def test_unknown_task_is_rejected(atlas_url):
    with pytest.raises(ValueError, match="unsupported task"):
        _adapter().run(SHA, "scan", _cfg(atlas_url))


def test_mapping_is_never_guessed(atlas_url):
    with pytest.raises(ValueError, match="atlas_job"):
        _adapter().run(SHA, "diff", {"atlas_url": atlas_url})
    with pytest.raises(ValueError, match="atlas_job"):
        _adapter().run(
            SHA, "diff", {"atlas_url": atlas_url, "atlas_job": JOB_A}
        )


def test_diff_records(atlas_url):
    records = _adapter().run(SHA, "diff", _cfg(atlas_url))
    assert len(records) == 3 + 2  # 3 aggregates + 2 hotspots
    for r in records:
        assert set(r) == set(RECORD_FIELDS)
        assert r["adapter"] == "atlas" and r["suite"] == "atlas"
        assert r["task"] == "diff"
        assert r["model_checkpoint_sha256"] == SHA
        assert isinstance(r["value"], float)
    by_metric = {r["metric"]: r for r in records}
    assert by_metric["delta_mean_change_pct"]["value"] == 0.0114
    assert by_metric["delta_n_changed_above_5pct"]["value"] == 2.0
    assert by_metric["hotspot_rank1_pct_change"]["value"] == 0.01
    assert f"tensor::blk.18.ssm_out.weight" in by_metric["hotspot_rank1_pct_change"]["artifacts"]
    assert JOB_A in by_metric["hotspot_rank1_pct_change"]["protocol"]
    assert JOB_B in by_metric["hotspot_rank1_pct_change"]["protocol"]


def test_nulls_are_omitted_not_zero_filled(atlas_url, monkeypatch):
    import adapters.atlas as mod

    null_body = dict(DELTA_BODY, summary={"mean_change_pct": None,
                                          "max_change_pct": None},
                     n_changed_above_5pct=0,
                     rows=[dict(DELTA_BODY["rows"][0], pct_change=None)])
    monkeypatch.setattr(mod, "_get", lambda *a, **k: null_body)
    records = _adapter().run(SHA, "diff", _cfg("http://unused"))
    assert [r["metric"] for r in records] == ["delta_n_changed_above_5pct"]
    assert all(r["value"] is not None for r in records)


def test_missing_scan_maps_to_atlas_error(atlas_url):
    with pytest.raises(AtlasError, match="HTTP 404"):
        _adapter().run(
            SHA, "diff",
            _cfg(atlas_url, atlas_job="missing"),
        )


def test_unreachable_server_is_an_atlas_error():
    with pytest.raises(AtlasError, match="cannot reach"):
        _adapter().run(
            SHA, "diff",
            {"atlas_url": "http://127.0.0.1:1", "atlas_job": JOB_A,
             "with_job": JOB_B},
        )


def test_store_roundtrip_isolated(atlas_url, tmp_path, monkeypatch):
    import store as store_mod

    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "isolated"))
    records = _adapter().run(SHA, "diff", _cfg(atlas_url))
    stored = store.put(records)
    assert stored == records
    back = store.query({"adapter": "atlas"})
    assert len(back) == len(records)
