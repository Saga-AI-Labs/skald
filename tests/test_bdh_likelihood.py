"""Tests for ``adapters.bdh_likelihood`` — Path A (weights-only) parity.

The torch helpers run in a subprocess that is not available in the skald
venv; these tests monkeypatch ``_run_helper`` and ``hash_checkpoint`` so the
adapter's orchestration, bundles, manifests, and records are exercised over
canned helper JSON with the stdlib interpreter. Real-capture integration
stays a documented manual step (like bdh_cl's NEEDS_SUITE pattern).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import store
from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.bdh_likelihood import (
    BdhLikelihoodAdapter,
    KL_METRICS,
    _vendor_pin,
    load_contexts,
)

_64_HEX = "a" * 64


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "vendor-bdh"
    r.mkdir()
    return r


@pytest.fixture()
def adapter():
    return BdhLikelihoodAdapter()


@pytest.fixture()
def cfg(tmp_path):
    return {"python": "/usr/bin/python3", "seed": 7}


def _checkpoint(tmp_path, name="ref.pt"):
    ckpt = tmp_path / name
    ckpt.write_text("fake checkpoint bytes")
    return ckpt


def test_adapter_interface(adapter):
    assert isinstance(adapter, SuiteAdapter)
    assert isinstance(adapter.run, object)


def test_requires_repo_dir(tmp_path, adapter):
    with pytest.raises(FileNotFoundError, match="repo not found"):
        adapter.run("x", "capture_reference", {"repo": str(tmp_path / "missing")})


def test_requires_torch_python(tmp_path, adapter, repo):
    ckpt = _checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="torch python not found"):
        adapter.run(str(ckpt), "capture_reference", {"repo": str(repo)})


def test_unknown_task_rejected(adapter, repo, cfg):
    with pytest.raises(ValueError, match="unsupported task"):
        adapter.run("x", "gsm8k", {"repo": str(repo), **cfg})


def test_capture_requires_checkpoint(adapter, repo, cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="reference checkpoint"):
        adapter.run(str(tmp_path / "no-ckpt.pt"), "capture_reference",
                    {"repo": str(repo), **cfg})


def test_capture_writes_bundle_and_records(
    adapter, repo, cfg, tmp_path, monkeypatch
):
    ckpt = _checkpoint(tmp_path)
    bundle = tmp_path / "bundle"
    canned = {
        "config": {
            "stored_dtype": "float16",
            "selfcheck_max_abs_logit_diff": 1.3e-6,
        },
        "domains": {"general": {"positions": 100}, "legal": {"positions": 40}},
    }
    monkeypatch.setattr(
        "adapters.bdh_likelihood._run_helper",
        lambda *a, **k: canned,
    )
    monkeypatch.setattr(
        "adapters.bdh_likelihood.hash_checkpoint", lambda p: _64_HEX
    )
    (repo / "PIN.md").write_text(f"Pinned commit: `{_64_HEX}`")

    records = adapter.run(
        str(ckpt), "capture_reference",
        {"repo": str(repo), "bundle_dir": str(bundle), **cfg,
         "contexts": [{"domain": "general", "text": "hello"},
                      {"domain": "legal", "text": "party"}]},
    )
    assert len(records) == 4  # 2 domains × (positions, selfcheck)
    for r in records:
        assert set(r) == set(RECORD_FIELDS)
        assert r["adapter"] == "bdh_likelihood"
        assert r["task"] == "capture_reference"
        assert r["model_checkpoint_sha256"] == _64_HEX
        assert r["n"] in (100, 40)
    metrics = {r["metric"] for r in records}
    assert metrics == {
        "reference_positions", "reference_selfcheck_max_abs_logit_diff",
    }

    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["created_by"] == "bdh_likelihood capture_reference"
    assert manifest["reference_checkpoint"] == str(ckpt)
    assert manifest["reference_checkpoint_sha256"] == _64_HEX
    assert manifest["vendor_pin"] == _64_HEX
    assert manifest["bundle_config"]["stored_dtype"] == "float16"
    assert manifest["runtime_sha256"] is not None
    assert manifest["seed"] == 7


def test_capture_fails_loudly_without_selfcheck_config(
    adapter, repo, cfg, tmp_path, monkeypatch
):
    ckpt = _checkpoint(tmp_path)
    monkeypatch.setattr(
        "adapters.bdh_likelihood._run_helper",
        lambda *a, **k: {
            "config": {"stored_dtype": "float16"},  # selfcheck key missing
            "domains": {"general": {"positions": 10}},
        },
    )
    monkeypatch.setattr(
        "adapters.bdh_likelihood.hash_checkpoint", lambda p: _64_HEX
    )
    with pytest.raises(KeyError, match="selfcheck_max_abs_logit_diff"):
        adapter.run(str(ckpt), "capture_reference",
                    {"repo": str(repo), **cfg,
                     "contexts": [{"domain": "general", "text": "hi"}]})


def test_parity_requires_candidate_and_bundle(adapter, repo, cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="candidate checkpoint"):
        adapter.run(str(tmp_path / "no-cand.pt"), "likelihood_parity",
                    {"repo": str(repo), **cfg})
    cand = _checkpoint(tmp_path, "cand.pt")
    with pytest.raises(FileNotFoundError, match="reference_bundle"):
        adapter.run(str(cand), "likelihood_parity",
                    {"repo": str(repo), "reference_bundle": "missing", **cfg})


def test_parity_emits_kl_family_per_domain(
    adapter, repo, cfg, tmp_path, monkeypatch
):
    cand = _checkpoint(tmp_path, "cand.pt")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "created_by": "bdh_likelihood capture_reference",
        "reference_checkpoint_sha256": _64_HEX,
        "vendor_pin": _64_HEX,
    }))
    monkeypatch.setattr(
        "adapters.bdh_likelihood._run_helper",
        lambda *a, **k: {
            "general": [0.1, 0.1, 2.0, 5.0],
            "legal": [0.05, 0.05],
        },
    )
    cand_sha = "b" * 64
    monkeypatch.setattr(
        "adapters.bdh_likelihood.hash_checkpoint", lambda p: cand_sha
    )

    records = adapter.run(
        str(cand), "likelihood_parity",
        {"repo": str(repo), "reference_bundle": str(bundle), **cfg},
    )
    by_metric = {r["metric"]: r for r in records}
    assert all(r["adapter"] == "bdh_likelihood" for r in records)
    assert all(r["task"] == "likelihood_parity" for r in records)
    assert all(r["model_checkpoint_sha256"] == cand_sha for r in records)
    for domain in ("general", "legal"):
        for m in KL_METRICS:
            assert f"{m}@{domain}" in by_metric
        assert by_metric[f"n_tokens@{domain}"]["value"] > 0
    # known distribution asserted exactly on the mean
    assert by_metric["kl_mean@general"]["value"] == pytest.approx(1.8)
    assert by_metric["kl_mean@legal"]["value"] == pytest.approx(0.05)
    protocol = by_metric["kl_mean@general"]["protocol"]
    assert f"{_64_HEX[:12]}" in protocol
    assert f"{cand_sha[:12]}" in protocol
    assert "Path B" in protocol  # cross-path disclaimer is explicit


def test_parity_refuses_empty_score(adapter, repo, cfg, tmp_path, monkeypatch):
    cand = _checkpoint(tmp_path, "cand.pt")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "reference_checkpoint_sha256": _64_HEX,
    }))
    monkeypatch.setattr(
        "adapters.bdh_likelihood._run_helper", lambda *a, **k: {}
    )
    with pytest.raises(RuntimeError, match="refusing to store an empty"):
        adapter.run(str(cand), "likelihood_parity",
                    {"repo": str(repo), "reference_bundle": str(bundle), **cfg})


def test_store_roundtrip_isolated(tmp_path, monkeypatch, adapter, repo, cfg):
    ckpt = _checkpoint(tmp_path, "cand.pt")  # reuse as candidate path
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "reference_checkpoint_sha256": _64_HEX,
    }))
    monkeypatch.setattr(
        "adapters.bdh_likelihood._run_helper",
        lambda *a, **k: {"general": [0.5]},
    )
    monkeypatch.setattr(
        "adapters.bdh_likelihood.hash_checkpoint", lambda p: "c" * 64
    )
    monkeypatch.setattr(
        "adapters.bdh_likelihood.runtime_digest", lambda: "d" * 64
    )
    import store as store_mod

    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "isolated"))
    records = adapter.run(str(ckpt), "likelihood_parity",
                          {"repo": str(repo),
                           "reference_bundle": str(bundle), **cfg})
    stored = store.put(records)
    assert stored == records
    back = store.query({"adapter": "bdh_likelihood"})
    assert len(back) == len(records)


# --- helpers -------------------------------------------------------------


def test_vendor_pin_parses_and_strips_backticks(repo):
    # no PIN.md -> None
    assert _vendor_pin(repo) is None
    (repo / "PIN.md").write_text("Pinned commit: `0123456789abcdef`\n")
    assert _vendor_pin(repo) == "0123456789abcdef"
    (repo / "PIN.md").write_text("Pinned commit: `<unknown>`\nnope")
    assert _vendor_pin(repo) == "<unknown>"


def test_load_contexts_inline_and_files(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("contract text")
    out = load_contexts({
        "contexts": [{"domain": "general", "text": "hello"},
                     {"domain": "legal", "text": "  "}],  # blank -> dropped
        "context_files": [{"domain": "legal", "path": str(f)}],
    })
    assert out == [
        {"domain": "general", "texts": ["hello"]},
        {"domain": "legal", "texts": ["contract text"]},
    ]


def test_load_contexts_caps_and_requires(tmp_path):
    ctx = [{"domain": "d", "text": f"t{i}"} for i in range(10)]
    out = load_contexts({"contexts": ctx, "max_contexts": 3})
    assert len(out[0]["texts"]) == 3
    out = load_contexts({"contexts": ctx, "max_chars": 2})
    assert out[0]["texts"] == [f"t{i}" for i in range(10)]  # 2-char clip
    out = load_contexts({"contexts": ctx, "max_chars": 1})
    assert out[0]["texts"] == ["t"] * 10  # clipped to first char
    with pytest.raises(ValueError, match="no usable contexts"):
        load_contexts({})