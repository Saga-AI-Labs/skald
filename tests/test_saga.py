"""Tests for ``adapters.saga`` — the saga general-purpose suite adapter (round 4).

Covers the round-1 adapter interface, unified record shape constraints
(non-empty ``model_checkpoint_sha256`` + ``protocol``, exact ``RECORD_FIELDS``
key set and order), mapping determinism, unknown-task/unknown-artifact
rejection, the refuse-on-empty guard, and real-material executions for MMLU
and HumanEval driven through the saga evaluation machinery over a real local
model (bounded only by saga's own config via ``max_samples``; truthful ``n``).
"""

from __future__ import annotations

import hashlib
import inspect
import re
import subprocess
from pathlib import Path

import pytest

import store
from adapters import RECORD_FIELDS, SuiteAdapter
from store import RECORD_FIELDS as STORE_RECORD_FIELDS
from adapters.saga import (
    SagaAdapter,
    TASKS,
    _RESULT_MARK,
    _checkpoint_sha256,
    _ci,
    _file_sha256,
    _humaneval_driver,
    _mmlu_driver,
)

REPO = Path("/media/data/coding/saga")
MODEL = Path("/media/data/coding/skald/.skald/saga_models/tiny-gpt2")
PYTHON = "/media/data/coding/OBLITERATUS/.venv/bin/python"
BENCH = REPO / "src" / "evaluation" / "benchmarks.py"

MMLU_PAYLOAD = {
    "name": "mmlu",
    "score": 0.25,
    "std_error": None,
    "num_samples": 8,
    "category_scores": {},
    "details": {},
    "effective": {"num_fewshot": 5, "max_samples": 8, "seed": 42, "max_new_tokens": 64},
}
HE_PAYLOAD = {
    "name": "humaneval",
    "score": 0.125,
    "std_error": None,
    "num_samples": 8,
    "category_scores": {},
    "details": {"passed": 1, "total": 8},
    "effective": {"num_fewshot": 0, "max_samples": 8, "seed": 42, "max_new_tokens": 96},
}


def _adapter() -> SagaAdapter:
    return SagaAdapter()


def test_round1_adapter_interface():
    adapter = _adapter()
    assert isinstance(adapter, SuiteAdapter)
    sig = inspect.signature(adapter.run)
    params = list(sig.parameters)
    assert params[:2] == ["model", "task"]
    assert "config" in params
    assert TASKS == {"mmlu", "humaneval"}


def test_mapping_determinism():
    adapter = _adapter()
    m1 = adapter._metrics_from_payload("mmlu", MMLU_PAYLOAD)
    m2 = adapter._metrics_from_payload("mmlu", MMLU_PAYLOAD)
    assert m1 == m2
    assert m1 == [{"metric": "accuracy", "value": 0.25, "n": 8,
                   "ci_low": 0.0, "ci_high": pytest.approx(0.55006250)}]
    he = adapter._metrics_from_payload("humaneval", HE_PAYLOAD)
    assert he[0]["metric"] == "pass_at_1"
    assert he[0]["n"] == 8
    # same payload -> same records, in the same order (deterministic mapping)
    assert _checkpoint_sha256(MODEL) == _checkpoint_sha256(MODEL)


def test_driver_source_determinism():
    d1 = _mmlu_driver(str(REPO), str(MODEL), 5, 6, 64, 42, 1800, "tiny-gpt2")
    d2 = _mmlu_driver(str(REPO), str(MODEL), 5, 6, 64, 42, 1800, "tiny-gpt2")
    assert d1 == d2
    d3 = _humaneval_driver(
        str(REPO), str(MODEL), 0, 3, 96, 42, 1800, "tiny-gpt2", 10
    )
    assert "run_mmlu" in d1 and "SAGA_RESULT_JSON" in d3


def test_ci_mapping():
    lo, hi = _ci(0.25, 8)
    assert 0.0 <= lo <= hi <= 1.0
    assert _ci(0.5, 0) == (None, None)


def test_records_carry_exact_field_order_and_hash(tmp_path):
    adapter = _adapter()
    records = adapter._records(
        model=str(MODEL),
        task="mmlu",
        repo=REPO,
        protocol="proto mmlu 5-shot seed 42",
        seed=42,
        metrics=adapter._metrics_from_payload("mmlu", MMLU_PAYLOAD),
        stdout="SAGA_RESULT_JSON: {}\n",
        artifact_dir=str(tmp_path / "raw"),
    )
    assert records
    for r in records:
        assert list(r) == list(STORE_RECORD_FIELDS), "record key order must equal RECORD_FIELDS"
        assert set(r) == set(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["model_checkpoint_sha256"]
        assert r["protocol"]
        assert r["adapter"] == "saga"
        assert r["suite"] == "saga"
        assert r["host"]
        assert r["script_sha256"] == _file_sha256(BENCH), (
            "script_sha256 must be the direct hash of the invoked saga script"
        )
        # exact checkpoint hash: directory model hashed deterministically,
        # single file via identity.hash_checkpoint
        normalized = store.normalize(r)
        assert normalized == r
        artifact = r["artifacts"][0]
        digest, path = artifact.split("  ", 1)
        assert _file_sha256(path) == digest


def test_file_model_hash_matches_identity(tmp_path):
    import identity

    artifact = tmp_path / "ckpt.bin"
    artifact.write_bytes(b"weights-0-1-2")
    assert _checkpoint_sha256(artifact) == identity.hash_checkpoint(artifact)
    assert re.fullmatch(r"[0-9a-f]{64}", _checkpoint_sha256(artifact))


def test_unknown_task_rejection():
    with pytest.raises(ValueError, match="unsupported task"):
        _adapter().run(str(MODEL), "bbq")


def test_unknown_artifact_rejection():
    with pytest.raises(FileNotFoundError, match="evaluated model not found"):
        _adapter().run(str(MODEL / "does-not-exist"), "mmlu")


def test_unknown_repo_rejection():
    with pytest.raises(FileNotFoundError, match="suite repo not found"):
        _adapter().run(
            str(MODEL), "mmlu", {"repo": "/no/such/saga/repo", "python": PYTHON}
        )


def test_empty_result_guard(monkeypatch):
    payload = dict(MMLU_PAYLOAD)
    payload["num_samples"] = 0
    payload["score"] = 0.0
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=f"{_RESULT_MARK} " +
        __import__("json").dumps(payload)
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="no samples"):
        _adapter()._capture(REPO, PYTHON, "print('')", timeout=60)
    assert BENCH.is_file(), "saga benchmarks.py must exist for real executions"


@pytest.mark.parametrize(
    ("task", "max_samples"),
    [
        pytest.param("mmlu", 6, id="mmlu"),
        pytest.param("humaneval", 3, id="humaneval"),
    ],
)
def test_real_material_execution(tmp_path, task, max_samples):
    """One real-material execution per suite from the saga machinery.

    Bounded by saga's own config (``max_samples``); the record's ``n`` is the
    truthful number of samples saga actually evaluated.  Requires the saga
    repo, its suite deps, and network access to saga's HF streaming datasets
    (saga's own contract).
    """
    records = _adapter().run(
        str(MODEL),
        task,
        {
            "python": PYTHON,
            "max_samples": max_samples,
            "artifact_dir": str(tmp_path / "raw"),
        },
    )
    assert len(records) == 1
    r = records[0]
    assert r["task"] == task
    assert r["metric"] == ("accuracy" if task == "mmlu" else "pass_at_1")
    assert 0.0 <= r["value"] <= 1.0
    # saga's own runner caps per-subject (max_samples // len(subjects)) and
    # silently skips any subject whose streaming dataset fails to load, so n
    # is truthful but may be below the configured cap; it must never be zero
    # (the adapter refuses to emit empty records) nor exceed the cap.
    assert 0 < r["n"] <= max_samples, "n is the truthfully evaluated count, capped by saga's config"
    assert r["ci_low"] is None or r["ci_low"] <= r["value"] <= (r["ci_high"] or 1.0)
    assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
    assert r["protocol"]
    assert r["script_sha256"] == _file_sha256(BENCH)
    assert r["seed"] == 42
    assert r["artifacts"] and Path(r["artifacts"][0].split("  ", 1)[1]).is_file()
    normalized = store.normalize(r)
    assert normalized == r