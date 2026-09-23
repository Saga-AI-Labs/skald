"""Tests for ``adapters.null_model`` — the §3.6 degenerate-reference floor.

No network, no weights: floors are scored over caller-supplied inline items
in the openai_compat shape. Identity anchors are fixed digests per null kind.
"""

from __future__ import annotations

import hashlib
import inspect

import pytest

import store
from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.null_model import NullModelAdapter

MMLU_ITEMS = [
    {"prompt": "2+2?", "choices": ["3", "4", "5", "6"], "gold": "A"},
    {"prompt": "Capital of France?", "choices": ["Rome", "Paris", "Oslo", "Bern"], "gold": "B"},
    {"prompt": "Water boils at?", "choices": ["90C", "100C", "110C", "120C"], "gold": "C"},
    {"prompt": "1+1?", "choices": ["2", "3", "4", "5"], "gold": "A"},
]

HUMANEVAL_ITEMS = [
    {
        "prompt": "def add(a, b):\n    \"\"\"Add.\"\"\"\n",
        "entry_point": "add",
        "test": "def check(f):\n    assert f(1, 2) == 3\n",
    },
    {
        "prompt": "def sub(a, b):\n    \"\"\"Subtract.\"\"\"\n",
        "entry_point": "sub",
        "test": "def check(f):\n    assert f(5, 3) == 2\n",
    },
]

STUB_ID = hashlib.sha256(b"null-model::stub").hexdigest()
RANDOM_ID = hashlib.sha256(b"null-model::random").hexdigest()


def _adapter():
    return NullModelAdapter()


def test_adapter_interface():
    adapter = _adapter()
    assert isinstance(adapter, SuiteAdapter)
    sig = inspect.signature(adapter.run)
    params = list(sig.parameters)
    assert params[:2] == ["model", "task"]
    assert "config" in params


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unsupported task"):
        _adapter().run("label", "gsm8k", {})


def test_unknown_null_kind_is_rejected():
    with pytest.raises(ValueError, match="null_kind"):
        _adapter().run("label", "mmlu", {"null_kind": "quantized",
                                         "mmlu_items": MMLU_ITEMS})


def test_items_are_required():
    with pytest.raises(ValueError, match="mmlu_items"):
        _adapter().run("label", "mmlu", {})
    with pytest.raises(ValueError, match="humaneval_items"):
        _adapter().run("label", "humaneval", {"null_kind": "random"})


def test_stub_mmlu_scores_constant_a():
    (r,) = _adapter().run("label", "mmlu", {"mmlu_items": MMLU_ITEMS})
    assert set(r) == set(RECORD_FIELDS)
    assert r["adapter"] == "null_model" and r["task"] == "mmlu"
    assert r["metric"] == "accuracy"
    assert r["value"] == 0.5 and r["n"] == 4  # 2 of 4 golds are A
    assert r["model_checkpoint_sha256"] == STUB_ID
    assert "null-stub" in r["protocol"]
    assert r["seed"] is None  # stub is deterministic; no sampling seed


def test_random_mmlu_is_seeded_and_fixed_identity():
    a = _adapter().run(
        "label", "mmlu",
        {"null_kind": "random", "seed": 7, "mmlu_items": MMLU_ITEMS},
    )[0]
    b = _adapter().run(
        "label", "mmlu",
        {"null_kind": "random", "seed": 7, "mmlu_items": MMLU_ITEMS},
    )[0]
    assert a["value"] == b["value"]
    assert a["model_checkpoint_sha256"] == RANDOM_ID != STUB_ID
    assert a["seed"] == 7
    assert 0.0 <= a["value"] <= 1.0 and a["n"] == 4


def test_stub_humaneval_is_measured_zero():
    (r,) = _adapter().run("label", "humaneval", {"humaneval_items": HUMANEVAL_ITEMS})
    assert r["metric"] == "pass_at_1"
    assert r["value"] == 0.0 and r["n"] == 2
    assert r["model_checkpoint_sha256"] == STUB_ID


def test_random_humaneval_is_zero_floor():
    (r,) = _adapter().run(
        "label", "humaneval",
        {"null_kind": "random", "humaneval_items": HUMANEVAL_ITEMS},
    )
    assert r["value"] == 0.0 and r["n"] == 2
    assert r["model_checkpoint_sha256"] == RANDOM_ID


def test_identity_ignores_model_label():
    a = _adapter().run("label-one", "mmlu", {"mmlu_items": MMLU_ITEMS})[0]
    b = _adapter().run("label-two", "mmlu", {"mmlu_items": MMLU_ITEMS})[0]
    assert a["model_checkpoint_sha256"] == b["model_checkpoint_sha256"] == STUB_ID


def test_store_roundtrip_isolated(tmp_path, monkeypatch):
    import store as store_mod

    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "isolated"))
    records = _adapter().run("label", "mmlu", {"mmlu_items": MMLU_ITEMS})
    stored = store.put(records)
    assert stored == records
    back = store.query({"adapter": "null_model"})
    assert len(back) == len(records)
