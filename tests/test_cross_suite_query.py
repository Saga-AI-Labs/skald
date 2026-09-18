"""Cross-suite query tests — one result-store, one query, both families.

Locks in the plan §3.3 unification claim (comparisons are queries): from a
single store holding records of both the ``bdh_cl`` and ``pi50`` benchmark
families, one ``store.query`` returns records from both families in one
result set, and every returned record carries the identical canonical
record shape (plan §4.2 field set) plus its own ``protocol`` label.
"""

from __future__ import annotations

import re

import pytest

import store
from store import normalize
from store.schema import RECORD_FIELDS

CKPT_BDH = "21ec6c220714fac44af5d8a15aa186821596254d2999503c587528adc7590000"
CKPT_PI50 = "a" * 64


def bdh_record(metric: str = "routed_perplexity", value: float = 20.5) -> dict:
    """A canonical ``bdh_cl``-family record (router shape, round-1 real run)."""
    return {
        "model_checkpoint_sha256": CKPT_BDH,
        "adapter": "bdh_cl",
        "suite": "bdh_cl",
        "task": "router",
        "metric": metric,
        "value": value,
        "n": 2,
        "protocol": "random-crop cold likelihood routing, teacher-forced",
        "seed": 1234,
    }


def pi50_record(metric: str = "manifest_current", value: float = 1.0) -> dict:
    """A canonical ``pi50``-family record (manifest_check shape)."""
    return {
        "model_checkpoint_sha256": CKPT_PI50,
        "adapter": "pi50",
        "suite": "pi50",
        "task": "manifest_check",
        "metric": metric,
        "value": value,
        "n": 1,
        "protocol": "phase-1 artifact manifest consistency check",
        "seed": None,
    }


def test_one_query_returns_both_families_from_one_store(tmp_path):
    s = store.Store(tmp_path)
    s.put([bdh_record(), pi50_record(value=0.0)])

    results = s.query()

    families = {r["suite"] for r in results}
    assert "bdh_cl" in families
    assert "pi50" in families
    assert [r["suite"] for r in results].count("bdh_cl") >= 1
    assert [r["suite"] for r in results].count("pi50") >= 1


def test_records_from_both_families_share_identical_shape(tmp_path):
    s = store.Store(tmp_path)
    s.put([bdh_record(), pi50_record(value=0.0)])

    results = s.query()

    key_sets = {frozenset(r) for r in results}
    assert key_sets == {frozenset(RECORD_FIELDS)}
    assert [r["suite"] for r in results].count("bdh_cl") >= 1
    assert [r["suite"] for r in results].count("pi50") >= 1


def test_cross_suite_comparison_is_direct_fieldwise_query(tmp_path):
    s = store.Store(tmp_path)
    s.put([bdh_record(value=20.5), pi50_record(value=0.0)])

    results = s.query()

    by_family = {r["suite"]: r for r in results}
    bdh, pi50 = by_family["bdh_cl"], by_family["pi50"]
    # Same canonical field names in both families -> field-wise dict access.
    assert list(bdh) == list(RECORD_FIELDS)
    assert list(pi50) == list(RECORD_FIELDS)
    assert set(bdh) == set(pi50)
    assert bdh["metric"] and pi50["metric"]


def test_default_store_cross_suite_query(tmp_path, monkeypatch):
    """The real default store returns records from both families in one query."""
    import store as store_mod

    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "isolated"))
    store.put([bdh_record(), pi50_record(value=1.0)])

    results = store.query()

    assert {r["suite"] for r in results} == {"bdh_cl", "pi50"}
    assert len(results) == 2
    for r in results:
        assert frozenset(r) == frozenset(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["protocol"]


def test_each_record_keeps_its_own_protocol_label(tmp_path):
    """No silent cross-protocol comparison: labels stay per family."""
    s = store.Store(tmp_path)
    s.put([bdh_record(), pi50_record(value=0.5)])

    results = s.query()
    protocols = {r["suite"]: r["protocol"] for r in results}

    assert protocols["bdh_cl"] != protocols["pi50"]
    assert "random-crop" in protocols["bdh_cl"]
    assert "manifest" in protocols["pi50"]