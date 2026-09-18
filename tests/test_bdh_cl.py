"""Tests for ``adapters.bdh_cl`` — the BDH-CL suite adapter (round 1).

Covers the round-1 adapter interface, unified record shape constraints (non-empty
``model_checkpoint_sha256`` + ``protocol``), real suite-script execution, and the
store end-to-end persistence gate (skipped until the sibling store-schema task
lands, at which point it must pass).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import store
from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.bdh_cl import BdhClAdapter, PROTOCOLS, _domain_spec, _routes_spec

REPO = Path("/media/data/coding/bdh-cl")
CKPT = REPO / "out/skald_raw/normalized_bdh_wikitext2_last.pt"
CORPUS = str(REPO / "data/tinyshakespeare.txt")

ROUTER_OUT = """\
router ckpt=out/skald_raw/normalized_bdh_wikitext2_last.pt | routes=[288, 576] n/head | window=32 tok | 8 crops/domain

confusion (rows=true domain, cols=routed prefix width):
                288      576
    prose         1        7
      mix         1        7

   domain   routed
    prose    20.78
      mix    19.34

joint full-width reference: ppl 18.89  (served positions only)
"""

P5_OUT = """\
parent out/a_last.pt: mult 128 | child out/b_last.pt: mult 160
P5[1] bit-exactness of the masked parent block through the child phase:
  encoder              BIT-EXACT
  decoder              BIT-EXACT
P5[2] grown segment: nonzero=True max|w|=1.2345e-06
P5[3] optimizer moments of the masked block in the child checkpoint:
  state[0] (nh,D,N) enc/encv: step=10000 v(masked)==0:True m(masked)==0:True v(grown)!=0:True
P5-VERDICT: PASS
"""


def _adapter():
    return BdhClAdapter()


def test_round1_adapter_interface():
    adapter = _adapter()
    assert isinstance(adapter, SuiteAdapter)
    sig = inspect.signature(adapter.run)
    params = list(sig.parameters)
    assert params[:2] == ["model", "task"]
    assert "config" in params


def test_spec_normalisers():
    assert _routes_spec([288, 576]) == "288,576"
    assert _domain_spec({"code": "/a", "prose": "/b"}) == "code:/a,prose:/b"


def test_router_records_have_required_fields():
    adapter = _adapter()
    parsed = adapter._parse_router(ROUTER_OUT, CROPS=8)
    assert [m["metric"] for m in parsed] == [
        "routing_confusion_288",
        "routing_confusion_576",
        "routing_confusion_288",
        "routing_confusion_576",
        "routed_perplexity",
        "routed_perplexity",
        "joint_fullwidth_perplexity",
    ]
    records = adapter._records(
        model=str(CKPT),
        task="router",
        repo=REPO,
        script=REPO / "scripts" / "eval_router.py",
        protocol=PROTOCOLS["router"],
        seed=1234,
        metrics=parsed,
        stdout=ROUTER_OUT,
        n_default=8,
    )
    assert records
    for r in records:
        assert set(r) == set(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["model_checkpoint_sha256"]
        assert r["protocol"]
        assert r["adapter"] == "bdh_cl"
        assert r["suite"] == "bdh_cl"
        assert r["host"]
        assert r["script_sha256"]


def test_p5_parser():
    parsed = _adapter()._parse_p5(P5_OUT)
    metrics = {m["metric"]: m["value"] for m in parsed}
    assert metrics == {
        "p5_bit_exact_encoder": 1.0,
        "p5_bit_exact_decoder": 1.0,
        "p5_grown_nonzero": 1.0,
        "p5_grown_max_abs_w": 1.2345e-06,
        "p5_verdict": 1.0,
    }


@pytest.mark.skipif(not CKPT.is_file(), reason="normalized BDH-CL checkpoint not present")
def test_router_real_execution():
    records = _adapter().run(
        str(CKPT),
        "router",
        {
            "routes": [288, 576],
            "domains": {"prose": CORPUS, "mix": CORPUS},
            "window": 32,
            "crops": 4,
            "batch": 2,
        },
    )
    ppl = [r for r in records if r["metric"] == "routed_perplexity"]
    assert len(ppl) == 2
    assert all(r["value"] > 0 for r in ppl)
    for r in records:
        assert r["model_checkpoint_sha256"]
        assert r["protocol"] == PROTOCOLS["router"]
        assert r["created_at"]
        assert r["script_sha256"]


@pytest.mark.skipif(not CKPT.is_file(), reason="normalized BDH-CL checkpoint not present")
def test_domain_eval_real_execution():
    records = _adapter().run(
        str(CKPT), "domain_eval", {"domains": {"prose": CORPUS}, "iters": 10, "batch": 2}
    )
    metrics = {r["metric"]: r["value"] for r in records}
    assert set(metrics) == {"prose:nll", "prose:ppl"}
    assert metrics["prose:ppl"] > 0


def test_store_end_to_end_persist_and_readback(tmp_path, monkeypatch):
    import store as store_mod

    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "isolated"))
    records = _adapter().run(
        str(CKPT),
        "router",
        {
            "routes": [288, 576],
            "domains": {"prose": CORPUS, "mix": CORPUS},
            "window": 32,
            "crops": 2,
            "batch": 1,
        },
    )
    try:
        store.put(records)
    except NotImplementedError:
        pytest.skip("store.put not implemented: sibling task skald-store-schema is still working")
    key = {
        "adapter": "bdh_cl",
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    }
    back = store.query(key)
    assert len(back) == len(records)
    assert all(r["protocol"] for r in back)
    saved_runs = list((tmp_path / "isolated" / "runs").glob("*.json"))
    assert saved_runs, "per-run immutable JSON files must be persisted"