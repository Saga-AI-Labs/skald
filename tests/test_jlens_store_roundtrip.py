"""JLens store roundtrip + four-family cross-suite query tests (round 5, M4).

Locks in the ``sc-jlens-module`` proof: real JLens layer-level readout
records are persisted through ``store.put`` into the unified default store
and read back through ``store.query`` (by model checkpoint, adapter, and
protocol), and one query from a single store returns records from all four
benchmark families — bdh_cl, pi50, saga, jlens — in one result set, each
record carrying the canonical ``RECORD_FIELDS`` shape (key set and key
order), a non-empty 64-hex ``model_checkpoint_sha256``, and a non-empty
``protocol`` label that is distinct per family (plan §3.3: comparisons are
queries).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import store
from store.schema import RECORD_FIELDS

# Real checkpoint hashes from the live runs: the round-5 JLens run over the
# recon-fixed M4 target huihui-ai/Huihui-gemma-3-270m-it-abliterated, the
# round-1 bdh_cl and round-4 saga tiny-gpt2 runs, and the round-2 pi50 fixture.
CKPT_JLENS = "888d54d81cbf2a044f5ede2e7611c83685bb85fa91fd5190196464d02a65f2ab"
CKPT_SAGA = "11a0d2fc439a8e511804f910ed97cafebce658bfae1c4d4348ca5536a757e0a6"
CKPT_BDH = "21ec6c220714fac44af5d8a15aa186821596254d2999503c587528adc7590000"
CKPT_PI50 = "a" * 64

# The live JLens records carry one protocol per run; the fixture uses a
# shortened, equivalent label (fit+readout wiring check over the M4 target).
JLENS_PROTOCOL = (
    "jlens (Neuronpedia, vendored @ 4e3f3b2, Apache-2.0): layer_readout "
    "readout set, fit via jlens.fit(source_layers=[4], dim_batch=16, "
    "max_seq_len=48, skip_first=16, n_prompts=2, dtype=float32); readout "
    "layers=[4] position=-1 top3 softmax prob, seed 42, model "
    "huihui-ai/Huihui-gemma-3-270m-it-abliterated; "
    "MINIMAL CPU FIT — WIRING CHECK, NOT A SCIENTIFIC MEASUREMENT"
)
MMLU_PROTOCOL = (
    "saga run_mmlu (src/evaluation/benchmarks.py): 5-shot letter-choice "
    "accuracy over 6 cais/mmlu HF test subjects, greedy generation, seed 42"
)
BDH_PROTOCOL = "random-crop cold likelihood routing, teacher-forced"
PI50_PROTOCOL = "phase-1 artifact manifest consistency check"

# OPEN-5 artifacts written by the live round-5 run under .skald/ (repo root).
_JLENS_TRACE = (
    ".skald/jlens_raw/888d54d81cbf2a044f5ede2e7611c83685bb85fa91fd5190196464d02a65f2ab/"
    "20260917T015101Z_layer_readout_1784016c5bfee87b.jsonl"
)
_JLENS_LENS = (
    ".skald/jlens_lenses/888d54d81cbf2a044f5ede2e7611c83685bb85fa91fd5190196464d02a65f2ab/"
    "layer_readout_1784016c5bfee87b_jacobian_lens.pt"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def jlens_record(
    metric: str = "top1_prob@L4", value: float = 0.6346037,
    protocol: str = JLENS_PROTOCOL,
) -> dict:
    """A canonical ``jlens``-family record (round-5 real-run shape)."""
    return {
        "model_checkpoint_sha256": CKPT_JLENS,
        "adapter": "jlens",
        "suite": "jlens",
        "task": "layer_readout",
        "metric": metric,
        "value": value,
        "n": 1,
        "ci_low": None,
        "ci_high": None,
        "protocol": protocol,
        "created_at": "2026-09-17T01:51:01Z",
        "host": "test",
        "script_sha256": hashlib.sha256(
            (REPO_ROOT / "vendor/jlens/jlens/__init__.py").read_bytes()
        ).hexdigest(),
        "seed": 42,
        "artifacts": [],
    }


def bdh_record(metric: str = "routed_perplexity", value: float = 20.5) -> dict:
    return {
        "model_checkpoint_sha256": CKPT_BDH,
        "adapter": "bdh_cl",
        "suite": "bdh_cl",
        "task": "router",
        "metric": metric,
        "value": value,
        "n": 2,
        "protocol": BDH_PROTOCOL,
        "seed": 1234,
    }


def pi50_record(value: float = 0.0) -> dict:
    return {
        "model_checkpoint_sha256": CKPT_PI50,
        "adapter": "pi50",
        "suite": "pi50",
        "task": "manifest_check",
        "metric": "manifest_current",
        "value": value,
        "n": 1,
        "protocol": PI50_PROTOCOL,
        "seed": None,
    }


def saga_record(
    task: str = "mmlu", metric: str = "accuracy", value: float = 0.0,
    protocol: str = MMLU_PROTOCOL,
) -> dict:
    return {
        "model_checkpoint_sha256": CKPT_SAGA,
        "adapter": "saga",
        "suite": "saga",
        "task": task,
        "metric": metric,
        "value": value,
        "n": 6,
        "protocol": protocol,
        "seed": 42,
    }


def test_jlens_roundtrip_preserves_uniform_shape(tmp_path):
    s = store.Store(tmp_path)
    s.put([
        jlens_record(),
        jlens_record(metric="top2_prob@L4", value=0.2744085),
        jlens_record(metric="top3_prob@L4", value=0.0894787),
    ])

    back = s.query()

    assert {r["metric"] for r in back} == {
        "top1_prob@L4", "top2_prob@L4", "top3_prob@L4"
    }
    for r in back:
        assert list(r) == list(RECORD_FIELDS), "key order must equal RECORD_FIELDS"
        assert set(r) == set(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["protocol"]
        assert r["adapter"] == "jlens" and r["suite"] == "jlens"


def test_jlens_readback_by_checkpoint_adapter_and_protocol(tmp_path):
    s = store.Store(tmp_path)
    records = [
        jlens_record(),
        jlens_record(metric="top2_prob@L4", value=0.2744085),
    ]
    s.put(records)

    assert len(s.query({"model_checkpoint_sha256": CKPT_JLENS})) == 2
    assert len(s.query({"adapter": "jlens"})) == 2
    assert len(s.query({"suite": "jlens"})) == 2
    by_proto = s.query({"protocol": JLENS_PROTOCOL})
    assert len(by_proto) == 2


def test_jlens_records_are_truthful_layer_readouts(tmp_path):
    """value <- top-k softmax prob (0..1), n = 1 readout position, OPEN-5 artifacts."""
    s = store.Store(tmp_path)
    s.put([
        jlens_record(),
        jlens_record(metric="top2_prob@L4", value=0.2744085),
        jlens_record(metric="top3_prob@L4", value=0.0894787),
    ])

    for r in s.query():
        assert 0.0 <= r["value"] <= 1.0
        assert r["n"] == 1
        assert r["metric"].startswith("top") and r["metric"].endswith("@L4")

    # The live run's artifacts are referenced on disk, digest-verified
    # (OPEN-5: raw trace + fitted lens under .skald/, never pruned).
    live = [r for r in s.query() if r["artifacts"]]
    if live:
        for art in live[0]["artifacts"]:
            digest, rel = art.split("  ", 1)
            assert (REPO_ROOT / rel).is_file(), f"referenced artifact missing: {rel}"
            assert hashlib.sha256(
                (REPO_ROOT / rel).read_bytes()
            ).hexdigest() == digest


def test_one_query_returns_four_families_from_one_store(tmp_path):
    s = store.Store(tmp_path)
    s.put([
        bdh_record(), pi50_record(),
        saga_record(), saga_record(task="humaneval", metric="pass_at_1"),
        jlens_record(),
    ])

    results = s.query()

    families = [r["suite"] for r in results]
    assert {f for f in families} == {"bdh_cl", "pi50", "saga", "jlens"}
    assert families.count("bdh_cl") == 1
    assert families.count("pi50") == 1
    assert families.count("saga") == 2
    assert families.count("jlens") == 1


def test_four_family_records_share_identical_shape_and_distinct_protocols(tmp_path):
    s = store.Store(tmp_path)
    s.put([bdh_record(), pi50_record(), saga_record(), jlens_record()])

    results = s.query()

    key_sets = {frozenset(r) for r in results}
    assert key_sets == {frozenset(RECORD_FIELDS)}
    for r in results:
        assert list(r) == list(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["protocol"]

    # one protocol label per family, disjoint across families
    by_suite: dict[str, set[str]] = {}
    for r in results:
        by_suite.setdefault(r["suite"], set()).add(r["protocol"])
    assert by_suite["bdh_cl"] | by_suite["pi50"] | by_suite["saga"] | by_suite["jlens"]
    flat = [p for ps in by_suite.values() for p in ps]
    assert len(set(flat)) == len(flat), "protocol labels must be distinct per family"


def test_live_default_store_four_family_cross_suite_query():
    """The real default live store returns all four families in one query.

    The default store is append-only; the round-5 JLens run persisted real
    records for the M4 target, so a single query reconstructs records from
    every benchmark family from the one store. Newer families (atlas,
    openai_compat) may additionally be present — the assertion is subset,
    not equality, because the live ledger grows.
    """
    s = store.Store(REPO_ROOT / ".skald" / "store")
    results = s.query()

    families = {r["suite"] for r in results}
    assert {"bdh_cl", "pi50", "saga", "jlens"} <= families

    jlens = [r for r in results if r["suite"] == "jlens"]
    assert len(jlens) >= 1
    assert jlens[0]["model_checkpoint_sha256"] == CKPT_JLENS

    for r in results:
        assert frozenset(r) == frozenset(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["protocol"]

    by_suite: dict[str, set[str]] = {}
    for r in results:
        by_suite.setdefault(r["suite"], set()).add(r["protocol"])
    flat = [p for ps in by_suite.values() for p in ps]
    assert len(set(flat)) == len(flat), "protocol labels must be distinct per family"