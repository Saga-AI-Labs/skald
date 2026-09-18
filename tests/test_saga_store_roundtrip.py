"""Saga store roundtrip + three-family cross-suite query tests (round 4, M3).

Locks in the ``sc-saga-suites`` proof (plan §3.3: comparisons are queries):
real saga MMLU + HumanEval unified records are persisted through
``store.put`` and read back through ``store.query`` (by model checkpoint,
suite, and protocol), and one query from the single default store returns
records from all three families — bdh_cl, pi50, saga — in one result set,
each record carrying the canonical ``RECORD_FIELDS`` shape (key set and
key order), a non-empty 64-hex ``model_checkpoint_sha256``, and a non-empty
``protocol`` label.
"""

from __future__ import annotations

import re

import store
from store.schema import RECORD_FIELDS

# Real checkpoint hashes from the live saga runs (skald-adapter-saga over
# tiny-gpt2) and the round-2 bdh_cl/pi50 fixtures.
CKPT_SAGA = "11a0d2fc439a8e511804f910ed97cafebce658bfae1c4d4348ca5536a757e0a6"
CKPT_BDH = "21ec6c220714fac44af5d8a15aa186821596254d2999503c587528adc7590000"
CKPT_PI50 = "a" * 64

MMLU_PROTOCOL = (
    "saga run_mmlu (src/evaluation/benchmarks.py): 5-shot letter-choice "
    "accuracy over 6 cais/mmlu HF test subjects, greedy generation, seed 42"
)
HE_PROTOCOL = (
    "saga humaneval config entry (configs/evaluation.yaml, 0-shot, "
    "openai/openai_humaneval): adapter-side greedy 0-shot pass@1 shim, seed 42"
)


def saga_record(
    task: str = "mmlu", metric: str = "accuracy", value: float = 0.0,
    n: int = 6, protocol: str = MMLU_PROTOCOL,
) -> dict:
    """A canonical ``saga``-family record (recon §open-4 mapping)."""
    return {
        "model_checkpoint_sha256": CKPT_SAGA,
        "adapter": "saga",
        "suite": "saga",
        "task": task,
        "metric": metric,
        "value": value,
        "n": n,
        "protocol": protocol,
        "seed": 42,
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
        "protocol": "random-crop cold likelihood routing, teacher-forced",
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
        "protocol": "phase-1 artifact manifest consistency check",
        "seed": None,
    }


def test_saga_roundtrip_preserves_uniform_shape(tmp_path):
    s = store.Store(tmp_path)
    s.put([saga_record(), saga_record(task="humaneval", metric="pass_at_1")])

    back = s.query()

    assert {r["task"] for r in back} == {"mmlu", "humaneval"}
    for r in back:
        assert list(r) == list(RECORD_FIELDS), "key order must equal RECORD_FIELDS"
        assert set(r) == set(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["model_checkpoint_sha256"]
        assert r["protocol"]
        assert r["adapter"] == "saga" and r["suite"] == "saga"


def test_saga_readback_by_checkpoint_suite_and_protocol(tmp_path):
    s = store.Store(tmp_path)
    records = [
        saga_record(),
        saga_record(task="humaneval", metric="pass_at_1", protocol=HE_PROTOCOL),
    ]
    s.put(records)

    assert len(s.query({"model_checkpoint_sha256": CKPT_SAGA})) == 2
    assert len(s.query({"suite": "saga"})) == 2
    mmlu = s.query({"protocol": MMLU_PROTOCOL})
    assert len(mmlu) == 1 and mmlu[0]["task"] == "mmlu"


def test_saga_n_and_value_are_truthful_measurements(tmp_path):
    """Recon mapping: value <- score (proportion), n <- num_samples (>0)."""
    s = store.Store(tmp_path)
    s.put([saga_record(value=0.0, n=6),
           saga_record(task="humaneval", metric="pass_at_1", value=0.0, n=3,
                       protocol=HE_PROTOCOL)])

    by_task = {r["task"]: r for r in s.query()}
    assert by_task["mmlu"]["metric"] == "accuracy"
    assert by_task["humaneval"]["metric"] == "pass_at_1"
    for r in by_task.values():
        assert 0.0 <= r["value"] <= 1.0
        assert isinstance(r["n"], int) and r["n"] > 0


def test_one_query_returns_three_families_from_one_store(tmp_path):
    s = store.Store(tmp_path)
    s.put([
        bdh_record(), pi50_record(),
        saga_record(), saga_record(task="humaneval", metric="pass_at_1",
                                   protocol=HE_PROTOCOL),
    ])

    results = s.query()

    families = [r["suite"] for r in results]
    assert families.count("bdh_cl") == 1
    assert families.count("pi50") == 1
    assert families.count("saga") == 2


def test_three_family_records_share_identical_shape(tmp_path):
    s = store.Store(tmp_path)
    s.put([bdh_record(), pi50_record(), saga_record()])

    results = s.query()

    key_sets = {frozenset(r) for r in results}
    assert key_sets == {frozenset(RECORD_FIELDS)}
    for r in results:
        assert list(r) == list(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["protocol"]


def test_default_store_three_family_cross_suite_query(tmp_path, monkeypatch):
    """The real default store returns all three families in one query."""
    import store as store_mod

    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "isolated"))
    store.put([
        bdh_record(), pi50_record(value=1.0),
        saga_record(), saga_record(task="humaneval", metric="pass_at_1",
                                   protocol=HE_PROTOCOL),
    ])

    results = store.query()

    families = {r["suite"] for r in results}
    assert families == {"bdh_cl", "pi50", "saga"}
    assert len(results) == 4
    for r in results:
        assert frozenset(r) == frozenset(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])
        assert r["protocol"]