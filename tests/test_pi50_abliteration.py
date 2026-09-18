"""Tests for the abliteration/refusal half of ``adapters.pi50``.

The ``pi50`` family carries two very different jobs: ``manifest_check`` (a BDH-CL
freshness audit, covered in ``tests/test_pi50.py``) and the Qwen3.8-Flash-Next
abliteration instruments vendored under ``vendor/pi50_eval/``.  This file covers
the second half only.

Everything here runs offline: no server, no network, and no 99 GiB weight tree.
Transcripts are synthesised per-test, and the checkpoint argument is a small
file, because ``hash_checkpoint`` reads the whole artifact.  The scorers
themselves are the real vendored scripts -- nothing is stubbed that the
measurement actually depends on.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.pi50 import (
    ABLIT_TASKS,
    MAX_FAILED_FRAC,
    Pi50Adapter,
    _pick_interval,
    _proportion,
)

HARNESS = Path(__file__).resolve().parent.parent / "vendor" / "pi50_eval" / "harness"

REFUSED = "I'm sorry, but I can't help with that."
COMPLIANT = (
    "To measure the temperature of the reactor core, place a type-K thermocouple "
    "against the housing, wait for the reading to settle for roughly thirty seconds, "
    "and record the value against the calibration curve. Repeat at three points and "
    "take the median so a single drifting sensor cannot move the reported figure."
)


def _row(i: int, *, content: str, safe: bool, category: str = "general") -> dict:
    return {
        "id": f"fixture:{i}",
        "category": category,
        "expected_safe": safe,
        "content": content,
        "error": None,
        "http_status": 200,
        "finish_reason": "stop",
        "budget_starved": False,
        "usage": {"completion_tokens": 40, "prompt_tokens": 20, "total_tokens": 60},
        "wall_s": 1.2,
    }


def _transcript(path: Path, *, arm: str, rows: list[dict]) -> Path:
    """Write a transcript shaped exactly like ``run_suite.py``'s output."""
    header = {"kind": "meta", "arm": arm, "request_model": f"{arm}-model-id", "params": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


@pytest.fixture()
def ckpt(tmp_path):
    """A small stand-in for the evaluated weights: hashing it must stay cheap."""
    p = tmp_path / "weights.bin"
    p.write_bytes(b"skald-fixture-weights" * 64)
    return str(p)


# --- adapter contract -------------------------------------------------------


def test_abliteration_tasks_are_dispatchable():
    adapter = Pi50Adapter()
    assert isinstance(adapter, SuiteAdapter)
    assert ABLIT_TASKS == {
        "run_suite", "score_refusal", "score_confab", "score_capability", "paired_compare",
    }
    for task in sorted(ABLIT_TASKS):
        # a bogus config must fail on configuration, never silently return nothing
        with pytest.raises(Exception):
            adapter.run("definitely-not-a-path", task, {})


def test_unknown_task_is_rejected():
    with pytest.raises(Exception):
        Pi50Adapter().run("definitely-not-a-path", "score_nonsense", {})


def test_missing_checkpoint_is_refused_not_padded(ckpt):
    # every record is keyed to its weights, so there is no placeholder for a
    # checkpoint that does not exist on this machine
    with pytest.raises(FileNotFoundError, match="no placeholder"):
        Pi50Adapter().run(str(Path(ckpt) / "missing.bin"), "score_refusal", {})


# --- record shape and the store dimensions ----------------------------------


def test_scored_records_are_record_fields_exact(ckpt, tmp_path):
    tx = _transcript(
        tmp_path / "stock_a.jsonl", arm="stock",
        rows=[_row(i, content=REFUSED, safe=False) for i in range(4)]
        + [_row(i, content=COMPLIANT, safe=True) for i in range(4)],
    )
    records = Pi50Adapter().run(ckpt, "score_refusal", {
        "python": sys.executable, "transcripts": [str(tx)], "protocol": "fixture",
    })
    assert records, "the vendored scorer produced nothing to assert on"
    for record in records:
        assert set(record) == set(RECORD_FIELDS)
        assert record["adapter"] == "pi50"
        assert record["model_checkpoint_sha256"]
        assert record["protocol"]


def test_suite_defaults_to_the_family_not_the_task(ckpt, tmp_path):
    """``suite`` is the family (cf. saga's ``"suite": "saga"``).

    Defaulting it to the task name would make ``list_suites_adapters`` report
    every task as if it were a suite, splitting one family across five rows.
    """
    tx = _transcript(tmp_path / "stock_a.jsonl", arm="stock",
                     rows=[_row(1, content=COMPLIANT, safe=True)])
    records = Pi50Adapter().run(ckpt, "score_refusal", {
        "python": sys.executable, "transcripts": [str(tx)], "protocol": "fixture",
    })
    assert {r["suite"] for r in records} == {"pi50"}
    assert {r["task"] for r in records} == {"score_refusal"}
    # an explicit suite still wins, so a caller can sub-partition deliberately
    override = Pi50Adapter().run(ckpt, "score_refusal", {
        "python": sys.executable, "transcripts": [str(tx)],
        "protocol": "fixture", "suite": "abliteration",
    })
    assert {r["suite"] for r in override} == {"abliteration"}


def test_rates_are_stored_as_proportions_like_saga(ckpt, tmp_path):
    """``saga.py`` stores accuracy as a proportion; mixing scales would make
    cross-adapter queries compare a percentage against a fraction."""
    tx = _transcript(tmp_path / "stock_a.jsonl", arm="stock", rows=[
        _row(i, content=REFUSED, safe=False) for i in range(8)
    ] + [
        _row(i, content=COMPLIANT, safe=True) for i in range(2)
    ])
    records = Pi50Adapter().run(ckpt, "score_refusal", {
        "python": sys.executable, "transcripts": [str(tx)], "protocol": "fixture",
    })
    rates = [r for r in records if r["metric"].endswith(".refusal_rate")]
    assert rates, "no refusal_rate metric came back"
    for record in rates:
        assert 0.0 <= record["value"] <= 1.0


def test_proportion_helper_divides_and_guards_a_zero_denominator():
    """``_proportion`` takes (numerator, denominator), not a pre-divided value.

    A zero or missing denominator must yield ``None`` rather than a fabricated
    0.0 -- "no items scored" and "nothing correct" are different statements.
    """
    assert _proportion(924, 1000) == pytest.approx(0.924)
    assert _proportion(0, 10) == 0.0
    assert _proportion(5, 0) is None
    assert _proportion(5, None) is None
    assert _proportion(None, 10) is None


# --- interval selection -------------------------------------------------------


def test_pick_interval_prefers_cluster_bootstrap_when_clustered():
    stats = {
        "wilson": [0.90, 0.99],
        "bootstrap": [0.88, 0.97],
        "clusters": 6,
    }
    lo, hi, method = _pick_interval(stats)
    assert (lo, hi) == (0.88, 0.97)
    assert "bootstrap" in method


def test_pick_interval_falls_back_to_wilson_when_unclustered():
    stats = {"wilson": [0.90, 0.99], "bootstrap": [0.88, 0.97], "clusters": 1}
    lo, hi, method = _pick_interval(stats)
    assert (lo, hi) == (0.90, 0.99)
    assert "wilson" in method.lower()


def test_pick_interval_reports_no_interval_rather_than_a_fabricated_one():
    lo, hi, method = _pick_interval({"clusters": 1})
    assert (lo, hi) == (None, None)
    assert method


# --- arm identity -------------------------------------------------------------


def test_arm_label_comes_from_the_meta_header_not_the_file_name(tmp_path):
    # the two twins are one directory name apart; a file name cannot tell them
    # apart, so the label must come from what the harness recorded about the server
    deceptive = _transcript(tmp_path / "stock_named.jsonl", arm="ablit",
                            rows=[_row(1, content=COMPLIANT, safe=True)])
    assert Pi50Adapter()._arm_label([str(deceptive)]) == "ablit"


def test_arm_label_reports_unlabelled_rather_than_guessing(tmp_path):
    bare = tmp_path / "bare.jsonl"
    bare.write_text(json.dumps(_row(1, content=COMPLIANT, safe=True)) + "\n", encoding="utf-8")
    assert Pi50Adapter()._arm_label([str(bare)]) == "unlabelled"


def test_transcript_files_honours_the_arm_prefix(tmp_path):
    _transcript(tmp_path / "stock_xstest.jsonl", arm="stock",
                rows=[_row(1, content=COMPLIANT, safe=True)])
    resolved = Pi50Adapter()._transcript_files(
        {"transcript_dir": str(tmp_path), "suites": ["xstest"], "arm": "stock"}, "transcripts")
    assert [Path(p).name for p in resolved] == ["stock_xstest.jsonl"]


def test_missing_transcript_raises_rather_than_scoring_a_shorter_set(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        Pi50Adapter()._transcript_files(
            {"transcripts": [str(tmp_path / "absent.jsonl")]}, "transcripts")


def test_no_transcripts_at_all_raises(tmp_path):
    with pytest.raises(ValueError, match="no transcripts"):
        Pi50Adapter()._transcript_files({"transcript_dir": str(tmp_path / "empty")}, "transcripts")


# --- run_suite guards ---------------------------------------------------------


def test_run_suite_demands_a_model_to_corroborate(ckpt, tmp_path):
    suite = tmp_path / "s.jsonl"
    suite.write_text(json.dumps(_row(1, content=COMPLIANT, safe=True)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="require_model"):
        Pi50Adapter().run(ckpt, "run_suite", {"suite": str(suite)})


def test_run_suite_demands_a_base_url(ckpt, tmp_path):
    suite = tmp_path / "s.jsonl"
    suite.write_text(json.dumps(_row(1, content=COMPLIANT, safe=True)) + "\n", encoding="utf-8")
    saved = os.environ.pop("FN_BASE", None)
    try:
        with pytest.raises(ValueError, match="base_url"):
            Pi50Adapter().run(
                ckpt, "run_suite", {"suite": str(suite), "require_model": "some-model"})
    finally:
        if saved is not None:
            os.environ["FN_BASE"] = saved


def test_collection_health_gate_refuses_a_half_failed_run(ckpt, tmp_path, monkeypatch):
    """An aggregate over a half-failed collection is not a measurement."""
    suite = tmp_path / "s.jsonl"
    suite.write_text(json.dumps(_row(1, content=COMPLIANT, safe=True)) + "\n", encoding="utf-8")
    out_dir = tmp_path / "out" / "stock"
    out_dir.mkdir(parents=True)
    rows = [_row(i, content=COMPLIANT, safe=True) for i in range(10)]
    for i in range(3):                      # 30% > MAX_FAILED_FRAC
        rows[i] = {**rows[i], "error": "http 500", "http_status": 500, "content": ""}
    _transcript(out_dir / "s.jsonl", arm="stock", rows=rows)

    def fake_run_tool(script, args, python, timeout, env_extra=None, ok_rc=(0,)):
        return "collection finished\n"

    adapter = Pi50Adapter()
    # patch the instance, not the class: a plain function on the class would
    # receive ``self`` as ``script`` and every argument would shift by one
    monkeypatch.setattr(adapter, "_run_tool", fake_run_tool)
    with pytest.raises(RuntimeError, match="collection unhealthy"):
        adapter.run(ckpt, "run_suite", {
            "python": sys.executable, "suite": str(suite),
            "require_model": "some-model", "base_url": "http://127.0.0.1:1/v1",
            "transcript_dir": str(tmp_path / "out"), "arm": "stock",
        })
    assert MAX_FAILED_FRAC == 0.02


# --- subprocess return-code policy -------------------------------------------


def test_run_tool_accepts_only_the_return_codes_a_caller_declared(tmp_path, monkeypatch):
    """``ok_rc`` is what lets ``compare_arms.py``'s ``return 1 if hard else 0`` be read
    as a measurement rather than a crash -- without letting any non-zero code slide."""
    import adapters.pi50 as pi50_module

    script = tmp_path / "rc.py"
    script.write_text("import sys\nsys.exit(int(sys.argv[1]))\n", encoding="utf-8")
    monkeypatch.setattr(pi50_module, "HARNESS_DIR", tmp_path)
    adapter = Pi50Adapter()
    assert adapter._run_tool("rc.py", ["0"], sys.executable, 60) == ""
    assert adapter._run_tool("rc.py", ["1"], sys.executable, 60, ok_rc=(0, 1)) == ""
    with pytest.raises(RuntimeError, match="rc=1"):
        adapter._run_tool("rc.py", ["1"], sys.executable, 60)


def test_run_tool_reports_the_missing_instrument_rather_than_the_os_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="vendor tree broken"):
        Pi50Adapter()._run_tool("no_such_tool.py", [], sys.executable, 60)


def test_score_json_surfaces_the_tool_output_on_a_half_written_document(monkeypatch):
    """A present-but-unparseable ``--json`` file means the tool died mid-write.

    ``json.dump`` streams into the open handle, so a serialisation crash used to
    leave a truncated document behind; a caller that only asked "did the file
    appear" would then parse half a JSON object and store it as a result.
    """
    def fake_run_tool(script, args, python, timeout, env_extra=None, ok_rc=(0,)):
        target = args[args.index("--json") + 1]
        Path(target).write_text('{"shared": 623, "only_a": ', encoding="utf-8")
        return "Traceback: TypeError during serialisation\n"

    adapter = Pi50Adapter()
    monkeypatch.setattr(adapter, "_run_tool", fake_run_tool)
    with pytest.raises(RuntimeError, match="unparseable"):
        adapter._score_json("compare_arms.py", [], [], sys.executable, 60, "paired")


# --- paired_compare -----------------------------------------------------------


def _pair(tmp_path):
    a = _transcript(tmp_path / "stock_mmlu.jsonl", arm="stock", rows=[
        _row(1, content=REFUSED, safe=False),
        _row(2, content=COMPLIANT, safe=False),
        _row(3, content=COMPLIANT, safe=True),
        _row(4, content=COMPLIANT, safe=True),
    ])
    b = _transcript(tmp_path / "ablit_mmlu.jsonl", arm="ablit", rows=[
        _row(1, content=COMPLIANT, safe=False),      # REFUSED -> COMPLIANT
        _row(2, content=COMPLIANT, safe=False),
        _row(3, content=COMPLIANT, safe=True),
        _row(4, content=COMPLIANT, safe=True),
    ])
    return a, b


def test_paired_compare_records_the_transitions_it_exists_for(ckpt, tmp_path):
    a, b = _pair(tmp_path)
    records = Pi50Adapter().run(ckpt, "paired_compare", {
        "python": sys.executable, "pair_a": str(a), "pair_b": str(b),
        "label_a": "stock", "label_b": "ablit", "protocol": "fixture",
    })
    metrics = {r["metric"]: r for r in records}
    assert records, "paired_compare stored nothing"
    for record in records:
        assert set(record) == set(RECORD_FIELDS)
    # the whole point of the task: the REFUSED->OFF_TARGET sites plan §4.1 names
    assert any(name.startswith("transition.") for name in metrics)
    assert "transition.REFUSED->COMPLIANT" in metrics
    assert metrics["shared"]["value"] == 4.0
    # disagreements/pairing_warnings are LISTS in the payload: the count is len(),
    # and an isinstance(int) test used to drop both fields without a word
    assert "disagreements" in metrics
    assert "pairing_warnings" in metrics


def test_paired_compare_needs_both_sides(ckpt, tmp_path):
    a, _ = _pair(tmp_path)
    with pytest.raises(ValueError, match="pair_a.*pair_b|pair_b"):
        Pi50Adapter().run(ckpt, "paired_compare", {
            "python": sys.executable, "pair_a": str(a), "protocol": "fixture",
        })


def test_paired_compare_survives_a_header_without_suite_file(ckpt, tmp_path):
    """Regression: the prompt-fallback joined ``HERE/data/None`` before its guard.

    A transcript whose provenance header omits ``suite_file`` used to die with
    ``TypeError: join() argument must be str... not NoneType`` instead of simply
    skipping the fallback -- which is the right answer when the prompts are
    already stored inline, as every current transcript does.
    """
    a, b = _pair(tmp_path)
    for path in (a, b):                      # strip suite_file from the header
        lines = path.read_text(encoding="utf-8").splitlines()
        head = json.loads(lines[0])
        head.pop("suite_file", None)
        path.write_text(json.dumps(head) + "\n" + "".join(l + "\n" for l in lines[1:]),
                        encoding="utf-8")
    records = Pi50Adapter().run(ckpt, "paired_compare", {
        "python": sys.executable, "pair_a": str(a), "pair_b": str(b), "protocol": "fixture",
    })
    assert {r["metric"] for r in records} & {"shared", "disagreements"}


def test_paired_compare_json_survives_tuple_keyed_counters(ckpt, tmp_path):
    """Regression: ``per_class`` is keyed by ``(classA, classB)`` tuples.

    ``json.dump`` rejects non-string keys, so ``--json`` used to die with
    ``TypeError: keys must be str... not tuple`` -- and because it streamed into
    the open handle, it also left a truncated file behind.
    """
    a, b = _pair(tmp_path)
    records = Pi50Adapter().run(ckpt, "paired_compare", {
        "python": sys.executable, "pair_a": str(a), "pair_b": str(b), "protocol": "fixture",
    })
    names = {r["metric"] for r in records}
    assert any("->" in name for name in names)
    assert any(name.startswith("transition.benign.") or name.startswith("transition.harmful.")
               for name in names), "per_class buckets never reached the store"


# --- identity interlock -------------------------------------------------------


def test_checkpoint_hash_agrees_with_sagas_implementation(ckpt, tmp_path):
    """One join key, one definition of it.

    ``model_checkpoint_sha256`` is how the surfaces join a result to a weight
    atlas entry; two adapters hashing the same artifact differently would split
    every model's history in two.
    """
    from adapters.saga import _checkpoint_sha256 as saga_hash
    from adapters.pi50 import _checkpoint_sha256 as pi50_hash

    assert pi50_hash(ckpt) == saga_hash(ckpt)
    directory = tmp_path / "snapshot"
    (directory / "sub").mkdir(parents=True)
    (directory / "a.safetensors").write_bytes(b"weights-a")
    (directory / "sub" / "b.json").write_bytes(b'{"x": 1}')
    assert pi50_hash(directory) == saga_hash(directory)
    # and a directory must hash at all: identity.hash_checkpoint opens() it and dies
    assert len(pi50_hash(directory)) == 64


def test_vendored_harness_is_present_and_importable():
    assert HARNESS.is_dir(), f"vendor tree broken: {HARNESS} is missing"
    for name in ("run_suite.py", "score_refusal.py", "score_confab.py",
                 "score_capability.py", "compare_arms.py"):
        assert (HARNESS / name).is_file(), f"vendored instrument missing: {name}"
