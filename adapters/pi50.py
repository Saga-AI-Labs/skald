"""Pi-50 phase-1 instrument suite adapter (plan §4.1, task skald-adapter-pi50).

Attribution: the instrument invoked here
(`scripts/pi50/phase1_manifest.py`) was written by the BDH-CL project
(© 2025 Pathway Technology, Inc.) and is vendored unmodified under
``vendor/bdh_cl/`` (see ``vendor/bdh_cl/PIN.md`` for pin and license).
Skald adds only this wrapper adapter. Upstream bugs belong upstream.

Runs the frozen Pi-50 phase-1 instrument for a target artifact and returns
unified result records (plan §4.2).

Interface: ``run(model, task, config) -> records[]`` (scaffold §5).  The
adapter invokes each instrument unchanged via subprocess and parses its
printed result; it does not re-implement the instruments and does not
persist — callers write records to the unified store via ``store.put``.

Input-selectivity policy (scripts/pi50/README.md): most phase-1 instruments
consume grown-ladder checkpoints (``out/bdh_europarl_ladRA2b-*.pt``), the
serving matrix (``docs/reports/data/2026-09-10_ra2b_matrix.csv``), or
seat-local data (Weight-Atlas server, ``~/bdh-review/*`` persisted-feature
trees).  Those are pinned where they consume them; the adapter selects only
tasks whose inputs are committed in the repo or runnable on this machine.

Runnable task on this box:

- ``manifest_check`` -> ``scripts/pi50/phase1_manifest.py --check``
  Git-tracked phase-1 evidence base vs committed
  ``docs/PHASE1-MANIFEST.md``; the script itself exits 0 for "manifest up
  to date" and 1 for "MANIFEST STALE — regenerate", both of which are
  *measurements*, not crashes, so the adapter records the verdict instead
  of raising on nonzero rc.

Every emitted record carries all canonical ``RECORD_FIELDS`` keys, a
non-empty ``model_checkpoint_sha256`` (SHA-256 of the evaluated artifact —
for ``manifest_check`` that is the committed manifest file) and a
non-empty ``protocol`` label so numbers from different evaluation protocols
are never silently compared.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from adapters import RECORD_FIELDS, SuiteAdapter
from identity import hash_checkpoint

# The instrument script itself is vendored (byte-identical upstream copy).
# BUT the check it performs is intrinsically bound to a BDH-CL git checkout:
# the script runs `git rev-parse/ls-files/log` against its own repository to
# audit evidence freshness. There is therefore no portable default for the
# checkout under audit — config["repo"] must name a BDH-CL checkout
# explicitly, and the adapter refuses to silently audit the wrong repo.
DEFAULT_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "vendor"
    / "bdh_cl"
    / "scripts"
    / "pi50"
    / "phase1_manifest.py"
)

# Raw stdout artifacts live in Skald's own artifact dir, never in the vendor
# tree (which must stay byte-identical to upstream).
DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / ".skald" / "pi50_raw"
)

TASKS = {"manifest_check"}

# --- abliteration / refusal diagnostics ------------------------------------
#
# Plan §4.1 assigns this family its measurement: "abliteration/refusal
# diagnostics: REFUSED->OFF_TARGET sites, fabrication rate".  The instruments
# that actually perform it are vendored under ``vendor/pi50_eval/harness/``
# (see ``vendor/pi50_eval/PIN.md``) and are driven **unmodified in a
# subprocess** — this adapter re-implements no scoring, it parses the JSON the
# instruments emit.  These tasks therefore need no BDH-CL checkout: their
# default path works from a fresh clone (CREDITS rule 4).
ABLIT_TASKS = {
    "run_suite",
    "score_refusal",
    "score_confab",
    "score_capability",
    "paired_compare",
}
TASKS = TASKS | ABLIT_TASKS

HARNESS_DIR = (
    Path(__file__).resolve().parent.parent / "vendor" / "pi50_eval" / "harness"
)
# ``score_capability.py --data-dir`` defaults to ``harness/data``, which does
# not exist (the suites are a sibling of the harness, not a child of it), so
# the adapter must pass this explicitly rather than inherit that default.
SUITE_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "vendor" / "pi50_eval" / "data"
)
# Transcripts are model *outputs* — including the abliterated arm complying
# with harmful requests.  They are not published data and never enter git:
# ``.skald/`` is gitignored, and the store keeps only the aggregate plus an
# artifact digest.  See the privacy boundary in
# ``docs/2026-09-14_ablit-eval-suite.md``.
DEFAULT_TRANSCRIPT_DIR = (
    Path(__file__).resolve().parent.parent / ".skald" / "pi50_transcripts"
)

# ``run_suite.py`` takes its endpoint from the environment, not from flags
# (``FN_BASE`` / ``FN_MODEL``), so the adapter sets them in the child env.
SERVER_BASE_ENV = "FN_BASE"
SERVER_MODEL_ENV = "FN_MODEL"

# ``run_suite.py``'s own collection gate.  A partially failed collection still
# yields a plausible-looking aggregate — a 404 storm once produced a 0%
# refusal rate that was entirely an artifact of the arm identity being wrong —
# so records from a tripped run are refused, not stored.
MAX_FAILED_FRAC = 0.02

PROTOCOLS = {
    "manifest_check": (
        "phase-1 artifact manifest consistency check "
        "(git-tracked table vs committed PHASE1-MANIFEST.md, "
        "generation stamp excluded, 2026-09-13 CI policy)"
    ),
}

# Interval methods, recorded in the protocol label: two numbers from
# different interval methods must not be silently compared either.
INTERVAL_CLUSTER_BOOTSTRAP = "cluster-bootstrap-95"
INTERVAL_WILSON = "wilson-95"
INTERVAL_NONE = "no-interval"


def _pick_interval(stats: dict[str, Any]) -> tuple[float | None, float | None, str]:
    """Choose the honest 95% interval for one scored block.

    The instruments report both a Wilson interval (assumes iid items) and a
    cluster-aware bootstrap.  Several suites are *not* iid — the SORRY-Bench-2
    variants re-express one base behaviour several ways, so the effective
    sample is the number of clusters, not the number of rows.  Where items are
    clustered and the bootstrap is available it is the defensible interval;
    Wilson would understate uncertainty.  Which one was used is recorded in
    the protocol label.
    """
    clusters = stats.get("clusters") or 1
    boot = stats.get("bootstrap")
    if (
        clusters > 1
        and isinstance(boot, (list, tuple))
        and len(boot) == 2
        and boot[0] is not None
        and boot[1] is not None
    ):
        return float(boot[0]), float(boot[1]), f"{INTERVAL_CLUSTER_BOOTSTRAP}({clusters} clusters)"
    wil = stats.get("wilson")
    if isinstance(wil, (list, tuple)) and len(wil) == 2 and wil[0] is not None:
        return float(wil[0]), float(wil[1]), INTERVAL_WILSON
    return None, None, INTERVAL_NONE


def _proportion(numerator: Any, denominator: Any) -> float | None:
    """Rate as a proportion in [0, 1] — the scale the store compares on.

    ``adapters/saga.py`` stores accuracy as a proportion (its ``_ci`` clamps to
    [0, 1]), so percentages must not enter ``value`` here; ``accuracy_pct``
    stays in the artifact only.
    """
    if not denominator:
        return None
    if numerator is None:
        # "nobody counted this" is not the same statement as "none of them were right"
        return None
    return float(numerator) / float(denominator)

_VERDICT_RE = re.compile(r"(manifest up to date|MANIFEST STALE - regenerate)")


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _hash_dir(path: Path) -> str:
    """Deterministic digest of a directory: sorted relative paths + contents.

    Identical in construction to ``adapters/saga.py``'s helper of the same
    name — deliberately, because the checkpoint hash is the store's join key
    and the same weights must hash the same way under every adapter.
    ``tests/test_pi50_abliteration.py`` asserts the agreement rather than
    trusting the comment.
    """
    h = hashlib.sha256()
    for rel in sorted(p.relative_to(path) for p in path.rglob("*") if p.is_file()):
        h.update(rel.as_posix().encode("utf-8"))
        h.update(b"\0")
        with open(path / rel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _checkpoint_sha256(model: str | Path) -> str:
    """SHA-256 identity of the evaluated model artifact.

    Single files use ``identity.hash_checkpoint``; directories (an
    ``huggingface_hub`` snapshot of sharded weights is one) are hashed over
    their sorted relative paths + contents.
    """
    path = Path(model)
    if path.is_file():
        return hash_checkpoint(path)
    return _hash_dir(path)


# --- abliteration / refusal task handlers ------------------------------------


class _Pi50AbliterationMixin:
    """Abliteration/refusal tasks for :class:`Pi50Adapter` (split for readability)."""

    # --- subprocess plumbing ------------------------------------------------

    def _run_tool(
        self,
        script: str,
        args: list[str],
        python: str,
        timeout: int,
        env_extra: dict[str, str] | None = None,
        ok_rc: tuple[int, ...] = (0,),
    ) -> str:
        """Run one vendored instrument from ``harness/`` and return its stdout.

        ``cwd`` and ``PYTHONPATH`` are the harness directory because the
        instruments import each other by bare module name
        (``from score_refusal import classify``).
        """
        target = HARNESS_DIR / script
        if not target.is_file():
            raise FileNotFoundError(
                f"pi50: vendored instrument missing: {target} "
                "(vendor tree broken — see vendor/pi50_eval/PIN.md)"
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(HARNESS_DIR)
        env.update(env_extra or {})
        try:
            proc = subprocess.run(
                [python, str(target), *args],
                cwd=HARNESS_DIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"pi50: {script} timed out after {timeout}s"
            ) from exc
        if proc.returncode not in ok_rc:
            raise RuntimeError(
                f"pi50: {script} failed (rc={proc.returncode})\n"
                f"cmd: {python} {target} {' '.join(args)}\n"
                f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
            )
        return proc.stdout

    def _score_json(
        self,
        script: str,
        files: list[str],
        extra: list[str],
        python: str,
        timeout: int,
        tag: str,
        ok_rc: tuple[int, ...] = (0,),
    ) -> tuple[dict[str, Any], str]:
        """Run a scorer over *files* with ``--json`` and return (payload, stdout).

        ``--json`` is used rather than stdout scraping so the parse is exact;
        the payload is additionally required to be a JSON object.
        """
        import tempfile

        with tempfile.TemporaryDirectory(prefix="skald-pi50-") as tmp:
            json_path = Path(tmp) / f"{tag}.json"
            out = self._run_tool(
                script,
                [*files, *extra, "--json", str(json_path)],
                python,
                timeout,
                ok_rc=ok_rc,
            )
            if not json_path.is_file():
                raise RuntimeError(
                    f"pi50: {script} wrote no --json file; refusing to guess "
                    f"the numbers from prose\n{out[-1500:]}"
                )
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                # A present-but-unparseable --json file means the tool died mid-write.
                # Surface the tool's own output instead of a bare decoder message.
                raise RuntimeError(
                    f"pi50: {script} wrote an unparseable --json file ({exc}); "
                    f"the tool most likely crashed mid-serialisation"
                    f" (rc was accepted as benign).\ntool output:\n{out[-2000:]}"
                ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"pi50: {script} produced {type(payload).__name__}, expected an object"
            )
        return payload, out

    @staticmethod
    def _transcript_files(config: dict[str, Any], key: str) -> list[str]:
        """Resolve the transcript files to score, from *key* or ``transcript_dir``.

        An explicit list wins; otherwise ``transcript_dir`` is globbed, or
        narrowed to ``config['suites']`` when given.  Missing files raise —
        scoring a silently-shorter file list would look like a real result.
        """
        files = config.get(key)
        if files is None and config.get("transcript_dir"):
            directory = Path(config["transcript_dir"])
            suites = config.get("suites")
            if suites:
                # run_suite.py writes arm-prefixed transcripts (``stock_mmlu.jsonl``),
                # so a bare ``suites`` list would miss every real file. ``arm`` composes
                # the prefix; a genuinely absent file still raises below, loudly.
                arm = config.get("arm")
                prefix = f"{arm}_" if arm else ""
                files = [str(directory / f"{prefix}{s}.jsonl") for s in suites]
            else:
                files = sorted(str(p) for p in directory.glob("*.jsonl"))
        if not files:
            raise ValueError(
                f"pi50: no transcripts to score — pass config[{key!r}] or "
                "config['transcript_dir'] (optionally with config['suites'])"
            )
        if isinstance(files, str):
            files = [files]
        missing = [f for f in files if not Path(f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"pi50: transcript file(s) not found: {missing}"
            )
        return [str(Path(f).resolve()) for f in files]

    # --- task: run_suite ----------------------------------------------------

    def _run_run_suite(
        self, model: str, config: dict[str, Any], python: str
    ) -> list[dict[str, Any]]:
        """Collect one suite's transcripts against a live OpenAI-compatible server.

        The endpoint and served model come from ``FN_BASE``/``FN_MODEL``
        because ``run_suite.py`` reads them from the environment, not from
        flags.  ``require_model`` is **mandatory**: arm identity has to be
        corroborated by the server answering ``/models``, never inferred from
        a filename — the ablit and stock twins are one directory name apart.
        """
        suite = config.get("suite")
        if not suite:
            raise ValueError("pi50: run_suite needs config['suite']")
        require_model = config.get("require_model")
        if not require_model:
            raise ValueError(
                "pi50: run_suite needs config['require_model'] — the served "
                "model id must be asserted and corroborated against the "
                "server's /models, not assumed from the file name"
            )
        base_url = config.get("base_url") or os.environ.get(SERVER_BASE_ENV)
        if not base_url:
            raise ValueError(
                f"pi50: run_suite needs config['base_url'] (or {SERVER_BASE_ENV})"
            )
        suite_path = Path(suite)
        if not suite_path.is_file():
            suite_path = SUITE_DATA_DIR / f"{suite}.jsonl"
        if not suite_path.is_file():
            raise FileNotFoundError(f"pi50: suite file not found: {suite}")

        arm = str(config.get("arm") or "unlabelled")
        out_dir = Path(config.get("transcript_dir") or DEFAULT_TRANSCRIPT_DIR) / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{suite_path.stem}.jsonl"

        args = [
            "--suite", str(suite_path),
            "--out", str(out_path),
            "--require-model", str(require_model),
            "--arm", arm,
        ]
        for flag, key in (("--max-tokens", "max_tokens"), ("--temperature", "temperature"),
                          ("--top-p", "top_p"), ("--seed", "seed"), ("--workers", "workers"),
                          ("--limit", "limit"), ("--offset", "offset"),
                          ("--system", "system"), ("--thinking-budget", "thinking_budget")):
            if config.get(key) is not None:
                args += [flag, str(config[key])]
        if config.get("thinking") is True:
            args.append("--thinking")
        if config.get("overwrite"):
            args.append("--overwrite")

        stdout = self._run_tool(
            "run_suite.py", args, python, int(config.get("timeout", 7200)),
            env_extra={
                SERVER_BASE_ENV: str(base_url),
                SERVER_MODEL_ENV: str(config.get("model_id") or require_model),
            },
        )

        rows = []
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            # run_suite.py's own provenance header is {"kind": "meta", ...}.
            if isinstance(row, dict) and row.get("kind") == "meta":
                continue
            rows.append(row)
        total = len(rows)
        failed = sum(1 for r in rows if r.get("error"))
        starved = sum(1 for r in rows if r.get("budget_starved"))
        if total == 0:
            raise RuntimeError(f"pi50: run_suite produced no rows for {suite}\n{stdout[-1500:]}")
        failed_frac = failed / total
        if failed_frac > MAX_FAILED_FRAC:
            raise RuntimeError(
                f"pi50: run_suite collection unhealthy — {failed}/{total} rows "
                f"carried an error ({failed_frac:.1%} > {MAX_FAILED_FRAC:.1%}). "
                "An aggregate over a half-failed collection is not a measurement; "
                f"nothing is stored.\n{stdout[-1500:]}"
            )
        metrics = [
            {"metric": "collected", "value": float(total), "n": total},
            {"metric": "request_errors", "value": float(failed), "n": total},
            {"metric": "budget_starved", "value": float(starved), "n": total},
        ]
        protocol = (
            f"suite collection against {base_url} serving {require_model!r} "
            f"(arm={arm}, identity corroborated via /models, "
            f"health gate max_failed_frac={MAX_FAILED_FRAC})"
        )
        return self._abl_records(
            model=model, task="run_suite", config=config, protocol=protocol,
            seed=config.get("seed"), metrics=metrics,
            artifacts=[f"{_file_sha256(out_path)}  {out_path}"],
            stdout=stdout,
        )

    # --- task: score_refusal --------------------------------------------------

    def _run_score_refusal(
        self, model: str, config: dict[str, Any], python: str
    ) -> list[dict[str, Any]]:
        files = self._transcript_files(config, "transcripts")
        payload, stdout = self._score_json(
            "score_refusal.py", files, [], python,
            int(config.get("timeout", 900)), "refusal",
        )
        blocks = payload.get("blocks")
        if not isinstance(blocks, dict) or not blocks:
            raise RuntimeError(
                f"pi50: score_refusal emitted no 'blocks'; got keys {sorted(payload)}"
            )
        metrics: list[dict[str, Any]] = []
        methods: set[str] = set()
        detail: list[str] = []
        for block, stats in sorted(blocks.items()):
            n = stats.get("n")
            lo, hi, method = _pick_interval(stats)
            methods.add(method.split("(")[0])
            detail.append(f"{block}:{method}")
            metrics.append({"metric": f"{block}.refusal_rate", "value": stats.get("rate"),
                            "n": n, "ci_low": lo, "ci_high": hi})
            # No interval on off_target_rate on purpose: the block's wilson/
            # bootstrap belong to ``rate`` (the refusal rate), so reusing them
            # here would attach someone else's interval to this number.
            metrics.append({"metric": f"{block}.off_target_rate",
                            "value": stats.get("off_target_rate"), "n": n})
            for field in ("refused", "complied", "off_target"):
                if field in stats:
                    metrics.append({"metric": f"{block}.{field}",
                                    "value": float(stats[field]), "n": n})
        starved = payload.get("budget_starved")
        if starved is not None:
            metrics.append({"metric": "budget_starved", "value": float(bool(starved))})
        scored = payload.get("scored")
        n_total = int(scored) if isinstance(scored, int) else None
        if n_total is not None:
            metrics.append({"metric": "scored", "value": float(n_total), "n": n_total})
        arm = self._arm_label(files)
        protocol = (
            f"refusal classification by score_refusal.classify (marker set + "
            f"PROCEDURAL rules) over {len(files)} transcript file(s) "
            f"[arm={arm}]; interval={'+'.join(sorted(methods))} "
            f"[{'; '.join(detail)}]"
        )
        return self._abl_records(
            model=model, task="score_refusal", config=config, protocol=protocol,
            seed=config.get("seed"), metrics=metrics, stdout=stdout,
            artifacts=self._artifact_refs(files),
        )

    # --- task: score_confab ---------------------------------------------------

    def _run_score_confab(
        self, model: str, config: dict[str, Any], python: str
    ) -> list[dict[str, Any]]:
        files = self._transcript_files(config, "transcripts")
        extra = []
        if config.get("suite_meta"):
            extra += ["--suite-meta", str(config["suite_meta"])]
        payload, stdout = self._score_json(
            "score_confab.py", files, extra, python,
            int(config.get("timeout", 600)), "confab",
        )
        tally = payload.get("tally")
        if not isinstance(tally, dict):
            raise RuntimeError(
                f"pi50: score_confab emitted no 'tally'; got keys {sorted(payload)}"
            )
        n_total = int(sum(v for v in tally.values() if isinstance(v, int)))
        metrics: list[dict[str, Any]] = [
            {"metric": "fabrication_rate", "value": payload.get("fabrication_rate"),
             "n": n_total},
        ]
        for verdict, count in sorted(tally.items()):
            metrics.append({"metric": f"verdict.{verdict.lower()}",
                            "value": float(count), "n": n_total})
            metrics.append({"metric": f"verdict.{verdict.lower()}_share",
                            "value": _proportion(count, n_total), "n": n_total})
        arm = self._arm_label(files)
        protocol = (
            f"fabrication/abstention tally over {len(files)} transcript file(s) "
            f"[arm={arm}]; denominators are the scored items, no interval is "
            "reported by the instrument"
        )
        return self._abl_records(
            model=model, task="score_confab", config=config, protocol=protocol,
            seed=config.get("seed"), metrics=metrics, stdout=stdout,
            artifacts=self._artifact_refs(files),
        )

    # --- task: score_capability -------------------------------------------------

    def _run_score_capability(
        self, model: str, config: dict[str, Any], python: str
    ) -> list[dict[str, Any]]:
        files = self._transcript_files(config, "transcripts")
        data_dir = str(Path(config.get("data_dir") or SUITE_DATA_DIR).resolve())
        if not Path(data_dir).is_dir():
            raise FileNotFoundError(f"pi50: suite data dir not found: {data_dir}")
        payload, stdout = self._score_json(
            "score_capability.py", files, ["--data-dir", data_dir], python,
            int(config.get("timeout", 900)), "capability",
        )
        suites = payload.get("suites")
        if not isinstance(suites, dict) or not suites:
            raise RuntimeError(
                f"pi50: score_capability emitted no 'suites'; got keys {sorted(payload)}"
            )
        metrics: list[dict[str, Any]] = []
        methods: set[str] = set()
        detail: list[str] = []
        for name, stats in sorted(suites.items()):
            n = stats.get("n")
            correct = stats.get("correct")
            lo, hi, method = _pick_interval(stats)
            methods.add(method.split("(")[0])
            detail.append(f"{name}:{method}")
            # value is a proportion, matching adapters/saga.py's scale.
            metrics.append({"metric": f"{name}.accuracy",
                            "value": _proportion(correct, n), "n": n,
                            "ci_low": lo, "ci_high": hi})
            tally = stats.get("tally") or {}
            for verdict, count in sorted(tally.items()):
                metrics.append({"metric": f"{name}.verdict.{verdict.lower()}",
                                "value": float(count), "n": n})
            for field in ("median_out_tokens", "median_wall_s"):
                if stats.get(field) is not None:
                    metrics.append({"metric": f"{name}.{field}",
                                    "value": float(stats[field]), "n": n})
            for cat, cs in sorted((stats.get("per_category") or {}).items()):
                metrics.append({"metric": f"{name}.category.{cat}.accuracy",
                                "value": _proportion(cs.get("correct"), cs.get("n")),
                                "n": cs.get("n")})
        arm = self._arm_label(files)
        protocol = (
            f"capability accuracy over {len(files)} transcript file(s) "
            f"[arm={arm}], data-dir={Path(data_dir).name}; value is a proportion; "
            f"interval={'+'.join(sorted(methods))} [{'; '.join(detail)}]"
        )
        return self._abl_records(
            model=model, task="score_capability", config=config, protocol=protocol,
            seed=config.get("seed"), metrics=metrics, stdout=stdout,
            artifacts=self._artifact_refs(files),
        )

    # --- task: paired_compare ---------------------------------------------------

    def _run_paired_compare(
        self, model: str, config: dict[str, Any], python: str
    ) -> list[dict[str, Any]]:
        """Paired arm comparison: the REFUSED->OFF_TARGET transitions plan §4.1 names."""
        pair_a = config.get("pair_a")
        pair_b = config.get("pair_b")
        if not pair_a or not pair_b:
            raise ValueError(
                "pi50: paired_compare needs config['pair_a'] and config['pair_b'] "
                "(one transcript file each, the two arms over the same suite)"
            )
        for label, path in (("pair_a", pair_a), ("pair_b", pair_b)):
            if not Path(path).is_file():
                raise FileNotFoundError(f"pi50: paired_compare {label} not found: {path}")
        extra: list[str] = []
        for flag, key in (("--label-a", "label_a"), ("--label-b", "label_b")):
            if config.get(key):
                extra += [flag, str(config[key])]
        payload, stdout = self._score_json(
            "compare_arms.py", [str(Path(pair_a).resolve()), str(Path(pair_b).resolve())],
            extra, python, int(config.get("timeout", 600)), "paired",
            # compare_arms.py ends with ``return 1 if hard else 0``: rc=1 means
            # pairing warnings, not a crash. rc=2 (no shared prompt ids) stays fatal.
            ok_rc=(0, 1),
        )
        metrics: list[dict[str, Any]] = []
        shared = payload.get("shared")
        if isinstance(shared, int):
            metrics.append({"metric": "shared", "value": float(shared), "n": shared})
        # compare_arms.py emits ``transitions`` as a FLAT {"A->B": n} map and
        # ``per_class`` as {bucket: {"A->B": n}}. ``disagreements`` and
        # ``pairing_warnings`` are LISTS, so their contribution is len(), not the
        # value itself -- the previous isinstance(int) test silently dropped all
        # of these fields and recorded nothing.
        n_shared = shared if isinstance(shared, int) else None
        flat: dict[str, int] = {}
        transitions = payload.get("transitions")
        if isinstance(transitions, dict):
            for key, count in transitions.items():
                if isinstance(count, int):
                    flat[str(key)] = count
        per_class = payload.get("per_class")
        if isinstance(per_class, dict):
            for bucket, mapping in per_class.items():
                if isinstance(mapping, dict):
                    for key, count in mapping.items():
                        if isinstance(count, int):
                            flat[f"{bucket}.{key}"] = count
        for name, count in sorted(flat.items()):
            metrics.append({"metric": f"transition.{name}", "value": float(count),
                            "n": n_shared})
        for field in ("only_a", "only_b"):
            value = payload.get(field)
            if isinstance(value, int):
                metrics.append({"metric": field, "value": float(value), "n": n_shared})
        for field in ("disagreements", "pairing_warnings"):
            value = payload.get(field)
            if isinstance(value, (list, dict)):
                metrics.append({"metric": field, "value": float(len(value)), "n": n_shared})
        protocol = (
            f"paired arm comparison over the same suite "
            f"[a={Path(pair_a).name}, b={Path(pair_b).name}]; transitions are "
            "the REFUSED->OFF_TARGET sites plan §4.1 names"
        )
        return self._abl_records(
            model=model, task="paired_compare", config=config, protocol=protocol,
            seed=config.get("seed"), metrics=metrics, stdout=stdout,
            artifacts=self._artifact_refs([str(pair_a), str(pair_b)]),
        )

    # --- shared record construction -------------------------------------------

    @staticmethod
    def _arm_label(files: list[str]) -> str:
        """Arm label read out of the transcripts' own provenance header.

        ``run_suite.py`` writes ``{"kind": "meta", "arm": ..., "params": ...,
        "request_model": ...}`` as the file's first line, so the label comes
        from what the harness recorded about the server it talked to — not
        from the file name, which cannot distinguish the two twins.
        """
        labels = set()
        for path in files:
            try:
                with open(path, encoding="utf-8") as handle:
                    for _ in range(4):
                        line = handle.readline()
                        if not line:
                            break
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(row, dict) and row.get("kind") == "meta":
                            arm = row.get("arm")
                            if arm:
                                labels.add(str(arm))
                            break
            except OSError:
                continue
        return "+".join(sorted(labels)) if labels else "unlabelled"

    @staticmethod
    def _artifact_refs(files: list[str]) -> list[str]:
        return [f"{_file_sha256(f)}  {f}" for f in files]

    def _abl_records(
        self,
        *,
        model: str,
        task: str,
        config: dict[str, Any],
        protocol: str,
        seed: Any,
        metrics: list[dict[str, Any]],
        stdout: str,
        artifacts: list[str],
    ) -> list[dict[str, Any]]:
        """Build RECORD_FIELDS-exact records for the abliteration tasks.

        ``suite`` is the instrument's own suite name when the task scored one
        suite, else the task, so ``store.query({'suite': ...})`` stays useful.
        """
        artifact = self._write_ablit_artifact(task, stdout, config.get("artifact_dir"))
        script = HARNESS_DIR / _TASK_SCRIPT[task]
        checkpoint = _checkpoint_sha256(model)
        # ``suite`` is the family, matching adapters/saga.py (``"suite": "saga"``) and
        # _build_records (``"suite": "pi50"``). Defaulting it to the *task* name would
        # make list_suites_adapters report every task as if it were a suite.
        suite = str(config.get("suite") or "pi50")
        records = []
        for m in metrics:
            record = {
                "model_checkpoint_sha256": checkpoint,
                "adapter": "pi50",
                "suite": suite,
                "task": task,
                "metric": m["metric"],
                "value": m.get("value"),
                "n": m.get("n"),
                "ci_low": m.get("ci_low"),
                "ci_high": m.get("ci_high"),
                "protocol": protocol,
                "created_at": _now(),
                "host": socket.gethostname(),
                "script_sha256": _file_sha256(script) if script.is_file() else None,
                "seed": seed,
                "artifacts": list(artifacts) + artifact,
            }
            assert set(record) == set(RECORD_FIELDS), set(record) ^ set(RECORD_FIELDS)
            records.append(record)
        return records

    def _write_ablit_artifact(
        self, task: str, stdout: str, artifact_dir: str | Path | None
    ) -> list[str]:
        out_dir = Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"{task}_{stamp}.txt"
        path.write_text(stdout, encoding="utf-8")
        return [f"{_file_sha256(path)}  {path}"]


_TASK_SCRIPT = {
    "run_suite": "run_suite.py",
    "score_refusal": "score_refusal.py",
    "score_confab": "score_confab.py",
    "score_capability": "score_capability.py",
    "paired_compare": "compare_arms.py",
}


class Pi50Adapter(SuiteAdapter, _Pi50AbliterationMixin):
    """Adapter over the frozen Pi-50 phase-1 instruments in ``DEFAULT_REPO``."""

    def run(
        self,
        model: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        config = config or {}
        if task not in TASKS:
            raise ValueError(
                f"pi50: unsupported task {task!r}; choose from {sorted(TASKS)}"
            )
        # The two task groups validate differently on purpose:
        # ``manifest_check`` is bound to a BDH-CL checkout and a single
        # artifact file, while the abliteration tasks run from the vendored
        # harness alone and key on a checkpoint that may be a directory.
        if task == "manifest_check":
            return self._manifest_entry(model, config)
        return self._abliteration_entry(model, task, config)

    def _manifest_entry(
        self, model: str, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        repo = config.get("repo")
        if not repo:
            raise ValueError(
                "pi50: manifest_check audits a BDH-CL checkout's evidence "
                "freshness via its own git history, so config['repo'] must "
                "name a BDH-CL checkout explicitly; there is no portable "
                "default (see vendor/bdh_cl/PIN.md)"
            )
        repo = Path(repo)
        if not repo.is_dir():
            raise FileNotFoundError(f"pi50: BDH-CL suite repo not found: {repo}")
        # The instrument is stdlib-only: any interpreter runs it.
        python = config.get("python") or sys.executable
        if not python:
            raise FileNotFoundError("pi50: no python interpreter found")
        model = str(model)
        if not Path(model).is_file():
            raise FileNotFoundError(f"pi50: evaluated artifact not found: {model}")
        return self._run_manifest_check(model, config, repo, python)

    def _abliteration_entry(
        self, model: str, task: str, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not HARNESS_DIR.is_dir():
            raise FileNotFoundError(
                f"pi50: vendored harness not found: {HARNESS_DIR} "
                "(vendor tree broken — see vendor/pi50_eval/PIN.md)"
            )
        model = str(model)
        if not Path(model).exists():
            raise FileNotFoundError(
                f"pi50: evaluated checkpoint not found: {model} — every record "
                "is keyed to its weights, so there is no placeholder for it"
            )
        python = config.get("python") or sys.executable
        if not python:
            raise FileNotFoundError("pi50: no python interpreter found")
        handlers = {
            "run_suite": self._run_run_suite,
            "score_refusal": self._run_score_refusal,
            "score_confab": self._run_score_confab,
            "score_capability": self._run_score_capability,
            "paired_compare": self._run_paired_compare,
        }
        return handlers[task](model, config, str(python))

    # --- task handlers ----------------------------------------------------

    def _run_manifest_check(
        self,
        model: str,
        config: dict[str, Any],
        repo: Path,
        python: Path,
    ) -> list[dict[str, Any]]:
        args = [str(python), str(DEFAULT_SCRIPT), "--check"]
        if not DEFAULT_SCRIPT.is_file():
            raise FileNotFoundError(
                f"pi50: vendored instrument missing: {DEFAULT_SCRIPT} "
                "(vendor tree broken — see vendor/bdh_cl/PIN.md)"
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo)
        out, rc = self._capture(repo, args, env, int(config.get("timeout", 300)))
        parsed = self._parse_manifest(out, rc)
        return self._records(
            model=model,
            task="manifest_check",
            repo=repo,
            script=DEFAULT_SCRIPT,
            protocol=PROTOCOLS["manifest_check"],
            seed=None,
            metrics=parsed,
            stdout=out,
            n_default=1,
            artifact_dir=config.get("artifact_dir"),
        )

    # --- machinery --------------------------------------------------------

    def _capture(
        self,
        repo: Path,
        args: list[str],
        env: dict[str, str],
        timeout: int,
    ) -> tuple[str, int]:
        """Run *args* in *repo* and return (stdout, returncode).

        Phase-1 instruments may use nonzero exit codes as measurement data
        (e.g. ``phase1_manifest --check`` exits 0 for *up to date* and 1
        for *stale*); the caller is responsible for interpreting the exit
        code.  A nonzero rc with no parseable verdict is treated as a crash
        and raises ``RuntimeError``; nonzero rc with a verdict is recorded.
        """
        try:
            proc = subprocess.run(
                args,
                cwd=repo,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"pi50: suite instrument timed out after {timeout}s"
            ) from exc
        # rc != 0 and != 1 is a genuine crash (instruments only use 0/1).
        if proc.returncode not in (0, 1):
            raise RuntimeError(
                f"pi50: suite instrument crashed (rc={proc.returncode})\n"
                f"cmd: {' '.join(args)}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        return proc.stdout, proc.returncode

    def _parse_manifest(self, out: str, rc: int) -> list[dict[str, Any]]:
        """Parse the phase-1 manifest check verdict into unified metrics."""
        m = _VERDICT_RE.search(out)
        if not m:
            raise RuntimeError(
                "pi50: manifest_check output did not carry a recognised verdict "
                f"(rc={rc}):\n{out[-2000:]}"
            )
        fresh = m.group(1) == "manifest up to date"
        return [
            {
                "metric": "manifest_current",
                "value": 1.0 if fresh else 0.0,
            }
        ]

    def _records(
        self,
        *,
        model: str,
        task: str,
        repo: Path,
        script: Path,
        protocol: str,
        seed: int | None,
        metrics: list[dict[str, Any]],
        stdout: str,
        n_default: int | None,
        artifact_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        artifact = self._write_artifact(task, stdout, artifact_dir)
        records = []
        for m in metrics:
            record = {
                # _checkpoint_sha256, not the raw identity.hash_checkpoint: an
                # huggingface_hub snapshot of sharded weights is a DIRECTORY, and the
                # raw helper opens it with open() and dies with IsADirectoryError.
                "model_checkpoint_sha256": _checkpoint_sha256(model),
                "adapter": "pi50",
                "suite": "pi50",
                "task": task,
                "metric": m["metric"],
                "value": m.get("value"),
                "n": m.get("n", n_default),
                "ci_low": m.get("ci_low"),
                "ci_high": m.get("ci_high"),
                "protocol": protocol,
                "created_at": _now(),
                "host": socket.gethostname(),
                "script_sha256": _file_sha256(script),
                "seed": seed,
                "artifacts": artifact,
            }
            assert set(record) == set(RECORD_FIELDS), set(record) ^ set(RECORD_FIELDS)
            records.append(record)
        return records

    def _write_artifact(
        self, task: str, stdout: str, artifact_dir: str | Path | None
    ) -> list[str]:
        out_dir = Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"{task}_{stamp}.txt"
        path.write_text(stdout)
        return [f"{_file_sha256(path)}  {path}"]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for the pi50 adapter: run an instrument, persist, read back.

    Invokes the frozen Pi-50 instrument for the target artifact, writes the
    resulting unified records to the unified result-store, and queries them
    back from the same store.

    Usage:
        python -m adapters.pi50 <evaluated_artifact> <task> [--config '{"repo": ...}']
    """
    import argparse
    import json
    import sys

    import store

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="path to the artifact the instrument evaluates "
                                  "(for manifest_check: the committed PHASE1-MANIFEST.md)")
    ap.add_argument("task", choices=sorted(TASKS), help="pi50 instrument to run")
    ap.add_argument("--config", default="{}", help="JSON config (repo, python, timeout)")
    args = ap.parse_args(argv)

    config = json.loads(args.config)
    records = Pi50Adapter().run(args.model, args.task, config)
    if not records:
        print("pi50: no records produced", file=sys.stderr)
        return 2

    store.put(records)
    key = {
        "adapter": "pi50",
        "task": args.task,
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    }
    back = store.query(key)
    print(f"persisted {len(records)} pi50 records; queried back {len(back)} matching")
    for r in back:
        print(
            f"  {r['task']}:{r['metric']} = {r['value']} "
            f"(n={r['n']}, protocol={r['protocol']!r})"
        )
    if len(back) < len(records):
        print(f"warning: read back {len(back)} of {len(records)} records", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
