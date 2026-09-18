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

PROTOCOLS = {
    "manifest_check": (
        "phase-1 artifact manifest consistency check "
        "(git-tracked table vs committed PHASE1-MANIFEST.md, "
        "generation stamp excluded, 2026-09-13 CI policy)"
    ),
}

_VERDICT_RE = re.compile(r"(manifest up to date|MANIFEST STALE - regenerate)")


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Pi50Adapter(SuiteAdapter):
    """Adapter over the frozen Pi-50 phase-1 instruments in ``DEFAULT_REPO``."""

    def run(
        self,
        model: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        config = config or {}
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
        if task not in TASKS:
            raise ValueError(
                f"pi50: unsupported task {task!r}; choose from {sorted(TASKS)}"
            )
        handlers = {
            "manifest_check": self._run_manifest_check,
        }
        return handlers[task](model, config, repo, python)

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
                "model_checkpoint_sha256": hash_checkpoint(model),
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
