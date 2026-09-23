"""Path A likelihood parity (proposal §3.2, Path A: hidden states, score offline).

Teacher-forced KL(reference ‖ candidate) over sealed held-out contexts,
reported as a distribution per domain (mean, median, p95, p99, p99.9, max,
CVaR95) — never a bare mean. Two tasks:

- ``capture_reference`` — one fp32 forward pass over sealed contexts with
  the reference checkpoint, saving post-LN trunk ``x`` per position (fp16)
  plus the ``lm_head`` once (fp32) into a reference bundle. The head is a
  plain matmul (``bdh.py:282``), so logits are recomputed offline exactly;
  the capture self-checks ``x @ head ≈ logits`` every call and fails loudly
  if the pin ever moves the head.
- ``likelihood_parity`` — the candidate checkpoint runs the same contexts;
  per-token KL is computed in the torch subprocess, quantiles here (stdlib).

The torch work runs in a subprocess under a torch-capable interpreter
(``config["python"]``, default ``<repo>/.venv/bin/python``) with the
pin-versioned tree as cwd/``PYTHONPATH`` — the same shape as
``adapters.bdh_cl``. Skald-side code stays stdlib-only. Contexts are
caller-supplied inline texts or ``.txt`` files (byte-level vocab: UTF-8
bytes are the token ids); parquet sources stay an operator conversion step.

Storage budget before running: positions × n_embd × 2 bytes per domain
(fp16 ``x``), plus the head once — tens of GB at external-suite scale.
``max_contexts``/``max_chars`` cap the capture; the bundle manifest records
what was actually sealed.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.likelihood_stats import KL_METRICS, summarize_kl
from identity import hash_checkpoint
from store.runtime import runtime_digest

TASKS = {"capture_reference", "likelihood_parity"}

DEFAULT_REPO = Path(__file__).resolve().parent.parent / "vendor" / "bdh_cl"

_DEFAULT_TIMEOUT = 1200
_DEFAULT_SEED = 42


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _self_sha256() -> str:
    h = hashlib.sha256()
    with open(__file__, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _vendor_pin(repo: Path) -> str | None:
    """Pinned upstream commit from the vendored PIN.md (or None)."""
    try:
        for line in (repo / "PIN.md").read_text().splitlines():
            if "Pinned commit:" in line:
                return line.split("Pinned commit:")[1].split()[0].strip("`")
    except OSError:
        pass
    return None


def _torch_python(repo: Path, config: dict) -> Path:
    python = Path(config.get("python") or repo / ".venv" / "bin" / "python")
    if not python.is_file():
        raise FileNotFoundError(
            f"bdh_likelihood: torch python not found: {python} "
            "(capture/scoring need torch; override via config['python'])"
        )
    return python


def _run_helper(python: Path, repo: Path, script: str, argv: dict,
                timeout: int) -> Any:
    """Run a torch helper script; return its stdout JSON. Fail loudly."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    try:
        proc = subprocess.run(
            [str(python), str(Path(__file__).parent / script), json.dumps(argv)],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"bdh_likelihood: {script} timed out after {timeout}s"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"bdh_likelihood: {script} failed (rc={proc.returncode})\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        raise RuntimeError(
            f"bdh_likelihood: {script} printed no usable JSON\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        ) from exc


def load_contexts(config: dict) -> list[dict[str, Any]]:
    """Collect sealed contexts as [{domain, texts}] from inline/files.

    Shared by both parity paths (Path A here, Path B in ``openai_compat``):
    the sealed context set is what makes reference and candidate comparable.
    """
    domains: dict[str, list[str]] = {}
    for item in config.get("contexts", []):
        domains.setdefault(item["domain"], []).append(item["text"])
    for item in config.get("context_files", []):
        raw = Path(item["path"]).read_bytes()
        domains.setdefault(item["domain"], []).append(
            raw.decode("utf-8", errors="replace")
        )
    max_contexts = config.get("max_contexts")
    max_chars = config.get("max_chars")
    out = []
    for domain in sorted(domains):
        texts = domains[domain]
        if max_contexts is not None:
            texts = texts[: int(max_contexts)]
        if max_chars is not None:
            texts = [t[: int(max_chars)] for t in texts]
        texts = [t for t in texts if t.strip()]
        if texts:
            out.append({"domain": domain, "texts": texts})
    if not out:
        raise ValueError(
            "bdh_likelihood: no usable contexts (config['contexts'] inline "
            "{domain, text} and/or config['context_files'] {domain, path})"
        )
    return out


class BdhLikelihoodAdapter(SuiteAdapter):
    """Reference capture + parity scoring for BDH-architecture checkpoints."""

    def run(
        self,
        model: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        config = config or {}
        repo = Path(config.get("repo", DEFAULT_REPO))
        if not repo.is_dir():
            raise FileNotFoundError(
                f"bdh_likelihood: model-code repo not found: {repo}"
            )
        python = _torch_python(repo, config)
        if task not in TASKS:
            raise ValueError(
                f"bdh_likelihood: unsupported task {task!r}; "
                f"choose from {sorted(TASKS)}"
            )
        timeout = int(config.get("timeout", _DEFAULT_TIMEOUT))
        seed = int(config.get("seed", _DEFAULT_SEED))
        if task == "capture_reference":
            return self._run_capture(model, config, repo, python, timeout, seed)
        return self._run_parity(model, config, repo, python, timeout, seed)

    # --- tasks ------------------------------------------------------------

    def _run_capture(self, model: str, config: dict, repo: Path,
                     python: Path, timeout: int, seed: int) -> list[dict]:
        ckpt = Path(model)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"bdh_likelihood: reference checkpoint not found: {ckpt}"
            )
        bundle = Path(config.get("bundle_dir") or
                      f"reference-{ckpt.stem}-{_now().replace(':', '')}")
        bundle.mkdir(parents=True, exist_ok=True)
        domains = load_contexts(config)
        result = _run_helper(python, repo, "bdh_likelihood_capture.py", {
            "checkpoint": str(ckpt),
            "bundle_dir": str(bundle),
            "domains": domains,
            "block_size": int(config.get("block_size", 0)),
            "dtype": config.get("dtype", "float16"),
        }, timeout)

        ref_sha = hash_checkpoint(ckpt)
        runtime = runtime_digest()
        (bundle / "manifest.json").write_text(json.dumps({
            "created_by": "bdh_likelihood capture_reference",
            "created_at": _now(),
            "reference_checkpoint": str(ckpt),
            "reference_checkpoint_sha256": ref_sha,
            "vendor_pin": _vendor_pin(repo),
            "bundle_config": result["config"],
            "domains": result["domains"],
            "runtime_sha256": runtime,
            "seed": seed,
        }, indent=2) + "\n")

        protocol = (
            f"bdh_likelihood capture-reference path-a over "
            f"{len(result['domains'])} sealed domains into {bundle.name}: "
            f"one fp32 forward pass of unquantised {ref_sha[:12]}…, post-LN "
            f"trunk x stored {result['config']['stored_dtype']}, head fp32, "
            f"self-check max abs logit diff reported per domain. "
            f"Capture once, reuse for every candidate arm."
        )
        selfcheck = float(result["config"]["selfcheck_max_abs_logit_diff"])
        records = []
        for domain in sorted(result["domains"]):
            info = result["domains"][domain]
            records.append(self._record(
                identity=ref_sha, task="capture_reference",
                metric="reference_positions", value=float(info["positions"]),
                n=info["positions"], protocol=protocol, seed=seed,
                runtime=runtime, artifacts=[f"reference-bundle::{bundle}"],
            ))
            records.append(self._record(
                identity=ref_sha, task="capture_reference",
                metric="reference_selfcheck_max_abs_logit_diff",
                value=selfcheck,
                n=info["positions"], protocol=protocol, seed=seed,
                runtime=runtime, artifacts=[f"reference-bundle::{bundle}"],
            ))
        return records

    def _run_parity(self, model: str, config: dict, repo: Path,
                    python: Path, timeout: int, seed: int) -> list[dict]:
        candidate = Path(model)
        if not candidate.is_file():
            raise FileNotFoundError(
                f"bdh_likelihood: candidate checkpoint not found: {candidate}"
            )
        bundle = Path(config.get("reference_bundle") or "")
        if not (bundle / "manifest.json").is_file():
            raise FileNotFoundError(
                "bdh_likelihood: likelihood_parity requires "
                "config['reference_bundle'] pointing at a capture_reference "
                f"bundle (got {bundle})"
            )
        manifest = json.loads((bundle / "manifest.json").read_text())
        kl_by_domain = _run_helper(python, repo, "bdh_likelihood_score.py", {
            "bundle_dir": str(bundle),
            "candidate": str(candidate),
        }, timeout)

        cand_sha = hash_checkpoint(candidate)
        ref_sha = manifest["reference_checkpoint_sha256"]
        runtime = runtime_digest()
        artifacts = [
            f"reference-bundle::{bundle}",
            f"candidate::{candidate}",
        ]
        records = []
        for domain in sorted(kl_by_domain):
            summary = summarize_kl(kl_by_domain[domain])
            protocol = (
                f"bdh_likelihood likelihood-parity path-a, domain {domain}: "
                f"per-token KL({ref_sha[:12]}… ‖ {cand_sha[:12]}…) over "
                f"{summary['n']} sealed positions from bundle {bundle.name}; "
                f"reference logits recomputed offline from stored x, "
                f"candidate full fp32 forward. Weights only — kernels not "
                f"measured (that is Path B). Same-checkpoint self-parity "
                f"should read ~0 and validates the pipeline."
            )
            for metric in KL_METRICS:
                records.append(self._record(
                    identity=cand_sha, task="likelihood_parity",
                    metric=f"{metric}@{domain}", value=summary[metric],
                    n=summary["n"], protocol=protocol, seed=seed,
                    runtime=runtime, artifacts=artifacts,
                ))
            records.append(self._record(
                identity=cand_sha, task="likelihood_parity",
                metric=f"n_tokens@{domain}", value=float(summary["n"]),
                n=summary["n"], protocol=protocol, seed=seed,
                runtime=runtime, artifacts=artifacts,
            ))
        if not records:
            raise RuntimeError(
                "bdh_likelihood: score helper returned no domains — "
                "refusing to store an empty parity run"
            )
        return records

    # --- records ----------------------------------------------------------

    def _record(self, *, identity: str, task: str, metric: str, value: float,
                n: int | None, protocol: str, seed: int | None,
                runtime: str | None, artifacts: list[str]) -> dict[str, Any]:
        record = {
            "model_checkpoint_sha256": identity,
            "adapter": "bdh_likelihood",
            "suite": "bdh_likelihood",
            "task": task,
            "metric": metric,
            "value": value,
            "n": n,
            "ci_low": None,
            "ci_high": None,
            "protocol": protocol,
            "created_at": _now(),
            "host": socket.gethostname(),
            "script_sha256": _self_sha256(),
            "runtime_sha256": runtime,
            "seed": seed,
            "artifacts": artifacts,
        }
        assert set(record) == set(RECORD_FIELDS), set(record) ^ set(RECORD_FIELDS)
        return record


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: capture a reference bundle or score parity, persist, read back.

    Usage:
        python -m adapters.bdh_likelihood <checkpoint> capture_reference
            --contexts '<json [{domain, text}...]>' --bundle-out <dir>
            [--config '{...}']
        python -m adapters.bdh_likelihood <candidate> likelihood_parity
            --bundle <dir> [--config '{...}']
    """
    import argparse
    import sys

    import store

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="reference checkpoint (capture) or candidate (parity)")
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--contexts", default="[]", help="JSON [{domain, text}]")
    ap.add_argument("--bundle-out", default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--config", default="{}")
    args = ap.parse_args(argv)

    config = json.loads(args.config)
    if args.task == "capture_reference":
        config["contexts"] = json.loads(args.contexts)
        if args.bundle_out:
            config["bundle_dir"] = args.bundle_out
    else:
        if args.bundle:
            config["reference_bundle"] = args.bundle
    records = BdhLikelihoodAdapter().run(args.model, args.task, config)
    if not records:
        print("bdh_likelihood: no records produced", file=sys.stderr)
        return 2

    store.put(records)
    back = store.query({
        "adapter": "bdh_likelihood",
        "task": args.task,
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    })
    print(f"persisted {len(records)} bdh_likelihood records; "
          f"queried back {len(back)} matching")
    for r in back:
        print(f"  {r['task']}:{r['metric']} = {r['value']} (n={r['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
