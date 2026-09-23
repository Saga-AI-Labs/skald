"""Section 3.4 router utilisation (proposal §3.4: the question eval_router.py
does not ask).

``eval_router.py`` reports routing *accuracy* (argmin choice, confusion
matrix, routed/oracle ppl). It never asks whether every capacity slot still
does anything. This adapter measures, per layer and gate side (``x`` =
encoder latents, ``y`` = decoder latents), over routed-domain text streams:

- ``util_dead_slot_fraction`` — slots never selected, or selected with mass
  below the stated ``mass_threshold``;
- ``util_gate_entropy_nats`` — entropy of the load distribution over slots;
- ``util_gate_entropy_spread`` — max-minus-min entropy across domains;
- ``util_load_mean`` / ``util_load_median`` / ``util_load_p90`` — scalars
  describing the load histogram (the full per-slot histogram lives in the
  run artifact, since the store holds scalars).

One task: ``router_utilization``. The gate work runs in a subprocess under
a torch-capable interpreter (``config["python"]``, default
``<repo>/.venv/bin/python``) with the pin-versioned tree as cwd/``PYTHONPATH``
— the same shape as ``adapters.bdh_likelihood``. Skald-side code stays
stdlib-only. Contexts are caller-supplied inline texts or ``.txt`` files
(byte-level vocab: UTF-8 bytes are the token ids).

Conditional probe (proposal §3.4 correction): at ``k_sparse_ratio <= 0``
there is no top-k gate (plain ReLU), so the run raises instead of storing
records — the honest result is *not applicable*, not zero. Every record's
protocol carries ``k_sparse_ratio`` and the measured width, because a
ratio-based k is not width-invariant across a growth step.
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
from adapters.bdh_likelihood import load_contexts
from adapters.utilization_stats import (
    UTIL_METRICS,
    entropy_spread,
    load_histogram,
    summarize_gate,
)
from identity import hash_checkpoint
from store.runtime import runtime_digest

TASKS = {"router_utilization"}

DEFAULT_REPO = Path(__file__).resolve().parent.parent / "vendor" / "bdh_cl"

# Run artifacts (full per-slot histograms) live in Skald's own artifact
# dir, never in the vendor tree (which must stay byte-identical to upstream).
DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / ".skald" / "bdh_router_util"
)

_DEFAULT_TIMEOUT = 1200
_DEFAULT_SEED = 42
_DEFAULT_MASS_THRESHOLD = 0.0


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
            f"bdh_router_util: torch python not found: {python} "
            "(gate capture needs torch; override via config['python'])"
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
            f"bdh_router_util: {script} timed out after {timeout}s"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"bdh_router_util: {script} failed (rc={proc.returncode})\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        raise RuntimeError(
            f"bdh_router_util: {script} printed no usable JSON\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        ) from exc


class BdhRouterUtilAdapter(SuiteAdapter):
    """Gate-utilisation probe for k-sparse BDH checkpoints."""

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
                f"bdh_router_util: model-code repo not found: {repo}"
            )
        python = _torch_python(repo, config)
        if task not in TASKS:
            raise ValueError(
                f"bdh_router_util: unsupported task {task!r}; "
                f"choose from {sorted(TASKS)}"
            )
        timeout = int(config.get("timeout", _DEFAULT_TIMEOUT))
        seed = int(config.get("seed", _DEFAULT_SEED))
        return self._run_utilization(model, config, repo, python,
                                     timeout, seed)

    # --- tasks ------------------------------------------------------------

    def _run_utilization(self, model: str, config: dict, repo: Path,
                         python: Path, timeout: int, seed: int) -> list[dict]:
        ckpt = Path(model)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"bdh_router_util: checkpoint not found: {ckpt}"
            )
        threshold = float(config.get("mass_threshold",
                                     _DEFAULT_MASS_THRESHOLD))
        if threshold < 0.0:
            raise ValueError(
                f"bdh_router_util: mass_threshold must be >= 0; "
                f"got {threshold}"
            )
        domains = load_contexts(config)
        result = _run_helper(python, repo, "bdh_router_util_capture.py", {
            "checkpoint": str(ckpt),
            "domains": domains,
            "block_size": int(config.get("block_size", 0)),
        }, timeout)

        if result.get("not_applicable"):
            raise ValueError(
                "bdh_router_util: not applicable for this checkpoint "
                f"(k_sparse_ratio={result.get('k_sparse_ratio')}). "
                f"{result.get('reason', '')} No records stored — a zero "
                "would claim measured utilisation where no gate exists."
            )

        cap_cfg = result["config"]
        ratio = float(cap_cfg["k_sparse_ratio"])
        width = int(cap_cfg["width"])
        n_layer = int(cap_cfg["n_layer"])
        ckpt_sha = hash_checkpoint(ckpt)
        runtime = runtime_digest()

        artifact_dir = Path(config.get("artifact_dir") or
                            DEFAULT_ARTIFACT_DIR / _now().replace(":", ""))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        histograms = {
            domain: {
                f"L{entry['layer']}{entry['side']}": entry["counts"]
                for entry in info["layers"]
            }
            for domain, info in result["domains"].items()
        }
        (artifact_dir / "histograms.json").write_text(
            json.dumps({
                "created_by": "bdh_router_util router_utilization",
                "created_at": _now(),
                "checkpoint": str(ckpt),
                "checkpoint_sha256": ckpt_sha,
                "capture_config": cap_cfg,
                "mass_threshold": threshold,
                "histograms": histograms,
            }, indent=2) + "\n")
        (artifact_dir / "manifest.json").write_text(json.dumps({
            "created_by": "bdh_router_util router_utilization",
            "created_at": _now(),
            "checkpoint": str(ckpt),
            "checkpoint_sha256": ckpt_sha,
            "vendor_pin": _vendor_pin(repo),
            "capture_config": cap_cfg,
            "domains": {
                d: info["positions"] for d, info in result["domains"].items()
            },
            "runtime_sha256": runtime,
            "seed": seed,
        }, indent=2) + "\n")
        artifacts = [f"router-util::{artifact_dir}"]

        records = []
        # Per (layer, side): pool raw counts across domains for the
        # aggregate gate, keep per-domain loads for the entropy spread.
        by_gate: dict[tuple[int, str], dict[str, list[int]]] = {}
        positions: dict[str, int] = {}
        for domain, info in result["domains"].items():
            positions[domain] = int(info["positions"])
            for entry in info["layers"]:
                key = (int(entry["layer"]), entry["side"])
                by_gate.setdefault(key, {})[domain] = entry["counts"]
        for (layer, side), per_domain in sorted(by_gate.items()):
            total_pos = sum(positions[d] for d in per_domain)
            pooled_counts = [0] * width
            for counts in per_domain.values():
                for i, count in enumerate(counts):
                    pooled_counts[i] += count
            pooled = [c / total_pos for c in pooled_counts]
            agg = dict(summarize_gate(pooled, threshold))
            agg["util_positions"] = float(total_pos)
            per_domain_entropy = {}
            for domain in sorted(per_domain):
                loads = [c / positions[domain] for c in per_domain[domain]]
                summary = dict(summarize_gate(loads, threshold))
                summary["util_positions"] = float(positions[domain])
                per_domain_entropy[domain] = \
                    summary["util_gate_entropy_nats"]
                protocol = (
                    f"bdh_router_util router-utilization §3.4, domain "
                    f"{domain}, layer {layer} side {side}: top-k gate load "
                    f"over {positions[domain]} streamed positions; "
                    f"k_sparse_ratio={ratio} (k={cap_cfg['k_absolute']} of "
                    f"width {width}), mass_threshold={threshold}. Ratio-k "
                    f"is not width-invariant: compare only at equal "
                    f"(k_sparse_ratio, width)."
                )
                for metric in UTIL_METRICS:
                    records.append(self._record(
                        identity=ckpt_sha, task="router_utilization",
                        metric=f"{metric}:L{layer}{side}@{domain}",
                        value=summary[metric], n=positions[domain],
                        protocol=protocol, seed=seed, runtime=runtime,
                        artifacts=artifacts,
                    ))
            spread = entropy_spread(list(per_domain_entropy.values()))
            spread_protocol = (
                f"bdh_router_util router-utilization §3.4, layer {layer} "
                f"side {side} across {len(per_domain_entropy)} domains "
                f"({', '.join(f'{d}={e:.4f}' for d, e in sorted(per_domain_entropy.items()))}): "
                f"max-minus-min gate entropy; k_sparse_ratio={ratio}, "
                f"width {width}, mass_threshold={threshold}."
            )
            records.append(self._record(
                identity=ckpt_sha, task="router_utilization",
                metric=f"util_gate_entropy_spread:L{layer}{side}",
                value=spread, n=total_pos, protocol=spread_protocol,
                seed=seed, runtime=runtime, artifacts=artifacts,
            ))
            agg_protocol = (
                f"bdh_router_util router-utilization §3.4, layer {layer} "
                f"side {side} pooled over {len(per_domain_entropy)} domains "
                f"({total_pos} positions): k_sparse_ratio={ratio} "
                f"(k={cap_cfg['k_absolute']} of width {width}), "
                f"mass_threshold={threshold}."
            )
            for metric in UTIL_METRICS:
                if metric == "util_positions":
                    continue
                records.append(self._record(
                    identity=ckpt_sha, task="router_utilization",
                    metric=f"{metric}:L{layer}{side}@pooled",
                    value=agg[metric], n=total_pos, protocol=agg_protocol,
                    seed=seed, runtime=runtime, artifacts=artifacts,
                ))
            # Full per-slot histogram for this gate lives in the artifact;
            # record its bucket shape as scalars alongside.
            buckets = load_histogram(pooled)
            for i, frac in enumerate(buckets):
                records.append(self._record(
                    identity=ckpt_sha, task="router_utilization",
                    metric=f"util_load_hist_b{i}:L{layer}{side}@pooled",
                    value=frac, n=total_pos, protocol=agg_protocol,
                    seed=seed, runtime=runtime, artifacts=artifacts,
                ))
        if not records:
            raise RuntimeError(
                "bdh_router_util: capture helper returned no domains — "
                "refusing to store an empty utilisation run"
            )
        return records

    # --- records ----------------------------------------------------------

    def _record(self, *, identity: str, task: str, metric: str, value: float,
                n: int | None, protocol: str, seed: int | None,
                runtime: str | None, artifacts: list[str]) -> dict[str, Any]:
        record = {
            "model_checkpoint_sha256": identity,
            "adapter": "bdh_router_util",
            "suite": "bdh_router_util",
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
    """CLI: measure gate utilisation, persist, read back.

    Usage:
        python -m adapters.bdh_router_util <checkpoint> router_utilization
            --contexts '<json [{domain, text}...]>'
            [--config '{...}']
    """
    import argparse
    import sys

    import store

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="checkpoint to probe")
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--contexts", default="[]", help="JSON [{domain, text}]")
    ap.add_argument("--config", default="{}")
    args = ap.parse_args(argv)

    config = json.loads(args.config)
    config["contexts"] = json.loads(args.contexts)
    records = BdhRouterUtilAdapter().run(args.model, args.task, config)
    if not records:
        print("bdh_router_util: no records produced", file=sys.stderr)
        return 2

    store.put(records)
    back = store.query({
        "adapter": "bdh_router_util",
        "task": args.task,
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    })
    print(f"persisted {len(records)} bdh_router_util records; "
          f"queried back {len(back)} matching")
    for r in back:
        print(f"  {r['task']}:{r['metric']} = {r['value']} (n={r['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
