"""BDH-CL continual-learning suite adapter (plan §4.1, task skald-adapter-bdh-cl).

Attribution: the suite scripts invoked here (`eval_router.py`,
`domain_eval.py`, `p5_inchain_check.py`) were written by the BDH-CL project
(© 2025 Pathway Technology, Inc.; research fork of pathwaycom/bdh) and are
vendored unmodified under ``vendor/bdh_cl/`` (see ``vendor/bdh_cl/PIN.md``
for pin and license, ``vendor/bdh_cl/LICENSE.md`` for the upstream license).
Skald adds only this wrapper adapter. Upstream bugs belong upstream.

Runs the vendored BDH-CL eval scripts against a target checkpoint and
returns unified result records (plan §4.2):

- ``eval_router.py``     -> task "router"      (label-free likelihood routing,
                                                 territory / routing)
- ``domain_eval.py``     -> task "domain_eval" (per-domain held-out cold eval)
- ``p5_inchain_check.py``-> task "p5_inchain"  (P5 storage thesis on grown models)

Interface: ``run(model, task, config) -> records[]`` (scaffold §5).  The
adapter reuses the suite's own scripts (subprocess) and parses their printed
result tables; it does not re-implement the evals and does not persist —
callers write records to the unified store via ``store.put``.

Every emitted record carries all canonical ``RECORD_FIELDS`` keys, a non-empty
``model_checkpoint_sha256`` and a non-empty ``protocol`` label so numbers from
different evaluation protocols are never silently compared.
"""

from __future__ import annotations

import hashlib
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapters import RECORD_FIELDS, SuiteAdapter
from identity import hash_checkpoint
from store.runtime import runtime_digest

# Vendored BDH-CL evaluation subset (see vendor/bdh_cl/PIN.md). A fresh clone
# works out of the box; point config["repo"] at an external BDH-CL checkout
# only to develop against upstream HEAD.
DEFAULT_REPO = Path(__file__).resolve().parent.parent / "vendor" / "bdh_cl"

# Raw stdout artifacts live in Skald's own artifact dir, never in the vendor
# tree (which must stay byte-identical to upstream).
DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / ".skald" / "bdh_raw"
)

TASKS = {"router", "domain_eval", "p5_inchain"}

# Protocol labels name the exact measurement protocol per task so cross-protocol
# comparisons are never silent (plan §4.2 "protocol").
PROTOCOLS = {
    "router": (
        "random-crop cold likelihood routing, teacher-forced, "
        "label-free argmin over prefix routes"
    ),
    "domain_eval": "random-crop cold held-out eval, teacher-forced per-domain streams",
    "p5_inchain": (
        "checkpoint-structural P5 in-chain check "
        "(bit-exact masked base block, zero moments, grown-nonzero)"
    ),
}

_FIXED_SEEDS = {"router": 1234, "domain_eval": 1234}

_ROUTES_RE = re.compile(r"routes=\[([\d,\s]+)\]")
_PPL_ROW_RE = re.compile(r"^\s*(\S+)\s+([\d.]+)(?:\s+([\d.]+))?$")
_JOINT_RE = re.compile(r"joint full-width reference: ppl ([\d.]+)")
_CONF_ROW_RE = re.compile(r"^\s*(\S+)\s+([\d\s]+)$")
_DOMAIN_RE = re.compile(r"^\s*(\S+): nll ([\d.]+) \| ppl ([\d.]+)$")
_P51_RE = re.compile(r"^\s*(\S+)\s+(BIT-EXACT|DIFFERS)$")
_P52_RE = re.compile(r"grown segment: nonzero=(True|False) max\|w\|=([\d.eE+-]+)")
_VERDICT_RE = re.compile(r"P5-VERDICT: (PASS|FAIL)")


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _domain_spec(domains: Any) -> str:
    """Normalise config['domains'] to the suite's ``name:path,name:path`` spec.

    Accepts a comma-joined string, a sequence of ``name:path`` strings, or a
    ``{name: path, ...}`` mapping.
    """
    if isinstance(domains, dict):
        return ",".join(f"{name}:{path}" for name, path in domains.items())
    if isinstance(domains, str):
        return domains
    return ",".join(str(d) for d in domains)


def _routes_spec(routes: Any) -> str:
    if isinstance(routes, str):
        return routes
    return ",".join(str(int(r)) for r in routes)


class BdhClAdapter(SuiteAdapter):
    """Adapter over the BDH-CL suite scripts in ``DEFAULT_REPO``."""

    def run(
        self,
        model: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        config = config or {}
        repo = Path(config.get("repo", DEFAULT_REPO))
        if not repo.is_dir():
            raise FileNotFoundError(f"bdh_cl: BDH-CL suite repo not found: {repo}")
        python = config.get("python") or str(repo / ".venv" / "bin" / "python")
        if not Path(python).is_file():
            raise FileNotFoundError(
                f"bdh_cl: suite python not found: {python} "
                "(the eval scripts need a torch-capable interpreter; "
                "override via config['python'])"
            )
        python = Path(python)
        model = str(model)
        if not Path(model).is_file():
            raise FileNotFoundError(f"bdh_cl: model checkpoint not found: {model}")
        if task not in TASKS:
            raise ValueError(
                f"bdh_cl: unsupported task {task!r}; choose from {sorted(TASKS)}"
            )

        handlers = {
            "router": self._run_router,
            "domain_eval": self._run_domain_eval,
            "p5_inchain": self._run_p5_inchain,
        }
        return handlers[task](model, config, repo, python)

    # --- task handlers ----------------------------------------------------

    def _run_router(
        self, model: str, config: dict[str, Any], repo: Path, python: Path
    ) -> list[dict[str, Any]]:
        routes = _routes_spec(config.get("routes") or "")
        domains = _domain_spec(config.get("domains") or "")
        if not routes or not domains:
            raise ValueError(
                "bdh_cl: router task requires config['routes'] and config['domains'] "
                "(comma-separated 'name:path' domain spec)"
            )
        window = int(config.get("window", 128))
        crops = int(config.get("crops", 40))
        batch = int(config.get("batch", 4))
        args = [
            str(python),
            "scripts/eval_router.py",
            model,
            "--routes",
            routes,
            "--domains",
            domains,
            "--window",
            str(window),
            "--crops",
            str(crops),
            "--batch",
            str(batch),
        ]
        oracle = config.get("oracle_routes")
        if oracle:
            args += ["--oracle-routes", str(oracle)]
        out = self._capture(repo, args, int(config.get("timeout", 3600)))
        parsed = self._parse_router(out, CROPS=crops)
        return self._records(
            model=model,
            task="router",
            repo=repo,
            script=repo / "scripts" / "eval_router.py",
            protocol=PROTOCOLS["router"],
            seed=_FIXED_SEEDS["router"],
            metrics=parsed,
            stdout=out,
            n_default=crops,
            artifact_dir=config.get("artifact_dir"),
        )

    def _run_domain_eval(
        self, model: str, config: dict[str, Any], repo: Path, python: Path
    ) -> list[dict[str, Any]]:
        domains = _domain_spec(config.get("domains") or "")
        if not domains:
            raise ValueError(
                "bdh_cl: domain_eval task requires config['domains'] "
                "(comma-separated 'name:path' domain spec)"
            )
        mb = int(config.get("mb", 30))
        batch = int(config.get("batch", 8))
        iters = int(config.get("iters", 100))
        args = [
            str(python),
            "scripts/domain_eval.py",
            model,
            domains,
            str(mb),
            "--batch",
            str(batch),
            "--iters",
            str(iters),
        ]
        out = self._capture(repo, args, int(config.get("timeout", 3600)))
        parsed = self._parse_domain_eval(out)
        return self._records(
            model=model,
            task="domain_eval",
            repo=repo,
            script=repo / "scripts" / "domain_eval.py",
            protocol=PROTOCOLS["domain_eval"],
            seed=_FIXED_SEEDS["domain_eval"],
            metrics=parsed,
            stdout=out,
            n_default=iters,
            artifact_dir=config.get("artifact_dir"),
        )

    def _run_p5_inchain(
        self, model: str, config: dict[str, Any], repo: Path, python: Path
    ) -> list[dict[str, Any]]:
        parent = config.get("parent")
        if not parent:
            raise ValueError(
                "bdh_cl: p5_inchain task requires config['parent'] "
                "(base-phase exit checkpoint; model= is the grown child)"
            )
        if not Path(str(parent)).is_file():
            raise FileNotFoundError(f"bdh_cl: parent checkpoint not found: {parent}")
        args = [
            str(python),
            "scripts/p5_inchain_check.py",
            str(parent),
            model,
        ]
        out = self._capture(repo, args, int(config.get("timeout", 1800)))
        parsed = self._parse_p5(out)
        return self._records(
            model=model,
            task="p5_inchain",
            repo=repo,
            script=repo / "scripts" / "p5_inchain_check.py",
            protocol=PROTOCOLS["p5_inchain"],
            seed=None,
            metrics=parsed,
            stdout=out,
            n_default=1,
            artifact_dir=config.get("artifact_dir"),
        )

    # --- machinery --------------------------------------------------------

    def _capture(self, repo: Path, args: list[str], timeout: int) -> str:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo)
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
                f"bdh_cl: suite script timed out after {timeout}s"
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"bdh_cl: suite script failed (rc={proc.returncode})\n"
                f"cmd: {' '.join(args)}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        return proc.stdout

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
                "adapter": "bdh_cl",
                "suite": "bdh_cl",
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
                "runtime_sha256": runtime_digest(),
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

    # --- parsers (testable, reused) --------------------------------------

    def _parse_router(self, out: str, *, CROPS: int) -> list[dict[str, Any]]:
        lines = out.splitlines()
        routes = [
            int(r)
            for r in re.split(r"[,\s]+", _ROUTES_RE.search(out).group(1).strip())
            if r
        ]
        metrics: list[dict[str, Any]] = []

        conf_start = next(
            (i for i, ln in enumerate(lines) if "cols=routed prefix width):" in ln),
            None,
        )
        if conf_start is not None:
            rows = []
            for ln in lines[conf_start + 2 :]:
                if not ln.strip():
                    break
                name, rest = _CONF_ROW_RE.match(ln).groups()
                counts = [int(c) for c in rest.split()]
                rows.append((name, counts))
            for name, counts in rows:
                for width, count in zip(routes, counts):
                    metrics.append(
                        {
                            "metric": f"routing_confusion_{width}",
                            "value": float(count),
                            "n": CROPS,
                        }
                    )

        ppl_reached = False
        for ln in lines:
            if ppl_reached and not ln.strip():
                break
            tokens = ln.split()
            if (
                len(tokens) >= 2
                and tokens[0] == "domain"
                and tokens[1].startswith("routed")
            ):
                ppl_reached = True
                continue
            if ppl_reached:
                m = _PPL_ROW_RE.match(ln)
                if m:
                    name, routed, oracle = m.groups()
                    metrics.append(
                        {"metric": "routed_perplexity", "value": float(routed), "n": CROPS}
                    )
                    if oracle is not None:
                        metrics.append(
                            {
                                "metric": "oracle_perplexity",
                                "value": float(oracle),
                                "n": CROPS,
                            }
                        )

        joint = _JOINT_RE.search(out)
        if joint:
            metrics.append(
                {
                    "metric": "joint_fullwidth_perplexity",
                    "value": float(joint.group(1)),
                    "n": CROPS * len(routes),
                }
            )
        return metrics

    def _parse_domain_eval(self, out: str) -> list[dict[str, Any]]:
        metrics = []
        for ln in out.splitlines():
            m = _DOMAIN_RE.match(ln)
            if m:
                name, nll, ppl = m.groups()
                metrics.append({"metric": f"{name}:nll", "value": float(nll)})
                metrics.append({"metric": f"{name}:ppl", "value": float(ppl)})
        return metrics

    def _parse_p5(self, out: str) -> list[dict[str, Any]]:
        metrics = []
        for ln in out.splitlines():
            m = _P51_RE.match(ln)
            if m:
                comp, verdict = m.groups()
                metrics.append(
                    {
                        "metric": f"p5_bit_exact_{comp}",
                        "value": 1.0 if verdict == "BIT-EXACT" else 0.0,
                    }
                )
            m2 = _P52_RE.search(ln)
            if m2:
                metrics.append(
                    {"metric": "p5_grown_nonzero", "value": 1.0 if m2.group(1) == "True" else 0.0}
                )
                metrics.append({"metric": "p5_grown_max_abs_w", "value": float(m2.group(2))})
            m3 = _VERDICT_RE.search(ln)
            if m3:
                metrics.append(
                    {"metric": "p5_verdict", "value": 1.0 if m3.group(1) == "PASS" else 0.0}
                )
        return metrics


if __name__ == "__main__":
    from adapters.__main__ import main

    raise SystemExit(main())