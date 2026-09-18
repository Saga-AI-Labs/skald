"""Saga general-purpose suite adapter (plan §4.1, task skald-adapter-saga).

Attribution: the benchmark machinery invoked here (`run_mmlu`, `run_gsm8k`,
`run_bbq`, `BenchmarkResult`, `FrozenModelWrapper`, `configs/evaluation.yaml`)
was written by the Saga project (Saga AI Labs) and is vendored unmodified
under ``vendor/saga_benchmarks/`` (see ``vendor/saga_benchmarks/PIN.md`` for
pin and license — AGPL-3.0, see root ``CREDITS.md`` for the combined-work
notice). Skald adds only this wrapper adapter (plus its HumanEval shim, which
exists because upstream Saga ships no `run_humaneval` runner). Upstream bugs
belong upstream.

Runs the vendored Saga evaluation machinery for a target model and returns
unified result records (plan §4.2):

- ``mmlu``     -> ``src/evaluation/benchmarks.py::run_mmlu``
- ``humaneval``-> adapter-side pass@1 shim over the saga ``humaneval`` config
                 entry (saga has no ``run_humaneval`` runner — see plan §8
                 OPEN-4; the shim is the thinnest possible execution of the
                 suite's own ``configs/evaluation.yaml`` ``humaneval`` entry,
                 ``num_fewshot: 0`` over ``openai/openai_humaneval``).

Interface: ``run(model, task, config) -> records[]`` (scaffold §5).  The
adapter invokes the existing saga suite code unmodified in a subprocess under
a saga-capable Python (torch/transformers/datasets): it imports saga's own
``run_mmlu`` and model loader (``src.models.loader.FrozenModelWrapper``) and
drives them over a local model path, then parses the printed result.  It does
not re-implement benchmark algorithms and does not persist — callers write
records to the unified store via ``store.put``.

Truthfulness contract: every emitted record carries all canonical
``RECORD_FIELDS`` keys, a non-empty 64-hex ``model_checkpoint_sha256`` (SHA-256
of the evaluated local model — a single file via ``identity.hash_checkpoint``,
a directory deterministically over its sorted contents) and a non-empty
``protocol`` label naming the exact evaluation protocol and its effective
parameters so numbers from different protocols are never silently compared.
Values are real measurements from saga's machinery over real model + data; no
cloud LLM participates (generation is a local transformers process).  If saga
reports ``num_samples == 0`` (e.g. its streaming dataset could not be loaded),
the adapter raises instead of emitting an empty/0-division record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from adapters import RECORD_FIELDS, SuiteAdapter
from identity import hash_checkpoint

# Vendored Saga benchmark subset (see vendor/saga_benchmarks/PIN.md). A fresh
# clone works out of the box; point config["repo"] at an external Saga
# checkout only to develop against upstream HEAD.
DEFAULT_REPO = (
    Path(__file__).resolve().parent.parent / "vendor" / "saga_benchmarks"
)

TASKS = {"mmlu", "humaneval"}

# Defaults mirror saga's own configuration (configs/evaluation.yaml + the
# BENCHMARK_CONFIGS / runner signatures in src/evaluation/benchmarks.py).
_DEFAULTS = {
    "mmlu": {"num_fewshot": 5, "max_samples": 2000, "max_new_tokens": 64},
    "humaneval": {"num_fewshot": 0, "max_samples": None, "max_new_tokens": 96},
}
_SEED = 42
_MMLU_SUBJECTS = 6
_EXEC_TIMEOUT = 10  # per-case watchdog for executing generated HumanEval code

_QUEUE = {"mmlu": "accuracy", "humaneval": "pass_at_1"}

_PROTOCOL = {
    "mmlu": (
        "saga run_mmlu (src/evaluation/benchmarks.py): {nf}-shot letter-choice "
        "accuracy over {subjects} cais/mmlu HF test subjects, greedy generation "
        "(max_new_tokens {mt}), max_samples {ms}, seed {seed}, model {mid}"
    ),
    "humaneval": (
        "saga humaneval config entry (configs/evaluation.yaml, {nf}-shot, "
        "openai/openai_humaneval): adapter-side greedy 0-shot pass@1 shim, "
        "max_new_tokens {mt}, max_samples {ms}, seed {seed}, "
        "{pt}s/case exec watchdog, model {mid}"
    ),
}

_RESULT_MARK = "SAGA_RESULT_JSON:"


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _checkpoint_sha256(model: str | Path) -> str:
    """SHA-256 identity of the evaluated model artifact.

    Single files use ``identity.hash_checkpoint``; directories are hashed
    deterministically over their sorted relative paths + contents.
    """
    path = Path(model)
    if path.is_file():
        return hash_checkpoint(path)
    return _hash_dir(path)


def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for rel in sorted(p.relative_to(path) for p in path.rglob("*") if p.is_file()):
        h.update(rel.as_posix().encode("utf-8"))
        h.update(b"\0")
        with open(path / rel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _ci(score: float, n: int | None) -> tuple[float | None, float | None]:
    """Wald ~95% confidence interval for a proportion (clamped to [0, 1])."""
    if not n or n <= 0:
        return None, None
    p = max(0.0, min(1.0, float(score)))
    se = math.sqrt(p * (1.0 - p) / n)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


class SagaAdapter(SuiteAdapter):
    """Adapter over the saga general-purpose evaluation machinery."""

    def run(
        self,
        model: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        config = config or {}
        repo = Path(config.get("repo", DEFAULT_REPO))
        if not repo.is_dir():
            raise FileNotFoundError(f"saga: suite repo not found: {repo}")
        python = self._resolve_python(repo, config)
        model = str(model)
        if not Path(model).exists():
            raise FileNotFoundError(f"saga: evaluated model not found: {model}")
        model = str(Path(model).resolve())
        if task not in TASKS:
            raise ValueError(
                f"saga: unsupported task {task!r}; choose from {sorted(TASKS)}"
            )
        handlers = {
            "mmlu": self._run_mmlu,
            "humaneval": self._run_humaneval,
        }
        return handlers[task](model, config, repo, python)

    # --- task handlers ----------------------------------------------------

    def _run_mmlu(
        self, model: str, config: dict[str, Any], repo: Path, python: str
    ) -> list[dict[str, Any]]:
        params = self._params("mmlu", config, model)
        driver = _mmlu_driver(repo=repo, model=model, **params)
        payload, stdout = self._capture(repo, python, driver, params["timeout"])
        metrics = self._metrics_from_payload("mmlu", payload)
        protocol = _PROTOCOL["mmlu"].format(
            nf=params["num_fewshot"],
            subjects=_MMLU_SUBJECTS,
            mt=params["max_new_tokens"],
            ms=_ms(params["max_samples"]),
            seed=params["seed"],
            mid=params["model_id"],
        )
        return self._records(
            model=model,
            task="mmlu",
            repo=repo,
            protocol=protocol,
            seed=payload["effective"]["seed"],
            metrics=metrics,
            stdout=stdout,
            artifact_dir=config.get("artifact_dir"),
        )

    def _run_humaneval(
        self, model: str, config: dict[str, Any], repo: Path, python: str
    ) -> list[dict[str, Any]]:
        params = self._params("humaneval", config, model)
        params["exec_timeout"] = int(config.get("exec_timeout", _EXEC_TIMEOUT))
        driver = _humaneval_driver(repo=repo, model=model, **params)
        payload, stdout = self._capture(repo, python, driver, params["timeout"])
        metrics = self._metrics_from_payload("humaneval", payload)
        protocol = _PROTOCOL["humaneval"].format(
            nf=params["num_fewshot"],
            mt=params["max_new_tokens"],
            ms=_ms(params["max_samples"]),
            seed=params["seed"],
            mid=params["model_id"],
            pt=params["exec_timeout"],
        )
        return self._records(
            model=model,
            task="humaneval",
            repo=repo,
            protocol=protocol,
            seed=payload["effective"]["seed"],
            metrics=metrics,
            stdout=stdout,
            artifact_dir=config.get("artifact_dir"),
        )

    # --- machinery --------------------------------------------------------

    def _resolve_python(self, repo: Path, config: dict[str, Any]) -> str:
        import shutil

        python = config.get("python")
        if python:
            cand = Path(python)
            if cand.is_file():
                return str(cand)
            resolved = shutil.which(python)
            if resolved:
                return resolved
            raise FileNotFoundError(f"saga: suite python not found: {python}")
        for cand in (repo / ".venv" / "bin" / "python",):
            if cand.is_file():
                return str(cand)
        resolved = shutil.which("python3")
        if resolved:
            return resolved
        raise FileNotFoundError(
            "saga: no python interpreter with suite deps found "
            "(set config['python'])"
        )

    def _params(
        self, task: str, config: dict[str, Any], model: str
    ) -> dict[str, Any]:
        defaults = _DEFAULTS[task]
        params = {
            "num_fewshot": int(config.get("num_fewshot", defaults["num_fewshot"])),
            "max_samples": config.get("max_samples", defaults["max_samples"]),
            "max_new_tokens": int(
                config.get("max_new_tokens", defaults["max_new_tokens"])
            ),
            "seed": int(config.get("seed", _SEED)),
            "timeout": int(config.get("timeout", 1800)),
            "model_id": str(config.get("model_id") or Path(model).name),
        }
        if params["max_samples"] is not None:
            params["max_samples"] = int(params["max_samples"])
        return params

    def _capture(
        self, repo: Path, python: str, driver: str, timeout: int
    ) -> tuple[dict[str, Any], str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo)
        try:
            proc = subprocess.run(
                [python, "-c", driver],
                cwd=repo,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"saga: suite driver timed out after {timeout}s"
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"saga: suite driver failed (rc={proc.returncode})\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        m = re.search(rf"^{re.escape(_RESULT_MARK)}(.*)$", proc.stdout, re.S | re.M)
        if not m:
            raise RuntimeError(
                "saga: suite driver printed no parseable result:\n"
                f"{proc.stdout[-2000:]}"
            )
        payload = json.loads(m.group(1))
        if not isinstance(payload.get("num_samples"), int) or payload["num_samples"] <= 0:
            raise RuntimeError(
                "saga: suite produced no samples "
                f"(n={payload.get('num_samples')!r}) — saga's streaming dataset "
                "could not be loaded (network/offline?); refusing to emit an "
                "empty record"
            )
        return payload, proc.stdout

    def _metrics_from_payload(
        self, task: str, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        score = float(payload["score"])
        n = int(payload["num_samples"])
        ci_low, ci_high = _ci(score, n)
        return [
            {
                "metric": _QUEUE[task],
                "value": score,
                "n": n,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        ]

    def _records(
        self,
        *,
        model: str,
        task: str,
        repo: Path,
        protocol: str,
        seed: int,
        metrics: list[dict[str, Any]],
        stdout: str,
        artifact_dir: str | Path | None,
    ) -> list[dict[str, Any]]:
        artifact = self._write_artifact(task, stdout, artifact_dir)
        script = repo / "src" / "evaluation" / "benchmarks.py"
        records = []
        for m in metrics:
            record = {
                "model_checkpoint_sha256": _checkpoint_sha256(model),
                "adapter": "saga",
                "suite": "saga",
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
                "artifacts": artifact,
            }
            assert set(record) == set(RECORD_FIELDS), set(record) ^ set(RECORD_FIELDS)
            records.append(record)
        return records

    def _write_artifact(
        self, task: str, stdout: str, artifact_dir: str | Path | None
    ) -> list[str]:
        default = Path(__file__).resolve().parent.parent / ".skald" / "saga_raw"
        out_dir = Path(artifact_dir) if artifact_dir else default
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"{task}_{stamp}.txt"
        path.write_text(stdout)
        return [f"{_file_sha256(path)}  {path}"]


def _ms(max_samples: int | None) -> str:
    return str(max_samples) if max_samples is not None else "null(all)"


def _common_driver_setup(repo: Path, model: str, mt: int) -> str:
    return (
        "import sys\n"
        f"sys.path.insert(0, {str(repo)!r})\n"
        "import json\n"
        "from src.evaluation.benchmarks import BenchmarkResult\n"
        "from src.models.loader import FrozenModelWrapper\n"
        "cfg = {"
        f"'id': 'under_test', 'hf_name': {str(model)!r}, 'commit': 'main', "
        "'hidden_dim': 768, 'dtype': 'float32'"
        "}\n"
        "wrapper = FrozenModelWrapper(cfg, encoding_device='cpu')\n"
        "wrapper.load_to_gpu()\n"
        "def generate_fn(prompt):\n"
        f"    return wrapper.generate([prompt], max_new_tokens={int(mt)})[0]\n"
    )


def _mmlu_driver(
    repo: Path, model: str, num_fewshot: int, max_samples: int | None,
    max_new_tokens: int, seed: int, timeout: int, model_id: str,
) -> str:
    ms = max_samples if max_samples is not None else 2000
    return (
        _common_driver_setup(repo, model, max_new_tokens)
        + "from src.evaluation.benchmarks import run_mmlu\n"
        + f"res = run_mmlu(generate_fn, num_fewshot={int(num_fewshot)}, "
        f"max_samples={int(ms)}, seed={int(seed)})\n"
        "out = {'name': res.name, 'score': float(res.score), "
        "'std_error': res.std_error, 'num_samples': int(res.num_samples), "
        "'category_scores': dict(res.category_scores), 'details': res.details, "
        "'effective': {'num_fewshot': " + str(int(num_fewshot)) + ", "
        "'max_samples': " + str(int(ms)) + ", 'seed': " + str(int(seed)) + ", "
        "'max_new_tokens': " + str(int(max_new_tokens)) + "}}\n"
        f"print({_RESULT_MARK!r} + ' ' + json.dumps(out, default=str))"
    )


def _humaneval_driver(
    repo: Path, model: str, num_fewshot: int, max_samples: int | None,
    max_new_tokens: int, seed: int, timeout: int, model_id: str,
    exec_timeout: int,
) -> str:
    ms = max_samples if max_samples is not None else "None"
    return (
        _common_driver_setup(repo, model, max_new_tokens)
        + "import random\n"
        "class _TO(Exception): pass\n"
        "def _alarm(sig, fr): raise _TO()\n"
        "import signal; signal.signal(signal.SIGALRM, _alarm)\n"
  "def _check(problem, gen_code):\n"
  "    ns = {}\n"
  "    lines = gen_code.split('\\n')\n"
  "    while lines and not lines[0].strip():\n"
  "        lines.pop(0)\n"
  "    body = '\\n'.join(lines)\n"
        "    if body.startswith('def '):\n"
        "        body = '\\n'.join(body.split('\\n')[1:]).lstrip('\\n')\n"
        f"    full = problem['prompt'] + '\\n' + body + '\\n' + problem['test']\n"
        f"    signal.setitimer(signal.ITIMER_REAL, {int(exec_timeout)})\n"
  "    try:\n"
  "        exec(full, ns)\n"
  "        ns['check'](ns[problem['entry_point']])\n"
  "        return True\n"  # HumanEval check() returns None; no exception means pass
        "    except _TO:\n"
        "        return False\n"
        "    except Exception:\n"
        "        return False\n"
        "    finally:\n"
        "        signal.setitimer(signal.ITIMER_REAL, 0)\n"
        "def run_humaneval(generate_fn, max_samples, seed):\n"
        "    from datasets import load_dataset\n"
        "    ds = load_dataset('openai/openai_humaneval', split='test', streaming=True)\n"
        "    problems = []\n"
        "    for ex in ds:\n"
        "        problems.append(ex)\n"
        "        if max_samples and len(problems) >= max_samples:\n"
        "            break\n"
        "    random.Random(seed).shuffle(problems)\n"
        "    passed = 0\n"
        "    for p in problems:\n"
        "        code = generate_fn(p['prompt'])\n"
        "        if _check(p, code):\n"
        "            passed += 1\n"
        "    n = len(problems)\n"
        "    return BenchmarkResult(name='humaneval', "
        "score=(passed / n if n else 0.0), num_samples=n, "
        "details={'passed': passed, 'total': n})\n"
        "res = run_humaneval(generate_fn, max_samples=" + str(ms) + ", seed="
        + str(int(seed)) + ")\n"
        "out = {'name': res.name, 'score': float(res.score), "
        "'std_error': res.std_error, 'num_samples': int(res.num_samples), "
        "'category_scores': dict(res.category_scores), 'details': res.details, "
        "'effective': {'num_fewshot': " + str(int(num_fewshot)) + ", "
        "'max_samples': " + str(ms) + ", 'seed': " + str(int(seed)) + ", "
        "'max_new_tokens': " + str(int(max_new_tokens)) + "}}\n"
        f"print({_RESULT_MARK!r} + ' ' + json.dumps(out, default=str))"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for the saga adapter: run a suite, persist, read back.

    Runs the saga evaluation machinery over the target model, writes the
    resulting unified records to the unified result-store, and queries them
    back from the same store.

    Usage:
        python -m adapters.saga <model_path> <task> [--config '{"max_samples": 200}']
    """
    import argparse
    import sys

    import store

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="path to the local model artifact under test")
    ap.add_argument(
        "task",
        choices=sorted(TASKS),
        help="saga suite to run (mmlu | humaneval)",
    )
    ap.add_argument(
        "--config",
        default="{}",
        help="JSON config (python, max_samples, num_fewshot, seed, timeout, ...)",
    )
    args = ap.parse_args(argv)

    config = json.loads(args.config)
    records = SagaAdapter().run(args.model, args.task, config)
    if not records:
        print("saga: no records produced", file=sys.stderr)
        return 2

    store.put(records)
    key = {
        "adapter": "saga",
        "task": args.task,
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    }
    back = store.query(key)
    print(f"persisted {len(records)} saga records; queried back {len(back)} matching")
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