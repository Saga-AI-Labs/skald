"""Null-model calibration adapter (proposal §3.6: unknown-checkpoint metrics).

Scores deliberately degenerate references against the capability batteries so
every capability record has a floor to be measured against:

- ``null-stub``   — constant outputs: always answer "A" on mmlu, emit an empty
  completion on humaneval (scored by the real HumanEval executor, so the 0.0
  is measured, not asserted).
- ``null-random`` — seeded uniform-random letter guesses on mmlu (the chance
  floor); humaneval stays 0.0 — random tokens never satisfy executable tests,
  and generating+executing random code would spend sandbox time to prove it.

Interface: ``run(model, task, config) -> records[]`` (scaffold §5). ``model``
is accepted but ignored — the floor is model-independent given the items; the
identity anchor is a fixed digest per null kind (``null-model::stub`` /
``null-model::random``), so floor records group together and are never
silently compared with weighed-in records (distinct ``adapter`` + distinct
``protocol``). Items are caller-supplied inline via ``config["mmlu_items"]`` /
``config["humaneval_items"]`` (same shape as ``adapters.openai_compat``) —
pass the same items used for the real run; there is deliberately no network
fetch, so the floor is reproducible offline.

A uniform-random-*weight* checkpoint at the same shape is out of scope: this
project stays black-box (HTTP only), so the chance floor is over outputs, not
weights. The record protocol states exactly this.
"""

from __future__ import annotations

import hashlib
import math
import random
import socket
from datetime import datetime, timezone
from typing import Any, Sequence

from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.openai_compat import _check

TASKS = {"mmlu", "humaneval"}

_NULL_KINDS = ("stub", "random")

# Fixed identity anchors: the floor is one thing, not per-model.
_NULL_IDENTITY = {
    kind: hashlib.sha256(f"null-model::{kind}".encode("utf-8")).hexdigest()
    for kind in _NULL_KINDS
}

_DEFAULT_SEED = 42


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ci(score: float, n: int | None) -> tuple[float | None, float | None]:
    """Wald ~95% confidence interval for a proportion (clamped to [0, 1])."""
    if not n or n <= 0:
        return None, None
    p = max(0.0, min(1.0, float(score)))
    se = math.sqrt(p * (1.0 - p) / n)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def _self_sha256() -> str:
    h = hashlib.sha256()
    with open(__file__, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class NullModelAdapter(SuiteAdapter):
    """Degenerate-reference floor for the capability batteries."""

    def run(
        self,
        model: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        config = config or {}
        if task not in TASKS:
            raise ValueError(
                f"null_model: unsupported task {task!r}; choose from {sorted(TASKS)}"
            )
        kind = str(config.get("null_kind", "stub"))
        if kind not in _NULL_KINDS:
            raise ValueError(
                f"null_model: null_kind must be one of {list(_NULL_KINDS)}; got {kind!r}"
            )
        seed = int(config.get("seed", _DEFAULT_SEED))
        handlers = {"mmlu": self._run_mmlu, "humaneval": self._run_humaneval}
        return handlers[task](kind, seed, config)

    # --- task handlers ----------------------------------------------------

    def _run_mmlu(self, kind: str, seed: int, config: dict) -> list[dict[str, Any]]:
        items = config.get("mmlu_items")
        if not items:
            raise ValueError(
                "null_model: mmlu requires config['mmlu_items'] "
                "(same inline shape as openai_compat; no network fetch by design)"
            )
        if kind == "stub":
            guesses = ["A"] * len(items)
            procedure = 'constant answer "A" on every item'
        else:
            rng = random.Random(seed)
            guesses = [rng.choice("ABCD") for _ in items]
            procedure = f"seeded uniform-random letter guess (seed {seed}) per item"
        hits = sum(1 for it, g in zip(items, guesses) if g == it["gold"])
        n = len(items)
        score = hits / n
        ci_low, ci_high = _ci(score, n)
        protocol = (
            f"null-{kind} mmlu floor over {n} caller-supplied inline "
            f"items: {procedure}. Fixed null identity, no endpoint, no "
            f"weights — the chance/constant floor for letter-choice accuracy, "
            f"never comparable with weighed-in records."
        )
        return [
            self._record(
                kind=kind,
                task="mmlu",
                metric="accuracy",
                value=score,
                n=n,
                ci_low=ci_low,
                ci_high=ci_high,
                protocol=protocol,
                seed=seed if kind == "random" else None,
            )
        ]

    def _run_humaneval(self, kind: str, seed: int, config: dict) -> list[dict[str, Any]]:
        items = config.get("humaneval_items")
        if not items:
            raise ValueError(
                "null_model: humaneval requires config['humaneval_items'] "
                "(same inline shape as openai_compat; no network fetch by design)"
            )
        # Both degenerate completions are scored by the real executor rather
        # than asserted: the empty stub defines no entry point (KeyError ->
        # False), and random tokens cannot satisfy executable tests, so the
        # honest random floor is the same measured 0.0 without spending
        # sandbox time executing noise.
        exec_timeout = int(config.get("exec_timeout", 10))
        passed = sum(1 for p in items if _check(p, "", exec_timeout))
        n = len(items)
        score = passed / n
        ci_low, ci_high = _ci(score, n)
        procedure = (
            "empty completion on every problem, scored by the real HumanEval "
            "executor"
            if kind == "stub"
            else "random-token completions cannot satisfy executable tests; "
            "floor 0.0 by construction, executor-verified on the empty case"
        )
        protocol = (
            f"null-{kind} humaneval floor over {n} caller-supplied "
            f"inline problems: {procedure}. Fixed null identity, no endpoint, "
            f"no weights — never comparable with weighed-in records."
        )
        return [
            self._record(
                kind=kind,
                task="humaneval",
                metric="pass_at_1",
                value=score,
                n=n,
                ci_low=ci_low,
                ci_high=ci_high,
                protocol=protocol,
                seed=None,
            )
        ]

    # --- records ----------------------------------------------------------

    def _record(self, *, kind: str, task: str, metric: str, value: float,
                n: int | None, ci_low: float | None, ci_high: float | None,
                protocol: str, seed: int | None) -> dict[str, Any]:
        record = {
            "model_checkpoint_sha256": _NULL_IDENTITY[kind],
            "adapter": "null_model",
            "suite": "null_model",
            "task": task,
            "metric": metric,
            "value": value,
            "n": n,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "protocol": protocol,
            "created_at": _now(),
            "host": socket.gethostname(),
            "script_sha256": _self_sha256(),
            "seed": seed,
            "artifacts": [],
        }
        assert set(record) == set(RECORD_FIELDS), set(record) ^ set(RECORD_FIELDS)
        return record


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: score a degenerate floor, persist, read back.

    Usage:
        python -m adapters.null_model <label> <task>
            --items '<json list>' [--config '{"null_kind": "random"}']
    """
    import argparse
    import json
    import sys

    import store

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="run label (ignored; identity is the fixed null digest)")
    ap.add_argument("task", choices=sorted(TASKS), help="battery to floor")
    ap.add_argument("--items", required=True, help="JSON list of inline items")
    ap.add_argument("--config", default="{}", help="JSON config")
    args = ap.parse_args(argv)

    items = json.loads(args.items)
    config = json.loads(args.config)
    key = "mmlu_items" if args.task == "mmlu" else "humaneval_items"
    config[key] = items
    records = NullModelAdapter().run(args.model, args.task, config)
    if not records:
        print("null_model: no records produced", file=sys.stderr)
        return 2

    store.put(records)
    back = store.query({
        "adapter": "null_model",
        "task": args.task,
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    })
    print(f"persisted {len(records)} null_model records; queried back {len(back)} matching")
    for r in back:
        print(f"  {r['task']}:{r['metric']} = {r['value']} (n={r['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
