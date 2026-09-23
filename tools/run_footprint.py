#!/usr/bin/env python3
"""Run a capability footprint against an OpenAI-compatible endpoint via the skald API.

Why a script rather than ad-hoc calls
--------------------------------------
The whole point is that Qwen3.8-Flash-Next and GLM-5.3-Flash-EXL3-K2 must be
measured with the *identical* instrument. Every knob that changes a score is
pinned here and echoed into the record's ``protocol`` string, because two of
them bit me in practice:

* ``num_fewshot`` defaults to 5 and is invisible unless you ask. Measured on
  identical MMLU items: 0-shot 0.20 vs 5-shot 0.40 -- a 2x swing, both stored
  under ``metric="accuracy"``, ``task="mmlu"``. 0.20 on four-option MC is
  *exactly* chance, so the 0-shot number is not a weak result, it is no
  result. Pinned to 5 below, deliberately and non-obviously.
* ``max_samples`` was once passed as 20 and the run silently used the default
  100. Pinned here.

Axes covered (see docs/plans/2026-09-23_benchmark-coverage-and-new-tasks.md):

    coding            humaneval        execution-verified pass@1
    math              gsm8k            free-form worked solutions (NOT multiple choice)
    logic/reasoning   mmlu             subject_set=reasoning (formal_logic, fallacies, maths)
    general knowledge mmlu + simpleqa  knowledge subjects + short-form factuality
    agentic           tool_use         multi-step tool driving + termination
    (gate)            determinism      reproducibility at temperature 0

Usage
-----
    python tools/run_footprint.py --base-url http://127.0.0.1:8888/v1 \
        --model qwen3.8-flash-next [--n 50] [--api http://127.0.0.1:8000]

Jobs are submitted one at a time and polled, on purpose: the endpoint serves
one model, so firing all jobs concurrently would make each measure the others'
queueing delay rather than the model's capability.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# --- the pinned protocol -------------------------------------------------
# Change any of these and you have a different benchmark, whose results are
# not comparable with the runs recorded in the store.
PINNED = {
    "mmlu":      {"num_fewshot": 5, "max_samples": 100, "max_tokens": 64,
                  "subject_set": "reasoning"},
    # Budgets must clear the deliberation: this endpoint spends ~1100 tokens
    # thinking before it emits content, so a small max_tokens leaves content
    # empty on every item and the task scores 0.0 for a reason unrelated to
    # capability. See budget_starved, which reports exactly that.
    "gsm8k":     {"max_samples": 50, "max_tokens": 2048},
    "simpleqa":  {"max_samples": 50, "max_tokens": 2048},
    "humaneval": {"max_samples": 20, "max_tokens": 320},
    "tool_use":  {"max_samples": 12, "steps": 4, "max_tool_calls": 24,
                  "max_tokens": 320},
    "determinism": {"repeats": 6, "max_tokens": 320},
}
SEED = 42
# One prompt family for the determinism gate; it measures reproducibility of
# *this* shape of request, so it should look like the scored tasks.
DET_PROMPT = ("A train travels 60 km/h for 30 minutes. How far does it go? "
              "Work it out step by step, then give the answer as '#### <number>'.")
ORDER = ["determinism", "gsm8k", "mmlu", "simpleqa", "humaneval", "tool_use"]


def _post(api: str, op: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{api}/api/v1/{op}", data=data,
                                headers={"Content-Type": "application/json"},
                                method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(api: str, op: str, **params: str) -> dict:
    q = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    with urllib.request.urlopen(f"{api}/api/v1/{op}?{q}", timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_one(api: str, base_url: str, model: str, task: str, n: int | None,
            timeout_s: int) -> list[dict]:
    cfg = dict(PINNED[task])
    if n is not None and "max_samples" in cfg:
        cfg["max_samples"] = n
    cfg["model"] = model
    cfg["seed"] = SEED
    if task == "determinism":
        cfg["prompt"] = DET_PROMPT

    job = _post(api, "run_benchmark",
                {"adapter": "openai_compat", "task": task,
                 "model": base_url, "config": cfg})
    jid = (job.get("job_id") or [None])[0]
    if not jid:
        print(f"    !! {task}: no job id: {job}")
        return []

    t0 = time.time()
    while True:
        st = _get(api, "job_status", job_id=jid)
        g = lambda k: (st.get(k) or [None])[0]
        status = g("status")
        if status in ("done", "failed", "orphaned"):
            break
        if time.time() - t0 > timeout_s:
            print(f"    !! {task}: exceeded {timeout_s}s (still {status})")
            return []
        time.sleep(5)

    if status != "done":
        print(f"    !! {task}: {status}: {str(g('error'))[:200]}")
        return []

    recs = _get(api, "query_results", adapter="openai_compat", task=task).get(
        "records", [])
    mine = [r for r in recs if r.get("protocol", "").find(f"model {model}:") >= 0]
    return mine or recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True,
                    help="endpoint under test, e.g. http://127.0.0.1:8888/v1")
    ap.add_argument("--model", required=True, help="served model id")
    ap.add_argument("--api", default="http://127.0.0.1:8000",
                    help="skald API (default %(default)s)")
    ap.add_argument("--n", type=int, default=None,
                    help="override max_samples for sample-limited tasks")
    ap.add_argument("--out", default=None,
                    help="write the collected records as JSON here")
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"subset of {ORDER}")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    tasks = [t for t in ORDER if not args.only or t in args.only]
    print(f"\n# footprint: {args.model} @ {args.base_url}")
    print(f"# pinned: seed={SEED} " + " ".join(
        f"{t}={PINNED[t].get('max_samples', PINNED[t].get('repeats', '-'))}"
        for t in tasks))
    print(f"# NOTE: determinism is a GATE. If reproducible=0, every accuracy "
          f"below carries run-to-run variance beyond its own CI.\n")

    rows: list[tuple[str, str, float, int | None]] = []
    for task in tasks:
        print(f"  {task:12} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            recs = run_one(args.api, args.base_url, args.model, task, args.n,
                          args.timeout)
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            print(f"ERROR {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if not recs:
            print("no records")
            continue
        print(f"{time.time() - t0:5.0f}s")
        for r in recs:
            rows.append((task, r["metric"], r["value"], r.get("n")))

    print(f"\n{'task':14} {'metric':26} {'value':>9} {'n':>5}")
    print("-" * 58)
    for task, metric, value, n in rows:
        flag = ""
        if metric == "reproducible" and value == 0.0:
            flag = "  <-- GATE FAILED: endpoint not reproducible at temp 0"
        if metric == "accuracy" and task == "mmlu" and abs(value - 0.25) < 0.03:
            flag = "  <-- at chance for 4-option MC: not a measurement"
        print(f"{task:14} {metric:26} {value:9.4f} {(n if n is not None else 0):5}"
              f"{flag}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
