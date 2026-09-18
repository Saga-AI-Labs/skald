#!/usr/bin/env python3
"""run_suite.py — drive the local engine over an evaluation suite, one JSONL row per item.

Design notes (see ABLITERATION-EVAL-PLAN.md):
  * Every row records finish_reason, usage and wall time. Items whose visible content is
    empty while reasoning consumed the budget are BUDGET_STARVED, not refusals, and must be
    excluded from the denominator rather than scored. This happened to three probes on
    2026-09-14, so it is a first-class state here.
  * Thinking ON/OFF is a treatment, not a setting: pass --thinking both ways and keep the
    result files separate.
  * Resume-safe: rerunning skips ids already present without an error.
  * No scoring happens here. Refusal/harm classification lives in score_refusal.py so the
    judgement can be changed without re-running the model.

Suite format (JSONL): {"id": str, "prompt": str, "category": str?, "expected_safe": bool?, "lang": str?}
"""
from __future__ import annotations
import argparse, json, os, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("FN_BASE", "http://localhost:8888/v1")
ROOT = BASE[:-3] if BASE.endswith("/v1") else BASE
MODEL = os.environ.get("FN_MODEL", "qwen3.8-flash-next")
lock = threading.Lock()


def load_items(path: str):
    items = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                items.append(json.loads(ln))
    return items


def done_ids(out: str) -> set[str]:
    ids = set()
    if os.path.exists(out):
        with open(out) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "meta" or "id" not in r:
                    continue
                if not r.get("error"):
                    ids.add(r["id"])
    return ids


def _get_json(url):
    try:
        return json.loads(urllib.request.urlopen(url, timeout=15).read())
    except Exception:  # noqa: BLE001 - an unreachable endpoint should not block a run
        return {}


def server_meta():
    """Identify WHICH SERVER this file came from. Filenames alone cannot: the ablit and stock arms
    differ only by a checkpoint directory, and a mislabelled arm would look like a result."""
    ver = _get_json(ROOT + "/version")
    models = (_get_json(BASE + "/models") or {}).get("data") or []
    return {"base_url": BASE, "served_models": [m.get("id") for m in models],
            "vllm_version": ver.get("version") if isinstance(ver, dict) else None,
            "recipe_repo": os.path.basename(os.path.dirname(HERE))}


def assert_served_model(need: str) -> int:
    """The two arms differ only by a checkpoint directory, so nothing in a results row proves which
    model produced it except the server we asked. If the operator's --require-model does not match
    /v1/models, refuse to run: a silently wrong arm is worse than no arm."""
    if not need:
        return 0
    ids = [m.get("id") for m in (_get_json(BASE + "/models") or {}).get("data") or []]
    if need in ids:
        return 0
    print(f"[FATAL] server at {BASE} serves {ids or 'nothing reachable'}, not {need!r}. "
          f"Refusing to record rows under arm label that the server cannot corroborate.",
          file=sys.stderr)
    return 3


def suite_provenance(suite):
    """Copy the dataset's own provenance record into the results header, so a comparison can be
    audited without opening manifest.json and hoping the same file was used."""
    stem = os.path.splitext(os.path.basename(suite))[0]
    mp = os.path.join(HERE, "data", "manifest.json")
    if not os.path.exists(mp):
        return {"suite_file": os.path.basename(suite)}
    man = json.load(open(mp))
    ent = man.get(stem, {})
    return {"suite_file": os.path.basename(suite), "suite_items": ent.get("normalised"),
            "suite_source": ent.get("repo") or ent.get("origin"),
            "suite_fetched_utc": ent.get("fetched_utc"), "suite_note": (ent.get("note") or "")[:300]}


def ask(item, args) -> dict:
    msgs = ([{"role": "system", "content": args.system}] if args.system else []) + \
           [{"role": "user", "content": item["prompt"]}]
    body = {"model": MODEL, "messages": msgs, "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "chat_template_kwargs": {"enable_thinking": args.thinking == "on"}}
    if args.thinking_budget:
        body["chat_template_kwargs"]["thinking_budget"] = args.thinking_budget
    if args.seed is not None:
        body["seed"] = args.seed
    t0 = time.time()
    row = {"id": item["id"], "category": item.get("category"), "lang": item.get("lang"),
           "expected_safe": item.get("expected_safe"), "category": item.get("category"),
           # full prompt stored alongside the answer: a transcript that can only be read next to an
           # unversioned dataset file is not a record. prompt_sha stays as the cheap join key.
           "prompt": item["prompt"], "prompt_sha": hash(item["prompt"]) & 0xffffffff,
           "params": {k: getattr(args, k) for k in ("thinking", "max_tokens", "temperature",
                                                    "thinking_budget", "seed", "system")},
           "t_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": None, "http_status": None,
           "finish_reason": None, "usage": None, "content": None, "reasoning": None,
           "error": None}
    try:
        req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        raw = urllib.request.urlopen(req, timeout=args.timeout).read()
        r = json.loads(raw)
        ch = r["choices"][0]
        m = ch.get("message") or {}
        row.update(http_status=200, finish_reason=ch.get("finish_reason"), usage=r.get("usage"),
                   content=m.get("content") or "",
                   # full reasoning text, not a head: comparing WHY two arms diverge is one of the
                   # few things only the transcripts can answer, and it is unrecoverable later
                   reasoning=m.get("reasoning") or "")
    except urllib.error.HTTPError as e:
        row["http_status"], row["error"] = e.code, (e.read().decode("utf-8", "ignore") or "")[:300]
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
    row["wall_s"] = round(time.time() - t0, 2)
    # classify the budget-starvation state explicitly
    u = row.get("usage") or {}
    rt = ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
    row["reasoning_tokens"] = rt
    row["budget_starved"] = bool(row["content"] is not None and not row["content"].strip()
                                 and rt >= 0.8 * (u.get("completion_tokens") or 1))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--system", default="")
    ap.add_argument("--workers", type=int, default=8, help="in-flight requests (<=MAX_NUM_SEQS)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--require-model", default="", help="abort unless this model id is served; "
                    "stops a mislabelled arm from being recorded as a result at all")
    ap.add_argument("--arm", default="", help="label written to the results header "
                    "(e.g. ablit / stock); defaults to the output filename")
    args = ap.parse_args()
    if not args.arm:
        args.arm = re.sub(r"^results_|\.jsonl$", "", os.path.basename(args.out)).split("_", 1)[0] \
            or "unnamed"

    items = load_items(args.suite)[args.offset:]
    if args.limit:
        items = items[:args.limit]
    # Resolve the output path FIRST. The previous order deleted ./name.jsonl while the run wrote to
    # results/name.jsonl, so --overwrite silently did nothing - caught by using the flag once.
    if not os.path.dirname(args.out):
        rd = os.path.join(HERE, "results")
        os.makedirs(rd, mode=0o700, exist_ok=True)
        os.chmod(rd, 0o700)
        args.out = os.path.join(rd, args.out)
    rc = assert_served_model(args.require_model)
    if rc:
        return rc
    global MODEL
    if args.require_model:
        # Verify one id and ask for another is exactly how a mislabelled arm gets recorded. If the
        # operator pinned the expected model, that pin IS the request model unless FN_MODEL says
        # otherwise - and say so loudly rather than silently sending 404s for 7k items.
        if MODEL != args.require_model and not os.environ.get("FN_MODEL"):
            print(f"[model] request model switched {MODEL!r} -> {args.require_model!r} "
                  f"(from --require-model)", file=sys.stderr)
            MODEL = args.require_model
        elif MODEL != args.require_model:
            print(f"[FATAL] FN_MODEL={MODEL!r} contradicts --require-model={args.require_model!r}; "
                  f"refusing to mix two model ids in one file.", file=sys.stderr)
            return 3
    if args.overwrite and os.path.exists(args.out):
        os.remove(args.out)
    need_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    have = done_ids(args.out)
    todo = [i for i in items if str(i["id"]) not in have]
    print(f"[suite] {len(items)} items | {len(have)} already recorded | {len(todo)} to run "
          f"| thinking={args.thinking} workers={args.workers}", file=sys.stderr)
    if not todo:
        return 0
    fh = open(args.out, "a")
    if need_header:
        hdr = {"kind": "meta", "arm": args.arm, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
               time.gmtime()),
               "harness": {"script": os.path.basename(__file__),
                           "script_mtime": int(os.path.getmtime(__file__)),
                           "script_sha": hash(open(__file__, "rb").read()) & 0xffffffff},
               "params": {k: getattr(args, k) for k in ("thinking", "max_tokens", "temperature",
                                                        "top_p", "seed", "system", "workers",
                                                        "require_model")},
               "request_model": MODEL,
               **suite_provenance(args.suite)}
        hdr.update(server_meta())
        fh.write(json.dumps(hdr, ensure_ascii=False) + "\n")
        fh.flush()
    os.chmod(args.out, 0o600)
    n = [0]

    def work(it):
        row = ask(it, args)
        with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n[0] += 1
            if n[0] % 25 == 0 or n[0] == len(todo):
                print(f"  {n[0]}/{len(todo)} last={row['id']} "
                      f"{'ERR ' + str(row['http_status']) if row['error'] else ('starved' if row['budget_starved'] else 'ok')}",
                      file=sys.stderr)
        return row

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    dur = time.time() - t0
    rows = [r for r in (json.loads(l) for l in open(args.out)) if r.get("kind") != "meta"]
    errs = sum(1 for r in rows if r.get("error"))
    starved = sum(1 for r in rows if r.get("budget_starved"))
    toks = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
    print(f"[done] {len(rows)} rows in {dur/60:.1f} min | errors={errs} budget_starved={starved} "
          f"| generated={toks:,} ({toks/dur:.1f} tok/s aggregate)", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
