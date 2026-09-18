#!/usr/bin/env python3
"""score_capability.py - grade axis C (capability tax) on the stored transcripts.

Gold answers are joined from eval/data/<suite>.jsonl by id rather than trusted from the result row:
the dataset is the authority, and a runner that silently lost a field should fail loudly here.

Verdicts per item: CORRECT | WRONG | NO_ANSWER (answered but no extractable answer) |
REFUSED | OFF_TARGET (the degenerate boilerplate class from score_refusal).
Accuracy denominators exclude nothing by default - a refusal IS wrong on a math test - but --strict-math
reports the "attempted-only" rate alongside so the two readings cannot be confused.
"""
import argparse, json, math, os, random, re, statistics as st, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_refusal import classify          # noqa: E402

NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def golds(suite_file):
    g = {}
    if not os.path.exists(suite_file):
        return g
    for ln in open(suite_file, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "meta" or "id" not in r:
            continue
        g[str(r["id"])] = r
    return g


def num(s):
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def pred_math(text):
    """Prefer an explicit #### line (that is what we asked for); else the last number written."""
    m = re.findall(r"####\s*([-+]?\s*\$?[\d,]*\.?\d+)", text)
    if m:
        return num(m[-1])
    lines = [l for l in text.strip().splitlines() if l.strip()]
    for l in reversed(lines):
        nums = NUM.findall(l)
        if nums:
            return num(nums[-1])
    return None


LET = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def pred_mc(text, item):
    t = text.strip()
    m = re.match(r"\s*([A-D])[\.\):\s]", t)
    if m:
        return m.group(1)
    letters = LET.findall(t[:120])
    if len(set(letters)) == 1:
        return letters[0]
    # fall back to matching the option text itself
    gl = (item.get("gold_text") or "").strip().lower()
    if gl and gl[:40] in t.lower():
        return item.get("gold")
    return letters[0] if letters else None


def boot_by_cluster(per_cat, reps=2000, seed=7):
    cats = sorted(per_cat)
    if len(cats) < 2:
        return (None, None)
    rng = random.Random(seed)
    rates = []
    for _ in range(reps):
        picks = [cats[rng.randrange(len(cats))] for _ in cats]
        k = sum(per_cat[c][0] for c in picks)
        n = sum(per_cat[c][1] for c in picks)
        if n:
            rates.append(100.0 * k / n)
    rates.sort()
    if not rates:
        return (None, None)
    return (rates[int(0.025 * len(rates))], rates[min(len(rates) - 1, int(0.975 * len(rates)))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--json", default="")
    ap.add_argument("--show", type=int, default=0, help="print N wrong/no-answer examples")
    args = ap.parse_args()

    out = {"suites": {}}
    for path in args.results:
        rows, meta = [], {}
        for ln in open(path, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "meta":
                meta = r
                continue
            if "id" in r:
                rows.append(r)
        # the header records which dataset produced this file; trust it over the filename convention
        stem = (meta.get("suite_file") or os.path.basename(path)).replace(".jsonl", "")
        stem = re.sub(r"^(stock|ablit|probe)_", "", stem)
        gd = golds(os.path.join(args.data_dir, f"{stem}.jsonl"))
        if not gd:
            print(f"!! {path}: no dataset file for suite '{stem}' in {args.data_dir} - cannot grade",
                  file=sys.stderr)
            continue
        kind = next(iter(gd.values())).get("kind", "gsm8k")
        per_cat = collections.defaultdict(lambda: [0, 0])
        tally = collections.Counter()
        miss_ids, toks, walls, bad = [], [], [], []
        for r in rows:
            g = gd.get(str(r["id"]))
            if not g:
                miss_ids.append(r["id"])
                continue
            beh = classify(r)
            txt = (r.get("content") or "").strip()
            if beh in ("REFUSED", "OFF_TARGET"):
                tally[beh] += 1
                per_cat[g.get("category", "?")][1] += 1
                bad.append((r["id"], beh, txt[:80]))
                continue
            if kind == "mmlu_mc":
                p, gv = pred_mc(txt, g), g.get("gold")
                unparsed = p is None
            else:
                p, gv = pred_math(txt), num(g.get("gold"))
                unparsed = (p is None or gv is None)
            if unparsed:
                # an answer we cannot read is NOT also an answer we read incorrectly: counting it in
                # both buckets made the tallies exceed n, which is how this line got written.
                tally["NO_ANSWER"] += 1
                per_cat[g.get("category", "?")][1] += 1
                bad.append((r["id"], "NO_ANSWER", txt[:80]))
                continue
            ok = (abs(p - gv) <= 1e-6) if kind != "mmlu_mc" else (p == gv)
            tally["CORRECT" if ok else "WRONG"] += 1
            per_cat[g.get("category", "?")][1] += 1
            per_cat[g.get("category", "?")][0] += int(ok)
            bad.append((r["id"], "WRONG" if not ok else "CORRECT", txt[:80]))
            u = r.get("usage") or {}
            if u.get("completion_tokens"):
                toks.append(u["completion_tokens"])
            if r.get("wall_s"):
                walls.append(r["wall_s"])
        n = sum(v[1] for v in per_cat.values())
        k = sum(v[0] for v in per_cat.values())
        lo, hi = wilson(k, n)
        blo, bhi = boot_by_cluster(dict(per_cat))
        attempted = k + tally["WRONG"]
        print(f"\n{os.path.basename(path)}  [{meta.get('arm','?')}]  suite={stem}  n={n}")
        print(f"  accuracy      {k}/{n} = {100.0*k/max(1,n):5.1f}%   Wilson[{lo:.1f},{hi:.1f}]"
              + (f"  cluster-boot[{blo:.1f},{bhi:.1f}]/cat" if blo is not None else ""))
        if attempted:
            print(f"  attempted-only {k}/{attempted} = {100.0*k/attempted:5.1f}%  "
                  f"(refusals+off-target excluded: {tally['REFUSED']}+{tally['OFF_TARGET']})")
        print("  tallies       " + ", ".join(f"{x}={tally[x]}" for x in
              ("CORRECT", "WRONG", "NO_ANSWER", "REFUSED", "OFF_TARGET") if tally[x]))
        if toks:
            print(f"  output tokens median {int(st.median(toks))}, wall median {st.median(walls):.1f}s"
                  if walls else f"  output tokens median {int(st.median(toks))}")
        if miss_ids:
            print(f"  !! {len(miss_ids)} result ids absent from the dataset file (not imputed)")
        out["suites"][stem] = {"arm": meta.get("arm"), "n": n, "correct": k,
                               "accuracy_pct": round(100.0 * k / max(1, n), 2),
                               "wilson": [round(lo, 2), round(hi, 2)],
                               "cluster_boot": [None if blo is None else round(blo, 2),
                                                None if bhi is None else round(bhi, 2)],
                               "tally": dict(tally),
                               "per_category": {c: {"correct": v[0], "n": v[1]}
                                                for c, v in sorted(per_cat.items())},
                               "median_out_tokens": int(st.median(toks)) if toks else None,
                               "median_wall_s": round(st.median(walls), 2) if walls else None}
        if args.show:
            for i, (rid, v, t) in enumerate(x for x in bad if x[1] != "CORRECT"):
                if i >= args.show:
                    break
                print(f"    [{v}] {rid}: {t!r}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
