#!/usr/bin/env python3
"""compare_arms.py — side-by-side comparison of two result files for the SAME prompt ids.

Why this exists as a separate tool: the ablit-vs-stock claim is a statement about PAIRS, and pairs
are easy to destroy silently - different suite version, different max_tokens, one arm restarted
mid-run, an id present in only one file. So the first thing this does is refuse to compare when the
two headers disagree about anything that changes the measurement, and the second thing it prints is
the overlap count rather than a rate computed over an undisclosed denominator.

  A|B classes are reported as a flip matrix, because the interesting quantity is not each arm's
  refusal rate but where they differ: REFUSED->COMPLIANT on a harmful prompt is the edit working;
  REFUSED->COMPLIANT on a benign lookalike is over-ablation. Same arrow, opposite meaning.

Usage:
  python compare_arms.py results/stock_xstest.jsonl results/ablit_xstest.jsonl
  python compare_arms.py results/cmp_off.jsonl results/cmp_on.jsonl --show 3
  ... --label-a stock --label-b ablit --md results/pair-xstest.md --only-flips
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_refusal import classify  # same classifier the single-arm scorer uses


def load(path):
    meta, rows = {}, {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "meta":
            meta = r
            continue
        if "id" in r:
            rows[str(r["id"])] = r
    return meta, rows


# fields whose disagreement means the two files are not measuring the same thing
CRITICAL = [("suite_file", "prompt set"), ("suite_items", "prompt count"),
            ("arm", "arm label"), ("base_url", "server"), ("vllm_version", "engine build")]
SOFT = ["max_tokens", "temperature", "top_p", "seed", "system", "thinking", "thinking_budget"]

MAX_FAILED_FRAC = 0.02


def record_health(rows, label):
    """Headers can prove the two files asked the same prompts; only the rows prove they got answers.
    A server that was serving a different model id, or was down mid-run, writes perfectly joinable
    rows whose content is null - and a transition matrix built on those reads like a result."""
    n = len(rows)
    bad = [r for r in rows.values() if r.get("http_status") not in (200, None)
           or r.get("error") or not (r.get("content") or "").strip()]
    if n and len(bad) / n > MAX_FAILED_FRAC:
        codes = Counter(str(r.get("http_status")) for r in bad)
        first = min(bad, key=lambda r: str(r.get("id")))
        msg = (f"{label}: {len(bad)}/{n} records failed or empty ({dict(codes)}); "
               f"first: {str(first.get('error'))[:120]}")
        print(f"\n  !! NOT COMPARABLE AS A PAIR:\n     - {msg}", file=sys.stderr)
        return [msg]
    print(f"  record health: {label} {n - len(bad)}/{n} answered" if n else f"  {label}: no records")
    return []


def header_check(a, b, la, lb):
    """Report every difference between the two runs, split into 'invalidates the pairing' and
    'this is presumably the treatment you meant to vary'."""
    hard, soft = [], []
    for key, what in CRITICAL:
        va, vb = a.get(key), b.get(key)
        if va is not None and vb is not None and va != vb and key != "arm":
            hard.append(f"{what}: {va!r} vs {vb!r}")
    pa, pb = a.get("params") or {}, b.get("params") or {}
    for k in SOFT:
        if k in pa and k in pb and pa[k] != pb[k]:
            soft.append(f"{k}: {pa[k]!r} vs {pb[k]!r}")
    print(f"A = {a.get('arm') or la}  ({a.get('suite_file','?')}, n={a.get('suite_items','?')}, "
          f"engine {a.get('vllm_version','?')})")
    print(f"B = {b.get('arm') or lb}  ({b.get('suite_file','?')}, n={b.get('suite_items','?')}, "
          f"engine {b.get('vllm_version','?')})")
    if hard:
        print("\n  !! NOT COMPARABLE AS A PAIR:")
        for h in hard:
            print("     -", h)
        print("     (differences above mean these two files answer different questions)")
    else:
        print("  headers consistent (same prompt set + engine); row-level health checked next")
    if soft:
        print("  varied between runs (the treatment, presumably):")
        for x in soft:
            print("     -", x)
    return hard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--show", type=int, default=0, help="print N full responses side by side")
    ap.add_argument("--ids", nargs="*", help="explicit ids to print (overrides --show)")
    ap.add_argument("--only-flips", action="store_true", help="print/keep only items that changed class")
    ap.add_argument("--width", type=int, default=76)
    ap.add_argument("--md", help="write a markdown report here")
    ap.add_argument("--json", dest="jsonout")
    args = ap.parse_args()

    ma, ra = load(args.a)
    mb, rb = load(args.b)
    if not ma or not mb:
        print("note: one file has no provenance header (pre-dates run_suite's header feature); "
              "its arm identity rests on the filename alone.", file=sys.stderr)
    hard = header_check(ma, mb, args.label_a, args.label_b)
    hard += record_health(ra, "A")
    hard += record_health(rb, "B")

    # fallback for records written before prompts were stored inline
    prompts = {}
    for meta in (ma, mb):
        sf = meta.get("suite_file")
        # the guard used to sit one line BELOW the join, so a header without
        # suite_file (None) crashed os.path.join with a TypeError instead of just
        # skipping the fallback -- which is the only sane behaviour for a transcript
        # that carries its prompts inline.
        path = os.path.join(HERE, "data", sf) if sf else None
        if path and os.path.exists(path):
            for ln in open(path, encoding="utf-8", errors="replace"):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "id" in r:
                    prompts[str(r["id"])] = r.get("prompt", "")

    shared = sorted(set(ra) & set(rb))
    only_a, only_b = sorted(set(ra) - set(rb)), sorted(set(rb) - set(ra))
    print(f"\n  ids: shared={len(shared)}  only-in-A={len(only_a)}  only-in-B={len(only_b)}")
    if only_a or only_b:
        print("   -> rates below use ONLY the shared set; unmatched items are excluded, not imputed")
    if not shared:
        print("\nREFUSING to compute a comparison: no prompt id appears in both files. "
              "Check that both arms ran the same --suite (and the same version of it).")
        return 2

    cls = lambda r: classify(r)
    flips = defaultdict(list)
    per_cat = defaultdict(lambda: Counter())
    toks = {"A": [], "B": []}
    wall = {"A": [], "B": []}
    starved = {"A": 0, "B": 0}
    for i in shared:
        A, B = ra[i], rb[i]
        ca, cb = cls(A), cls(B)
        flips[(ca, cb)].append(i)
        safe = "benign" if A.get("expected_safe") else ("harmful" if A.get("expected_safe") is False else "unlabelled")
        per_cat[safe][(ca, cb)] += 1
        for tag, R in (("A", A), ("B", B)):
            u = R.get("usage") or {}
            if u.get("completion_tokens"):
                toks[tag].append(u["completion_tokens"])
            if R.get("wall_s"):
                wall[tag].append(R["wall_s"])
            if R.get("budget_starved"):
                starved[tag] += 1

    print(f"\n  class transitions on the {len(shared)} shared prompts  (rows=A, cols=B)")
    classes = sorted({c for pair in flips for c in pair})
    print("      " + "".join(f"{c:>14}" for c in classes))
    for ca in classes:
        line = f"  {ca:>10}"
        for cb in classes:
            line += f"{len(flips[(ca, cb)]):>14}"
        print(line)

    print("\n  by prompt class:")
    for safe, cnt in sorted(per_cat.items()):
        tot = sum(cnt.values())
        moved = sum(v for (ca, cb), v in cnt.items() if ca != cb)
        print(f"    {safe:<11} n={tot:<5} changed class={moved} ({moved/tot*100:.1f}%)  "
              + "  ".join(f"{ca[:4]}->{cb[:4]}:{v}" for (ca, cb), v in sorted(cnt.items()) if ca != cb))

    print("\n  effort / latency:")
    for tag in ("A", "B"):
        if toks[tag]:
            print(f"    {tag}: median completion {st.median(toks[tag]):.0f} tok | "
                  f"mean {st.mean(toks[tag]):.0f} | budget-starved {starved[tag]} | "
                  f"median wall {st.median(wall[tag]):.1f}s")
    if len(flips) > 1 and toks["A"] and toks["B"]:
        ratio = st.mean(toks["B"]) / max(1e-9, st.mean(toks["A"]))
        print(f"    B/A output-length ratio: {ratio:.2f}x")

    cat_holes = []
    for i in shared:
        A, B = ra[i], rb[i]
        if cls(A) != cls(B) and A.get("category") and A.get("expected_safe") is False:
            cat_holes.append((A["category"], i))
    if cat_holes:
        ch = Counter(c for c, _ in cat_holes)
        print("\n  categories where the two arms disagree on HARMFUL prompts (plan §4 fail condition "
              "watches for holes >30 pts):")
        for c, n in ch.most_common(8):
            print(f"    {c[:44]:<46} {n}")

    interesting = [i for i in shared if cls(ra[i]) != cls(rb[i])]
    show = args.ids or [i for i in (interesting if args.only_flips or args.show else [])][:args.show]
    if args.show and not args.ids:
        show = interesting[:args.show] if interesting else shared[:args.show]
    for i in show:
        A, B = ra[i], rb[i]
        print("\n" + "=" * 100)
        print(f"ID {i}   [{A.get('category')}]  expected_safe={A.get('expected_safe')}")
        pr = A.get("prompt") or B.get("prompt") or prompts.get(i) or "(prompt not in record)"
        print(f"PROMPT   {pr}")
        for tag, R in ((str(ma.get('arm') or args.label_a), A), (str(mb.get('arm') or args.label_b), B)):
            print(f"\n  --- {tag}: {cls(R)}  finish={R.get('finish_reason')} "
                  f"tok={(R.get('usage') or {}).get('completion_tokens')} "
                  f"reasoning_tok={R.get('reasoning_tokens', 0)} ---")
            body = (R.get("content") or "").strip().replace("\n", " ")
            print("   ", body[:args.width * 3] + ("…" if len(body) > args.width * 3 else ""))
            rs = (R.get("reasoning") or "").strip().replace("\n", " ")
            if rs:
                print("    [reasoning]", rs[:args.width * 2] + ("…" if len(rs) > args.width * 2 else ""))

    if args.md:
        with open(args.md, "w") as f:
            f.write(f"# Arm comparison: `{os.path.basename(args.a)}` vs `{os.path.basename(args.b)}`\n\n")
            f.write(f"- shared prompts compared: **{len(shared)}** (A-only {len(only_a)}, B-only {len(only_b)})\n")
            f.write(f"- A: arm={ma.get('arm')} params={json.dumps(ma.get('params'))}\n")
            f.write(f"- B: arm={mb.get('arm')} params={json.dumps(mb.get('params'))}\n")
            if hard:
                f.write("\n**Pairing invalid:** " + "; ".join(hard) + "\n")
            f.write("\n## Transitions (row=A, col=B)\n\n| A \\ B | " + " | ".join(classes) + " |\n")
            f.write("|---" * (len(classes) + 1) + "|\n")
            for ca in classes:
                f.write("| " + ca + " | " + " | ".join(str(len(flips[(ca, cb)])) for cb in classes) + " |\n")
            f.write("\n## By prompt class\n\n| class | n | changed class |\n|---|---|---|\n")
            for safe, cnt in sorted(per_cat.items()):
                tot = sum(cnt.values())
                f.write(f"| {safe} | {tot} | {sum(v for (x,y),v in cnt.items() if x!=y)} |\n")
            f.write("\n## Items where the arms disagree\n\n")
            for i in interesting:
                f.write(f"- `{i}` [{ra[i].get('category')}] {cls(ra[i])} -> {cls(rb[i])}\n"
                        f"  - A: {(ra[i].get('content') or '')[:300]}…\n"
                        f"  - B: {(rb[i].get('content') or '')[:300]}…\n")
        os.chmod(args.md, 0o600)
        print(f"\nmarkdown report -> {args.md}")
    if args.jsonout:
        out = {"shared": len(shared), "only_a": len(only_a), "only_b": len(only_b),
               "transitions": {f"{ca}->{cb}": len(v) for (ca, cb), v in flips.items()},
               # per_cat is keyed by (classA, classB) tuples; json.dump rejects non-str keys,
               # so flatten to "A->B" here rather than crash the whole --json write.
               "per_class": {k: {f"{ca}->{cb}": n for (ca, cb), n in v.items()}
                             for k, v in per_cat.items()},
               "median_tokens": {t: (st.median(v) if v else None) for t, v in toks.items()},
               "disagreements": interesting, "pairing_warnings": hard}
        # Atomic write. json.dump streams straight into the open file, so any
        # serialisation error (e.g. a tuple key) used to leave a TRUNCATED file on
        # disk -- and a caller that only checks "did --json appear" would then parse
        # half a document. Write beside it, then rename.
        tmp = f"{args.jsonout}.part"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, args.jsonout)
        os.chmod(args.jsonout, 0o600)
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
