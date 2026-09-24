#!/usr/bin/env python3
"""One-shot vendoring of the BBH reasoning subset + PopQA factuality corpus.

Why this script exists instead of a datasets-server fetch: the served
datasets mirror only covers the two datasets openai_compat already used, and
the adapter must run offline. So the corpora are fetched ONCE here (needs
network), normalised to the local_bench jsonl shape, pinned by SHA-256, and
committed under vendor/bbh_popqa/. The adapter never touches the network
for these tasks.

Sources (verified 2026-09-24, recorded in vendor/bbh_popqa/PIN.md):
- BBH items: lighteval/bbh parquet configs read through the public
  HuggingFace datasets-server rows API (pure JSON, stdlib urllib).
  Upstream: BIG-bench Apache-2.0 family (arXiv:2210.09261); the
  maveriq/bigbenchhard mirror tags MIT. Both claims recorded, not resolved.
- PopQA: akariasai/PopQA test.tsv (14k rows, Wikidata pageviews included).
  Upstream repo AlexTMallen/adaptive-retrieval carries an MIT LICENSE
  covering the repo including data/popQA.tsv; the HF mirror states no
  license. Both claims recorded, not resolved.

Run:  python3 tools/vendor_bbh_popqa.py
Writes: vendor/bbh_popqa/data/bbh_<task>.jsonl, vendor/bbh_popqa/data/popqa.jsonl,
        vendor/bbh_popqa/data/manifest.json, vendor/bbh_popqa/PIN.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "vendor" / "bbh_popqa" / "data"

BBH_TASKS = [
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "date_understanding",
    "dyck_languages",
]
BBH_ROWS_URL = ("https://datasets-server.huggingface.co/rows"
                "?dataset=lighteval/bbh&config={task}&split=train"
                "&offset={offset}&length=100")
POPQA_URL = ("https://huggingface.co/datasets/akariasai/PopQA"
             "/resolve/main/test.tsv")


def _get(url: str, retries: int = 8) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                # Rate-limited: honour Retry-After, else back off hard. This
                # bit us on dyck_languages (the 8th task): 4 quick retries
                # all burned inside the same limit window.
                try:
                    wait = int(exc.headers.get("Retry-After", "0"))
                except ValueError:
                    wait = 0
                time.sleep(wait or min(60, 5 * (attempt + 1)))
            else:
                time.sleep(2 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 -- transient net, retry
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"vendor: GET failed after {retries}x: {url}: {last}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_bbh() -> dict:
    """Fetch the 8 reasoning tasks; return {task: [row, ...]}.

    The rows endpoint can serve partial pages; paginate to num_rows_total
    and refuse a short task rather than vendoring a silent subset.
    """
    tasks: dict[str, list[dict]] = {}
    for task in BBH_TASKS:
        rows: list[dict] = []
        total: int | None = None
        offset = 0
        while total is None or offset < total:
            payload = json.loads(_get(
                BBH_ROWS_URL.format(task=task, offset=offset)))
            if total is None:
                total = int(payload["num_rows_total"])
            page = [r["row"] for r in payload["rows"]]
            if not page:
                raise RuntimeError(
                    f"vendor: BBH {task}: empty page at offset {offset}")
            rows.extend(page)
            offset += len(page)
        if len(rows) != total:
            raise RuntimeError(
                f"vendor: BBH {task}: got {len(rows)} rows, want {total}")
        tasks[task] = rows
        print(f"  bbh {task}: {len(rows)} rows")
    return tasks


def normalise_bbh(tasks: dict) -> tuple[dict, int]:
    """Normalise to the local_bench shape, one answer kind per task.

    Two shapes exist and score differently, so each row records its own:
    - ``letter`` (tracking, deduction, dates): the gold option is a full
      phrase; the prompt letters the options and the gold is the letter.
    - ``text`` (dyck): the gold itself is a short bracket string (<= 8
      chars); choices are the permitted-symbol alphabet, and lettering an
      84-symbol alphabet would measure code-reading, not reasoning. Scored
      by normalised equality against the exact symbol string.
    A task whose golds straddle the boundary, or a row with no choices at
    all, is refused: either would silently change what the instrument
    measures.
    """
    out: dict[str, list[dict]] = {}
    for task, rows in tasks.items():
        for r in rows:
            if not (r.get("choices") or []):
                raise RuntimeError(
                    f"vendor: BBH {task}: row {r.get('id')} has no choices -- "
                    "refusing (instrument needs options or an alphabet)")
        golds = []
        for r in rows:
            choices = r.get("choices") or []
            idx = int(r["target_idx"])
            if idx >= len(choices):
                raise RuntimeError(
                    f"vendor: BBH {task}: row {r.get('id')}: target_idx "
                    f"{idx} outside {len(choices)} choices")
            golds.append(str(choices[idx]))
        kind = "text" if max(len(g) for g in golds) <= 8 else "letter"
        norm = []
        for r, g in zip(rows, golds):
            choices = r.get("choices") or []
            idx = int(r["target_idx"])
            if kind == "letter" and idx >= 26:
                raise RuntimeError(
                    f"vendor: BBH {task}: row {r.get('id')}: {len(choices)} "
                    "options do not fit letter-choice scoring")
            norm.append({
                "id": f"{task}:{r['id']}",
                "task": task,
                "answer_kind": kind,
                "input": r["input"],
                "task_prefix": r.get("task_prefix") or "",
                "choices": choices,
                "gold": g if kind == "text" else
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[idx],
                "gold_idx": idx,
            })
        print(f"  bbh {task}: answer_kind={kind}")
        out[task] = norm
    return out, sum(len(v) for v in out.values())


def _norm_simple(s: str) -> str:
    import re
    return re.sub(r"\s+", " ",
                  re.sub(r"[^a-z0-9 ]", "", str(s).lower())).strip()


def fetch_popqa() -> list[dict]:
    """Download PopQA, bucket subjects by pageview tertiles, filter junk alts.

    possible_answers carries truncation artefacts ("pol.", "pol") that are
    strict substrings of the real answers; as containment alts they would
    hand out unearned credit ("pol" is in "Poland"). An alt that is a
    strict substring of the primary gold or of a longer alt is dropped,
    and so is anything under 2 normalised characters.
    """
    raw = _get(POPQA_URL)
    print(f"  popqa tsv: {len(raw)} bytes")
    text = raw.decode("utf-8")
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    rows = list(reader)
    pops = sorted(int(r["s_pop"]) for r in rows if (r.get("s_pop") or "").strip().isdigit())
    if len(pops) != len(rows):
        raise RuntimeError(
            f"vendor: popqa: {len(rows) - len(pops)} rows lack s_pop")
    cut1, cut2 = statistics.quantiles(pops, n=3)
    print(f"  popqa rows: {len(rows)}, s_pop tertiles: {cut1:.0f} / {cut2:.0f}")
    norm = []
    for r in rows:
        try:
            alts = json.loads(r["possible_answers"])
        except ValueError:
            continue
        gold = (r.get("obj") or "").strip()
        if not gold or not (r.get("question") or "").strip():
            continue
        pool = [gold] + [a for a in alts if isinstance(a, str)]
        kept: list[str] = []
        norms = [_norm_simple(a) for a in pool]
        for a, na in zip(pool, norms):
            if not na or len(na) < 2:
                continue
            if any(na != nb and na in nb for nb in norms):
                continue  # truncation artefact, see docstring
            if a not in kept:
                kept.append(a)
        if not kept:
            continue
        spop = int(r["s_pop"])
        bucket = "tail" if spop < cut1 else ("head" if spop > cut2 else "mid")
        norm.append({
            "id": f"popqa:{r['id']}",
            "prompt": r["question"].strip(),
            "gold": kept[0],
            "gold_alts": kept[1:],
            "s_pop": spop,
            "bucket": bucket,
        })
    if not norm:
        raise RuntimeError("vendor: popqa: zero usable rows after filtering")
    return norm, (cut1, cut2)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/vendor_bbh_popqa.py (one-shot, network at vendor time only)",
    }

    print("fetching BBH ...")
    bbh, bbh_total = normalise_bbh(fetch_bbh())
    manifest["bbh"] = {
        "source": "lighteval/bbh via public datasets-server rows API",
        "upstream": "BIG-bench Apache-2.0 family, arXiv:2210.09261; "
                    "maveriq/bigbenchhard mirror tags MIT",
        "license_note": "both upstream claims recorded, not resolved; "
                        "no license tag on the lighteval mirror",
        "tasks": {},
    }
    for task, rows in bbh.items():
        path = OUT / f"bbh_{task}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                        encoding="utf-8")
        manifest["bbh"]["tasks"][task] = {
            "rows": len(rows), "sha256": _sha256(path)}
    print(f"  wrote {bbh_total} BBH rows across {len(bbh)} tasks")

    print("fetching PopQA ...")
    popqa, (cut1, cut2) = fetch_popqa()
    popqa_path = OUT / "popqa.jsonl"
    popqa_path.write_text("".join(json.dumps(r) + "\n" for r in popqa),
                          encoding="utf-8")
    counts: dict[str, int] = {}
    for r in popqa:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
    manifest["popqa"] = {
        "source": "akariasai/PopQA test.tsv",
        "upstream": "upstream repo AlexTMallen/adaptive-retrieval carries an "
                    "MIT LICENSE covering the repo including data/popQA.tsv; "
                    "the HF mirror states no license",
        "license_note": "both claims recorded, not resolved",
        "rows": len(popqa),
        "sha256": _sha256(popqa_path),
        "s_pop_tertile_cutoffs": [cut1, cut2],
        "bucket_counts": counts,
        "alt_filter": "drop empty, <2 chars, or strict substring of another alt",
    }

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    pin = f"""# Vendored upstream: BBH reasoning subset + PopQA factuality

One-shot vendor product of `tools/vendor_bbh_popqa.py` (network at vendor
time only; the adapter reads the committed jsonl offline).

## What and why

- **BBH** (`bbh_<task>.jsonl`, {bbh_total} rows, 8 tasks): multi-step reasoning
  with exact-gradeable outputs (tracking shuffled objects x3, logical
  deduction x3, date understanding, dyck languages). Chosen for the
  logic/reasoning axis: the *output* is small (a choice letter, or one
  bracket for dyck -- chance floor, read it as such) but reaching it
  requires the chain, unlike MMLU recall. Each row records `answer_kind`
  (`letter` vs `text`) so the scorer cannot mix the two.
- **PopQA** (`popqa.jsonl`, {len(popqa)} rows): entity-centric short factual
  questions WITH subject pageviews, bucketed to head/mid/tail by s_pop
  tertiles ({cut1:.0f} / {cut2:.0f}). The head-vs-tail gap is the
  quant-sensitive number: aggressive rounding prunes rare knowledge first.

## Provenance and licenses

- BBH upstream: BIG-bench Apache-2.0 family (arXiv:2210.09261), read via the
  lighteval/bbh mirror through the public datasets-server rows API.
  maveriq/bigbenchhard tags MIT. Both claims recorded, not resolved; no
  license tag sits on the lighteval mirror itself.
- PopQA upstream repo AlexTMallen/adaptive-retrieval carries an MIT LICENSE
  covering the repo including data/popQA.tsv; the akariasai/PopQA mirror
  states no license. Both claims recorded, not resolved.
- Full per-file rows + SHA-256: `data/manifest.json` (generated
  {manifest["generated_at"]}).

## Shaping decisions (instrument, not benchmark)

- BBH prompts are built at run time: task_prefix + input + lettered choices
  + "reply with only the letter" instruction. The committed rows carry the
  raw parts, so a prompt change is a code change with a protocol echo, not
  silent data drift.
- PopQA `gold_alts` exclude truncation artefacts (strict substrings of
  another alt) so containment scoring cannot earn credit by accident.
"""
    (OUT.parent / "PIN.md").write_text(pin, encoding="utf-8")
    print("wrote manifest.json + PIN.md")
    print(json.dumps({"bbh_rows": bbh_total, "popqa_rows": len(popqa),
                      "buckets": counts}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
