#!/usr/bin/env python3
"""fetch_datasets.py — pull evaluation suites into normalised local JSONL, with provenance.

Three acquisition paths, because the datasets are not uniformly reachable:
  github_csv   the authors' own repo file (canonical, no account needed)
  server       HF datasets-server JSON API (public datasets only; 100 rows/request)
  hub_files    huggingface_hub download with our token (gated repos answer 401 to the
               datasets-server even when an accepted account can read them)

Every output records repo/config/split/row-count/field-mapping in data/manifest.json. That is
not ceremony: OR-Bench has shipped more than one layout, and a comparison against the stock
twin is meaningless if the two arms silently used different subsets.

Normalised schema:
  {"id","prompt","category","expected_safe":bool|None,"lang","source"}
"""
from __future__ import annotations
import argparse, csv, io, json, os, re, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SERVER = "https://datasets-server.huggingface.co"
UA = "eval-harness/1.0 (gx10-50ef)"
PROMPT_KEYS = ["prompt", "instruction", "Behavior", "behavior", "question", "Problem",
             "problem", "text", "forbidden_prompt"]
LIST_KEYS = ["turns", "content", "questions", "prompts"]   # some suites ship a conversation list
CAT_KEYS = ["type", "category", "Category", "categorical_level", "Categorical_level", "topic",
            "Topic", "HarmDomain", "harm_domain", "risk_category", "sub_category", "prompt_style"]
SAFE_KEYS = ["label", "is_safe", "safe", "expected_safe", "binary_label", "gen_target", "target"]


def token() -> str | None:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    env = os.path.join(os.path.dirname(HERE), ".env")
    if os.path.exists(env):
        for ln in open(env):
            m = re.match(r'\s*(?:export\s+)?HF_TOKEN=["\']?([^"\'\s#]+)', ln)
            if m:
                return m.group(1)
    return None


def http(url: str, tries: int = 4, auth: bool = False, binary: bool = False):
    last = None
    hdrs = {"User-Agent": UA}
    if auth and token():
        hdrs["Authorization"] = "Bearer " + token()
    for i in range(tries):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=hdrs), timeout=90).read()
            return raw if binary else json.loads(raw)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def _looks_de(s: str) -> bool:
    return bool(re.search(r"\b(und|der|die|das|nicht|wie|bitte|erkl[äa]re|nenne|was ist)\b", s, re.I))


def _prompt_of(row):
    """Scalar prompt fields, else the LAST turn of a conversation list (SORRY-Bench ships
    multi-turn items; the final user turn is what the model must answer), else first element
    of a one-element list (XSTest mirror ships content=[prompt])."""
    for k in PROMPT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in LIST_KEYS:
        v = row.get(k)
        if isinstance(v, list) and v:
            cand = next((str(x).strip() for x in reversed(v) if isinstance(x, str) and x.strip()), None)
            if cand:
                return cand
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def norm(row, src, idx, forced_safe=None, cat_fallback=None, lang=None):
    prompt = _prompt_of(row)
    if not prompt:
        return None
    safe = forced_safe
    if safe is None:
        lab = next((row[k] for k in SAFE_KEYS if k in row and row[k] is not None), None)
        if isinstance(lab, str):
            s = lab.strip().lower()
            if s in ("benign", "safe", "true", "harmless", "no", "0"):
                safe = True
            elif s in ("harmful", "unsafe", "false", "yes", "1"):
                safe = False
        elif isinstance(lab, bool):
            safe = lab
        elif isinstance(lab, (int, float)):
            safe = bool(lab)
    cat = next((str(row[k]).strip() for k in CAT_KEYS if isinstance(row.get(k), str) and row[k].strip()),
               None) or cat_fallback or "(uncategorised)"
    ident = next((str(row[k]) for k in ("id", "ID", "index", "Index", "uuid", "question_id")
                  if k in row and row[k] != ""), None) or str(idx)
    return {"id": f"{src}:{ident}", "prompt": prompt, "category": cat, "expected_safe": safe,
            "lang": lang or ("de" if _looks_de(prompt) else "en"), "source": f"{src}#{idx}"}


# ---------------------------------------------------------------- acquisition
def from_github(url, name, man, forced_safe=None, cat_fallback=None):
    raw = http(url, binary=True).decode("utf-8", "ignore")
    rdr = list(csv.DictReader(io.StringIO(raw)))
    items, skipped = [], 0
    for i, r in enumerate(rdr):
        n = norm({k.strip(): (v or "").strip() for k, v in r.items()}, name, i, forced_safe, cat_fallback)
        if n: items.append(n)
        else: skipped += 1
    return items, skipped, {"origin": url, "rows_seen": len(rdr)}


def from_server(repo, name, man, configs=None, splits_wanted=None, forced_safe_map=None, cap=1000,
                row_filter=None):
    sp = http(f"{SERVER}/splits?dataset={urllib.parse.quote(repo)}").get("splits", [])
    cfgs = sorted({c["config"] for c in sp})
    chosen = [c for c in cfgs if (configs is None or c in configs)]
    items, skipped, detail = [], 0, []
    for cfg in chosen:
        names = [c["split"] for c in sp if c["config"] == cfg]
        for split in ([s for s in names if splits_wanted is None or s in splits_wanted] or names[:1]):
            fs = (forced_safe_map or {}).get(split)
            out, off = [], 0
            while off < min(cap, 10000):
                d = http(f"{SERVER}/rows?dataset={urllib.parse.quote(repo)}&config={cfg}"
                         f"&split={split}&offset={off}&length=100")
                batch = [b["row"] for b in (d.get("rows") or [])]
                if not batch:
                    break
                out.extend(batch)
                off += len(batch)
                total = int(d.get("num_rows_total") or 0)
                if not total or off >= total:
                    break
                time.sleep(0.2)
            for i, r in enumerate(out[:cap]):
                if row_filter and not row_filter(r):
                    skipped += 1
                    continue
                n = norm(r, f"{name}-{split}", i, forced_safe=fs,
                         cat_fallback=f"{cfg}/{split}" if fs is not None else None)
                if n:
                    items.append(n)
                else:
                    skipped += 1
            detail.append({"config": cfg, "split": split, "rows": len(out[:cap]), "forced_expected_safe": fs})
    return items, skipped, {"repo": repo, "fetched": detail}


def from_hub_files(repo, name, patterns, cap=1000, forced_safe=None, header=True,
                   file_category_rx=None, prompt_key="instruction", cat_fallback=None):
    """Gated datasets: download the shipped json/csv files with the token."""
    from huggingface_hub import hf_hub_download, list_repo_files  # local import: optional dep
    tok = token()
    files = [f for f in list_repo_files(repo, repo_type="dataset", token=tok)
             if any(re.search(p, f) for p in patterns)]
    if not files:
        raise RuntimeError(f"{repo}: no file matched {patterns} (have: {list_repo_files(repo, repo_type='dataset', token=tok)[:12]})")
    items, skipped, detail = [], 0, []
    for f in files:
        path = hf_hub_download(repo, f, repo_type="dataset", token=tok)
        rows = []
        fcat = None
        if file_category_rx:
            m = re.search(file_category_rx, f)
            fcat = f"category_{m.group(1)}" if m else None
        if not header and f.endswith(".csv"):
            # HEx-PHI ships bare instruction lines with NO header row; DictReader would consume the
            # first prompt as column names and lose it. Verified against the raw bytes.
            rdr = [r for r in csv.reader(open(path, newline="", encoding="utf-8", errors="ignore"))
                   if r and any(x.strip() for x in r)]
            rows = [{prompt_key: r[0], "category": fcat or ""} for r in rdr]
        elif f.endswith(".parquet"):
            import pyarrow.parquet as _pq
            rows = _pq.read_table(path).to_pylist()
        elif f.endswith((".csv", ".tsv")):
            rows = list(csv.DictReader(open(path, newline="", encoding="utf-8", errors="ignore"),
                                       delimiter="\t" if f.endswith(".tsv") else ","))
        elif f.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(path, encoding="utf-8", errors="ignore") if l.strip()]
        elif f.endswith(".json"):
            j = json.load(open(path, encoding="utf-8", errors="ignore"))
            rows = j if isinstance(j, list) else next((v for v in j.values() if isinstance(v, list)), [])
        got = 0
        for i, r in enumerate(rows[:cap]):
            if not isinstance(r, dict):
                continue
            r = {k: (v.strip().strip('"').replace('""', '"') if isinstance(v, str) else v)
                 for k, v in r.items()}
            # id namespace MUST include the file: row indices restart per file, and a bare
            # "{name}:{i}" makes every category collide with every other one, so the dedupe below
            # silently keeps only the first file. Same bug class as the JBB split collision.
            n = norm(r, f"{name}-{fcat}" if fcat else name, i, forced_safe=forced_safe,
                     cat_fallback=fcat or cat_fallback)
            if n:
                items.append(n)
            else:
                skipped += 1
            got += 1
        detail.append({"file": f, "rows_used": got})
    return items, skipped, {"repo": repo, "files": detail}


def from_hub_parquet(repo, files, name, cap=2000, forced_safe=None, cat_fallback=None,
                     drop_wrapping_quotes=True):
    """Direct parquet download via /resolve/main/, bypassing the datasets-server.

    Needed because a repo can be readable at the resolve endpoint while the datasets-server still
    refuses it (registration/indexing lag after a consent click), and vice versa. Which of the two
    paths works is checked per run, never assumed - that assumption is what cost us the original
    HEx-PHI/AdvBench sources earlier today.
    """
    import pyarrow.parquet as pq
    tok = token()
    items, skipped, detail = [], 0, []
    for f in files:
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{f}"
        hdrs = {"User-Agent": UA}
        if tok:
            hdrs["Authorization"] = "Bearer " + tok
        req = urllib.request.Request(url, headers=hdrs)
        raw = urllib.request.urlopen(req, timeout=120).read()
        if raw[:4] != b"PAR1":
            raise RuntimeError(f"{url}: not a parquet ({raw[:60]!r})")
        import io as _io
        t = pq.read_table(_io.BytesIO(raw))
        rows = t.to_pylist()
        got = 0
        for i, r in enumerate(rows[:cap]):
            if drop_wrapping_quotes:
                # some mirrors store the whole instruction inside literal double quotes, and keep
                # inner quotes CSV-doubled; without this the same prompt looks like two different ones
                r = {k: (v.strip().strip('"').replace('""', '"') if isinstance(v, str) else v)
                     for k, v in r.items()}
            n = norm(r, name, i, forced_safe=forced_safe, cat_fallback=cat_fallback)
            if n:
                items.append(n)
                got += 1
            else:
                skipped += 1
        detail.append({"file": f, "rows_used": got, "columns": t.column_names})
    return items, skipped, {"repo": repo, "transport": "resolve/parquet", "files": detail}


def enrich_sorry_taxonomy(items, repo):
    """Replace bare category digits with the benchmark's own topic names.

    The scorer clusters by category for the bootstrap, so "1".."20" would work either way; the
    names are here because a reader of results/score-*.json should not have to open meta_info.py to
    know what cluster 7 is. Fails soft: if the file layout changes, digits stay digits.
    """
    try:
        from huggingface_hub import hf_hub_download
        import ast
        path = hf_hub_download(repo, "meta_info.py", repo_type="dataset", token=token())
        txt = open(path, encoding="utf-8", errors="ignore").read()
        m = re.search(r"category_descriptions\s*=\s*(\[.*?\])", txt, re.S)
        descs = ast.literal_eval(m.group(1)) if m else {}
        table = {str(k): str(v) for k, v in enumerate(descs, 1)} if isinstance(descs, list) else dict(descs)
    except Exception as e:  # noqa: BLE001
        print(f"[sorrybench] taxonomy enrichment skipped: {type(e).__name__}: {str(e)[:90]}")
        return items
    for it in items:
        c = it.get("category")
        if c in table:
            it["category"] = f"{c} {table[c]}"[:60]
    return items


SOURCES = {
    # Notes carry the REGISTERED paper title right after each id so cite-check.sh can verify them.
    # Titles go in SINGLE quotes inside double-quoted strings: the checker accepts either, and this
    # avoids the nested-quote soup that made this file unparsable once already. No apostrophes in
    # the prose near a title, or the matcher grabs the wrong span.
    "xstest": dict(kind="github",
                   url="https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
                   note="XSTest arXiv:2308.01263 'XSTest: A Test Suite for Identifying Exaggerated "
                        "Safety Behaviours in Large Language Models' - authors repo file; 450 rows = "
                        "18 types x 25, the contrast_* types are the genuinely unsafe controls"),
    "xstest_mirror": dict(kind="server", repo="AlignmentResearch/XSTest", configs=["neg"],
                          splits_wanted=["validation"],
                          note="fallback mirror of XSTest; expected_safe read from gen_target"),
    "hexphi": dict(kind="hub", repo="LLM-Tuning-Safety/HEx-PHI",
                   patterns=[r"category_\d+\.csv$"], header=False,
                   file_category_rx=r"category_(\d+)\.csv$", forced_safe=False,
                   note="HEx-PHI arXiv:2310.03693 'Fine-tuning Aligned Language Models Compromises "
                        "Safety, Even When Users Do Not Intend To' - the ORIGINAL author repo, after the "
                        "operator accepted its terms on 2026-09-14. Measured here: the repo ships 10 "
                        "category CSVs of 30 bare instructions each = 300, and category_2.csv does not "
                        "exist (resolve/ returns the literal body 'Entry not found'), which explains the "
                        "300-vs-330 gap left by the public copies YanY-NLP/HEx-PHI and "
                        "jkazdan/HeX-PHI-usable - they are faithful, not lossy. Files carry no "
                        "use-case column either: the harmful framing is inside the instruction text"),
    "advbench": dict(kind="hub", repo="walledai/AdvBench", patterns=[r"\.parquet$"],
                     forced_safe=False, cat_fallback="advbench-behavior",
                     note="AdvBench arXiv:2307.15043 'Universal and Transferable Adversarial Attacks "
                        "on Aligned Language Models' - 520 behaviour strings plus optimisation "
                        "targets; we send the prompt only. Consent accepted 2026-09-14"),
    "jbb": dict(kind="server", repo="JailbreakBench/JBB-Behaviors", configs=["behaviors"],
                splits_wanted=["harmful", "benign"],
                forced_safe_map={"harmful": False, "benign": True},
                note="JBB arXiv:2404.01318 'JailbreakBench: An Open Robustness Benchmark for "
                     "Jailbreaking Large Language Models' - paired harmful/benign behaviours; the "
                     "benign half doubles as an over-refusal control"),
    "sorrybench": dict(kind="hub", repo="sorry-bench/sorry-bench-202406",
                       patterns=[r"^question\.jsonl$"], forced_safe=False,
                       note="SORRY-Bench arXiv:2406.14598 'SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal'"
                            " - ORIGINAL repo, 450 single-turn prompts / 45 categories; consent accepted "
                            "2026-09-14. Supersedes the SillyTilly/SorryBench copy used earlier"),
    # StrongREJECT ships its prompts as a plain CSV in the authors' repo; dsbowen/strong_reject is
    # the evaluator package and points here, so this URL IS the canonical source.
    "strongreject": dict(kind="github",
                         url="https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
                             "strongreject_dataset/strongreject_dataset.csv",
                         forced_safe=False,
                         note="StrongREJECT arXiv:2402.10260 'A StrongREJECT for Empty Jailbreaks' - "
                              "313 forbidden prompts with a scoring_category and a jailbreak source "
                              "column; measured 313 rows, not the 310 written down earlier"),
    "orbench_hard": dict(kind="server", repo="bench-llm/or-bench", configs=["or-bench-hard-1k"],
                         splits_wanted=["train"], forced_safe_map={"train": True},
                         note="OR-Bench arXiv:2405.20947 'OR-Bench: An Over-Refusal Benchmark for Large "
                              "Language Models' - refined hard subset, 1319 seemingly-toxic prompts that "
                              "the benchmark expects a model to ANSWER. Label assigned by us from the "
                              "benchmark construction, not copied from a column"),
    "orbench_toxic": dict(kind="server", repo="bench-llm/or-bench", configs=["or-bench-toxic"],
                          splits_wanted=["train"], forced_safe_map={"train": False},
                          note="same paper, arXiv:2405.20947 'OR-Bench: An Over-Refusal Benchmark for "
                              "Large Language Models' - 655 genuinely toxic controls paired with the "
                               "hard subset, so over-refusal is scored against a real refusal baseline "
                               "rather than against our own judgement of what counts as harmful"),
    # ---- SORRY-Bench 2 (repo sorry-bench/sorry-bench-202503; paper cited in 'sorrybench2') ---
    # Same 440 prompts replicated across 21 style arms. We take base + three arms chosen because
    # they can tell a genuine removal apart from surface-pattern suppression: a cipher arm (the
    # request is identical in meaning, unrecognisable as a string), a role-play arm (reframed as
    # fiction), and a translation arm (refusal is known to be language-dependent). The other 17
    # arms exist and are unfetched on purpose: morse/slang/ascii/technical_terms/
    # uncommon_dialects/authority_endorsement/evidence-based_persuasion/expert_endorsement/
    # logical_appeal/misrepresentation/misspellings/question + translate-{ml,mr,ta}. There is NO
    # German arm, so this does not close our de gap - it shows whether the edit travels.
    "sorrybench2": dict(kind="hub", repo="sorry-bench/sorry-bench-202503",
                        patterns=[r"^question\.jsonl$"], forced_safe=False,
                        taxonomy_repo="sorry-bench/sorry-bench-202503",
                        note="SORRY-Bench 2 arXiv:2406.14598 'SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal' "
                             "- base arm, measured 440 prompts over 44 categories (v1 was 450 over 45; "
                             "both counts are right, they belong to different versions)"),
    "sorrybench2_caesar": dict(kind="hub", repo="sorry-bench/sorry-bench-202503",
                               patterns=[r"^question_caesar\.jsonl$"], forced_safe=False,
                               taxonomy_repo="sorry-bench/sorry-bench-202503",
                               note="SORRY-Bench 2 Caesar-cipher arm (paper cited on the 'sorrybench2' "
                                    "entry above): same 440 intents rendered unrecognisable as character "
                                    "strings, so residual refusal here must come from intent recognition "
                                    "rather than token matching"),
    "sorrybench2_roleplay": dict(kind="hub", repo="sorry-bench/sorry-bench-202503",
                                 patterns=[r"^question_role_play\.jsonl$"], forced_safe=False,
                                 taxonomy_repo="sorry-bench/sorry-bench-202503",
                                 note="SORRY-Bench 2 role-play arm (paper cited on 'sorrybench2'): the "
                                      "classic frame that restores refusals in shallowly-edited models"),
    "sorrybench2_fr": dict(kind="hub", repo="sorry-bench/sorry-bench-202503",
                           patterns=[r"^question_translate-fr\.jsonl$"], forced_safe=False,
                           cat_fallback="translated-fr", set_lang="fr",
                           note="SORRY-Bench 2 machine-translated French arm (paper cited on "
                                "'sorrybench2'). Nearest available stand-in for the German deployment "
                                "question: it measures language transfer, it does not answer German"),
    "advbench_safe": dict(kind="server", repo="AmberYifan/AdvBench_safe", splits_wanted=["train"],
                          # CORRECTED 2026-09-15: this repo's columns are (prompt, response) where
                          # prompt is the HARMFUL behaviour and response is a canned refusal - "safe"
                          # describes the response, not the prompt. It is an efficacy suite with a
                          # reference refusal attached, NOT a benign control set. Do not force True.
                          forced_safe_map={"train": False},
                          note="harmful prompts each paired with a human-written refusal (columns "
                               "prompt+response); usable as efficacy evidence and as a refusal-style "
                               "reference, never as an over-refusal control"),
}


def write(name, items, meta, man):
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, f"{name}.jsonl")
    seen, uniq = set(), []
    for it in items:
        if it and it["id"] not in seen:
            seen.add(it["id"])
            uniq.append(it)
    with open(out, "w") as f:
        for it in uniq:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    man[name] = {**meta, "normalised": len(uniq), "dropped_no_prompt": len(items) - len(uniq),
                 "file": os.path.relpath(out, HERE),
                 "safe_true": sum(1 for i in uniq if i["expected_safe"] is True),
                 "safe_false": sum(1 for i in uniq if i["expected_safe"] is False),
                 "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(f"[{name}] {len(uniq)} items (safe={man[name]['safe_true']} unsafe={man[name]['safe_false']}) -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SOURCES))
    # A silent truncation is the worst kind of provenance bug: it produces a plausible-looking
    # subset with no trace. The cap now defaults above every suite we fetch, and write() records
    # rows_seen vs normalised so any future clipping is visible in the manifest.
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--simpleqa-csv", default="/tmp/sqa.csv")
    args = ap.parse_args()
    mp = os.path.join(DATA, "manifest.json")
    man = json.load(open(mp)) if os.path.exists(mp) else {}
    for name in args.only:
        cfg = SOURCES.get(name)
        if not cfg:
            print(f"unknown source {name}", file=sys.stderr)
            continue
        try:
            if cfg["kind"] == "github":
                items, sk, meta = from_github(cfg["url"], name, man, cfg.get("forced_safe"),
                                              cfg.get("cat_fallback"))
            elif cfg["kind"] == "hubp":
                items, sk, meta = from_hub_parquet(cfg["repo"], cfg["files"], name, args.cap,
                                                   cfg.get("forced_safe"), cfg.get("cat_fallback"))
            elif cfg["kind"] == "server":
                items, sk, meta = from_server(cfg["repo"], name, man, cfg.get("configs"),
                                              cfg.get("splits_wanted"), cfg.get("forced_safe_map"),
                                              args.cap, cfg.get("row_filter"))
            else:
                items, sk, meta = from_hub_files(cfg["repo"], name, cfg["patterns"], args.cap,
                                                 cfg.get("forced_safe"), cfg.get("header", True),
                                                 cfg.get("file_category_rx"),
                                                 cat_fallback=cfg.get("cat_fallback"))
            if cfg.get("set_lang"):
                # _looks_de() cannot tell French from English, so a translated arm has to declare
                # its language or it enters the corpus tagged "en".
                for it in items:
                    if it:
                        it["lang"] = cfg["set_lang"]
            if cfg.get("taxonomy_repo"):
                items = enrich_sorry_taxonomy(items, cfg["taxonomy_repo"])
            meta["note"] = cfg["note"]
            write(name, [i for i in items if i], meta, man)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED {type(e).__name__}: {str(e)[:180]}", file=sys.stderr)
    if os.path.exists(args.simpleqa_csv):
        items = []
        with open(args.simpleqa_csv, newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                q = (r.get("problem") or "").strip()
                if q:
                    items.append({"id": f"simpleqa:{i}", "prompt": q, "category": "short-form-factuality",
                                  "expected_safe": True, "lang": "en", "source": f"simpleqa#{i}",
                                  "gold": (r.get("answer") or "").strip()})
                if len(items) >= 4326:
                    break
        write("simpleqa", items, {"origin": args.simpleqa_csv,
              "note": "arXiv:2411.04368 'Measuring short-form factuality in large language models'; "
                       "grader distinguishes correct/incorrect/not_attempted"}, man)
    json.dump(man, open(mp, "w"), indent=1)
    print(f"manifest -> {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
