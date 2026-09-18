# Vendored upstream: Pi-50 abliteration / refusal diagnostic instrument

## Authorship and provenance

Two different things live in this directory, and they have different authors.
Conflating them would misattribute both.

**1. The harness (`harness/`) — written by the Pi-50 agent, not third-party code.**
`run_suite.py`, `score_refusal.py`, `score_confab.py`, `score_capability.py`,
`compare_arms.py`, `fetch_datasets.py`, `make_custom.py` were written by
**pi-50** (the agent seat on host `gx10-50ef`) for the
`Qwen3.8-Flash-Next-Single-DGX-Spark` abliteration evaluation, between
2026-09-14 and 2026-09-17. They are vendored here because Skald's adapters
drive instruments rather than re-implement them (CREDITS rule 1), not because
they are somebody else's code. They are stdlib-only (`urllib`, no `numpy`,
no `requests`, no `torch`), so any Python >= 3.11 runs them.

**2. The suite data (`data/`) — third-party published benchmarks.**
The 18 `.jsonl` suites are other people's published research datasets
(XSTest, JBB, AdvBench, SORRY-Bench, OR-Bench, StrongREJECT,
GSM8K, MMLU, SimpleQA) plus two batteries pi-50 built (`custom_ccb`,
`custom_fpb`). A 19th, HEx-PHI, was removed in the 2026-09-19 license
sweep (see below) — its gated terms forbid redistribution. Their
authoritative provenance record is
[`data/manifest.json`](data/manifest.json), copied verbatim from the source
project; it carries per suite the upstream `repo`/`origin`, the fetched
`config`/`split`/`rows`, the label-source (`forced_expected_safe`), the
normalised/dropped counts, and an arXiv id + title in `note`.

## Pinned state — pinned by content digest, deliberately not by commit

The source project is `/srv/coding/Qwen3.8-Flash-Next-Single-DGX-Spark`
(Asus GX10 / DGX Spark, host `gx10-50ef`). At the time of vendoring its
`HEAD` was:

```
d03809008834124e80223c3482f2ddb59577a48f
2026-09-09T18:16:20+01:00  Ship reduced-vocab drafting as the default (#39)
```

**That commit does not contain these files.** `git ls-files eval/` in the
source repo returns **0 files** — the whole `eval/` tree is untracked
working-tree content there. Recording `d038090` as "the pinned commit" would
therefore be a false statement about upstream, so this pin is by **SHA-256 of
each file**, which is verifiable from a fresh clone of *Skald* alone. The
commit above is recorded only as context for which build of the surrounding
project the harness was developed against.

Consequence for maintainers: if the harness is ever committed upstream, this
PIN should be re-pinned to that commit *and* the digests re-verified; the
digests remain the authority either way.

## What was vendored

### `harness/` (byte-identical copies)

| file | sha256 | bytes |
|---|---|---|
| `harness/compare_arms.py` | `f7c37d122114e39bd81769e3561c32b0f657e5dcd745e5180023d3676cf92cd1` | 13,573 |
| `harness/fetch_datasets.py` | `ae29c395846394fc363f8b8aaaf7af733c163a77f6fb3f2e0c907ebc09075490` | 26,995 |
| `harness/make_custom.py` | `07f7ec035a926655f739ca08be3ffdf81aa46a7ab6941e457a76e9a8e9503a8c` | 7,149 |
| `harness/run_suite.py` | `1db82d28d06f2aad7fd94eded3e4bc98442dbdcccdf3034641c86251b0a65ed8` | 12,358 |
| `harness/score_capability.py` | `21d61787f11263951fb4a16c2be066788ee93f2c83f02b9da7f035ee35386d13` | 8,700 |
| `harness/score_confab.py` | `6ca32a1bbf0c558adc2c79f2754aa7f433325a2635456661ee51ec1ba2ba7639` | 5,659 |
| `harness/score_refusal.py` | `e2a720f69d2e47e2ffc8186e4ae34cab711ea1e7315e14557e4fce3c115ea134` | 11,355 |

> **Re-pin note (2026-09-18).** `compare_arms.py` was fixed **upstream** in
> `/srv/coding/Qwen3.8-Flash-Next-Single-DGX-Spark/eval/` and re-vendored byte-identically,
> rather than patched in place here (rule 1). Three defects were closed:
>
> 1. `--json` crashed with `TypeError: keys must be str… not tuple` because `per_class` was
>    keyed by `(classA, classB)` tuples.
> 2. That crash left a **truncated** `--json` file on disk, because `json.dump` streams into the
>    open handle; the write is now `tmp` + `os.replace`.
> 3. The prompt fallback called `os.path.join(HERE, "data", sf)` one line **above** its own
>    `if sf` guard, so a provenance header without `suite_file` died with
>    `TypeError: join() argument must be str… not NoneType` instead of skipping the fallback.
>
> The digest above therefore supersedes `ae6043a2…fce35c3` (12,624 B) and the intermediate
> `4753a468…19519a22` (13,268 B). Each fix is pinned by a test in
> `tests/test_pi50_abliteration.py`.

`fetch_datasets.py` is the acquisition tool and the reason `manifest.json` is
trustworthy: it is what recorded repo/config/split/row-count/field-mapping per
suite, including the corrections it forced (e.g. StrongREJECT measured 313
rows, not the 310 the paper writes down). `make_custom.py` built the two
pi-50 batteries with resolver-verified identifiers.

### `data/` (18 suites + two manifests)

`data/MANIFEST.json` is generated, not hand-written: per suite the item count
counted from the file, its SHA-256, the count `manifest.json` declares, and
whether the two agree. All 18 agree; total **12,627 items**. (19 suites /
12,927 before the HEx-PHI removal documented below.)

| suite | items | sha256 | upstream | license (swept 2026-09-19) |
|---|---|---|---|---|
| `advbench` | 520 | `236dfea57f22b849…` | `walledai/AdvBench` | MIT (HF tag; content from `llm-attacks/llm-attacks`, MIT) |
| `advbench_safe` | 416 | `9d65cec70609f121…` | `AmberYifan/AdvBench_safe` | **none stated** — prompts derive from AdvBench (MIT); the paired reference refusals are the mirror author's, unstated |
| `custom_ccb` | 18 | `65dbd677c3f0fd67…` | `harness/make_custom.py` (pi-50) | own work, no third party involved |
| `custom_fpb` | 8 | `3d92924806b76877…` | `harness/make_custom.py` (pi-50) | own work, no third party involved |
| `gsm8k` | 1,319 | `d2cea8e5fcdf74d7…` | `openai/gsm8k` | MIT (HF tag) |
| `jbb` | 200 | `4f6037e4bddd4d2d…` | `JailbreakBench/JBB-Behaviors` | MIT (HF tag; `JailbreakBench/jailbreakbench` repo MIT) |
| `mmlu` | 623 | `e59dbcddccc48d63…` | `cais/mmlu` | MIT (HF tag) |
| `orbench_hard` | 1,319 | `c1f008be8143dc40…` | `bench-llm/or-bench` | CC-BY-4.0 (HF tag; attribution required — cite Cui et al., arXiv:2405.20947) |
| `orbench_toxic` | 655 | `78736ea11952deee…` | `bench-llm/or-bench` | CC-BY-4.0, as above |
| `simpleqa` | 4,326 | `ffc1feb0152abbb6…` | see known gaps below | MIT for the content (`openai/simple-evals`, MIT) — but the recorded origin is a deleted `/tmp/sqa.csv`, so the chain from that file to SimpleQA is unverified |
| `sorrybench` | 450 | `c8825d75ff4747d2…` | `sorry-bench/sorry-bench-202406` | MIT (project license: `sorry-bench/sorry-bench` repo MIT; HF tags say `other`) |
| `sorrybench2` | 440 | `f0121e9676b8bd47…` | `sorry-bench/sorry-bench-202503` | MIT, as above |
| `sorrybench2_caesar` | 440 | (see `data/MANIFEST.json`) | `sorry-bench/sorry-bench-202503` | MIT, as above |
| `sorrybench2_fr` | 440 | (see `data/MANIFEST.json`) | `sorry-bench/sorry-bench-202503` | MIT, as above |
| `sorrybench2_roleplay` | 440 | (see `data/MANIFEST.json`) | `sorry-bench/sorry-bench-202503` | MIT, as above |
| `strongreject` | 313 | (see `data/MANIFEST.json`) | `alexandrasouly/strongreject` | MIT (repo license) |
| `xstest` | 450 | (see `data/MANIFEST.json`) | `paul-rottger/xstest` (authors' CSV) | CC-BY-4.0 (repo license; attribution required — cite Röttger et al., arXiv:2308.01263) |
| `xstest_mirror` | 250 | (see `data/MANIFEST.json`) | `AlignmentResearch/XSTest` (`neg` config) | CC-BY-4.0 by content (mirror of XSTest; the mirror itself states no license) |

License sources: Hugging Face dataset API `license` tags and GitHub license
API `spdx_id`, swept 2026-09-19. CC-BY-4.0 suites require attribution on
reuse — the paper citations above satisfy it.

### Removed: `hexphi` (HEx-PHI, 300 items)

`data/hexphi.jsonl` (ex-`LLM-Tuning-Safety/HEx-PHI`, sha
`1045ecfc19931bce…`) was **deleted from this repository** in the 2026-09-19
license sweep and its `MANIFEST.json` entry with it. Reason: HEx-PHI ships
under a custom gated Dataset License Agreement whose **Prohibited Transfers**
clause forbids distributing, copying, embedding or hosting the dataset, with
manual access approval and a right-to-deletion/termination clause —
redistributing it from a public repo violates the terms under which it was
obtained. No credit line can cure that; only removal can. To run the hexphi
arm, fetch it yourself under your own approved access
(`harness/fetch_datasets.py` knows the source) — do not re-commit the file.

Do not "restore" this from the MIT-licensed code repo:
`LLM-Tuning-Safety/LLMs-Finetuning-Safety` on GitHub (MIT) holds the paper's
code and demo data, but **not** the `category_*.csv` benchmark — that lives
only in the gated `LLM-Tuning-Safety/HEx-PHI` HF dataset, which is a separate
object under separate terms. The vendored rows were fetched from the gated
dataset (see `manifest.json`: `category_1/3/4/5/6/7/8/9/10/11.csv`, terms
accepted 2026-09-14 for local eval use), so the MIT code-repo license never
covered them. Lawful access is not a redistribution right.

## Known provenance gaps (stated, not papered over)

These are real weaknesses in the record, found while vendoring. Do not treat
`manifest.json` as complete:

1. **`simpleqa`'s recorded origin is a deleted temp file.** Its `origin` is
   `/tmp/sqa.csv`, which no longer exists. The durable identifier is the arXiv
   id in its `note` (`arXiv:2411.04368`, "Measuring short-form factuality in
   large language models"). Re-acquisition must go through that, not through
   the recorded path.
2. **Nine suites record `repo: null` and `origin: null`.** For those the only
   provenance is the free-text `note` (which does carry an arXiv id and title
   in each case checked). `harness/fetch_datasets.py`'s `SOURCES` table is the
   machine-readable counterpart and should be read together with the manifest.
3. **Dataset licences, swept 2026-09-19.** Every suite in the table above
   now carries its upstream license (HF + GitHub license APIs). Standing
   caveats: `advbench_safe` states no license (AdvBench-derived prompts +
   unattributed refusals); `simpleqa`'s content is MIT but its recorded
   origin is a deleted temp file, so the chain is unverified; SORRY-Bench
   content is MIT via the project repo while HF tags say `other`;
   `xstest_mirror` inherits CC-BY-4.0 by content with no statement of its
   own. And see the HEx-PHI removal above — the one suite whose terms
   forbade redistribution outright.
4. **`advbench_safe` is a relabelled derivative of `advbench`** (upstream
   `AmberYifan/AdvBench_safe`), not an independent sample — see the
   `advbench_safe` correction note in `docs/2026-09-14_ablit-eval-suite.md`.
   Counting it alongside `advbench` double-counts the same behaviours.

## Deliberately excluded

- **`run_ablit_night.sh`, `run_stock_chain.sh`, `run_capability.sh`,
  `run_cap_both_arms.sh`** — the source project's arm-scheduling drivers. They
  shell out to `./switch-arm.sh`, `./start.sh`, `docker` and a watchdog on the
  serving container, i.e. they are bound to one host's vLLM deployment and
  cannot run from a fresh Skald clone. Vendoring them would vendor a broken
  default path (CREDITS rule 4). The arm swap they perform is an operator
  action, recorded in `docs/2026-09-14_ablit-eval-suite.md`.
- **`results/` — every transcript, score file and pair report.** These are
  model *outputs*, including the abliterated model complying with harmful
  requests. They are not published data and they do not belong in a public
  repository. The adapter writes them under `.skald/` (already gitignored) and
  the store keeps only the aggregate plus an artifact digest. See the
  privacy boundary in `docs/2026-09-14_ablit-eval-suite.md`.
- **`ABLITERATION-EVAL-PLAN.md` / `EDIT-FOOTPRINT.md`** — these are prose, not
  instruments; they live in `docs/` instead of being duplicated here.

## How the adapter uses this directory

`adapters/pi50.py` invokes these scripts **unmodified in a subprocess** with
`cwd=harness/` (the scripts import each other by bare module name, so the
harness directory must be the working directory / `PYTHONPATH` entry). It
re-implements no scoring and parses only their emitted JSON. Any upstream bug
belongs upstream, in the source project — per CREDITS rule 1, workarounds
live visibly in the adapter, never here.
