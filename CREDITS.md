# Credits — whose code lives in this repository

Skald's own code (adapters' wrappers, `store/`, `api/`, `ui/`, `surfaces/`,
`identity/`, tests, docs) is original to this project. The benchmark
instruments it drives are not: each family below is someone else's work,
vendored unmodified with its license. The per-directory `PIN.md` files carry
the full provenance (source, pinned commit, file list, digests where
applicable); this file is the index.

| directory | what | author / source | license |
|---|---|---|---|
| `vendor/jlens/` | Jacobian-Lens library (`jlens` package) | **Anthropic PBC** (copyright headers), via Johnny Lin's **Neuronpedia** (`github.com/hijohnnylin/neuronpedia`); background paper "Verbalizable Representations Form a Global Workspace in Language Models" (Gurnee et al., 2026) | Apache-2.0 (`vendor/jlens/LICENSE`) |
| `vendor/bdh_cl/` | BDH-CL eval scripts (`eval_router`, `domain_eval`, `p5_inchain_check`), Pi-50 manifest instrument, `pipeline/` + `bdh.py` import closure | **Pathway Technology, Inc.** (© 2025, research fork of `pathwaycom/bdh` — *The Dragon Hatchling*, Kosowski et al.) | MIT-style (`vendor/bdh_cl/LICENSE.md`) |
| `vendor/saga_benchmarks/` | Saga general-purpose benchmarks (`run_mmlu`, `run_gsm8k`, `run_bbq`, `BenchmarkResult`), model loader, eval config | **Saga AI Labs** (Saga "Mixture of Agents" repo; previously unlicensed — AGPL-3.0 `LICENSE` placed at its root during this consolidation) | **AGPL-3.0** (`vendor/saga_benchmarks/LICENSE`) |
| `vendor/pi50_eval/` | Abliteration/refusal instrument: 7 stdlib scorers + 18 public prompt suites (12,627 items). **Mixed authorship** — see the note below the table | **harness:** pi-50 (this project's own instrument, same as `adapters/pi50.py`). **data:** MIT — `JailbreakBench`, `AdvBench`, `SORRY-Bench`/`-2`, `MMLU` (`cais/mmlu`), `GSM8K`, `StrongREJECT`, SimpleQA content; CC-BY-4.0 — `XSTest` (+mirror), `OR-Bench` (attribution: Röttger et al.; Cui et al.); own work — `custom_ccb`, `custom_fpb`; unstated — `AdvBench_safe` mirror (AdvBench-derived). Per-suite table in `vendor/pi50_eval/PIN.md` | **swept 2026-09-19** (HF + GitHub license APIs); one suite **removed**, see note |

### The `vendor/pi50_eval/` entry, stated plainly

This directory is unlike the other three, and two of rule 2's requirements are
only partly met for it. Rather than paper over that:

- **The harness is not borrowed.** `harness/*.py` is pi-50's own instrument — the
  same authorship as `adapters/pi50.py` — so it carries no third-party license and
  no upstream to pin. It is vendored (rather than left only in the adapter) so the
  scorers stay byte-checkable and runnable from a fresh clone, per rule 4.
- **The data is borrowed, and its licenses were swept 2026-09-19**
  (Hugging Face + GitHub license APIs; per-suite table in
  `vendor/pi50_eval/PIN.md`). MIT covers `JailbreakBench`, `AdvBench`,
  `SORRY-Bench`/`-2` (via the project repo; HF tags say `other`),
  `MMLU`, `GSM8K`, `StrongREJECT`, and the SimpleQA content (whose
  recorded origin is still a deleted temp file — chain unverified).
  CC-BY-4.0 covers `XSTest` (+mirror, by content) and `OR-Bench`, both
  needing attribution on reuse. `AdvBench_safe` states no license.
  The two `custom_*` batteries are pi-50-authored.
- **One suite was removed in that sweep: HEx-PHI.** Its custom gated
  Dataset License Agreement forbids redistribution outright (Prohibited
  Transfers, access approval, right-to-deletion), so `hexphi.jsonl` was
  deleted from the tree and its manifest entry with it — no credit line
  could cure that. Re-fetch under your own approved access; do not
  re-commit the file.
- **Pinned by content, not by commit.** The source `eval/` directory is not
  git-tracked, so no upstream commit exists to name; each file is pinned by SHA-256.
- **One file is a derivative, not an independent sample:** `advbench_safe.jsonl` is
  `advbench.jsonl` with its `expected_safe` column relabelled. Do not average the two.

## Combined-work license notice

Skald's own code is Apache-2.0 (root `LICENSE`). The MIT and Apache-2.0
vendored parts combine without restriction. The AGPL-3.0 part does not:
**the combined work that executes `vendor/saga_benchmarks/` (i.e. Skald's
saga adapter driving Saga's benchmark code) is AGPL-3.0-covered** — anyone
distributing it, or offering its results over a network, owes AGPL
compliance (source availability) for the whole. Per-file licenses are
unchanged; this notice describes the combination. If the saga adapter is
never executed, the AGPL part is inert data on disk.

## Rules for borrowed code (maintainers)

1. Vendor byte-identical subsets; never fork or patch in place. If upstream
   must be worked around, the workaround lives in Skald's adapter, visibly.
2. Every `vendor/<name>/` carries `PIN.md` (authorship, source, pinned
   commit, file list, why this subset) and the upstream `LICENSE` verbatim.
3. Every borrowing is listed in this file and in the README credits section.
4. Adapters default to the vendored copy. An adapter may accept an external
   checkout via config (for development against upstream HEAD), but the
   default path must work from a fresh clone plus documented interpreters.
