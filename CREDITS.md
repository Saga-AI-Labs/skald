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
