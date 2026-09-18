# Vendored upstream: Saga general-purpose benchmark subset

## Authorship and provenance

The code in this directory comes from the **Saga** project (Saga AI Labs) —
the "Mixture of Agents Coordination Layer" repo: `src/evaluation/benchmarks.py`
(`run_mmlu`, `run_gsm8k`, `run_bbq`, `BenchmarkResult`), `src/models/loader.py`
(`FrozenModelWrapper`), and `configs/evaluation.yaml` (suite configuration,
seed 42). Skald's authors did not write this benchmark machinery:
`adapters/saga.py` is a wrapper that invokes it unmodified in a subprocess,
and this directory is a byte-identical subset copy, not a fork.

- Source repo: `/media/data/coding/saga` (local checkout)
- Pinned commit: `5657fcedb854e9a5aadcfaffb37b97b298b4a2db`
- License: **GNU AGPL-3.0** (`LICENSE` here is the verbatim AGPL-3.0 text;
  the source repo previously carried no license file — one was placed at its
  root as part of this consolidation). See the root `CREDITS.md` for the
  combined-work consequence: Skald's own code stays Apache-2.0, but the
  combined work that executes this directory is AGPL-covered.

## What was vendored (and what was not)

Only the benchmark closure Skald's adapter invokes:

- `src/evaluation/benchmarks.py` (+ package `__init__` files, comment-only)
- `src/models/loader.py` (+ package `__init__`, comment-only)
- `configs/evaluation.yaml` — suite configuration mirrored by the adapter's
  defaults (max_samples, num_fewshot, seed)

Deliberately excluded: everything else in Saga (`alignment/`, `meta_model/`,
`orchestrator/`, `router/`, training scripts `01_`–`07_`, poisoning eval
`08_`, full-evaluation orchestration `10_`, `data/`, Docker). The adapter's
HumanEval shim exists because upstream Saga has no `run_humaneval` runner
(HumanEval is a config entry only) — the shim executes Saga's own
`humaneval` config entry and dataset, nothing else.

## Portability notes

- The driver runs with `PYTHONPATH` pointed here and imports
  `src.evaluation.benchmarks` / `src.models.loader` exactly as in the source
  repo; the package-relative layout (`src/...`) is preserved for that reason.
- Runtime needs torch + transformers + datasets + pyyaml in the executing
  interpreter (pass via adapter `config["python"]`); none of that ships here.
- Model weights and HF datasets are never vendored: callers supply model
  paths via adapter config; datasets stream from Hugging Face at run time.
