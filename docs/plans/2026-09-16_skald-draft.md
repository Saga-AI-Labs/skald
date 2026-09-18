# Skald - Consolidated Local Evaluation Framework (Draft v0.1)

Status: DRAFT v0.1 - 2026-09-16 - Quinn (A0 seat) - Operator GO: pending
Intended location: /media/data/coding/skald/docs/plans/2026-09-16_skald-draft.md
(staged here because that repo does not exist yet and /media/data/coding is not writable by a0-quinn)

> Reconstruction note. The original design sketch was written in an A0 chat session and was
> subsequently thinned by history compaction. This draft is rebuilt from (a) the surviving
> checklist entry, (b) the archived conversation summaries that still carry the Skald
> description verbatim, (c) the Jacobian-Lens reconnaissance artifacts of 2026-09-15, and
> (d) the operator's four corner points. It is not a verbatim recovery. See section 9.

## 1. What Skald is

A single, local, API-first evaluation service that consolidates the benchmark families which
today live apart - BDH-CL continual-learning evals, Pi-50 abliteration/refusal diagnostics,
and the general-purpose benchmark suites already used by the original Saga repo - behind one
result-store and two functionally equivalent surfaces: an API for LLM clients and a UI for
humans. It exists because ablated models cannot be evaluated through cloud APIs (they refuse,
and the analysis itself is the object of study), and because three separate eval pipelines
means three incompatible result formats and no cross-model comparison.

## 2. Why now

- Abliteration damage is quantified: refusal removal is clean, but it produces OFF_TARGET
  degeneration in 43 paired transitions and raises fabrication rates. A framework that
  measures where the damage sits is the natural next instrument.
- The DMT misclassification mystery: an abliterated model answered a DMT chemistry question
  context-dependently wrong; the hypothesis is abliteration-induced filter damage.
  Reproducing and localizing it needs layer-level readout.
- The Jacobian-Lens instrument exists and is Apache-2.0: integrate rather than reimplement.
- Fragmentation: BDH-CL, Pi-50 and Saga each carry their own scripts and result formats.
- Own-AI constraint: no cloud LLM in the analysis chain.

## 3. Design principles

1. Two equal surfaces: API and UI expose the same functional surface. Neither is a
   second-class view of the other; enforced by a conformance test, not by convention.
2. Local-only: no cloud LLM anywhere in the analysis or evaluation path.
3. One result-store: every adapter writes the same record shape; comparisons are queries.
4. Adapters, not a monolith: adding a benchmark family must not touch the core.
5. Identity by checkpoint hash: every result is keyed to the exact weights it measured.

## 4. Architecture

### 4.1 Suite adapters

| Adapter | Source | What it measures |
|---|---|---|
| bdh_cl | BDH-CL eval suite (eval_router.py, p5_inchain_check.py, Sonde B/C) | territory / routing / retention / P5 storage on grown models |
| pi50 | Pi-50 scripts (scripts/pi50/*) | abliteration/refusal diagnostics: REFUSED->OFF_TARGET sites, fabrication rate |
| saga | original Saga repo (/media/data/coding/saga) | general-purpose suites, incl. MMLU and HumanEval (operator-stated), plus the existing poisoning / answer-level evaluation (see dated note) |

> Note (2026-09-16, OPEN-4 resolution in task `skald-saga-recon`): the saga general-purpose
> machinery per actual source is `run_mmlu`/`run_gsm8k`/`run_bbq` in
> `src/evaluation/benchmarks.py` plus the eval orchestration `scripts/10_full_evaluation.py`.
> `humaneval` exists only as a config entry (`configs/evaluation.yaml`, `num_fewshot: 0`,
> "all 164 problems") — no runner and no handling branch in `scripts/10`, so HumanEval is
> **not runnable in the current saga machinery**; the M3 adapter must supply a runner or keep
> HumanEval claims truthful. The poisoning/answer-level evaluation lives in
> `scripts/08_run_poisoning_eval.py` (inline implementation) — there is **no
> `/api/benchmarks/poisoning[/per-sample]` HTTP route** in the repo (no FastAPI/Flask/uvicorn
> anywhere in `src/` or `scripts/`), and per-sample results are not persisted (only the
> aggregate `results/poisoning/report.json` plus TensorBoard events). No persisted saga eval
> outputs exist anywhere searched (in-repo `results/`, `output/`, `report/` absent; sibling
> content probe negative).
| jlens | github.com/hijohnnylin/neuronpedia (Apache-2.0) | layer-by-layer workspace readout (Jacobian Lens) over ablated models |

### 4.2 Unified result-store

One append-only store; one record per (model, suite, task, metric):

- model_checkpoint_sha256 - identity anchor (see 4.4)
- adapter, suite, task, metric, value, n, ci_low, ci_high
- protocol - eval protocol label (e.g. random-crop cold, teacher-forced) so numbers from
  different protocols are never silently compared
- created_at, host, script_sha256, seed
- artifacts[] - paths/hashes of raw outputs

Storage: SQLite as the query index over immutable per-run files (one JSON/Parquet per run),
so the store stays inspectable without a server.

### 4.3 Two surfaces

- API - the LLM-facing surface: list suites, query results, run a benchmark, same call shapes
  as any other tool.
- UI - the human-facing surface: result browsing, per-task drilldown, anomaly view.
- Both generated from one schema; a capability added to one appears in both by construction.

### 4.4 weight-atlas interlock

Every result carries the checkpoint hash. Skald inherits weight-atlas identity: a model's
weight-level atlas entry and its evaluation history are the same object seen from two sides
(the atlas asks what the weights are; Skald asks what the weights do). No separate model
registry - the hash is the join key.

### 4.5 Jacobian-Lens module

Wrap the Apache-2.0 Neuronpedia JLens implementation; do not fork the algorithm. Surving recon
(2026-09-15): hosted lens page per model (DeepSeek V4 Flash, Qwen 3.6, Gemma 3) with the
readout set Verbal Report, Directed Modulation, Multi-Hop Reasoning, General Broadcast,
Selective Mediation; Jlens API answers (probe: HTTP 200, docs route 308); the repo carries
~74 lens-relevant files, incl. apps/inference/neuronpedia_inference/endpoints/lens/
{lens_loader,model_specific,prompt,residual_spec}.py, schemas/lens.py, saes/saelens.py, and
the webapp app/[modelId]/jlens/* client.

**JLens adapter contract (OPEN-2, resolved 2026-09-17).** The generic
`run(model, task, config) -> records[]` contract is sufficient; no interface change is required.
Evidence from the upstream surface: the hosted path `POST /lens/prompt`
(`apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:2232`) accepts a self-contained
`LensPromptRequest` (`schemas/lens.py`: `model`, `type[]` in {LOGIT_LENS, JACOBIAN_LENS}, `prompt`,
client-supplied cached token ids via `LensSteerToken(token, type)`) and streams NDJSON frames
through `StreamingResponse`; there is no server-side session id and no cross-request turn state.
The lower-level library we wrap is likewise pure functions: `jlens.from_hf(hf, tok) -> HFLensModel`,
`jlens.fit(lm, prompts, source_layers=..., dim_batch=..., max_seq_len=...) -> JacobianLens`,
`JacobianLens.apply(lm, text, layers=[...]) -> per-layer logits`. Session-like use (the webapp's
per-model page and its five demo readout sets - Verbal Report, Directed Modulation, Multi-Hop
Reasoning, General Broadcast, Selective Mediation; `apps/webapp/app/[modelId]/jlens/
jlens-model-selector.tsx`) is a client-side sequence of independent `run()` calls, each carrying any
prior token ids in `config`. Mapping: `task` = the readout set (or generic `layer_readout`);
`config` = lens source (a fitted `*_jacobian_lens.pt`, else fit params `prompts`, `source_layers`,
`dim_batch`, `max_seq_len`, `skip_first`, dtype) + readout params (`layers`, `position`, `top_n`,
`seed`) + subprocess `python`; `records[]` flatten each per-layer vector readout to scalar records
(one per `(layer, position, rank)`, `metric` e.g. `top1_prob@L{l}`, `value` = prob/logit, `n`),
with the full vector kept in `artifacts[]` (see OPEN-5) and `protocol` naming the fit+readout
config. Execute under a torch-capable interpreter in a subprocess (the saga pattern), importing the
unmodified `jlens` package - wrap, do not fork.

**M4 end-to-end proof (2026-09-17, round-5 task `skald-jlens-store-roundtrip`).** The JLens
adapter ran end-to-end over the M4 target `huihui-ai/Huihui-gemma-3-270m-it-abliterated`
(local CPU fit: `source_layers=[4]`, `dim_batch=16`, `max_seq_len=48`, `skip_first=16`,
`n_prompts=2`, `dtype=float32`, seed 42; readout `layers=[4]` position `-1` top-3 softmax
prob) and persisted 3 unified `layer_readout` records into the default live store
(`run 2026-09-17T01:51:01Z`, checkpoint hash `888d54d8…`). The default store now holds 22
real records {bdh_cl 16, pi50 1, saga 2, jlens 3}; one live `store.query()` returns all four
families from the single store with per-family non-empty protocol labels distinct across
families. Stored top-k values recompute 1:1 (float32 tolerance) from the OPEN-5 raw trace
`.skald/jlens_raw/888d54d8…/20260917T015101Z_layer_readout_1784016c5bfee87b.jsonl`
(kind `jlens_raw_v1`); the fitted lens is at
`.skald/jlens_lenses/888d54d8…/layer_readout_1784016c5bfee87b_jacobian_lens.pt`; both are
referenced by sha256 from `artifacts[]`. As prescribed, this is a wiring check, not a
scientific measurement; a faithful all-layer lens needs the GPU box.

**M4 run target (resolved 2026-09-17).** This box is CPU-only (`lspci` shows only an Intel UHD 630;
no NVIDIA/AMD device, so `torch.cuda.is_available()` is False locally) and holds no abliterated HF
decoder checkpoint (the operator's abliterated Qwen twin is on another box; locally only the stock
`tiny-gpt2` under `.skald/saga_models/`, which is gpt2-layout and loadable, and BDH `.pt`
checkpoints, which are a custom architecture and not JLens-loadable). M4 therefore runs the smallest
abliterated HF decoder LM the wrapper can load: `huihui-ai/Huihui-gemma-3-270m-it-abliterated`
(~270M params, `model_type: gemma3_text`, safetensors; resolves via `_LAYOUTS[0] = Layout("model")`),
downloaded at run time; fallback `huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3` (Qwen2).
Interpreter: `/media/data/coding/OBLITERATUS/.venv/bin/python` (torch 2.13.0+cpu, transformers
5.14.0, datasets 5.0.0) - the only local env with all three. Upstream pin currently
`/tmp/opencode/neuronpedia` @ `4e3f3b2` (2026-09-15); `/tmp` is ephemeral, so M4 must vendor/pin the
upstream `jlens` package into the repo. CPU honesty: `fit_lens.py` hard-requires CUDA, but the
library `jlens.fit` is device-agnostic (probe `/tmp/opencode/jlens_probe.py` ran on CPU), so M4
calls the library, not the CLI; the default fit (every layer, ~100-200 prompts) is GPU-scale, so M4
fits a minimal lens (few `source_layers`, 1-2 prompts, `max_seq_len` ~32-64) to prove the wiring
end-to-end, records the reduced settings in `protocol`, and labels the readout a wiring check, not a
scientific measurement; a faithful lens requires the GPU box.

## 5. Components and interfaces

| Component | Responsibility | Interface |
|---|---|---|
| adapters/* | one per benchmark family | run(model, task, config) -> records[] |
| store | persist + query results | put(records), query(filters) |
| api | LLM-facing surface | HTTP/JSON, same schema as UI |
| ui | human-facing surface | same schema; conformance-tested against api |
| identity | checkpoint hashing / weight-atlas join | hash(checkpoint) -> sha256 |

## 6. Milestones

- M0 skeleton: repo, adapter interface, store schema, one adapter (bdh_cl) writing and reading
  back real results.
- M1 second adapter (pi50) + first cross-suite query.
- M2 two surfaces: API + UI over one schema, conformance test green.
- M3 saga suites: MMLU + HumanEval adapters over existing Saga evaluation data.
- M4 JLens module: layer readout wired to the store, one ablated model end-to-end.

## 7. Non-goals

- No cloud LLM in the analysis or evaluation chain.
- No re-implementation of the Jacobian Lens (integrate the Apache-2.0 work).
- No training/fine-tuning - Skald measures.
- No serving stack for the models under test.

## 8. Open questions

- OPEN-1: ledger format - SQLite index over immutable per-run files, or a single Parquet
  dataset at expected volume?
- OPEN-2: adapter contract - is run(model, task, config) too narrow for interactive JLens
  probes, which are session-like rather than batch?
  - RESOLVED 2026-09-17 (round-5 recon, task `skald-jlens-recon`, evidence from direct source
    inspection of the upstream clone `/tmp/opencode/neuronpedia` @ `4e3f3b2`). **No contract
    change needed.** Upstream is stateless/batch at both layers: the hosted endpoint
    `POST /lens/prompt` (`endpoints/lens/prompt.py:2232`) takes a self-contained
    `LensPromptRequest` (`schemas/lens.py`) - no session id, no cross-request state; the client
    supplies cached token ids per `LensSteerToken(token, type)`, and the response is an NDJSON
    `StreamingResponse`. The library we wrap is pure functions (`from_hf` / `fit` /
    `JacobianLens.apply`, no hidden state), and the interactive webapp page is a client-side
    sequence of independent probes. Sessions are therefore the caller's concern: thread prior
    token ids through `config`; each `run()` stays self-contained. Per-layer vector readouts
    flatten to scalar records with the raw vector in `artifacts[]`. Final contract and run
    target recorded in §4.5.
- OPEN-3: UI framework - weight-atlas / HAK pattern is the stated precedent; confirm which
  before M2.
- OPEN-4: which general-purpose suites beyond MMLU/HumanEval are already available in the
  original Saga repo, and in which result format.
  - RESOLVED 2026-09-16 (round-4 recon, task `skald-saga-recon`, evidence from direct source
    inspection of `/media/data/coding/saga`). Inventory: (1) `src/evaluation/benchmarks.py`
    runners `run_mmlu` (5-shot, max 2000 samples, 6 HF subjects, letter-choice accuracy),
    `run_gsm8k` (8-shot, full ~1319-sample test set, numeric exact match), `run_bbq` (0-shot,
    9 bias categories, accuracy disaggregated per category; overall = unweighted mean of
    category accuracies). Result type everywhere is `BenchmarkResult` (name, score,
    std_error=None, num_samples, category_scores, details). (2) `humaneval` is a **config
    entry only** (`num_fewshot: 0`, "all 164 problems") — there is no `run_humaneval` runner
    and no handling branch in `scripts/10_full_evaluation.py`, so as-written `10` raises
    `KeyError('humaneval')`; pass@1 was never implemented. (3) Poisoning/answer-level eval is
    `scripts/08_run_poisoning_eval.py` (inline implementation; `src/evaluation/poisoning.py`
    is stubs). Formats: `10` persists only per-benchmark `score` scalars to
    `results/full_eval/report.json` (`single_models`, `ensemble`,
    `best_single_per_benchmark`, `success`); `08` persists only the aggregate
    `results/poisoning/report.json` (`trigger`, `tau`, `recall`, `fpr`, `auc`,
    `clean_mean_score`, `triggered_mean_score`, `num_clean`, `num_triggered`, `passed`) plus
    TensorBoard events; per-sample detail is in-memory only. No persisted saga eval outputs
    exist in `/media/data/coding/saga` nor in sibling repos (search scope recorded in the
    task). The M3 adapter therefore executes the instruments and maps onto `RECORD_FIELDS`:
    `score→value`, `num_samples→n`, per-category BBQ→one record per category (never a single
    aggregate claim), `seed: 42`, `script_sha256` = hash of the invoked script, `ci_*`
    from the (currently always-None) `std_error` or a computed CI.
- OPEN-5: storage location and retention policy for raw JLens traces (larger than scalars).
  - RESOLVED 2026-09-17 (round-5 recon, task `skald-jlens-recon`, evidence from the existing
    artifact-store convention `.skald/saga_raw/`, `.skald/saga_models/`,
    `bdh-cl/out/skald_raw/`). **Location:** raw per-layer logit readouts are written
    append-only, one normalized JSONL file per run, under
    `.skald/jlens_raw/<model_slug>/<YYYYMMDDTHHMMSSZ>_<task>_<config_hash>.jsonl`
    (checkpoint-hash in `<model_slug>` where known). Fitted lenses (`*_jacobian_lens.pt`,
    expensive to refit) live under `.skald/jlens_lenses/<model_slug>/<name>.pt`. Store records
    never embed raw vectors: `artifacts[]` holds repo-relative paths + sha256 into
    `.skald/jlens_raw/`, matching the `RECORD_FIELDS.artifacts` contract. **Retention:** any
    trace or lens referenced by at least one store record is never pruned; unreferenced/orphan
    artifacts are eligible for pruning by age (default 90 days) or a total-size cap (default
    10 GiB) via an explicit `prune` operation. Deleting a store record does not delete its
    referenced raw trace unless `--cascade` is passed. Artifacts are immutable and
    content-addressed by sha256.

## 9. Provenance and reconstruction notes

- Verbatim, surviving: bdh/docs/tasks/2026-09-15_phase1-hf-release-checklist.md, section
  Related - "Skald (working name, operator-liked) - consolidated eval-haus: suite adapters
  (BDH-CL, pi-50 abliteration/refusal, saga benchmarks, Jacobian-Lens module), unified
  result-store, dual API+UI (weight-atlas / HAK pattern), no cloud-LLM in the analysis chain.
  Design-doc next, after upload housekeeping. Identity link to weight-atlas via checkpoint-hash."
- Verbatim, archived summary: the A0 chat history still carries an un-thinned summary line
  describing Skald as a "unified evaluation framework (formerly Eval-Haus) ... local,
  API-first service ... using the Jacobian Lens (adapted from the open-source Neuronpedia
  repo) to inspect internal J-Space representations", with the DMT misclassification named as
  a motivating case.
- Recon artifacts (2026-09-15, 06:10-06:32): Neuronpedia JLens page fetch; JLENS-API-PROBE
  (jlens_api_http=200); Neuronpedia repo metadata/tree (74 lens-related files, Apache-2.0).
- Operator corner points (2026-09-16): dual API+UI of equal functional completeness; general
  benchmark suites beyond abliteration (MMLU, HumanEval already used in the original Saga
  repo); interlock with weight-atlas; integration of the interactive Jacobian Lens
  (Apache licence).
- Not recovered: the full original sketch text (component list, schema, milestone detail).
  Sections 4-6 are the reconstruction; section 8 marks what remains genuinely undecided.

- Quinn, @quinn-the-builder, 2026-09-16
