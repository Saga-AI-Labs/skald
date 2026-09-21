# Qwen3.8-Flash-Next stock-vs-abliterated: interim measurement report

Date: 2026-09-21. Ledger: 59 Skald records (6 families) + weight-atlas
scans/deltas + the abliteration decision docs (`docs/plans/2026-09-14_…`,
`docs/reports/2026-09-18_edit-footprint.md`). Purpose: a reference for
which Skald benchmarks answer which questions about this model pair, what
each has actually shown so far, and which instruments are still missing.
Status of every number below is stated inline; nothing here is a final
verdict.

## 1. The pair

- **Stock:** `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` (snapshot `925d7be6`),
  Skald identity `7f55edf2…` (weighed) / served as `qwen3.8-flash-next[-stock]`.
- **Abliterated:** `drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only`
  (snapshot `b3b60a4c`), Skald identity `cbe78dd2…`. Server startup log:
  refusals removed; MTP, PLE, experts, chat template stock.
- Atlas scans: 6× identical stock NVFP4 scans + 2 fresh twins (jobs
  `b467…` ablit / `7662…` stock, 151,176 tensors each).

## 2. Weight forensics (atlas diff + compare report)

| instrument | result |
|---|---|
| `atlas diff` (frobenius, 151k tensors) | max −0.33% @ L23 o_proj; hotspots L15–47, **all `self_attn.o_proj.weight`** (+ lockstep quant-scale refits); embeddings, lm_head, MLPs, hyper-connections exactly 0.00% |
| Compare job `2359dbcf` (cartographic tier) | height rel-L2 0.166 / cosine 0.987, hotspot L47 attn_o; tint 0.077/0.997, hotspot L31; rough cosine **1.000000**; every hotspot row in every channel is `attn_o` |
| Null control (same weights, 1 min apart) | all-0.0 over 149k tensors — scanner deterministic, control honest |
| Cross-check (Qwen3.8-27B stock-vs-Huihui-ablit) | uniform 0.01% dust, no hotspot — either not a true pair or below resolution |

**Learning:** the edit is confined to one slot class (mid-to-late attention
output projections). Capability machinery is structurally untouched.
**Limit:** statistic tier (norm percentages); the finer weight-space tier
needs an edit-preset compare job pairing the exact scans.

## 3. Served-model capability (openai_compat mmlu / humaneval)

| run (UTC) | arm (attribution) | mmlu (n=100) | humaneval (n=20) |
|---|---|---|---|
| Sep-20 21:34/21:37 | stock (convergent, §5) | 0.40 | 0.00 |
| Sep-20 22:10/22:13 | **ablit** (convergent, §5) | 0.20 | 0.00 |
| Sep-21 18:20/18:35 | stock (root-verified) | 0.35 | 0.00 |
| Sep-21 18:32 | stock (root-verified) | 0.39 | — |
| Sep-21 19:29/19:31 | **ablit** (root-verified) | 0.20 | 0.15 |

**Learnings:**
- Stock MMLU clusters 0.35–0.40 (3 runs); ablit reads 0.20 twice, including
  an exact replication across evenings. Capability tax ≈ **0.15–0.20
  absolute** on 5-shot letter-choice — replicated, not a one-point fluke.
  (Absolute levels are low for this class; reasoning-budget pressure at
  64–128 max_tokens plus strict letter extraction both push down. Do not
  compare with the docs' 89.7/82.2 — different budget, subjects, protocol.)
- Humaneval: 0.0/0.0/0.0/0.15. The zeros predate the fence-aware extractor
  (raw completions, prose/fences → SyntaxError → fail); the 0.15 (3/20,
  post-fix) is the only format-fair point. Directionally the ablit writes
  runnable code *more* often — n=20, curiosity only, needs a bigger sample.
- Temp-0 does not mean deterministic here: the stock 0.35–0.40 spread on
  identical weights is reasoning-trace nondeterminism. Served-model numbers
  need repeats; single runs are anecdotes.

## 4. Refusal / honesty batteries (pi50_eval + decision docs)

Skald store status: **no refusal-battery records filed yet** — the numbers
below live in the decision docs and (for transcripts) on the measuring box,
not in this ledger.

| axis (decision docs) | stock | ablit |
|---|---|---|
| refusal, unsafe (n=2,538) | 93.5% | 0.0% |
| refusal, safe (n=2,085) | 77.0% | 0.0% |
| MMLU capability, 1024-token budget (n=623) | 89.7% | 82.2% |
| fabrication, CCB battery | 61.1% | 88.9% |

**Learning:** efficacy is total (both axes → 0) with a modest capability tax
and a large honesty tax (fabrication +28 pts). The honest-reading caveat:
safe-prompt refusal at 77% stock means "0% everywhere" also removes a lot
of (over-)refusal — the ablit looks cleaner partly because stock was jumpy.
**To file:** run `pi50 run_suite` per arm on .50 and import the runs; the
adapter, harness, and import script all exist — only the runs are missing.

## 5. What the other families contribute

- **bdh_cl router** (16 records): BDH continual-learning routing/retention —
  unrelated to this pair (different architecture); no learning about Qwen.
- **saga mmlu/humaneval** (3 records, tiny-gpt2): smoke-scale wiring proofs
  (0.0s); no learning about Qwen. Note: the saga humaneval shim carried the
  same `bool(check())` + `lstrip()` bugs as openai_compat — fixed, but the
  stored 0.0s predate the fix.
- **jlens layer_readout** (3 records, Gemma-270M): CPU wiring check; no
  learning about Qwen. A Qwen-scale lens needs the GPU box.
- **pi50 manifest_check** (2 records, one a byte-duplicate): repo-freshness
  signal; no learning about Qwen.

## 6. Provenance incidents (resolved, recorded so future readers don't repeat)

- **Endpoint identity collision:** both arms served under one self-reported
  name, so Sep-20 runs shared identity `00f39f2e…` with no arm marker.
  Resolved by timestamp + operator recall + exact replication (models.yaml
  documents the attribution per record group). Procedure going forward:
  explicit arm labels per run (distinct identities `…-stock` / `…-ablit`).
- **Harness bugs found by measurement:** `bool(check())` never-passing,
  `lstrip()` SyntaxErrors, null-content reasoning budgets, first-match
  letter extraction, datasets-server 100-row cap + throttling. All fixed in
  the adapters; stored zeros predating each fix are marked.
- **HEx-PHI removed** from `vendor/pi50_eval` (gated license forbids
  redistribution); history rewrite still pending.

## 7. Missing instruments (build list, ordered)

1. **Refusal-battery records in the store** — no code needed; run + import.
2. **Bigger humaneval samples** (n≥100/arm) with the fence-aware extractor.
3. **Edit-preset compare job** for the Qwen twins → tier-1 `rel_l2`
   hotspots (the adapter already speaks the shape).
4. **Activity/lesion maps** (atlas pre/post forward-pass capture) — the
   functional complement to the weight map; answers *what broke*, not just
   *where*.
5. **Qwen-scale JLens** (GPU box) — layer-level readout of the actual pair.
6. **HEx-PHI history rewrite** (coordination with all clones first).
7. **Run-from-UI + readable UI** — already shipped (spec v2 ops,
   models.yaml names, dark theme); needs operator road-testing, not code.

## 8. Bottom line to date

Weights: one slot class, nine layers, everything else silent — **freed,
surgically**. Behavior: refusal efficacy total, capability tax ~0.15–0.20
MMLU absolute, honesty tax large, coding impact unknown (one 3/20
curiosity). The two halves point the same way, and every number above
traces to a store record, a scan, or a named doc — with its caveats
attached rather than footnoted away.
