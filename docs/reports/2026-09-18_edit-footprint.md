# What this abliteration actually changed (weight-level forensics)

Method note: every number below was measured on the two local HF snapshots, not taken from the
model card or from memory. Scripts: `/tmp/tensor_diff.py`, `/tmp/edit_forensics.py` (pure Python +
mmap, no numpy in `.venv-hf`). Citation gate: `../cite-check.sh eval/EDIT-FOOTPRINT.md`.

## 1. The arms really are the same model except for the edit

| | ablit | stock twin |
|---|---|---|
| repo | `drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only` | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` |
| snapshot | `b3b60a4c…` | `925d7be6…` |
| tensor names | 301,730 | 301,730 (identical set, 0 unique to either side) |
| shard map | 35 shards | identical placement on 301,730/301,730 tensors |
| payload `total_size` | 105,839,538,520 B | 105,839,538,520 B |
| `config.json` | — | zero differing keys |

`ABLIT_META.json` on the ablit side declares `"stock": "Mia-AiLab/Qwen3.8-Flash-Next-NVFP4"` and the
tag `Qwen3.8-Flash-Next-Abliterated-NVFP4-spark-oproj-L15-47`. So axis C/D deltas are attributable to
the edit, not to a different base, a different quantisation, or a different packaging pass.

That last point matters: **26 of 35 shards are byte-identical** between the arms. A wholesale
re-quantisation or re-packaging by the ablit author would have perturbed every shard, so the
differences we see are the edit itself, not pipeline noise.

## 2. Edit footprint: 32 tensors, 0.011 % of all tensors

Scanned 301,730 tensors across 35 shards (~6.5 min, both sides read once). Changed tensors:

```
dtype      count  what
F8_E4M3      18   self_attn.o_proj.weight (9), self_attn.q_proj.weight (9)
U8           14   block weight_scale: linear_attn.in_proj_qkv (6), .in_proj_z (6),
                  self_attn.q_proj (1), self_attn.v_proj (1)
```

Layers touched: 15, 19, 20, 23, 24, 27, 28, 31, 32, 35, 36, 39, 43, 44, 47.

The pattern is architectural, not arbitrary. This is a hybrid-attention model: the nine
**full-attention** layers in that span (15, 19, 23, 27, 31, 35, 39, 43, 47) each had **both**
`o_proj.weight` and `q_proj.weight` rewritten; the six **linear-attention** layers (20, 24, 28, 32, 36,
44) had only their input-projection quantisation scales touched. Layer 47 additionally carries
`q_proj`/`v_proj` scale changes.

So this is an attention-pathway edit spanning mid-to-late layers — matching the author's own
`oproj-L15-47` tag — and it edits **every** full-attention layer in that range rather than a selected
subset of heads.

## 3. How violent is it

Fraction of bytes that differ per edited tensor:

| tensor | bytes changed |
|---|---|
| `self_attn.o_proj.weight` × 9 | **51.3 % – 64.5 %** |
| `self_attn.q_proj.weight` × 9 | 2.1 % – 5.2 % |
| `linear_attn.*.weight_scale` × 12 | 0.8 % – 6.0 % |
| `self_attn.{q,v}_proj.weight_scale` × 2 | 1.7 %, 4.6 % |

Median 5.0 %, max 64.5 %. Decoding the float8 `weight_scale` tensors as e4m3 block scales: most moved
by single-digit percent, i.e. entire quantisation blocks were rescaled.

Byte-level difference overstates semantic difference in a quantised tensor (one code step flips a
byte). But over half the codes changing in nine output projections is not a light touch. For
orientation only: the untouched MLP/expert weights, embeddings, and all other layers are bit-exact.

## 4. Why this is the wrong shape for a "clean" abliteration

The canonical result in this area is that refusal is carried by a low-dimensional direction which can
be erased from the residual stream with little else disturbed — arXiv:2406.11717v3 "Refusal in
Language Models Is Mediated by a Single Direction", which describes its own method as surgically
disabling refusal "with minimal effect on other capabilities". Note the claim being made there:
**preservation**, not improvement.

This build does not implement that recipe. It rewrites dense attention projection matrices — the
mixing hardware every downstream circuit reads through — instead of deleting one direction at one
site. That is a legitimate alternative family of attack (attention-head ablation), but it predicts
exactly the outcome we measured: refusal gone completely, plus a measurable tax on tasks that depend
on mid-layer attention doing precise retrieval-and-combine work.

Cross-check against published measurements of capability impact, where they exist:
arXiv:2512.13655v2 "Comparative Analysis of LLM Abliteration Methods: A Cross-Architecture
Evaluation" reports average GSM8K change for single-pass tools of **−0.28 pp and −0.13 pp** across
three models, while Bayesian-optimised abliteration produced distribution shift with KL divergence
0.043–1.646 and model-dependent capability impact. Our GSM8K delta on this build is **−1.7 pp** —
roughly six times the larger of those two, on a metric where the field's expectation is fractions of
a point.

And the off-target-effects literature agrees with our direction, though not usually at our size:
arXiv:2607.17427v1 "Abliteration Is Not a Scalpel: Off-Target Effects of Refusal Removal on Decision
Disposition Across Model Families" uses a task that elicits no refusals at all, so any arm delta is
pure side effect, and finds abliterated models systematically more optimistic (+12.2 pp and +7.4 pp
on two MoE families), self-justifying at greater length, and using fewer explicit uncertainty words.
Our epistemics result is the same phenomenon seen through a different probe: abstention 8/18 → 0/18
(paired exact p = 0.0078), fabricated-citation rate 61.1 % → 88.9 %.

## 5. What the model card claims as evidence

The ablit card's only quantitative capability statement is a refusal suite:

> `Refusal suite | 32/32 BYPASS (QuantTrio-style hard suite) · 0 refuse · 0 garble · 0 errors`

That measures axis A and eyeballs for gross degeneration. It cannot detect a 7.5-point knowledge
delta or a 28-point calibration delta, and the card makes no comparison against the stock twin on any
capability metric. Nothing here contradicts the author's claims; it extends them into a region the
author did not measure.

## 6. Consequences for how we talk about this build

1. Axis A/B are unambiguous successes: 0/2,538 harmful refusals, 0/2,085 benign refusals
   (`results/scores/{stock,ablit}-refusal.json`, `blocks/efficacy_on_unsafe` and
   `blocks/over_refusal_on_safe`; the two foot to `blocks/all` n=4,623). Stock twin for the same
   blocks: 93.5 % and 77.0 %.
2. Axis C fails the pre-registered band, and the failure is **content-specific**: no positional bias
   (mean signed letter offset −0.03 stock vs −0.01 ablit), 86.2 % letter agreement between arms,
   near-equal answer lengths (GSM8K median 1,095 vs 1,023 chars), and divergent GSM8K answers that
   are mostly arithmetic slips (44/127 within 1 % of gold; 12 at exact integer factors 2×/3×/4×/6×).
   This is not laziness, verbosity, or format drift; it is the model being confidently wrong about
   facts and arithmetic.
3. Before generalising "abliteration costs capability", test whether the cost is representational or
   behavioural: token-level likelihood parity (`prompt_logprobs` on held-out text, both arms) is the
   style-free metric the papers use, and a thinking-ON re-run tests whether deliberation buys the gap
   back. Both need one arm swap (~12 min) plus ≲30 min GPU each.
4. The clean scientific move, if we want a real contribution: apply a rank-1 refusal-direction edit
   to the stock twin ourselves and run the identical harness against this o_proj build. Same lineage,
   same serving stack, two recipes — that isolates recipe choice from refusal removal itself.
