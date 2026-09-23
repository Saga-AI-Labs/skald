# Proposal: runtime provenance facet + unknown-checkpoint metric family

Status: **proposal, not implemented.** No code in this change set is committed; nothing here has been
run against a live endpoint.
Date: 2026-09-22. Author: pi-50 (`gx10-50ef`), on operator request.
Related: `2026-09-16_skald-draft.md` (OPEN-1..OPEN-5), `docs/reports/2026-09-18_edit-footprint.md`.

---

## 0. Why this proposal exists, and what it is not

Skald's stated premise is that *"every result is keyed to the exact weights it measured"*
(`README.md`). That premise is sound and is the reason this hub is worth having: the checkpoint
SHA-256 as identity anchor, with deliberately **no model registry**, is what makes Skald a measurement
store rather than a leaderboard.

This proposal argues two things:

1. The keying is **one dimension short**. It captures weights and harness code, but not the code path
   that *serves* the weights. Two runs of byte-identical weights through different kernels therefore
   land as the same measurement.
2. The current metric families answer *"how good is this model?"* and *"how damaged is it?"* well, but
   answer *"what **is** this checkpoint?"* poorly — which is the question a hub for a not-yet-functional
   BDH-CL model will be asked most.

It is explicitly **not** a claim that any existing number in the store is wrong. Every concrete example
below is drawn from an external, third-party quantisation project, not from Skald's own records, and is
used only to show a failure shape. Where I assert something about Skald's code I cite file and line;
where I assert something about a measurement I have not made, I say so.

---

## 1. Verified starting state

Checked in this working tree (`/srv/coding/skald`, HEAD `9c30f02`):

| Claim | Evidence |
|---|---|
| `RECORD_FIELDS` has no runtime/kernel/image/driver facet | `store/schema.py:14-30`; `grep -riE 'runtime\|kernel\|image\|driver\|container\|commit\|version' store/schema.py` → **no matches** |
| The schema **rejects** unknown fields, so a new facet is a schema change, not an additive no-op | `store/schema.py:99-102` — `unknown = set(record) - set(RECORD_FIELDS)` → `ValidationError` |
| `host` is only a hostname | `store/schema.py:171` — `out.setdefault("host", socket.gethostname())` |
| `protocol` is a free-text description label, set per adapter | `adapters/bdh_cl.py:53-66` (`PROTOCOLS`), `adapters/atlas.py:111` |
| The anomaly rule already enforces "do not silently compare across protocols" | `surfaces/spec.py:39` `ANOMALY_RULE = "cross-protocol-checkpoint"`, `flag_anomalies()` |
| Adapters drive vendored instruments unmodified in a subprocess and re-implement no scoring | `adapters/pi50.py:80-87` comment block; `vendor/*/PIN.md` |
| `eval_router.py` **does** emit a per-domain × per-width confusion matrix, routed/oracle/joint ppl | `vendor/bdh_cl/scripts/eval_router.py:106` (`choice = scores.argmin(dim=0)`), `:110` (`routed_ppl`), `:128-140` (confusion matrix, joint reference) |

The last row is a correction I owe the reader: an earlier draft of this proposal said Skald "has no
routing diagnostics". That was wrong — `eval_router.py` measures routing **accuracy** against the true
domain and prints a full confusion matrix. What is absent is routing **utilisation**. §3.4 is scoped
accordingly.

---

## 2. The hole: weights and code are keyed, the serving path is not

### 2.1 The failure shape, from an external project

A third-party DGX-Spark quantisation recipe publishes, for **one** checkpoint
(`vcruz305/GLM-5.3-Flash-EXL3-K2`, `declared_bits: 2.0`), two teacher-KL numbers that differ by 2.2×:

| measurement | mean KL vs BF16 teacher | top-1 agreement |
|---|---:|---:|
| recipe `docs/KLD.md` — captured **as served**, fused MoE kernels, fp8 KV | **0.3346** | 0.788 |
| fidelity-suite `reports/vcruz-k2-2bpw-packed-kld.json` — **offline packed reader**, `codebook: mcg` | **0.1552** | 0.873 |

Same weights. Same nominal protocol. Different code path. The second report's own
`routed_bits_decode_histogram` reads `{K4: 907200}` — i.e. every scored position decoded at K4 — which
is a further reason the two are not the same measurement even in spirit.

I am **not** adjudicating which number is right and the reader should not infer that either is. The
point is structural: **a provenance key that cannot distinguish these two runs will silently store them
as one measurement of one thing.** Under Skald's own rule (§4.2 of the draft plan, forbidding silent
comparison across protocols) that is the exact shape the anomaly view exists to catch — except the
anomaly view keys on `protocol`, a human-written label, and nothing in the record would have differed
had the kernel path changed.

### 2.2 Why `protocol` does not already cover it

`protocol` is a description (`adapters/bdh_cl.py:58-66`: *"random-crop cold likelihood routing,
teacher-forced, label-free argmin over prefix routes"*). It is authored by the adapter, not derived
from the environment, and the two runs above would carry an identical `protocol` string. It records
*what was measured*, not *what executed it*.

### 2.3 Proposal: a derived runtime facet

Add one field, populated by **derivation rather than assertion**:

```
runtime_sha256   # digest over a normalised runtime manifest
```

The manifest is the thing that, held fixed, makes two runs the same measurement *in the serving case*:

- server/engine identity and version as reported by the endpoint (the GLM recipe distinguishes its
  container runtime from its local source build by `root: /model` vs `root: /home/markus/models/...` —
  a distinction that today lives only in prose and in no digest at all);
- quantisation kernel identity where the path is kernel-dependent (ExLlama-class versions, fused-MoE
  on/off, KV dtype);
- the adapter's own vendored-instrument pin (`vendor/*/PIN.md` commit), so a re-pin is visible in the data.

Design constraints, so this does not become a second identity problem:

- **Optional.** Absent from `REQUIRED_TEXT` (`store/schema.py:33-42`). Records without it stay valid;
  no back-fill of existing run files.
- **Not a join key.** The checkpoint SHA-256 remains the identity anchor. `runtime_sha256` is a
  *facet for comparison*, like `seed` — you filter on it, you do not key on it.
- **Derived, never typed.** If it cannot be computed from the environment, it is `None`. An adapter
  that guesses it is worse than an adapter that omits it. (Precedent for this discipline:
  `adapters/atlas.py:14` — *"the asserted mapping is recorded verbatim … the adapter never guesses it"*.)
- **Extend the anomaly rule, do not replace it.** A checkpoint appearing under two
  `runtime_sha256` values for the same `protocol` is the same hazard class as
  `cross-protocol-checkpoint` and should surface in the same view.

Cost: one column, one derivation helper, one extra predicate in `flag_anomalies()`. It touches
`store/schema.py`, `surfaces/spec.py`, and the adapters that talk to a server
(`adapters/pi50.py`, `adapters/openai_compat.py`). It does not touch the store's file format.

---

## 3. Metric family for an *unknown* checkpoint

Ordered by information gained per GPU-hour. Each is written as the hub would ask it, not as a paper.

### 3.1 Determinism probe — a gate, not a metric

> Same prompt, `temperature=0`, N repeats. Report **distinct-output rate** and the first-divergence
> position distribution.

Why first: every capability number in the store is a sample from a distribution. If the greedy path is
not reproducible, a single run is a coin flip and the record should say so.

**This must gate the others, not sit beside them.** A checkpoint failing this probe should have its
capability records marked non-reproducible rather than stored as a bare scalar.

Evidence that the distinction is real and not hypothetical: the same external recipe asserts
`bitwise_deterministic: true` in its KLD reports while a separate determinism study in the same suite
reports a noise floor of 8.7e-4 nats; and on our own endpoint an independent probe measured **12/12
distinct outputs at `temperature=0`** on one prompt family. Both statements can be true of the same
system and mean different things. A field named in the record resolves the ambiguity; a boolean in a
README does not.

Cost: N forward passes, no training, no reference model. Cheapest item in this proposal.

### 3.2 Teacher-forced likelihood parity — reported as a distribution, not a mean

> Per token, KL(reference ‖ candidate) over sealed held-out contexts, reference logprobs precomputed.
> Report **mean, median, p95, p99, p99.9, max, CVaR₉₅**, stratified by domain.

Two clauses are load-bearing and both are cheap to violate:

**(a) The tail is the measurement.** A mean-only KL is a gate that can be passed by a distribution whose
worst 1% disagrees hard. In the external data above, the arm with mean 0.155 carries `p99: 1.65` and
`max: 5.86`; the arm with mean 0.335 carries `p99: 3.33`, `p99.9: 6.47`. A gate of
`mean_tokenwise_kld < 0.06` — and such a gate exists in that project, `quality_gate: {metric:
mean_tokenwise_kld, threshold_lt: 0.06}` — is a statement about the bulk of the distribution and says
nothing about the positions where low-bit quantisation actually does its damage.

**(b) Stratify by domain, and treat the spread as a finding.** The same external report's per-domain
split:

| domain | mean | p50 | p99 | max |
|---|---:|---:|---:|---:|
| general | 0.173 | 0.046 | 1.65 | 5.86 |
| legal | 0.251 | 0.099 | 2.54 | 8.27 |
| code/agentic | 0.127 | 0.013 | 1.65 | 7.27 |
| reasoning/termination | 0.067 | 0.00015 | 1.23 | 6.04 |

Damage varies by ~3.7× across domains and the ordering differs by statistic (legal is worst on mean and
p50; code is worst on max relative to its own bulk). Pooling those into one scalar would hide a
result of independent interest — and for a capacity-routed model, domain-dependent damage is precisely
the phenomenon under study.

Cost: one forward pass per context, no generation. The reference is precomputed, so this is the
highest-value item per GPU-hour in this proposal.

**Prerequisite, stated plainly:** this needs a reference capture **of the model being scored**. The
external fidelity suite is single-model — its manifest names one BF16 model, one `hidden_size`, one
`vocab_size`, and ships that model's own `final_norm` + `lm_head`. Replaying a different model's hidden
states through it is meaningless. Adopting the *protocol* therefore requires producing our own
reference capture; adopting the external *data* does not.

### 3.3 Length-stress curve — against generated length, not prompt length

> Accuracy, refusal rate, and failure rate as a function of **completion** length at fixed prompt
> length. Report the length at which behaviour changes, and the failure mode at each step.

The distinction matters. In the external project, every long-*prompt* probe passed — 65k and 82k token
prompts returned normally — while the crash was carried by a **76-token prompt driving a 32,768-token
generation** whose constraints were near-unsatisfiable, so the model looped until it exhausted its
budget. Long-prompt benches did not find it; the common serving path is the one that did.

For an unknown checkpoint this is also the cheapest place to see a model's failure *shape* rather than
its score: a model that degrades gracefully and a model that loops are different objects, and a single
accuracy number does not tell them apart.

Cost: generation, so budget accordingly — cap the sweep and run it after §3.1 has established whether
repeats are even meaningful.

### 3.4 Utilisation — the question `eval_router.py` does not ask

`eval_router.py` asks: given a block, which prefix width should serve it, and was that the true
domain? It reports an argmin choice, a confusion matrix, and routed/oracle/joint ppl
(`vendor/bdh_cl/scripts/eval_router.py:106,110,128-140`). That is **accuracy**.

It does not ask: **does every expert still do anything?**

For a capacity-routed model, add per layer:

- the load histogram over experts/capacity slots,
- **dead-slot fraction** — slots never selected, or selected with mass below a stated threshold,
- gate entropy, and its spread across domains.

Why this belongs in the hub and nowhere else in it: a model can achieve a routing accuracy of 100%
while half its capacity slots are dead, because accuracy counts only the served positions. Dead slots
are invisible to every metric currently in the store, and they are exactly what the P5 storage thesis
and the grow-without-retraining premise rest on. `p5_inchain` checks structural properties of the
checkpoint (`adapters/bdh_cl.py:63-66`: *bit-exact masked base block, zero moments, grown-nonzero*);
it does not observe what the router does at run time.

**Correction, in the interest of not over-claiming.** An earlier draft of §3.4 said the vendored model
has "no gate outputs to hook". That is not accurate for the default configuration. In
`vendor/bdh_cl/bdh.py:252,267` the selection is

```python
x_sparse = _k_sparse_relu(x_latent, C.k_sparse_ratio) if C.k_sparse_ratio > 0 else F.relu(x_latent)
```

with `k_sparse_ratio: float = 0.0` (`bdh.py:39`, `pipeline/config.py:56`) — i.e. **by default there is
no top-k selection at all**, plain ReLU, and there is no gate to instrument. The probe is therefore
conditional on `k_sparse_ratio > 0`, and where it is zero the honest result is *not applicable*, not
zero.

That default also carries a warning this proposal should state rather than bury. The vendored source
records, at `bdh.py:11-17`:

> *"ratio-based k is NOT width-invariant. Under growth (N → N′) k grows with the axis and the retained
> set of OLD activations changes … Sparse growth requires holding k ABSOLUTE across the growth step …
> masking the new block in either order does NOT restore exactness (retraction 6754e83, derivation
> section 8)."*

So a utilisation measurement taken at `k_sparse_ratio > 0` is **not comparable across a growth step**
unless k is held absolute. Any record produced by this probe must carry `k_sparse_ratio` and the width
it was measured at, or the hub will accumulate utilisation numbers that cannot be put in the same
chart. This is the same hazard §2 is about, arriving from the other direction.

Cost: one forward pass with the selection already computed; near-free when `k_sparse_ratio == 0`, in
which case do not run it.

### 3.5 Perturbation sensitivity

> Apply a small, stated perturbation to the input — substitute one token in a fixed position, permute
> whitespace in a code block, reorder two independent clauses — and report output stability (Jaccard
> over tokens, and whether the verdict changes).

Catches a model that has learned the surface of the task rather than the task. Costs no generation
budget beyond a second pass per item, and unlike §3.3 it needs no long completions at all.

### 3.6 Null-model calibration — so that a score means something

Run the capability suites against a **deliberately degenerate** reference: a constant-output stub, and a
uniform-random-weight checkpoint at the same shape. Store the results as ordinary records under a
distinct `protocol` (`null-stub`, `null-random`) and a fixed `model_checkpoint_sha256`.

Then every capability record has a floor to be measured against, and "84.17" is a statement relative to
something. Today the hub can say that two checkpoints differ; it cannot say whether either is above
noise. The determinism study in the external suite publishes a noise floor (8.7e-4 nats) for exactly
this reason and we have no equivalent.

Cost: trivial, once. Value: it is what turns a list of numbers into a measurement.

---

## 4. Sequencing, and what I am not asking for

Nothing here requires a schema rewrite of existing data, and nothing here requires deleting a record.

1. **§3.6 null-model calibration** — cheapest, unblocks interpretation of everything else.
2. **§3.1 determinism probe** — cheap, and it gates §3.2-§3.5.
3. **§2 runtime facet** — small, and it makes §3.2 trustworthy by making the serving path visible.
4. **§3.2 likelihood parity** — highest value per GPU-hour, but it has a prerequisite (our own
   reference capture) and should not start before §2 exists, or its numbers will be exactly as
   ambiguous as the pair in §2.1.
5. **§3.4 utilisation** — near-free, but conditional on `k_sparse_ratio > 0`; run it when the
   configuration has one.
6. **§3.3 length-stress** — the only item that spends real generation budget; schedule it last and
   cap it.

I am **not** asking to replace `protocol`, to make `runtime_sha256` a join key, to back-fill existing
run files, or to add a model registry — the absence of one is a feature.

---

## 5. Open questions for the maintainer

1. Is `runtime_sha256` better as a column beside `script_sha256`, or as a second element of the
   `protocol` label with the anomaly rule split on the digest? The former keeps `protocol` human; the
   latter needs no schema change. §2 prefers the former and says why.
2. Who owns the runtime manifest — the adapter, or the store at `put()` time? Derivation at `put()`
   makes it uniform; derivation in the adapter lets a server-backed adapter record what only it can
   see.
3. For §3.2, is a reference capture of the BDH-CL model in scope for this project, or does the hub
   stay black-box (HTTP only)? If black-box only, §3.2 must be implemented over `/v1/completions`
   with `prompt_logprobs`, which our endpoint supports, and the tail statistics become the whole
   result — no hidden states are available to check them against.
4. §3.4 presumes the router is reachable from the adapter. The vendored instrument is driven in a
   subprocess and re-implements no scoring (`adapters/pi50.py:80-87`); a utilisation probe needs either
   a new vendored script under `vendor/bdh_cl/scripts/` or an exception to that rule. Which?

---

## 6. Provenance of the external figures

Every third-party number in this document was read on 2026-09-22 from public artefacts of a DGX-Spark
quantisation recipe and its associated fidelity suite (`vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe`,
`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`, `malaiwah/GLM-5.3-Flash-fidelity-suite-v1`), by way of
their published reports and manifests. None was measured by Skald, none was measured by me, and no
number here is offered as a measurement of a Skald record. The 12/12 determinism observation is ours,
taken on our own endpoint, and is quoted as an observation about one prompt family on one day — it is
not a property of the engine and should not be read as one.
