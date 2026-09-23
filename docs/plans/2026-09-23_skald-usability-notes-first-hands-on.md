# Skald usability notes — first hands-on session against a live OpenAI-compatible endpoint

Date: 2026-09-23 · Session: pi-50 on `gx10-50ef` · Repo at `ff38cad`
Task: drive skald over HTTP against `http://127.0.0.1:8888/v1`
(`qwen3.8-flash-next`, `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`).

This is a **usability** report, not a defect list. Nothing here blocked the work; the
items are ranked by how much friction they cost a person who is not already inside
the code.

---

## 0. Verdict up front

**The API surface is good and it works end-to-end.** `run_benchmark` → `job_status` →
`query_results` is a clean three-call loop, jobs are durable, records carry CIs, and
the capability spec is genuinely the single source of truth (the UI is generated from
it, and `docs/usage.md` matched reality everywhere I checked).

Two things cost me real time, and one is a **measurement-validity trap** that will
directly affect the Qwen-vs-GLM footprint. That third one is the reason this file
exists.

---

## 1. Measurement validity: the shot count is a silent free-text field ⚠ **do this first**

Observed in one session, same model, same task, same items:

| run | `num_fewshot` | `max_samples` | accuracy |
|---|---|---|---|
| 1 | 5 (default) | 100 | **0.38** |
| 2 | 0 | 3 | 0.00 |
| 3 | 0 | 10 | **0.20** |
| 4 | 5 | 10 | **0.40** |

**0-shot and 5-shot differ by 2× on identical items.** Both are stored as
`metric="accuracy"`, `task="mmlu"`, `adapter="openai_compat"`. A footprint query that
forgets the shot count averages them, or worse, compares a Qwen 5-shot against a GLM
0-shot and reports a real-looking difference that is entirely an artefact.

0.20 on 4-option MC is *exactly* chance (`P(X≤2 | p=0.25, n=10) = 0.53`), so the 0-shot
number is not a weak result, it is **no information** — and it is indistinguishable
from a broken prompt.

**Why it is easy to hit:** `num_fewshot` has a non-obvious default (5) and is set only
through the opaque `config` blob. The two runs above differ by one key in a JSON
string.

**The mitigation that already exists** — and it is good: the shot count *is* recorded,
inside the free-text `protocol` string (`"… 5-shot letter-choice accuracy …"`), and
that string also carries the endpoint URL, served model name, temperature, `max_tokens`,
seed, and the honest `UNVERIFIED endpoint identity … never compare with weighed-in
records` warning. So nothing is lost.

**The gap:** it is recoverable only by knowing the whole string in advance. `protocol`
is equality-filterable (verified: exact string → `count=2`), so there is no way to ask
*"all 5-shot mmlu"* without already possessing the full 145-character string.

**Proposals** (smallest first):
1. Promote the shot count to a first-class field (`num_fewshot`, or reuse `seed`-style
   numeric fields) so it is equality-filterable like `n` and `value`. ~10 lines, and
   `SPEC_VERSION` → 3.
2. Until then: **make `num_fewshot` mandatory in any footprint config** and pin it in
   the run script, so it can never silently default.
3. Consider refusing `num_fewshot=0` for MMLU, or labelling such records with a
   distinct `metric` (`accuracy_zero_shot`). A number equal to chance is not a
   measurement, and storing it under the same metric name as a real one is the
   dangerous part.

---

## 2. `run_benchmark` has no timeout, and "running" is indistinguishable from "wedged"

`jobs.py:103` executes in a **daemon thread inside the API process** — correct for a
local tool, and the `orphaned`-on-restart handling is the right call. But:

- there is **no per-job deadline**; the only timeout is the 300 s *per-request* HTTP
  timeout (`adapters/openai_compat.py:149`), which does not bound a job that makes many
  requests;
- `job_status` reports `record_count: 0` for the whole duration, so a job doing 100
  sequential samples is **visually identical** to a job blocked forever;
- `record_count` only updates at completion, so there is no progress signal at all.

My first 20-sample job sat at `running / 0` for ~5 minutes and I had to reach for
`/proc/<pid>/wchan` (`poll_schedule_timeout`) and `ss -tnp` to prove it was alive and
waiting on HTTP. It was — it was queueing behind **my own session**, which is served by
the same endpoint. That is a legitimate contention, not a bug, but the tool gave me no
way to know that.

**Proposals:**
1. Surface progress: update `record_count` incrementally, or add `items_done`/`items_total`.
2. Add an optional per-job `deadline_s` → `failed` with a clear error, rather than
   relying on the per-request timeout.
3. Cheap and high-value: have `job_status` echo the **endpoint URL and last activity
   timestamp**, so "waiting on a busy server" is readable without `ss`/`/proc`.

---

## 3. Two small API-surface papercuts

**(a) `config` is typed `string` but the docs send an object.** `surfaces/spec.py`
declares `config` as `kind="string"` ("JSON object string"); `docs/usage.md:96` sends a
nested object. **Both work** — `api/app.py:311` folds a dict into a JSON string — so
this is not a bug, but the spec and the docs describe different contracts, and a person
writing a client from the spec alone will send a string and a person following the docs
will send an object. Accepting both is right; saying so in one line in the spec
description would remove the doubt.

**(b) `list_suites_adapters` on an empty store returns `[]` for both keys.** Correct —
it lists values *present in the store*, not registered adapters. But the name reads like
"what can I run", and my first instinct was to file it as a bug. A one-line summary
tweak ("values present in the store; for what *can* be run see `docs/usage.md`") would
prevent that misreading. The genuinely useful thing — "which adapters/tasks are
available" — currently has **no API operation at all**; it lives only in `docs/usage.md`.
Worth considering a `list_capabilities` operation, since the data already exists in
`surfaces.spec`/the adapter registry.

---

## 4. Things that worked, and deserve saying

- **Zero regressions from the 6-commit pull**, verified properly: a worktree at
  `b21b9e8` gave the **same 15 failures** as `ff38cad` (171 → 269 passed). The 15 are
  environmental: `FileNotFoundError` on `/media/data/coding/OBLITER…` (saga/jlens/pi50
  external suite binaries) plus three "live default store" tests asserting
  `{bdh_cl, pi50}` families exist in an empty store.
- **`runtime_sha256` is live and populated** on real records
  (`bd850a5aa519…`), i.e. the §2 facet is already doing its job, and the
  "derived, never typed" discipline holds.
- **CIs are correct**, not decorative: stored `[0.2849, 0.4751]` reproduces the 95 % CI
  for `n=100` and *not* for `n=20` — so `n` is trustworthy, which matters because `n` is
  how you will catch a footprint that silently shrank.
- **`script_sha256` + `model_checkpoint_sha256` + `runtime_sha256` all land**, and the
  checkpoint hash is stable across runs (`00f39f2e…`), which is what makes a cross-model
  comparison auditable later.
- The `protocol` string's `UNVERIFIED endpoint identity … never compare with
  weighed-in records` warning is exactly the right instinct, and it is the reason the
  §1 problem is recoverable at all.

---

## 5. Method notes (my errors, so the next person does not repeat them)

Four of my own tooling errors each produced a **false clean result**, which is the
dangerous failure mode:

1. **pytest `FAILED` lines carry ANSI colour.** Grepping for `FAILED` without stripping
   `sed -E 's/\x1b\[[0-9;]*m//g'` matches **nothing** → a spurious `before=0 after=0` →
   a free "no regressions". Strip colour *before* any pass/fail comparison.
2. **Comparing two trees needs path normalisation** (`sed -E 's#^.*/tests/#tests/#'`),
   otherwise `comm` reports every test as simultaneously new *and* fixed.
3. **A `cd` at the top of a shell block moves the rest of the block.** My first
   before/after comparison ran both sides in the *same* tree and therefore agreed.
4. **No system `pytest`** — use `.venv/bin/python -m pytest`.

And two claims I made and then retracted, recorded because the retraction is the point:
*"protocol filtering is broken"* and *"max_samples is silently ignored"* were both
**my** encoding/reading errors, not skald's. Both fields work.

---

## 6. Recommended order

1. **§1** — promote the shot count to a filterable field, and pin `num_fewshot`
   explicitly in every footprint run. Do this *before* the Qwen/GLM comparison, or the
   comparison is not trustworthy.
2. **§2.1** — incremental progress in `job_status` (removes the `ss`/`/proc` archaeology).
3. **§3b** — a `list_capabilities` operation.
4. **§2.2/§3a** — per-job deadline; one clarifying line in the spec.

None of these are architectural. §1 is the only one that changes what you can
*believe*, so it goes first.
