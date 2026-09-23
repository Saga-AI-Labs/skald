# Benchmark coverage for a capability footprint: what existed, what was missing, what was added

Date: 2026-09-23 · Author: pi-50 (`gx10-50ef`) · Status: **implemented**, see §4
Companion to `2026-09-23_skald-usability-notes-first-hands-on.md`.

---

## 1. The question

A footprint of Qwen3.8-Flash-Next (and later GLM-5.3-Flash-EXL3-K2 through the
identical instrument) was asked to cover five axes: **coding, math, agentic
tasks, logic/reasoning, general knowledge** — measuring precision and
intelligence, not speed.

The question was whether MMLU + HumanEval are sufficient. They are not, and the
gap is not evenly distributed.

## 2. What `openai_compat` actually measured

Read from the dispatch table, not from the docs (`adapters/openai_compat.py`,
`handlers` in `OpenAICompatAdapter.run`): exactly **two scored tasks**, `mmlu`
and `humaneval`, plus the four diagnostics added on 2026-09-23.

MMLU was drawn from **four subjects** (`_MMLU_SUBJECTS`):
`college_mathematics`, `high_school_statistics`, `college_biology`,
`logical_fallacies`.

| axis | status | why |
|---|---|---|
| coding | **covered** | `humaneval`, execution-verified pass@1 |
| math | **partial** | MMLU math subjects are **four-way multiple choice** |
| logic/reasoning | **partial** | one subject (`logical_fallacies`), MC |
| general knowledge | **thin** | one subject (`college_biology`); no factuality measure |
| agentic | **absent** | no multi-step or tool task existed anywhere in the repo |

Two of those five were therefore effectively unmeasured, and the math number
carried a specific defect worth naming:

**Multiple-choice math is not a math test.** Guessing scores 0.25, and a model
can select the correct number without computing anything. A 0.25 "math" score
and a 0.25 "no maths at all" score are the same number. The only thing that
distinguishes them is a free-form instrument.

## 3. The constraint that decided the design

The obvious move — add GSM8K, MMLU-redux, BBH, TruthfulQA — was **not
available**, and I verified this rather than assumed it:

```
cais/mmlu            college_mathematics   OK   n=100     <- control, known-good
gsm8k                default               FAIL
openai/gsm8k         default               FAIL
lukaemon/hellaswag   default               FAIL
```

`_DEFAULT_DATASETS_SERVER` mirrors **only the two datasets this adapter already
used**. Every new dataset id fails to resolve, while the control succeeds — so
this is a serving-scope limit, not a network fault, and no hub-fetchable
benchmark could have been used. (`datasets` is not installed either.)

What *was* available, already on disk under `vendor/pi50_eval/data/` with
provenance in `PIN.md`:

| corpus | items | gold | usable for |
|---|---|---|---|
| `gsm8k.jsonl` | 1,319 | numeric | **free-form math** |
| `simpleqa.jsonl` | 4,326 | short text | **factuality / knowledge** |
| `cais/mmlu` (server) | 57 subjects | letter | **widen reasoning/knowledge** |

That last row is the non-obvious one: **MMLU has ~57 subjects and the server
serves them**, so logic/reasoning and knowledge could be widened substantially
using the dataset already permitted, with no new dependency at all. Verified
resolving: `formal_logic`, `logical_fallacies`, `college_mathematics`,
`high_school_mathematics`, `high_school_statistics`, `college_computer_science`,
`machine_learning`, `conceptual_physics`, `astronomy`, `college_biology`,
`high_school_us_history`, `professional_law`, `college_medicine` (13/14 probed).

## 4. What was added

`adapters/local_bench.py` (new) + three handlers in `openai_compat`:

**`gsm8k` — free-form math.** The single highest-value addition: it is what
makes the math axis mean something. Answers are extracted by a preference order
(`#### N`, then a labelled *final answer*, then the last number on the last
line — *last*, because in a worked solution the final number is the answer and
the first is usually a given). Reports `accuracy` **and** `answerable`.

**`simpleqa` — short-form factuality.** Normalised containment against a gold
phrase. Deliberately kept as its own metric: gold is free text, so scoring
under-counts a correct answer phrased differently and over-counts a model that
quotes the gold back without meaning it. It must not be averaged with the
exact-match tasks.

**`tool_use` — multi-step tool driving.** The agentic axis. N chained
arithmetic steps that must be worked one `calc` call at a time; reports
`solved`, `tool_calls_per_item`, `no_tool_call`.

**`subject_set`** on `mmlu`: named presets (`default`, `reasoning`,
`knowledge`, `footprint`). The default is unchanged so no existing run is
silently widened.

### Two design decisions worth defending

**The agentic gold is computed, not authored.** There is no published agentic
corpus on disk and none was fetchable, so the task is constructed — which is
where an "indicative metric" normally becomes theatre. What keeps it honest:
the answer is the value of the expression the model is asked to work through,
so correctness is checkable with **no answer key anyone wrote**, and `seed`
reproduces the item exactly. It was verified by replaying each chain through
the same `calc` tool the model is given (3/3 seeds matched). The measured skill
is not arithmetic — the tool does that — but whether the model issues the
calls, carries each observation into the next, and stops. The record's
`protocol` says `CONSTRUCTED task — no published agentic corpus was available;
treat as indicative, not as a named benchmark`.

**Unanswerable is a third bucket, never folded into WRONG.** Taken from the
vendored harness, which recorded its own lesson verbatim:

> *"an answer we cannot read is NOT also an answer we read incorrectly: counting
> it in both buckets made the tallies exceed n, which is how this line got
> written."*

So `accuracy = correct / n` over **all** items — an item the model never
answered is a failure, not an exclusion. Dropping unparseable answers would let
a model that never formats an answer score 1.0 on zero answers. `answerable`
is reported separately so a formatting failure can be told apart from an
arithmetic failure.

**The `calc` tool is not `eval` plus a shrug.** The model supplies the
expression, so the gate is load-bearing: `**` is rejected outright (it is legal
arithmetic and would pass a character whitelist, but `2**99999999` is a
denial-of-service on the serving host — and the resulting integer is so large
that *stringifying* it raises, outside any `try`); result magnitude is capped;
evaluation runs with no builtins. All three were found by attacking my own
implementation before shipping it.

## 5. Findings from the first footprint

**The endpoint is not reproducible at temperature 0.** 6 identical requests
produced **6 distinct outputs**; skald's own probe reports
`reproducible = 0.0`, 2 distinct outputs from 6 repeats, first divergence at
character 2. This is a gate, not a footnote: every accuracy figure below
carries run-to-run variance beyond its own binomial CI, and a single run of any
task is a sample, not a measurement. It also makes `max_tokens` truncation
non-deterministic — the same GSM8K item was observed correct on one run and
`NO_ANSWER` on the next.

**`num_fewshot` is a silent 2× swing.** 0-shot 0.20 vs 5-shot 0.40 on identical
items, both stored as `metric="accuracy"`, `task="mmlu"`. 0.20 on four-option MC
is *exactly* chance (`P(X≤2 | p=0.25, n=10) = 0.53`), so the 0-shot figure is
not a weak result but **no** result. It is pinned to 5 in the runner and should
become a first-class filterable field.

## 6. Still open

1. **`num_fewshot` as a filterable field** — see the usability notes §1. Blocks
   trustworthy cross-model comparison; do it before the GLM run.
2. **`simpleqa` scoring is soft.** Containment on free text. A stricter
   EM/F1 with normalisation, or an LLM judge behind an explicit opt-in, would
   make it comparable to the others. Until then it is directional only.
3. **`tool_use` is constructed.** If a published agentic suite becomes
   available on the datasets server, prefer it. The current task should be
   read as "can it drive a tool loop and terminate", not as an agentic score.
4. **GSM8K truncation sensitivity.** At `max_tokens=640` some items still
   truncate into `NO_ANSWER`. `answerable` makes this visible; if it is low,
   the accuracy figure is partly a budget artefact rather than a capability
   result.
5. **One run is not a measurement** while the endpoint is non-deterministic.
   Repeats with a spread are needed before any number is quoted as *the* score.
