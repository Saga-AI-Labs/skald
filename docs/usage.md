# Skald usage guide

How to install Skald and run each surface end to end. See the authoritative
design plan in
[`docs/plans/2026-09-16_skald-draft.md`](plans/2026-09-16_skald-draft.md) and
the quick start in [`../README.md`](../README.md). All commands below were run
verbatim inside a repository checkout; commands guarded by "requires" notes
depend on machines or data that are not shipped in this repo.

Skald is local-only: no cloud LLM participates anywhere in the evaluation or
serving chain. Every surface is read-only over the store except the adapters,
which write new records.

## Install and environment

Requirements: Python >= 3.11. The only runtime dependency is the stdlib; the
`dev` extra adds pytest for the test suite.

```bash
cd skald
python3 -m venv .venv
. .venv/bin/activate

pip install -e ".[dev]"

pytest            # full suite, including real-material adapter executions
```

`pytest` runs every test in `tests/`. Two tests execute actual adapters over
real local models (`tests/test_jlens.py`, `tests/test_saga.py`) and need the
suite prerequisites listed below (saga fetches its HF datasets over the
network); the rest are hermetic and offline.

### The result store

- Default location: `.skald/store/` under the directory you run from.
- Override: the `SKALD_STORE_DIR` environment variable, or the `--store-dir`
  flag on the two surfaces (`python -m api`, `python -m ui`).
- Layout: SQLite (`index.sqlite3`) indexes immutable per-run JSON files in
  `runs/`. The JSON files are authoritative; SQLite is a query index. Records
  are append-only — nothing is ever overwritten or deleted.

The adapter CLIs persist through the same store, so point them at a scratch
directory with `SKALD_STORE_DIR` when you only want to try something out:

```bash
export SKALD_STORE_DIR="$PWD/.skald/demo-store"
```

## Running a suite adapter

Every adapter shares one shape:

```bash
python -m adapters.<name> <model> <task> [--config '<json>']
```

It runs the real suite machinery for `<model>` + `<task>`, writes the unified
records to the store, then queries them back and prints a summary. Exit codes:
`0` = all records persisted and read back, `2` = no records produced, `3` =
read back fewer records than persisted. `--help` prints the exact upstream
usage for each adapter.

All four suites, tasks, and the `--config` keys they honor:

| Adapter | `python -m` | Tasks | `--config` keys |
|---|---|---|---|
| bdh_cl | `adapters.bdh_cl` | `router`, `domain_eval`, `p5_inchain` | `repo`, `python`, `routes`, `domains`, `window`, `crops`, `batch`, `oracle_routes`, `mb`, `iters`, `parent`, `timeout` |
| bdh_likelihood | `adapters.bdh_likelihood` | `capture_reference`, `likelihood_parity` | `repo`, `python`, `bundle_dir`, `reference_bundle`, `contexts`, `context_files`, `max_contexts`, `max_chars`, `block_size`, `dtype`, `seed`, `timeout` |
| bdh_router_util | `adapters.bdh_router_util` | `router_utilization` | `repo`, `python`, `contexts`, `context_files`, `max_contexts`, `max_chars`, `block_size`, `mass_threshold`, `artifact_dir`, `seed`, `timeout` |
| pi50 | `adapters.pi50` | `manifest_check`, `run_suite`, `score_refusal`, `score_confab`, `score_capability`, `paired_compare` | `repo`, `python`, `timeout`, `suites`, `arm`, `require_model`, `protocol`, `files`, `base_url` |
| saga | `adapters.saga` | `mmlu`, `humaneval` | `repo`, `python`, `num_fewshot`, `max_samples`, `max_new_tokens`, `seed`, `timeout`, `exec_timeout`, `model_id`, `artifact_dir` |
| openai_compat | `adapters.openai_compat` | `mmlu`, `humaneval`, `determinism` | `model`, `api_key`, `timeout`, `max_tokens`, `max_samples`, `num_fewshot`, `subjects`, `seed`, `exec_timeout`, `mmlu_items`, `humaneval_items`, `datasets_server`, `datasets_cache`, `prompt`, `repeats` |
| null_model | `adapters.null_model` | `mmlu`, `humaneval` | `null_kind` (`stub`/`random`), `seed`, `mmlu_items`, `humaneval_items`, `exec_timeout` |
| atlas | `adapters.atlas` | `diff` | `atlas_url`, `atlas_job`, `with_job`, `metric`, `top_n`, `min_change_pct`, `seed`, `timeout` |

### Running benchmarks from the surfaces (no CLI needed)

Both surfaces expose `run_benchmark` (launch) and `job_status` (poll) —
the same operation set, so API and UI stay equal. A run executes the
adapter in a background job and persists its records to the same store;
closing the page does not stop it. Jobs live under `<store>/jobs/`; a
restarted server marks interrupted jobs `orphaned` instead of pretending
they continue.

- **UI:** open the `run_benchmark` view (linked from the landing page):
  pick adapter + task from dropdowns, name the run target per that
  adapter's contract (checkpoint path, endpoint URL, or hash), paste
  config as a JSON object, submit. The status page tracks the job and
  links to its records when done.
- **API:** `POST /api/v1/run_benchmark` with
  `{"adapter": ..., "task": ..., "model": ..., "config": {...}}`, then
  `GET /api/v1/job_status?job_id=JOB-…`.

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/run_benchmark \
  -d '{"adapter": "openai_compat", "task": "mmlu",
       "model": "http://127.0.0.1:8888/v1",
       "config": {"model": "qwen3.8-flash-next-stock", "max_samples": 20}}'
# {"spec_id": ..., "operation": "run_benchmark",
#  "job_id": ["JOB-…"], "status": ["queued"]}
```

Trust note: submitting a run equals running the adapter CLI locally.
The surfaces are local-only; do not expose them where you would not run
the CLI.
| jlens | `adapters.jlens` | `layer_readout`, `verbal_report`, `directed_modulation`, `multi_hop_reasoning`, `general_broadcast`, `selective_mediation` | `python`, `vendor_dir`, `lens_source`, `prompts`, `source_layers`, `dim_batch`, `max_seq_len`, `skip_first`, `dtype`, `layers`, `position`, `top_n`, `seed`, `timeout`, `artifact_dir`, `readout_prompt` |

### bdh_cl — continual-learning suite (`python -m adapters.bdh_cl`)

Runs the BDH-CL eval scripts against a grown-ladder checkpoint:
`scripts/eval_router.py` (`router`), `scripts/domain_eval.py`
(`domain_eval`), `scripts/p5_inchain_check.py` (`p5_inchain`).

Requires the BDH-CL eval code — vendored under `vendor/bdh_cl/`, which is
the adapter default, so a fresh clone works — plus a torch-capable Python
(`--config '{"python": ...}'`; the adapter falls back to
`repo/.venv/bin/python`, which exists only in a full BDH-CL checkout):

```bash
# router: label-free routing likelihood over a territory/domain layout
python -m adapters.bdh_cl /path/to/checkpoint.pt router \
  --config '{"routes": "64,128,256", "domains": "wiki:/path/wiki.txt,legal:/path/legal.txt"}'

# domain_eval: per-domain held-out cold eval
python -m adapters.bdh_cl /path/to/checkpoint.pt domain_eval \
  --config '{"domains": "math:/path/math.txt"}'
```

`router` requires `routes` and `domains`; `domain_eval` requires `domains`
(each a comma-joined `name:path` spec, also accepted as a JSON list or
mapping). `p5_inchain` requires `config['parent']` — the base-phase exit
checkpoint whose grown child is the `model` argument — and **cannot be run
locally without a grown parent/child checkpoint pair**.

### pi50 — phase-1 instrument (`python -m adapters.pi50`)

Runs the frozen Pi-50 phase-1 instrument `scripts/pi50/phase1_manifest.py
--check` (vendored under `vendor/bdh_cl/`). The `model` argument is the
evaluated artifact; for `manifest_check` that is the committed
`docs/PHASE1-MANIFEST.md` in the suite repo.

The check audits a BDH-CL checkout's evidence freshness through its own git
history, so it is checkout-bound by nature: `config['repo']` must name a
BDH-CL checkout explicitly (there is deliberately no default — the adapter
refuses to silently audit the wrong repository). The instrument itself is
stdlib-only and runs under any Python. Verified end-to-end on this box (see
the walkthrough below):

```bash
python -m adapters.pi50 /media/data/coding/bdh-cl/docs/PHASE1-MANIFEST.md manifest_check \
  --config '{"repo": "/media/data/coding/bdh-cl"}'
# persisted 1 pi50 records; queried back N matching (N grows with the store)
#   manifest_check:manifest_current = 0.0 (n=1, protocol='phase-1 artifact ...')
```

The instrument treats a nonzero exit as *measurement data* (manifest fresh vs
stale), so a "0.0 / stale" verdict is a result, not a failure.

#### pi50 abliteration / refusal tasks

The same family also carries the Qwen ablit-vs-stock instrument
(`vendor/pi50_eval/`), which judges whether a refusal-removal edit is *good*:
collateral over-refusal precision, capability tax, and epistemic honesty. It is
one family, not a new one — the store's `adapter`/`suite` dimensions stay
comparable across everything Skald records.

| task | reads | metrics it emits |
|---|---|---|
| `run_suite` | a live OpenAI-compatible server | nothing directly; writes transcripts and records collection health |
| `score_refusal` | transcripts | `<block>.refusal_rate`, `.off_target_rate`, `.compliance_rate` |
| `score_confab` | transcripts | `<battery>.fabrication_rate`, `.honesty_rate`, `.attempted_accuracy` |
| `score_capability` | transcripts | `<suite>.accuracy`, `.attempted_accuracy`, `.empty_rate` |
| `paired_compare` | two transcripts | `shared`, `disagreements`, `transitions:<A->B>`, `per_class:<bucket>:<A->B>` |

```bash
# collect (arm identity is corroborated by the server, never inferred from a filename)
python -m adapters.pi50 /models/qwen3.8-flash-next-stock run_suite \
  --config '{"suites": ["xstest"], "arm": "stock", "require_model": "qwen3.8-flash-next-stock", "protocol": "v1"}'

# score the four axes from transcripts already on disk
python -m adapters.pi50 /models/qwen3.8-flash-next-stock score_refusal \
  --config '{"suites": ["xstest", "jbb"], "arm": "stock", "protocol": "v1"}'
python -m adapters.pi50 /models/qwen3.8-flash-next-stock score_confab \
  --config '{"files": ["ablit_ccb.jsonl", "ablit_fpb.jsonl"], "protocol": "v1"}'
python -m adapters.pi50 /models/qwen3.8-flash-next-stock score_capability \
  --config '{"suites": ["mmlu_c1024", "gsm8k"], "arm": "stock", "protocol": "v1"}'

# the two-arm flip table, including the REFUSED->OFF_TARGET sites
python -m adapters.pi50 /models/qwen3.8-flash-next-stock paired_compare \
  --config '{"pair_a": "stock_xstest.jsonl", "pair_b": "ablit_xstest.jsonl", "protocol": "v1"}'
```

Config keys: `python` (default `sys.executable`), `base_url`/`model` (exported
to the tool as `FN_BASE`/`FN_MODEL`), `transcript_dir`, `data_dir`, `arm`,
`suites`, `files`, `pair_a`, `pair_b`, `workers`, `max_tokens`, `timeout`,
`require_model`, `protocol` (**required** — it is the join key), `repo`.

Four guards are deliberate, because a half-failed collection otherwise produces
plausible-looking aggregates:

- `run_suite` **requires** `require_model` and aborts if fewer than 98 % of
  items were answered — a 404 storm against a swapped checkpoint looks exactly
  like a model that stopped refusing.
- `paired_compare` accepts the tool's `rc=1` **only** when the `--json` was
  written: that code means "pairing warnings", which is a measurement.
- Intervals are cluster-aware bootstrap where the suite clusters, Wilson
  otherwise, and the choice is recorded in `protocol` — a per-item interval on
  a cluster-sampled suite would be pseudo-replication.
- Rates are stored as **proportions**, matching `saga.py`, so one query can
  compare across adapters.

Transcripts are model outputs and stay on disk; only aggregate records enter
the store. See `docs/2026-09-14_ablit-eval-suite.md` for the full manual and
`docs/plans/2026-09-14_abliteration-eval-plan.md` for the pre-registered
thresholds.

### saga — general-purpose suites (`python -m adapters.saga`)

Runs the saga evaluation machinery over a local model:
`mmlu` (5-shot letter-choice accuracy over the cais/mmlu test subjects) and
`humaneval` (an adapter-side greedy 0-shot pass@1 shim over saga's
`configs/evaluation.yaml` entry — saga has no `run_humaneval` runner of its
own).

Requires the saga benchmark code — vendored under `vendor/saga_benchmarks/`,
which is the adapter default, so a fresh clone works — and a Python with
torch/transformers/datasets (`--config '{"python": ...}'`; the adapter falls
back to `repo/.venv/bin/python`, then `python3`). Saga fetches its HF
streaming datasets over the network — offline it raises instead of emitting
an empty record. There is no `.venv` in the saga repo on this box; the
operator's torch-capable venv
`/media/data/coding/OBLITERATUS/.venv/bin/python` is used. A bounded example
against the small model already cached under `.skald/saga_models/tiny-gpt2`:

```bash
python -m adapters.saga .skald/saga_models/tiny-gpt2 mmlu \
  --config '{"python": "/media/data/coding/OBLITERATUS/.venv/bin/python", "max_samples": 8}'
```

(`mmlu` and `humaneval` bounded runs are exercised by the suite's own
real-material tests in `tests/test_saga.py`.)

### openai_compat — served models (`python -m adapters.openai_compat`)

Benchmarks any OpenAI-compatible `/chat/completions` endpoint (vLLM,
llama.cpp server, …) with the same `mmlu` / `humaneval` task shapes as the
saga adapter, so served-model numbers land in the store next to weighed-in
ones. `model` is the endpoint base URL; the served model id comes from
`config['model']` (default: the server's first `/models` entry). Transport
is stdlib-only; items load from Hugging Face datasets-server over HTTP by
default, or inline via `config['mmlu_items']` / `config['humaneval_items']`
for offline or custom batteries.

```bash
python -m adapters.openai_compat http://host:8888/v1 mmlu \
  --config '{"model": "my-served-model", "max_samples": 20}'
```

Identity honesty: a served model exposes no weights, so
`model_checkpoint_sha256` is the hash of
`openai-compat::<url>::<model>` — a stable *endpoint* identity, and every
record's `protocol` carries an UNVERIFIED marker. Never silently compare
these numbers with weighed-in records. Verified live against a vLLM box
(`mmlu` accuracy 1.0 on a 2-item probe). Two lessons from that run, now in
the code: reasoning models need `max_tokens` headroom (default 64 — a 16
budget spent itself thinking and returned null content, which the client
now reads from the `reasoning`/`reasoning_content` fallback fields), and
letter extraction takes the *last* A–D match (completions echo the
question's own options first).

### atlas — weight diffs (`python -m adapters.atlas`)

Queries a weight-atlas API for the weight-space difference between two
scans — the abliteration-forensics instrument: scan stock + edited twin,
diff, and the hotspots say where the edit landed, layer and tensor.
`model` is the Skald checkpoint SHA-256 the scan is asserted to belong to;
`config['atlas_job']` / `config['with_job']` are the two scan ids. The
mapping is caller-asserted and recorded verbatim in `protocol` — the
adapter never guesses it (atlas addresses scans by job id, not by hash,
so no automatic join exists).

```bash
python -m adapters.atlas <checkpoint_sha256> diff \
  --config '{"atlas_job": "<scan-a>", "with_job": "<scan-b>"}'
```

Emits `delta_mean/max_change_pct`, `delta_n_changed_above_5pct`, and
`hotspot_rank{k}_pct_change` records (tensor names in `artifacts[]`).
Verified live: a same-weights null pair reads all-0.0 (deterministic
scanner, honest null control); stock-vs-abliterated Qwen3.8-27B reads
max 0.0127% concentrated in mid-layer `ssm_out`/`o_proj`/`down_proj` —
below any surgical-edit signature, which is itself a finding (see PIN
note below). Default `atlas_url` is the atlas box on this LAN; the real
stock-vs-abliterated Qwen3.8-Flash-Next pair is not scanned yet — that
scan is the prerequisite for the harmed-vs-freed query this adapter
exists to serve.

### jlens — layer readout (`python -m adapters.jlens`)

Reads out early-layer residual-stream representations of an ablated model as
per-`(layer, rank)` softmax-probability records, using the vendored
Neuronpedia `jlens` library under `vendor/jlens/` (pinned, unmodified).

`saga`-style but with the model argument optional:

```bash
# default target + task, both optional
python -m adapters.jlens
# explicit
python -m adapters.jlens huihui-ai/Huihui-gemma-3-270m-it-abliterated layer_readout \
  --config '{"source_layers": [4], "layers": [4], "max_seq_len": 48}'
```

Requires a torch/transformers-capable Python. The default is the operator's
`/media/data/coding/OBLITERATUS/.venv/bin/python` — machine-specific; set
`config['python']` on any other box. A remote HF model id is resolved through
`snapshot_download` into `.skald/jlens_models/<slug>/` (weights are reused
across runs, but the snapshot check contacts the hub); a local file/directory
path is used as-is and never touches the network — pass the cached directory,
e.g. `.skald/jlens_models/Huihui-gemma-3-270m-it-abliterated`, for a fully
offline run. Expect the fit to take several minutes on a CPU box.

Fitted lenses land in `.skald/jlens_lenses/<checkpoint_sha>/` and append-only
raw logit traces in `.skald/jlens_raw/<checkpoint_sha>/`; both are referenced
from each record's `artifacts` by path + sha256.

> **Constraint.** A local run is a *CPU wiring check*, not a scientific
> measurement: the protocol string is explicitly labelled
> `MINIMAL CPU FIT — WIRING CHECK, NOT A SCIENTIFIC MEASUREMENT`. A faithful
> all-layer readout over a production checkpoint needs the GPU box.

## Surface 1: HTTP/JSON API (`python -m api`)

Serves the spec-derived JSON surface. Default `127.0.0.1:8000`, backed by the
default store (override via `--store-dir` or `SKALD_STORE_DIR`).

```bash
python -m api                       # -> listening on http://127.0.0.1:8000
python -m api --port 8001 --store-dir /tmp/demo-store
```

Read-only, GET/HEAD only. Routes are generated from
[`surfaces/spec.py`](../surfaces/spec.py) as `/api/v1/<operation_id>`:

| Route | Params | Returns |
|---|---|---|
| `/api/v1/list_suites_adapters` | — | `{spec_id, spec_version, operation, suites, adapters}` |
| `/api/v1/query_results` | any filterable field + `limit` | `{…, count, records}` |
| `/api/v1/task_drilldown` | `model_checkpoint_sha256`* , `task`* , `limit` | `{…, model_checkpoint_sha256, task, count, records}` |
| `/api/v1/list_anomalies` | any filterable field + `limit` | `{…, count, anomalies, records}` |

Filterable fields: `model_checkpoint_sha256`, `adapter`, `suite`, `task`,
`metric`, `value`, `n`, `ci_low`, `ci_high`, `protocol`, `created_at`, `host`,
`script_sha256`, `runtime_sha256`, `seed`. Every filter is an equality filter
passed straight to `store.query`; `limit` bounds the number of records.
Unknown parameters and missing required ones are rejected with 400.
`runtime_sha256` is the serving-path facet (digest over the normalised runtime
manifest — interpreter, OS, endpoint-observed model list; see
`docs/plans/2026-09-23_runtime-manifest-spec.md`): filter on it, never key on
it. `null` means unknown.

```bash
curl http://127.0.0.1:8000/api/v1/list_suites_adapters
curl "http://127.0.0.1:8000/api/v1/query_results?adapter=pi50&limit=5"
curl "http://127.0.0.1:8000/api/v1/task_drilldown?model_checkpoint_sha256=<sha>&task=manifest_check"
```

`list_anomalies` implements the declared deterministic rules: any checkpoint
that appears under two or more distinct `protocol` labels is flagged, because
numbers from different protocols must never be silently compared — and any
(checkpoint, protocol) measured under two or more distinct known
`runtime_sha256` digests is flagged as `cross-runtime-checkpoint`, the same
hazard class through the serving path.

## Surface 2: HTML UI (`python -m ui`)

Same functional surface, rendered as HTML. Default `127.0.0.1:8080`; one view
per operation at `/ui/v1/<operation_id>`; `/` redirects to the browsing entry
point `list_suites_adapters`.

```bash
python -m ui                       # -> listening on http://127.0.0.1:8080
```

Open `http://127.0.0.1:8080/`. Browse suites → filter results → drill into a
checkpoint+task. Every page embeds the exact JSON envelope the API would
return for the same query (script block `#skald-surface-data`), so parity
between the surfaces is by construction, not by duplicate rendering.

## Using the store programmatically

```python
import store
from identity import hash_checkpoint

checksum = hash_checkpoint("path/to/checkpoint.pt")   # -> 64-hex sha256

record = {
    "model_checkpoint_sha256": checksum,   # required, non-empty
    "adapter": "demo",
    "suite": "demo",
    "task": "smoke",
    "metric": "answer",
    "value": 42.0,                          # must be finite
    "n": 1,
    "protocol": "some-protocol-label",       # required, non-empty
}
store.put([record])                     # append-only; returns normalized records
store.query({"adapter": "demo"})        # equality filters
store.query()                           # all records
store.Store("/tmp/my-store")            # an explicit store root
store.query(filters={"adapter": "demo"}, limit=10)
```

`SKALD_STORE_DIR` selects the default store's root; `store.normalize` and
`store.ValidationError` (from `store`) validate/sanitize single records. The
full canonical field set is `store.RECORD_FIELDS` — extra fields are rejected,
optional fields are defaulted (`created_at` = now-UTC, `host` = local
hostname, nulls for the others).

## Checkpoint identity and the join key

`identity.hash_checkpoint(path)` returns the SHA-256 hex digest of one
checkpoint file. This hash is the `model_checkpoint_sha256` recorded on every
result and is **the join key** between a model's evaluation history and its
weight-level atlas entry — there is no separate model registry. Directories
(a HF model folder, a multi-file checkpoint) are hashed by the relevant
adapters deterministically over their sorted contents (HF `.cache` metadata
excluded), so the same weights always produce the same key.

Because the hash is the identity, comparing numbers is only valid within one
`protocol` label: a checkpoint that appears under several protocols is exactly
what the anomaly view flags.

## End-to-end walkthrough

**1. Install.**

```bash
cd skald
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

export SKALD_STORE_DIR="$PWD/.skald/demo-store"   # scratch store for this demo
```

**2. Produce a record with a real adapter** (requires a torch-capable
Python for the BDH-CL eval scripts; the evaluated artifact is the
suite's committed `docs/PHASE1-MANIFEST.md`):

```bash
python -m adapters.pi50 /media/data/coding/bdh-cl/docs/PHASE1-MANIFEST.md manifest_check \
  --config '{"repo": "/media/data/coding/bdh-cl"}'
# persisted 1 pi50 records; queried back N matching (N grows with the store)
#   manifest_check:manifest_current = 0.0 (n=1, protocol='phase-1 artifact manifest consistency check ...')
```

No BDH-CL checkout handy? The fully self-contained alternative is a one-record
store write (`store.put([...])` per the section above) — that is exactly what
the adapter does internally.

**3. Serve both surfaces over the same store** (two terminals):

```bash
python -m api        # http://127.0.0.1:8000
python -m ui         # http://127.0.0.1:8080
```

**4. Read the record back from both.**

```bash
curl http://127.0.0.1:8000/api/v1/list_suites_adapters
# {"spec_id": "skald-functional-surface", "spec_version": "1", "operation": "list_suites_adapters", "suites": ["pi50"], "adapters": ["pi50"]}

curl "http://127.0.0.1:8000/api/v1/query_results?task=manifest_check&limit=1"
# {"spec_id":..., "operation":"query_results","count":1,"records":[{... "adapter":"pi50","task":"manifest_check", ...}]}
```

Open `http://127.0.0.1:8080/ui/v1/query_results?task=manifest_check` in a
browser (or `curl -s` it) — the same record, rendered as a table, with the
same embedded JSON envelope.

**5. Clean up:** `rm -rf .skald/demo-store` (or keep it — the default store is
`.skald/store` and already holds real runs from the saga/jlens/pi50/bdh_cl
suites under `.skald/jlens_raw`, `.skald/jlens_lenses`,
`.skald/saga_raw`, `.skald/saga_models`).

## Known constraints

- **No cloud LLM anywhere** in the evaluation, serving, or analysis chain —
  generation is a local transformers process; the surfaces are read-only over
  the local store.
- **jlens local runs are CPU wiring checks**, explicitly labelled as such in
  the `protocol` field; a faithful all-layer readout needs the GPU box.
- **`bdh_cl` `p5_inchain` is not executable locally** without a grown
  parent/child checkpoint pair (`config['parent']`).
- **saga needs network access** to its HF streaming datasets; offline it
  raises rather than emit 0-sample records. HumanEval is an adapter-side
  shim over saga's config entry, not a saga runner.
- Benchmark code ships vendored (`vendor/`); what remains machine-specific
  is the torch-capable interpreter (`--config '{"python": ...}'`, e.g.
  `/media/data/coding/OBLITERATUS/.venv` on this box), the evaluated model
  weights/corpora, and network access for HF datasets. A full BDH-CL
  checkout is needed only for `pi50/manifest_check`, which audits that
  checkout's own git history by design.