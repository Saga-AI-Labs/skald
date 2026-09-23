# Runtime-manifest spec (proposal §2 implementation spec)

Answers proposal open Q1 (column beside `script_sha256`) and Q2 (adapter
collects via shared helper, store validates). Written before §2 lands, per
the §3.6/§3.1-first sequencing.

## 1. What the digest covers

`runtime_sha256 = sha256(canonical_json(manifest))`, where the manifest is:

```json
{
  "local":  {"python": "3.12.3", "os": "Linux", "arch": "x86_64"},
  "served": {"models": ["stub-model"], "server": "uvicorn"},
  "extra":  {"vendor_pin": "abc123..."}
}
```

- `local` — stdlib `platform` facts only. Always computable, never guessed.
- `served` — present only when the adapter observed an endpoint (`/models`
  id list, `Server` response header). `null` for local-weight adapters.
- `extra` — adapter-supplied pinned facts only (vendored-instrument pin
  commits). No free-form strings.

Deliberately excluded: hostname (already a separate field — including it
would make every host a distinct runtime), timestamps, PIDs, absolute paths,
cwd, env vars, full package SBOM (volatile across pip upgrades; the facet
targets serving-path identity, not reproducibility auditing). The GLM-recipe
`root: /model` vs `/home/...` distinction (§2.1) is captured structurally:
container vs checkout shows up as `served` present/absent and in the vendor
pin, not as a raw path.

## 2. Canonicalization

1. Drop `None`-valued keys recursively (`served: null` and `extra: null`
   vanish; a manifest that is empty after dropping hashes nothing → digest
   is `None`, never typed).
2. Sort the `served.models` list (order-insignificant).
3. `json.dumps(manifest, sort_keys=True, separators=(",", ":"),
   ensure_ascii=True)` → UTF-8 → SHA-256 hex.

Same manifest → same digest across hosts, key orderings, and runs.
`None` means "unknown", never "default".

## 3. Ownership (Q2)

The adapter collects via the shared helper (`store/runtime.py`:
`collect_manifest(served=None, extra=None)`, `digest_manifest(manifest)`),
because only the adapter knows whether the serving path is local or remote —
the store at `put()` time sees only its own host and would mis-stamp
imported or server-backed records. The store owns validation
(`normalize()` accepts `None` or 64-hex lowercase only — an adapter that
stuffs prose into the digest fails loudly) and index migration. No guessing
anywhere: unobservable parts stay `None`.

## 4. Schema and migration

- `RECORD_FIELDS` gains `runtime_sha256` beside `script_sha256`;
  `FILTERABLE` gains it too (a facet you filter on, not a join key — Q1).
  Not in `REQUIRED_TEXT`: old records stay valid, no back-fill.
- `adapters/__init__.py` `RECORD_FIELDS` frozenset gains it in lockstep
  (every adapter asserts set-exactness against that copy).
- SQLite: `record_index` gains nullable `TEXT` + index. Existing stores
  migrate via guarded `ALTER TABLE` in `initialize()` (column presence
  checked through `PRAGMA table_info`); the JSON run files are untouched —
  new records simply carry the new key.
- `SPEC_VERSION` "1" → "2" (the `query_results` filter set grows).

## 5. Anomaly rule extension

`flag_anomalies()` keeps `cross-protocol-checkpoint` and adds reason
`cross-runtime-checkpoint`: one entry per (checkpoint, protocol) with more
than one distinct **non-null** runtime digest — entry keys
`model_checkpoint_sha256`, `protocol`, `runtimes` (sorted), `runtime_count`.
All-`None` groups are unknown, not divergent, and never flagged. Same view,
same determinism (sorted checkpoints, sorted digests).

## 6. Adapter wiring

- `openai_compat` (all tasks incl. `determinism`): local + served facts.
- `bdh_cl`, `pi50`, `saga`, `jlens`, `atlas`: local-only digest via helper.
- `null_model`: `None` — the floor is defined runtime-independent; stamping
  the local interpreter would fragment the fixed-identity grouping.
- Raw manifests are not stored (one column, per §2 cost box); the digest is
  reproducible from the helper + adapter inputs, which the protocol text
  already describes.
