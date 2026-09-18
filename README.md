# Skald

Consolidated **local** evaluation framework for comparing model benchmarks
behind one result-store and two equivalent surfaces (API + UI).

- **No cloud LLM anywhere** in the evaluation or serving chain.
- Every result is keyed to the exact weights it measured
  (`model_checkpoint_sha256` — see below).
- Adding a benchmark family means adding an adapter, not touching the core.

Start here for a quick intro; the full how-to-run guide is
[`docs/usage.md`](docs/usage.md). The authoritative design plan is
[`docs/plans/2026-09-16_skald-draft.md`](docs/plans/2026-09-16_skald-draft.md).

## Repository layout

```
skald/
  adapters/       Suite-adapter interface and per-family implementations.
                  Each adapter exposes run(model, task, config) -> records[].
  store/          Unified result-store: put(records), query(filters).
                  SQLite index over immutable per-run files (plan §4.2).
  api/            LLM-facing HTTP/JSON surface (plan §4.3).
  ui/             Human-facing HTML result-browsing surface (plan §4.3).
  identity/       Checkpoint SHA-256 hashing — the result's join key (plan §4.4).
  surfaces/       Shared spec both surfaces generate from (one capability set).
  tests/          pytest test suite.
  docs/           Design plans and usage documentation.
```

## Component interfaces (plan §5)

| Component   | Interface                                  |
|-------------|--------------------------------------------|
| adapters/*  | `run(model, task, config) -> records[]`    |
| store       | `put(records)`, `query(filters)`, `Store(dir)` |
| api         | HTTP/JSON at `/api/v1/<operation>` (default port 8000) |
| ui          | HTML at `/ui/v1/<operation>` (default port 8080); same spec as api |
| identity    | `hash_checkpoint(path) -> sha256`          |

## Getting started

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

pytest
```

Requirements: Python >= 3.11. Results are stored under `.skald/store/` by
default; override with the `SKALD_STORE_DIR` environment variable or the
`--store-dir` flag on both surfaces.

## Quick tour

```bash
# Run one adapter end to end (pi50 manifest check over the BDH-CL suite;
# see docs/usage.md for prerequisites and the other three adapters):
python -m adapters.pi50 <path>/PHASE1-MANIFEST.md manifest_check

# Serve the two surfaces over the result store (two terminals):
python -m api        # HTTP/JSON -> http://127.0.0.1:8000
python -m ui         # HTML      -> http://127.0.0.1:8080

# Query the store from Python:
python - <<'PY'
import store
print(store.query({"adapter": "pi50"}))
PY
```

Each adapter CLI (`python -m adapters.bdh_cl`, `python -m adapters.pi50`,
`python -m adapters.saga`, `python -m adapters.jlens`) runs the real suite
machinery, persists the unified records, and reads them back. All surface
routes, adapter tasks, and `--config` keys are documented in
[`docs/usage.md`](docs/usage.md).

## Credits — borrowed code

Skald's own code is original to this project. The benchmark instruments it
drives are not — each is vendored unmodified under `vendor/` with its
license and a `PIN.md` provenance record. Full index: [`CREDITS.md`](CREDITS.md).

- **Jacobian Lens** (`vendor/jlens/`, via `adapters/jlens.py`): written by
  **Anthropic PBC** (`Copyright 2026 Anthropic PBC`), published through
  **Johnny Lin's Neuronpedia**
  ([github.com/hijohnnylin/neuronpedia](https://github.com/hijohnnylin/neuronpedia));
  the lens from "Verbalizable Representations Form a Global Workspace in
  Language Models" (Gurnee et al., 2026). Pinned at commit `4e3f3b2`,
  Apache-2.0.
- **BDH-CL evals + Pi-50 instrument** (`vendor/bdh_cl/`, via
  `adapters/bdh_cl.py`, `adapters/pi50.py`): **Pathway Technology, Inc.**
  (© 2025; research fork of `pathwaycom/bdh` — *The Dragon Hatchling*,
  Kosowski et al.). Pinned at commit `8c28c6e`, MIT-style.
- **Saga general-purpose benchmarks** (`vendor/saga_benchmarks/`, via
  `adapters/saga.py`): **Saga AI Labs** (previously unlicensed; AGPL-3.0
  `LICENSE` placed at its root during this consolidation). Pinned at commit
  `5657fce`, **AGPL-3.0** — see `CREDITS.md`: the combined work that
  executes this directory is AGPL-covered.

Everything else in this repository is Skald's own code, Apache-2.0.