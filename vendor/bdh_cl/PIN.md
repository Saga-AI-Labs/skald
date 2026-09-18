# Vendored upstream: BDH-CL evaluation subset

## Authorship and provenance

The code in this directory comes from the **BDH-CL** research project —
continual-learning research on BDH-GPU. The base model code is
© 2025 **Pathway Technology, Inc.** (see `LICENSE.md`, MIT-style, copied
verbatim from the source repo). BDH-CL is a research fork of
[**pathwaycom/bdh**](https://github.com/pathwaycom/bdh), the official
implementation of *The Dragon Hatchling: The Missing Link between the
Transformer and Models of the Brain* (Kosowski et al., arXiv:2509.26507).
Skald's authors wrote none of this code: `adapters/bdh_cl.py` and
`adapters/pi50.py` are wrappers that invoke these scripts unmodified in a
subprocess, and this directory is a byte-identical subset copy, not a fork.

- Source repo: `/media/data/coding/bdh-cl` (local checkout)
- Pinned commit: `8c28c6e99a0ffe097e792a30484a9e8785588d6e`
- License: MIT-style, © 2025 Pathway Technology, Inc. (`LICENSE.md` here is
  the verbatim copy; its copyright + permission notice must be preserved in
  all copies per its own terms)

## What was vendored (and what was not)

Only the evaluation closure Skald's adapters invoke, plus its license and
dependency manifest:

- `scripts/eval_router.py`, `scripts/domain_eval.py`,
  `scripts/p5_inchain_check.py` — the three BDH-CL eval instruments
- `scripts/pi50/phase1_manifest.py` + `docs/PHASE1-MANIFEST.md` — the Pi-50
  phase-1 manifest instrument and its committed manifest (provenance
  reference; note below on portability)
- `pipeline/` (whole package) + `bdh.py`, `bdh_linear.py`, `bdh_prime.py`
  — the import closure of the eval scripts (`pipeline.analyze._load_model`,
  `from bdh import BDH`, lazy model-variant imports in `pipeline/config.py`)
- `requirements.txt` — upstream dependency manifest (`torch>=2.3`, numpy,
  requests, pyarrow)

Deliberately excluded: training scripts and ladder shells (`train.py`,
`ladder_*.sh`), corpora (`data/`), checkpoints and run outputs (`out/`),
reports (`docs/reports/`, except the manifest above). Skald measures; it
does not train and does not ship weights or data.

## Portability notes

- The eval scripts expect to run with this directory as cwd
  (`sys.path.insert(0, ".")`, `from pipeline.analyze import _load_model`);
  the adapter enforces that.
- `scripts/pi50/phase1_manifest.py` is intrinsically bound to a BDH-CL git
  checkout (it runs `git rev-parse/ls-files/log` against its own repo to
  audit evidence freshness). It is vendored for provenance, but the
  `manifest_check` task remains checkout-bound by nature: the adapter
  requires an explicit `repo` pointing at a BDH-CL checkout and will not
  silently audit the wrong repository.
- Checkpoints and eval corpora are never vendored: callers supply model
  paths and domain data via adapter config.
