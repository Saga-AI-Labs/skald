"""Import run files from another Skald store (stdlib only).

A store is just immutable per-run JSON files plus a SQLite index, so moving
records between boxes is a file copy plus this script: it reads each given
run file, normalizes its records, skips records already present (exact
match, so re-running the import is safe), and ``put()``s the new ones as
one run.

Usage:
    python scripts/import_run_files.py [--store-dir DIR] RUN.json [RUN.json ...]

Typical flow (records measured on box B, ledger on box A):
    # on B: tar -czf /tmp/skald-runs.tar.gz -C <repo> .skald/store/runs
    # move the tarball to A, unpack, then on A:
    python scripts/import_run_files.py /tmp/from-b/runs/RUN-*.json

Caveat: records' ``artifacts[]`` are machine-local paths. The records stay
valid, but raw files (transcripts, traces) must be copied separately if
you want them resolvable on this box too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store
from store.schema import normalize


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_files", nargs="+", help="run JSON files to import")
    ap.add_argument(
        "--store-dir",
        default=None,
        help="target store (default: $SKALD_STORE_DIR or .skald/store)",
    )
    args = ap.parse_args(argv)

    target = store.Store(
        args.store_dir or os.environ.get("SKALD_STORE_DIR", ".skald/store")
    )
    seen = {_canonical(r) for r in target.query()}
    imported_files = 0
    imported_records = 0
    skipped_records = 0
    for name in args.run_files:
        try:
            payload = json.loads(Path(name).read_text())
        except (OSError, ValueError) as exc:
            print(f"skip {name}: unreadable ({exc})", file=sys.stderr)
            continue
        records = payload.get("records", []) if isinstance(payload, dict) else []
        fresh = []
        for record in records:
            try:
                norm = normalize(record)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"skip record in {name}: invalid ({exc})", file=sys.stderr)
                continue
            if _canonical(norm) in seen:
                skipped_records += 1
                continue
            seen.add(_canonical(norm))
            fresh.append(norm)
        if not fresh:
            continue
        target.put(fresh)
        imported_files += 1
        imported_records += len(fresh)
    print(
        f"imported {imported_records} records from {imported_files} files; "
        f"skipped {skipped_records} already-present records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
