"""CLI for the bdh_cl adapter: run an eval, persist, read back.

This is the M0 "bdh_cl execution": it runs the real BDH-CL suite script for a
target model, writes the resulting unified records to the unified result-store,
and queries them back from the same store.

Usage:
    python -m adapters.bdh_cl <model_ckpt> <task> [--config '{"routes": ...}']
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import store
from adapters.bdh_cl import BdhClAdapter, TASKS


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="path to the model checkpoint under test")
    ap.add_argument("task", choices=sorted(TASKS), help="bdh_cl eval to run")
    ap.add_argument("--config", default="{}", help="JSON config (routes, domains, ...)")
    args = ap.parse_args(argv)

    config = json.loads(args.config)
    records = BdhClAdapter().run(args.model, args.task, config)
    if not records:
        print("bdh_cl: no records produced", file=sys.stderr)
        return 2

    store.put(records)
    key = {
        "adapter": "bdh_cl",
        "task": args.task,
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    }
    back = store.query(key)
    print(f"persisted {len(records)} bdh_cl records; queried back {len(back)} matching")
    for r in back:
        print(
            f"  {r['task']}:{r['metric']} = {r['value']} "
            f"(n={r['n']}, protocol={r['protocol']!r})"
        )
    if len(back) < len(records):
        print(f"warning: read back {len(back)} of {len(records)} records", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())