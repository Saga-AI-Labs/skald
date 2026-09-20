"""Hash a Hugging Face snapshot directory (stdlib only, no install needed).

Reproduces Skald's ``identity.hash_dir`` semantics exactly: sorted relative
paths (UTF-8, NUL-terminated) interleaved with chunked file contents, one
SHA-256 over all of it. Run it where the weights live and paste the output
back so hotspot records can be filed under the true checkpoint identity.

Usage:
    python3 scripts/hash_snapshot_dir.py <snapshot-dir> [<snapshot-dir> ...]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def hash_snapshot_dir(root: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"hash_snapshot_dir: no files under {root}")
    for rel in files:
        h.update(rel.as_posix().encode("utf-8"))
        h.update(b"\0")
        with open(root / rel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) < 2 or any(a in ("-h", "--help") for a in argv):
        print(__doc__.strip().splitlines()[-2].strip())
        return 2 if len(argv) < 2 else 0
    for arg in argv[1:]:
        root = Path(arg)
        if not root.is_dir():
            print(f"not a directory: {arg}", file=sys.stderr)
            return 1
        print(f"{root.name}  {hash_snapshot_dir(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
