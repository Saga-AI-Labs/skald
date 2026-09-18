"""Checkpoint identity — weight-atlas interlock (plan §4.4).

Every result carries ``model_checkpoint_sha256``.  Skald inherits
weight-atlas identity: the hash *is* the join key between a model's
weight-level atlas entry and its evaluation history.  There is no
separate model registry.

The concrete ``hash(checkpoint) -> sha256`` implementation will resolve
local checkpoint paths to their SHA-256 digest.  For now this module
documents the contract and provides a reference stub.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_checkpoint(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a checkpoint file.

    Parameters
    ----------
    path:
        Local filesystem path to the model checkpoint.

    Returns
    -------
    str
        64-character lowercase hex SHA-256 digest.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
