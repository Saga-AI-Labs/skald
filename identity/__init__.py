"""Checkpoint identity — weight-atlas interlock (plan §4.4).

Every result carries ``model_checkpoint_sha256``.  Skald inherits
weight-atlas identity: the hash *is* the join key between a model's
weight-level atlas entry and its evaluation history.  There is no
separate model registry.

Single entry point: :func:`hash_model` dispatches files to
:func:`hash_checkpoint` and directories to :func:`hash_dir`.  All three
adapters that weigh models in (bdh_cl, saga, jlens) hash through here,
so one checkpoint always yields one identity string no matter which
adapter measured it.

Format notes (quantized checkpoints — EXL3, GGUF, NVFP4/MXFP4, AWQ):
identity is computed over the **stored bytes**, not dequantized values.
Re-quantizing the same base model produces a different identity, which is
correct: the bytes under test changed, so the measurement key must change
too.  Directory hashing is deterministic (sorted relative paths, NUL
separators, chunked reads) with an ``exclude`` predicate for fetch
metadata — Hugging Face ``snapshot_download`` caches bookkeeping under
``.cache/`` that must not participate in identity, or re-downloads would
re-identify the same weights.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

__all__ = ["hash_checkpoint", "hash_dir", "hash_model"]


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


def _default_exclude(rel: Path) -> bool:
    """Exclude Hugging Face fetch metadata from directory identity."""
    return ".cache" in rel.parts


def hash_dir(
    path: str | Path,
    exclude: Callable[[Path], bool] | None = _default_exclude,
) -> str:
    """Return the deterministic SHA-256 hex digest of a checkpoint directory.

    Walks files in sorted-relative-path order, feeding each relative path
    (UTF-8, NUL-terminated) followed by its chunked contents into one
    SHA-256.  Pass ``exclude=None`` to hash every file, or a predicate
    taking the path relative to *path*.

    Raises
    ------
    FileNotFoundError
        If *path* is not an existing directory.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"hash_dir: not a directory: {path}")
    h = hashlib.sha256()
    files = sorted(
        p.relative_to(root) for p in root.rglob("*") if p.is_file()
    )
    for rel in files:
        if exclude is not None and exclude(rel):
            continue
        h.update(rel.as_posix().encode("utf-8"))
        h.update(b"\0")
        with open(root / rel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def hash_model(
    path: str | Path,
    exclude: Callable[[Path], bool] | None = _default_exclude,
) -> str:
    """Return the identity SHA-256 for a model artifact, file or directory.

    Single files hash by content; directories hash deterministically (see
    :func:`hash_dir`).  This is the only hashing entry point adapters
    should use.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)
    if p.is_file():
        return hash_checkpoint(p)
    if p.is_dir():
        return hash_dir(p, exclude=exclude)
    raise FileNotFoundError(f"hash_model: model artifact not found: {path}")
