"""Unified result store (plan §4.2).

Persistence follows the OPEN-1 decision: SQLite acts as the query index over
immutable per-run files (one JSON file per ``put`` call), so the store stays
inspectable without a server.

Public interface:

- ``put(records)`` — persist one run of unified records (append-only).
- ``query(filters=None)`` — query records by equality on scalar fields.
- ``Store`` — a store rooted at a specific directory.

The module-level ``put``/``query`` use a default store; point it elsewhere
with the ``SKALD_STORE_DIR`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .backend import Store
from .runtime import collect_manifest, digest_manifest, runtime_digest
from .schema import RECORD_FIELDS, ValidationError, normalize

__all__ = [
    "Store",
    "RECORD_FIELDS",
    "ValidationError",
    "collect_manifest",
    "digest_manifest",
    "normalize",
    "put",
    "query",
    "runtime_digest",
]

_DEFAULT_STORE: Store | None = None


def _default_store() -> Store:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = Store(Path(os.environ.get("SKALD_STORE_DIR", ".skald/store")))
    return _DEFAULT_STORE


def put(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist one or more unified result records to the default store.

    Returns the normalized records as persisted (used for round-trip checks).
    """
    return _default_store().put(records)


def query(
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Query the default store by filter criteria; ``None`` returns all.

    Filters are equality matches on scalar record fields — at least
    ``model_checkpoint_sha256``, ``suite``, ``adapter``, and ``protocol``.
    """
    return _default_store().query(filters=filters, limit=limit)