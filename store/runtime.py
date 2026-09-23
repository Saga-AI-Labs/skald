"""Runtime-manifest derivation for the ``runtime_sha256`` facet.

Spec: ``docs/plans/2026-09-23_runtime-manifest-spec.md``. The adapter collects
via this helper — only it knows whether the serving path is local or remote —
and the store validates the resulting digest in ``normalize()``.

Discipline (proposal §2: "derived, never typed"): every fact here is observed
from the stdlib or passed in by the adapter from something it actually saw.
Unobservable parts stay ``None`` and are dropped from the canonical form, so
a manifest with no observable content digests to ``None``, never to a guess.
"""

from __future__ import annotations

import hashlib
import json
import platform
from typing import Any


def collect_manifest(
    served: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the raw runtime manifest.

    ``served`` carries endpoint-observed facts (e.g. ``{"models": [...],
    "server": ...}``) — pass ``None`` when no endpoint was involved.
    ``extra`` carries adapter-observed pinned facts (e.g. vendored-instrument
    pin commits) — ``None`` when there are none. The local section (stdlib
    ``platform`` facts; hostname deliberately excluded — it already has its
    own record field) is always collected.
    """
    manifest: dict[str, Any] = {
        "local": {
            "python": platform.python_version(),
            "os": platform.system(),
            "arch": platform.machine(),
        },
    }
    if served is not None:
        manifest["served"] = dict(served)
    if extra is not None:
        manifest["extra"] = dict(extra)
    return manifest


def canonical_json(manifest: dict[str, Any]) -> str | None:
    """Canonical JSON for *manifest*, or ``None`` when it has no content.

    Drops ``None``-valued keys recursively, sorts order-insignificant lists
    (``served.models``), and serializes with sorted keys and no whitespace.
    """
    cleaned = _drop_nones(manifest)
    if not cleaned:
        return None
    served = cleaned.get("served")
    if isinstance(served, dict) and isinstance(served.get("models"), list):
        served = {**served, "models": sorted(served["models"])}
        cleaned = {**cleaned, "served": served}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def digest_manifest(manifest: dict[str, Any]) -> str | None:
    """SHA-256 hex over the canonical manifest JSON, or ``None`` if empty."""
    canonical = canonical_json(manifest)
    if canonical is None:
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def runtime_digest(
    served: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> str | None:
    """One-call manifest collection + digest for adapter record builders."""
    return digest_manifest(collect_manifest(served=served, extra=extra))


def _drop_nones(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: value for key, value in
            ((k, _drop_nones(v)) for k, v in node.items())
            if value is not None
        }
    if isinstance(node, list):
        return [_drop_nones(v) for v in node]
    return node
