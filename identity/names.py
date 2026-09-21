"""Human names for checkpoint identities (models.yaml directory).

Every record is keyed by hash; humans cannot read hashes. This module maps
known identities to names, with an explicit unnamed fallback — a missing
entry renders as a short-hash plus "(unnamed)", never as a silently
wrong name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ModelName", "display_name", "load_directory", "resolve"]


@dataclass(frozen=True)
class ModelName:
    sha256: str
    name: str
    description: str
    kind: str  # "weights" | "endpoint"


def _directory_path() -> Path:
    return Path(__file__).resolve().parent.parent / "models.yaml"


def load_directory(path: str | Path | None = None) -> dict[str, ModelName]:
    """Load the name directory; returns {} when the file is absent."""
    src = Path(path) if path else _directory_path()
    if not src.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_minimal_yaml(src.read_text())
    raw = yaml.safe_load(src.read_text()) or {}
    out: dict[str, ModelName] = {}
    for entry in raw.get("models", []) or []:
        try:
            out[str(entry["sha256"]).strip()] = ModelName(
                sha256=str(entry["sha256"]).strip(),
                name=str(entry["name"]),
                description=str(entry.get("description", "")),
                kind=str(entry.get("kind", "weights")),
            )
        except (KeyError, TypeError, AttributeError):
            continue
    return out


def _parse_minimal_yaml(text: str) -> dict[str, ModelName]:
    """Fallback parser for the models.yaml subset (no PyYAML needed)."""
    out: dict[str, ModelName] = {}
    current: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- sha256:"):
            if current.get("sha256"):
                out[current["sha256"]] = ModelName(
                    sha256=current["sha256"],
                    name=current.get("name", current["sha256"][:12]),
                    description=current.get("description", ""),
                    kind=current.get("kind", "weights"),
                )
            current = {"sha256": stripped.split(":", 1)[1].strip()}
        elif current and ":" in stripped and not stripped.startswith("#"):
            key, _, value = stripped.partition(":")
            current[key.strip()] = value.strip()
    if current.get("sha256"):
        out[current["sha256"]] = ModelName(
            sha256=current["sha256"],
            name=current.get("name", current["sha256"][:12]),
            description=current.get("description", ""),
            kind=current.get("kind", "weights"),
        )
    return out


def resolve(sha256: str, directory: dict[str, ModelName] | None = None) -> ModelName:
    """Resolve a checkpoint hash to its directory entry or an unnamed stub."""
    directory = directory if directory is not None else load_directory()
    entry = directory.get(str(sha256).strip())
    if entry is not None:
        return entry
    short = str(sha256).strip()[:12]
    return ModelName(
        sha256=str(sha256).strip(),
        name=f"{short}… (unnamed)",
        description="No models.yaml entry — add one via pull request.",
        kind="weights",
    )


def display_name(sha256: str, directory: dict[str, ModelName] | None = None) -> str:
    """The short human label for a checkpoint hash (tables, dropdowns)."""
    entry = resolve(sha256, directory)
    if entry.name.endswith("(unnamed)"):
        return entry.name
    suffix = " [endpoint, unverified]" if entry.kind == "endpoint" else ""
    return f"{entry.name}{suffix}"
