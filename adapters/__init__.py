"""Suite adapters — one per benchmark family.

Each adapter exposes the common ``run`` interface (plan §5):

    run(model, task, config) -> records[]

A *record* is a dict conforming to the unified result-store schema (plan §4.2).
Concrete adapter implementations go in submodules (e.g. ``adapters.bdh_cl``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# Canonical record fields from plan §4.2.  The actual Pydantic / dataclass
# schema lives in ``store``; adapters emit plain dicts that satisfy it.
RECORD_FIELDS = frozenset(
    {
        "model_checkpoint_sha256",
        "adapter",
        "suite",
        "task",
        "metric",
        "value",
        "n",
        "ci_low",
        "ci_high",
        "protocol",
        "created_at",
        "host",
        "script_sha256",
        "seed",
        "artifacts",
    }
)


class SuiteAdapter(ABC):
    """Base class for all benchmark-family adapters."""

    @abstractmethod
    def run(
        self,
        model: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a benchmark task and return unified result records.

        Parameters
        ----------
        model:
            Model identifier or checkpoint path.
        task:
            Task identifier within the adapter's suite.
        config:
            Adapter-specific configuration overrides.

        Returns
        -------
        list[dict]
            One dict per result metric, conforming to the unified record shape.
        """
        ...
