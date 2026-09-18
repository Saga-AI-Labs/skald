"""Weight-atlas diff adapter (abliteration forensics).

Queries a weight-atlas API for the weight-space difference between two
scans and returns unified result records (plan §4.2):

- ``diff`` -> ``GET /api/model/{job}/delta?with={other}`` — top changed
  tensors plus summary statistics (mean/max change, most-affected type
  and layer range, count above threshold)

Interface: ``run(model, task, config) -> records[]`` (scaffold §5). Here
``model`` is the Skald checkpoint SHA-256 the caller asserts the atlas scan
belongs to; ``config["atlas_job"]`` is that scan's atlas job id and
``config["with_job"]`` the comparison scan. The asserted mapping is
recorded verbatim in every record's ``protocol`` — the adapter never
guesses it. Transport is stdlib ``urllib`` only.

Attribution: weight-atlas is our own code (same org); no third-party
borrowing here. The statistics compared (frobenius, spectral_norm, ranks,
kurtosis, sparsity) are computed by atlas's scanner, not by this wrapper.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Sequence

from adapters import RECORD_FIELDS, SuiteAdapter

TASKS = {"diff"}

DEFAULT_ATLAS_URL = "http://192.168.178.200:8000"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class AtlasError(RuntimeError):
    """A failed or malformed response from the weight-atlas API."""


class AtlasAdapter(SuiteAdapter):
    """Adapter over a weight-atlas query API (cross-scan weight diff)."""

    def run(
        self,
        model: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        config = config or {}
        checkpoint = str(model or "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint):
            raise ValueError(
                "atlas: model must be the 64-hex Skald checkpoint SHA-256 "
                "the atlas scan is asserted to belong to; "
                f"got {model!r}"
            )
        if task not in TASKS:
            raise ValueError(
                f"atlas: unsupported task {task!r}; choose from {sorted(TASKS)}"
            )
        base = str(config.get("atlas_url", DEFAULT_ATLAS_URL)).rstrip("/")
        job = config.get("atlas_job")
        with_job = config.get("with_job")
        if not job or not with_job:
            raise ValueError(
                "atlas: diff requires config['atlas_job'] (the scan asserted "
                "to be this checkpoint) and config['with_job'] (the "
                "comparison scan); Skald never guesses the mapping"
            )
        metric = str(config.get("metric", "frobenius"))
        top_n = int(config.get("top_n", 10))
        min_pct = float(config.get("min_change_pct", 0.0))
        seed = config.get("seed", None)
        body = _get(
            base,
            f"/api/model/{urllib.parse.quote(str(job), safe='')}/delta",
            {
                "with": str(with_job),
                "metric": metric,
                "n": str(top_n),
                "min_change_pct": str(min_pct),
            },
            timeout=int(config.get("timeout", 300)),
        )
        return self._records(
            checkpoint=checkpoint,
            job=str(job),
            with_job=str(with_job),
            metric_name=metric,
            body=body,
            top_n=top_n,
            seed=seed,
        )

    # --- records --------------------------------------------------------

    def _records(self, *, checkpoint: str, job: str, with_job: str,
                 metric_name: str, body: dict, top_n: int,
                 seed: Any) -> list[dict[str, Any]]:
        summary = body.get("summary", {}) or {}
        tier = body.get("tier", "unknown")
        protocol = (
            f"atlas diff via asserted mapping checkpoint {checkpoint[:12]}… "
            f"== scan {job} vs scan {with_job}: top-{top_n} tensor changes "
            f"in {metric_name} (tier {tier}). Mapping caller-asserted, "
            f"recorded verbatim — verify before comparing across pairs."
        )
        base_artifacts = [f"atlas::{job}", f"atlas-with::{with_job}"]
        # The store requires finite values: null summary fields (the API
        # returns nulls when there is nothing to average) are omitted, not
        # zero-filled — a missing record states less than a false 0.0.
        candidates = [
            ("delta_mean_change_pct", _number(summary.get("mean_change_pct"))),
            ("delta_max_change_pct", _number(summary.get("max_change_pct"))),
            ("delta_n_changed_above_5pct", _number(body.get("n_changed_above_5pct"))),
        ]
        records = [
            self._record(
                checkpoint=checkpoint, metric=name, value=value,
                protocol=protocol, seed=seed, artifacts=base_artifacts,
            )
            for name, value in candidates
            if value is not None
        ]
        for rank, row in enumerate(body.get("rows", [])[:top_n], start=1):
            # Tier-2 (statistic_diff) rows carry pct_change + tensor_name;
            # tier-1 (weight_space) rows carry rel_l2 + name_a/name_b.
            pct = _number(row.get("pct_change"))
            rel = _number(row.get("rel_l2"))
            if pct is not None:
                metric, value = f"hotspot_rank{rank}_pct_change", pct
            elif rel is not None:
                metric, value = f"hotspot_rank{rank}_rel_l2", rel
            else:
                continue
            tensor = row.get("tensor_name") or row.get("name_a")
            records.append(
                self._record(
                    checkpoint=checkpoint,
                    metric=metric,
                    value=value,
                    protocol=protocol,
                    seed=seed,
                    artifacts=base_artifacts + [f"tensor::{tensor}"],
                )
            )
        if not records:
            raise AtlasError(
                "atlas: diff returned no numeric values at all "
                f"(tier {tier}); nothing truthful to record"
            )
        return records

    def _record(self, *, checkpoint: str, metric: str, value: float | None,
                protocol: str, seed: Any, artifacts: list[str]) -> dict[str, Any]:
        record = {
            "model_checkpoint_sha256": checkpoint,
            "adapter": "atlas",
            "suite": "atlas",
            "task": "diff",
            "metric": metric,
            "value": value,
            "n": None,
            "ci_low": None,
            "ci_high": None,
            "protocol": protocol,
            "created_at": _now(),
            "host": socket.gethostname(),
            "script_sha256": _self_sha256(),
            "seed": seed,
            "artifacts": artifacts,
        }
        assert set(record) == set(RECORD_FIELDS), set(record) ^ set(RECORD_FIELDS)
        return record


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _self_sha256() -> str:
    h = hashlib.sha256()
    with open(__file__, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(base: str, path: str, params: dict[str, str], timeout: int) -> dict:
    url = base + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise AtlasError(
            f"atlas: {url} returned HTTP {exc.code} "
            f"({exc.read().decode('utf-8', 'replace')[:300]})"
        ) from exc
    except urllib.error.URLError as exc:
        raise AtlasError(f"atlas: cannot reach {url}: {exc.reason}") from exc
    if not isinstance(body, dict):
        raise AtlasError(f"atlas: {url} returned no usable object: {body!r}")
    return body


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run an atlas diff, persist, read back.

    Usage:
        python -m adapters.atlas <checkpoint_sha256> diff
            --config '{"atlas_job": "<scan-a>", "with_job": "<scan-b>"}'
    """
    import argparse
    import sys

    import store

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="64-hex checkpoint SHA-256 (asserted scan owner)")
    ap.add_argument("task", choices=sorted(TASKS), help="atlas task")
    ap.add_argument("--config", default="{}", help="JSON config")
    args = ap.parse_args(argv)

    config = json.loads(args.config)
    records = AtlasAdapter().run(args.model, args.task, config)
    if not records:
        print("atlas: no records produced", file=sys.stderr)
        return 2

    store.put(records)
    key = {
        "adapter": "atlas",
        "task": args.task,
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    }
    back = store.query(key)
    print(f"persisted {len(records)} atlas records; queried back {len(back)} matching")
    for r in back:
        print(f"  {r['task']}:{r['metric']} = {r['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
