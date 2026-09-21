"""Background benchmark jobs for the run_benchmark surface operation.

A job records an adapter run (adapter/task/model/config) from submission to
completion, executes it in a worker thread, persists resulting records to
the same store the surfaces read, and keeps its own metadata as one JSON
file per job under ``<store-root>/jobs/``. Listing is a directory scan;
there is no daemon and no queue server — this is a local tool, and a
restarted server marks interrupted jobs ``orphaned`` rather than
pretending they continue.

Trust: submitting a run is equivalent to running the adapter CLI locally
(same code, same local interpreters). The surfaces are local-only; do not
expose them to a network you would not run the CLI on.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from store.schema import utcnow

_STATUSES = ("queued", "running", "done", "failed", "orphaned")


def _adapter_registry() -> dict[str, dict[str, Any]]:
    """Map adapter names to their class + task set (imported lazily)."""
    from adapters import atlas as m_atlas
    from adapters import bdh_cl as m_bdh_cl
    from adapters import jlens as m_jlens
    from adapters import openai_compat as m_openai_compat
    from adapters import pi50 as m_pi50
    from adapters import saga as m_saga

    def entry(module, cls_name: str) -> dict[str, Any]:
        return {"class": getattr(module, cls_name), "tasks": set(module.TASKS)}

    return {
        "atlas": entry(m_atlas, "AtlasAdapter"),
        "bdh_cl": entry(m_bdh_cl, "BdhClAdapter"),
        "jlens": entry(m_jlens, "JLensAdapter"),
        "openai_compat": entry(m_openai_compat, "OpenAICompatAdapter"),
        "pi50": entry(m_pi50, "Pi50Adapter"),
        "saga": entry(m_saga, "SagaAdapter"),
    }


class JobError(ValueError):
    """A rejected run submission (bad adapter/task/model/config)."""


class JobRegistry:
    """Create, execute, and inspect benchmark jobs under one directory."""

    def __init__(self, jobs_dir: str | os.PathLike,
                 adapter_factory: Callable[[], dict[str, Any]] | None = None,
                 put: Callable[[list[dict]], list[dict]] | None = None) -> None:
        self.jobs_dir = Path(jobs_dir)
        self._factory = adapter_factory or _adapter_registry
        self._put = put
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # --- submission -----------------------------------------------------

    def submit(self, adapter: str, task: str, model: str,
               config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate and persist a job, start its worker, return the record."""
        registry = self._factory()
        if adapter not in registry:
            raise JobError(
                f"unknown adapter {adapter!r}; choose from {sorted(registry)}"
            )
        entry = registry[adapter]
        tasks = set(entry["tasks"])
        if task not in tasks:
            raise JobError(
                f"adapter {adapter!r} has no task {task!r}; "
                f"choose from {sorted(tasks)}"
            )
        if not str(model or "").strip():
            raise JobError("model must be a non-empty string")
        job = {
            "job_id": f"JOB-{uuid.uuid4().hex[:12]}",
            "adapter": adapter,
            "task": task,
            "model": str(model),
            "config": dict(config or {}),
            "status": "queued",
            "created_at": utcnow(),
            "started_at": None,
            "finished_at": None,
            "record_count": 0,
            "checkpoint": None,
            "error": None,
        }
        self._write(job)
        thread = threading.Thread(
            target=self._execute, args=(job["job_id"],), daemon=True,
            name=f"skald-job-{job['job_id']}",
        )
        with self._lock:
            self._threads[job["job_id"]] = thread
        thread.start()
        return self.get(job["job_id"])

    # --- inspection -----------------------------------------------------

    def get(self, job_id: str) -> dict[str, Any]:
        """Return the job record, marking stale live states orphaned."""
        path = self.jobs_dir / f"{job_id}.json"
        if not path.is_file():
            raise KeyError(f"unknown job {job_id!r}")
        job = json.loads(path.read_text())
        if job.get("status") in ("queued", "running"):
            with self._lock:
                thread = self._threads.get(job_id)
            if thread is None or not thread.is_alive():
                job["status"] = "orphaned"
                job["finished_at"] = job.get("finished_at") or utcnow()
                if not job.get("error"):
                    job["error"] = (
                        "worker is gone (server restart or crash); "
                        "resubmit to retry"
                    )
                self._write(job)
        return job

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest-first job records (by creation time)."""
        if not self.jobs_dir.is_dir():
            return []
        jobs = []
        for path in self.jobs_dir.glob("JOB-*.json"):
            try:
                jobs.append(self.get(path.stem))
            except (ValueError, OSError):
                continue
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return jobs[: max(0, limit)]

    # --- execution ------------------------------------------------------

    def _execute(self, job_id: str) -> None:
        path = self.jobs_dir / f"{job_id}.json"
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        job["status"] = "running"
        job["started_at"] = utcnow()
        self._write(job)
        try:
            adapter = self._factory()[job["adapter"]]["class"]()
            records = adapter.run(job["model"], job["task"], job["config"])
            put = self._put
            if put is None:  # pragma: no cover - wired by both surfaces
                raise JobError("job registry has no store sink configured")
            stored = put(records)
            job["status"] = "done"
            job["record_count"] = len(stored)
            if stored:
                job["checkpoint"] = stored[0].get("model_checkpoint_sha256")
        except Exception as exc:  # noqa: BLE001 - failure is a job outcome
            job["status"] = "failed"
            job["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        job["finished_at"] = utcnow()
        self._write(job)

    # --- persistence ----------------------------------------------------

    def _write(self, job: dict[str, Any]) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.jobs_dir / f"{job['job_id']}.json.tmp"
        tmp.write_text(json.dumps(job, indent=2) + "\n")
        os.replace(tmp, self.jobs_dir / f"{job['job_id']}.json")


_registries: dict[str, JobRegistry] = {}


def registry_for_store(store, jobs_subdir: str = "jobs") -> JobRegistry:
    """One process-wide job registry per store root (threads live here)."""
    key = str(getattr(store, "root", ".skald/store"))
    registry = _registries.get(key)
    if registry is None:
        registry = JobRegistry(Path(key) / jobs_subdir, put=store.put)
        _registries[key] = registry
    return registry


def adapter_tasks() -> dict[str, list[str]]:
    """Adapter names with their task lists (for run forms and validation)."""
    return {name: sorted(entry["tasks"])
            for name, entry in _adapter_registry().items()}
