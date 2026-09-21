"""Tests for ``jobs`` — background benchmark runs behind run_benchmark.

Covers: submission validation, the queued->running->done lifecycle with a
stub adapter, failure capture, unknown-job handling, orphan marking after
a lost worker, newest-first listing, and API/UI parity of the two run
operations (same job semantics on both surfaces).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from jobs import JobError, JobRegistry, adapter_tasks, registry_for_store


class _StubAdapter:
    TASKS = {"quick", "boom"}

    def run(self, model, task, config=None):
        if task == "boom":
            raise RuntimeError("stub went bang")
        time.sleep(0.05)
        return [{
            "model_checkpoint_sha256": "c" * 64,
            "adapter": "stub",
            "suite": "stub",
            "task": task,
            "metric": "m",
            "value": 1.0,
            "n": 1,
            "ci_low": None,
            "ci_high": None,
            "protocol": "stub protocol",
            "created_at": "2026-01-01T00:00:00Z",
            "host": "testbox",
            "script_sha256": None,
            "seed": None,
            "artifacts": [],
        }]


def _registry(tmp_path, **over):
    import store as store_mod

    store = store_mod.Store(tmp_path / "store")
    store.initialize()
    registry = JobRegistry(
        tmp_path / "jobs",
        adapter_factory=lambda: {"stub": {
            "class": _StubAdapter, "tasks": set(_StubAdapter.TASKS)}},
        put=over.get("put", store.put),
    )
    return registry, store


def _wait_done(registry, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = registry.get(job_id)
        if job["status"] in ("done", "failed", "orphaned"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never settled")


def test_adapter_tasks_lists_every_family():
    tasks = adapter_tasks()
    assert set(tasks) == {"atlas", "bdh_cl", "jlens", "openai_compat",
                          "pi50", "saga"}
    assert tasks["saga"] == ["humaneval", "mmlu"]


def test_submit_validates_adapter_task_model(tmp_path):
    registry, _ = _registry(tmp_path)
    with pytest.raises(JobError, match="unknown adapter"):
        registry.submit("nope", "quick", "m")
    with pytest.raises(JobError, match="no task"):
        registry.submit("stub", "nope", "m")
    with pytest.raises(JobError, match="non-empty"):
        registry.submit("stub", "quick", "  ")


def test_happy_path_lifecycle(tmp_path):
    registry, store = _registry(tmp_path)
    job = registry.submit("stub", "quick", "model-x", {"k": "v"})
    assert job["status"] in ("queued", "running")
    assert job["job_id"].startswith("JOB-")
    done = _wait_done(registry, job["job_id"])
    assert done["status"] == "done"
    assert done["record_count"] == 1
    assert done["checkpoint"] == "c" * 64
    assert done["started_at"] and done["finished_at"]
    assert len(store.query({"adapter": "stub"})) == 1


def test_failure_is_captured_not_raised(tmp_path):
    registry, _ = _registry(tmp_path)
    job = registry.submit("stub", "boom", "model-x")
    done = _wait_done(registry, job["job_id"])
    assert done["status"] == "failed"
    assert "stub went bang" in done["error"]


def test_unknown_job_raises_key_error(tmp_path):
    registry, _ = _registry(tmp_path)
    with pytest.raises(KeyError):
        registry.get("JOB-deadbeef")


def test_dead_worker_marks_orphaned(tmp_path):
    registry, _ = _registry(tmp_path)
    path = registry.jobs_dir
    path.mkdir(parents=True, exist_ok=True)
    (path / "JOB-orphan.json").write_text(json.dumps({
        "job_id": "JOB-orphan", "adapter": "stub", "task": "quick",
        "model": "m", "config": {}, "status": "running",
        "created_at": "2026-01-01T00:00:00Z", "started_at": None,
        "finished_at": None, "record_count": 0, "checkpoint": None,
        "error": None,
    }))
    job = registry.get("JOB-orphan")
    assert job["status"] == "orphaned"
    assert "resubmit" in job["error"]


def test_list_is_newest_first(tmp_path):
    registry, _ = _registry(tmp_path)
    first = registry.submit("stub", "quick", "m")
    # ensure distinct timestamps without sleeping a whole second
    time.sleep(1.05)
    second = registry.submit("stub", "quick", "m")
    ids = [j["job_id"] for j in registry.list()]
    assert ids.index(second["job_id"]) < ids.index(first["job_id"])
    _wait_done(registry, first["job_id"])
    _wait_done(registry, second["job_id"])


def test_registry_for_store_is_shared_per_root(tmp_path):
    import store as store_mod

    store = store_mod.Store(tmp_path / "store")
    assert registry_for_store(store) is registry_for_store(store)


# --- surface parity: API POST and UI POST submit the same way --------------


def _stubbed_surfaces(monkeypatch, tmp_path):
    import api.app as api_app
    import ui.app as ui_app

    registry, store = _registry(tmp_path)
    monkeypatch.setattr(api_app, "registry_for_store", lambda s: registry)
    monkeypatch.setattr(ui_app, "registry_for_store", lambda s: registry)
    return registry, store


def test_api_post_submits_and_status_tracks(tmp_path, monkeypatch):
    from api.app import dispatch

    registry, store = _stubbed_surfaces(monkeypatch, tmp_path)
    status, body = dispatch(
        "POST", "/api/v1/run_benchmark", {},
        store, body=json.dumps(
            {"adapter": "stub", "task": "quick", "model": "m",
             "config": {"k": "v"}}).encode(),
    )
    assert status == 200, body
    job_id = body["job_id"][0]
    done = _wait_done(registry, job_id)
    assert done["status"] == "done"

    status, body = dispatch(
        "GET", "/api/v1/job_status", {"job_id": [job_id]}, store)
    assert status == 200, body
    assert body["operation"] == "job_status"
    assert body["status"] == ["done"]
    assert body["record_count"] == [1]
    assert body["checkpoint"] == ["c" * 64]


def test_api_post_rejects_bad_input(tmp_path, monkeypatch):
    from api.app import dispatch

    _, store = _stubbed_surfaces(monkeypatch, tmp_path)
    status, body = dispatch(
        "POST", "/api/v1/run_benchmark", {}, store, body=b"not json")
    assert status == 400
    status, body = dispatch(
        "POST", "/api/v1/run_benchmark", {}, store,
        body=json.dumps({"adapter": "stub"}).encode())
    assert status == 400  # missing task/model
    status, body = dispatch(
        "POST", "/api/v1/query_results", {}, store, body=b"{}")
    assert status == 405  # only run_benchmark accepts POST
    status, body = dispatch(
        "GET", "/api/v1/job_status", {"job_id": ["JOB-nope"]}, store)
    assert status == 404


def test_ui_post_submits_and_renders_status_page(tmp_path, monkeypatch):
    from ui.app import render_page

    registry, store = _stubbed_surfaces(monkeypatch, tmp_path)
    from urllib.parse import urlencode

    body = urlencode({"adapter": "stub", "task": "quick", "model": "m",
                      "config": '{"k": "v"}'}).encode()
    status, page = render_page("POST", "/ui/v1/run_benchmark", {}, store,
                               body=body)
    assert status == 200, page[:300]
    assert "Benchmark job JOB-" in page
    job_id = registry.list()[0]["job_id"]
    assert job_id in page
    _wait_done(registry, job_id)

    status, page = render_page(
        "GET", "/ui/v1/job_status", {"job_id": [job_id]}, store)
    assert status == 200
    assert "done" in page
    assert "View the persisted records" in page


def test_ui_run_form_lists_adapters_and_tasks(tmp_path, monkeypatch):
    from ui.app import render_page

    _, store = _stubbed_surfaces(monkeypatch, tmp_path)
    status, page = render_page("GET", "/ui/v1/run_benchmark", {}, store)
    assert status == 200
    assert "Run benchmark" in page
    assert '<select name="adapter"' in page
    assert "textarea" in page  # config as JSON, not a one-liner


def test_ui_post_rejects_unknown_adapter(tmp_path, monkeypatch):
    from ui.app import render_page
    from urllib.parse import urlencode

    _, store = _stubbed_surfaces(monkeypatch, tmp_path)
    body = urlencode({"adapter": "nope", "task": "t", "model": "m"}).encode()
    status, page = render_page("POST", "/ui/v1/run_benchmark", {}, store,
                               body=body)
    assert status == 400
    assert "unknown adapter" in page
