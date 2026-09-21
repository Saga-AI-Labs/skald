"""Spec-driven HTTP/JSON API surface (plan §4.3, §5 api row).

The endpoint set is not hand-maintained: every route below is generated from
``surfaces.spec.OPERATIONS`` at import time, so a capability added to the
shared spec appears here automatically (and the parity test locks this in).
Query operations are read-only over the unified store; ``run_benchmark``
launches a background benchmark job (equivalent to the adapter CLI) whose
records land in the same store — no cloud LLM anywhere in the chain.

URL scheme: one GET endpoint per operation at ``/api/v1/<operation.id>``
(e.g. ``/api/v1/query_results``), except ``run_benchmark`` which is POST
with a JSON body ``{"adapter": ..., "task": ..., "model": ...,
"config": {...}}``.  Request parameters are the operation's declared
request params carried as query-string pairs; every filterable
parameter is an equality filter passed straight to ``store.query``.

Response envelope for record-carrying operations (spec ``RecordsResponse``):

.. code-block:: json

    {
      "spec_id": "skald-functional-surface",
      "spec_version": "1",
      "operation": "query_results",
      "<declared metadata keys>": ...,
      "records": [ {RECORD_FIELDS...}, ... ]
    }

Listings (spec ``ValuesResponse``) carry their declared value keys in place of
``records``.

The dispatch logic in :func:`dispatch` and :func:`execute_operation` is pure
(no socket), so the whole surface is unit-testable without a live server; the
socket layer is the ``BaseHTTPRequestHandler`` built by :func:`make_handler`.
"""

from __future__ import annotations

import json
import logging
import math
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from store.schema import FILTERABLE
from jobs import JobError, registry_for_store
from surfaces.spec import (
    OPERATIONS,
    SPEC_ID,
    SPEC_VERSION,
    RecordsResponse,
    ValuesResponse,
    flag_anomalies,
    get_operation,
)

API_VERSION = "v1"
_PATH_PREFIX = f"/api/{API_VERSION}"

logger = logging.getLogger("skald.api")


class ApiError(Exception):
    """A request-level failure mapped to an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def endpoint_path(operation_id: str) -> str:
    """The canonical URL path for one spec operation."""
    return f"{_PATH_PREFIX}/{operation_id}"


def operation_routes() -> dict[str, str]:
    """Map every spec operation id to its endpoint path (spec-derived)."""
    return {op.id: endpoint_path(op.id) for op in OPERATIONS}


_PATH_TO_OPERATION = {path: op_id for op_id, path in operation_routes().items()}


def _coerce(kind: str, name: str, raw: str) -> Any:
    """Coerce a query-string value to the declared parameter kind."""
    try:
        if kind == "int":
            return int(raw)
        if kind == "number":
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
            return value
        if kind in ("string", "scalar"):
            return raw
    except (TypeError, ValueError):
        raise ApiError(
            400, f"parameter {name!r} must be a valid {kind}; got {raw!r}"
        ) from None
    raise ApiError(400, f"parameter {name!r} has unknown kind {kind!r}")


def parse_request(operation, query: dict[str, list[str]]) -> dict[str, Any]:
    """Coerce and validate query params against the operation declaration.

    Unknown query parameters and missing required parameters are rejected
    with a 400, so a filter typo is surfaced instead of silently ignored.
    """
    declared = {p.name: p for p in operation.request}
    unknown = sorted(set(query) - set(declared))
    if unknown:
        raise ApiError(
            400,
            f"unknown parameter(s) for {operation.id}: {unknown}; "
            f"declared: {sorted(declared)}",
        )

    missing = sorted(p.name for p in operation.request if p.required and p.name not in query)
    if missing:
        raise ApiError(400, f"missing required parameter(s): {missing}")

    values: dict[str, Any] = {}
    for name, param in declared.items():
        if name not in query:
            continue
        raw_values = query[name]
        if len(raw_values) != 1:
            raise ApiError(
                400, f"parameter {name!r} must be given exactly once"
            )
        values[name] = _coerce(param.kind, name, raw_values[0])
    return values


def _filters_and_limit(operation, values: dict[str, Any]) -> tuple[dict, int | None]:
    filters = {
        p.name: values[p.name]
        for p in operation.request
        if p.filterable and p.name in values
    }
    limit = None
    if "limit" in values:
        limit = values["limit"]
        if limit < 0:
            raise ApiError(400, "limit must be a non-negative integer")
    return filters, limit


def _list_suites_adapters(operation, values, store) -> tuple[dict[str, Any], list | None]:
    records = store.query()
    return {
        "suites": sorted({r["suite"] for r in records}),
        "adapters": sorted({r["adapter"] for r in records}),
    }, None


def _query_records(operation, values, store) -> tuple[dict[str, Any], list]:
    filters, limit = _filters_and_limit(operation, values)
    records = store.query(filters=filters, limit=limit)
    return {"count": len(records)}, records


def _task_drilldown(operation, values, store) -> tuple[dict[str, Any], list]:
    filters, limit = _filters_and_limit(operation, values)
    records = store.query(filters=filters, limit=limit)
    return {
        "model_checkpoint_sha256": values["model_checkpoint_sha256"],
        "task": values["task"],
        "count": len(records),
    }, records


def _list_anomalies(operation, values, store) -> tuple[dict[str, Any], list]:
    filters, limit = _filters_and_limit(operation, values)
    scoped = store.query(filters=filters, limit=limit)
    anomalies = flag_anomalies(scoped)
    flagged = {a["model_checkpoint_sha256"] for a in anomalies}
    records = [r for r in scoped if r["model_checkpoint_sha256"] in flagged]
    return {"count": len(records), "anomalies": anomalies}, records


_HANDLERS: dict[str, Callable[..., tuple[dict[str, Any], list | None]]] = {
    "list_suites_adapters": _list_suites_adapters,
    "query_results": _query_records,
    "task_drilldown": _task_drilldown,
    "list_anomalies": _list_anomalies,
    "run_benchmark": None,  # wired below (needs the job registry)
    "job_status": None,
}


def _registry_for(store):
    return registry_for_store(store)


def _run_benchmark(operation, values, store) -> tuple[dict[str, Any], None]:
    config_raw = values.get("config")
    if config_raw is None:
        config = {}
    else:
        try:
            config = json.loads(config_raw)
        except ValueError as exc:
            raise ApiError(400, f"parameter 'config' must be a JSON object: {exc}") from exc
        if not isinstance(config, dict):
            raise ApiError(400, "parameter 'config' must be a JSON object")
    try:
        job = _registry_for(store).submit(
            values["adapter"], values["task"], values["model"], config
        )
    except JobError as exc:
        raise ApiError(400, str(exc)) from exc
    return {"job_id": [job["job_id"]], "status": [job["status"]]}, None


def _job_status(operation, values, store) -> tuple[dict[str, Any], None]:
    try:
        job = _registry_for(store).get(values["job_id"])
    except KeyError as exc:
        raise ApiError(404, f"unknown job {values['job_id']!r}") from exc
    single = lambda v: [] if v is None else [v]  # noqa: E731
    return {
        "job_id": [job["job_id"]],
        "status": [job["status"]],
        "adapter": [job["adapter"]],
        "task": [job["task"]],
        "model": [job["model"]],
        "record_count": [job["record_count"]],
        "checkpoint": single(job["checkpoint"]),
        "error": single(job["error"]),
    }, None


_HANDLERS["run_benchmark"] = _run_benchmark
_HANDLERS["job_status"] = _job_status


def _registry_for(store):
    return registry_for_store(store)


def _envelope(operation, metadata: dict[str, Any], records: list | None) -> dict:
    body: dict[str, Any] = {
        "spec_id": SPEC_ID,
        "spec_version": SPEC_VERSION,
        "operation": operation.id,
    }
    response = operation.response
    if isinstance(response, RecordsResponse):
        for key in response.metadata_keys:
            body[key] = metadata[key]
        body["records"] = records
    elif isinstance(response, ValuesResponse):
        for key in response.value_keys:
            body[key] = metadata[key]
    else:  # pragma: no cover - spec validation forbids other response kinds
        raise ApiError(500, f"unsupported response kind for {operation.id}")
    return body


def execute_operation(
    operation_id: str,
    query: dict[str, list[str]],
    store,
) -> dict[str, Any]:
    """Run one spec operation against *store* and return the response body."""
    operation = get_operation(operation_id)
    values = parse_request(operation, query)
    handler = _HANDLERS.get(operation_id)
    if handler is None:  # pragma: no cover - every op id has a handler
        raise ApiError(500, f"no handler registered for {operation_id!r}")
    metadata, records = handler(operation, values, store)
    return _envelope(operation, metadata, records)


def error_body(message: str, operation_id: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"message": message}}
    if operation_id is not None:
        body["error"]["operation"] = operation_id
    return body


def dispatch(method: str, path: str, query: dict[str, list[str]], store,
               body: bytes | None = None):
    """Pure request dispatch: return ``(status, json-serializable body)``.

    GET/HEAD carry parameters in *query*; POST carries a JSON object body
    (``run_benchmark`` only — every other endpoint is GET).
    """
    operation_id = _PATH_TO_OPERATION.get(path)
    if operation_id is None:
        return 404, error_body("unknown endpoint; available: "
                               f"{sorted(operation_routes().values())}")
    if method == "POST":
        if operation_id != "run_benchmark":
            return 405, error_body("only run_benchmark accepts POST", operation_id)
        try:
            payload = json.loads((body or b"{}").decode("utf-8"))
        except ValueError as exc:
            return 400, error_body(f"POST body must be JSON: {exc}", operation_id)
        if not isinstance(payload, dict):
            return 400, error_body("POST body must be a JSON object", operation_id)
        # Fold the JSON body into the spec's query-string shape (one string
        # or list of strings per parameter); the config object travels as a
        # JSON string into the spec's string param.
        query = {}
        for key, value in payload.items():
            if key == "config" and isinstance(value, dict):
                query[key] = [json.dumps(value)]
            elif isinstance(value, list):
                query[key] = [v if isinstance(v, str) else json.dumps(v)
                              for v in value]
            elif isinstance(value, str):
                query[key] = [value]
            else:
                query[key] = [json.dumps(value)]
    elif method not in ("GET", "HEAD"):
        return 405, error_body("only GET is supported on this endpoint", operation_id)
    try:
        return 200, execute_operation(operation_id, query, store)
    except ApiError as exc:
        return exc.status, error_body(exc.message, operation_id)
    except Exception as exc:  # noqa: BLE001 - last-resort gate for the HTTP edge
        logger.warning("internal error on %s: %r", operation_id, exc)
        return 500, error_body("internal error", operation_id)


def make_handler(
    store_resolver: Callable[[], Any],
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class backed by *store_resolver*'s store."""

    class Handler(BaseHTTPRequestHandler):
        """Minimal GET/HEAD/POST JSON server over the spec-derived routes."""

        def do_GET(self) -> None:
            self._handle("GET")

        def do_HEAD(self) -> None:
            self._handle("HEAD")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length > 0 else b"{}"
            self._handle("POST", body=body)

        def _handle(self, method: str, body: bytes | None = None) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            head_only = method == "HEAD"
            try:
                status, body = dispatch(
                    method,
                    parsed.path,
                    query,
                    store_resolver(),
                    body=body,
                )
            except Exception as exc:  # pragma: no cover - dispatch is total
                logger.error("unhandled dispatch failure: %r", exc)
                status, body = 500, error_body("internal error")
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)

        def log_message(self, fmt, *args) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

    return Handler