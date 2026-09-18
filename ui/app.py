"""Spec-driven human-facing HTML surface (plan §4.3, §5 ui row).

One HTML view per operation in ``surfaces.spec.OPERATIONS`` — the view set is
generated from the spec at import time and never hand-maintained, so a
capability added to the shared spec appears here automatically (locked by the
parity test in ``tests/test_ui_surface.py``).  Views are read-only queries over
the same ``store.query`` the API uses, with the same declared request
parameters and the same coercion rules, so the UI exposes the same functional
surface as the API by construction.

URL scheme: one GET view per operation at ``/ui/v1/<operation.id>`` (e.g.
``/ui/v1/query_results``); ``/`` redirects to the result-browsing entry point
(``list_suites_adapters``).  Request parameters are the operation's declared
request params carried as query-string pairs; every filterable parameter is an
equality filter passed straight to ``store.query`` — identical semantics to the
API.

Every page embeds the exact record envelope the API returns for the same query
(as an ``application/json`` script block, ``id="skald-surface-data"``), so
conformance parity (identical values, counts, metadata) is true by construction
rather than by duplicated rendering.  The human surface is the HTML tables,
lists, drilldown links and filter forms; the embedded data block just carries
the same canonical records unchanged.

The dispatch logic (:func:`render_page`, :func:`render_view`,
:func:`envelope_for`) is pure (no socket), so the whole surface is
unit-testable without a live server; the socket layer is the
``BaseHTTPRequestHandler`` built by :func:`make_handler`.  No external JS
framework, no cloud LLM anywhere in the chain.
"""

from __future__ import annotations

import html
import json
import logging
import math
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlparse

from store.schema import RECORD_FIELDS
from surfaces.spec import (
    OPERATIONS,
    SPEC_ID,
    SPEC_VERSION,
    RecordsResponse,
    ValuesResponse,
    flag_anomalies,
    get_operation,
)

UI_VERSION = "v1"
_PAGE_PREFIX = f"/ui/{UI_VERSION}"

logger = logging.getLogger("skald.ui")


class UiError(Exception):
    """A request-level failure mapped to an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def view_path(operation_id: str) -> str:
    """The canonical URL path for one spec operation's HTML view."""
    return f"{_PAGE_PREFIX}/{operation_id}"


def view_routes() -> dict[str, str]:
    """Map every spec operation id to its view path (spec-derived)."""
    return {op.id: view_path(op.id) for op in OPERATIONS}


_PATH_TO_OPERATION = {path: op_id for op_id, path in view_routes().items()}


# --- request parsing (mirrors the API surface's spec-driven rules) -----------
#
# Same declared parameters, same coercion, same rejection of unknown/missing
# parameters and the same filter->store.query passthrough as ``api.app``, all
# derived from ``surfaces.spec``.  The conformance sibling asserts the two
# surfaces expose equal capability; both must therefore speak identical
# request semantics.


def _coerce(kind: str, name: str, raw: str) -> Any:
    """Coerce a query-string value to the declared parameter kind."""
    try:
        if kind == "int":
            return int(raw)
        if kind == "number":
            value = float(raw)
            if not _is_finite(value):
                raise ValueError
            return value
        if kind in ("string", "scalar"):
            return raw
    except (TypeError, ValueError):
        raise UiError(
            400, f"parameter {name!r} must be a valid {kind}; got {raw!r}"
        ) from None
    raise UiError(400, f"parameter {name!r} has unknown kind {kind!r}")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def parse_request(operation, query: dict[str, list[str]]) -> dict[str, Any]:
    """Coerce and validate query params against the operation declaration.

    Unknown query parameters and missing required parameters are rejected with
    a 400, so a filter typo is surfaced instead of silently ignored — the same
    contract as the API surface.
    """
    declared = {p.name: p for p in operation.request}
    unknown = sorted(set(query) - set(declared))
    if unknown:
        raise UiError(
            400,
            f"unknown parameter(s) for {operation.id}: {unknown}; "
            f"declared: {sorted(declared)}",
        )

    missing = sorted(
        p.name for p in operation.request if p.required and p.name not in query
    )
    if missing:
        raise UiError(400, f"missing required parameter(s): {missing}")

    values: dict[str, Any] = {}
    for name, param in declared.items():
        if name not in query:
            continue
        raw_values = query[name]
        if len(raw_values) != 1:
            raise UiError(400, f"parameter {name!r} must be given exactly once")
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
            raise UiError(400, "limit must be a non-negative integer")
    return filters, limit


# --- operation execution (the same store queries the API makes) ---------------


def _run(operation, values: dict[str, Any], store) -> tuple[dict[str, Any], list | None]:
    """Execute one operation: ``(metadata, records)`` over ``store.query``.

    Functionally identical to the API surface's handlers: the same filters, the
    same ``store.query`` calls, the same ``flag_anomalies`` rule.  The only
    difference is presentation, which lives in the renderers below.
    """
    op_id = operation.id
    if op_id == "list_suites_adapters":
        records = store.query()
        return {
            "suites": sorted({r["suite"] for r in records}),
            "adapters": sorted({r["adapter"] for r in records}),
        }, None
    if op_id == "query_results":
        filters, limit = _filters_and_limit(operation, values)
        records = store.query(filters=filters, limit=limit)
        return {"count": len(records)}, records
    if op_id == "task_drilldown":
        filters, limit = _filters_and_limit(operation, values)
        records = store.query(filters=filters, limit=limit)
        return {
            "model_checkpoint_sha256": values["model_checkpoint_sha256"],
            "task": values["task"],
            "count": len(records),
        }, records
    if op_id == "list_anomalies":
        filters, limit = _filters_and_limit(operation, values)
        scoped = store.query(filters=filters, limit=limit)
        anomalies = flag_anomalies(scoped)
        flagged = {a["model_checkpoint_sha256"] for a in anomalies}
        records = [r for r in scoped if r["model_checkpoint_sha256"] in flagged]
        return {"count": len(records), "anomalies": anomalies}, records
    raise UiError(500, f"no view handler registered for {op_id!r}")


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
        raise UiError(500, f"unsupported response kind for {operation.id}")
    return body


def envelope_for(
    operation_id: str, query: dict[str, list[str]], store
) -> dict[str, Any]:
    """The API-equivalent response envelope for one operation.

    Pure and spec-derived.  Because the UI executes the same declared
    operation, request parameters and ``store.query`` as the API, this equals
    the API envelope for the equivalent query (asserted by this task's parity
    tests and by the conformance sibling).
    """
    operation = get_operation(operation_id)
    values = parse_request(operation, query)
    metadata, records = _run(operation, values, store)
    return _envelope(operation, metadata, records)


# --- rendering ----------------------------------------------------------------


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _fmt(value: Any) -> str:
    """Human-friendly presentation of one record value."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return " · ".join(str(item) for item in value)
    return str(value)


def _filter_link(field: str, value: Any) -> str:
    query = urlencode({field: str(value)})
    return f"{view_path('query_results')}?{query}"


# Presentation-only mapping from a listing value key (the plural dimension the
# spec's ValuesResponse declares, e.g. "suites") to the singular record field
# that query_results filters on.  Not a capability declaration — it exists only
# so the browsing links land on a valid filter.
_VALUE_FILTER_FIELD = {"suites": "suite", "adapters": "adapter"}


def _drilldown_link(record: dict[str, Any]) -> str:
    query = urlencode(
        {
            "model_checkpoint_sha256": record["model_checkpoint_sha256"],
            "task": record["task"],
        }
    )
    return f"{view_path('task_drilldown')}?{query}"


def _nav(active_id: str) -> str:
    items = []
    for op in OPERATIONS:
        cls = " class=active" if op.id == active_id else ""
        items.append(
            f'<a href="{_esc(view_path(op.id))}" title="{_esc(op.summary)}" '
            f'{cls}>{_esc(op.id)}</a>'
        )
    return '<nav role="navigation">' + "\n".join(items) + "</nav>"


def _filter_form(operation, values: dict[str, Any]) -> str:
    fields = []
    for param in operation.request:
        kind = param.kind
        input_type = "number" if kind in ("number", "int") else "text"
        current = values.get(param.name)
        if isinstance(current, bool):
            current = ""
        value_attr = "" if current is None else f' value="{_esc(str(current))}"'
        required = " required" if param.required else ""
        fields.append(
            f'<div class="field">'
            f'<label for="{_esc(param.name)}">{_esc(param.name)}'
            f'{"*" if param.required else ""}</label>'
            f'<input name="{_esc(param.name)}" id="{_esc(param.name)}" '
            f'type="{input_type}"{value_attr}{required} '
            f'title="{_esc(param.description)}">'
            f"</div>"
        )
    return (
        f'<form class="filters" method="get" '
        f'action="{_esc(view_path(operation.id))}">'
        + "".join(fields)
        + '<button type="submit">Query</button>'
        + "</form>"
    )


def _records_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return '<p class="empty">No records.</p>'
    header = "".join(f"<th>{_esc(field)}</th>" for field in RECORD_FIELDS)
    rows = []
    for record in records:
        cells = []
        for field in RECORD_FIELDS:
            text = _fmt(record[field])
            if field == "model_checkpoint_sha256":
                cell = f'<td class="mono"><a href="{_esc(_drilldown_link(record))}">{_esc(text)}</a></td>'
            elif field == "task":
                cell = f'<td><a href="{_esc(_drilldown_link(record))}">{_esc(text)}</a></td>'
            else:
                cell = f"<td>{_esc(text)}</td>"
            cells.append(cell)
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _active_filters_summary(operation, values: dict[str, Any]) -> str:
    filters = {
        p.name: values[p.name]
        for p in operation.request
        if p.filterable and p.name in values
    }
    if not filters:
        return ""
    chips = " ".join(
        f'<span class="chip">{_esc(name)}={_esc(value)}</span>'
        for name, value in sorted(filters.items())
    )
    return f'<p class="meta">active filters: {chips}</p>'


def _records_view(operation, values: dict[str, Any], envelope: dict[str, Any]) -> str:
    parts = [f"<h2>{_esc(operation.summary)}</h2>"]
    op_id = operation.id
    if op_id == "task_drilldown":
        parts.append(
            f'<p class="meta">drilldown: checkpoint '
            f'<code>{_esc(envelope["model_checkpoint_sha256"])}</code> '
            f'· task <code>{_esc(envelope["task"])}</code> '
            f'· {envelope["count"]} record{"s" if envelope["count"] != 1 else ""}</p>'
        )
    elif op_id == "query_results":
        parts.append(
            f'<p class="meta">{envelope["count"]} record'
            f'{"s" if envelope["count"] != 1 else ""}</p>'
        )
        parts.append(_active_filters_summary(operation, values))
    elif op_id == "list_anomalies":
        parts.append(
            f'<p class="meta">{envelope["count"]} record'
            f'{"s" if envelope["count"] != 1 else ""} belonging to flagged '
            f"cross-protocol checkpoints</p>"
        )
        cards = [
            f'<li class="anomaly"><a class="mono" '
            f'href="{_esc(_filter_link("model_checkpoint_sha256", a["model_checkpoint_sha256"]))}">'
            f'{_esc(a["model_checkpoint_sha256"])}</a> — '
            f'{a["protocol_count"]} protocols '
            f'({_esc(", ".join(a["protocols"]))})'
            f' · {_esc(a["reason"])}</li>'
            for a in envelope["anomalies"]
        ]
        if cards:
            parts.append("<h3>Anomalies</h3><ul>" + "".join(cards) + "</ul>")
        else:
            parts.append('<p class="empty">No anomalies under the declared rule.</p>')
    parts.append(_filter_form(operation, values))
    parts.append("<h3>Records</h3>")
    parts.append(_records_table(envelope["records"]))
    return "".join(parts)


def _values_view(operation, values: dict[str, Any], envelope: dict[str, Any]) -> str:
    parts = [f"<h2>{_esc(operation.summary)}</h2>"]
    for key in operation.response.value_keys:
        items = envelope[key]
        parts.append(f"<h3>{_esc(key)}</h3>")
        if not items:
            parts.append('<p class="empty">No values yet.</p>')
            continue
        links = "".join(
            f'<li><a href="{_esc(_filter_link(_VALUE_FILTER_FIELD.get(key, key), item))}">{_esc(item)}</a></li>'
            for item in items
        )
        parts.append(f'<ul class="values">{links}</ul>')
    return "".join(parts)


def _embed_json(envelope: dict[str, Any]) -> str:
    payload = json.dumps(envelope, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    return (
        '<script type="application/json" id="skald-surface-data">'
        + payload
        + "</script>"
    )


def _document(
    title: str,
    operation,
    content: str,
    envelope: dict[str, Any],
) -> str:
    active = operation.id if operation is not None else ""
    heading = f"Skald UI — {active}" if active else "Skald UI"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
body{{max-width:120ch;margin:2rem auto;padding:0 1rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px;line-height:1.5}}
nav a{{margin-right:1rem;color:#0366d6;text-decoration:none}}
nav a.active{{font-weight:bold}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:.25rem .5rem;text-align:left;vertical-align:top;font-size:13px}}
th{{background:#f6f8fa}}
td.mono{{font-size:12px}}
code{{background:#f6f8fa;padding:0 .25rem}}
form.filters{{display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-end;margin:1rem 0;padding:1rem;border:1px solid #ddd;border-radius:4px}}
form.filters .field{{display:flex;flex-direction:column;font-size:12px}}
input,button{{padding:.25rem .4rem;font:inherit}}
.meta{{color:#57606a}}
.chip{{background:#f6f8fa;border:1px solid #ddd;border-radius:3px;padding:0 .35rem;margin-right:.35rem}}
.anomaly{{margin:.35rem 0}}
.empty{{color:#57606a}}
.error p{{color:#b35900}}
</style>
</head>
<body>
<header>
<h1>{heading}</h1>
{_nav(active)}
</header>
<main>
{content}
</main>
{_embed_json(envelope)}
</body>
</html>"""


def _error_document(status: int, message: str) -> tuple[int, str]:
    content = f'<section class="error"><h2>{status}</h2><p>{_esc(message)}</p></section>'
    page = _document("Skald UI — error", None, content, {})
    return status, page


def render_view(operation_id: str, query: dict[str, list[str]], store) -> tuple[int, str]:
    """Build the full HTML document for one spec operation (pure; no socket)."""
    try:
        operation = get_operation(operation_id)
        values = parse_request(operation, query)
        metadata, records = _run(operation, values, store)
        envelope = _envelope(operation, metadata, records)
    except UiError as exc:
        return _error_document(exc.status, exc.message)
    except Exception as exc:  # noqa: BLE001 - last-resort gate for the HTTP edge
        logger.warning("UI render failure on %s: %r", operation_id, exc)
        return _error_document(500, "internal error")

    response = operation.response
    if isinstance(response, RecordsResponse):
        content = _records_view(operation, values, envelope)
    elif isinstance(response, ValuesResponse):
        content = _values_view(operation, values, envelope)
    else:  # pragma: no cover - spec validation forbids other response kinds
        content = '<p class="empty">Unsupported response kind.</p>'
    return 200, _document(f"Skald UI — {operation.id}", operation, content, envelope)


def render_page(
    method: str, path: str, query: dict[str, list[str]], store
) -> tuple[int, str]:
    """Pure request dispatch: return ``(status, html document)``."""
    if path == "/":
        return 302, ""
    operation_id = _PATH_TO_OPERATION.get(path)
    if operation_id is None:
        return _error_document(
            404,
            "unknown view; available: " f"{sorted(view_routes().values())}",
        )
    if method not in ("GET", "HEAD"):
        return _error_document(
            405, "only GET is supported on this view (HEAD sends headers only)"
        )
    return render_view(operation_id, query, store)


def make_handler(
    store_resolver: Callable[[], Any],
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class backed by *store_resolver*'s store."""

    class Handler(BaseHTTPRequestHandler):
        """Minimal GET/HEAD HTML server over the spec-derived views."""

        server_version = "SkaldUI/1.0"

        def do_GET(self) -> None:
            self._handle(head_only=False)

        def do_HEAD(self) -> None:
            self._handle(head_only=True)

        def _handle(self, head_only: bool) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                status, body = render_page(
                    "HEAD" if head_only else "GET",
                    parsed.path,
                    query,
                    store_resolver(),
                )
            except Exception as exc:  # pragma: no cover - render_page is total
                logger.error("unhandled UI dispatch failure: %r", exc)
                status, body = _error_document(500, "internal error")
            payload = body.encode("utf-8")
            self.send_response(status)
            if status == 302:
                self.send_header("Location", view_path("list_suites_adapters"))
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if not head_only and payload:
                self.wfile.write(payload)

        def log_message(self, fmt, *args) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

    return Handler