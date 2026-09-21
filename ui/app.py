"""Spec-driven human-facing HTML surface (plan §4.3, §5 ui row).

One HTML view per operation in ``surfaces.spec.OPERATIONS`` — the view set is
generated from the spec at import time and never hand-maintained, so a
capability added to the shared spec appears here automatically (locked by the
parity test in ``tests/test_ui_surface.py``).  Views are read-only queries over
the same ``store.query`` the API uses, with the same declared request
parameters and the same coercion rules, so the UI exposes the same functional
surface as the API by construction.

URL scheme: one GET view per operation at ``/ui/v1/<operation.id>`` (e.g.
``/ui/v1/query_results``); ``/`` is a landing page explaining the app.  Request
parameters are the operation's declared request params carried as
query-string pairs; every filterable parameter is an equality filter passed
straight to ``store.query`` — identical semantics to the API.

Checkpoint hashes render through the ``models.yaml`` name directory
(``identity.names``): known models show human names, unknown ones an
explicit "(unnamed)" stub.  Names are presentation-only — the embedded
data block carries the same canonical records unchanged, so API/UI
envelope parity holds by construction.


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
import re
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlparse

from store.schema import RECORD_FIELDS
from identity.names import display_name, load_directory, resolve
from jobs import adapter_tasks, registry_for_store
from jobs import JobError as _JobError
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
    items = [f'<a href="/" title="What Skald is and where to start">home</a>']
    for op in OPERATIONS:
        cls = " class=active" if op.id == active_id else ""
        items.append(
            f'<a href="{_esc(view_path(op.id))}" title="{_esc(op.summary)}" '
            f'{cls}>{_esc(op.id)}</a>'
        )
    return '<nav role="navigation">' + "\n".join(items) + "</nav>"


def _distinct_options(store) -> dict[str, list[str]]:
    """Sorted distinct values per filterable dimension, from live records."""
    options: dict[str, set[str]] = {}
    try:
        records = store.query()
    except Exception:  # noqa: BLE001 - empty/broken store renders empty selects
        records = []
    for record in records:
        for field in (
            "adapter", "suite", "task", "metric", "model_checkpoint_sha256",
            "protocol", "host",
        ):
            options.setdefault(field, set()).add(str(record.get(field)))
    return {field: sorted(values) for field, values in options.items()}


def _landing(store) -> tuple[int, str]:
    """The home page: what Skald is, known models, entry points."""
    from surfaces.spec import get_operation as _get_op

    try:
        records = store.query()
    except Exception:  # noqa: BLE001
        records = []
    counts: dict[str, int] = {}
    for record in records:
        counts[record.get("adapter", "?")] = counts.get(record.get("adapter", "?"), 0) + 1
    directory = load_directory()
    seen_hashes = sorted({str(r.get("model_checkpoint_sha256")) for r in records})
    model_rows = []
    for sha in seen_hashes:
        entry = resolve(sha, directory)
        model_rows.append(
            f"<tr><td>{_esc(entry.name)}</td>"
            f'<td class="mono">{_esc(sha[:12])}…</td>'
            f"<td>{_esc(entry.kind)}</td>"
            f"<td>{_esc(entry.description)}</td></tr>"
        )
    cards = []
    for op in OPERATIONS:
        cards.append(
            f'<li><a href="{_esc(view_path(op.id))}">{_esc(op.id)}</a> — '
            f"{_esc(op.summary)}</li>"
        )
    counts_text = (
        ", ".join(f"{adapter}: {n}" for adapter, n in sorted(counts.items()))
        or "store is empty"
    )
    content = f"""<h2>Skald — consolidated local evaluation</h2>
<p>One result store behind two equivalent surfaces (this UI and the JSON API).
Every benchmark family writes the same record shape; comparisons are queries.
Pick a view below, or start from a model.</p>
<p class="meta">Store holds {len(records)} records ({_esc(counts_text)}).</p>
<h3>Views</h3><ul>{"".join(cards)}</ul>
<h3>Models in this store</h3>
<table><thead><tr><th>Model</th><th>Checkpoint</th><th>Kind</th><th>What it is</th></tr></thead>
<tbody>{"".join(model_rows) or '<tr><td colspan="4">No records yet.</td></tr>'}</tbody></table>
<p class="meta">Names come from <code>models.yaml</code> in the repo root —
hashes without an entry render as "(unnamed)". JLens layer readouts render
as per-layer bars on their drilldown pages.</p>"""
    operation = _get_op("list_suites_adapters")
    return 200, _document("Skald — home", operation, content, {})


def _filter_form(operation, values: dict[str, Any],
                 options: dict[str, list[str]] | None = None,
                 directory: dict | None = None,
                 method: str = "get") -> str:
    """One input per declared request parameter.

    Dimensions backed by live store values (adapter, suite, task, metric,
    checkpoint, protocol, host) render as dropdowns — no guessing identifiers.
    Free dimensions keep text/number inputs. Every field carries its spec
    description as help text.
    """
    options = options or {}
    fields = []
    for param in operation.request:
        kind = param.kind
        current = values.get(param.name)
        if isinstance(current, bool):
            current = ""
        current_text = "" if current is None else str(current)
        required = " required" if param.required else ""
        help_text = (
            f'<small class="help">{_esc(param.description)}</small>'
        )
        choices = options.get(param.name)
        if param.name == "config":
            # Free-form JSON object; a single-line input would hide mistakes.
            current_json = current_text
            control = (
                f'<textarea name="{_esc(param.name)}" id="{_esc(param.name)}" '
                f'rows="4" cols="48" placeholder=\'{{"max_samples": 20}}\' '
                f'title="{_esc(param.description)}">{_esc(current_json)}</textarea>'
            )
        elif choices:
            opts = []
            if not param.required:
                opts.append("<option value=\"\">— any —</option>")
            for choice in choices:
                selected = " selected" if choice == current_text else ""
                if param.name == "model_checkpoint_sha256":
                    label = display_name(choice, directory)
                else:
                    label = choice if len(choice) <= 90 else choice[:87] + "…"
                opts.append(
                    f'<option value="{_esc(choice)}"{selected}>'
                    f"{_esc(label)}</option>"
                )
            control = (
                f'<select name="{_esc(param.name)}" id="{_esc(param.name)}"'
                f"{required}>" + "".join(opts) + "</select>"
            )
        else:
            input_type = "number" if kind in ("number", "int") else "text"
            value_attr = (
                "" if current is None else f' value="{_esc(str(current))}"'
            )
            control = (
                f'<input name="{_esc(param.name)}" id="{_esc(param.name)}" '
                f'type="{input_type}"{value_attr}{required} '
                f'title="{_esc(param.description)}">'
            )
        fields.append(
            f'<div class="field">'
            f'<label for="{_esc(param.name)}">{_esc(param.name)}'
            f'{"*" if param.required else ""}</label>'
            f"{control}{help_text}"
            f"</div>"
        )
    return (
        f'<form class="filters" method="{method}" '
        f'action="{_esc(view_path(operation.id))}">'
        + "".join(fields)
        + '<button type="submit">'
        + ("Run benchmark" if method == "post" else "Query")
        + "</button>"
        + "</form>"
    )


def _records_table(records: list[dict[str, Any]],
                   directory: dict | None = None) -> str:
    """Full canonical table (every RECORD_FIELDS column, per the parity test)
    led by a human Model column; long protocols clamp via CSS with the full
    text on hover so nothing is hidden, only shortened on screen."""
    if not records:
        return '<p class="empty">No records.</p>'
    header = "<th>Model</th>" + "".join(
        f"<th>{_esc(field)}</th>" for field in RECORD_FIELDS
    )
    rows = []
    for record in records:
        sha = str(record["model_checkpoint_sha256"])
        model_cell = (
            f'<td class="modelname"><a href="{_esc(_drilldown_link(record))}" '
            f'title="{_esc(sha)}">{_esc(display_name(sha, directory))}</a></td>'
        )
        cells = [model_cell]
        for field in RECORD_FIELDS:
            text = _fmt(record[field])
            if field == "model_checkpoint_sha256":
                cell = (
                    f'<td class="mono"><a href="{_esc(_drilldown_link(record))}" '
                    f'title="{_esc(text)}">{_esc(text[:12])}…</a></td>'
                )
            elif field == "task":
                cell = f'<td><a href="{_esc(_drilldown_link(record))}">{_esc(text)}</a></td>'
            elif field == "protocol":
                cell = (
                    f'<td class="proto" title="{_esc(text)}">{_esc(text)}</td>'
                )
            else:
                cell = f"<td>{_esc(text)}</td>"
            cells.append(cell)
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


_JLENS_METRIC_RE = re.compile(r"^top(\d+)_prob@L(\d+)$")


def _jlens_viz(records: list[dict[str, Any]]) -> str:
    """Per-layer top-k readout bars for JLens records, with an explainer.

    The Jacobian lens transports an early layer's residual into the final
    layer's basis and decodes it with the model's own unembedding: each bar
    is the probability the layer's state assigns to that token — what the
    layer is "disposed to say". Bars share a 0–1 scale so layers are
    comparable; layers are ordered numerically.
    """
    by_layer: dict[int, list[tuple[int, float]]] = {}
    for record in records:
        m = _JLENS_METRIC_RE.match(str(record.get("metric", "")))
        if m is None:
            continue
        rank, layer = int(m.group(1)), int(m.group(2))
        try:
            prob = float(record.get("value"))
        except (TypeError, ValueError):
            continue
        by_layer.setdefault(layer, []).append((rank, prob))
    if not by_layer:
        return ""
    parts = [
        "<h3>Layer readout</h3>"
        '<p class="meta">What each layer is disposed to say, per the '
        "Jacobian lens (Anthropic, via Neuronpedia — see CREDITS.md): "
        "top predicted tokens and their probabilities, 0–1 scale shared "
        "across layers. A sharp top-1 means the layer already speaks; a "
        "flat spread means it does not.</p>"
    ]
    for layer in sorted(by_layer):
        rows = sorted(by_layer[layer])
        bars = "".join(
            f'<div class="bar-row"><span class="bar-rank">#{rank}</span>'
            f'<span class="bar-track"><span class="bar-fill" '
            f'style="width:{max(0.0, min(1.0, prob)) * 100:.1f}%"></span></span>'
            f'<span class="bar-val">{prob:.4f}</span></div>'
            for rank, prob in rows
        )
        parts.append(f"<h4>Layer {layer}</h4>{bars}")
    return "".join(parts)


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


def _records_view(operation, values: dict[str, Any], envelope: dict[str, Any],
                  store=None) -> str:
    directory = load_directory()
    options = _distinct_options(store) if store is not None else {}
    parts = [f"<h2>{_esc(operation.summary)}</h2>"]
    op_id = operation.id
    if op_id == "task_drilldown":
        parts.append(
            f'<p class="meta">drilldown: checkpoint '
            f'<code>{_esc(envelope["model_checkpoint_sha256"])}</code> '
            f'· task <code>{_esc(envelope["task"])}</code> '
            f'· {envelope["count"]} record{"s" if envelope["count"] != 1 else ""}</p>'
        )
        entry = resolve(envelope["model_checkpoint_sha256"], directory)
        parts.append(
            f'<div class="modelcard"><strong>{_esc(entry.name)}</strong> '
            f'<span class="kind">{_esc(entry.kind)}</span><br>'
            f'<span class="meta">{_esc(entry.description)}</span></div>'
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
            f'<li class="anomaly"><a '
            f'href="{_esc(_filter_link("model_checkpoint_sha256", a["model_checkpoint_sha256"]))}">'
            f'{_esc(display_name(a["model_checkpoint_sha256"], directory))}</a> '
            f'<span class="mono">{_esc(a["model_checkpoint_sha256"][:12])}…</span> — '
            f'{a["protocol_count"]} protocols '
            f'({_esc(", ".join(a["protocols"]))})'
            f' · {_esc(a["reason"])}</li>'
            for a in envelope["anomalies"]
        ]
        if cards:
            parts.append("<h3>Anomalies</h3><ul>" + "".join(cards) + "</ul>")
        else:
            parts.append('<p class="empty">No anomalies under the declared rule.</p>')
    parts.append(_filter_form(operation, values, options, directory))
    parts.append("<h3>Records</h3>")
    parts.append(_jlens_viz(envelope["records"]))
    parts.append(_records_table(envelope["records"], directory))
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
td.modelname{{font-weight:bold;white-space:nowrap}}
td.proto{{max-width:38ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;color:#57606a}}
form.filters select{{max-width:44ch;padding:.25rem .4rem;font:inherit}}
small.help{{display:block;color:#57606a;font-size:12px;max-width:44ch}}
.modelcard{{border:1px solid #ddd;border-radius:4px;padding:.6rem 1rem;margin:.6rem 0;background:#f6f8fa}}
.kind{{background:#ddf4ff;border:1px solid #54aeff;border-radius:2em;padding:0 .6rem;font-size:12px}}
.bar-row{{display:flex;align-items:center;gap:.6rem;margin:.15rem 0;max-width:70ch}}
.bar-rank{{width:3ch;text-align:right;color:#57606a}}
.bar-track{{flex:1;background:#eaeef2;border-radius:3px;height:1.1em;overflow:hidden}}
.bar-fill{{display:block;background:#0969da;height:100%}}
.bar-val{{width:7ch;font-variant-numeric:tabular-nums}}
h4{{margin-bottom:.2rem}}
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


def _run_form_options() -> dict[str, list[str]]:
    """Adapter/task dropdown values for the run form (from the registry)."""
    tasks = adapter_tasks()
    all_tasks = sorted({task for names in tasks.values() for task in names})
    return {"adapter": sorted(tasks), "task": all_tasks}


def _job_page(operation, job: dict[str, Any], store,
              notice: str = "") -> tuple[int, str]:
    """Status page for one benchmark job (pure; no socket)."""
    status = job["status"]
    rows = [
        ("Job", job["job_id"]),
        ("Status", status),
        ("Adapter", job["adapter"]),
        ("Task", job["task"]),
        ("Model", job["model"]),
        ("Records persisted", str(job["record_count"])),
    ]
    if job.get("checkpoint"):
        rows.append(("Checkpoint", job["checkpoint"]))
    if job.get("error"):
        rows.append(("Error", job["error"]))
    body_rows = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows
    )
    extra = ""
    if notice:
        extra += f'<p class="meta">{_esc(notice)}</p>'
    if status in ("queued", "running"):
        extra += (
            '<p class="meta">The benchmark is still running — '
            '<a href="">reload</a> to refresh. Long runs survive in the '
            "background; closing this page does not stop them.</p>"
        )
    elif status == "done" and job.get("checkpoint"):
        query = urlencode(
            {"model_checkpoint_sha256": job["checkpoint"], "task": job["task"]}
        )
        extra += (
            f'<p><a href="{_esc(view_path("task_drilldown"))}?{query}">'
            "View the persisted records</a></p>"
        )
    elif status == "failed":
        extra += (
            "<p>Fix the input and resubmit from the "
            f'<a href="{_esc(view_path("run_benchmark"))}">run form</a>.</p>'
        )
    elif status == "orphaned":
        extra += "<p>Resubmit from the run form to retry.</p>"
    content = (
        f"<h2>Benchmark job { _esc(job['job_id'])}</h2>"
        f"<table><tbody>{body_rows}</tbody></table>{extra}"
    )
    return 200, _document(
        f"Skald UI — job {job['job_id']}", operation, content,
        _job_envelope(operation, job),
    )


def _job_envelope(operation, job: dict[str, Any]) -> dict[str, Any]:
    """The API-equivalent job_status envelope for one job record."""
    single = lambda v: [] if v is None else [v]  # noqa: E731
    return {
        "spec_id": SPEC_ID,
        "spec_version": SPEC_VERSION,
        "operation": operation.id,
        "job_id": [job["job_id"]],
        "status": [job["status"]],
        "adapter": [job["adapter"]],
        "task": [job["task"]],
        "model": [job["model"]],
        "record_count": [job["record_count"]],
        "checkpoint": single(job["checkpoint"]),
        "error": single(job["error"]),
    }


def render_view(operation_id: str, query: dict[str, list[str]], store) -> tuple[int, str]:
    """Build the full HTML document for one spec operation (pure; no socket)."""
    try:
        operation = get_operation(operation_id)
        if operation_id == "run_benchmark" and not any(
            key in query for key in ("adapter", "task", "model")
        ):
            # Bare GET: the run form, not an error. Submission is POST.
            content = (
                f"<h2>{_esc(operation.summary)}</h2>"
                '<p class="meta">Pick an adapter and task, name the run '
                "target per that adapter's contract (checkpoint path, "
                "endpoint URL, or hash), and add config as a JSON object. "
                "The run executes in the background; its records land in "
                "this store like any adapter CLI run.</p>"
                + _filter_form(operation, {}, _run_form_options(),
                               load_directory(), method="post")
            )
            return 200, _document("Skald UI — run a benchmark", operation,
                                  content, {})
        values = parse_request(operation, query)
        if operation_id in ("run_benchmark", "job_status"):
            return _submit_or_status(operation, operation_id, values, store)
        metadata, records = _run(operation, values, store)
        envelope = _envelope(operation, metadata, records)
    except UiError as exc:
        return _error_document(exc.status, exc.message)
    except Exception as exc:  # noqa: BLE001 - last-resort gate for the HTTP edge
        logger.warning("UI render failure on %s: %r", operation_id, exc)
        return _error_document(500, "internal error")

    response = operation.response
    if isinstance(response, RecordsResponse):
        content = _records_view(operation, values, envelope, store)
    elif isinstance(response, ValuesResponse):
        content = _values_view(operation, values, envelope)
    else:  # pragma: no cover - spec validation forbids other response kinds
        content = '<p class="empty">Unsupported response kind.</p>'
    return 200, _document(f"Skald UI — {operation.id}", operation, content, envelope)


def _submit_or_status(operation, operation_id: str,
                      values: dict[str, Any], store) -> tuple[int, str]:
    """Execute the run_benchmark / job_status operations for the UI."""
    registry = registry_for_store(store)
    if operation_id == "run_benchmark":
        config_raw = values.get("config")
        try:
            config = json.loads(config_raw) if config_raw else {}
        except ValueError:
            raise UiError(400, "parameter 'config' must be a JSON object") from None
        if not isinstance(config, dict):
            raise UiError(400, "parameter 'config' must be a JSON object")
        try:
            job = registry.submit(
                values["adapter"], values["task"], values["model"], config
            )
        except _JobError as exc:
            raise UiError(400, str(exc)) from exc
        return _job_page(
            operation, registry.get(job["job_id"]), store,
            notice="Run submitted — this page tracks it to completion.",
        )
    try:
        job = registry.get(values["job_id"])
    except KeyError as exc:
        raise UiError(404, f"unknown job {values['job_id']!r}") from exc
    return _job_page(operation, job, store)


def render_page(
    method: str, path: str, query: dict[str, list[str]], store,
    body: bytes | None = None,
) -> tuple[int, str]:
    """Pure request dispatch: return ``(status, html document)``.

    GET/HEAD carry parameters in *query*; POST carries a form-encoded body
    (the run form only — every other view is GET).
    """
    if path == "/":
        return _landing(store)
    operation_id = _PATH_TO_OPERATION.get(path)
    if operation_id is None:
        return _error_document(
            404,
            "unknown view; available: " f"{sorted(view_routes().values())}",
        )
    if method == "POST":
        if operation_id != "run_benchmark":
            return _error_document(
                405, "only the run form accepts POST"
            )
        try:
            form = parse_qs((body or b"").decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return _error_document(400, "unreadable form body")
        return render_view(operation_id, form, store)
    if method not in ("GET", "HEAD"):
        return _error_document(
            405, "views are GET (HEAD sends headers only); only the run "
            "form accepts POST"
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
            self._handle("GET")

        def do_HEAD(self) -> None:
            self._handle("HEAD")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length > 0 else b""
            self._handle("POST", body=body)

        def _handle(self, method: str, body: bytes | None = None) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                status, body = render_page(
                    method,
                    parsed.path,
                    query,
                    store_resolver(),
                    body=body,
                )
            except Exception as exc:  # pragma: no cover - render_page is total
                logger.error("unhandled UI dispatch failure: %r", exc)
                status, body = _error_document(500, "internal error")
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if method != "HEAD" and payload:
                self.wfile.write(payload)

        def log_message(self, fmt, *args) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

    return Handler