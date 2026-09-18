"""Tests for ``ui`` — the spec-driven human-facing HTML surface (M2).

Covers: view set == spec operation set (parity by construction), record
rendering from canned/real records with full ``RECORD_FIELDS`` and HTML
escaping, per-task drilldown rendering the same records as the API for the
equivalent query, the anomaly view, request validation parity, one real
end-to-end request over a live socket with a seeded store, and a live-default-
store render smoke check (the real persisted records, bdh_cl + pi50).
"""

from __future__ import annotations

import json
import re
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs
from urllib.request import urlopen

import pytest

from api import dispatch
from api.app import endpoint_path
from store.schema import RECORD_FIELDS
from surfaces.spec import ANOMALY_RULE, OPERATIONS, flag_anomalies, get_operation_ids
from ui import render_page, view_path, view_routes

CKPT_BDH = "21ec6c220714fac44af5d8a15aa186821596254d2999503c587528adc7590000"
CKPT_PI50 = "a" * 64
PROTO_BDH = "random-crop cold likelihood routing, teacher-forced"
PROTO_PI50 = "phase-1 artifact manifest consistency check"

LIVE_STORE = Path(__file__).resolve().parents[1] / ".skald" / "store"

_DATA_RE = re.compile(
    r'<script type="application/json" id="skald-surface-data">(.*?)</script>',
    re.S,
)


def bdh_record(value: float = 20.5, ckpt: str = CKPT_BDH, **overrides) -> dict:
    record = {
        "model_checkpoint_sha256": ckpt,
        "adapter": "bdh_cl",
        "suite": "bdh_cl",
        "task": "router",
        "metric": "routed_perplexity",
        "value": value,
        "n": 2,
        "protocol": PROTO_BDH,
        "seed": 1234,
    }
    record.update(overrides)
    return record


def pi50_record(value: float = 0.0, ckpt: str = CKPT_PI50, **overrides) -> dict:
    record = {
        "model_checkpoint_sha256": ckpt,
        "adapter": "pi50",
        "suite": "pi50",
        "task": "manifest_check",
        "metric": "manifest_current",
        "value": value,
        "n": 1,
        "protocol": PROTO_PI50,
        "seed": None,
    }
    record.update(overrides)
    return record


def seeded_store(tmp_path, records):
    from store.backend import Store

    store = Store(tmp_path / "store")
    store.put(records)
    return store


def embedded(html_text: str) -> dict:
    match = _DATA_RE.search(html_text)
    assert match is not None, "no embedded surface data block in page"
    return json.loads(match.group(1))


def ui_call(store, operation_id, query_string="", method="GET"):
    return render_page(method, view_path(operation_id), parse_qs(query_string), store)


def api_call(store, operation_id, query_string=""):
    status, body = dispatch(
        "GET", endpoint_path(operation_id), parse_qs(query_string), store
    )
    assert status == 200, body
    return body


def run_server(store) -> tuple[ThreadingHTTPServer, str, threading.Thread]:
    from ui.app import make_handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(lambda: store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/ui/v1", thread


def http_get(base_url, operation_id, query=""):
    url = f"{base_url}/{operation_id}"
    if query:
        url = f"{url}?{query}"
    with urlopen(url, timeout=10) as resp:
        return resp.status, resp.read().decode("utf-8")


# --- spec operation <-> view parity ------------------------------------------


def test_view_set_equals_spec_operation_set(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])
    views = view_routes()

    assert set(views) == set(get_operation_ids())
    assert set(views) == {op.id for op in OPERATIONS}
    assert len(views) == len(OPERATIONS)
    assert len(set(views.values())) == len(views)
    for op_id in get_operation_ids():
        assert view_path(op_id).startswith("/ui/v1/")

    # Every registered view actually renders a complete page.
    for op_id, path in views.items():
        if op_id == "task_drilldown":
            status, body = render_page(
                "GET",
                path,
                {"model_checkpoint_sha256": [CKPT_BDH], "task": ["router"]},
                store,
            )
        else:
            status, body = render_page("GET", path, {}, store)
        assert status == 200, (op_id, body[:200])
        assert embedded(body)["operation"] == op_id
        assert "skald-surface-data" in body


# --- parity: UI embedded records == API records for the equivalent query -----


def test_query_results_renders_same_records_as_api(tmp_path):
    store = seeded_store(tmp_path, [bdh_record(), pi50_record(value=1.0)])

    status, page = ui_call(store, "query_results", "suite=pi50")
    assert status == 200

    ui_envelope = embedded(page)
    api_body = api_call(store, "query_results", "suite=pi50")

    assert ui_envelope["spec_id"] == api_body["spec_id"]
    assert ui_envelope["spec_version"] == api_body["spec_version"]
    assert ui_envelope["operation"] == api_body["operation"]
    assert ui_envelope["count"] == api_body["count"]
    assert ui_envelope["records"] == api_body["records"]


def test_task_drilldown_renders_same_records_as_api(tmp_path):
    store = seeded_store(
        tmp_path,
        [
            bdh_record(),
            bdh_record(task="retention", value=10.0, metric="retention_score"),
            pi50_record(value=1.0),
        ],
    )

    status, page = ui_call(
        store,
        "task_drilldown",
        f"model_checkpoint_sha256={CKPT_BDH}&task=router",
    )
    assert status == 200

    ui_envelope = embedded(page)
    api_body = api_call(
        store,
        "task_drilldown",
        f"model_checkpoint_sha256={CKPT_BDH}&task=router",
    )

    assert ui_envelope["model_checkpoint_sha256"] == CKPT_BDH
    assert ui_envelope["task"] == "router"
    assert ui_envelope["count"] == api_body["count"]
    assert ui_envelope["records"] == api_body["records"]
    assert ui_envelope["records"]

    for record in ui_envelope["records"]:
        assert record["model_checkpoint_sha256"] == CKPT_BDH
        assert record["task"] == "router"
        # The acceptance bar: same identity anchor, metric, value, protocol.
    assert {
        (r["model_checkpoint_sha256"], r["metric"], r["value"], r["protocol"])
        for r in ui_envelope["records"]
    } == {
        (r["model_checkpoint_sha256"], r["metric"], r["value"], r["protocol"])
        for r in api_body["records"]
    }

    # The human page also shows those values verbatim.
    assert "routed_perplexity" in page
    assert str(ui_envelope["records"][0]["value"]) in page


# --- record rendering from canned records -------------------------------------


def test_record_rendering_shows_all_record_fields_and_escapes(tmp_path):
    tricky = bdh_record(protocol="<b>P1 & <i>tricky</i>")
    store = seeded_store(tmp_path, [tricky])

    status, page = ui_call(store, "query_results", "task=router")

    assert status == 200
    for field in RECORD_FIELDS:
        assert f"<th>{field}</th>" in page

    # Values render html-escaped (never raw), and the only <script> tag is the
    # embedded application/json block.
    assert "&lt;b&gt;P1 &amp; &lt;i&gt;tricky&lt;/i&gt;" in page
    data = _DATA_RE.search(page).group(1)
    assert "<b>" not in page.replace(data, "")
    assert "<i>" not in page.replace(data, "")
    assert page.count("<script") == 1

    ui_envelope = embedded(page)
    assert ui_envelope["records"] == [store.query()[0]]
    assert ui_envelope["records"][0]["value"] == 20.5


def test_list_suites_adapters_lists_distinct_values_with_links(tmp_path):
    store = seeded_store(
        tmp_path, [bdh_record(), bdh_record(task="retention"), pi50_record()]
    )

    status, page = ui_call(store, "list_suites_adapters")

    assert status == 200
    ui_envelope = embedded(page)
    assert ui_envelope["suites"] == ["bdh_cl", "pi50"]
    assert ui_envelope["adapters"] == ["bdh_cl", "pi50"]

    # Result-browsing: suite/adapter values link to an equivalent query_results.
    assert f'href="{view_path("query_results")}?suite=bdh_cl"' in page
    assert f'href="{view_path("query_results")}?adapter=pi50"' in page


def test_records_table_links_to_task_drilldown(tmp_path):
    import html as html_mod
    from urllib.parse import urlencode

    store = seeded_store(tmp_path, [bdh_record()])

    status, page = ui_call(store, "query_results", "")

    assert status == 200
    url = view_path("task_drilldown") + "?" + urlencode(
        {"model_checkpoint_sha256": CKPT_BDH, "task": "router"}
    )
    assert f'href="{html_mod.escape(url, quote=True)}"' in page


# --- anomaly view -------------------------------------------------------------


def test_anomaly_view_renders_flagged_checkpoints(tmp_path):
    cross_proto = "teacher-forced, single-shot"
    store = seeded_store(
        tmp_path,
        [
            bdh_record(),
            bdh_record(ckpt=CKPT_PI50, task="retention"),
            pi50_record(value=1.0),
            pi50_record(value=0.5, task="other_task", protocol=cross_proto),
        ],
    )

    status, page = ui_call(store, "list_anomalies")

    assert status == 200
    ui_envelope = embedded(page)

    expected = flag_anomalies(store.query())
    assert ui_envelope["anomalies"] == expected
    assert ui_envelope["anomalies"][0]["protocol_count"] == 3
    assert ui_envelope["anomalies"][0]["reason"] == ANOMALY_RULE
    for record in ui_envelope["records"]:
        assert record["model_checkpoint_sha256"] == CKPT_PI50
    assert ui_envelope["count"] == len(ui_envelope["records"])

    # The human page names the flagged checkpoint and its protocols.
    assert CKPT_PI50 in page
    assert PROTO_PI50 in page
    assert PROTO_BDH in page
    assert cross_proto in page
    assert ANOMALY_RULE in page


# --- request validation parity -------------------------------------------------


def test_task_drilldown_requires_its_parameters(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])

    status, page = ui_call(store, "task_drilldown", "model_checkpoint_sha256=" + CKPT_BDH)

    assert status == 400
    assert "missing required parameter(s)" in page
    # list repr ['task'] html-escaped inside the error message
    assert "[&#x27;task&#x27;]" in page


def test_unknown_filter_is_rejected_like_api(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])

    status, page = ui_call(store, "query_results", "record_id=42")

    assert status == 400
    assert "unknown parameter(s)" in page


def test_bad_kind_is_rejected(tmp_path):
    store = seeded_store(tmp_path, [bdh_record(value=20.5)])

    status, page = ui_call(store, "query_results", "value=not-a-number")

    assert status == 400
    assert "valid number" in page


def test_unknown_view_is_404_and_non_get_is_405(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])

    status, page = render_page("GET", "/ui/v1/no_such_op", {}, store)
    assert status == 404
    assert "unknown view" in page

    status, page = render_page("POST", view_path("query_results"), {}, store)
    assert status == 405
    assert "only GET" in page


def test_root_redirects_to_browsing_entry_point(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])

    status, body = render_page("GET", "/", {}, store)

    assert status == 302
    assert body == ""


# --- end-to-end over a real socket --------------------------------------------


def test_end_to_end_ui_server_over_seeded_store(tmp_path):
    store = seeded_store(tmp_path, [bdh_record(), pi50_record(value=1.0)])
    server, base_url, thread = run_server(store)
    try:
        status, page = http_get(base_url, "query_results", "suite=pi50")
        assert status == 200
        assert {r["suite"] for r in embedded(page)["records"]} == {"pi50"}

        status, page = http_get(base_url, "list_suites_adapters")
        assert status == 200
        assert embedded(page)["suites"] == ["bdh_cl", "pi50"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- live default store --------------------------------------------------------


@pytest.mark.skipif(
    not LIVE_STORE.is_dir(),
    reason="live default store not present",
)
def test_live_default_store_render_smoke():
    from store.backend import Store

    store = Store(LIVE_STORE)

    status, page = ui_call(store, "query_results", "")
    assert status == 200
    records = embedded(page)["records"]

    assert len(records) >= 2
    families = {r["suite"] for r in records}
    assert {"bdh_cl", "pi50"} <= families
    assert len({r["protocol"] for r in records}) >= 2
    for record in records:
        assert list(record) == list(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", record["model_checkpoint_sha256"])
        assert record["protocol"]

    # Browse the listing too.
    status, page = ui_call(store, "list_suites_adapters")
    assert status == 200
    assert {"bdh_cl", "pi50"} <= set(embedded(page)["suites"])

    # Per-task drilldown against a real persisted checkpoint+task renders the
    # same records the API returns for that task.
    probe = records[0]
    query = (
        f"model_checkpoint_sha256={probe['model_checkpoint_sha256']}"
        f"&task={probe['task']}"
    )
    status, page = ui_call(store, "task_drilldown", query)
    assert status == 200
    ui_envelope = embedded(page)
    api_body = api_call(store, "task_drilldown", query)
    assert ui_envelope["count"] == api_body["count"]
    assert ui_envelope["records"] == api_body["records"]

    # Anomaly view over the whole live record set renders.
    status, page = ui_call(store, "list_anomalies")
    assert status == 200
    ui_envelope = embedded(page)
    assert ui_envelope["anomalies"] == flag_anomalies(store.query())
    assert ui_envelope["count"] == len(ui_envelope["records"])