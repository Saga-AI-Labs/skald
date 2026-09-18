"""Tests for ``api`` — the spec-driven LLM-facing HTTP/JSON surface (M2).

Covers: endpoint/operation parity against ``surfaces.spec``, equality-filter
acceptance and unknown-parameter rejection, exact ``RECORD_FIELDS`` JSON shape
inside the spec envelope, the anomaly-view rule, task drilling, one real
end-to-end request over a live socket with a seeded store, and a live-default-
store smoke request (the real persisted records, bdh_cl + pi50).
"""

from __future__ import annotations

import json
import re
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

import api
from api import dispatch, make_handler
from api.app import endpoint_path, operation_routes
from store.schema import RECORD_FIELDS
from surfaces.spec import ANOMALY_RULE, OPERATIONS, get_operation_ids

CKPT_BDH = "21ec6c220714fac44af5d8a15aa186821596254d2999503c587528adc7590000"
CKPT_PI50 = "a" * 64
PROTO_BDH = "random-crop cold likelihood routing, teacher-forced"
PROTO_PI50 = "phase-1 artifact manifest consistency check"

LIVE_STORE = (
    Path(__file__).resolve().parents[1] / ".skald" / "store"
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


def call(store, operation_id, query_string="", method="GET"):
    return dispatch(method, endpoint_path(operation_id), _qs(query_string), store)


def _qs(query_string: str) -> dict[str, list[str]]:
    from urllib.parse import parse_qs

    return parse_qs(query_string)


def run_server(store) -> tuple[ThreadingHTTPServer, str, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(lambda: store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/api/v1", thread


def http_get(base_url, operation_id, query=""):
    url = f"{base_url}/{operation_id}"
    if query:
        url = f"{url}?{query}"
    with urlopen(url, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


# --- spec-operation <-> endpoint parity ---------------------------------------


def test_endpoint_set_equals_spec_operation_set(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])
    routes = operation_routes()

    assert set(routes) == set(get_operation_ids())
    assert set(routes) == {op.id for op in OPERATIONS}
    assert len(routes) == len(OPERATIONS)
    assert len(set(routes.values())) == len(routes)
    for op_id in get_operation_ids():
        assert endpoint_path(op_id).startswith("/api/v1/")

    # Every registered endpoint actually answers with a spec response.
    for op_id, path in routes.items():
        if op_id == "task_drilldown":
            status, body = dispatch(
                "GET", path,
                {
                    "model_checkpoint_sha256": [CKPT_BDH],
                    "task": ["router"],
                },
                store,
            )
        else:
            status, body = dispatch("GET", path, {}, store)
        assert status == 200, (op_id, body)
        assert body["operation"] == op_id


# --- filter validation --------------------------------------------------------


def test_query_results_accepts_equality_filter(tmp_path):
    store = seeded_store(
        tmp_path, [bdh_record(), pi50_record(value=1.0)]
    )

    status, body = call(store, "query_results", "suite=pi50")

    assert status == 200
    records = body["records"]
    assert records
    assert {r["suite"] for r in records} == {"pi50"}
    assert body["count"] == len(records)


def test_query_results_accepts_numeric_filter_with_kind_coercion(tmp_path):
    store = seeded_store(
        tmp_path, [bdh_record(value=20.5), pi50_record(value=1.0)]
    )

    status, body = call(store, "query_results", "value=1.0")

    assert status == 200
    assert [r["value"] for r in body["records"]] == [1.0]


def test_query_results_rejects_unknown_filter(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])

    status, body = call(store, "query_results", "record_id=42")

    assert status == 400
    assert "unknown parameter(s)" in body["error"]["message"]
    assert body["error"]["operation"] == "query_results"


def test_list_suites_adapters_rejects_unknown_parameter(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])

    status, body = call(store, "list_suites_adapters", "bogus=1")

    assert status == 400
    assert "unknown parameter(s)" in body["error"]["message"]


def test_task_drilldown_requires_its_parameters(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])

    status, body = call(store, "task_drilldown", "model_checkpoint_sha256=" + CKPT_BDH)

    assert status == 400
    assert "missing required parameter(s)" in body["error"]["message"]
    assert "task" in body["error"]["message"]


def test_bad_kind_is_rejected(tmp_path):
    store = seeded_store(tmp_path, [bdh_record(value=20.5)])

    status, body = call(store, "query_results", "value=not-a-number")

    assert status == 400
    assert "valid number" in body["error"]["message"]


# --- JSON shape ---------------------------------------------------------------


def test_records_response_is_rect_fields_exact_inside_spec_envelope(tmp_path):
    store = seeded_store(
        tmp_path, [bdh_record(), pi50_record(value=1.0)]
    )

    status, body = call(store, "query_results")

    assert status == 200
    assert body["spec_id"] == "skald-functional-surface"
    assert body["spec_version"]
    assert body["operation"] == "query_results"
    assert body["count"] == 2
    for record in body["records"]:
        assert list(record) == list(RECORD_FIELDS)
        assert set(record) == set(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", record["model_checkpoint_sha256"])
        assert record["protocol"]


def test_task_drilldown_returns_single_task_and_echoes_metadata(tmp_path):
    store = seeded_store(
        tmp_path,
        [bdh_record(), bdh_record(task="retention", value=10.0), pi50_record()],
    )

    status, body = call(
        store, "task_drilldown", "model_checkpoint_sha256=" + CKPT_BDH + "&task=router"
    )

    assert status == 200
    assert body["model_checkpoint_sha256"] == CKPT_BDH
    assert body["task"] == "router"
    records = body["records"]
    assert records and all(r["task"] == "router" for r in records)
    assert body["count"] == len(records)


def test_list_suites_adapters_lists_distinct_values(tmp_path):
    store = seeded_store(
        tmp_path, [bdh_record(), bdh_record(task="retention"), pi50_record()]
    )

    status, body = call(store, "list_suites_adapters")

    assert status == 200
    assert body["suites"] == ["bdh_cl", "pi50"]
    assert body["adapters"] == ["bdh_cl", "pi50"]


# --- anomaly view -------------------------------------------------------------


def test_anomaly_view_flags_cross_protocol_checkpoints_deterministically(tmp_path):
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

    status, body = call(store, "list_anomalies")

    assert status == 200
    anomalies = body["anomalies"]
    assert anomalies == [
        {
            "model_checkpoint_sha256": CKPT_PI50,
            "protocols": sorted([PROTO_PI50, PROTO_BDH, cross_proto]),
            "protocol_count": 3,
            "reason": ANOMALY_RULE,
        }
    ]
    for record in body["records"]:
        assert record["model_checkpoint_sha256"] == CKPT_PI50
    assert body["count"] == len(body["records"])

    second_status, second_body = call(store, "list_anomalies")
    assert second_status == 200
    assert second_body["anomalies"] == body["anomalies"]


# --- method / path handling ---------------------------------------------------


def test_unknown_endpoint_is_404(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])

    status, body = dispatch("GET", "/api/v1/no_such_op", {}, store)

    assert status == 404
    assert "unknown endpoint" in body["error"]["message"]


def test_non_get_method_is_405(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])

    status, body = dispatch("POST", endpoint_path("query_results"), {}, store)

    assert status == 405
    assert body["error"]["operation"] == "query_results"


# --- end-to-end over a real socket --------------------------------------------


def test_end_to_end_server_over_seeded_store(tmp_path):
    store = seeded_store(
        tmp_path, [bdh_record(), pi50_record(value=1.0)]
    )
    server, base_url, thread = run_server(store)
    try:
        status, body = http_get(base_url, "query_results", "suite=pi50")
        assert status == 200
        assert {r["suite"] for r in body["records"]} == {"pi50"}

        status, body = http_get(base_url, "list_suites_adapters")
        assert status == 200
        assert body["suites"] == ["bdh_cl", "pi50"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_task_drilldown_end_to_end_missing_param_is_400(tmp_path):
    from urllib.error import HTTPError

    store = seeded_store(tmp_path, [bdh_record()])
    server, base_url, thread = run_server(store)
    try:
        with pytest.raises(HTTPError) as excinfo:
            http_get(base_url, "task_drilldown")
        assert excinfo.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- live default store -------------------------------------------------------


@pytest.mark.skipif(
    not LIVE_STORE.is_dir(),
    reason="live default store not present",
)
def test_live_default_store_smoke_request():
    from store.backend import Store

    store = Store(LIVE_STORE)
    server, base_url, thread = run_server(store)
    try:
        status, body = http_get(base_url, "query_results", "")
        assert status == 200
        records = body["records"]

        assert len(records) >= 2
        families = {r["suite"] for r in records}
        assert {"bdh_cl", "pi50"} <= families
        assert len({r["protocol"] for r in records}) >= 2
        for record in records:
            assert list(record) == list(RECORD_FIELDS)
            assert re.fullmatch(r"[0-9a-f]{64}", record["model_checkpoint_sha256"])
            assert record["protocol"]

        status, body = http_get(base_url, "list_suites_adapters")
        assert status == 200
        assert {"bdh_cl", "pi50"} <= set(body["suites"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)