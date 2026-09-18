"""Dual-surface conformance tests (plan §3.1, §4.3; M2).

Asserts that the API surface and the UI surface expose equal functional
capability over the one result schema, generated from the shared capability
spec (``surfaces.spec``).  Two levels, as the acceptance criterion requires:

- Structural:  the API endpoint set == the UI view set == the spec operation
  set — no surface invents or drops a capability.
- Behavioral:  for the same filter/request executed against one seeded store,
  the API JSON response and the UI-rendered page return the same records
  (same record keys, values, and counts) — a comparison of functional
  capability, not string equality of formats.

The UI embeds the exact response envelope in the page as an
``application/json`` script block (``id="skald-surface-data"``), so the parity
check reads the data the rendered page actually carries.  Everything runs
read-only against real local modules and a local store; no cloud LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from api import dispatch
from api.app import endpoint_path, operation_routes
from store.schema import RECORD_FIELDS
from surfaces.spec import ANOMALY_RULE, OPERATIONS, flag_anomalies, get_operation_ids
from ui import render_page
from ui.app import view_path, view_routes

CKPT_BDH = "21ec6c220714fac44af5d8a15aa186821596254d2999503c587528adc7590000"
CKPT_PI50 = "a" * 64
CKPT_CROSS = "b" * 64
PROTO_BDH = "random-crop cold likelihood routing, teacher-forced"
PROTO_PI50 = "phase-1 artifact manifest consistency check"
PROTO_CROSS = "teacher-forced, single-shot"

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


def embedded(page: str) -> dict:
    """The response envelope the rendered UI page actually carries."""
    match = _DATA_RE.search(page)
    assert match is not None, "no embedded surface data block in page"
    return json.loads(match.group(1))


def api_call(store, operation_id: str, query_string: str = ""):
    status, body = dispatch(
        "GET", endpoint_path(operation_id), parse_qs(query_string), store
    )
    assert status == 200, body
    return body


def ui_call(store, operation_id: str, query_string: str = ""):
    status, page = render_page(
        "GET", view_path(operation_id), parse_qs(query_string), store
    )
    assert status == 200, (operation_id, page[:200])
    return embedded(page)


# --- structural parity: endpoint set == view set == spec operation set --------


def test_endpoint_set_view_set_operation_set_are_equal(tmp_path):
    store = seeded_store(tmp_path, [bdh_record(), pi50_record(value=1.0)])

    api_ops = set(operation_routes())
    ui_ops = set(view_routes())
    spec_ops = set(get_operation_ids())

    assert api_ops == ui_ops == spec_ops
    assert len(api_ops) == len(OPERATIONS)
    assert set(operation_routes().values()) == {
        endpoint_path(op) for op in spec_ops
    }
    assert set(view_routes().values()) == {view_path(op) for op in spec_ops}

    # Every capability is reachable on both surfaces, not just declared.
    for op_id in spec_ops:
        query = (
            f"model_checkpoint_sha256={CKPT_BDH}&task=router"
            if op_id == "task_drilldown"
            else ""
        )
        api_status, _ = dispatch(
            "GET", endpoint_path(op_id), parse_qs(query), store
        )
        ui_status, _ = render_page(
            "GET", view_path(op_id), parse_qs(query), store
        )
        assert api_status == 200, op_id
        assert ui_status == 200, op_id


# --- behavioral parity: same filter, same records, on one seeded store --------


def test_query_results_same_filter_same_records_on_both_surfaces(tmp_path):
    store = seeded_store(
        tmp_path,
        [
            bdh_record(),
            bdh_record(task="retention", value=10.0),
            pi50_record(value=1.0),
        ],
    )

    filters = [
        "",
        "suite=pi50",
        "adapter=pi50",
        "task=router",
        f"model_checkpoint_sha256={CKPT_BDH}",
        "protocol=" + PROTO_BDH,
        "value=1.0",
    ]
    for query in filters:
        api_body = api_call(store, "query_results", query)
        ui_body = ui_call(store, "query_results", query)
        assert ui_body["records"] == api_body["records"], query
        assert ui_body["count"] == api_body["count"] == len(api_body["records"])
        for record in api_body["records"]:
            assert list(record) == list(RECORD_FIELDS)


def test_list_suites_adapters_parity(tmp_path):
    store = seeded_store(
        tmp_path, [bdh_record(), bdh_record(task="retention"), pi50_record()]
    )

    api_body = api_call(store, "list_suites_adapters")
    ui_body = ui_call(store, "list_suites_adapters")

    assert ui_body["suites"] == api_body["suites"] == ["bdh_cl", "pi50"]
    assert ui_body["adapters"] == api_body["adapters"] == ["bdh_cl", "pi50"]
    assert ui_body["spec_id"] == api_body["spec_id"]
    assert ui_body["spec_version"] == api_body["spec_version"]


def test_task_drilldown_parity(tmp_path):
    store = seeded_store(
        tmp_path,
        [
            bdh_record(),
            bdh_record(task="retention", value=10.0),
            pi50_record(value=1.0),
        ],
    )
    query = f"model_checkpoint_sha256={CKPT_BDH}&task=router"

    api_body = api_call(store, "task_drilldown", query)
    ui_body = ui_call(store, "task_drilldown", query)

    assert ui_body["model_checkpoint_sha256"] == CKPT_BDH
    assert ui_body["task"] == "router"
    assert ui_body["count"] == api_body["count"]
    assert ui_body["records"] == api_body["records"]
    assert ui_body["records"]
    for record in api_body["records"]:
        assert (record["model_checkpoint_sha256"], record["task"]) == (
            CKPT_BDH,
            "router",
        )


def test_anomaly_view_parity(tmp_path):
    store = seeded_store(
        tmp_path,
        [
            # CKPT_PI50 spans three protocols -> one flagged checkpoint.
            pi50_record(value=1.0),
            pi50_record(value=0.5, task="other_task", protocol=PROTO_CROSS),
            bdh_record(ckpt=CKPT_PI50, task="retention"),
            # CKPT_BDH stays under a single protocol -> not flagged.
            bdh_record(),
        ],
    )

    api_body = api_call(store, "list_anomalies")
    ui_body = ui_call(store, "list_anomalies")

    expected = flag_anomalies(store.query())
    assert ui_body["anomalies"] == api_body["anomalies"] == expected
    assert ui_body["records"] == api_body["records"]
    assert ui_body["count"] == api_body["count"] == len(api_body["records"])
    assert api_body["anomalies"][0]["reason"] == ANOMALY_RULE
    for record in api_body["records"]:
        assert record["model_checkpoint_sha256"] == CKPT_PI50


# --- request semantics parity (same rejections on both surfaces) --------------


def test_unknown_filter_rejected_on_both_surfaces(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])

    api_status, api_body = dispatch(
        "GET", endpoint_path("query_results"), parse_qs("record_id=42"), store
    )
    ui_status, ui_page = render_page(
        "GET", view_path("query_results"), parse_qs("record_id=42"), store
    )

    assert api_status == 400
    assert ui_status == 400
    assert "unknown parameter(s)" in api_body["error"]["message"]
    assert "unknown parameter(s)" in ui_page


def test_missing_required_param_rejected_on_both_surfaces(tmp_path):
    store = seeded_store(tmp_path, [bdh_record()])
    query = f"model_checkpoint_sha256={CKPT_BDH}"

    api_status, api_body = dispatch(
        "GET", endpoint_path("task_drilldown"), parse_qs(query), store
    )
    ui_status, ui_page = render_page(
        "GET", view_path("task_drilldown"), parse_qs(query), store
    )

    assert api_status == 400
    assert ui_status == 400
    assert "missing required parameter(s)" in api_body["error"]["message"]
    assert "missing required parameter(s)" in ui_page


# --- live default store (the persisted bdh_cl + pi50 records) -----------------


@pytest.mark.skipif(
    not LIVE_STORE.is_dir(),
    reason="live default store not present",
)
def test_live_default_store_conformance():
    from store.backend import Store

    store = Store(LIVE_STORE)

    api_body = api_call(store, "query_results", "")
    ui_body = ui_call(store, "query_results", "")

    assert ui_body["records"] == api_body["records"]
    assert ui_body["count"] == api_body["count"]
    families = {r["suite"] for r in api_body["records"]}
    assert {"bdh_cl", "pi50"} <= families
    assert len({r["protocol"] for r in api_body["records"]}) >= 2
    for record in api_body["records"]:
        assert list(record) == list(RECORD_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", record["model_checkpoint_sha256"])
        assert record["protocol"]

    # The two family counts are the sc-dual-surfaces evidence tally.
    by_family: dict[str, int] = {}
    for record in api_body["records"]:
        by_family[record["suite"]] = by_family.get(record["suite"], 0) + 1
    assert by_family["bdh_cl"] >= 1 and by_family["pi50"] >= 1

    probe = api_body["records"][0]
    query = (
        f"model_checkpoint_sha256={probe['model_checkpoint_sha256']}"
        f"&task={probe['task']}"
    )
    api_drill = api_call(store, "task_drilldown", query)
    ui_drill = ui_call(store, "task_drilldown", query)
    assert ui_drill["records"] == api_drill["records"]
    assert ui_drill["count"] == api_drill["count"]

    api_listing = api_call(store, "list_suites_adapters")
    ui_listing = ui_call(store, "list_suites_adapters")
    assert ui_listing == api_listing