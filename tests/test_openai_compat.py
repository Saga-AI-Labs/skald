"""Tests for ``adapters.openai_compat`` — the served-model endpoint adapter.

All tests run against an in-process stub OpenAI-compatible server (stdlib
``http.server`` in a thread): no network, no weights, no datasets package.
Real-endpoint verification is a documented manual step (docs/usage.md), not
part of this suite.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import store
from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.likelihood_stats import KLD_METRICS
from adapters.openai_compat import EndpointError, OpenAICompatAdapter

KLD_METRICS_NAMES = KLD_METRICS

MMLU_ITEMS = [
    {"prompt": "2+2?", "choices": ["3", "4", "5", "6"], "gold": "B"},
    {"prompt": "Capital of France?", "choices": ["Rome", "Paris", "Oslo", "Bern"], "gold": "B"},
    {"prompt": "Water boils at?", "choices": ["90C", "100C", "110C", "120C"], "gold": "B"},
]

HUMANEVAL_ITEMS = [
    {
        "prompt": "def add(a, b):\n    \"\"\"Add.\"\"\"\n",
        "entry_point": "add",
        "test": "def check(f):\n    assert f(1, 2) == 3\n",
    },
    {
        "prompt": "def sub(a, b):\n    \"\"\"Subtract.\"\"\"\n",
        "entry_point": "sub",
        "test": "def check(f):\n    assert f(5, 3) == 2\n",
    },
]


class _Stub(BaseHTTPRequestHandler):
    """Canned OpenAI-compatible server: answers B, emits correct code."""

    completions_seen: list = []
    logprob_delta: float = 0.0

    def _json(self, body, status=200):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/models":
            self._json({"object": "list", "data": [{"id": "stub-model"}]})
        else:
            self._json({"error": "nope"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/completions":
            self._completions(req)
            return
        prompt = req.get("messages", [{}])[-1].get("content", "")
        type(self).completions_seen.append(prompt)
        if "def add" in prompt or "def sub" in prompt:
            # echo a correct body for whichever function was asked
            name = "add" if "def add" in prompt else "sub"
            op = "+" if name == "add" else "-"
            text = f"    return a {op} b"
        else:
            text = "B"
        self._json({"choices": [{"message": {"content": text}}]})

    def _completions(self, req):
        # Legacy /completions with echo: true + logprobs: 0. Tokens are
        # whitespace runs; logprobs are deterministic in the token text so
        # capture and parity agree byte-for-byte when nothing changed.
        prompt = req.get("prompt", "")
        tokens = re.findall(r"\S+\s*|\S", prompt)
        token_logprobs = [
            -(len(t) + 1.0) + type(self).logprob_delta for t in tokens
        ]
        self._json({"choices": [{"logprobs": {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
        }}]})

    def log_message(self, *a):
        pass


@pytest.fixture()
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    _Stub.completions_seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _adapter():
    return OpenAICompatAdapter()


def test_adapter_interface():
    adapter = _adapter()
    assert isinstance(adapter, SuiteAdapter)
    sig = inspect.signature(adapter.run)
    params = list(sig.parameters)
    assert params[:2] == ["model", "task"]
    assert "config" in params


def test_model_must_be_a_url():
    with pytest.raises(ValueError, match="must be the endpoint base URL"):
        _adapter().run("/local/path/model", "mmlu", {})


def test_unknown_task_is_rejected(endpoint):
    with pytest.raises(ValueError, match="unsupported task"):
        _adapter().run(endpoint, "gsm8k", {})


def test_mmlu_inline_items(endpoint):
    records = _adapter().run(
        endpoint, "mmlu",
        {"model": "stub-model", "mmlu_items": MMLU_ITEMS, "num_fewshot": 1},
    )
    assert len(records) == 1
    r = records[0]
    assert set(r) == set(RECORD_FIELDS)
    assert r["adapter"] == "openai_compat" and r["task"] == "mmlu"
    assert r["metric"] == "accuracy"
    assert r["value"] == 1.0 and r["n"] == 2  # 3 items, 1 used as shot
    assert "UNVERIFIED" in r["protocol"]
    assert r["artifacts"] == [f"endpoint::{endpoint}"]
    assert re.fullmatch(r"[0-9a-f]{64}", r["model_checkpoint_sha256"])


def test_humaneval_inline_items(endpoint):
    records = _adapter().run(
        endpoint, "humaneval",
        {"model": "stub-model", "humaneval_items": HUMANEVAL_ITEMS},
    )
    assert len(records) == 1
    r = records[0]
    assert r["metric"] == "pass_at_1"
    assert r["value"] == 1.0 and r["n"] == 2
    assert "UNVERIFIED" in r["protocol"]


def test_default_model_comes_from_the_server(endpoint):
    records = _adapter().run(
        endpoint, "mmlu", {"mmlu_items": MMLU_ITEMS, "num_fewshot": 1}
    )
    assert "stub-model" in records[0]["protocol"]


def test_identity_is_stable_and_endpoint_scoped(endpoint):
    a = _adapter().run(
        endpoint, "mmlu", {"mmlu_items": MMLU_ITEMS, "num_fewshot": 1}
    )[0]
    b = _adapter().run(endpoint, "humaneval", {"humaneval_items": HUMANEVAL_ITEMS[:1]})[0]
    assert a["model_checkpoint_sha256"] == b["model_checkpoint_sha256"]
    c = _adapter().run(
        endpoint, "mmlu",
        {"model": "other-model", "mmlu_items": MMLU_ITEMS, "num_fewshot": 1},
    )[0]
    assert c["model_checkpoint_sha256"] != a["model_checkpoint_sha256"]


def test_reasoning_fallback_when_content_is_null(endpoint):
    from adapters.openai_compat import _Client

    class _ReasoningStub(_Stub):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self._json({"choices": [{"message": {
                "role": "assistant", "content": None,
                "reasoning": "thinking... so the answer is B",
            }}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReasoningStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = _Client(
            f"http://127.0.0.1:{server.server_port}",
            api_key=None, timeout=30,
        )
        client.model = "reasoning-stub"
        assert "B" in client.complete("pick B", 16)
    finally:
        server.shutdown()


def test_unreachable_endpoint_is_an_endpoint_error():
    with pytest.raises(EndpointError, match="cannot reach"):
        _adapter().run("http://127.0.0.1:1", "mmlu", {"mmlu_items": MMLU_ITEMS})


def test_empty_items_raise(endpoint):
    with pytest.raises(EndpointError, match="zero evaluation items"):
        _adapter().run(endpoint, "mmlu", {"mmlu_items": []})


def test_store_roundtrip_isolated(endpoint, tmp_path, monkeypatch):
    import store as store_mod

    store_mod._DEFAULT_STORE = None
    monkeypatch.setenv("SKALD_STORE_DIR", str(tmp_path / "isolated"))
    records = _adapter().run(
        endpoint, "mmlu", {"mmlu_items": MMLU_ITEMS, "num_fewshot": 1}
    )
    stored = store.put(records)
    assert stored == records
    back = store.query({"adapter": "openai_compat"})
    assert len(back) == len(records)


def test_extract_code_prefers_python_fences():
    from adapters.openai_compat import _extract_code

    assert _extract_code("```python\n    return 1\n```") == "    return 1"
    assert _extract_code("text\n```\n    return 2\n```") == "    return 2"
    multi = "```text\nnope\n```\n```python\n    return 3\n```"
    assert _extract_code(multi) == "    return 3"


def test_extract_code_preserves_indentation():
    from adapters.openai_compat import _extract_code

    assert _extract_code("\n\n    return a + b") == "    return a + b"
    assert _extract_code("    return a + b") == "    return a + b"


def test_determinism_requires_a_prompt(endpoint):
    with pytest.raises(ValueError, match="config\\['prompt'\\]"):
        _adapter().run(endpoint, "determinism", {"model": "stub-model"})
    with pytest.raises(ValueError, match="repeats >= 2"):
        _adapter().run(
            endpoint, "determinism",
            {"model": "stub-model", "prompt": "say hi", "repeats": 1},
        )


def test_determinism_stable_endpoint(endpoint):
    records = _adapter().run(
        endpoint, "determinism",
        {"model": "stub-model", "prompt": "say hi", "repeats": 12},
    )
    by_metric = {r["metric"]: r for r in records}
    assert set(by_metric) == {
        "distinct_output_rate", "reproducible", "num_distinct_outputs",
    }  # no divergence metric when all outputs agree
    assert by_metric["distinct_output_rate"]["value"] == pytest.approx(1 / 12)
    assert by_metric["reproducible"]["value"] == 1.0
    assert by_metric["num_distinct_outputs"]["value"] == 1.0
    for r in records:
        assert set(r) == set(RECORD_FIELDS)
        assert r["task"] == "determinism" and r["n"] == 12
        assert "UNVERIFIED" in r["protocol"]


def test_determinism_reports_first_divergence():
    class _FlipFlop(_Stub):
        calls = 0

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            type(self).calls += 1
            text = "answer: yes" if type(self).calls % 2 else "answer: no!"
            self._json({"choices": [{"message": {"content": text}}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), _FlipFlop)
    _FlipFlop.calls = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        records = _adapter().run(
            f"http://127.0.0.1:{server.server_port}", "determinism",
            {"model": "stub-model", "prompt": "yes or no?", "repeats": 10},
        )
    finally:
        server.shutdown()
    by_metric = {r["metric"]: r for r in records}
    assert by_metric["distinct_output_rate"]["value"] == pytest.approx(2 / 10)
    assert by_metric["reproducible"]["value"] == 0.0
    assert by_metric["num_distinct_outputs"]["value"] == 2.0
    # "answer: yes" vs "answer: no!": first difference at offset 8
    assert by_metric["first_divergence_char"]["value"] == 8.0


def test_check_runs_fenced_code():
    from adapters.openai_compat import _check

    problem = {
        "prompt": "def add(a, b):\n",
        "entry_point": "add",
        "test": "def check(f):\n    assert f(1, 2) == 3\n",
    }
    chatty = "Here you go:\n```python\n    return a + b\n```\nHope that helps!"
    assert _check(problem, chatty, 10) is True
    assert _check(problem, "definitely not code at all ((((", 10) is False


PATH_B_CONTEXTS = [
    {"domain": "legal", "text": "This contract is governed by law."},
    {"domain": "legal", "text": "The party shall indemnify the other."},
    {"domain": "general", "text": "The quick brown fox jumps over the dog."},
]


def test_capture_reference_writes_bundle(endpoint, tmp_path):
    bundle_out = tmp_path / "ref-b.json"
    records = _adapter().run(
        endpoint, "capture_reference",
        {"model": "stub-model", "contexts": PATH_B_CONTEXTS,
         "bundle_out": str(bundle_out)},
    )
    assert bundle_out.is_file()
    bundle = json.loads(bundle_out.read_text())
    assert bundle["created_by"] == "openai_compat capture_reference (path b)"
    assert bundle["endpoint"] == endpoint
    assert bundle["served"] == "stub-model"
    assert len(bundle["reference"]) == 2
    assert len(bundle["reference"]["legal"]) == 2
    assert len(bundle["reference"]["general"]) == 1
    # every captured context has one non-empty logprob list
    for domain_items in bundle["reference"].values():
        for item in domain_items:
            assert item["logprobs"]
            assert all(isinstance(v, float) for v in item["logprobs"])

    by_metric = {r["metric"]: r for r in records}
    assert set(r["task"] for r in records) == {"capture_reference"}
    assert all(r["adapter"] == "openai_compat" for r in records)
    assert "reference_positions@legal" in by_metric
    assert "reference_positions@general" in by_metric
    assert by_metric["reference_positions@general"]["value"] > 0
    for r in records:
        assert set(r) == set(RECORD_FIELDS)
        assert "UNVERIFIED" in r["protocol"]
        assert f"reference-bundle::{bundle_out}" in r["artifacts"]


def test_capture_reference_requires_contexts(endpoint):
    with pytest.raises(ValueError, match="no usable contexts"):
        _adapter().run(endpoint, "capture_reference", {"model": "stub-model"})


def test_parity_against_identical_serving_is_zero(endpoint, tmp_path):
    bundle_out = tmp_path / "ref.json"
    _adapter().run(
        endpoint, "capture_reference",
        {"model": "stub-model", "contexts": PATH_B_CONTEXTS,
         "bundle_out": str(bundle_out)},
    )
    records = _adapter().run(
        endpoint, "likelihood_parity",
        {"model": "stub-model", "reference_bundle": str(bundle_out)},
    )
    by_metric = {r["metric"]: r for r in records}
    assert all(r["task"] == "likelihood_parity" for r in records)
    for domain in ("legal", "general"):
        assert by_metric[f"kld_mean@{domain}"]["value"] == pytest.approx(0.0)
        assert by_metric[f"kld_max@{domain}"]["value"] == pytest.approx(0.0)
        assert by_metric[f"kld_cvar95@{domain}"]["value"] == pytest.approx(0.0)
        assert by_metric[f"n_tokens@{domain}"]["value"] > 0
    # same serving, same tokens -> every kld_*@domain metric exact 0 besides n
    kld_records = [r for r in records if "kld_" in r["metric"]]
    assert len(kld_records) == 2 * len(KLD_METRICS_NAMES)
    assert all("UNVERIFIED" in r["protocol"] for r in kld_records)


def test_parity_requires_a_reference_bundle(endpoint):
    with pytest.raises(FileNotFoundError, match="reference_bundle"):
        _adapter().run(endpoint, "likelihood_parity", {"model": "stub-model"})


def test_parity_requires_a_path_b_bundle(endpoint, tmp_path):
    not_bundle = tmp_path / "not-b.json"
    not_bundle.write_text(json.dumps({"created_by": "something else"}))
    with pytest.raises(ValueError, match="not a path-b reference bundle"):
        _adapter().run(
            endpoint, "likelihood_parity",
            {"model": "stub-model", "reference_bundle": str(not_bundle)},
        )


def test_parity_rejects_tokenization_drift(endpoint, tmp_path):
    bundle_out = tmp_path / "ref.json"
    _adapter().run(
        endpoint, "capture_reference",
        {"model": "stub-model", "contexts": PATH_B_CONTEXTS,
         "bundle_out": str(bundle_out)},
    )

    # A serving whose /completions tokenizes differently (per-character):
    # per-token pairing across tokenizers is meaningless -> must fail loudly.
    class _Drifted(_Stub):
        def _completions(self, req):
            prompt = req.get("prompt", "")
            tokens = list(prompt)
            self._json({"choices": [{"logprobs": {
                "tokens": tokens,
                "token_logprobs": [-1.0] * len(tokens),
            }}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Drifted)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(EndpointError, match="tokenization changed"):
            _adapter().run(
                f"http://127.0.0.1:{server.server_port}",
                "likelihood_parity",
                {"model": "stub-model", "reference_bundle": str(bundle_out)},
            )
    finally:
        server.shutdown()


def test_parity_detects_serving_change(endpoint, tmp_path):
    bundle_out = tmp_path / "ref.json"
    _adapter().run(
        endpoint, "capture_reference",
        {"model": "stub-model", "contexts": PATH_B_CONTEXTS,
         "bundle_out": str(bundle_out)},
    )

    # Shift every served logprob by a constant: same tokens (no drift), but
    # the distribution the candidate assigns has moved -> kld_mean is the
    # constant offset (ref - cand), nowhere near zero.
    class _Shifted(_Stub):
        logprob_delta = 2.0

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Shifted)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        records = _adapter().run(
            f"http://127.0.0.1:{server.server_port}",
            "likelihood_parity",
            {"model": "stub-model", "reference_bundle": str(bundle_out)},
        )
    finally:
        server.shutdown()

    by_metric = {r["metric"]: r for r in records}
    assert by_metric["kld_max@general"]["value"] == pytest.approx(-2.0)
    assert by_metric["kld_mean@legal"]["value"] == pytest.approx(-2.0)


# --- length_stress (§3.3) --------------------------------------------------


class _StressStub(_Stub):
    """Budget-dependent behavior: clean short stop, then a looping tail."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        budget = int(req.get("max_tokens", 0))
        if budget <= 64:
            text = "a short clean answer."
            finish = "stop"
        else:
            text = "the story begins. " + " ".join(["and then again"] * 40)
            finish = "length"
        self._json({"choices": [{"message": {"content": text},
                                 "finish_reason": finish}]})


def _stress_endpoint(stub):
    server = ThreadingHTTPServer(("127.0.0.1", 0), stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_length_stress_requires_prompt(endpoint):
    with pytest.raises(ValueError, match="requires config\\['prompt'\\]"):
        _adapter().run(endpoint, "length_stress",
                       {"model": "stub-model"})
    with pytest.raises(ValueError, match="non-empty"):
        _adapter().run(endpoint, "length_stress",
                       {"model": "stub-model",
                        "prompts": [{"prompt": "  "}]})
    with pytest.raises(ValueError, match="ladder"):
        _adapter().run(endpoint, "length_stress",
                       {"model": "stub-model",
                        "prompt": "hi", "lengths": []})
    with pytest.raises(ValueError, match="1\\.\\.32768"):
        _adapter().run(endpoint, "length_stress",
                       {"model": "stub-model",
                        "prompt": "hi", "lengths": [32, 99999]})
    with pytest.raises(ValueError, match="caps the sweep"):
        _adapter().run(endpoint, "length_stress",
                       {"model": "stub-model",
                        "prompt": "hi", "lengths": list(range(1, 14))})


def test_length_stress_flags_and_curve():
    server = _stress_endpoint(_StressStub)
    try:
        records = _adapter().run(
            f"http://127.0.0.1:{server.server_port}", "length_stress",
            {"model": "stub-model", "prompt": "tell me a story",
             "lengths": [32, 512]},
        )
    finally:
        server.shutdown()
    for r in records:
        assert set(r) == set(RECORD_FIELDS)
        assert r["task"] == "length_stress"
        assert "UNVERIFIED" in r["protocol"]
    by_metric = {r["metric"]: r for r in records}
    sha = next(iter(by_metric)) .split("@p")[1]
    # short budget: clean natural stop
    assert by_metric[f"stress_ok:L32@p{sha}"]["value"] == 1.0
    assert by_metric[f"stress_loop:L32@p{sha}"]["value"] == 0.0
    assert by_metric[f"stress_completion_chars:L32@p{sha}"]["value"] > 0
    # large budget: looping tail that exhausted the budget
    assert by_metric[f"stress_loop:L512@p{sha}"]["value"] == 1.0
    assert by_metric[f"stress_ok:L512@p{sha}"]["value"] == 0.0
    # curve: half the steps failed, first failure at 512
    assert by_metric[f"stress_failure_rate@p{sha}"]["value"] == \
        pytest.approx(0.5)
    assert by_metric[f"stress_first_failure_at@p{sha}"]["value"] == \
        pytest.approx(512)
    assert by_metric[f"stress_max_clean_chars@p{sha}"]["value"] > 0


def test_length_stress_clean_sweep_sentinel():
    server = _stress_endpoint(_Stub)  # always answers "B", no finish reason
    try:
        records = _adapter().run(
            f"http://127.0.0.1:{server.server_port}", "length_stress",
            {"model": "stub-model", "prompt": "pick B", "lengths": [16, 32]},
        )
    finally:
        server.shutdown()
    by_metric = {r["metric"]: r for r in records}
    sha = next(iter(by_metric)).split("@p")[1]
    assert by_metric[f"stress_failure_rate@p{sha}"]["value"] == \
        pytest.approx(0.0)
    assert by_metric[f"stress_first_failure_at@p{sha}"]["value"] == \
        pytest.approx(-1.0)


def test_length_stress_expect_flag():
    server = _stress_endpoint(_Stub)
    try:
        records = _adapter().run(
            f"http://127.0.0.1:{server.server_port}", "length_stress",
            {"model": "stub-model",
             "prompts": [{"prompt": "pick B", "expect": "B"}],
             "lengths": [16]},
        )
    finally:
        server.shutdown()
    by_metric = {r["metric"]: r for r in records}
    sha = next(iter(by_metric)).split("@p")[1]
    assert by_metric[f"stress_contains_expected:L16@p{sha}"]["value"] == 1.0


# --- perturbation (§3.5) ---------------------------------------------------


class _PerturbStub(_Stub):
    """Baseline answers 'B'; ws_jitter/clause_swap are invisible, so the
    verdict holds. token_sub inserts '___', and the stub flips to 'A'."""

    call_count = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        content = (req.get("messages") or [{}])[0].get("content", "")
        if "___" in content:
            text = "answer: A"
        else:
            text = "answer: B"
        self._json({"choices": [{"message": {"content": text}}],
                    "finish_reason": "stop"})


def _perturb_endpoint(stub):
    server = ThreadingHTTPServer(("127.0.0.1", 0), stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_perturbation_requires_prompts(endpoint):
    with pytest.raises(ValueError, match="config\\['prompt'\\]"):
        _adapter().run(endpoint, "perturbation", {"model": "stub-model"})
    with pytest.raises(ValueError, match="non-empty"):
        _adapter().run(endpoint, "perturbation",
                       {"model": "stub-model", "prompts": []})
    with pytest.raises(ValueError, match="list/tuple"):
        _adapter().run(endpoint, "perturbation",
                       {"model": "stub-model", "prompt": "hi",
                        "perturbations": "ws_jitter"})
    with pytest.raises(ValueError, match="unknown perturbations"):
        _adapter().run(endpoint, "perturbation",
                       {"model": "stub-model", "prompt": "hi",
                        "perturbations": ["nope"]})
    with pytest.raises(ValueError, match="1\\.\\.32768"):
        _adapter().run(endpoint, "perturbation",
                       {"model": "stub-model", "prompt": "x",
                        "max_tokens": 99999})


def test_perturbation_stability_and_agg():
    server = _perturb_endpoint(_PerturbStub)
    try:
        records = _adapter().run(
            f"http://127.0.0.1:{server.server_port}", "perturbation",
            {"model": "stub-model", "prompt": "1 + 1 =? A:2 B:2 C:3 D:4",
             "gold": "B",
             "perturbations": ["ws_jitter", "token_sub", "clause_swap"]},
        )
    finally:
        server.shutdown()
    for r in records:
        assert set(r) == set(RECORD_FIELDS)
        assert r["task"] == "perturbation"
        assert "UNVERIFIED" in r["protocol"]
    by_metric = {r["metric"]: r for r in records}
    sha = next(iter(by_metric)).split("@p")[1]
    # whitespace jitter: stub still answers B -> stable
    assert by_metric[f"perturb_jaccard:Pws_jitter@p{sha}"]["value"] == 1.0
    assert by_metric[f"perturb_verdict_changed:Pws_jitter@p{sha}"]["value"] == 0.0
    # clause swap: unchanged verdict
    assert by_metric[f"perturb_verdict_changed:Pclause_swap@p{sha}"]["value"] == 0.0
    # token substitution flips the verdict
    assert by_metric[f"perturb_verdict_changed:Ptoken_sub@p{sha}"]["value"] == 1.0
    # token_sub stubs returns "answer: A" vs baseline "answer: B" -> 1/3 overlap
    assert by_metric[f"perturb_jaccard:Ptoken_sub@p{sha}"]["value"] == \
        pytest.approx(1 / 3)
    # aggregates over n=1 item
    assert by_metric["perturb_mean_jaccard:Ptoken_sub"]["value"] == \
        pytest.approx(1 / 3)
    assert by_metric["perturb_verdict_change_rate:Ptoken_sub"]["value"] == \
        pytest.approx(1.0)
    assert by_metric["perturb_verdict_change_rate:Pws_jitter"]["value"] == \
        pytest.approx(0.0)


def test_perturbation_aggregate_over_items():
    server = _perturb_endpoint(_PerturbStub)
    try:
        records = _adapter().run(
            f"http://127.0.0.1:{server.server_port}", "perturbation",
            {"model": "stub-model",
             "prompts": [{"prompt": "q1 " + "a" * 8, "gold": "B"},
                         {"prompt": "q1 " + "b" * 8, "gold": "B"}],
             "perturbations": ["token_sub"]},
        )
    finally:
        server.shutdown()
    by_metric = {r["metric"]: r for r in records}
    # both items flip (each contains '___') -> 100% churn, n=2
    assert by_metric["perturb_verdict_change_rate:Ptoken_sub"]["n"] == 2
    assert by_metric["perturb_verdict_change_rate:Ptoken_sub"]["value"] == \
        pytest.approx(1.0)
