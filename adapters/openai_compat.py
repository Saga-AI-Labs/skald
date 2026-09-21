"""OpenAI-compatible endpoint adapter (task: benchmark served models).

Drives any OpenAI-compatible `/chat/completions` endpoint (vLLM, llama.cpp
server, text-generation-webui, ...) for a target model and returns unified
result records (plan §4.2):

- ``mmlu``      -> letter-choice accuracy over MMLU subjects
- ``humaneval`` -> greedy 0-shot pass@1 over HumanEval problems

Interface: ``run(model, task, config) -> records[]`` (scaffold §5). Here
``model`` is the endpoint base URL (e.g. ``http://host:8888/v1``) and the
served model id comes from ``config["model"]`` (default: the server's first
``/models`` entry). Transport is stdlib ``urllib`` only — no new
dependencies. Items come from Hugging Face datasets-server over HTTP by
default, or inline via ``config["mmlu_items"]`` / ``config["humaneval_items"]``
(each ``{"prompt": ..., "gold": ...}`` for mmlu, each HumanEval row dict for
humaneval) for offline/custom batteries.

Identity honesty: a served model exposes no weights to hash, so
``model_checkpoint_sha256`` is the SHA-256 of
``"openai-compat::<base_url>::<model>"`` — a stable *endpoint* identity, not
a weight identity. Every record's ``protocol`` carries the UNVERIFIED marker:
never silently compare these numbers with weighed-in records. The endpoint
URL is also recorded in ``artifacts[]`` as ``endpoint::<url>``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import signal
import socket
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from adapters import RECORD_FIELDS, SuiteAdapter

TASKS = {"mmlu", "humaneval"}

_DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co"
_MMLU_DATASET = "cais/mmlu"
_MMLU_SUBJECTS = [
    "college_mathematics",
    "high_school_statistics",
    "college_biology",
    "logical_fallacies",
]
_HUMANEVAL_DATASET = "openai/openai_humaneval"
_HUMANEVAL_CONFIG = "openai_humaneval"

_DEFAULTS = {
    "mmlu": {"num_fewshot": 5, "max_samples": 100, "max_tokens": 64},
    "humaneval": {"max_samples": 20, "max_tokens": 256},
}
_SEED = 42
_EXEC_TIMEOUT = 10  # per-case watchdog for executing generated HumanEval code

_LETTER_RE = re.compile(r"\b([A-D])\b")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ci(score: float, n: int | None) -> tuple[float | None, float | None]:
    """Wald ~95% confidence interval for a proportion (clamped to [0, 1])."""
    if not n or n <= 0:
        return None, None
    p = max(0.0, min(1.0, float(score)))
    se = math.sqrt(p * (1.0 - p) / n)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


class EndpointError(RuntimeError):
    """A failed or malformed response from the served endpoint."""


class OpenAICompatAdapter(SuiteAdapter):
    """Adapter over an OpenAI-compatible chat-completions endpoint."""

    def run(
        self,
        model: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        config = config or {}
        base = str(model or "").rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise ValueError(
                "openai_compat: model must be the endpoint base URL "
                f"(e.g. 'http://host:8888/v1'); got {model!r}"
            )
        if task not in TASKS:
            raise ValueError(
                f"openai_compat: unsupported task {task!r}; choose from {sorted(TASKS)}"
            )
        client = _Client(
            base,
            api_key=config.get("api_key"),
            timeout=int(config.get("timeout", 300)),
        )
        served = str(config.get("model") or client.default_model())
        client.model = served
        identity = hashlib.sha256(
            f"openai-compat::{base}::{served}".encode("utf-8")
        ).hexdigest()
        handlers = {"mmlu": self._run_mmlu, "humaneval": self._run_humaneval}
        return handlers[task](client, served, base, identity, config)

    # --- task handlers ----------------------------------------------------

    def _run_mmlu(
        self, client: "_Client", served: str, base: str, identity: str, config: dict
    ) -> list[dict[str, Any]]:
        d = _DEFAULTS["mmlu"]
        nf = int(config.get("num_fewshot", d["num_fewshot"]))
        ms = config.get("max_samples", d["max_samples"])
        ms = int(ms) if ms is not None else None
        mt = int(config.get("max_tokens", d["max_tokens"]))
        seed = int(config.get("seed", _SEED))
        subjects = list(config.get("subjects", _MMLU_SUBJECTS))
        items = config.get("mmlu_items")
        if items is None:
            cache_dir = config.get("datasets_cache")
            items = _fetch_mmlu(
                config.get("datasets_server", _DEFAULT_DATASETS_SERVER),
                subjects,
                nf if ms is None else nf + (ms or 0),
                cache_dir,
            )
            coverage = (
                f"over {len(subjects)} MMLU subjects ({','.join(subjects)})"
            )
        else:
            coverage = f"over {len(items)} caller-supplied inline items"
        shots, rest = items[:nf], items[nf:]
        if ms is not None:
            rest = rest[:ms]
        if not rest:
            raise EndpointError("openai_compat: mmlu fetched zero evaluation items")
        preamble = "".join(_mmlu_prompt(s, with_answer=True) for s in shots)
        hits = 0
        for it in rest:
            completion = client.complete(
                preamble + _mmlu_prompt(it, with_answer=False), mt
            )
            # Last match wins: completions often echo the question (whose
            # options contain A-D) before giving the final answer letter.
            hits_here = _LETTER_RE.findall(completion or "")
            if hits_here and hits_here[-1] == it["gold"]:
                hits += 1
        n = len(rest)
        score = hits / n
        ci_low, ci_high = _ci(score, n)
        protocol = (
            f"openai_compat mmlu via {base} model {served}: {nf}-shot "
            f"letter-choice accuracy {coverage}, temperature 0, max_tokens "
            f"{mt}, seed {seed}. UNVERIFIED endpoint identity (model name "
            f"self-reported by server, not a weight hash) — never compare "
            f"with weighed-in records."
        )
        return [
            self._record(
                identity=identity,
                task="mmlu",
                metric="accuracy",
                value=score,
                n=n,
                ci_low=ci_low,
                ci_high=ci_high,
                protocol=protocol,
                seed=seed,
                base=base,
            )
        ]

    def _run_humaneval(
        self, client: "_Client", served: str, base: str, identity: str, config: dict
    ) -> list[dict[str, Any]]:
        d = _DEFAULTS["humaneval"]
        ms = config.get("max_samples", d["max_samples"])
        ms = int(ms) if ms is not None else None
        mt = int(config.get("max_tokens", d["max_tokens"]))
        seed = int(config.get("seed", _SEED))
        exec_timeout = int(config.get("exec_timeout", _EXEC_TIMEOUT))
        items = config.get("humaneval_items")
        if items is None:
            items = _fetch_humaneval(
                config.get("datasets_server", _DEFAULT_DATASETS_SERVER),
                ms if ms is not None else 164,
                config.get("datasets_cache"),
            )
            coverage = "over openai/openai_humaneval"
        else:
            coverage = f"over {len(items)} caller-supplied inline problems"
        if ms is not None:
            items = items[:ms]
        if not items:
            raise EndpointError("openai_compat: humaneval fetched zero problems")
        passed = 0
        for p in items:
            code = client.complete(p["prompt"], mt)
            if _check(p, code or "", exec_timeout):
                passed += 1
        n = len(items)
        score = passed / n
        ci_low, ci_high = _ci(score, n)
        protocol = (
            f"openai_compat humaneval via {base} model {served}: greedy 0-shot "
            f"pass@1 {coverage}, max_tokens {mt}, seed "
            f"{seed}, {exec_timeout}s/case exec watchdog. UNVERIFIED endpoint "
            f"identity (model name self-reported by server, not a weight "
            f"hash) — never compare with weighed-in records."
        )
        return [
            self._record(
                identity=identity,
                task="humaneval",
                metric="pass_at_1",
                value=score,
                n=n,
                ci_low=ci_low,
                ci_high=ci_high,
                protocol=protocol,
                seed=seed,
                base=base,
            )
        ]

    # --- records ----------------------------------------------------------

    def _record(self, *, identity: str, task: str, metric: str, value: float,
                n: int | None, ci_low: float | None, ci_high: float | None,
                protocol: str, seed: int, base: str) -> dict[str, Any]:
        record = {
            "model_checkpoint_sha256": identity,
            "adapter": "openai_compat",
            "suite": "openai_compat",
            "task": task,
            "metric": metric,
            "value": value,
            "n": n,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "protocol": protocol,
            "created_at": _now(),
            "host": socket.gethostname(),
            "script_sha256": _self_sha256(),
            "seed": seed,
            "artifacts": [f"endpoint::{base}"],
        }
        assert set(record) == set(RECORD_FIELDS), set(record) ^ set(RECORD_FIELDS)
        return record


def _self_sha256() -> str:
    h = hashlib.sha256()
    with open(__file__, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mmlu_prompt(item: dict, *, with_answer: bool) -> str:
    lines = [f"Question: {item['prompt']}"]
    for letter, choice in zip("ABCD", item["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    if with_answer:
        lines[-1] += f" {item['gold']}"
    return "\n".join(lines) + "\n\n"


class _Client:
    """Minimal OpenAI-compatible chat client over stdlib urllib."""

    def __init__(self, base: str, *, api_key: str | None, timeout: int) -> None:
        self.base = base
        self.api_key = api_key
        self.timeout = timeout
        self.model = ""

    def _request(self, path: str, payload: dict | None) -> Any:
        url = self.base + path
        data = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise EndpointError(
                f"openai_compat: {url} returned HTTP {exc.code} "
                f"({exc.read().decode('utf-8', 'replace')[:300]})"
            ) from exc
        except urllib.error.URLError as exc:
            raise EndpointError(
                f"openai_compat: cannot reach {url}: {exc.reason}"
            ) from exc
        except (json.JSONDecodeError, TimeoutError) as exc:
            raise EndpointError(
                f"openai_compat: bad response from {url}: {exc}"
            ) from exc

    def default_model(self) -> str:
        body = self._request("/models", None)
        try:
            return body["data"][0]["id"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(
                f"openai_compat: /models returned no usable model id: {body!r}"
            ) from exc

    def complete(self, prompt: str, max_tokens: int) -> str:
        body = self._request(
            "/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
        )
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(
                f"openai_compat: /chat/completions returned no usable "
                f"content: {body!r}"
            ) from exc
        # Reasoning models may leave content null while thinking fits the
        # budget: fall back through the known reasoning fields (vLLM uses
        # "reasoning", others "reasoning_content").
        for field in ("content", "reasoning_content", "reasoning"):
            text = message.get(field)
            if isinstance(text, str) and text.strip():
                return text
        raise EndpointError(
            f"openai_compat: /chat/completions returned empty content and "
            f"no reasoning text: {body!r}"
        )


def _fetch_rows(server: str, dataset: str, config: str, n: int,
                 cache_dir: str | Path | None = None) -> list[dict]:
    """Fetch *n* test rows, paginated: datasets-server caps length at 100.

    Successful fetches accumulate in a local JSON cache
    (``<root>/.skald/dataset_cache/``, overridable), so a won fetch is
    never repeated and later runs survive throttling or offline boxes.
    Test splits are static; the cache carries no TTL by design.
    """
    import time

    cached = _read_cache(cache_dir, dataset, config)
    rows: list[dict] = list(cached)
    offset = len(rows)
    while len(rows) < n:
        length = min(100, n - len(rows))
        url = (
            f"{server}/rows?dataset={urllib.parse.quote(dataset, safe='')}"
            f"&config={urllib.parse.quote(config, safe='')}"
            f"&split=test&offset={offset}&length={length}"
        )
        page = _fetch_page(url)
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        _write_cache(cache_dir, dataset, config, rows)
        time.sleep(2.0)  # politeness gap; bursts get 429s otherwise
    return rows


def _cache_path(cache_dir: str | Path | None, dataset: str, config: str) -> Path:
    root = Path(cache_dir) if cache_dir else (
        Path(__file__).resolve().parent.parent / ".skald" / "dataset_cache"
    )
    safe = lambda s: re.sub(r"[^0-9A-Za-z._-]+", "-", s)
    return root / safe(dataset) / f"{safe(config)}.json"


def _read_cache(cache_dir: str | Path | None, dataset: str, config: str) -> list[dict]:
    path = _cache_path(cache_dir, dataset, config)
    if not path.is_file():
        return []
    try:
        body = json.loads(path.read_text())
    except (ValueError, OSError):
        return []
    return body if isinstance(body, list) else []


def _write_cache(cache_dir: str | Path | None, dataset: str,
                 config: str, rows: list[dict]) -> None:
    path = _cache_path(cache_dir, dataset, config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows))
        import os

        os.replace(tmp, path)
    except OSError:
        pass  # cache is best-effort; the fetch itself already succeeded


def _fetch_page(url: str, retries: int = 8) -> list[dict]:
    import time

    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            # datasets-server rate-limits bursts (429) and occasionally
            # sheds load (502/503/504): back off and retry, honoring
            # Retry-After when the server names a wait.
            if exc.code in (429, 502, 503, 504) and attempt < retries:
                wait = exc.headers.get("Retry-After")
                delay = float(wait) if wait is not None else 2.0 * (2 ** attempt)
                time.sleep(min(delay, 120.0))
                attempt += 1
                continue
            raise EndpointError(
                f"openai_compat: datasets-server {url} returned HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise EndpointError(
                f"openai_compat: cannot reach datasets-server {url}: {exc.reason} "
                "(offline? pass inline mmlu_items/humaneval_items via config)"
            ) from exc
    try:
        return [r["row"] for r in body["rows"]]
    except (KeyError, TypeError) as exc:
        raise EndpointError(
            f"openai_compat: datasets-server returned no rows: {body!r}"
        ) from exc


def _fetch_mmlu(server: str, subjects: list[str], per_subject: int,
               cache_dir: str | Path | None = None) -> list[dict]:
    items = []
    for subject in subjects:
        for row in _fetch_rows(server, _MMLU_DATASET, subject, per_subject,
                               cache_dir):
            letters = "ABCD"
            gold = letters[int(row["answer"])]
            items.append(
                {"prompt": row["question"], "choices": list(row["choices"]), "gold": gold}
            )
    return items


def _fetch_humaneval(server: str, n: int,
                     cache_dir: str | Path | None = None) -> list[dict]:
    return _fetch_rows(server, _HUMANEVAL_DATASET, _HUMANEVAL_CONFIG, n,
                       cache_dir)


class _Timeout(Exception):
    pass


_FENCE_RE = re.compile(r"```(\w*)\s*\n(.*?)```", re.S)


def _extract_code(gen_code: str) -> str:
    """Pull executable code out of a chatty completion.

    Preference order: a ```python-fenced block, then any fenced block,
    then the raw text with leading blank lines dropped (preserving the
    first line's indentation — stripping all leading whitespace dedents
    the body out of the function and SyntaxErrors). Callers still drop a
    leading ``def`` line when the model echoed the whole function.
    """
    fences = _FENCE_RE.findall(gen_code or "")
    if fences:
        python_first = [body for tag, body in fences if tag == "python"]
        chosen = (python_first or [fences[0][1]])[0]
    else:
        chosen = gen_code or ""
    lines = chosen.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _check(problem: dict, gen_code: str, exec_timeout: int) -> bool:
    """Execute the generated completion against the problem's tests."""
    def _alarm(_sig, _frm) -> None:
        raise _Timeout()

    ns: dict = {}
    # Drop leading blank lines only: stripping all leading whitespace would
    # dedent the first code line out of the function body (SyntaxError).
    body = _extract_code(gen_code)
    if body.startswith("def "):
        body = "\n".join(body.split("\n")[1:]).lstrip("\n")
    if body.startswith("def "):
        body = "\n".join(body.split("\n")[1:]).lstrip("\n")
    full = problem["prompt"] + "\n" + body + "\n" + problem["test"]
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, exec_timeout)
    try:
        exec(full, ns)  # noqa: S102 - the HumanEval protocol is code execution
        ns["check"](ns[problem["entry_point"]])
        return True  # HumanEval check() returns None; no exception means pass
    except _Timeout:
        return False
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run an endpoint benchmark, persist, read back.

    Usage:
        python -m adapters.openai_compat <base_url> <task>
            [--config '{"model": "<served-id>", "max_samples": 20}']
    """
    import argparse
    import sys

    import store

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="endpoint base URL, e.g. http://host:8888/v1")
    ap.add_argument("task", choices=sorted(TASKS), help="benchmark task")
    ap.add_argument("--config", default="{}", help="JSON config")
    args = ap.parse_args(argv)

    config = json.loads(args.config)
    records = OpenAICompatAdapter().run(args.model, args.task, config)
    if not records:
        print("openai_compat: no records produced", file=sys.stderr)
        return 2

    store.put(records)
    key = {
        "adapter": "openai_compat",
        "task": args.task,
        "model_checkpoint_sha256": records[0]["model_checkpoint_sha256"],
    }
    back = store.query(key)
    print(f"persisted {len(records)} openai_compat records; queried back {len(back)} matching")
    for r in back:
        print(f"  {r['task']}:{r['metric']} = {r['value']} (n={r['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
