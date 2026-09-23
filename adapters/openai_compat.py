"""OpenAI-compatible endpoint adapter (task: benchmark served models).

Drives any OpenAI-compatible `/chat/completions` endpoint (vLLM, llama.cpp
server, text-generation-webui, ...) for a target model and returns unified
result records (plan §4.2):

- ``mmlu``        -> letter-choice accuracy over MMLU subjects
- ``humaneval``   -> greedy 0-shot pass@1 over HumanEval problems
- ``determinism`` -> repeat-sampling gate (proposal §3.1): distinct-output
  rate + first-divergence offset for one prompt at temperature 0
- ``capture_reference`` -> Path B §3.2 serving-path reference capture:
  per-token served logprobs over sealed contexts, into a JSON bundle
- ``likelihood_parity`` -> Path B §3.2 parity against a reference bundle:
  per-domain tokenwise KLD (``kld_*@domain``); weights+kernels as served
- ``length_stress`` -> completion-length sweep (proposal §3.3): fixed
  prompt(s), capped max_tokens ladder; per-step failure-mode flags
  (ok/refusal/loop/truncated/empty) plus per-prompt curve records
  (failure rate, first failure budget, max clean length)
- ``perturbation`` -> prompt-surface sensitivity (proposal §3.5): per item,
  each of three small stated perturbations (whitespace jitter, one-token
  substitution, first-two-clause swap) applied to the same prompt; report
  output stability as token-set Jaccard vs the unperturbed completion and
  whether the answer verdict changed.

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
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from adapters import RECORD_FIELDS, SuiteAdapter
from adapters.length_stats import (
    STRESS_CURVE_METRICS,
    STRESS_STEP_METRICS,
    classify,
    summarize_curve,
)
from adapters.likelihood_stats import KLD_METRICS, summarize_kld
from adapters import local_bench
from adapters.perturb_stats import (
    PERTURB_AGG_METRICS,
    PERTURB_STEP_METRICS,
    PERTURBATIONS,
    apply_perturbation,
    jaccard,
    verdict,
)
from store.runtime import runtime_digest

TASKS = {"mmlu", "humaneval", "determinism", "capture_reference",
         "likelihood_parity", "length_stress", "perturbation",
         "gsm8k", "simpleqa", "tool_use"}

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
    # Measured on GLM-5.3-Flash-EXL3 (6 items, 5-shot, seed 42), sweeping the
    # budget: 64 -> 100% starved, silent chance-level score. 512 -> still 100%
    # starved. 1024 and 2048 -> 0% starved. 4096 -> starvation RETURNS (0.33,
    # then 0.17 on a repeat), because the server runs with
    # --max-num-batched-tokens 2048: a request asking for more completion
    # budget than the batched-token cap starves unpredictably.
    # So the usable window is [1024, 2048]; 1024 is the safe pick, under the cap.
    "mmlu": {"num_fewshot": 5, "max_samples": 100, "max_tokens": 1024},
    "humaneval": {"max_samples": 20, "max_tokens": 256},
    # Free-form math: no few-shot prefix (the corpus prompts are zero-shot by
    # construction), and a long budget because a worked solution is long.
    # Budgets are sized for a THINKING endpoint: this server spends ~1100
    # tokens on deliberation before it emits content, so a small max_tokens
    # yields empty content on every item and the task scores 0.0 for a reason
    # that has nothing to do with capability. budget_starved makes that visible
    # rather than leaving it to be misread as ignorance.
    "gsm8k": {"max_samples": 100, "max_tokens": 2048},
    "simpleqa": {"max_samples": 100, "max_tokens": 2048},
    "tool_use": {"max_samples": 20, "max_tokens": 256, "steps": 4,
                 "max_tool_calls": 24},
}
_SEED = 42
_EXEC_TIMEOUT = 10  # per-case watchdog for executing generated HumanEval code

_LETTER_RE = re.compile(r"\b([A-D])\b")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_arg(raw: Any, key: str) -> Any:
    """Pull one field out of a tool call's JSON ``arguments`` string.

    Tolerates the two shapes servers actually emit: a JSON object string, or
    an already-decoded dict. Returns ``None`` rather than raising when the
    payload is unusable, so one malformed tool call degrades to a failed item
    instead of aborting the run.
    """
    if isinstance(raw, dict):
        return raw.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    return obj.get(key) if isinstance(obj, dict) else None


def _ci(score: float, n: int | None) -> tuple[float | None, float | None]:
    """Wald ~95% confidence interval for a proportion (clamped to [0, 1])."""
    if not n or n <= 0:
        return None, None
    p = max(0.0, min(1.0, float(score)))
    se = math.sqrt(p * (1.0 - p) / n)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def _first_divergence(a: str, b: str) -> int:
    """Earliest character offset where *a* and *b* differ (min length if
    one is a strict prefix of the other). Inputs are assumed distinct."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


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
        client.echo_max_tokens = int(config.get("echo_max_tokens", 0))
        served = str(config.get("model") or client.default_model())
        client.model = served
        identity = hashlib.sha256(
            f"openai-compat::{base}::{served}".encode("utf-8")
        ).hexdigest()
        # Serving-path facet (runtime-manifest spec §6): endpoint-observed
        # facts plus the local interpreter. None-served degrades to a
        # local-only digest, never a guess.
        runtime = runtime_digest(served=client.server_facts())
        handlers = {
            "mmlu": self._run_mmlu,
            "humaneval": self._run_humaneval,
            "determinism": self._run_determinism,
            "capture_reference": self._run_capture_reference,
            "likelihood_parity": self._run_parity,
            "length_stress": self._run_length_stress,
            "perturbation": self._run_perturbation,
            "gsm8k": self._run_gsm8k,
            "simpleqa": self._run_simpleqa,
            "tool_use": self._run_tool_use,
        }
        return handlers[task](client, served, base, identity, runtime, config)

    # --- task handlers ----------------------------------------------------

    def _run_mmlu(
        self, client: "_Client", served: str, base: str, identity: str,
        runtime: str | None, config: dict
    ) -> list[dict[str, Any]]:
        d = _DEFAULTS["mmlu"]
        nf = int(config.get("num_fewshot", d["num_fewshot"]))
        ms = config.get("max_samples", d["max_samples"])
        ms = int(ms) if ms is not None else None
        mt = int(config.get("max_tokens", d["max_tokens"]))
        seed = int(config.get("seed", _SEED))
        subjects = list(config.get("subjects",
                        local_bench.SUBJECT_SETS.get(
                            str(config.get("subject_set", "default")),
                            _MMLU_SUBJECTS)))
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
        answerable = 0
        starved = 0
        for it in rest:
            # Score ``content`` ONLY, never the reasoning fallback: a thinking
            # model that exhausts its budget leaves content empty and puts a
            # restatement of the question -- options included -- into
            # reasoning. Letting _LETTER_RE see that text manufactures answers
            # the model never gave.
            content, _reasoning, finish = client.answer_and_reasoning(
                preamble + _mmlu_prompt(it, with_answer=False), mt
            )
            if finish == "length":
                starved += 1
            # Last match wins: content often echoes the question (whose options
            # contain A-D) before giving the final answer letter.
            hits_here = _LETTER_RE.findall(content or "")
            if hits_here:
                answerable += 1
                if hits_here[-1] == it["gold"]:
                    hits += 1
        n = len(rest)
        # Accuracy is over ALL items, so truncation depresses it instead of
        # being hidden by scoring only the items that happened to answer.
        score = hits / n
        ci_low, ci_high = _ci(score, n)
        protocol = (
            f"openai_compat mmlu via {base} model {served}: {nf}-shot "
            f"letter-choice accuracy {coverage}, temperature 0, max_tokens "
            f"{mt}, seed {seed}. Scored on content only; reasoning is never "
            f"scanned for an answer letter. accuracy is over all {n} items, so "
            f"a truncated item counts as wrong rather than being dropped. "
            f"UNVERIFIED endpoint identity (model name "
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
                runtime=runtime,
            ),
            self._record(
                identity=identity,
                task="mmlu",
                metric="answerable",
                value=(answerable / n) if n else 0.0,
                n=n,
                ci_low=None,
                ci_high=None,
                protocol=(
                    protocol + f". answerable {answerable}/{n} produced an "
                    f"answer letter at all."
                ),
                seed=seed,
                base=base,
                runtime=runtime,
            ),
            self._record(
                identity=identity,
                task="mmlu",
                metric="budget_starved",
                value=(starved / n) if n else 0.0,
                n=n,
                ci_low=None,
                ci_high=None,
                protocol=(
                    protocol + f". budget_starved {starved}/{n} hit "
                    f"finish_reason=length. This is a BUDGET result: a starved "
                    f"item is not evidence of model ignorance, so accuracy "
                    f"alone must not be read as capability when it is high."
                ),
                seed=seed,
                base=base,
                runtime=runtime,
            ),
        ]

    def _run_humaneval(
        self, client: "_Client", served: str, base: str, identity: str,
        runtime: str | None, config: dict
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
        starved = 0
        no_code = 0
        for p in items:
            # Score ``content`` ONLY. ``complete()`` falls back onto the
            # reasoning fields, so on a thinking model that exhausts its
            # budget it hands back private deliberation -- and _check() will
            # happily extract and execute a code block quoted *inside* that
            # deliberation. That is not a pass@1 of the model's answer, so it
            # must not share the number with one.
            content, _reasoning, finish = client.answer_and_reasoning(
                p["prompt"], mt
            )
            if finish == "length":
                starved += 1
            if not (content or "").strip():
                no_code += 1
            if _check(p, content or "", exec_timeout):
                passed += 1
        n = len(items)
        score = passed / n
        ci_low, ci_high = _ci(score, n)
        protocol = (
            f"openai_compat humaneval via {base} model {served}: greedy 0-shot "
            f"pass@1 {coverage}, max_tokens {mt}, seed "
            f"{seed}, {exec_timeout}s/case exec watchdog. Scored on the public "
            f"answer field only (reasoning is never executed as code). "
            f"UNVERIFIED endpoint "
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
                runtime=runtime,
            ),
            # Truncation and empty answers are reported as their own buckets,
            # not folded silently into a lower pass@1: a model that never
            # emits code and a model that is merely capped look identical in
            # the headline number.
            self._record(
                identity=identity,
                task="humaneval",
                metric="budget_starved",
                value=starved / n,
                n=n,
                ci_low=None,
                ci_high=None,
                protocol=protocol,
                seed=seed,
                base=base,
                runtime=runtime,
            ),
            self._record(
                identity=identity,
                task="humaneval",
                metric="no_code",
                value=no_code / n,
                n=n,
                ci_low=None,
                ci_high=None,
                protocol=protocol,
                seed=seed,
                base=base,
                runtime=runtime,
            ),
        ]

    def _run_determinism(
        self, client: "_Client", served: str, base: str, identity: str,
        runtime: str | None, config: dict
    ) -> list[dict[str, Any]]:
        """Determinism probe (proposal §3.1): a gate, not a metric.

        Same prompt, ``temperature=0`` (the client's fixed setting), N
        repeats. Reports the distinct-output rate plus the first-divergence
        character offset (the earliest position where any two completions
        differ). A checkpoint failing this probe — ``reproducible == 0`` —
        should have its capability records treated as non-reproducible
        samples rather than bare scalars; the probe gates the other
        measurements, it does not sit beside them.
        """
        prompt = config.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "openai_compat: determinism requires config['prompt'] "
                "(one non-empty prompt string; run once per prompt family)"
            )
        repeats = int(config.get("repeats", 12))
        if repeats < 2:
            raise ValueError(
                f"openai_compat: determinism requires repeats >= 2; got {repeats}"
            )
        mt = int(config.get("max_tokens", _DEFAULTS["mmlu"]["max_tokens"]))
        seed = int(config.get("seed", _SEED))
        outputs = [client.complete(prompt, mt) or "" for _ in range(repeats)]
        distinct = sorted(set(outputs))
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        protocol = (
            f"openai_compat determinism via {base} model {served}: prompt "
            f"sha {prompt_sha}, {repeats} repeats at temperature 0, "
            f"max_tokens {mt}, seed {seed}. Compares the full returned text "
            f"(content, or reasoning when a truncated thinking model leaves "
            f"content empty) — this is a GATE on repeatability, not an "
            f"answer-accuracy measure. UNVERIFIED endpoint identity "
            f"(model name self-reported by server, not a weight hash) — "
            f"never compare with weighed-in records."
        )
        records = [
            self._record(
                identity=identity,
                task="determinism",
                metric="distinct_output_rate",
                value=len(distinct) / repeats,
                n=repeats,
                ci_low=None,
                ci_high=None,
                protocol=protocol,
                seed=seed,
                base=base,
                runtime=runtime,
            ),
            self._record(
                identity=identity,
                task="determinism",
                metric="reproducible",
                value=1.0 if len(distinct) == 1 else 0.0,
                n=repeats,
                ci_low=None,
                ci_high=None,
                protocol=protocol,
                seed=seed,
                base=base,
                runtime=runtime,
            ),
            self._record(
                identity=identity,
                task="determinism",
                metric="num_distinct_outputs",
                value=float(len(distinct)),
                n=repeats,
                ci_low=None,
                ci_high=None,
                protocol=protocol,
                seed=seed,
                base=base,
                runtime=runtime,
            ),
        ]
        if len(distinct) > 1:
            # Earliest character offset where any two completions differ;
            # min length when one output is a strict prefix of another.
            first = min(
                _first_divergence(a, b) for i, a in enumerate(distinct)
                for b in distinct[i + 1:]
            )
            records.append(
                self._record(
                    identity=identity,
                    task="determinism",
                    metric="first_divergence_char",
                    value=float(first),
                    n=repeats,
                    ci_low=None,
                    ci_high=None,
                    protocol=protocol,
                    seed=seed,
                    base=base,
                    runtime=runtime,
                )
            )
        return records

    def _run_length_stress(
        self, client: "_Client", served: str, base: str, identity: str,
        runtime: str | None, config: dict
    ) -> list[dict[str, Any]]:
        """Completion-length sweep (proposal §3.3): failure shape, not score.

        Fixed prompt(s), capped ``max_tokens`` ladder. Each step gets one
        failure-mode label (``classify``: empty/refusal/loop/truncated/ok);
        per-prompt curve records report the failure rate, the first failing
        budget (-1 when the sweep is clean), and the longest clean
        completion. "A model that degrades gracefully and a model that
        loops are different objects" — the flags tell them apart where a
        single accuracy number cannot.

        Optional per-prompt ``expect`` substring adds a
        ``stress_contains_expected`` flag per step (accuracy against
        completion length, when the operator supplies a gold string).
        """
        prompts = config.get("prompts")
        if prompts is None:
            single = config.get("prompt")
            if not isinstance(single, str) or not single.strip():
                raise ValueError(
                    "openai_compat: length_stress requires config['prompt'] "
                    "(one non-empty prompt string) or config['prompts'] "
                    "([{prompt, expect?}...]; run once per prompt family)"
                )
            prompts = [{"prompt": single}]
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(
                "openai_compat: length_stress requires a non-empty "
                "config['prompts'] list"
            )
        for item in prompts:
            if not isinstance(item, dict) or not isinstance(
                    item.get("prompt"), str) or not item["prompt"].strip():
                raise ValueError(
                    "openai_compat: every prompts entry needs a non-empty "
                    f"'prompt' string; got {item!r}"
                )
        lengths = config.get("lengths", [32, 128, 512, 2048])
        if not isinstance(lengths, list) or not lengths:
            raise ValueError(
                "openai_compat: length_stress requires a non-empty "
                "config['lengths'] ladder"
            )
        budgets = [int(v) for v in lengths]
        if any(b < 1 or b > 32768 for b in budgets):
            raise ValueError(
                "openai_compat: length_stress budgets must lie in "
                f"1..32768; got {budgets}"
            )
        if len(budgets) > 12:
            raise ValueError(
                "openai_compat: length_stress caps the sweep at 12 steps "
                f"(generation budget); got {len(budgets)}"
            )
        seed = int(config.get("seed", _SEED))
        records = []
        for item in prompts:
            prompt = item["prompt"]
            expect = item.get("expect")
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            modes, chars = [], []
            for budget in budgets:
                text, finish = client.complete_full(prompt, budget)
                mode = classify(text or "", finish)
                modes.append(mode)
                chars.append(len(text or ""))
                flags = {m: 1.0 if mode == m else 0.0
                         for m in ("ok", "refusal", "loop",
                                   "truncated", "empty")}
                step_protocol = (
                    f"openai_compat length_stress via {base} model {served}: "
                    f"prompt sha {prompt_sha}, max_tokens {budget}, "
                    f"temperature 0, seed {seed}; completion "
                    f"{len(text or '')} chars, finish_reason "
                    f"{finish!r} -> mode {mode}. UNVERIFIED endpoint "
                    f"identity (model name self-reported by server, not a "
                    f"weight hash) — never compare with weighed-in records."
                )
                for name in STRESS_STEP_METRICS:
                    value = (float(chars[-1]) if name == "stress_completion_chars"
                             else flags[name[len("stress_"):]])
                    records.append(self._record(
                        identity=identity, task="length_stress",
                        metric=f"{name}:L{budget}@p{prompt_sha}",
                        value=value, n=1, ci_low=None, ci_high=None,
                        protocol=step_protocol, seed=seed, base=base,
                        runtime=runtime,
                    ))
                if expect is not None:
                    records.append(self._record(
                        identity=identity, task="length_stress",
                        metric=f"stress_contains_expected:L{budget}@p{prompt_sha}",
                        value=1.0 if expect in (text or "") else 0.0,
                        n=1, ci_low=None, ci_high=None,
                        protocol=step_protocol + f" expect {expect!r}.",
                        seed=seed, base=base, runtime=runtime,
                    ))
            curve = summarize_curve(modes, chars, budgets)
            curve_protocol = (
                f"openai_compat length_stress curve via {base} model "
                f"{served}: prompt sha {prompt_sha} over budgets {budgets}; "
                f"modes {[f'{b}:{m}' for b, m in zip(budgets, modes)]}; "
                f"first_failure_at -1 means the whole sweep stayed ok, "
                f"max_clean_chars -1 means no step was clean. UNVERIFIED "
                f"endpoint identity — never compare with weighed-in records."
            )
            for name in STRESS_CURVE_METRICS:
                records.append(self._record(
                    identity=identity, task="length_stress",
                    metric=f"{name}@p{prompt_sha}", value=curve[name],
                    n=len(budgets), ci_low=None, ci_high=None,
                    protocol=curve_protocol, seed=seed, base=base,
                    runtime=runtime,
                ))
        return records

    def _run_perturbation(
        self, client: "_Client", served: str, base: str, identity: str,
        runtime: str | None, config: dict
    ) -> list[dict[str, Any]]:
        """Prompt-surface sensitivity (proposal §3.5): stability over input
        perturbations.

        For each prompt, first get the unperturbed completion, then apply
        each of a stated set of small deterministic perturbations to the
        *input* (whitespace jitter, one-token substitution, first-two-clause
        swap) and re-ask. Reports, per (item, perturbation): token-set
        Jaccard between the two completions, and whether the verdict moved.
        ``verdict`` is the last A–D letter when the item carries a ``gold``
        (letter-choice items like mmlu), else the raw completion — so
        ``perturb_verdict_changed`` always means "the answer moved".

        "Catches a model that has learned the surface of the task rather
        than the task" (proposal §3.5): a model whose output is stable
        under a perturbed input that changes nothing semantically is
        reading the words, not the intent.

        Config: ``prompts`` (or single ``prompt``), optional per-item
        ``gold``, ``perturbations`` (default: all of
        ``ws_jitter, token_sub, clause_swap``), ``max_tokens``.
        """
        prompts = config.get("prompts")
        if prompts is None:
            single = config.get("prompt")
            if not isinstance(single, str) or not single.strip():
                raise ValueError(
                    "openai_compat: perturbation requires config['prompt'] "
                    "(one non-empty prompt string) or config['prompts'] "
                    "([{prompt, gold?}...]; run once per item)"
                )
            prompts = [{"prompt": single}]
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(
                "openai_compat: perturbation requires a non-empty "
                "config['prompts'] list"
            )
        for item in prompts:
            if not isinstance(item, dict) or not isinstance(
                    item.get("prompt"), str) or not item["prompt"].strip():
                raise ValueError(
                    "openai_compat: every prompts entry needs a non-empty "
                    f"'prompt' string; got {item!r}"
                )
        raw_chosen = config.get("perturbations", PERTURBATIONS)
        if isinstance(raw_chosen, str) or not isinstance(raw_chosen, (list, tuple)):
            raise ValueError(
                "openai_compat: config['perturbations'] must be a list/tuple "
                "of operator names, not a bare string"
            )
        chosen = list(raw_chosen)
        if not chosen:
            raise ValueError(
                "openai_compat: perturbation requires a non-empty "
                "config['perturbations'] list"
            )
        unknown = set(chosen) - set(PERTURBATIONS)
        if unknown:
            raise ValueError(
                f"openai_compat: unknown perturbations {sorted(unknown)}; "
                f"choose from {PERTURBATIONS}"
            )
        if any(not isinstance(p, str) for p in chosen):
            raise ValueError(
                "openai_compat: config['perturbations'] must be a list of "
                "operator names"
            )
        mt = int(config.get("max_tokens", _DEFAULTS["mmlu"]["max_tokens"]))
        if mt < 1 or mt > 32768:
            raise ValueError(
                f"openai_compat: perturbation max_tokens must lie in "
                f"1..32768; got {mt}"
            )
        seed = int(config.get("seed", _SEED))
        records = []
        agg = {name: {"jaccs": [], "changed": []} for name in chosen}
        for item in prompts:
            prompt = item["prompt"]
            gold = item.get("gold")
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            baseline = client.complete(prompt, mt) or ""
            base_verdict = verdict(baseline, gold)
            for name in chosen:
                perturbed_prompt = apply_perturbation(name, prompt)
                if perturbed_prompt == prompt:
                    completion = baseline
                else:
                    completion = client.complete(perturbed_prompt, mt) or ""
                sim = jaccard(baseline, completion)
                moved = 1.0 if verdict(completion, gold) != base_verdict else 0.0
                agg[name]["jaccs"].append(sim)
                agg[name]["changed"].append(moved)
                step_protocol = (
                    f"openai_compat perturbation via {base} model {served}: "
                    f"item {prompt_sha}, perturbation {name}, max_tokens {mt}, "
                    f"temperature 0, seed {seed}; prompt perturbed "
                    f"deterministically (token_sub at fixed middle position, "
                    f"ws_jitter permutes whitespace, clause_swap reorders the "
                    f"first two sentences) — semantic content unchanged; "
                    f"jaccard {sim:.4f} over token sets, verdict "
                    f"{'moved' if moved else 'stable'} "
                    f"(gold {gold!r}). UNVERIFIED endpoint identity — never "
                    f"compare with weighed-in records."
                )
                for metric in PERTURB_STEP_METRICS:
                    value = sim if metric == "perturb_jaccard" else moved
                    records.append(self._record(
                        identity=identity, task="perturbation",
                        metric=f"{metric}:P{name}@p{prompt_sha}",
                        value=value, n=1, ci_low=None, ci_high=None,
                        protocol=step_protocol, seed=seed, base=base,
                        runtime=runtime,
                    ))
        n = len(prompts)
        for name in chosen:
            mean_j = sum(agg[name]["jaccs"]) / n
            rate = sum(agg[name]["changed"]) / n
            agg_protocol = (
                f"openai_compat perturbation aggregate via {base} model "
                f"{served}: perturbation {name} over {n} item(s); mean "
                f"token-set Jaccard and verdict-change rate across items. "
                f"UNVERIFIED endpoint identity — never compare with "
                f"weighed-in records."
            )
            lo_j, hi_j = _ci(mean_j, n)
            lo_r, hi_r = _ci(rate, n)
            records.append(self._record(
                identity=identity, task="perturbation",
                metric=f"perturb_mean_jaccard:P{name}",
                value=mean_j, n=n, ci_low=lo_j, ci_high=hi_j,
                protocol=agg_protocol, seed=seed, base=base, runtime=runtime,
            ))
            records.append(self._record(
                identity=identity, task="perturbation",
                metric=f"perturb_verdict_change_rate:P{name}",
                value=rate, n=n, ci_low=lo_r, ci_high=hi_r,
                protocol=agg_protocol, seed=seed, base=base, runtime=runtime,
            ))
        return records

    def _run_capture_reference(
        self, client: "_Client", served: str, base: str, identity: str,
        runtime: str | None, config: dict
    ) -> list[dict[str, Any]]:
        """Path B reference capture (proposal §3.2): served logprobs.

        For each sealed context, records the *served* per-token logprobs
        from ``/completions`` (``echo:true, logprobs:0``) into a JSON
        reference bundle, and emits ``reference_positions`` per domain. The
        bundle is the serving-path reference floor: this measures
        weights *and* kernels as actually served — the A/B difference is
        exactly the point (Path A in ``bdh_likelihood`` measures weights
        only, offline).
        """
        from adapters.bdh_likelihood import load_contexts

        bundle_path = Path(config.get("bundle_out") or
                           f"reference-{served}-{_now().replace(':', '')}.json")
        domains = load_contexts(config)
        bundle: dict[str, Any] = {
            "created_by": "openai_compat capture_reference (path b)",
            "created_at": _now(),
            "reference_identity": identity,
            "endpoint": base,
            "served": served,
            "runtime_sha256": runtime,
            "reference": {},
        }
        records: list[dict[str, Any]] = []
        seed = int(config.get("seed", _SEED))
        for dom in domains:
            domain = dom["domain"]
            texts_out = []
            for text in dom["texts"]:
                logprobs = client.prompt_logprobs(text)
                texts_out.append({"text": text, "logprobs": logprobs})
            bundle["reference"][domain] = texts_out
            total = sum(len(t["logprobs"]) for t in texts_out)
            bundle["positions_per_domain"] = bundle.get(
                "positions_per_domain", {})
            bundle["positions_per_domain"][domain] = total
            records.append(self._record(
                identity=identity, task="capture_reference",
                metric=f"reference_positions@{domain}", value=float(total),
                n=total, ci_low=None, ci_high=None,
                protocol=(
                    f"openai_compat capture-reference (path b) via {base} "
                    f"model {served}: served per-token logprobs over "
                    f"{len(dom['texts'])} sealed contexts in domain "
                    f"{domain!r} ({total} positions), weights+kernels as "
                    f"served. UNVERIFIED endpoint identity — never compare "
                    f"with weighed-in records."
                ),
                seed=seed, base=base, runtime=runtime,
            ))
        bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
        for r in records:
            r["artifacts"] = [
                f"endpoint::{base}", f"reference-bundle::{bundle_path}",
            ]
        return records

    def _run_parity(
        self, client: "_Client", served: str, base: str, identity: str,
        runtime: str | None, config: dict
    ) -> list[dict[str, Any]]:
        """Path B parity (proposal §3.2): served reference vs served candidate.

        Re-queries the serving path over the sealed contexts captured by
        ``capture_reference`` and reports the per-domain *tokenwise* KLD
        (realized-token ``ref_lp - cand_lp``) — metrics are ``kld_*@domain``,
        deliberately distinct from Path A's full-vocab ``kl_*@domain`` so
        the two paths are never conflated in the store.
        """
        bundle_path = Path(config.get("reference_bundle") or "")
        if not bundle_path.is_file():
            raise FileNotFoundError(
                "openai_compat: likelihood_parity requires "
                "config['reference_bundle'] pointing at a "
                "capture_reference bundle (got "
                f"{bundle_path})"
            )
        bundle = json.loads(bundle_path.read_text())
        if bundle.get("created_by") != "openai_compat capture_reference (path b)":
            raise ValueError(
                f"openai_compat: {bundle_path} is not a path-b reference "
                f"bundle (created_by={bundle.get('created_by')!r})"
            )
        seed = int(config.get("seed", _SEED))
        kld_by_domain: dict[str, list[float]] = {}
        total_by_domain: dict[str, int] = {}
        for domain, texts_out in sorted(bundle["reference"].items()):
            domain_total = 0
            for item in texts_out:
                cand = client.prompt_logprobs(item["text"])
                ref = item["logprobs"]
                if len(cand) != len(ref):
                    raise EndpointError(
                        f"openai_compat: serving tokenization changed for "
                        f"domain {domain!r}: reference bundle has "
                        f"{len(ref)} logprobs for this context but the "
                        f"current serving returned {len(cand)}. Per-token "
                        f"pairing is meaningless across tokenizers — "
                        f"re-capture with capture_reference."
                    )
                domain_total += len(ref)
                kld_by_domain.setdefault(domain, []).extend(
                    r - c for r, c in zip(ref, cand)
                )
            total_by_domain[domain] = domain_total
        records: list[dict[str, Any]] = []
        for domain in sorted(kld_by_domain):
            summary = summarize_kld(kld_by_domain[domain])
            protocol = (
                f"openai_compat likelihood-parity (path b) via {base} model "
                f"{served}: tokenwise KLD per realized token, reference "
                f"bundle {bundle_path.name} (captured "
                f"{bundle['created_at']} from served "
                f"{bundle['served']}@{bundle['endpoint']}), candidate "
                f"{served}@{base}; domain {domain!r}, {summary['n']} "
                f"positions. Measures weights+kernels as served — the "
                f"A/B difference is the finding (Path A = weights only, "
                f"offline). UNVERIFIED endpoint identity — never compare "
                f"with weighed-in records."
            )
            for metric in KLD_METRICS:
                records.append(self._record(
                    identity=identity, task="likelihood_parity",
                    metric=f"{metric}@{domain}",
                    value=summary[metric], n=summary["n"],
                    ci_low=None, ci_high=None, protocol=protocol,
                    seed=seed, base=base, runtime=runtime,
                ))
            records.append(self._record(
                identity=identity, task="likelihood_parity",
                metric=f"n_tokens@{domain}",
                value=float(total_by_domain[domain]),
                n=total_by_domain[domain], ci_low=None, ci_high=None,
                protocol=protocol, seed=seed, base=base, runtime=runtime,
            ))
        if not records:
            raise RuntimeError(
                "openai_compat: parity bundle has no contexts — refusing "
                "to store an empty run"
            )
        return records

    # --- records ----------------------------------------------------------

    # --- local-corpus tasks (adapters.local_bench) -------------------------

    def _run_gsm8k(
        self, client: "_Client", served: str, base: str, identity: str,
        runtime: str | None, config: dict
    ) -> list[dict[str, Any]]:
        """Free-form grade-school math: worked solutions, numeric answer.

        The point of this task is that it is *not* multiple choice. MMLU's
        math subjects are four-way MC, where guessing scores 0.25 and a model
        can select the right number without computing anything; here the model
        must produce the number, so 0.0 means it could not do the arithmetic.
        """
        d = _DEFAULTS["gsm8k"]
        ms = config.get("max_samples", d["max_samples"])
        ms = int(ms) if ms is not None else None
        mt = int(config.get("max_tokens", d["max_tokens"]))
        seed = int(config.get("seed", _SEED))
        items = config.get("gsm8k_items") or local_bench.load_jsonl(
            local_bench.GSM8K_FILE, data_dir=config.get("data_dir"))
        if ms is not None:
            items = items[:ms]
        if not items:
            raise EndpointError("openai_compat: gsm8k has zero evaluation items")
        prompts = [f"{it['prompt']}\n\nWork it out step by step, then give "
                   f"the answer on a final line as '#### <number>'."
                   for it in items]
        # content only, never the reasoning fallback: scoring a thinking
        # model's private deliberation against a gold is a different measurement
        pairs = [client.answer_and_reasoning(pr, mt) for pr in prompts]
        s = local_bench.score_math(items, [p for p, _r, _f in pairs],
                                   [r for _p, r, _f in pairs])
        protocol = (
            f"openai_compat gsm8k via {base} model {served}: free-form "
            f"grade-school math, zero-shot, numeric answer extracted by "
            f"####/last-number rule, temperature 0, max_tokens {mt}, seed "
            f"{seed}. unparseable answers are a separate NO_ANSWER bucket and "
            f"count as failures (accuracy {s['correct']}/{s['n']}, "
            f"unanswered {s['unanswered']}). UNVERIFIED endpoint identity "
            f"(model name self-reported by server, not a weight hash) -- "
            f"never compare with weighed-in records."
        )
        recs = [self._record(
            identity=identity, task="gsm8k", metric="accuracy",
            value=s["accuracy"], n=s["n"], ci_low=ci, ci_high=ch,
            protocol=protocol, seed=seed, base=base, runtime=runtime,
        ) for ci, ch in [_ci(s["accuracy"], s["n"])]]
        recs.append(self._record(
            identity=identity, task="gsm8k", metric="answerable",
            value=s["answerable"], n=s["n"], ci_low=None, ci_high=None,
            protocol=protocol, seed=seed, base=base, runtime=runtime,
        ))
        recs.append(self._record(
            identity=identity, task="gsm8k", metric="budget_starved",
            value=(s["budget_starved"] / s["n"]) if s["n"] else 0.0,
            n=s["n"], ci_low=None, ci_high=None,
            protocol=(protocol + f". budget_starved {s['budget_starved']}/{s['n']}"
                      f": no answer emitted because the token budget was spent"
                      f" thinking (max_tokens {mt}); a budget setting, not a"
                      f" capability result"),
            seed=seed, base=base, runtime=runtime,
        ))
        return recs

    def _run_simpleqa(
        self, client: "_Client", served: str, base: str, identity: str,
        runtime: str | None, config: dict
    ) -> list[dict[str, Any]]:
        """Short-form factuality. Softer than the other metrics -- read it as such.

        Gold is free text, so scoring is normalised containment. It
        under-counts correct answers phrased differently from the gold and
        over-counts a model that quotes the gold string back without meaning
        it. It therefore has its own metric name and must not be averaged with
        the exact-match tasks.
        """
        d = _DEFAULTS["simpleqa"]
        ms = config.get("max_samples", d["max_samples"])
        ms = int(ms) if ms is not None else None
        mt = int(config.get("max_tokens", d["max_tokens"]))
        seed = int(config.get("seed", _SEED))
        items = config.get("simpleqa_items") or local_bench.load_jsonl(
            local_bench.SIMPLEQA_FILE, data_dir=config.get("data_dir"))
        if ms is not None:
            items = items[:ms]
        if not items:
            raise EndpointError("openai_compat: simpleqa has zero evaluation items")
        prompts = [f"Answer with a short factual phrase, no explanation."
                   f"\n\n{it['prompt']}" for it in items]
        # content only: a thinking model's deliberation must never be scored
        # as its answer
        pairs = [client.answer_and_reasoning(pr, mt) for pr in prompts]
        s = local_bench.score_factuality(items, [p for p, _r, _f in pairs],
                                         [r for _p, r, _f in pairs])
        protocol = (
            f"openai_compat simpleqa via {base} model {served}: short-form "
            f"factuality, normalised-containment scoring (SOFTER than exact "
            f"match: under-counts paraphrase, over-counts quoted gold), "
            f"temperature 0, max_tokens {mt}, seed {seed}. UNVERIFIED endpoint "
            f"identity (model name self-reported by server, not a weight hash) "
            f"-- never compare with weighed-in records."
        )
        recs = [self._record(
            identity=identity, task="simpleqa", metric="accuracy",
            value=s["accuracy"], n=s["n"], ci_low=ci, ci_high=ch,
            protocol=protocol, seed=seed, base=base, runtime=runtime,
        ) for ci, ch in [_ci(s["accuracy"], s["n"])]]
        recs.append(self._record(
            identity=identity, task="simpleqa", metric="budget_starved",
            value=(s["budget_starved"] / s["n"]) if s["n"] else 0.0,
            n=s["n"], ci_low=None, ci_high=None,
            protocol=(protocol + f". budget_starved {s['budget_starved']}/{s['n']}"
                      f": no answer emitted, the token budget was spent thinking"
                      f" (max_tokens {mt}). A zero here with a high"
                      f" budget_starved is a BUDGET result, not evidence that"
                      f" the model lacks the fact"),
            seed=seed, base=base, runtime=runtime,
        ))
        return recs

    def _run_tool_use(
        self, client: "_Client", served: str, base: str, identity: str,
        runtime: str | None, config: dict
    ) -> list[dict[str, Any]]:
        """Multi-step tool use: plan N calc calls, read observations, terminate.

        Constructed, not published -- there is no agentic corpus on disk and
        the datasets server mirrors only the two datasets already in use. What
        keeps it honest is that the gold is COMPUTED: the chain's result is
        obtained by evaluating the same expression the model is asked to work
        through, so correctness is checkable with no authored answer key, and
        ``seed`` reproduces the item exactly. The measured skill is not
        arithmetic (the tool does that) but whether the model issues the calls,
        carries each observation into the next, and stops.
        """
        d = _DEFAULTS["tool_use"]
        ms = int(config.get("max_samples", d["max_samples"]))
        mt = int(config.get("max_tokens", d["max_tokens"]))
        steps = int(config.get("steps", d["steps"]))
        budget = int(config.get("max_tool_calls", d["max_tool_calls"]))
        seed = int(config.get("seed", _SEED))
        if ms <= 0 or steps < 1:
            raise ValueError("openai_compat: tool_use needs max_samples>0 and steps>=1")

        solved = 0
        calls_total = 0
        no_terminate = 0
        for i in range(ms):
            prompt, _terms, gold = local_bench.build_task(seed + i, steps)
            msgs: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            calls = 0
            text = ""
            for _ in range(budget):
                turn = client.chat_tools(msgs, local_bench.CALC_SCHEMA, mt)
                calls += 1
                text = turn.get("content") or ""
                tcs = turn.get("tool_calls") or []
                if not tcs:
                    break
                msgs.append({"role": "assistant", "content": text or None,
                            "tool_calls": tcs})
                for tc in tcs:
                    fn = (tc.get("function") or {})
                    expr = _json_arg(fn.get("arguments"), "expr")
                    obs = local_bench.run_calc(expr if isinstance(expr, str) else "")
                    msgs.append({"role": "tool", "tool_call_id": tc.get("id"),
                                "content": str(obs)})
            calls_total += calls
            if calls == 0:
                no_terminate += 1
            pred = local_bench.predict_math(text)
            if pred is not None and abs(pred - gold) <= 1e-6:
                solved += 1
        rate = solved / ms
        protocol = (
            f"openai_compat tool_use via {base} model {served}: {steps}-step "
            f"arithmetic chains driven through a calc tool, {ms} items, gold "
            f"COMPUTED from the chain (not authored), seed {seed} reproduces "
            f"items, tool-call budget {budget}/item, temperature 0. Measures "
            f"multi-step tool driving and termination, not arithmetic. "
            f"CONSTRUCTED task -- no published agentic corpus was available; "
            f"treat as indicative, not as a named benchmark. UNVERIFIED "
            f"endpoint identity (model name self-reported by server, not a "
            f"weight hash) -- never compare with weighed-in records."
        )
        recs = [self._record(
            identity=identity, task="tool_use", metric="solved",
            value=rate, n=ms, ci_low=ci, ci_high=ch,
            protocol=protocol, seed=seed, base=base, runtime=runtime,
        ) for ci, ch in [_ci(rate, ms)]]
        recs.append(self._record(
            identity=identity, task="tool_use", metric="tool_calls_per_item",
            value=calls_total / ms, n=ms, ci_low=None, ci_high=None,
            protocol=protocol, seed=seed, base=base, runtime=runtime))
        recs.append(self._record(
            identity=identity, task="tool_use", metric="no_tool_call",
            value=no_terminate / ms, n=ms, ci_low=None, ci_high=None,
            protocol=protocol, seed=seed, base=base, runtime=runtime))
        return recs

    def _record(self, *, identity: str, task: str, metric: str, value: float,
                n: int | None, ci_low: float | None, ci_high: float | None,
                protocol: str, seed: int, base: str,
                runtime: str | None) -> dict[str, Any]:
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
            "runtime_sha256": runtime,
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
        self.last_headers: dict[str, str] = {}
        self.echo_max_tokens = 0  # 0 = echo prompt only; >0 drops that many
        # trailing generated tokens before treating the rest as the prompt.

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
                self.last_headers = dict(resp.headers.items())
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

    def server_facts(self) -> dict[str, Any] | None:
        """Endpoint-observed facts for the runtime manifest (spec §1).

        Returns ``{"models": [...], "server": ...}`` from ``GET /models``,
        or ``None`` when the endpoint cannot be introspected — the caller
        then degrades to a local-only digest, never a guess.
        """
        try:
            body = self._request("/models", None)
            models = [m["id"] for m in body["data"] if isinstance(m, dict) and m.get("id")]
        except (EndpointError, KeyError, TypeError):
            return None
        if not models:
            return None
        facts: dict[str, Any] = {"models": sorted(set(models))}
        lowered = {k.lower(): v for k, v in self.last_headers.items()}
        if lowered.get("server"):
            facts["server"] = lowered["server"]
        return facts

    def chat_tools(self, messages: list[dict], tools: list[dict],
                   max_tokens: int) -> dict[str, Any]:
        """One chat turn that may return tool calls instead of an answer.

        Returns the assistant ``message`` object verbatim (``content`` and/or
        ``tool_calls``). Kept separate from ``complete`` because a tool turn is
        a different contract: the caller must feed observations back and the
        turn may legitimately carry no content at all.
        """
        body = self._request(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "tools": tools,
                "tool_choice": "auto",
            },
        )
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(
                f"openai_compat: /chat/completions returned no message: {body!r}"
            ) from exc
        if not isinstance(message, dict):
            raise EndpointError(
                f"openai_compat: tool turn message is not an object: {message!r}"
            )
        return message

    def complete(self, prompt: str, max_tokens: int) -> str:
        """Completion text, falling back onto the reasoning fields.

        Preserved deliberately for existing callers (humaneval, perturbation,
        determinism): a reasoning model that leaves ``content`` null still
        yields *some* text here. Scorers must not use this -- they use
        ``answer_and_reasoning`` and score ``content`` only, because measuring
        a model's private deliberation against a gold answer is a different
        measurement from measuring its public answer.
        """
        message = self._chat_message(prompt, max_tokens)
        for field in ("content", "reasoning_content", "reasoning"):
            text = message.get(field)
            if isinstance(text, str) and text.strip():
                return text
        raise EndpointError(
            f"openai_compat: /chat/completions returned empty content and "
            f"no reasoning text: {message!r}"
        )

    def answer_and_reasoning(self, prompt: str, max_tokens: int
                             ) -> tuple[str, str, str | None]:
        """Return ``(content, reasoning, finish_reason)`` for one prompt.

        Kept separate from ``complete`` on purpose. ``complete`` falls back
        onto the reasoning fields, which is right for a caller that just wants
        *some* text but is wrong for a scorer: a thinking model that exhausts
        its budget leaves ``content`` empty and puts everything in
        ``reasoning``, so scoring the fallback measures the model's private
        deliberation against the gold instead of its public answer. Those are
        different measurements and must not share a number.

        The three states stay distinct -- empty content with empty reasoning
        (no output), empty content with non-empty reasoning (the budget went
        into thinking and no answer was ever produced), and non-empty content
        (a real answer). Only the last is a capability result; collapsing the
        first two is what would make a footprint lie.
        """
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
            choice = body["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(
                f"openai_compat: /chat/completions returned no usable "
                f"content: {body!r}"
            ) from exc
        message = choice.get("message") or {}
        content = message.get("content")
        reasoning = (message.get("reasoning_content")
                     or message.get("reasoning"))
        content = content if isinstance(content, str) else ""
        reasoning = reasoning if isinstance(reasoning, str) else ""
        reason = choice.get("finish_reason") or choice.get("stop_reason")
        return content, reasoning, (reason if isinstance(reason, str) else None)

    def _chat_message(self, prompt: str, max_tokens: int) -> dict:
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
        return message or {}

    def complete_full(self, prompt: str, max_tokens: int
                      ) -> tuple[str, str | None]:
        """Completion text plus the server's ``finish_reason`` (§3.3).

        ``complete`` drops the finish reason; length-stress needs it to
        tell a budget-exhausting loop (``length``) from a natural stop.
        A missing/empty reason degrades to ``None`` — the classifiers
        treat ``None`` as "not the budget", never as a guess. An empty
        completion is returned as ``""`` (a failure mode, §3.3 ``empty``),
        not raised.
        """
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
            choice = body["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(
                f"openai_compat: /chat/completions returned no usable "
                f"content: {body!r}"
            ) from exc
        message = choice.get("message") or {}
        text = ""
        for field in ("content", "reasoning_content", "reasoning"):
            candidate = message.get(field)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate
                break
        finish = choice.get("finish_reason")
        return text, finish if isinstance(finish, str) else None

    def prompt_logprobs(self, prompt: str) -> list[float]:
        """Per-position log probabilities (nats) over ``prompt``'s tokens.

        Path B (§3.2): the serving-path realized-token distribution. Uses
        the legacy ``/completions`` endpoint with ``echo: true`` and
        ``logprobs: 0`` — the one place OpenAI-compatible servers put
        *prompt* logprobs (``/chat/completions`` only returns completion
        token logprobs, which is a different, generated-token distribution).

        Returns one logprob per prompt token (the probability the *served*
        model assigned to the token that actually appears), or raises
        ``EndpointError`` when the endpoint does not expose prompt logprobs.
        """
        body = self._request(
            "/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": self.echo_max_tokens,
                "echo": True,
                "logprobs": 0,
            },
        )
        try:
            slot = body["choices"][0]["logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(
                f"openai_compat: /completions returned no logprobs slot: "
                f"{body!r}"
            ) from exc
        # 'echo' prepends the prompt tokens, then any generated tokens
        # (none when echo_max_tokens=0; strip them otherwise).
        tokens = slot.get("tokens")
        token_logprobs = slot.get("token_logprobs")
        if not isinstance(tokens, list) or not isinstance(token_logprobs, list):
            raise EndpointError(
                f"openai_compat: /completions logprobs malformed: {slot!r}"
            )
        if len(tokens) != len(token_logprobs):
            raise EndpointError(
                f"openai_compat: /completions tokens/logprobs misaligned "
                f"({len(tokens)} vs {len(token_logprobs)})"
            )
        n_gen = min(self.echo_max_tokens, len(tokens))
        prompt_logprobs = token_logprobs[: len(tokens) - n_gen] if n_gen else token_logprobs
        values: list[float] = []
        for lp in prompt_logprobs:
            if lp is None:
                raise EndpointError(
                    f"openai_compat: /completions returned a null token "
                    f"logprob (logprobs: 0 unsupported?): {token_logprobs!r}"
                )
            values.append(float(lp))
        if not values:
            raise EndpointError(
                f"openai_compat: /completions echoed no prompt tokens "
                f"(empty prompt?): {body!r}"
            )
        return values


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
    """Execute the generated completion against the problem's tests.

    Runs in a *subprocess*, for two reasons that are both load-bearing:

    1. Thread safety. The previous implementation armed ``SIGALRM`` via
       ``signal.setitimer``, which raises ``ValueError: signal only works in
       main thread of the main interpreter`` whenever the adapter is driven
       from a worker thread -- which is exactly how the API runs jobs. So
       HumanEval was unusable through ``/api/v1/run_benchmark``, the documented
       way to drive skald. A subprocess owns its own main thread, so the alarm
       is legal there, and this matches how ``saga.py`` already does it.
    2. Isolation. The HumanEval protocol is code execution, and the code being
       executed is whatever the model emitted. It should not run inside the
       API process, where it could take the whole hub down with it.

    Semantics are unchanged: pass means ``check()`` returned without raising,
    anything else -- exception, timeout, non-zero exit -- is a failure.
    """
    def _alarm(_sig, _frm) -> None:
        raise _Timeout()

    # Drop leading blank lines only: stripping all leading whitespace would
    # dedent the first code line out of the function body (SyntaxError).
    body = _extract_code(gen_code)
    while body and not body.split("\n")[0].strip():
        body = body.split("\n", 1)[1] if "\n" in body else ""
    while body.startswith("def "):
        body = "\n".join(body.split("\n")[1:]).lstrip("\n")
    full = problem["prompt"] + "\n" + body + "\n" + problem["test"] + "\n"
    child = (
        "import signal, sys\n"
        "class _TO(Exception): pass\n"
        "def _h(s, f): raise _TO()\n"
        "signal.signal(signal.SIGALRM, _h)\n"
        f"signal.setitimer(signal.ITIMER_REAL, {int(exec_timeout)})\n"
        "ns = {}\n"
        f"src = {full!r}\n"
        "try:\n"
        "    exec(compile(src, '<humaneval>', 'exec'), ns)\n"
        f"    ns['check'](ns[{problem['entry_point']!r}])\n"
        "except BaseException:\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", child],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # wall-clock backstop beyond the child's own alarm, in case the
            # model's code blocks somewhere the alarm cannot interrupt
            timeout=exec_timeout + 10,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return False


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
