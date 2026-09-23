"""Local-corpus benchmarks for ``openai_compat``: free-form math, factuality,
and multi-step tool use.

Why this module exists
----------------------
``openai_compat`` shipped with two tasks. ``mmlu`` is four-way multiple choice
drawn from four MMLU subjects; ``humaneval`` is code generation. Against the
five capability axes a footprint is supposed to cover, that leaves *coding*
covered and everything else thin or absent:

    coding           COVERED   humaneval (execution-verified pass@1)
    math             PARTIAL   MMLU multiple choice only -- no free-form items
    logic/reasoning  PARTIAL  one MMLU subject (logical_fallacies), MC
    general knowledge THIN     one MMLU subject (college_biology); no factuality
    agentic          ABSENT    no multi-step / tool task existed anywhere

Multiple-choice math is not a math test: guessing scores 25%, and a model can
pick the right number without computing anything. The three tasks here close
those holes using corpora that are already on disk under ``vendor/``
(``PIN.md`` records their provenance), so no new network dependency is added --
the datasets server only mirrors the two datasets this adapter already used,
so nothing fetchable from the hub could have been used anyway.

Scoring discipline borrowed from ``vendor/pi50_eval/harness/score_capability.py``
-- including its hardest-won lesson, quoted at the bottom of this file where it
governs the ``NO_ANSWER`` bucket.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# --- corpora --------------------------------------------------------------
# Vendored, not fetched. See vendor/pi50_eval/PIN.md for authorship and hashes.
_DATA = Path(__file__).resolve().parent.parent / "vendor" / "pi50_eval" / "data"

GSM8K_FILE = "gsm8k.jsonl"
SIMPLEQA_FILE = "simpleqa.jsonl"

# MMLU subject sets. The default stays the historical four so an existing run
# is never silently widened; a footprint asks for one of these by name.
SUBJECT_SETS: dict[str, list[str]] = {
    "default": [
        "college_mathematics",
        "high_school_statistics",
        "college_biology",
        "logical_fallacies",
    ],
    # Reasoning-heavy: logic and mathematics, no life sciences.
    "reasoning": [
        "formal_logic",
        "logical_fallacies",
        "college_mathematics",
        "high_school_mathematics",
        "high_school_statistics",
    ],
    # Broad knowledge across sciences and humanities, for the knowledge axis.
    "knowledge": [
        "college_biology",
        "college_chemistry",
        "astronomy",
        "college_medicine",
        "conceptual_physics",
        "high_school_us_history",
        "us_presidential_history",
        "professional_law",
    ],
    # Everything a footprint should ask for when it wants one number per axis.
    "footprint": [
        "college_mathematics",
        "high_school_mathematics",
        "formal_logic",
        "logical_fallacies",
        "college_biology",
        "college_computer_science",
    ],
}


class CorpusError(RuntimeError):
    """A vendored corpus is missing or malformed -- a *setup* failure."""


def load_jsonl(name: str, *, data_dir: str | Path | None = None) -> list[dict]:
    """Read a vendored ``.jsonl`` corpus. Raises rather than returning [].

    A missing file is a configuration error and must not degrade into an empty
    list: an empty corpus scores 0.0, which is indistinguishable from a model
    that answered every item wrong.
    """
    root = Path(data_dir) if data_dir else _DATA
    path = root / name
    if not path.is_file():
        raise CorpusError(f"local_bench: corpus not found: {path}")
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            raise CorpusError(f"local_bench: {path}:{lineno} not JSON: {exc}") from exc
        if isinstance(obj, dict) and "prompt" in obj:
            rows.append(obj)
    if not rows:
        raise CorpusError(f"local_bench: {path} held no usable rows")
    return rows


# --- free-form math (GSM8K) ------------------------------------------------
# An answer we cannot read is NOT an answer we read wrong. The vendored
# harness learned this the hard way -- its own comment records it:
#   "an answer we cannot read is NOT also an answer we read incorrectly:
#    counting it in both buckets made the tallies exceed n, which is how this
#    line got written."
# Unparseable responses are therefore a third bucket, never folded into WRONG,
# and never folded into the denominator of a silent pass.

_NUM_RE = re.compile(r"[-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?")
_HASH_RE = re.compile(r"####\s*([-+]?\s*\$?[\d,]*\.?\d+)")
_FINAL_ANS_RE = re.compile(r"(?:final answer|answer)\s*(?:is|:)?\s*([-+]?\$?[\d,]+\.?\d*)",
                           re.IGNORECASE)


def _as_number(s: str | None) -> float | None:
    if s is None:
        return None
    t = str(s).strip().replace(",", "").replace("$", "").rstrip(".").strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def predict_math(text: str) -> float | None:
    """Extract the answer a free-form math response commits to.

    Preference order is deliberate: an explicit ``####`` line is what the
    prompt asks for, so it wins; then a labelled "final answer"; failing both,
    the last number written anywhere -- last, because in a worked solution the
    final number is the answer and the first is usually a given.
    """
    if not text:
        return None
    m = _HASH_RE.findall(text)
    if m:
        v = _as_number(m[-1])
        if v is not None:
            return v
    m = _FINAL_ANS_RE.findall(text)
    if m:
        v = _as_number(m[-1])
        if v is not None:
            return v
    for line in reversed([l for l in text.strip().splitlines() if l.strip()]):
        nums = _NUM_RE.findall(line)
        if nums:
            return _as_number(nums[-1])
    return None


def score_math(items: list[dict], completions: list[str],
               reasonings: list[str] | None = None) -> dict[str, Any]:
    """Score free-form math. Returns correct/wrong/unanswered and accuracy.

    ``accuracy`` is correct / n over *all* items: an unanswered item is a
    failure, not an exclusion. Dropping unparseable answers would let a model
    that never formats an answer score 1.0 on zero answers.

    ``reasonings``, when given, is the model's private deliberation per item.
    It is never scored -- it exists only so ``budget_starved`` can say how many
    of the unanswered items were unanswered because a thinking model spent the
    whole token budget thinking and never emitted an answer. That is a
    different failure from getting it wrong, and reporting the accuracy without
    it would attribute a budget setting to the model's arithmetic.
    """
    n = len(items)
    correct = unanswered = starved = 0
    misses: list[dict] = []
    for idx, (item, text) in enumerate(zip(items, completions)):
        gold = _as_number(item.get("gold"))
        pred = predict_math(text or "")
        if gold is None or pred is None:
            unanswered += 1
            thought = ""
            if reasonings is not None and idx < len(reasonings):
                thought = (reasonings[idx] or "").strip()
            if thought:
                starved += 1
            misses.append({"id": item.get("id"), "why": "NO_ANSWER",
                          "budget_starved": bool(thought),
                          "gold": item.get("gold"), "saw": (text or "")[:120]})
            continue
        if abs(pred - gold) <= 1e-6:
            correct += 1
        else:
            misses.append({"id": item.get("id"), "why": "WRONG",
                          "gold": gold, "pred": pred, "saw": (text or "")[:120]})
    wrong = n - correct - unanswered
    return {"n": n, "correct": correct, "wrong": wrong,
            "unanswered": unanswered, "budget_starved": starved,
            "accuracy": (correct / n) if n else 0.0,
            "answerable": ((n - unanswered) / n) if n else 0.0,
            "misses": misses[:25]}


# --- factuality (SimpleQA) -------------------------------------------------
# Gold here is free text, so scoring is containment, not equality. That makes
# this metric *softer* than the others and it must be read as such: it
# under-counts correct answers that are phrased differently and over-counts a
# confident model that quotes the gold string back without meaning it. It is
# reported under its own metric name for exactly that reason.

_WS_RE = re.compile(r"[^a-z0-9 ]")
_SPACES_RE = re.compile(r"\s+")
# Below this many normalised characters the answer is too short to be matched
# against the gold by containment without earning credit by accident.
_MIN_REVERSE_MATCH = 3


def _norm(s: str) -> str:
    return _SPACES_RE.sub(" ", _WS_RE.sub("", str(s).lower())).strip()


def score_factuality(items: list[dict], completions: list[str],
                     reasonings: list[str] | None = None) -> dict[str, Any]:
    """Score short-form factual answers by normalised containment.

    ``reasonings`` is never scored; it only populates ``budget_starved`` so a
    zero caused by a thinking model spending the whole budget thinking is
    readable as a budget setting rather than mistaken for ignorance.

    Containment runs one way for the gold (gold appears in the answer) and the
    reverse direction is guarded by a minimum length: without it a one-character
    reply is a substring of almost any gold and earns unearned credit.
    """
    n = len(items)
    correct = unanswered = starved = 0
    misses: list[dict] = []
    for idx, (item, text) in enumerate(zip(items, completions)):
        accepted = [_norm(item.get("gold") or item.get("gold_text") or "")]
        # gold_alts is a list when present. The corpus stores the literal string
        # "None" for "no alternates", which must not be iterated character by
        # character -- that would accept any answer containing 'N', 'o' or 'e'.
        alts = item.get("gold_alts")
        if isinstance(alts, (list, tuple)):
            accepted += [_norm(a) for a in alts]
        accepted = [a for a in accepted if a]

        got = _norm(text or "")
        if not got:
            unanswered += 1
            thought = ""
            if reasonings is not None and idx < len(reasonings):
                thought = (reasonings[idx] or "").strip()
            if thought:
                starved += 1
            misses.append({"id": item.get("id"), "why": "NO_ANSWER",
                          "budget_starved": bool(thought)})
            continue
        hit = any(gold in got for gold in accepted) or any(
            len(got) >= _MIN_REVERSE_MATCH and got == gold for gold in accepted)
        if hit:
            correct += 1
        else:
            misses.append({"id": item.get("id"), "why": "WRONG",
                          "gold": (accepted[0] if accepted else "")[:60],
                          "saw": got[:120]})
    return {"n": n, "correct": correct, "wrong": n - correct - unanswered,
            "unanswered": unanswered, "budget_starved": starved,
            "accuracy": (correct / n) if n else 0.0,
            "misses": misses[:25]}


# --- multi-step tool use ---------------------------------------------------
# The agentic axis. There is no published agentic corpus on disk and the
# datasets server mirrors only the two datasets this adapter already used, so
# this task is *constructed*. What makes it defensible rather than theatre is
# that the gold is COMPUTED, not authored: every item is a chain of arithmetic
# whose result is obtained by evaluating the same expression, so correctness is
# checkable without anyone writing down an answer key. The task is not whether
# the model can do arithmetic -- it is whether it can plan N tool calls, read
# the observations, and terminate with the accumulated result.

CALC_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "calc",
        "description": "Evaluate one arithmetic expression over +,-,*,/,// and parentheses.",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string", "description": "e.g. (3+4)*2"}},
            "required": ["expr"],
        },
    },
}]

_SAFE_EVAL = re.compile(r"^[\d\s\.\+\-\*\/\(\)]+$")


_MAX_MAGNITUDE = 1e30


def run_calc(expr: str) -> float | str:
    """Execute one arithmetic expression. Returns an error string, never raises.

    The model supplies ``expr``, so this is untrusted input and the gate is
    load-bearing. Three separate reasons it is not just a regex plus ``eval``:

    * ``**`` is rejected outright. It is a legal arithmetic operator and would
      pass a character whitelist, but ``2**99999999`` is a denial of service on
      the serving host -- and the resulting integer is so large that even
      *stringifying* it raises, outside any try block here. Exponentiation buys
      nothing this task needs, so it is refused rather than bounded.
    * the result magnitude is capped, so a chain of ``*`` steps cannot run away
      into an unprintable number either;
    * evaluation happens with no builtins, so a generated expression cannot
      reach ``__import__``/``open``/``getattr`` even if it got past the gate.
    """
    if not _SAFE_EVAL.match(expr or ""):
        return f"error: illegal expression {expr!r}"
    if "**" in expr:
        return "error: exponentiation is not supported"
    try:
        value = eval(expr, {"__builtins__": None}, {})  # noqa: S307 -- gated above
    except (ZeroDivisionError, SyntaxError, ValueError, OverflowError, MemoryError) as exc:
        return f"error: {type(exc).__name__}: {str(exc)[:120]}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "error: not a number"
    try:
        if abs(float(value)) > _MAX_MAGNITUDE:
            return f"error: magnitude {value!r} exceeds {_MAX_MAGNITUDE:.0e}"
    except (OverflowError, ValueError):
        return "error: magnitude out of range"
    return value


def build_task(seed: int, steps: int) -> tuple[str, list[dict], float]:
    """Return (prompt, terms, gold) for one multi-step tool-use item."""
    import random

    rng = random.Random(f"tooluse-{seed}")
    start = float(rng.randint(3, 40))
    terms: list[dict] = []
    acc = start
    for i in range(steps):
        op = rng.choice(["+", "-", "*"])
        operand = rng.randint(2, 12) if op != "*" else rng.randint(2, 5)
        acc = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
               "*": lambda a, b: a * b}[op](acc, operand)
        terms.append({"step": i + 1, "op": op, "operand": operand})
    lines = [
        "You must use the calc tool for every arithmetic step; do not compute",
        "in your head. Work the chain IN ORDER, one calc call per step, each",
        "step starting from the result of the previous step.",
        "",
        f"STARTING VALUE: {int(start)}",
    ]
    for t in terms:
        lines.append(f"  Step {t['step']}: {t['op']} {t['operand']}")
    lines += ["", "When the chain is finished, reply with only the final",
              "number on a line of its own.", "", "Final number:"]
    return "\n".join(lines), terms, acc
