"""Shared failure-mode classifiers for the §3.3 length-stress family.

Proposal §3.3: accuracy, refusal rate, and failure rate as a function of
*completion* length at fixed prompt length — "a model that degrades
gracefully and a model that loops are different objects, and a single
accuracy number does not tell them apart."

Per sweep step (one fixed prompt, one ``max_tokens`` budget) a completion
gets exactly one failure-mode label, by precedence:

- ``empty`` — no completion text at all;
- ``refusal`` — refusal-marker phrases (a deliberate short stop);
- ``loop`` — the tail is periodic (the model looped; with
  ``finish_reason == "length"`` it looped until it exhausted its budget —
  the failure shape the proposal's 76-token-prompt/32768-token-generation
  crash exhibited);
- ``truncated`` — hit the token budget (``finish_reason == "length"``)
  without looping;
- ``ok`` — natural stop, non-empty, no loop, no refusal.

Tokenization here is whitespace runs — deliberately tokenizer-free so the
classifiers stay stdlib-only and deterministic. They measure shape
(repetition, refusal markers), not token-exact budgets.

Stdlib only, deterministic.
"""

from __future__ import annotations

import re

# Failure-mode labels in precedence order (first match wins).
MODES = ("empty", "refusal", "loop", "truncated", "ok")

# Per-step metric names in stable report order.
STRESS_STEP_METRICS = (
    "stress_ok",
    "stress_refusal",
    "stress_loop",
    "stress_truncated",
    "stress_empty",
    "stress_completion_chars",
)

# Curve-level (per prompt) metric names in stable report order.
STRESS_CURVE_METRICS = (
    "stress_failure_rate",
    "stress_first_failure_at",
    "stress_max_clean_chars",
)

_REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "as a language model",
    "unable to",
    "i refuse",
    "not able to",
    "against my",
)

_MAX_PERIOD = 50
_MIN_REPEATS = 3
_TAIL_TOKENS = 200


def tokens(text: str) -> list[str]:
    """Whitespace-run tokenization for shape analysis."""
    return re.findall(r"\S+", text)


def refusal_detected(text: str) -> bool:
    """True when the completion carries refusal-marker phrases."""
    lowered = text.casefold()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def loop_detected(text: str) -> tuple[bool, int, int]:
    """Detect a periodic tail: (loop, period_tokens, span_tokens).

    Checks the last ``_TAIL_TOKENS`` tokens for a block of ``p`` tokens
    repeated ``_MIN_REPEATS``+ times consecutively, for ``p`` in
    ``1.._MAX_PERIOD``. Returns the smallest such period and the span it
    covers; ``(False, 0, 0)`` when no periodic tail is found.
    """
    toks = tokens(text)[-_TAIL_TOKENS:]
    for period in range(1, min(_MAX_PERIOD, len(toks) // _MIN_REPEATS) + 1):
        block = toks[-period:]
        span = period * _MIN_REPEATS
        if toks[-span:] == block * _MIN_REPEATS:
            return True, period, span
    return False, 0, 0


def classify(text: str, finish_reason: str | None) -> str:
    """Assign the single failure-mode label for one completion."""
    if not text.strip():
        return "empty"
    if refusal_detected(text):
        return "refusal"
    loop, _, _ = loop_detected(text)
    if loop:
        return "loop"
    if (finish_reason or "") == "length":
        return "truncated"
    return "ok"


def summarize_curve(modes: list[str], chars: list[int],
                    budgets: list[int]) -> dict[str, float]:
    """Summarize one prompt's sweep into STRESS_CURVE_METRICS.

    - ``stress_failure_rate`` — fraction of steps not ``ok``;
    - ``stress_first_failure_at`` — first failing budget, or -1.0 when the
      whole sweep is clean (stated in the record protocol);
    - ``stress_max_clean_chars`` — largest completion among ``ok`` steps,
      or -1.0 when no step is clean.
    """
    if not modes:
        raise ValueError("summarize_curve of empty sweep")
    failing = [b for m, b in zip(modes, budgets) if m != "ok"]
    clean = [c for m, c in zip(modes, chars) if m == "ok"]
    return {
        "stress_failure_rate": sum(1 for m in modes if m != "ok") / len(modes),
        "stress_first_failure_at": float(min(failing)) if failing else -1.0,
        "stress_max_clean_chars": float(max(clean)) if clean else -1.0,
    }
