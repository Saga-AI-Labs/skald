"""Shared perturbation operators for the §3.5 sensitivity family.

Proposal §3.5: "Apply a small, stated perturbation to the input ...
and report output stability (Jaccard over tokens, and whether the
verdict changes)." The three operators, each deterministic and stated in
the record protocol:

- ``ws_jitter`` — whitespace permutation: tabs become 4 spaces, then
  leading 4-space groups become tabs, then non-leading runs of 2+
  spaces collapse to exactly 2 spaces;
- ``token_sub`` — substitute one token at a fixed position (default the
  middle token) with ``___``, preserving surrounding whitespace exactly;
- ``clause_swap`` — split on sentence boundaries (". " plus terminal
  punctuation) and swap the first two sentences; inputs with fewer than
  two sentences pass through unchanged.

``jaccard`` is classic set-Jaccard over whitespace tokens (1.0 when both
sides are empty, 0.0 when exactly one is). ``verdict`` is the last A–D
letter when a gold letter is supplied (letter-choice items), else the raw
output itself — so ``verdict_changed`` always means "the answer moved",
never "the wording wiggled".

Stdlib only, deterministic.
"""

from __future__ import annotations

import re

# Perturbation operators in stable application order.
PERTURBATIONS = ("ws_jitter", "token_sub", "clause_swap")

# Per (item, perturbation) metric names in stable report order.
PERTURB_STEP_METRICS = (
    "perturb_jaccard",
    "perturb_verdict_changed",
)

# Per-perturbation aggregate metric names in stable report order.
PERTURB_AGG_METRICS = (
    "perturb_mean_jaccard",
    "perturb_verdict_change_rate",
)

_LETTER_RE = re.compile(r"\b([A-D])\b")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")


def apply_perturbation(name: str, text: str) -> str:
    """Dispatch a named perturbation operator (stable application order)."""
    if name == "ws_jitter":
        return ws_jitter(text)
    if name == "token_sub":
        return token_sub(text)
    if name == "clause_swap":
        return clause_swap(text)
    raise ValueError(
        f"perturb_stats: unknown perturbation {name!r}; "
        f"choose from {PERTURBATIONS}"
    )


def ws_jitter(text: str) -> str:
    """Whitespace permutation: tabs out, leading indents retabbed."""
    lines = text.replace("\t", "    ").split("\n")
    out = []
    for line in lines:
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        tabs, rest = divmod(indent, 4)
        rebuilt = "\t" * tabs + " " * rest + stripped
        rebuilt = re.sub(r"(?<=\S) {2,}", "  ", rebuilt)
        out.append(rebuilt)
    return "\n".join(out)


def token_sub(text: str, position: str | int = "middle",
              token: str = "___") -> str:
    """Substitute one token at a fixed position, whitespace preserved.

    ``position`` is ``"middle"`` (default), ``"first"``, ``"last"``, or a
    0-based token index (negative counts from the end, clamped).
    """
    runs = list(re.finditer(r"\S+", text))
    if not runs:
        return text
    if position == "middle":
        idx = len(runs) // 2
    elif position == "first":
        idx = 0
    elif position == "last":
        idx = len(runs) - 1
    else:
        idx = int(position)
        if idx >= 0:
            idx = min(idx, len(runs) - 1)
        else:
            idx = max(0, len(runs) + idx)
    start, end = runs[idx].span()
    return text[:start] + token + text[end:]


def clause_swap(text: str) -> str:
    """Swap the first two sentences; fewer than two passes through."""
    sentences = [m.group(0) for m in _SENTENCE_RE.finditer(text)]
    if len(sentences) < 2:
        return text
    sentences[0], sentences[1] = sentences[1], sentences[0]
    # Rebuild from the swapped sentences plus whatever followed the last
    # sentence terminator (trailing whitespace, uncapped fragments).
    last_end = 0
    for m in _SENTENCE_RE.finditer(text):
        last_end = m.end()
    return " ".join(s.strip() for s in sentences) + text[last_end:]


def jaccard(a: str, b: str) -> float:
    """Set-Jaccard similarity over whitespace tokens."""
    set_a = set(re.findall(r"\S+", a))
    set_b = set(re.findall(r"\S+", b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def verdict(output: str, gold: str | None = None) -> str:
    """The answer carried by an output: last A–D letter with gold, else raw."""
    if gold is not None:
        hits = _LETTER_RE.findall(output or "")
        return hits[-1] if hits else ""
    return output or ""


def apply_perturbation(name: str, text: str) -> str:
    """Apply the named perturbation operator — the stable public dispatch.

    ``name`` must be one of :data:`PERTURBATIONS`. ``token_sub`` always
    substitutes the middle token (the fixed position, so the effect is a
    stated, re-run-able perturbation — not a random pick).
    """
    if name == "ws_jitter":
        return ws_jitter(text)
    if name == "token_sub":
        return token_sub(text)
    if name == "clause_swap":
        return clause_swap(text)
    raise ValueError(f"unknown perturbation {name!r}; choose from {PERTURBATIONS}")
