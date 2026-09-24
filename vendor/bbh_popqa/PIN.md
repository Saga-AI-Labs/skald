# Vendored upstream: BBH reasoning subset + PopQA factuality

One-shot vendor product of `tools/vendor_bbh_popqa.py` (network at vendor
time only; the adapter reads the committed jsonl offline).

## What and why

- **BBH** (`bbh_<task>.jsonl`, 6169 rows, 8 tasks): multi-step reasoning
  with exact-gradeable outputs (tracking shuffled objects x3, logical
  deduction x3, date understanding, dyck languages). Chosen for the
  logic/reasoning axis: the *output* is small (a choice letter, or one
  bracket for dyck -- chance floor, read it as such) but reaching it
  requires the chain, unlike MMLU recall. Each row records `answer_kind`
  (`letter` vs `text`) so the scorer cannot mix the two.
- **PopQA** (`popqa.jsonl`, 14267 rows): entity-centric short factual
  questions WITH subject pageviews, bucketed to head/mid/tail by s_pop
  tertiles (375 / 2879). The head-vs-tail gap is the
  quant-sensitive number: aggressive rounding prunes rare knowledge first.

## Provenance and licenses

- BBH upstream: BIG-bench Apache-2.0 family (arXiv:2210.09261), read via the
  lighteval/bbh mirror through the public datasets-server rows API.
  maveriq/bigbenchhard tags MIT. Both claims recorded, not resolved; no
  license tag sits on the lighteval mirror itself.
- PopQA upstream repo AlexTMallen/adaptive-retrieval carries an MIT LICENSE
  covering the repo including data/popQA.tsv; the akariasai/PopQA mirror
  states no license. Both claims recorded, not resolved.
- Full per-file rows + SHA-256: `data/manifest.json` (generated
  2026-09-24T03:21:48.413232+00:00).

## Shaping decisions (instrument, not benchmark)

- BBH prompts are built at run time: task_prefix + input + lettered choices
  + "reply with only the letter" instruction. The committed rows carry the
  raw parts, so a prompt change is a code change with a protocol echo, not
  silent data drift.
- PopQA `gold_alts` exclude truncation artefacts (strict substrings of
  another alt) so containment scoring cannot earn credit by accident.
