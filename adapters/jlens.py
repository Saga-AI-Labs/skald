"""JLens suite adapter (plan §4.5, task skald-adapter-jlens).

Wraps the Apache-2.0 Neuronpedia ``jlens`` library (pinned, vendored,
unmodified under ``vendor/jlens/``) to read out early-layer residual-stream
representations of an ablated model as scalar probability records — one
record per ``(layer, rank)`` at a fixed token position.

Interface: ``run(model, task, config) -> records[]`` (scaffold §5).  The
adapter invokes the vendored upstream package unmodified in a subprocess
under a torch-capable Python (default the operator's OBLITERATUS venv), the
saga pattern: it builds a driver, executes ``python -c <driver>`` with
``PYTHONPATH`` pointing at ``vendor/jlens``, and parses the printed result.
It does not re-implement the lens algorithm and does not persist — callers
write records to the unified store via ``store.put``.

Contract per the recon-fixed plan (§4.5, OPEN-2/OPEN-5, M4):

- ``model`` is the run target: an abliterated HF decoder LM id (recon M4
  target ``huihui-ai/Huihui-gemma-3-270m-it-abliterated``), or a local
  checkpoint file/directory.  Remote ids are ``snapshot_download``-ed to a
  deterministic local dir at run time; ``model_checkpoint_sha256`` is the
  SHA-256 of the resolved local checkpoint (single file via
  ``identity.hash_checkpoint``, directory deterministically over its sorted
  contents, HF cache metadata excluded).
- ``task`` is the readout set — the generic ``layer_readout`` or one of the
  webapp's five named sets (Verbal Report, Directed Modulation, Multi-Hop
  Reasoning, General Broadcast, Selective Mediation).  The named sets share
  the same instrumentation; the label is preserved in the record and the
  protocol.
- ``config`` = lens source (a fitted ``*_jacobian_lens.pt`` via
  ``lens_source``, else fit params ``prompts``/``source_layers``/
  ``dim_batch``/``max_seq_len``/``skip_first``/``dtype``) + readout params
  (``layers``, ``position``, ``top_n``, ``seed``) + subprocess ``python``.
- ``records[]`` flatten each per-layer vector readout to scalar records
  (one per ``(layer, rank)``, ``metric`` ``top{k}_prob@L{l}``, ``value`` =
  softmax probability).  The full per-layer logit vectors go to raw traces
  referenced from ``artifacts[]`` (OPEN-5): a fitted lens is saved to
  ``.skald/jlens_lenses/<model_checkpoint_sha256>/<name>.pt`` and the raw
  per-layer logit readouts are written append-only to
  ``.skald/jlens_raw/<model_checkpoint_sha256>/<YYYYMMDDTHHMMSSZ>_<task>_
  <config_hash>.jsonl``.  ``artifacts[]`` holds repo-relative paths + sha256,
  matching the ``RECORD_FIELDS.artifacts`` contract.
- ``protocol`` names the exact fit+readout configuration and labels the
  result a wiring check, not a scientific measurement (the recon's M4 run
  target is CPU-fitted with a minimal lens; a faithful lens needs the GPU
  box).

Truthfulness: every emitted record carries all canonical ``RECORD_FIELDS``
keys, a non-empty 64-hex ``model_checkpoint_sha256``, a non-empty
``protocol``, and ``script_sha256`` equal to the direct SHA-256 of the
invoked upstream file — the vendored ``vendor/jlens/jlens/__init__.py``
(the package entry that dispatches ``fit``/``from_hf``/``apply``; see
``vendor/jlens/PIN.md`` for the pinned commit and file digests).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import string
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from adapters import RECORD_FIELDS, SuiteAdapter
from identity import hash_checkpoint

DEFAULT_VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "jlens"
DEFAULT_PYTHON = Path("/media/data/coding/OBLITERATUS/.venv/bin/python")

# A readout set per the webapp + the generic label (plan §4.5).
TASKS = frozenset(
    {
        "layer_readout",
        "verbal_report",
        "directed_modulation",
        "multi_hop_reasoning",
        "general_broadcast",
        "selective_mediation",
    }
)

_SEED = 42

# Recon M4 default run target (smallest abliterated HF decoder LM the wrapper
# can load on this CPU box); fallback documented in the plan.
DEFAULT_MODEL = "huihui-ai/Huihui-gemma-3-270m-it-abliterated"

# Default minimal CPU fit (recon M4: few source_layers, 1-2 prompts,
# max_seq_len ~32-64) + readout.  Prompts must tokenize > 17 tokens
# (upstream SKIP_FIRST_N_POSITIONS=16).
_DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the river bank while "
    "the sun sets slowly in the western sky.",
    "A small glass of milk sat on the wooden table beneath the warm light "
    "of the kitchen window.",
]
_DEFAULT_READOUT_PROMPT = (
    "In the heart of the quiet forest the old wooden bridge crossed the "
    "stream where the children had played all afternoon."
)

_UPSTREAM_SCRIPT = "jlens/__init__.py"  # relative to vendor dir

_RESULT_MARK = "JLENS_RESULT_JSON:"

_DEFAULTS = {
    "source_layers": [4],
    "dim_batch": 16,
    "max_seq_len": 48,
    "skip_first": 16,
    "dtype": "float32",
    "top_n": 3,
    "position": -1,
    "seed": _SEED,
}


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _config_hash(fit: dict[str, Any], readout: dict[str, Any]) -> str:
    """Short content hash naming one run's raw-trace file (OPEN-5)."""
    canonical = {
        "fit": {
            "prompts": list(fit.get("prompts") or []),
            "source_layers": list(fit["source_layers"]),
            "dim_batch": int(fit["dim_batch"]),
            "max_seq_len": int(fit["max_seq_len"]),
            "skip_first": int(fit["skip_first"]),
            "dtype": fit["dtype"],
        },
        "readout": {
            "layers": list(readout["layers"]),
            "position": int(readout["position"]),
            "top_n": int(readout["top_n"]),
            "seed": int(readout["seed"]),
        },
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _hash_dir(path: Path) -> str:
    """Deterministic SHA-256 over a checkpoint *directory*'s sorted contents.

    HF ``snapshot_download(local_dir=...)`` caches fetch metadata under
    ``.cache/huggingface/`` which is excluded — the hash covers the checkpoint
    and tokenizer artifacts only, so it is stable across re-downloads.
    """
    h = hashlib.sha256()
    for rel in sorted(
        p.relative_to(path)
        for p in path.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(path).parts
    ):
        h.update(rel.as_posix().encode("utf-8"))
        h.update(b"\0")
        with open(path / rel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _checkpoint_sha256(model: str | Path) -> str:
    """SHA-256 identity of the evaluated model artifact.

    Single files use ``identity.hash_checkpoint``; directories are hashed
    deterministically over their sorted contents (HF cache metadata aside).
    """
    path = Path(model)
    if path.is_file():
        return hash_checkpoint(path)
    if path.is_dir():
        return _hash_dir(path)
    raise FileNotFoundError(f"jlens: model artifact not found: {path}")


class JLensAdapter(SuiteAdapter):
    """Adapter over the vendored Neuronpedia jlens library."""

    def run(
        self,
        model: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        config = config or {}
        vendor = Path(config.get("vendor_dir", DEFAULT_VENDOR))
        if not (vendor / "jlens").is_dir():
            raise FileNotFoundError(
                f"jlens: vendored package not found at {vendor / 'jlens'} "
                "(expected vendor/jlens/jlens from the upstream pin)"
            )
        script = vendor / _UPSTREAM_SCRIPT
        if not script.is_file():
            raise FileNotFoundError(f"jlens: upstream file not found: {script}")
        python = self._resolve_python(config)
        model = str(model or DEFAULT_MODEL).strip()
        task = str(task).strip()
        if task not in TASKS:
            raise ValueError(
                f"jlens: unsupported task {task!r}; choose from {sorted(TASKS)}"
            )

        fit, readout = self._params(config)
        root = Path(__file__).resolve().parent.parent
        artifact_root = Path(config["artifact_dir"]).resolve() if config.get(
            "artifact_dir"
        ) else root
        timeout = int(config.get("timeout", 3600))

        cfg = {
            "model": model,
            "vendor": str(vendor),
            "root": str(root),
            "artifact_root": str(artifact_root),
            "task": task,
            "fit": fit,
            "readout": readout,
            "script": str(script),
            "config_hash": _config_hash(fit, readout),
        }
        payload, stdout = self._capture(root, python, cfg, timeout)

        # Global checkpoint hash from the resolved local artifact the driver
        # used (single file or deterministic dir includes).
        model_dir = payload.get("model_dir")
        if not model_dir or not Path(model_dir).exists():
            raise RuntimeError(
                f"jlens: driver reported no usable local model artifact "
                f"({model_dir!r})"
            )
        checkpoint_sha = _checkpoint_sha256(model_dir)
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
            raise RuntimeError(
                f"jlens: invalid model_checkpoint_sha256 {checkpoint_sha!r}"
            )

        protocol = self._protocol(task, model, fit, readout, payload)
        return self._records(
            task=task,
            protocol=protocol,
            checkpoint_sha=checkpoint_sha,
            seed=int(payload.get("seed", readout["seed"])),
            readout_rows=payload["readout"],
            payload=payload,
            artifact_root=artifact_root,
        )

    # --- machinery --------------------------------------------------------

    def _resolve_python(self, config: dict[str, Any]) -> str:
        import shutil

        python = config.get("python")
        if python:
            cand = Path(python)
            if cand.is_file():
                return str(cand)
            resolved = shutil.which(python)
            if resolved:
                return resolved
            raise FileNotFoundError(f"jlens: suite python not found: {python}")
        if DEFAULT_PYTHON.is_file():
            return str(DEFAULT_PYTHON)
        resolved = shutil.which("python3")
        if resolved:
            return resolved
        raise FileNotFoundError(
            "jlens: no python interpreter with torch/transformers found "
            "(set config['python'])"
        )

    def _params(self, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        fit = {
            "prompts": list(config.get("prompts", _DEFAULT_PROMPTS)),
            "source_layers": [
                int(x) for x in config.get("source_layers", _DEFAULTS["source_layers"])
            ],
            "dim_batch": int(config.get("dim_batch", _DEFAULTS["dim_batch"])),
            "max_seq_len": int(config.get("max_seq_len", _DEFAULTS["max_seq_len"])),
            "skip_first": int(config.get("skip_first", _DEFAULTS["skip_first"])),
        }
        dtype = str(config.get("dtype", _DEFAULTS["dtype"])) or "float32"
        fit["dtype"] = dtype
        lens_source = config.get("lens_source")
        if lens_source:
            lens_source = str(lens_source)
            if not Path(lens_source).is_file():
                raise FileNotFoundError(
                    f"jlens: fitted lens source not found: {lens_source}"
                )
        fit["lens_source"] = lens_source

        readout_layers = [
            int(x) for x in config.get("layers", fit["source_layers"])
        ]
        readout = {
            "prompt": str(config.get("readout_prompt", _DEFAULT_READOUT_PROMPT)),
            "layers": readout_layers,
            "position": int(config.get("position", _DEFAULTS["position"])),
            "top_n": int(config.get("top_n", _DEFAULTS["top_n"])),
            "seed": int(config.get("seed", _DEFAULTS["seed"])),
        }
        return fit, readout

    def _capture(
        self,
        root: Path,
        python: str,
        cfg: dict[str, Any],
        timeout: int,
    ) -> tuple[dict[str, Any], str]:
        body = _DRIVER.substitute(vendor=json.dumps(cfg["vendor"], ensure_ascii=True))
        driver = (
            "import json, os, sys\n"
            "cfg = json.loads(os.environ['JLENS_CFG'])\n"
            "sys.path.insert(0, cfg['vendor'])\n"
            + body
        )
        env = dict(os.environ)
        env["JLENS_CFG"] = json.dumps(cfg)
        env["PYTHONPATH"] = str(root)
        try:
            proc = subprocess.run(
                [python, "-c", driver],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"jlens: driver timed out after {timeout}s") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"jlens: driver failed (rc={proc.returncode})\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        m = re.search(rf"^{re.escape(_RESULT_MARK)}(.*)$", proc.stdout, re.S | re.M)
        if not m:
            raise RuntimeError(
                "jlens: driver printed no parseable result:\n"
                f"{proc.stdout[-2000:]}"
            )
        payload = json.loads(m.group(1))
        if not payload.get("readout"):
            raise RuntimeError(
                "jlens: driver produced no readout records "
                f"(readout={payload.get('readout')!r})"
            )
        return payload, proc.stdout

    def _protocol(
        self,
        task: str,
        model: str,
        fit: dict[str, Any],
        readout: dict[str, Any],
        payload: dict[str, Any],
    ) -> str:
        layout = payload.get("layout", {})
        source = (
            f"pre-fitted lens loaded from {fit['lens_source']}"
            if fit.get("lens_source")
            else (
                f"fit via jlens.fit(source_layers={fit['source_layers']}, "
                f"dim_batch={fit['dim_batch']}, max_seq_len={fit['max_seq_len']}, "
                f"skip_first={fit['skip_first']}, n_prompts={len(fit['prompts'])}, "
                f"dtype={fit['dtype']})"
            )
        )
        return (
            "jlens (Neuronpedia, vendored @ 4e3f3b2, Apache-2.0): "
            f"{task} readout set, {source}; readout "
            f"layers={readout['layers']} position={readout['position']} "
            f"top{readout['top_n']} softmax prob, seed {readout['seed']}, "
            f"model {model} "
            f"(n_layers={layout.get('n_layers')}, d_model={layout.get('d_model')}); "
            "MINIMAL CPU FIT — WIRING CHECK, NOT A SCIENTIFIC MEASUREMENT"
        )

    def _records(
        self,
        *,
        task: str,
        protocol: str,
        checkpoint_sha: str,
        seed: int,
        readout_rows: list[dict[str, Any]],
        payload: dict[str, Any],
        artifact_root: Path,
    ) -> list[dict[str, Any]]:
        artifacts = self._artifacts(artifact_root, payload)
        records = []
        for row in readout_rows:
            rank = int(row["rank"])
            layer = int(row["layer"])
            record = {
                "model_checkpoint_sha256": checkpoint_sha,
                "adapter": "jlens",
                "suite": "jlens",
                "task": task,
                "metric": f"top{rank}_prob@L{layer}",
                "value": float(row["prob"]),
                "n": 1,  # one readout position; see protocol
                "ci_low": None,
                "ci_high": None,
                "protocol": protocol,
                "created_at": _now(),
                "host": socket.gethostname(),
                "script_sha256": payload["script_sha256"],
                "seed": seed,
                "artifacts": artifacts,
            }
            assert set(record) == set(RECORD_FIELDS), set(record) ^ set(RECORD_FIELDS)
            records.append(record)
        return records

    def _artifacts(self, base: Path, payload: dict[str, Any]) -> list[str]:
        """Registry of referenced raw traces + fitted lens (OPEN-5).

        Returns ``[f"{sha256}  <repo-relative path>", ...]`` for every artifact
        the driver wrote, so the store never embeds raw vectors and any
        referenced artifact is never pruned.
        """
        entries = []
        for key in ("trace_rel", "lens_rel"):
            rel = payload.get(key)
            if not rel:
                continue
            abs_path = base / rel
            if not abs_path.is_file():
                raise RuntimeError(f"jlens: referenced artifact missing: {abs_path}")
            entries.append(f"{_file_sha256(abs_path)}  {rel}")
        return entries


# --- driver -----------------------------------------------------------------

# The driver body is a plain Template: only ``$vendor`` is substituted (the
# ``$``-prefixed name cannot collide with the Python code below it), and the
# fixed upstream-path values are read from the ``JLENS_CFG`` environment
# variable so the Python source stays a single immutable string (mirrors the
# saga subprocess pattern).  The driver writes OPEN-5 artifacts (append-only
# raw logit trace + fitted lens) and prints one JSON payload line.
_DRIVER = string.Template(
    r"""
import hashlib
import torch
import transformers
import jlens

def _slug(model):
    import re as _re
    return _re.sub(r"[^0-9A-Za-z._-]+", "-", model.rstrip("/").split("/")[-1]).strip("-") or "model"

def _hash_dir(path):
    h = hashlib.sha256()
    for rel in sorted(p.relative_to(path) for p in path.rglob("*") if p.is_file()):
        if ".cache" in rel.parts:
            continue
        h.update(rel.as_posix().encode("utf-8")); h.update(b"\0")
        with open(path / rel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()

def _checkpoint_sha256(path):
    p = os.path.abspath(path)
    if os.path.isfile(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    return _hash_dir(__import__("pathlib").Path(p))

model_arg = cfg["model"]
fit = cfg["fit"]
readout = cfg["readout"]
root = __import__("pathlib").Path(cfg["root"])
artifact_root = __import__("pathlib").Path(cfg["artifact_root"])

# --- resolve the model artifact ---------------------------------------------
if os.path.exists(model_arg) or os.path.exists(os.path.abspath(model_arg)):
    model_dir = os.path.abspath(model_arg)
else:
    from huggingface_hub import snapshot_download
    slug = _slug(model_arg)
    cache = root / ".skald" / "jlens_models"
    cache.mkdir(parents=True, exist_ok=True)
    model_dir = snapshot_download(model_arg, local_dir=str(cache / slug))

# --- load + wrap -------------------------------------------------------------
torch_dtype = getattr(torch, fit["dtype"], torch.float32)
tok = transformers.AutoTokenizer.from_pretrained(model_dir)
hf = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch_dtype)
lm = jlens.from_hf(hf, tok, compile=False)
layout = {"n_layers": int(lm.n_layers), "d_model": int(lm.d_model)}

# --- fit or load the lens ----------------------------------------------------
lens_source = fit.get("lens_source")
lens = None
if lens_source:
    from jlens.lens import JacobianLens
    lens = JacobianLens.load(lens_source)
else:
    lens = jlens.fit(
        lm,
        list(fit["prompts"]),
        source_layers=list(fit["source_layers"]),
        dim_batch=int(fit["dim_batch"]),
        max_seq_len=int(fit["max_seq_len"]),
        skip_first=int(fit["skip_first"]),
    )

# --- readout -----------------------------------------------------------------
prompt = readout["prompt"]
layers = sorted(set(int(x) for x in readout["layers"]))
lens_logits, model_logits, input_ids = lens.apply(
    lm, prompt, layers=layers,
    position=int(readout["position"]), max_seq_len=int(fit["max_seq_len"]),
)

def _topk(logits, n):
    probs = torch.softmax(logits.float(), -1)
    v, i = torch.topk(probs, n)
    return [(rank, int(tid), float(p))
            for rank, (p, tid) in enumerate(zip(v.tolist(), i.tolist()), start=1)]

readout_rows = []
for layer in layers:
    if layer not in lens_logits:
        continue
    for rank, tid, prob in _topk(lens_logits[layer], int(readout["top_n"])):
        readout_rows.append({
            "layer": int(layer), "rank": rank, "token_id": tid,
            "token": tok.convert_ids_to_tokens([tid])[0], "prob": prob,
        })
if not readout_rows:
    raise RuntimeError("jlens: no layers produced logits for the readout")

# --- OPEN-5 artifacts (append-only raw trace + fitted lens) ------------------
checkpoint_sha = _checkpoint_sha256(model_dir)
raw_dir = artifact_root / ".skald" / "jlens_raw" / checkpoint_sha
lens_dir = artifact_root / ".skald" / "jlens_lenses" / checkpoint_sha
raw_dir.mkdir(parents=True, exist_ok=True)
lens_dir.mkdir(parents=True, exist_ok=True)
import datetime
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
config_hash = cfg["config_hash"]
trace_path = raw_dir / f"{stamp}_{cfg['task']}_{config_hash}.jsonl"
lens_path = None
if lens_source is None:
    lens_path = lens_dir / f"{cfg['task']}_{config_hash}_jacobian_lens.pt"
    lens.save(str(lens_path))

with open(trace_path, "a") as f:
    header = {
        "kind": "jlens_raw_v1", "model": model_arg,
        "model_checkpoint_sha256": checkpoint_sha, "task": cfg["task"],
        "stamp": stamp, "layout": layout, "readout_prompt": prompt,
        "positions": int(readout["position"]),
    }
    f.write(json.dumps(header) + "\n")
    seq = input_ids.tolist()[0] if hasattr(input_ids, "tolist") else None
    for layer in layers:
        if layer not in lens_logits:
            continue
        row = {
            "kind": "layer_logits", "layer": int(layer),
            "position": int(readout["position"]), "input_token_ids": seq,
            "logits": lens_logits[layer].tolist(),
        }
        f.write(json.dumps(row) + "\n")

trace_rel = str(trace_path.relative_to(artifact_root))
lens_rel = str(lens_path.relative_to(artifact_root)) if lens_path else None

out = {
    "layout": layout,
    "model_dir": model_dir,
    "model_checkpoint_sha256": checkpoint_sha,
    "script_sha256": hashlib.sha256(open(cfg["script"], "rb").read()).hexdigest(),
    "seed": int(cfg["readout"]["seed"]),
    "trace_rel": trace_rel,
    "lens_rel": lens_rel,
    "readout": readout_rows,
}
print("PLACEHOLDER_RESULT_MARK" + " " + json.dumps(out, default=str))
""".replace("PLACEHOLDER_RESULT_MARK", _RESULT_MARK)
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for the jlens adapter: run a readout, persist, read back.

    Usage:
        python -m adapters.jlens [model] [task] [--config '{"source_layers": [4]}']
    """
    import argparse
    import sys

    import store

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help="abliterated HF model id or local checkpoint path "
        f"(default {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "task",
        nargs="?",
        default="layer_readout",
        choices=sorted(TASKS),
        help="readout set (default layer_readout)",
    )
    ap.add_argument(
        "--config",
        default="{}",
        help="JSON config (python, prompts, source_layers, dim_batch, "
        "max_seq_len, skip_first, dtype, layers, position, top_n, seed, "
        "lens_source, timeout, ...)",
    )
    args = ap.parse_args(argv)

    config = json.loads(args.config)
    records = JLensAdapter().run(args.model, args.task, config)
    if not records:
        print("jlens: no records produced", file=sys.stderr)
        return 2

    store.put(records)
    back = store.query({"adapter": "jlens", "task": args.task})
    print(f"persisted {len(records)} jlens records; queried back {len(back)} matching")
    for r in back:
        print(
            f"  {r['task']}:{r['metric']} = {r['value']} "
            f"(n={r['n']}, protocol={r['protocol']!r})"
        )
    if len(back) < len(records):
        print(
            f"warning: read back {len(back)} of {len(records)} records",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())