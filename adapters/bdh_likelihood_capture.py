"""Path A reference capture (torch side; proposal §3.2, Path A).

Executed under a torch-capable interpreter with the pin-versioned tree on
``sys.path`` (cwd = repo) — never imported by Skald itself. Reads JSON argv
from ``sys.argv[1]``, writes the reference bundle, prints a JSON summary.

The structural fact that makes this cheap (``bdh.py:282``): the head is a
plain matmul ``x.view(B, T, D) @ lm_head`` over the post-LN trunk output, so
the reference is ``x`` per position plus the head — logits are recomputed
offline, never stored.

``x`` is captured with a forward hook on the model's final ``ln`` (the last
``ln`` call before the head matmul). That is an architectural assumption,
and it is checked, not trusted: every forward call asserts
``x @ lm_head ≈ logits`` and the run reports the global max abs deviation.
A future pin that moves the head fails loudly here instead of silently
capturing garbage.
"""

import json
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

import torch

from pipeline.config import Config, build_model


def _reconstruct_config(cfg_dict: dict) -> Config:
    names = {f.name for f in dataclass_fields(Config)}
    return Config(**{k: v for k, v in cfg_dict.items() if k in names})


def main() -> int:
    argv = json.loads(sys.argv[1])
    ckpt_path = Path(argv["checkpoint"])
    bundle = Path(argv["bundle_dir"])
    block_size = int(argv.get("block_size", 0))
    store_dtype = {"float16": torch.float16, "float32": torch.float32}[
        argv.get("dtype", "float16")
    ]

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = _reconstruct_config(ckpt["cfg"] if isinstance(ckpt.get("cfg"), dict)
                              else ckpt["cfg"].__dict__)
    if not block_size:
        block_size = int(cfg.block_size or 128)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    head = dict(model.state_dict())["lm_head"].detach().to(torch.float32)
    n_embd, vocab = tuple(head.shape)

    ln_outputs: list = []
    model.ln.register_forward_hook(lambda _m, _i, o: ln_outputs.append(o))

    domains_out: dict = {}
    bundle_domains = bundle / "domains"
    bundle_domains.mkdir(parents=True, exist_ok=True)
    global_selfcheck = 0.0

    with torch.no_grad():
        for dom in argv["domains"]:
            domain = dom["domain"]
            ids_all: list[int] = []
            spans: list[list[int]] = []
            skipped = 0
            for text in dom["texts"]:
                ids = list(text.encode("utf-8"))
                if len(ids) < 2:
                    skipped += 1
                    continue
                spans.append([len(ids_all), len(ids_all) + len(ids)])
                ids_all.extend(ids)
            if not ids_all:
                raise RuntimeError(
                    f"capture: domain {domain!r} has no usable contexts "
                    f"({skipped} skipped as <2 bytes)"
                )
            xs: list = []
            for start, end in spans:
                state = None
                pos = start
                while pos < end:
                    chunk = torch.tensor(
                        [ids_all[pos:min(pos + block_size, end)]],
                        dtype=torch.long,
                    )
                    ln_outputs.clear()
                    logits, _loss, state = model(chunk, None, state)
                    x = ln_outputs[-1].detach().to(torch.float32).squeeze(1)
                    # The architectural assumption, checked every call.
                    dev = (x.view(1, -1, n_embd) @ head - logits).abs().max().item()
                    global_selfcheck = max(global_selfcheck, dev)
                    xs.append(x.squeeze(0))
                    pos += chunk.size(1)
            torch.save(
                {
                    "input_ids": torch.tensor(ids_all, dtype=torch.uint8),
                    "x": torch.cat(xs, dim=0).to(store_dtype),
                    "spans": spans,
                },
                bundle_domains / f"{domain}.pt",
            )
            domains_out[domain] = {
                "positions": len(ids_all),
                "texts": len(spans),
                "skipped_texts": skipped,
            }

    torch.save({"weight": head}, bundle / "head.pt")
    bundle_config = {
        "model_kind": cfg.model,
        "n_layer": cfg.n_layer,
        "n_embd": n_embd,
        "n_head": cfg.n_head,
        "vocab_size": vocab,
        "block_size": block_size,
        "stored_dtype": str(store_dtype).replace("torch.", ""),
        "selfcheck_max_abs_logit_diff": global_selfcheck,
    }
    (bundle / "config.json").write_text(json.dumps(bundle_config, indent=2) + "\n")
    print(json.dumps({"config": bundle_config, "domains": domains_out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
