"""Path A parity scoring (torch side; proposal §3.2, Path A).

Executed under a torch-capable interpreter with the pin-versioned tree on
``sys.path`` — never imported by Skald itself. Reads JSON argv, prints a
JSON object ``{domain: [kl_per_position...]}`` to stdout.

Reference logits are recomputed offline from the bundle (stored ``x`` cast
to fp32 @ fp32 head); the candidate runs a full fp32 forward over the same
``input_ids`` with fresh state per text span. KL(reference ‖ candidate) is
reduced in fp64. Quantiles and records are the adapter's job (stdlib).
"""

import json
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

import torch
import torch.nn.functional as F

from pipeline.config import Config, build_model


def main() -> int:
    argv = json.loads(sys.argv[1])
    bundle = Path(argv["bundle_dir"])
    bconf = json.loads((bundle / "config.json").read_text())
    head = torch.load(bundle / "head.pt", map_location="cpu",
                      weights_only=True)["weight"].to(torch.float32)

    ckpt = torch.load(str(argv["candidate"]), map_location="cpu",
                      weights_only=False)
    raw_cfg = ckpt["cfg"] if isinstance(ckpt.get("cfg"), dict) \
        else ckpt["cfg"].__dict__
    names = {f.name for f in dataclass_fields(Config)}
    cfg = Config(**{k: v for k, v in raw_cfg.items() if k in names})
    if cfg.model != bconf["model_kind"] or cfg.n_embd != bconf["n_embd"] \
            or cfg.vocab_size != bconf["vocab_size"]:
        raise RuntimeError(
            f"score: candidate arch ({cfg.model}, n_embd={cfg.n_embd}, "
            f"vocab={cfg.vocab_size}) does not match the reference bundle "
            f"({bconf['model_kind']}, n_embd={bconf['n_embd']}, "
            f"vocab={bconf['vocab_size']}) — scoring across architectures "
            f"is meaningless"
        )
    block_size = int(bconf["block_size"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out: dict = {}
    with torch.no_grad():
        for part in sorted((bundle / "domains").glob("*.pt")):
            saved = torch.load(part, map_location="cpu", weights_only=True)
            ids = saved["input_ids"].to(torch.long)
            x_ref = saved["x"].to(torch.float32)
            ref_logits = x_ref.view(-1, bconf["n_embd"]) @ head
            kl_all: list[float] = []
            for start, end in saved["spans"]:
                state = None
                pos = start
                cand_chunks: list = []
                while pos < end:
                    chunk = ids[pos:min(pos + block_size, end)].unsqueeze(0)
                    logits, _loss, state = model(chunk, None, state)
                    cand_chunks.append(logits.squeeze(0))
                    pos += chunk.size(1)
                cand_logits = torch.cat(cand_chunks, dim=0)
                seg = slice(start, end)
                logp = F.log_softmax(ref_logits[seg].double(), dim=-1)
                logq = F.log_softmax(cand_logits.double(), dim=-1)
                kl = (logp.exp() * (logp - logq)).sum(dim=-1)
                kl_all.extend(kl.tolist())
            out[part.stem] = kl_all
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
