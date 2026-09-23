"""Section 3.4 router-utilisation capture (torch side; proposal §3.4).

Executed under a torch-capable interpreter with the pin-versioned tree on
``sys.path`` (cwd = repo) — never imported by Skald itself. Reads JSON argv
from ``sys.argv[1]``, prints a JSON summary.

The vendored model selects capacity slots with ``_k_sparse_relu``
(``bdh.py:19,252,267``): top-k of the ReLU'd latents per position, applied
twice per layer (encoder side, then decoder side). This capture records,
per layer and side, how often each slot is selected over routed-domain
text streams — the raw material for load histograms, dead-slot fractions,
and gate entropy (proposal §3.4: "does every expert still do anything?").
Indices are pooled across heads and batch positions, so each gate call
contributes n_head * k selections.

Two structural facts make this sound without touching the vendor tree:

- The gate is patched *in this process only* (module attribute swap). The
  original function still computes the forward values; the wrapper
  recomputes the top-k indices from the same inputs and asserts the
  recomputed mask reproduces the original output exactly. A future pin
  that changes the gate fails loudly here.
- The call order inside ``BDH.forward`` is fixed (encoder call, then
  decoder call, per layer), so call index ``i`` maps to layer ``i // 2``
  and side ``x`` (even) / ``y`` (odd). Every forward call asserts exactly
  ``2 * n_layer`` gate calls.

Conditional probe (proposal §3.4 correction): when ``k_sparse_ratio <= 0``
there is no top-k gate (plain ReLU), so there is nothing to instrument.
The script then returns ``not_applicable: true`` — the adapter refuses to
store records, because the honest result is *not applicable*, not zero.
"""

import json
import sys

import torch

from pipeline.analyze import _load_model


def main() -> int:
    argv = json.loads(sys.argv[1])
    ckpt_path = argv["checkpoint"]
    block_size = int(argv.get("block_size", 0))

    import bdh as bdh_mod

    model, cfg = _load_model(ckpt_path)
    ratio = float(cfg.get("k_sparse_ratio", 0.0) or 0.0)
    n_layer = int(cfg.get("n_layer", model.config.n_layer))
    n_head = int(model.config.n_head)
    width = int(model.config.mlp_internal_dim_multiplier
                * model.config.n_embd // n_head)
    if not block_size:
        block_size = int(cfg.get("block_size") or 128)

    if ratio <= 0.0:
        print(json.dumps({
            "not_applicable": True,
            "k_sparse_ratio": ratio,
            "reason": (
                "k_sparse_ratio <= 0: plain ReLU, no top-k gate exists to "
                "instrument (bdh.py:252,267). Section 3.4 is not applicable, "
                "not zero."
            ),
        }))
        return 0

    orig_gate = bdh_mod._k_sparse_relu
    state = {"calls": [], "domain": None}

    def counting_gate(x, gate_ratio):
        x_pos = torch.relu(x)
        k = max(1, int(gate_ratio * x_pos.shape[-1]))
        _, indices = torch.topk(x_pos, k, dim=-1)
        mask = torch.zeros_like(x_pos).scatter_(-1, indices, 1.0)
        out = orig_gate(x, gate_ratio)
        # The gate assumption, checked every call: the wrapper's
        # recomputed selection must reproduce the original exactly.
        if not torch.equal(out, x_pos * mask):
            raise RuntimeError(
                "util-capture: recomputed gate mask disagrees with "
                "bdh._k_sparse_relu — pin moved the gate, refusing to "
                "record garbage"
            )
        state["calls"].append(indices.detach().cpu())
        return out

    bdh_mod._k_sparse_relu = counting_gate
    model.eval()

    domains_out: dict = {}
    with torch.no_grad():
        for dom in argv["domains"]:
            domain = dom["domain"]
            ids_all: list[int] = []
            for text in dom["texts"]:
                ids = list(text.encode("utf-8"))
                if len(ids) < 2:
                    continue
                ids_all.extend(ids)
            if not ids_all:
                raise RuntimeError(
                    f"util-capture: domain {domain!r} has no usable contexts"
                )
            counts = torch.zeros(2 * n_layer, width, dtype=torch.int64)
            positions = 0
            pos = 0
            # Sequential streaming over every domain byte: deterministic,
            # complete coverage (no random crops to seed).
            while pos < len(ids_all):
                chunk = torch.tensor(
                    [ids_all[pos:min(pos + block_size, len(ids_all))]],
                    dtype=torch.long,
                )
                state["calls"].clear()
                model(chunk)
                if len(state["calls"]) != 2 * n_layer:
                    raise RuntimeError(
                        "util-capture: expected "
                        f"{2 * n_layer} gate calls per forward, got "
                        f"{len(state['calls'])} — forward moved the gate"
                    )
                for call_i, indices in enumerate(state["calls"]):
                    flat = indices.reshape(-1)
                    counts[call_i] += torch.bincount(flat, minlength=width)
                positions += chunk.size(1)
                pos += chunk.size(1)
            layers = []
            for layer in range(n_layer):
                for side_i, side in ((0, "x"), (1, "y")):
                    layers.append({
                        "layer": layer,
                        "side": side,
                        "counts": counts[2 * layer + side_i].tolist(),
                    })
            domains_out[domain] = {"positions": positions, "layers": layers}

    print(json.dumps({
        "not_applicable": False,
        "config": {
            "k_sparse_ratio": ratio,
            "k_absolute": max(1, int(ratio * width)),
            "width": width,
            "n_layer": n_layer,
            "n_head": n_head,
            "block_size": block_size,
        },
        "domains": domains_out,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
