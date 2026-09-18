# Vendored upstream: Neuronpedia jlens package

This directory pins an unmodified copy of the Apache-2.0 `jlens` Python
package from the Neuronpedia repository, integrated for reproducible
invocation from the Skald JLens adapter (`adapters/jlens.py`). The copy is
byte-for-byte identical to the upstream tree at the pinned commit; it is
**not** forked or patched. Fitting/readout logic lives upstream.

- Upstream repo: `https://github.com/hijohnnylin/neuronpedia`
- Pinned commit: `4e3f3b2cf1d85a4821a6fb1c46970efea872004d`
  (2026-09-15, commit message: "fix(inference): return an empty steer
  completion instead of a 500 when the model generates no text")
- Source subtree (copy source):
  `utils/neuronpedia-utils/neuronpedia_utils/jlens/`
- License: Apache-2.0, header SPDX `Apache-2.0` on every module; the
  upstream `LICENSE` file is copied alongside as `vendor/jlens/LICENSE`
  (sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).

## How the pin was made

```
mkdir -p vendor/jlens
cp -r <upstream>/utils/neuronpedia-utils/neuronpedia_utils/jlens/jlens vendor/jlens/jlens
cp <upstream>/utils/neuronpedia-utils/neuronpedia_utils/jlens/LICENSE vendor/jlens/LICENSE
```

`import jlens` resolves when the parent directory `vendor/jlens` is on
`PYTHONPATH` (mirrors upstream layout). The Skald adapter sets that path in
its subprocess driver.

## Pinned file digests (sha256, verified against upstream)

| file | sha256 |
| --- | --- |
| `jlens/__init__.py` | `a735834b95986243e4bba226ab1fddac73a6330ed8d784a4bcc9a5b99decd63d` |
| `jlens/_logging.py` | `051791435992d867bfd5473fc026560fb57da60295f8166532074cea6cacbbe9` |
| `jlens/fitting.py` | `367e7ed45c95d806d1f1f7f7b32bd6002dca6337aa0e21283bf2bd10d898a8e6` |
| `jlens/hf.py` | `fbf3fef1ef6bd4520e24ead1d0ce14fdc2e55690f13d2d287570e7c5410e52ef` |
| `jlens/hooks.py` | `0d29609b01e56cf001e2aa2665bafd4c0e7e390edf08cd2fec4c4c45d81a64f7` |
| `jlens/lens.py` | `334b736af706a52836f91c21944028a7ed0b43c8ea031fad8cf78cdb50120240` |
| `jlens/protocol.py` | `54b392d1e178fdc0f0923b8f81bacd496b2232374c3f0571945124392f551c2e` |

## The invoked upstream script/file (script_sha256)

The upstream library entry shared by the driver is its top-level
`__init__.py` (dispatches `fit`, `from_hf`, `JacobianLens.apply`). The Skald
adapter records `script_sha256` as the SHA-256 of this vendored
`jlens/__init__.py` file — i.e. the direct SHA-256 of the invoked upstream
file. All seven package files are pinned here at the commit above, so the
record is auditable against this manifest.