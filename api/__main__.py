"""Run the Skald HTTP/JSON API surface locally (``python -m api``).

Dependency-light serving consistent with the store's stdlib posture: uses
``http.server.ThreadingHTTPServer`` and the spec-driven handler from
``api.app``.  Read-only over the store; no cloud LLM anywhere in the chain.
"""

from __future__ import annotations

import argparse
import os
import sys
from http.server import ThreadingHTTPServer

from api.app import make_handler
from store.backend import Store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m api", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "--store-dir",
        default=None,
        help="store directory (default: $SKALD_STORE_DIR or .skald/store)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store_dir = args.store_dir or os.environ.get("SKALD_STORE_DIR", ".skald/store")
    handler = make_handler(lambda: Store(store_dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"skald api listening on http://{args.host}:{args.port} "
        f"(store: {store_dir})",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())