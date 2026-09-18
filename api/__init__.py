"""LLM-facing API surface (plan §4.3, §5 api row).

HTTP/JSON endpoints generated from the shared capability spec
(``surfaces.spec``) and backed by the same ``store.query`` as the UI — the
same functional surface, exposed over the network.  One GET endpoint per spec
operation at ``/api/v1/<operation.id>``; read-only over the unified store.

Run the server with ``python -m api``.  For embedding, the pure dispatch
surface is :func:`api.app.dispatch`.
"""

from __future__ import annotations

from .app import (
    API_VERSION,
    ApiError,
    dispatch,
    endpoint_path,
    error_body,
    execute_operation,
    make_handler,
    operation_routes,
)

__all__ = [
    "API_VERSION",
    "ApiError",
    "dispatch",
    "endpoint_path",
    "error_body",
    "execute_operation",
    "make_handler",
    "operation_routes",
]