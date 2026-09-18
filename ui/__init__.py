"""Human-facing surface (plan §4.3).

Result browsing, per-task drilldown, anomaly view.  One HTML view per
operation in ``surfaces.spec.OPERATIONS``, generated from the shared spec and
backed by the same ``store.query`` as the API.  Conformance-tested against the
API surface.
"""

from .app import (
    envelope_for,
    make_handler,
    parse_request,
    render_page,
    render_view,
    view_path,
    view_routes,
)

__all__ = [
    "envelope_for",
    "make_handler",
    "parse_request",
    "render_page",
    "render_view",
    "view_path",
    "view_routes",
]