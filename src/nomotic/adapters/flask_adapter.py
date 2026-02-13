"""Flask integration for Nomotic gateway.

Usage::

    from nomotic.middleware import NomoticGateway, GatewayConfig
    from nomotic.adapters.flask_adapter import nomotic_required

    gateway = NomoticGateway(config=GatewayConfig(require_cert=True, min_trust=0.6))

    @app.route("/data")
    @nomotic_required(gateway)
    def get_data():
        context = g.nomotic  # GatewayResult
        return jsonify(data)
"""

from __future__ import annotations

import functools
from typing import Any, Callable

try:
    from flask import g, jsonify, request as flask_request
except ImportError as exc:
    raise ImportError(
        "Flask is required for the Flask adapter. "
        "Install it with: pip install 'nomotic[flask]'"
    ) from exc

from nomotic.middleware import NomoticGateway, REASON_NO_CERTIFICATE

__all__ = ["nomotic_required", "nomotic_before_request"]


def nomotic_required(gateway: NomoticGateway) -> Callable:
    """Decorator that validates Nomotic certificates on Flask routes."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            headers = dict(flask_request.headers)
            body = flask_request.get_data()
            result = gateway.check(headers, body)

            if not result.allowed:
                status_code = 401 if result.reason == REASON_NO_CERTIFICATE else 403
                return jsonify(result.to_dict()), status_code

            g.nomotic = result
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def nomotic_before_request(gateway: NomoticGateway) -> Callable:
    """Register as a Flask before_request handler for app-wide validation.

    Usage::

        app.before_request(nomotic_before_request(gateway))
    """

    def handler() -> Any:
        headers = dict(flask_request.headers)
        body = flask_request.get_data()
        result = gateway.check(headers, body)

        if not result.allowed:
            status_code = 401 if result.reason == REASON_NO_CERTIFICATE else 403
            return jsonify(result.to_dict()), status_code

        g.nomotic = result
        return None

    return handler
