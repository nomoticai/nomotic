"""FastAPI middleware adapter for Nomotic gateway.

Usage::

    from nomotic.middleware import NomoticGateway, GatewayConfig
    from nomotic.adapters.fastapi_adapter import NomoticMiddleware

    gateway = NomoticGateway(config=GatewayConfig(require_cert=True, min_trust=0.6))
    app.add_middleware(NomoticMiddleware, gateway=gateway)

    # Access certificate context in route handlers:
    @app.get("/data")
    async def get_data(request: Request):
        context = request.state.nomotic  # GatewayResult
        if context.trust_score >= 0.8:
            return full_data()
        return limited_data()
"""

from __future__ import annotations

from typing import Any

try:
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.types import ASGIApp, Receive, Scope, Send
except ImportError as exc:
    raise ImportError(
        "FastAPI/Starlette is required for the FastAPI adapter. "
        "Install it with: pip install 'nomotic[fastapi]'"
    ) from exc

from nomotic.middleware import NomoticGateway, REASON_NO_CERTIFICATE

__all__ = ["NomoticMiddleware"]


class NomoticMiddleware:
    """ASGI middleware that validates Nomotic certificates."""

    def __init__(self, app: ASGIApp, gateway: NomoticGateway) -> None:
        self.app = app
        self.gateway = gateway

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract headers
        headers: dict[str, str] = {}
        for name, value in scope.get("headers", []):
            headers[name.decode("latin-1")] = value.decode("latin-1")

        # Consume the body so we can validate it, then replay for downstream.
        body = b""
        message = await receive()
        if message.get("type") == "http.request":
            body = message.get("body", b"")

        result = self.gateway.check(headers, body)

        if not result.allowed:
            status_code = 401 if result.reason == REASON_NO_CERTIFICATE else 403
            response = JSONResponse(
                content=result.to_dict(),
                status_code=status_code,
            )
            await response(scope, receive, send)
            return

        # Inject context into scope state for access via request.state
        scope.setdefault("state", {})
        scope["state"]["nomotic"] = result

        # Replay the body for the downstream app
        body_sent = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay_receive, send)
