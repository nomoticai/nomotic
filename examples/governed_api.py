#!/usr/bin/env python3
"""Example: FastAPI service with Nomotic certificate validation.

Shows how to protect API endpoints with certificate-based trust.

    pip install nomotic fastapi uvicorn
    python examples/governed_api.py

This example requires FastAPI and uvicorn to run.
"""

from __future__ import annotations

import sys
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("This example requires FastAPI and uvicorn:")
    print("  pip install 'nomotic[fastapi]'")
    sys.exit(1)

from nomotic.adapters.fastapi_adapter import NomoticMiddleware
from nomotic.middleware import GatewayConfig, NomoticGateway

# ── App setup ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Governed API Example",
    description="API endpoints protected by Nomotic certificate validation",
)

# Configure the gateway: require certificates, verify signatures
gateway = NomoticGateway(config=GatewayConfig(
    require_cert=True,
    min_trust=0.3,
    verify_signature=False,  # No local CA in this example
))

app.add_middleware(NomoticMiddleware, gateway=gateway)


# ── Public endpoint (still requires cert, but low trust) ──────────────


@app.get("/api/public")
async def public_data(request: Request) -> dict[str, Any]:
    """Public data - any certified agent can access."""
    context = getattr(request.state, "nomotic", None)
    return {
        "data": "public information",
        "agent": context.certificate_id if context else None,
    }


# ── Standard endpoint (requires moderate trust) ──────────────────────


@app.get("/api/standard")
async def standard_data(request: Request) -> Any:
    """Standard data - requires trust >= 0.5."""
    context = getattr(request.state, "nomotic", None)
    trust = context.trust_score if context else 0.0

    if trust < 0.5:
        return JSONResponse(
            status_code=403,
            content={
                "error": "insufficient_trust",
                "message": f"Trust {trust:.2f} < 0.50 required",
                "required_trust": 0.5,
            },
        )

    return {
        "data": "standard business data",
        "records": 100,
        "trust": trust,
    }


# ── Sensitive endpoint (requires high trust) ─────────────────────────


@app.get("/api/sensitive")
async def sensitive_data(request: Request) -> Any:
    """Sensitive data - requires trust >= 0.8 and minimum age."""
    context = getattr(request.state, "nomotic", None)
    trust = context.trust_score if context else 0.0
    age = context.behavioral_age if context else 0

    issues: list[str] = []
    if trust < 0.8:
        issues.append(f"trust {trust:.2f} < 0.80 required")
    if age < 10:
        issues.append(f"behavioral age {age} < 10 required")

    if issues:
        return JSONResponse(
            status_code=403,
            content={
                "error": "insufficient_credentials",
                "message": "; ".join(issues),
                "required_trust": 0.8,
                "required_age": 10,
            },
        )

    return {
        "data": "sensitive financial records",
        "records": 500,
        "trust": trust,
        "age": age,
    }


# ── Health check (no auth needed - but gateway still runs) ────────────


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Entry point ──────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Starting governed API on http://localhost:8000")
    print()
    print("Endpoints:")
    print("  GET /api/public    - any certified agent")
    print("  GET /api/standard  - trust >= 0.5")
    print("  GET /api/sensitive - trust >= 0.8, age >= 10")
    print()
    print("Send requests with X-Nomotic-* headers to test.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
