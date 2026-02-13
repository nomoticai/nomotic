#!/usr/bin/env python3
"""Example: Multiple governed agents with different trust levels.

Shows how the gateway treats agents differently based on their
certificates, archetypes, and trust histories.

Run:
    python examples/multi_agent.py

All in-memory, no files created, no external dependencies.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from nomotic.authority import CertificateAuthority
from nomotic.keys import SigningKey
from nomotic.middleware import GatewayConfig, NomoticGateway
from nomotic.sdk import GovernedAgent
from nomotic.store import MemoryCertificateStore

# ── ANSI colors ───────────────────────────────────────────────────────

_NO_COLOR = not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(t: str) -> str:
    return _c("1", t)


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _cyan(t: str) -> str:
    return _c("36", t)


# ── Example service ───────────────────────────────────────────────────

_gateway: NomoticGateway | None = None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        assert _gateway is not None
        headers = {k: v for k, v in self.headers.items()}
        result = _gateway.check(headers)

        if not result.allowed:
            body = json.dumps(result.to_dict(), indent=2).encode()
            self.send_response(403)
        else:
            trust = result.trust_score or 0.0
            data = {
                "access": "full" if trust >= 0.7 else "standard" if trust >= 0.5 else "limited",
                "agent": result.certificate_id,
                "trust": trust,
                "archetype": result.archetype,
            }
            body = json.dumps(data, indent=2).encode()
            self.send_response(200)

        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Demo ──────────────────────────────────────────────────────────────


def run_demo() -> None:
    """Run the multi-agent demo."""
    global _gateway

    print()
    print(_bold(_cyan("=" * 60)))
    print(_bold(_cyan("  Multi-Agent Governance Demo")))
    print(_bold(_cyan("=" * 60)))
    print()
    print("  Three agents with different archetypes and trust levels")
    print("  access the same governed service.")
    print()

    # Set up infrastructure
    issuer_sk, _ = SigningKey.generate()
    store = MemoryCertificateStore()
    ca = CertificateAuthority(
        issuer_id="multi-agent-issuer",
        signing_key=issuer_sk,
        store=store,
    )

    # Issue certificates for three agents
    print(_bold("  Issuing certificates:"))
    print()

    # Agent 1: Experienced customer-experience agent (high trust)
    cert1, sk1 = ca.issue(
        agent_id="experienced-agent",
        archetype="customer-experience",
        organization="acme-corp",
        zone_path="global/us",
    )
    # Simulate history: boost trust and age
    for _ in range(20):
        ca.record_action(cert1.certificate_id, min(0.95, cert1.trust_score + 0.02))

    print(f"  1. {_green('experienced-agent')}")
    print(f"     Archetype: customer-experience")
    print(f"     Trust: {_green(f'{cert1.trust_score:.2f}')}, Age: {cert1.behavioral_age}")
    print()

    # Agent 2: New data-processing agent (baseline trust)
    cert2, sk2 = ca.issue(
        agent_id="new-agent",
        archetype="data-processing",
        organization="acme-corp",
        zone_path="global/us",
    )
    print(f"  2. {_yellow('new-agent')}")
    print(f"     Archetype: data-processing")
    print(f"     Trust: {_yellow(f'{cert2.trust_score:.2f}')}, Age: {cert2.behavioral_age}")
    print()

    # Agent 3: Suspended agent
    cert3, sk3 = ca.issue(
        agent_id="suspended-agent",
        archetype="analytics",
        organization="acme-corp",
        zone_path="global/us",
    )
    ca.suspend(cert3.certificate_id, "policy violation investigation")
    print(f"  3. {_red('suspended-agent')}")
    print(f"     Archetype: analytics")
    print(f"     Status: {_red('SUSPENDED')}")
    print()

    # Set up gateway requiring certificates with local CA
    _gateway = NomoticGateway(config=GatewayConfig(
        require_cert=True,
        min_trust=0.3,
        local_ca=ca,
        verify_signature=True,
    ))

    # Start service
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    # Create governed agents
    agent1 = GovernedAgent(certificate=cert1, signing_key=sk1, base_url=base_url)
    agent2 = GovernedAgent(certificate=cert2, signing_key=sk2, base_url=base_url)
    agent3 = GovernedAgent(certificate=cert3, signing_key=sk3, base_url=base_url)

    # Test each agent
    print(_bold(_cyan("-" * 60)))
    print(_bold("  Requests:"))
    print()

    agents = [
        ("experienced-agent", agent1, cert1),
        ("new-agent", agent2, cert2),
        ("suspended-agent", agent3, cert3),
    ]

    for name, agent, cert in agents:
        print(f"  {_bold(name)} -> GET /api/data")
        resp = agent.get("/api/data")

        if resp.ok:
            data = resp.json()
            access = data.get("access", "unknown")
            trust = data.get("trust", 0.0)
            color = _green if access == "full" else _yellow
            print(f"    {_green('200 OK')} - Access: {color(access)}, Trust: {trust:.2f}")
        else:
            try:
                data = resp.json()
                reason = data.get("reason", "unknown")
            except Exception:
                reason = "unknown"
            print(f"    {_red(f'{resp.status} DENIED')} - Reason: {reason}")
        print()

    # Summary
    print(_bold(_cyan("-" * 60)))
    print(_bold("  Summary:"))
    print()
    print(f"    {_green('experienced-agent')}: High trust ({cert1.trust_score:.2f}), "
          f"age {cert1.behavioral_age} -> full access")
    print(f"    {_yellow('new-agent')}:         Baseline trust ({cert2.trust_score:.2f}), "
          f"age {cert2.behavioral_age} -> standard access")
    print(f"    {_red('suspended-agent')}:   Suspended -> {_red('rejected')}")
    print()
    print("  The same gateway, same service, same endpoint -- but different")
    print("  outcomes based on each agent's certificate and trust history.")
    print()

    # Cleanup
    server.shutdown()
    _gateway = None


if __name__ == "__main__":
    run_demo()
