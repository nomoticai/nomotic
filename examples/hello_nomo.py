#!/usr/bin/env python3
"""hello-nomo: See Nomotic governance in action in 30 seconds.

Run:
    python examples/hello_nomo.py

What happens:
    1. Creates a temporary governance environment (in-memory, no files)
    2. Registers an organization
    3. Issues a birth certificate for an agent
    4. Starts a tiny governed API server
    5. The agent makes requests to the API
    6. You see: certificate headers, governance evaluation, trust calibration
    7. The agent does something suspicious -- watch governance respond
    8. Cleanup

No installation beyond ``pip install nomotic`` required.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from nomotic.authority import CertificateAuthority
from nomotic.headers import generate_headers, parse_headers
from nomotic.keys import SigningKey
from nomotic.middleware import GatewayConfig, GatewayResult, NomoticGateway
from nomotic.sdk import GovernedAgent, GovernedResponse
from nomotic.store import MemoryCertificateStore

# ── ANSI colors (with no-color fallback) ──────────────────────────────

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


def _dim(t: str) -> str:
    return _c("2", t)


def _header(text: str) -> None:
    print()
    print(_bold(_cyan(f"{'=' * 60}")))
    print(_bold(_cyan(f"  {text}")))
    print(_bold(_cyan(f"{'=' * 60}")))


def _step(n: int, text: str) -> None:
    print()
    print(_bold(f"  [{n}] {text}"))


def _info(text: str) -> None:
    print(f"      {text}")


def _ok(text: str) -> None:
    print(f"      {_green('OK')} {text}")


def _warn(text: str) -> None:
    print(f"      {_yellow('!!')} {text}")


def _fail(text: str) -> None:
    print(f"      {_red('DENIED')} {text}")


# ── Tiny example service ──────────────────────────────────────────────

_gateway: NomoticGateway | None = None
_request_log: list[dict[str, Any]] = []


class _ExampleHandler(BaseHTTPRequestHandler):
    """A tiny API that validates Nomotic certificates via the gateway."""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress stderr

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_DELETE(self) -> None:
        self._handle("DELETE")

    def _handle(self, method: str) -> None:
        assert _gateway is not None
        headers = {k: v for k, v in self.headers.items()}
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""

        result = _gateway.check(headers, body)

        log_entry: dict[str, Any] = {
            "method": method,
            "path": self.path,
            "allowed": result.allowed,
            "reason": result.reason,
            "trust_score": result.trust_score,
            "verified_level": result.verified_level,
        }
        _request_log.append(log_entry)

        if not result.allowed:
            resp = json.dumps(result.to_dict(), indent=2).encode()
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        # Simulate graduated responses based on trust
        trust = result.trust_score or 0.0
        if self.path == "/api/data":
            if trust >= 0.7:
                data = {"level": "full", "records": 100, "trust": trust}
            elif trust >= 0.5:
                data = {"level": "standard", "records": 50, "trust": trust}
            else:
                data = {"level": "limited", "records": 10, "trust": trust}
        elif self.path == "/api/sensitive/delete":
            data = {"error": "this endpoint requires manual approval", "trust": trust}
            resp = json.dumps(data, indent=2).encode()
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return
        else:
            data = {"message": "ok", "path": self.path, "trust": trust}

        resp = json.dumps(data, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


# ── Main demo ─────────────────────────────────────────────────────────


def run_demo() -> None:
    """Run the hello-nomo demo. All in-memory, no files created."""
    global _gateway

    _header("Nomotic Governance Demo")
    print()
    print("  This demo creates a complete governance environment in-memory,")
    print("  issues a certificate, and shows how governance works end-to-end.")

    # Step 1: Create governance infrastructure
    _step(1, "Creating governance infrastructure")
    issuer_sk, issuer_vk = SigningKey.generate()
    store = MemoryCertificateStore()
    ca = CertificateAuthority(
        issuer_id="hello-nomo-issuer",
        signing_key=issuer_sk,
        store=store,
    )
    _ok("CertificateAuthority created")
    _info(f"Issuer: hello-nomo-issuer")
    _info(f"Fingerprint: {issuer_vk.fingerprint()[:32]}...")

    # Step 2: Issue a birth certificate
    _step(2, "Issuing birth certificate for hello-agent")
    cert, agent_sk = ca.issue(
        agent_id="hello-agent",
        archetype="customer-experience",
        organization="hello-corp",
        zone_path="global/us",
    )
    _ok(f"Certificate: {cert.certificate_id}")
    _info(f"Agent:     hello-agent")
    _info(f"Archetype: customer-experience")
    _info(f"Org:       hello-corp")
    _info(f"Zone:      global/us")
    _info(f"Trust:     {cert.trust_score}")
    _info(f"Age:       {cert.behavioral_age}")

    # Step 3: Set up the gateway
    _step(3, "Setting up NomoticGateway middleware")
    _gateway = NomoticGateway(config=GatewayConfig(
        require_cert=True,
        min_trust=0.3,
        local_ca=ca,
        verify_signature=True,
    ))
    _ok("Gateway configured: require_cert=True, min_trust=0.3")

    # Step 4: Start example service
    _step(4, "Starting governed API service")
    server = HTTPServer(("127.0.0.1", 0), _ExampleHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _ok(f"Service running on http://127.0.0.1:{port}")

    # Step 5: Create governed agent
    _step(5, "Creating GovernedAgent from certificate")
    agent = GovernedAgent(
        certificate=cert,
        signing_key=agent_sk,
        base_url=f"http://127.0.0.1:{port}",
    )
    _ok(f"Agent ready: {agent.cert_id}")

    # Step 6: Normal requests
    _header("Normal Operations")
    trust_trajectory: list[float] = [cert.trust_score]

    for i in range(5):
        _step(i + 1, f"GET /api/data (request {i + 1}/5)")
        resp = agent.get("/api/data")

        if resp.ok:
            data = resp.json()
            _ok(f"Status {resp.status} - Access level: {data.get('level', 'unknown')}")
        else:
            _fail(f"Status {resp.status}")

        # Show what headers were sent
        hdrs = generate_headers(cert, agent_sk, b"")
        _info(f"X-Nomotic-Cert-ID:   {hdrs['X-Nomotic-Cert-ID'][:20]}...")
        _info(f"X-Nomotic-Trust:     {hdrs['X-Nomotic-Trust']}")
        _info(f"X-Nomotic-Age:       {hdrs['X-Nomotic-Age']}")
        _info(f"X-Nomotic-Archetype: {hdrs['X-Nomotic-Archetype']}")

        # Simulate trust climbing with successful actions
        ca.record_action(cert.certificate_id, min(0.95, cert.trust_score + 0.05))
        trust_trajectory.append(cert.trust_score)
        _info(f"Trust after action:  {_green(f'{cert.trust_score:.2f}')}")

        time.sleep(0.05)

    # Step 7: Suspicious action
    _header("Suspicious Action")
    _step(1, "DELETE /api/sensitive/delete (suspicious!)")
    resp = agent.delete("/api/sensitive/delete")
    _fail(f"Status {resp.status} - Service denied the sensitive operation")
    _info("The service recognized this as a sensitive endpoint")
    _info("and rejected the request despite valid certificate.")

    # Step 8: What if trust drops?
    _header("Trust Degradation")
    _step(1, "Simulating trust drop due to policy violations")
    ca.update_trust(cert.certificate_id, 0.2)
    trust_trajectory.append(cert.trust_score)
    _warn(f"Trust dropped to {_red(f'{cert.trust_score:.2f}')} (below min_trust=0.3)")

    _step(2, "GET /api/data (with degraded trust)")
    # Need to recreate agent to pick up new trust score on headers
    agent = GovernedAgent(
        certificate=cert,
        signing_key=agent_sk,
        base_url=f"http://127.0.0.1:{port}",
    )
    resp = agent.get("/api/data")
    if resp.ok:
        _ok(f"Status {resp.status}")
    else:
        data = resp.json()
        _fail(f"Status {resp.status} - {data.get('reason', 'rejected')}")
        _info("The gateway rejected the request because trust")
        _info(f"score {cert.trust_score:.2f} < min_trust 0.30")

    # Step 9: Recovery
    _header("Trust Recovery")
    _step(1, "Trust restored after remediation")
    ca.update_trust(cert.certificate_id, 0.6)
    trust_trajectory.append(cert.trust_score)
    _ok(f"Trust restored to {_green(f'{cert.trust_score:.2f}')}")

    agent = GovernedAgent(
        certificate=cert,
        signing_key=agent_sk,
        base_url=f"http://127.0.0.1:{port}",
    )
    resp = agent.get("/api/data")
    _ok(f"Status {resp.status} - Access restored")

    # Summary
    _header("Summary")
    print()
    _info("Trust trajectory:")
    for i, t in enumerate(trust_trajectory):
        bar_len = int(t * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        color = _green if t >= 0.5 else (_yellow if t >= 0.3 else _red)
        _info(f"  [{i:2d}] {color(f'{t:.2f}')} |{bar}|")

    print()
    _info(f"Total requests:   {len(_request_log)}")
    _info(f"  Allowed:        {sum(1 for r in _request_log if r['allowed'])}")
    _info(f"  Denied:         {sum(1 for r in _request_log if not r['allowed'])}")
    print()
    _info("Key takeaways:")
    _info("  - Certificates carry identity and trust through HTTP headers")
    _info("  - The gateway validates certificates on every request")
    _info("  - Trust is dynamic: it increases with good behavior")
    _info("  - Trust degradation leads to access denial")
    _info("  - Recovery is possible through remediation")
    print()

    # Cleanup
    server.shutdown()
    _gateway = None
    _request_log.clear()


if __name__ == "__main__":
    run_demo()
