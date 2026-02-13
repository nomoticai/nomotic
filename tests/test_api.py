"""Tests for the Nomotic REST API server."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest

from nomotic.api import NomoticAPIServer
from nomotic.authority import CertificateAuthority
from nomotic.keys import SigningKey
from nomotic.registry import ArchetypeRegistry, OrganizationRegistry, ZoneValidator
from nomotic.store import MemoryCertificateStore


# ── Test fixtures ───────────────────────────────────────────────────────

def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class APITestClient:
    """Minimal HTTP client for testing the API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def get(self, path: str) -> tuple[int, dict[str, Any]]:
        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
                return resp.status, body
        except urllib.error.HTTPError as e:
            body = json.loads(e.read())
            return e.code, body

    def post(self, path: str, data: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        url = f"{self.base_url}{path}"
        payload = json.dumps(data or {}).encode("utf-8")
        try:
            req = urllib.request.Request(
                url, data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
                return resp.status, body
        except urllib.error.HTTPError as e:
            body = json.loads(e.read())
            return e.code, body

    def patch(self, path: str, data: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        url = f"{self.base_url}{path}"
        payload = json.dumps(data or {}).encode("utf-8")
        try:
            req = urllib.request.Request(
                url, data=payload, method="PATCH",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
                return resp.status, body
        except urllib.error.HTTPError as e:
            body = json.loads(e.read())
            return e.code, body


@pytest.fixture(scope="module")
def api_server() -> tuple[APITestClient, CertificateAuthority]:
    """Start a test API server and return a client + the CA."""
    sk, _vk = SigningKey.generate()
    store = MemoryCertificateStore()
    ca = CertificateAuthority(issuer_id="test-issuer", signing_key=sk, store=store)
    arch_reg = ArchetypeRegistry.with_defaults()
    zone_val = ZoneValidator()
    org_reg = OrganizationRegistry()

    port = _find_free_port()
    server = NomoticAPIServer(
        ca,
        archetype_registry=arch_reg,
        zone_validator=zone_val,
        org_registry=org_reg,
        host="127.0.0.1",
        port=port,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)  # Give server time to start

    client = APITestClient(f"http://127.0.0.1:{port}")
    yield client, ca
    server.shutdown()


# ═══════════════════════════════════════════════════════════════════════
# Health & Info
# ═══════════════════════════════════════════════════════════════════════


class TestHealthEndpoints:
    def test_health(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/health")
        assert status == 200
        assert body["status"] == "ok"

    def test_info(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/info")
        assert status == 200
        assert "version" in body
        assert "issuer_id" in body
        assert "issuer_fingerprint" in body
        assert "uptime_seconds" in body
        assert "certificate_count" in body


# ═══════════════════════════════════════════════════════════════════════
# Certificates
# ═══════════════════════════════════════════════════════════════════════


class TestCertificateEndpoints:
    def test_issue_certificate(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/certificates", {
            "agent_id": "agent-1",
            "archetype": "customer-experience",
            "organization": "test-org",
            "zone_path": "global/us",
        })
        assert status == 201
        assert body["agent_id"] == "agent-1"
        assert body["archetype"] == "customer-experience"
        assert body["status"] == "ACTIVE"
        assert body["certificate_id"].startswith("nmc-")

    def test_issue_missing_fields(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/certificates", {
            "agent_id": "agent-x",
        })
        assert status == 400
        assert body["error"] == "validation_error"
        assert "Missing fields" in body["message"]

    def test_list_certificates(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/certificates")
        assert status == 200
        assert isinstance(body, list)
        assert len(body) >= 1

    def test_get_certificate(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        # Issue first
        _, cert_data = client.post("/v1/certificates", {
            "agent_id": "agent-get",
            "archetype": "analytics",
            "organization": "test-org",
            "zone_path": "global",
        })
        cert_id = cert_data["certificate_id"]

        status, body = client.get(f"/v1/certificates/{cert_id}")
        assert status == 200
        assert body["certificate_id"] == cert_id

    def test_get_certificate_not_found(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/certificates/nmc-nonexistent")
        assert status == 404
        assert body["error"] == "not_found"

    def test_verify_certificate(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        _, cert_data = client.post("/v1/certificates", {
            "agent_id": "agent-verify",
            "archetype": "analytics",
            "organization": "test-org",
            "zone_path": "global",
        })
        cert_id = cert_data["certificate_id"]

        status, body = client.post(f"/v1/certificates/{cert_id}/verify")
        assert status == 200
        assert body["valid"] is True

    def test_verify_live(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        _, cert_data = client.post("/v1/certificates", {
            "agent_id": "agent-live",
            "archetype": "analytics",
            "organization": "test-org",
            "zone_path": "global",
        })
        cert_id = cert_data["certificate_id"]

        status, body = client.post(f"/v1/certificates/{cert_id}/verify/live")
        assert status == 200
        assert body["valid"] is True
        assert "trust_score" in body
        assert "behavioral_age" in body
        assert "healthy" in body

    def test_suspend_certificate(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        _, cert_data = client.post("/v1/certificates", {
            "agent_id": "agent-suspend",
            "archetype": "analytics",
            "organization": "test-org",
            "zone_path": "global",
        })
        cert_id = cert_data["certificate_id"]

        status, body = client.patch(f"/v1/certificates/{cert_id}/suspend", {"reason": "test"})
        assert status == 200
        assert body["status"] == "SUSPENDED"

    def test_reactivate_certificate(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        _, cert_data = client.post("/v1/certificates", {
            "agent_id": "agent-reactivate",
            "archetype": "analytics",
            "organization": "test-org",
            "zone_path": "global",
        })
        cert_id = cert_data["certificate_id"]

        client.patch(f"/v1/certificates/{cert_id}/suspend", {"reason": "test"})
        status, body = client.patch(f"/v1/certificates/{cert_id}/reactivate")
        assert status == 200
        assert body["status"] == "ACTIVE"

    def test_revoke_certificate(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        _, cert_data = client.post("/v1/certificates", {
            "agent_id": "agent-revoke",
            "archetype": "analytics",
            "organization": "test-org",
            "zone_path": "global",
        })
        cert_id = cert_data["certificate_id"]

        status, body = client.patch(f"/v1/certificates/{cert_id}/revoke", {"reason": "decommissioned"})
        assert status == 200
        assert body["status"] == "REVOKED"

    def test_renew_certificate(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        _, cert_data = client.post("/v1/certificates", {
            "agent_id": "agent-renew",
            "archetype": "analytics",
            "organization": "test-org",
            "zone_path": "global",
        })
        cert_id = cert_data["certificate_id"]

        status, body = client.post(f"/v1/certificates/{cert_id}/renew")
        assert status == 200
        assert body["lineage"] == cert_id

    def test_transfer_zone(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        _, cert_data = client.post("/v1/certificates", {
            "agent_id": "agent-zone",
            "archetype": "analytics",
            "organization": "test-org",
            "zone_path": "global",
        })
        cert_id = cert_data["certificate_id"]

        status, body = client.patch(f"/v1/certificates/{cert_id}/zone", {"zone_path": "global/eu"})
        assert status == 200
        assert body["zone_path"] == "global/eu"

    def test_reputation(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        _, cert_data = client.post("/v1/certificates", {
            "agent_id": "agent-rep",
            "archetype": "analytics",
            "organization": "test-org",
            "zone_path": "global",
        })
        cert_id = cert_data["certificate_id"]

        status, body = client.get(f"/v1/certificates/{cert_id}/reputation")
        assert status == 200
        assert body["trust_score"] == 0.5
        assert body["behavioral_age"] == 0


# ═══════════════════════════════════════════════════════════════════════
# Quick Verify
# ═══════════════════════════════════════════════════════════════════════


class TestQuickVerify:
    def test_quick_verify(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        _, cert_data = client.post("/v1/certificates", {
            "agent_id": "agent-qv",
            "archetype": "analytics",
            "organization": "test-org",
            "zone_path": "global/us",
        })
        cert_id = cert_data["certificate_id"]

        status, body = client.get(f"/v1/verify/{cert_id}")
        assert status == 200
        assert body["valid"] is True
        assert body["certificate_id"] == cert_id
        assert body["status"] == "ACTIVE"
        assert "trust_score" in body
        assert "behavioral_age" in body
        assert "archetype" in body
        assert "organization" in body
        assert "zone_path" in body
        assert "healthy" in body

    def test_quick_verify_not_found(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/verify/nmc-nonexistent")
        assert status == 404


# ═══════════════════════════════════════════════════════════════════════
# Revocations
# ═══════════════════════════════════════════════════════════════════════


class TestRevocationsEndpoint:
    def test_revocations_list(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/revocations")
        assert status == 200
        assert "revoked" in body
        assert "generated_at" in body
        assert isinstance(body["revoked"], list)


# ═══════════════════════════════════════════════════════════════════════
# Archetypes
# ═══════════════════════════════════════════════════════════════════════


class TestArchetypeEndpoints:
    def test_list_archetypes(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/archetypes")
        assert status == 200
        assert isinstance(body, list)
        assert len(body) >= 16  # built-in count

    def test_list_archetypes_by_category(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/archetypes?category=financial")
        assert status == 200
        assert all(a["category"] == "financial" for a in body)

    def test_get_archetype(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/archetypes/customer-experience")
        assert status == 200
        assert body["name"] == "customer-experience"
        assert body["builtin"] is True

    def test_get_archetype_not_found(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/archetypes/nonexistent-arch")
        assert status == 404

    def test_register_archetype(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/archetypes", {
            "name": "my-custom-arch",
            "description": "A custom archetype",
            "category": "custom",
        })
        assert status == 201
        assert body["name"] == "my-custom-arch"
        assert body["builtin"] is False

    def test_register_archetype_duplicate(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/archetypes", {
            "name": "customer-experience",
            "description": "dup",
            "category": "dup",
        })
        assert status == 400

    def test_validate_archetype(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/archetypes/validate", {"name": "customer-experience"})
        assert status == 200
        assert body["valid"] is True

    def test_validate_archetype_invalid(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/archetypes/validate", {"name": "Bad Name!"})
        assert status == 200
        assert body["valid"] is False
        assert len(body["errors"]) > 0


# ═══════════════════════════════════════════════════════════════════════
# Organizations
# ═══════════════════════════════════════════════════════════════════════


class TestOrganizationEndpoints:
    def test_register_org(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/organizations", {"name": "Test Org Alpha"})
        assert status == 201
        assert body["name"] == "test-org-alpha"

    def test_register_org_duplicate(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        client.post("/v1/organizations", {"name": "Dup Corp"})
        status, body = client.post("/v1/organizations", {"name": "Dup Corp"})
        assert status == 409
        assert body["error"] == "conflict"

    def test_register_org_missing_name(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/organizations", {})
        assert status == 400

    def test_list_orgs(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/organizations")
        assert status == 200
        assert isinstance(body, list)

    def test_get_org(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        client.post("/v1/organizations", {"name": "Get Org Test"})
        status, body = client.get("/v1/organizations/get-org-test")
        assert status == 200
        assert body["name"] == "get-org-test"

    def test_get_org_not_found(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/organizations/nonexistent-org")
        assert status == 404

    def test_validate_org(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/organizations/validate", {"name": "new-org-name"})
        assert status == 200
        assert body["valid"] is True

    def test_suspend_org(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        client.post("/v1/organizations", {"name": "Suspend Org"})
        status, body = client.patch("/v1/organizations/suspend-org/suspend", {"reason": "test"})
        assert status == 200
        assert body["status"] == "SUSPENDED"

    def test_revoke_org(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        client.post("/v1/organizations", {"name": "Revoke Org"})
        status, body = client.patch("/v1/organizations/revoke-org/revoke", {"reason": "test"})
        assert status == 200
        assert body["status"] == "REVOKED"


# ═══════════════════════════════════════════════════════════════════════
# Zones
# ═══════════════════════════════════════════════════════════════════════


class TestZoneEndpoints:
    def test_validate_zone_valid(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/zones/validate", {"zone_path": "global/us/production"})
        assert status == 200
        assert body["valid"] is True

    def test_validate_zone_invalid(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.post("/v1/zones/validate", {"zone_path": "/bad/path/"})
        assert status == 200
        assert body["valid"] is False
        assert len(body["errors"]) > 0


# ═══════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    def test_404_unknown_route(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/nonexistent")
        assert status == 404
        assert body["error"] == "not_found"

    def test_error_structure(self, api_server: tuple[APITestClient, CertificateAuthority]) -> None:
        client, _ca = api_server
        status, body = client.get("/v1/certificates/nmc-nonexistent")
        assert status == 404
        assert "error" in body
        assert "message" in body
        assert "status_code" in body
