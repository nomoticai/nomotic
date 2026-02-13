"""Tests for the Nomotic gateway middleware."""

import io
import json

import pytest

from nomotic.authority import CertificateAuthority
from nomotic.certificate import CertStatus
from nomotic.headers import generate_headers
from nomotic.keys import SigningKey
from nomotic.middleware import (
    REASON_AGE_BELOW_THRESHOLD,
    REASON_ALLOWED,
    REASON_ARCHETYPE_NOT_ALLOWED,
    REASON_CERT_INACTIVE,
    REASON_CERT_NOT_FOUND,
    REASON_INVALID_HEADERS,
    REASON_INVALID_SIGNATURE,
    REASON_NO_CERTIFICATE,
    REASON_ORG_NOT_ALLOWED,
    REASON_TRUST_BELOW_THRESHOLD,
    REASON_ZONE_NOT_ALLOWED,
    GatewayConfig,
    GatewayResult,
    NomoticGateway,
)
from nomotic.store import MemoryCertificateStore


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_ca():
    """Create a CA and issue a certificate for testing."""
    issuer_sk, _ = SigningKey.generate()
    store = MemoryCertificateStore()
    ca = CertificateAuthority(issuer_id="test-issuer", signing_key=issuer_sk, store=store)
    cert, agent_sk = ca.issue("test-agent", "customer-experience", "test-org", "global/us")
    return ca, cert, agent_sk


def _make_headers(cert, agent_sk, body=b""):
    """Generate valid X-Nomotic-* headers for a request."""
    return generate_headers(cert, agent_sk, body)


# ── No-header tests ──────────────────────────────────────────────────


class TestNoHeaders:
    def test_no_headers_require_cert_true(self):
        gateway = NomoticGateway(config=GatewayConfig(require_cert=True))
        result = gateway.check({})
        assert not result.allowed
        assert result.reason == REASON_NO_CERTIFICATE

    def test_no_headers_require_cert_false(self):
        gateway = NomoticGateway(config=GatewayConfig(require_cert=False))
        result = gateway.check({})
        assert result.allowed
        assert result.reason == REASON_ALLOWED


# ── Header-only validation (no CA) ───────────────────────────────────


class TestHeaderOnlyValidation:
    def test_valid_headers_no_ca(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(require_cert=True))
        result = gateway.check(headers)
        assert result.allowed
        assert result.verified_level == 1
        assert result.certificate_id == cert.certificate_id
        assert result.archetype == cert.archetype

    def test_invalid_headers(self):
        gateway = NomoticGateway(config=GatewayConfig(require_cert=True))
        result = gateway.check({"X-Nomotic-Cert-ID": "nmc-test"})
        assert not result.allowed
        assert result.reason == REASON_INVALID_HEADERS


# ── Local CA verification (Level 2) ──────────────────────────────────


class TestLocalCAVerification:
    def test_valid_cert_local_ca(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk, b"body")
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            local_ca=ca,
        ))
        result = gateway.check(headers, b"body")
        assert result.allowed
        assert result.verified_level == 2
        assert result.organization == "test-org"

    def test_invalid_signature(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk, b"original body")
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            local_ca=ca,
        ))
        result = gateway.check(headers, b"tampered body")
        assert not result.allowed
        assert result.reason == REASON_INVALID_SIGNATURE

    def test_cert_not_found_in_ca(self):
        ca, cert, agent_sk = _make_ca()
        # Create a second CA without the cert
        other_sk, _ = SigningKey.generate()
        other_ca = CertificateAuthority(
            issuer_id="other-issuer",
            signing_key=other_sk,
            store=MemoryCertificateStore(),
        )
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            local_ca=other_ca,
        ))
        result = gateway.check(headers)
        assert not result.allowed
        assert result.reason == REASON_CERT_NOT_FOUND

    def test_suspended_cert(self):
        ca, cert, agent_sk = _make_ca()
        ca.suspend(cert.certificate_id, "test suspension")
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert not result.allowed
        assert result.reason == REASON_CERT_INACTIVE


# ── Trust threshold tests ─────────────────────────────────────────────


class TestTrustThreshold:
    def test_trust_below_threshold(self):
        ca, cert, agent_sk = _make_ca()
        ca.update_trust(cert.certificate_id, 0.3)
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            min_trust=0.5,
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert not result.allowed
        assert result.reason == REASON_TRUST_BELOW_THRESHOLD

    def test_trust_at_threshold(self):
        ca, cert, agent_sk = _make_ca()
        # Default trust is 0.5
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            min_trust=0.5,
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert result.allowed

    def test_trust_above_threshold(self):
        ca, cert, agent_sk = _make_ca()
        ca.update_trust(cert.certificate_id, 0.9)
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            min_trust=0.5,
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert result.allowed


# ── Age threshold tests ───────────────────────────────────────────────


class TestAgeThreshold:
    def test_age_below_threshold(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            min_age=10,
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert not result.allowed
        assert result.reason == REASON_AGE_BELOW_THRESHOLD

    def test_age_meets_threshold(self):
        ca, cert, agent_sk = _make_ca()
        for _ in range(10):
            ca.increment_behavioral_age(cert.certificate_id)
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            min_age=10,
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert result.allowed


# ── Archetype filter tests ────────────────────────────────────────────


class TestArchetypeFilter:
    def test_archetype_allowed(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            allowed_archetypes={"customer-experience", "data-processing"},
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert result.allowed

    def test_archetype_not_allowed(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            allowed_archetypes={"data-processing", "analytics"},
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert not result.allowed
        assert result.reason == REASON_ARCHETYPE_NOT_ALLOWED


# ── Zone filter tests ─────────────────────────────────────────────────


class TestZoneFilter:
    def test_zone_exact_match(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            allowed_zones={"global/us"},
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert result.allowed

    def test_zone_prefix_match(self):
        ca, cert, agent_sk = _make_ca()
        # cert zone is "global/us"
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            allowed_zones={"global"},
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert result.allowed

    def test_zone_not_allowed(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            allowed_zones={"global/eu"},
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert not result.allowed
        assert result.reason == REASON_ZONE_NOT_ALLOWED

    def test_multiple_allowed_zones(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            allowed_zones={"global/eu", "global/us"},
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert result.allowed


# ── Organization filter tests ─────────────────────────────────────────


class TestOrgFilter:
    def test_org_allowed(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            allowed_orgs={"test-org"},
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert result.allowed

    def test_org_not_allowed(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            allowed_orgs={"other-org"},
            local_ca=ca,
        ))
        result = gateway.check(headers)
        assert not result.allowed
        assert result.reason == REASON_ORG_NOT_ALLOWED


# ── on_deny="flag" mode ──────────────────────────────────────────────


class TestFlagMode:
    def test_flag_mode_always_allows(self):
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            on_deny="flag",
        ))
        result = gateway.check({})
        assert result.allowed
        assert result.reason == REASON_NO_CERTIFICATE
        assert len(result.issues) > 0

    def test_flag_mode_with_trust_below(self):
        ca, cert, agent_sk = _make_ca()
        ca.update_trust(cert.certificate_id, 0.2)
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            min_trust=0.5,
            local_ca=ca,
            on_deny="flag",
        ))
        result = gateway.check(headers)
        assert result.allowed
        assert result.reason == REASON_TRUST_BELOW_THRESHOLD


# ── WSGI middleware tests ─────────────────────────────────────────────


class TestWSGIMiddleware:
    def _make_wsgi_environ(self, headers, body=b""):
        """Build a minimal WSGI environ dict."""
        environ = {
            "REQUEST_METHOD": "GET",
            "PATH_INFO": "/test",
            "SERVER_NAME": "localhost",
            "SERVER_PORT": "80",
            "wsgi.input": io.BytesIO(body),
        }
        if body:
            environ["CONTENT_LENGTH"] = str(len(body))
        for key, value in headers.items():
            wsgi_key = "HTTP_" + key.upper().replace("-", "_")
            environ[wsgi_key] = value
        return environ

    def test_wsgi_denied_returns_403(self):
        ca, cert, agent_sk = _make_ca()
        ca.update_trust(cert.certificate_id, 0.1)
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            min_trust=0.5,
            local_ca=ca,
        ))

        def dummy_app(environ, start_response):
            raise AssertionError("Should not reach app")

        wsgi_app = gateway.wrap_wsgi(dummy_app)
        environ = self._make_wsgi_environ(headers)
        responses = []

        def start_response(status, headers, exc_info=None):
            responses.append((status, headers))

        body = wsgi_app(environ, start_response)
        assert responses[0][0] == "403 Forbidden"
        response_body = b"".join(body)
        data = json.loads(response_body)
        assert data["reason"] == REASON_TRUST_BELOW_THRESHOLD

    def test_wsgi_no_cert_returns_401(self):
        gateway = NomoticGateway(config=GatewayConfig(require_cert=True))

        def dummy_app(environ, start_response):
            raise AssertionError("Should not reach app")

        wsgi_app = gateway.wrap_wsgi(dummy_app)
        environ = self._make_wsgi_environ({})
        responses = []

        def start_response(status, headers, exc_info=None):
            responses.append((status, headers))

        body = wsgi_app(environ, start_response)
        assert responses[0][0] == "401 Unauthorized"

    def test_wsgi_allowed_passes_through(self):
        ca, cert, agent_sk = _make_ca()
        headers = _make_headers(cert, agent_sk)
        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            local_ca=ca,
        ))

        app_called = []

        def dummy_app(environ, start_response):
            app_called.append(True)
            context = environ.get("nomotic.context", {})
            assert context["certificate_id"] == cert.certificate_id
            body = b"ok"
            start_response("200 OK", [("Content-Length", str(len(body)))])
            return [body]

        wsgi_app = gateway.wrap_wsgi(dummy_app)
        environ = self._make_wsgi_environ(headers)
        responses = []

        def start_response(status, headers, exc_info=None):
            responses.append((status, headers))

        wsgi_app(environ, start_response)
        assert len(app_called) == 1


# ── GatewayResult tests ──────────────────────────────────────────────


class TestGatewayResult:
    def test_to_json(self):
        result = GatewayResult(
            allowed=True,
            reason="allowed",
            certificate_id="nmc-test",
            trust_score=0.8,
        )
        data = json.loads(result.to_json())
        assert data["allowed"] is True
        assert data["certificate_id"] == "nmc-test"

    def test_to_dict(self):
        result = GatewayResult(
            allowed=False,
            reason="trust_below_threshold",
            trust_score=0.3,
            issues=["trust too low"],
        )
        d = result.to_dict()
        assert d["allowed"] is False
        assert d["issues"] == ["trust too low"]

    def test_context_property(self):
        result = GatewayResult(
            allowed=True,
            reason="allowed",
            certificate_id="nmc-test",
            trust_score=0.8,
            behavioral_age=5,
            archetype="customer-experience",
            organization="test-org",
            zone_path="global/us",
            verified_level=2,
        )
        ctx = result.context
        assert ctx["certificate_id"] == "nmc-test"
        assert ctx["trust_score"] == 0.8
        assert ctx["verified_level"] == 2
