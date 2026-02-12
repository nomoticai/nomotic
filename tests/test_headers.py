"""Tests for HTTP header generation, parsing, and verification."""

from nomotic.authority import CertificateAuthority
from nomotic.headers import (
    CertificateHeaders,
    RequestVerifyResult,
    generate_headers,
    parse_headers,
    verify_request,
)
from nomotic.keys import SigningKey


def _setup():
    """Create a CA and issue a certificate for testing."""
    issuer_sk, _ = SigningKey.generate()
    ca = CertificateAuthority(issuer_id="test-issuer", signing_key=issuer_sk)
    cert, agent_sk = ca.issue("agent-1", "customer-experience", "acme", "global/us")
    return ca, cert, agent_sk


class TestGenerateHeaders:
    def test_returns_all_required_headers(self):
        _ca, cert, agent_sk = _setup()
        headers = generate_headers(cert, agent_sk, b"body")
        assert "X-Nomotic-Cert-ID" in headers
        assert "X-Nomotic-Trust" in headers
        assert "X-Nomotic-Age" in headers
        assert "X-Nomotic-Archetype" in headers
        assert "X-Nomotic-Zone" in headers
        assert "X-Nomotic-Signature" in headers
        assert "X-Nomotic-Gov-Hash" in headers

    def test_header_values(self):
        _ca, cert, agent_sk = _setup()
        headers = generate_headers(cert, agent_sk, b"body")
        assert headers["X-Nomotic-Cert-ID"] == cert.certificate_id
        assert headers["X-Nomotic-Trust"] == str(cert.trust_score)
        assert headers["X-Nomotic-Age"] == str(cert.behavioral_age)
        assert headers["X-Nomotic-Archetype"] == cert.archetype
        assert headers["X-Nomotic-Zone"] == cert.zone_path
        assert headers["X-Nomotic-Gov-Hash"] == cert.governance_hash


class TestParseHeaders:
    def test_parse_round_trip(self):
        _ca, cert, agent_sk = _setup()
        headers = generate_headers(cert, agent_sk, b"body")
        parsed = parse_headers(headers)
        assert isinstance(parsed, CertificateHeaders)
        assert parsed.certificate_id == cert.certificate_id
        assert parsed.trust_score == cert.trust_score
        assert parsed.behavioral_age == cert.behavioral_age
        assert parsed.archetype == cert.archetype
        assert parsed.zone_path == cert.zone_path
        assert parsed.governance_hash == cert.governance_hash
        assert len(parsed.signature) > 0

    def test_parse_missing_header_raises(self):
        import pytest
        with pytest.raises(KeyError):
            parse_headers({"X-Nomotic-Cert-ID": "nmc-1"})  # Missing others


class TestVerifyRequest:
    def test_valid_request(self):
        ca, cert, agent_sk = _setup()
        body = b"request payload"
        headers = generate_headers(cert, agent_sk, body)
        result = verify_request(headers, body, ca)
        assert result.valid
        assert result.certificate_id == cert.certificate_id
        assert result.issues == []

    def test_tampered_body_fails(self):
        ca, cert, agent_sk = _setup()
        headers = generate_headers(cert, agent_sk, b"original")
        result = verify_request(headers, b"tampered", ca)
        assert not result.valid
        assert any("signature" in i for i in result.issues)

    def test_unknown_cert_fails(self):
        ca, cert, agent_sk = _setup()
        body = b"body"
        headers = generate_headers(cert, agent_sk, body)
        headers["X-Nomotic-Cert-ID"] = "nmc-nonexistent"
        result = verify_request(headers, body, ca)
        assert not result.valid
        assert any("not found" in i for i in result.issues)

    def test_suspended_cert_fails(self):
        ca, cert, agent_sk = _setup()
        ca.suspend(cert.certificate_id, "test")
        body = b"body"
        headers = generate_headers(cert, agent_sk, body)
        result = verify_request(headers, body, ca)
        assert not result.valid

    def test_malformed_headers_fail(self):
        ca, cert, agent_sk = _setup()
        result = verify_request({}, b"body", ca)
        assert not result.valid
        assert any("parse error" in i for i in result.issues)

    def test_empty_body(self):
        ca, cert, agent_sk = _setup()
        body = b""
        headers = generate_headers(cert, agent_sk, body)
        result = verify_request(headers, body, ca)
        assert result.valid
