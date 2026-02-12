"""Tests for the certificate data structure."""

import json
from datetime import datetime, timedelta, timezone

from nomotic.certificate import AgentCertificate, CertStatus, CertVerifyResult, LiveVerifyResult
from nomotic.keys import SigningKey


def _make_cert(**overrides) -> AgentCertificate:
    """Create a certificate with sensible defaults for testing."""
    _sk, vk = SigningKey.generate()
    defaults = {
        "certificate_id": "nmc-test-1234",
        "agent_id": "agent-1",
        "archetype": "customer-experience",
        "organization": "acme-corp",
        "zone_path": "global/us/acme-corp/retail",
        "issued_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "trust_score": 0.50,
        "behavioral_age": 0,
        "status": CertStatus.ACTIVE,
        "public_key": vk.to_bytes(),
        "fingerprint": vk.fingerprint(),
        "governance_hash": "abc123",
        "lineage": None,
        "issuer_signature": b"\x00" * 32,
        "expires_at": None,
    }
    defaults.update(overrides)
    return AgentCertificate(**defaults)


class TestCertificateFields:
    def test_certificate_id_prefix(self):
        cert = _make_cert(certificate_id="nmc-abcd-1234")
        assert cert.certificate_id.startswith("nmc-")

    def test_baseline_trust(self):
        cert = _make_cert()
        assert cert.trust_score == 0.50

    def test_zero_behavioral_age(self):
        cert = _make_cert()
        assert cert.behavioral_age == 0

    def test_active_status(self):
        cert = _make_cert()
        assert cert.status == CertStatus.ACTIVE

    def test_no_lineage_by_default(self):
        cert = _make_cert()
        assert cert.lineage is None

    def test_no_expiration_by_default(self):
        cert = _make_cert()
        assert cert.expires_at is None

    def test_expiration_set(self):
        exp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        cert = _make_cert(expires_at=exp)
        assert cert.expires_at == exp


class TestCertStatus:
    def test_all_statuses_exist(self):
        assert CertStatus.ACTIVE
        assert CertStatus.SUSPENDED
        assert CertStatus.REVOKED
        assert CertStatus.EXPIRED

    def test_status_names(self):
        assert CertStatus.ACTIVE.name == "ACTIVE"
        assert CertStatus.SUSPENDED.name == "SUSPENDED"
        assert CertStatus.REVOKED.name == "REVOKED"
        assert CertStatus.EXPIRED.name == "EXPIRED"


class TestCertificateSerialization:
    def test_json_round_trip(self):
        cert = _make_cert()
        json_str = cert.to_json()
        restored = AgentCertificate.from_json(json_str)
        assert restored.certificate_id == cert.certificate_id
        assert restored.agent_id == cert.agent_id
        assert restored.archetype == cert.archetype
        assert restored.organization == cert.organization
        assert restored.zone_path == cert.zone_path
        assert restored.issued_at == cert.issued_at
        assert restored.trust_score == cert.trust_score
        assert restored.behavioral_age == cert.behavioral_age
        assert restored.status == cert.status
        assert restored.public_key == cert.public_key
        assert restored.fingerprint == cert.fingerprint
        assert restored.governance_hash == cert.governance_hash
        assert restored.lineage == cert.lineage
        assert restored.issuer_signature == cert.issuer_signature
        assert restored.expires_at == cert.expires_at

    def test_json_round_trip_with_expiration(self):
        exp = datetime(2025, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        cert = _make_cert(expires_at=exp)
        restored = AgentCertificate.from_json(cert.to_json())
        assert restored.expires_at == exp

    def test_json_round_trip_with_lineage(self):
        cert = _make_cert(lineage="nmc-previous-id")
        restored = AgentCertificate.from_json(cert.to_json())
        assert restored.lineage == "nmc-previous-id"

    def test_json_is_canonical(self):
        cert = _make_cert()
        j = cert.to_json()
        # Canonical: sorted keys, no whitespace
        parsed = json.loads(j)
        re_serialized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        assert j == re_serialized

    def test_dict_round_trip(self):
        cert = _make_cert()
        d = cert.to_dict()
        restored = AgentCertificate.from_dict(d)
        assert restored.certificate_id == cert.certificate_id
        assert restored.status == cert.status

    def test_binary_round_trip(self):
        cert = _make_cert()
        binary = cert.to_binary()
        restored = AgentCertificate.from_binary(binary)
        assert restored.certificate_id == cert.certificate_id

    def test_to_signing_bytes_only_identity_fields(self):
        cert = _make_cert()
        signing_bytes = cert.to_signing_bytes()
        d = json.loads(signing_bytes)
        # Must include identity fields
        assert "certificate_id" in d
        assert "agent_id" in d
        assert "archetype" in d
        assert "organization" in d
        assert "zone_path" in d
        assert "issued_at" in d
        assert "public_key" in d
        assert "fingerprint" in d
        assert "lineage" in d
        assert "expires_at" in d
        # Must NOT include mutable state fields
        assert "issuer_signature" not in d
        assert "trust_score" not in d
        assert "behavioral_age" not in d
        assert "status" not in d
        assert "governance_hash" not in d

    def test_to_signing_bytes_is_deterministic(self):
        cert = _make_cert()
        assert cert.to_signing_bytes() == cert.to_signing_bytes()

    def test_to_signing_bytes_stable_across_state_changes(self):
        """Mutable state changes must not affect the signing payload."""
        cert = _make_cert()
        before = cert.to_signing_bytes()
        cert.trust_score = 0.99
        cert.behavioral_age = 42
        cert.status = CertStatus.SUSPENDED
        cert.governance_hash = "changed"
        after = cert.to_signing_bytes()
        assert before == after


class TestVerifyResults:
    def test_cert_verify_result_valid(self):
        r = CertVerifyResult(valid=True, certificate_id="nmc-1", status=CertStatus.ACTIVE)
        assert r.valid
        assert r.issues == []

    def test_cert_verify_result_invalid(self):
        r = CertVerifyResult(valid=False, certificate_id="nmc-1", issues=["expired"])
        assert not r.valid
        assert "expired" in r.issues

    def test_live_verify_result(self):
        r = LiveVerifyResult(
            valid=True,
            certificate_id="nmc-1",
            status=CertStatus.ACTIVE,
            trust_score=0.75,
            behavioral_age=42,
            governance_hash="abc",
            healthy=True,
        )
        assert r.valid
        assert r.trust_score == 0.75
        assert r.behavioral_age == 42
        assert r.healthy
