"""Tests for the certificate authority."""

import pytest
from datetime import timedelta

from nomotic.authority import CertificateAuthority
from nomotic.certificate import AgentCertificate, CertStatus
from nomotic.keys import SigningKey, VerifyKey
from nomotic.store import MemoryCertificateStore


def _ca() -> CertificateAuthority:
    sk, _vk = SigningKey.generate()
    return CertificateAuthority(issuer_id="test-issuer", signing_key=sk)


class TestCertificateIssuance:
    def test_issue_returns_certificate_and_key(self):
        ca = _ca()
        cert, agent_sk = ca.issue("agent-1", "customer-experience", "acme", "global/us")
        assert isinstance(cert, AgentCertificate)
        assert isinstance(agent_sk, SigningKey)

    def test_issued_cert_has_correct_fields(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "customer-experience", "acme", "global/us")
        assert cert.certificate_id.startswith("nmc-")
        assert cert.agent_id == "agent-1"
        assert cert.archetype == "customer-experience"
        assert cert.organization == "acme"
        assert cert.zone_path == "global/us"

    def test_baseline_trust_and_zero_age(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "claims-processor", "acme", "global")
        assert cert.trust_score == 0.50
        assert cert.behavioral_age == 0

    def test_active_status(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "claims-processor", "acme", "global")
        assert cert.status == CertStatus.ACTIVE

    def test_no_lineage_by_default(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "claims-processor", "acme", "global")
        assert cert.lineage is None

    def test_issuer_signature_is_set(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "claims-processor", "acme", "global")
        assert len(cert.issuer_signature) > 0

    def test_fingerprint_matches_public_key(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "claims-processor", "acme", "global")
        vk = VerifyKey.from_bytes(cert.public_key)
        assert cert.fingerprint == vk.fingerprint()

    def test_agent_key_can_sign_and_verify(self):
        ca = _ca()
        cert, agent_sk = ca.issue("agent-1", "claims-processor", "acme", "global")
        data = b"test data"
        sig = agent_sk.sign(data)
        vk = VerifyKey.from_bytes(cert.public_key)
        assert vk.verify(sig, data)

    def test_each_cert_gets_unique_id(self):
        ca = _ca()
        cert1, _ = ca.issue("agent-1", "arch", "org", "zone")
        cert2, _ = ca.issue("agent-2", "arch", "org", "zone")
        assert cert1.certificate_id != cert2.certificate_id

    def test_each_cert_gets_unique_key_pair(self):
        ca = _ca()
        cert1, sk1 = ca.issue("agent-1", "arch", "org", "zone")
        cert2, sk2 = ca.issue("agent-2", "arch", "org", "zone")
        assert cert1.public_key != cert2.public_key
        assert sk1.to_bytes() != sk2.to_bytes()

    def test_issue_with_expiration(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone", expires_in=timedelta(days=30))
        assert cert.expires_at is not None

    def test_issue_with_lineage(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone", lineage="nmc-old")
        assert cert.lineage == "nmc-old"

    def test_certificate_stored_after_issue(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        retrieved = ca.get(cert.certificate_id)
        assert retrieved is not None
        assert retrieved.certificate_id == cert.certificate_id


class TestSignatureVerification:
    def test_verify_agent_signature(self):
        """Level 1: agent signed this data."""
        ca = _ca()
        cert, agent_sk = ca.issue("agent-1", "arch", "org", "zone")
        data = b"request body"
        sig = agent_sk.sign(data)
        assert ca.verify_signature(cert, sig, data)

    def test_verify_agent_signature_wrong_data(self):
        ca = _ca()
        cert, agent_sk = ca.issue("agent-1", "arch", "org", "zone")
        sig = agent_sk.sign(b"original")
        assert not ca.verify_signature(cert, sig, b"tampered")

    def test_verify_certificate_valid(self):
        """Level 2: issuer signature + status + expiration."""
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        result = ca.verify_certificate(cert)
        assert result.valid
        assert result.issues == []
        assert result.status == CertStatus.ACTIVE

    def test_verify_certificate_tampered(self):
        """Tampered certificate should fail Level 2."""
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        cert.trust_score = 0.99  # Tamper without re-signing
        result = ca.verify_certificate(cert)
        assert not result.valid
        assert any("signature" in i for i in result.issues)

    def test_verify_certificate_suspended(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.suspend(cert.certificate_id, "test")
        cert = ca.get(cert.certificate_id)
        result = ca.verify_certificate(cert)
        assert not result.valid
        assert any("SUSPENDED" in i for i in result.issues)

    def test_verify_live(self):
        """Level 3: full verification + behavioral state."""
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        result = ca.verify_live(cert)
        assert result.valid
        assert result.trust_score == 0.50
        assert result.behavioral_age == 0
        assert result.healthy

    def test_verify_live_not_found(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        # Create a different cert not in the store
        from nomotic.certificate import AgentCertificate, _generate_cert_id, _utcnow
        fake = AgentCertificate(
            certificate_id="nmc-nonexistent",
            agent_id="ghost",
            archetype="arch",
            organization="org",
            zone_path="zone",
            issued_at=_utcnow(),
            trust_score=0.5,
            behavioral_age=0,
            status=CertStatus.ACTIVE,
            public_key=cert.public_key,
            fingerprint=cert.fingerprint,
            governance_hash="hash",
            lineage=None,
            issuer_signature=b"\x00" * 32,
        )
        result = ca.verify_live(fake)
        assert not result.valid
        assert any("not found" in i for i in result.issues)


class TestCertificateLifecycle:
    def test_suspend(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        suspended = ca.suspend(cert.certificate_id, "policy violation")
        assert suspended.status == CertStatus.SUSPENDED

    def test_suspend_non_active_raises(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.suspend(cert.certificate_id, "test")
        with pytest.raises(ValueError, match="SUSPENDED"):
            ca.suspend(cert.certificate_id, "again")

    def test_reactivate(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.suspend(cert.certificate_id, "test")
        reactivated = ca.reactivate(cert.certificate_id)
        assert reactivated.status == CertStatus.ACTIVE

    def test_reactivate_non_suspended_raises(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        with pytest.raises(ValueError, match="ACTIVE"):
            ca.reactivate(cert.certificate_id)

    def test_revoke(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        revoked = ca.revoke(cert.certificate_id, "decommissioned")
        assert revoked.status == CertStatus.REVOKED

    def test_revoke_already_revoked_raises(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.revoke(cert.certificate_id, "first")
        with pytest.raises(ValueError, match="already revoked"):
            ca.revoke(cert.certificate_id, "second")

    def test_revoke_moves_to_revoked_store(self):
        store = MemoryCertificateStore()
        sk, _ = SigningKey.generate()
        ca = CertificateAuthority(issuer_id="test", signing_key=sk, store=store)
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.revoke(cert.certificate_id, "test")
        # Active list should be empty
        assert store.list() == []
        # But get should still find it in revoked
        assert store.get(cert.certificate_id) is not None

    def test_suspend_reactivate_cycle(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.suspend(cert.certificate_id, "temp")
        ca.reactivate(cert.certificate_id)
        current = ca.get(cert.certificate_id)
        assert current.status == CertStatus.ACTIVE

    def test_revoked_cert_fails_verification(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.revoke(cert.certificate_id, "test")
        current = ca.get(cert.certificate_id)
        result = ca.verify_certificate(current)
        assert not result.valid


class TestRenewal:
    def test_renew_creates_new_cert_with_lineage(self):
        ca = _ca()
        old, _ = ca.issue("agent-1", "arch", "org", "zone")
        new, new_sk = ca.renew(old.certificate_id)
        assert new.certificate_id != old.certificate_id
        assert new.lineage == old.certificate_id

    def test_renewed_cert_inherits_identity(self):
        ca = _ca()
        old, _ = ca.issue("agent-1", "customer-experience", "acme", "global/us")
        new, _ = ca.renew(old.certificate_id)
        assert new.agent_id == old.agent_id
        assert new.archetype == old.archetype
        assert new.organization == old.organization
        assert new.zone_path == old.zone_path

    def test_renewed_cert_starts_fresh(self):
        ca = _ca()
        old, _ = ca.issue("agent-1", "arch", "org", "zone")
        # Modify old cert's behavioral age
        ca.increment_behavioral_age(old.certificate_id)
        ca.update_trust(old.certificate_id, 0.75)
        new, _ = ca.renew(old.certificate_id)
        assert new.behavioral_age == 0
        assert new.trust_score == 0.50

    def test_renew_nonexistent_raises(self):
        ca = _ca()
        with pytest.raises(KeyError):
            ca.renew("nmc-nonexistent")


class TestZoneTransfer:
    def test_transfer_zone(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "global/us")
        transferred = ca.transfer_zone(cert.certificate_id, "global/eu/acme/retail")
        assert transferred.zone_path == "global/eu/acme/retail"

    def test_transfer_zone_updates_store(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "global/us")
        ca.transfer_zone(cert.certificate_id, "global/eu")
        current = ca.get(cert.certificate_id)
        assert current.zone_path == "global/eu"


class TestTrustAndAge:
    def test_update_trust(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.update_trust(cert.certificate_id, 0.75)
        current = ca.get(cert.certificate_id)
        assert current.trust_score == 0.75

    def test_trust_bounded_high(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.update_trust(cert.certificate_id, 1.5)
        current = ca.get(cert.certificate_id)
        assert current.trust_score <= 0.95

    def test_trust_bounded_low(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.update_trust(cert.certificate_id, -0.5)
        current = ca.get(cert.certificate_id)
        assert current.trust_score >= 0.05

    def test_increment_behavioral_age(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.increment_behavioral_age(cert.certificate_id)
        current = ca.get(cert.certificate_id)
        assert current.behavioral_age == 1

    def test_behavioral_age_monotonic(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        for i in range(10):
            ca.increment_behavioral_age(cert.certificate_id)
        current = ca.get(cert.certificate_id)
        assert current.behavioral_age == 10


class TestLookup:
    def test_get_existing(self):
        ca = _ca()
        cert, _ = ca.issue("agent-1", "arch", "org", "zone")
        found = ca.get(cert.certificate_id)
        assert found is not None
        assert found.certificate_id == cert.certificate_id

    def test_get_nonexistent(self):
        ca = _ca()
        assert ca.get("nmc-nonexistent") is None

    def test_list_all(self):
        ca = _ca()
        ca.issue("agent-1", "arch", "org-a", "zone")
        ca.issue("agent-2", "arch", "org-b", "zone")
        assert len(ca.list()) == 2

    def test_list_by_org(self):
        ca = _ca()
        ca.issue("agent-1", "arch", "acme", "zone")
        ca.issue("agent-2", "arch", "globex", "zone")
        assert len(ca.list(org="acme")) == 1

    def test_list_by_status(self):
        ca = _ca()
        cert1, _ = ca.issue("agent-1", "arch", "org", "zone")
        ca.issue("agent-2", "arch", "org", "zone")
        ca.suspend(cert1.certificate_id, "test")
        assert len(ca.list(status=CertStatus.ACTIVE)) == 1
        assert len(ca.list(status=CertStatus.SUSPENDED)) == 1

    def test_list_by_archetype(self):
        ca = _ca()
        ca.issue("agent-1", "customer-experience", "org", "zone")
        ca.issue("agent-2", "claims-processor", "org", "zone")
        assert len(ca.list(archetype="customer-experience")) == 1
