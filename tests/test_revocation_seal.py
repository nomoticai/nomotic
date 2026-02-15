"""Tests for revocation sealing — immutable final records for revoked agents."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nomotic.certificate import AgentCertificate, CertStatus
from nomotic.keys import SigningKey
from nomotic.revocation import create_revocation_seal, load_revocation_seal


def _make_cert(**overrides) -> AgentCertificate:
    """Create a certificate with sensible defaults for testing."""
    _sk, vk = SigningKey.generate()
    defaults = {
        "certificate_id": "nmc-test-1234",
        "agent_id": "test-agent",
        "owner": "alice@acme.com",
        "archetype": "customer-experience",
        "organization": "acme-corp",
        "zone_path": "global",
        "issued_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "trust_score": 0.50,
        "behavioral_age": 0,
        "status": CertStatus.REVOKED,
        "public_key": vk.to_bytes(),
        "fingerprint": vk.fingerprint(),
        "governance_hash": "abc123",
        "lineage": None,
        "issuer_signature": b"\x00" * 32,
        "expires_at": None,
        "agent_numeric_id": 1000,
    }
    defaults.update(overrides)
    return AgentCertificate(**defaults)


class TestCreateRevocationSeal:
    """Test revocation seal creation."""

    def test_creates_seal_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cert = _make_cert(trust_score=0.15, behavioral_age=50)
            seal = create_revocation_seal(
                base,
                agent_id=1000,
                agent_name="test-agent",
                certificate_id="nmc-test-1234",
                reason="Decommissioned after audit",
                cert=cert,
            )
            assert seal["agent_id"] == 1000
            assert seal["agent_name"] == "test-agent"
            assert seal["certificate_id"] == "nmc-test-1234"
            assert seal["reason"] == "Decommissioned after audit"
            assert seal["revoked_by"] == "cli-user"
            assert "revoked_at" in seal
            assert seal["final_trust"] == 0.15

            # Check file was written
            seal_path = base / "revocations" / "1000.json"
            assert seal_path.exists()
            loaded = json.loads(seal_path.read_text())
            assert loaded["agent_id"] == 1000

    def test_seal_with_audit_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cert = _make_cert(trust_score=0.3, behavioral_age=100)
            audit_records = [
                {"verdict": "ALLOW", "agent_id": "test-agent"},
                {"verdict": "DENY", "agent_id": "test-agent"},
                {"verdict": "ALLOW", "agent_id": "test-agent"},
                {"verdict": "ESCALATE", "agent_id": "test-agent"},
            ]
            seal = create_revocation_seal(
                base,
                agent_id=1000,
                agent_name="test-agent",
                certificate_id="nmc-test-1234",
                reason="Test",
                cert=cert,
                audit_records=audit_records,
            )
            assert seal["total_evaluations"] == 4
            assert seal["total_violations"] == 2  # DENY + ESCALATE
            assert seal["audit_seal"].startswith("sha256:")
            assert seal["chain_records"] == 4

    def test_seal_without_cert(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            seal = create_revocation_seal(
                base,
                agent_id=1001,
                agent_name="no-cert-agent",
                certificate_id="nmc-no-cert",
                reason="Gone",
            )
            assert seal["final_trust"] == 0.05
            assert seal["total_evaluations"] == 0

    def test_peak_trust_at_least_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cert = _make_cert(trust_score=0.05)
            seal = create_revocation_seal(
                base,
                agent_id=1000,
                agent_name="bot",
                certificate_id="nmc-1",
                reason="bad",
                cert=cert,
            )
            assert seal["peak_trust"] >= 0.50  # At least the starting trust


class TestLoadRevocationSeal:
    """Test loading revocation seals."""

    def test_load_existing_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cert = _make_cert()
            create_revocation_seal(
                base,
                agent_id=1000,
                agent_name="bot",
                certificate_id="nmc-1",
                reason="test",
                cert=cert,
            )
            loaded = load_revocation_seal(base, 1000)
            assert loaded is not None
            assert loaded["agent_name"] == "bot"
            assert loaded["reason"] == "test"

    def test_load_nonexistent_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = load_revocation_seal(Path(tmp), 9999)
            assert result is None


class TestRevocationSealIntegrity:
    """Test that the audit seal hash is deterministic and verifiable."""

    def test_same_records_same_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            records = [{"verdict": "ALLOW", "id": "1"}, {"verdict": "DENY", "id": "2"}]
            seal1 = create_revocation_seal(
                base,
                agent_id=1000,
                agent_name="bot",
                certificate_id="nmc-1",
                reason="test",
                audit_records=records,
            )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            seal2 = create_revocation_seal(
                base,
                agent_id=1000,
                agent_name="bot",
                certificate_id="nmc-1",
                reason="test",
                audit_records=records,
            )

        assert seal1["audit_seal"] == seal2["audit_seal"]

    def test_different_records_different_seal(self):
        records_a = [{"verdict": "ALLOW", "id": "1"}]
        records_b = [{"verdict": "DENY", "id": "1"}]

        with tempfile.TemporaryDirectory() as tmp:
            seal_a = create_revocation_seal(
                Path(tmp),
                agent_id=1000,
                agent_name="bot",
                certificate_id="nmc-1",
                reason="test",
                audit_records=records_a,
            )
        with tempfile.TemporaryDirectory() as tmp:
            seal_b = create_revocation_seal(
                Path(tmp),
                agent_id=1000,
                agent_name="bot",
                certificate_id="nmc-1",
                reason="test",
                audit_records=records_b,
            )
        assert seal_a["audit_seal"] != seal_b["audit_seal"]
