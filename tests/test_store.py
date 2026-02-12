"""Tests for certificate storage backends."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from nomotic.certificate import AgentCertificate, CertStatus
from nomotic.keys import SigningKey, VerifyKey
from nomotic.store import MemoryCertificateStore, FileCertificateStore


def _cert(cert_id: str = "nmc-test-1", **overrides) -> AgentCertificate:
    _sk, vk = SigningKey.generate()
    defaults = {
        "certificate_id": cert_id,
        "agent_id": "agent-1",
        "archetype": "customer-experience",
        "organization": "acme",
        "zone_path": "global/us",
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


class TestMemoryCertificateStore:
    def test_save_and_get(self):
        store = MemoryCertificateStore()
        cert = _cert()
        store.save(cert)
        retrieved = store.get("nmc-test-1")
        assert retrieved is not None
        assert retrieved.certificate_id == "nmc-test-1"

    def test_get_nonexistent(self):
        store = MemoryCertificateStore()
        assert store.get("nmc-nope") is None

    def test_list_all(self):
        store = MemoryCertificateStore()
        store.save(_cert("nmc-1"))
        store.save(_cert("nmc-2"))
        assert len(store.list()) == 2

    def test_list_by_org(self):
        store = MemoryCertificateStore()
        store.save(_cert("nmc-1", organization="acme"))
        store.save(_cert("nmc-2", organization="globex"))
        assert len(store.list(org="acme")) == 1

    def test_list_by_status(self):
        store = MemoryCertificateStore()
        store.save(_cert("nmc-1", status=CertStatus.ACTIVE))
        store.save(_cert("nmc-2", status=CertStatus.SUSPENDED))
        assert len(store.list(status=CertStatus.ACTIVE)) == 1

    def test_list_by_archetype(self):
        store = MemoryCertificateStore()
        store.save(_cert("nmc-1", archetype="customer-experience"))
        store.save(_cert("nmc-2", archetype="claims-processor"))
        assert len(store.list(archetype="customer-experience")) == 1

    def test_update(self):
        store = MemoryCertificateStore()
        cert = _cert()
        store.save(cert)
        cert.trust_score = 0.75
        store.update(cert)
        retrieved = store.get("nmc-test-1")
        assert retrieved.trust_score == 0.75

    def test_move_to_revoked(self):
        store = MemoryCertificateStore()
        cert = _cert()
        store.save(cert)
        store.move_to_revoked("nmc-test-1")
        # Should be gone from active list
        assert store.list() == []
        # But still accessible via get (from revoked)
        assert store.get("nmc-test-1") is not None

    def test_move_nonexistent_no_error(self):
        store = MemoryCertificateStore()
        store.move_to_revoked("nmc-nope")  # Should not raise


class TestFileCertificateStore:
    def test_save_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCertificateStore(tmp)
            cert = _cert()
            store.save(cert)
            retrieved = store.get("nmc-test-1")
            assert retrieved is not None
            assert retrieved.certificate_id == "nmc-test-1"

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCertificateStore(tmp)
            assert store.get("nmc-nope") is None

    def test_list_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCertificateStore(tmp)
            store.save(_cert("nmc-1"))
            store.save(_cert("nmc-2"))
            assert len(store.list()) == 2

    def test_list_by_org(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCertificateStore(tmp)
            store.save(_cert("nmc-1", organization="acme"))
            store.save(_cert("nmc-2", organization="globex"))
            assert len(store.list(org="acme")) == 1

    def test_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCertificateStore(tmp)
            cert = _cert()
            store.save(cert)
            cert.trust_score = 0.80
            store.update(cert)
            retrieved = store.get("nmc-test-1")
            assert retrieved.trust_score == 0.80

    def test_move_to_revoked(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCertificateStore(tmp)
            cert = _cert()
            store.save(cert)
            store.move_to_revoked("nmc-test-1")
            assert store.list() == []
            # Should still be findable in revoked dir
            assert store.get("nmc-test-1") is not None

    def test_json_round_trip_preserves_all_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCertificateStore(tmp)
            cert = _cert(
                lineage="nmc-old",
                trust_score=0.73,
                behavioral_age=42,
            )
            store.save(cert)
            retrieved = store.get("nmc-test-1")
            assert retrieved.lineage == "nmc-old"
            assert retrieved.trust_score == 0.73
            assert retrieved.behavioral_age == 42

    def test_save_and_get_agent_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCertificateStore(tmp)
            key_bytes = b"\x01" * 32
            store.save_agent_key("nmc-test-1", key_bytes)
            assert store.get_agent_key("nmc-test-1") == key_bytes

    def test_save_and_get_agent_pub(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCertificateStore(tmp)
            pub_bytes = b"\x02" * 32
            store.save_agent_pub("nmc-test-1", pub_bytes)
            assert store.get_agent_pub("nmc-test-1") == pub_bytes

    def test_get_nonexistent_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCertificateStore(tmp)
            assert store.get_agent_key("nmc-nope") is None
            assert store.get_agent_pub("nmc-nope") is None

    def test_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "deep" / "path"
            store = FileCertificateStore(base)
            assert (base / "certs").is_dir()
            assert (base / "revoked").is_dir()
