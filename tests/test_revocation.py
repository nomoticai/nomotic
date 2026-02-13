"""Tests for revocation list functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nomotic.authority import CertificateAuthority
from nomotic.certificate import CertStatus
from nomotic.keys import SigningKey
from nomotic.store import FileCertificateStore, MemoryCertificateStore


class TestRevocationListMemory:
    """Revocation list with MemoryCertificateStore."""

    def _make_ca(self) -> CertificateAuthority:
        sk, _vk = SigningKey.generate()
        store = MemoryCertificateStore()
        return CertificateAuthority(issuer_id="test", signing_key=sk, store=store)

    def test_empty_revocation_list(self) -> None:
        ca = self._make_ca()
        rl = ca.get_revocation_list()
        assert rl == []

    def test_revoked_cert_appears_in_list(self) -> None:
        ca = self._make_ca()
        cert, _sk = ca.issue("agent-1", "analytics", "org", "global")
        ca.revoke(cert.certificate_id, "decommissioned")
        rl = ca.get_revocation_list()
        assert len(rl) == 1
        assert rl[0]["certificate_id"] == cert.certificate_id
        assert rl[0]["agent_id"] == "agent-1"
        assert rl[0]["organization"] == "org"

    def test_multiple_revocations(self) -> None:
        ca = self._make_ca()
        cert1, _ = ca.issue("a1", "analytics", "org", "global")
        cert2, _ = ca.issue("a2", "analytics", "org", "global")
        cert3, _ = ca.issue("a3", "analytics", "org", "global")
        ca.revoke(cert1.certificate_id, "reason1")
        ca.revoke(cert3.certificate_id, "reason3")
        rl = ca.get_revocation_list()
        assert len(rl) == 2
        ids = {entry["certificate_id"] for entry in rl}
        assert cert1.certificate_id in ids
        assert cert3.certificate_id in ids
        assert cert2.certificate_id not in ids

    def test_active_certs_not_in_revocation_list(self) -> None:
        ca = self._make_ca()
        cert, _ = ca.issue("a1", "analytics", "org", "global")
        rl = ca.get_revocation_list()
        assert len(rl) == 0

    def test_suspended_certs_not_in_revocation_list(self) -> None:
        ca = self._make_ca()
        cert, _ = ca.issue("a1", "analytics", "org", "global")
        ca.suspend(cert.certificate_id, "testing")
        rl = ca.get_revocation_list()
        assert len(rl) == 0


class TestRevocationListFile:
    """Revocation list with FileCertificateStore."""

    def _make_ca(self, base: Path) -> CertificateAuthority:
        sk, _vk = SigningKey.generate()
        store = FileCertificateStore(base)
        return CertificateAuthority(issuer_id="test", signing_key=sk, store=store)

    def test_revocation_list_file_store(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ca = self._make_ca(Path(td))
            cert, _ = ca.issue("agent-1", "analytics", "org", "global")
            ca.revoke(cert.certificate_id, "decommissioned")
            rl = ca.get_revocation_list()
            assert len(rl) == 1
            assert rl[0]["certificate_id"] == cert.certificate_id


class TestListRevokedMemory:
    """Test list_revoked() on MemoryCertificateStore."""

    def test_list_revoked_empty(self) -> None:
        store = MemoryCertificateStore()
        assert store.list_revoked() == []

    def test_list_revoked_after_move(self) -> None:
        sk, _vk = SigningKey.generate()
        store = MemoryCertificateStore()
        ca = CertificateAuthority(issuer_id="test", signing_key=sk, store=store)
        cert, _ = ca.issue("a1", "arch", "org", "global")
        ca.revoke(cert.certificate_id, "reason")
        revoked = store.list_revoked()
        assert len(revoked) == 1
        assert revoked[0].certificate_id == cert.certificate_id


class TestListRevokedFile:
    """Test list_revoked() on FileCertificateStore."""

    def test_list_revoked_file_store(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sk, _vk = SigningKey.generate()
            store = FileCertificateStore(Path(td))
            ca = CertificateAuthority(issuer_id="test", signing_key=sk, store=store)
            cert, _ = ca.issue("a1", "arch", "org", "global")
            ca.revoke(cert.certificate_id, "reason")
            revoked = store.list_revoked()
            assert len(revoked) == 1
            assert revoked[0].certificate_id == cert.certificate_id
