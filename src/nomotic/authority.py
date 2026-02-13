"""Certificate Authority — issuance, verification, and lifecycle management.

The CertificateAuthority is the single point of control for agent certificates.
It issues new certificates, verifies them at three levels of depth, and manages
the full lifecycle (suspend, reactivate, revoke, renew, transfer).

The authority holds the Nomotic instance's issuer signing key and uses it to
sign every certificate it issues.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from nomotic.certificate import (
    AgentCertificate,
    CertStatus,
    CertVerifyResult,
    LiveVerifyResult,
    _generate_cert_id,
    _utcnow,
)
from nomotic.keys import SigningKey, VerifyKey
from nomotic.store import CertificateStore, MemoryCertificateStore

__all__ = ["CertificateAuthority"]

_BASELINE_TRUST = 0.50
_MIN_TRUST = 0.05
_MAX_TRUST = 0.95


def _compute_governance_hash(config: dict[str, Any] | None = None) -> str:
    """SHA-256 hash of the governance configuration."""
    if config is None:
        config = {}
    import json

    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CertificateAuthority:
    """Manages the full lifecycle of Nomotic agent certificates.

    The authority owns the issuer key pair and an in-memory (v1) or
    pluggable certificate store.

    Usage::

        sk, vk = SigningKey.generate()
        ca = CertificateAuthority(issuer_id="nomotic-instance-1", signing_key=sk)
        cert = ca.issue("agent-1", "customer-experience", "acme-corp", "global/us")
    """

    def __init__(
        self,
        issuer_id: str,
        signing_key: SigningKey,
        store: CertificateStore | None = None,
        governance_config: dict[str, Any] | None = None,
    ) -> None:
        self.issuer_id = issuer_id
        self._signing_key = signing_key
        self._verify_key = signing_key.verify_key()
        self._store: CertificateStore = store or MemoryCertificateStore()
        self._governance_config = governance_config
        self._governance_hash = _compute_governance_hash(governance_config)

    # ── Issuance ─────────────────────────────────────────────────────

    def issue(
        self,
        agent_id: str,
        archetype: str,
        organization: str,
        zone_path: str,
        *,
        owner: str = "",
        expires_in: timedelta | None = None,
        lineage: str | None = None,
        governance_config: dict[str, Any] | None = None,
    ) -> tuple[AgentCertificate, SigningKey]:
        """Issue a new agent certificate.

        Generates an agent key pair, creates the certificate with baseline
        trust and zero behavioral age, signs it with the issuer key, and
        stores it.

        *owner* is the person or entity accountable for this agent — who
        birthed it, who gets notified on status changes, and who answers
        for its actions.

        Returns the certificate and the agent's private signing key.
        The caller is responsible for storing the private key securely.
        """
        agent_sk, agent_vk = SigningKey.generate()
        now = _utcnow()

        gov_hash = (
            _compute_governance_hash(governance_config)
            if governance_config is not None
            else self._governance_hash
        )

        cert = AgentCertificate(
            certificate_id=_generate_cert_id(),
            agent_id=agent_id,
            owner=owner,
            archetype=archetype,
            organization=organization,
            zone_path=zone_path,
            issued_at=now,
            trust_score=_BASELINE_TRUST,
            behavioral_age=0,
            status=CertStatus.ACTIVE,
            public_key=agent_vk.to_bytes(),
            fingerprint=agent_vk.fingerprint(),
            governance_hash=gov_hash,
            lineage=lineage,
            issuer_signature=b"",  # placeholder — signed below
            expires_at=now + expires_in if expires_in else None,
        )

        # Sign the certificate with the issuer key
        cert.issuer_signature = self._signing_key.sign(cert.to_signing_bytes())

        self._store.save(cert)
        return cert, agent_sk

    # ── Verification (three levels) ──────────────────────────────────

    def verify_signature(
        self,
        certificate: AgentCertificate,
        signature: bytes,
        data: bytes,
    ) -> bool:
        """Level 1: verify the agent signed *data*.

        Uses the public key embedded in the certificate.
        """
        vk = VerifyKey.from_bytes(certificate.public_key)
        return vk.verify(signature, data)

    def verify_certificate(self, certificate: AgentCertificate) -> CertVerifyResult:
        """Level 2: verify issuer signature, status, and expiration."""
        issues: list[str] = []

        # Check issuer signature
        expected_sig = self._signing_key.sign(certificate.to_signing_bytes())
        if not self._verify_key.verify(certificate.issuer_signature, certificate.to_signing_bytes()):
            issues.append("invalid issuer signature")

        # Check status
        if certificate.status != CertStatus.ACTIVE:
            issues.append(f"certificate status is {certificate.status.name}")

        # Check expiration
        if certificate.expires_at is not None:
            now = _utcnow()
            if now >= certificate.expires_at:
                issues.append("certificate has expired")

        return CertVerifyResult(
            valid=len(issues) == 0,
            certificate_id=certificate.certificate_id,
            issues=issues,
            status=certificate.status,
        )

    def verify_live(self, certificate: AgentCertificate) -> LiveVerifyResult:
        """Level 3: full verification plus current behavioral state.

        Pulls the latest certificate from the store to get current trust
        and behavioral age.
        """
        # Get the latest version from the store
        current = self._store.get(certificate.certificate_id)
        if current is None:
            return LiveVerifyResult(
                valid=False,
                certificate_id=certificate.certificate_id,
                issues=["certificate not found in store"],
            )

        level2 = self.verify_certificate(current)
        healthy = level2.valid and current.trust_score >= _MIN_TRUST

        return LiveVerifyResult(
            valid=level2.valid,
            certificate_id=current.certificate_id,
            issues=level2.issues,
            status=current.status,
            trust_score=current.trust_score,
            behavioral_age=current.behavioral_age,
            governance_hash=current.governance_hash,
            healthy=healthy,
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    def suspend(self, certificate_id: str, reason: str) -> AgentCertificate:
        """Suspend a certificate. Only ACTIVE certificates can be suspended."""
        cert = self._require(certificate_id)
        if cert.status != CertStatus.ACTIVE:
            raise ValueError(
                f"Cannot suspend certificate in {cert.status.name} state"
            )
        cert.status = CertStatus.SUSPENDED
        cert.issuer_signature = self._signing_key.sign(cert.to_signing_bytes())
        self._store.update(cert)
        return cert

    def reactivate(self, certificate_id: str) -> AgentCertificate:
        """Reactivate a suspended certificate."""
        cert = self._require(certificate_id)
        if cert.status != CertStatus.SUSPENDED:
            raise ValueError(
                f"Cannot reactivate certificate in {cert.status.name} state"
            )
        cert.status = CertStatus.ACTIVE
        cert.issuer_signature = self._signing_key.sign(cert.to_signing_bytes())
        self._store.update(cert)
        return cert

    def revoke(self, certificate_id: str, reason: str) -> AgentCertificate:
        """Permanently revoke a certificate. Cannot be undone."""
        cert = self._require(certificate_id)
        if cert.status == CertStatus.REVOKED:
            raise ValueError("Certificate is already revoked")
        cert.status = CertStatus.REVOKED
        cert.issuer_signature = self._signing_key.sign(cert.to_signing_bytes())
        self._store.update(cert)
        self._store.move_to_revoked(certificate_id)
        return cert

    def renew(self, certificate_id: str) -> tuple[AgentCertificate, SigningKey]:
        """Issue a new certificate linked to the old one via lineage.

        The old certificate is not revoked — it remains in whatever state
        it was in.  The new certificate inherits agent_id, archetype,
        organization, zone_path, **trust_score**, and **behavioral_age**
        from the old certificate so the agent's earned reputation carries
        forward across renewals.
        """
        old = self._require(certificate_id)
        cert, agent_sk = self.issue(
            agent_id=old.agent_id,
            archetype=old.archetype,
            organization=old.organization,
            zone_path=old.zone_path,
            owner=old.owner,
            lineage=old.certificate_id,
        )
        # Carry forward earned reputation
        cert.trust_score = old.trust_score
        cert.behavioral_age = old.behavioral_age
        self._store.update(cert)
        return cert, agent_sk

    def transfer_zone(
        self, certificate_id: str, new_zone_path: str
    ) -> AgentCertificate:
        """Move a certificate to a different governance zone."""
        cert = self._require(certificate_id)
        cert.zone_path = new_zone_path
        cert.issuer_signature = self._signing_key.sign(cert.to_signing_bytes())
        self._store.update(cert)
        return cert

    # ── Lookup ───────────────────────────────────────────────────────

    def get(self, certificate_id: str) -> AgentCertificate | None:
        return self._store.get(certificate_id)

    def list(
        self,
        org: str | None = None,
        status: CertStatus | None = None,
        archetype: str | None = None,
    ) -> list[AgentCertificate]:
        return self._store.list(org=org, status=status, archetype=archetype)

    # ── Trust and behavioral age updates ─────────────────────────────

    def update_trust(self, certificate_id: str, new_trust: float) -> None:
        """Update the trust score on a certificate.

        Trust is a mutable behavioral field — it is *not* part of the
        signed identity payload, so no re-signing is needed.
        """
        cert = self._require(certificate_id)
        cert.trust_score = max(_MIN_TRUST, min(_MAX_TRUST, new_trust))
        self._store.update(cert)

    def increment_behavioral_age(self, certificate_id: str) -> None:
        """Increment behavioral age by one governed action.

        Behavioral age is a mutable state field — it is *not* part of the
        signed identity payload, so no re-signing is needed.
        """
        cert = self._require(certificate_id)
        cert.behavioral_age += 1
        self._store.update(cert)

    def record_action(
        self, certificate_id: str, new_trust: float
    ) -> None:
        """Update trust and increment behavioral age in a single store write.

        Called after each governed action to batch both mutable-state
        updates into one operation.
        """
        cert = self._require(certificate_id)
        cert.trust_score = max(_MIN_TRUST, min(_MAX_TRUST, new_trust))
        cert.behavioral_age += 1
        self._store.update(cert)

    def update_governance_hash(
        self, config: dict[str, Any] | None = None
    ) -> str:
        """Recompute and store the governance hash."""
        self._governance_config = config
        self._governance_hash = _compute_governance_hash(config)
        return self._governance_hash

    # ── Internal ─────────────────────────────────────────────────────

    def _require(self, certificate_id: str) -> AgentCertificate:
        cert = self._store.get(certificate_id)
        if cert is None:
            raise KeyError(f"Certificate not found: {certificate_id}")
        return cert
