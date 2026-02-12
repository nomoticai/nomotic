"""Agent Birth Certificate — the foundational identity primitive.

A birth certificate is a signed, structured identity document for a governed
AI agent. It carries the agent's identity, behavioral archetype, governance
zone, trust score, behavioral age, and cryptographic proof of issuance.

Certificates are the authoritative source of agent identity and trust.
Everything else in Nomotic builds on them.
"""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any

__all__ = [
    "AgentCertificate",
    "CertStatus",
    "CertVerifyResult",
    "LiveVerifyResult",
]


class CertStatus(Enum):
    """Lifecycle status of an agent certificate."""

    ACTIVE = auto()
    SUSPENDED = auto()
    REVOKED = auto()
    EXPIRED = auto()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _generate_cert_id() -> str:
    return f"nmc-{uuid.uuid4()}"


@dataclass
class AgentCertificate:
    """A Nomotic agent birth certificate.

    This is the agent's identity. The certificate is signed by the Nomotic
    instance's issuer key and carries the agent's public key, trust score,
    behavioral age, and governance configuration hash.

    The JSON serialization (sorted keys) is the canonical form used for
    hashing and signing.
    """

    certificate_id: str
    agent_id: str
    archetype: str
    organization: str
    zone_path: str
    issued_at: datetime
    trust_score: float
    behavioral_age: int
    status: CertStatus
    public_key: bytes
    fingerprint: str
    governance_hash: str
    lineage: str | None
    issuer_signature: bytes
    expires_at: datetime | None = None

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict with sorted keys."""
        return {
            "agent_id": self.agent_id,
            "archetype": self.archetype,
            "behavioral_age": self.behavioral_age,
            "certificate_id": self.certificate_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "fingerprint": self.fingerprint,
            "governance_hash": self.governance_hash,
            "issued_at": self.issued_at.isoformat(),
            "issuer_signature": base64.b64encode(self.issuer_signature).decode("ascii"),
            "lineage": self.lineage,
            "organization": self.organization,
            "public_key": base64.b64encode(self.public_key).decode("ascii"),
            "status": self.status.name,
            "trust_score": self.trust_score,
            "zone_path": self.zone_path,
        }

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, no extra whitespace)."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_signing_bytes(self) -> bytes:
        """Bytes used for issuer signature — immutable identity fields only.

        The signature proves *who this agent is* and *who issued it*, not
        the agent's current behavioral state.  Mutable fields (trust_score,
        behavioral_age, status, governance_hash) are deliberately excluded
        so that routine state updates don't invalidate the signature.

        Signed fields: certificate_id, agent_id, archetype, organization,
        zone_path, issued_at, public_key, fingerprint, lineage, expires_at.
        """
        d = {
            "agent_id": self.agent_id,
            "archetype": self.archetype,
            "certificate_id": self.certificate_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "fingerprint": self.fingerprint,
            "issued_at": self.issued_at.isoformat(),
            "lineage": self.lineage,
            "organization": self.organization,
            "public_key": base64.b64encode(self.public_key).decode("ascii"),
            "zone_path": self.zone_path,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def to_binary(self) -> bytes:
        """Compact binary representation (UTF-8 encoded canonical JSON)."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentCertificate:
        """Reconstruct a certificate from a dict (e.g. parsed JSON)."""
        expires_raw = d.get("expires_at")
        return cls(
            certificate_id=d["certificate_id"],
            agent_id=d["agent_id"],
            archetype=d["archetype"],
            organization=d["organization"],
            zone_path=d["zone_path"],
            issued_at=datetime.fromisoformat(d["issued_at"]),
            trust_score=d["trust_score"],
            behavioral_age=d["behavioral_age"],
            status=CertStatus[d["status"]],
            public_key=base64.b64decode(d["public_key"]),
            fingerprint=d["fingerprint"],
            governance_hash=d["governance_hash"],
            lineage=d.get("lineage"),
            issuer_signature=base64.b64decode(d["issuer_signature"]),
            expires_at=datetime.fromisoformat(expires_raw) if expires_raw else None,
        )

    @classmethod
    def from_json(cls, data: str) -> AgentCertificate:
        """Parse a certificate from canonical JSON."""
        return cls.from_dict(json.loads(data))

    @classmethod
    def from_binary(cls, data: bytes) -> AgentCertificate:
        """Parse a certificate from compact binary (UTF-8 JSON)."""
        return cls.from_json(data.decode("utf-8"))


# ── Verification result types ────────────────────────────────────────────


@dataclass
class CertVerifyResult:
    """Result of Level 2 certificate verification.

    Checks issuer signature validity, certificate status, and expiration.
    """

    valid: bool
    certificate_id: str
    issues: list[str] = field(default_factory=list)
    status: CertStatus | None = None


@dataclass
class LiveVerifyResult:
    """Result of Level 3 live certificate verification.

    Full verification plus current trust score, behavioral age, and health.
    """

    valid: bool
    certificate_id: str
    issues: list[str] = field(default_factory=list)
    status: CertStatus | None = None
    trust_score: float = 0.0
    behavioral_age: int = 0
    governance_hash: str = ""
    healthy: bool = False
