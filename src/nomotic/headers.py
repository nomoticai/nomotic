"""HTTP header utilities for Nomotic-governed requests.

Generates, parses, and verifies the ``X-Nomotic-*`` headers that carry
certificate identity and request signatures through HTTP.

Headers:
    X-Nomotic-Cert-ID      certificate_id
    X-Nomotic-Owner        owner
    X-Nomotic-Trust        trust_score
    X-Nomotic-Age          behavioral_age
    X-Nomotic-Archetype    archetype
    X-Nomotic-Zone         zone_path
    X-Nomotic-Signature    agent signature over the request body
    X-Nomotic-Gov-Hash     governance_hash
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

from nomotic.authority import CertificateAuthority
from nomotic.certificate import AgentCertificate
from nomotic.keys import SigningKey, VerifyKey

__all__ = [
    "CertificateHeaders",
    "RequestVerifyResult",
    "generate_headers",
    "parse_headers",
    "verify_request",
]

# Header name constants
_HDR_CERT_ID = "X-Nomotic-Cert-ID"
_HDR_OWNER = "X-Nomotic-Owner"
_HDR_TRUST = "X-Nomotic-Trust"
_HDR_AGE = "X-Nomotic-Age"
_HDR_ARCHETYPE = "X-Nomotic-Archetype"
_HDR_ZONE = "X-Nomotic-Zone"
_HDR_SIGNATURE = "X-Nomotic-Signature"
_HDR_GOV_HASH = "X-Nomotic-Gov-Hash"


@dataclass
class CertificateHeaders:
    """Parsed Nomotic HTTP headers."""

    certificate_id: str
    owner: str
    trust_score: float
    behavioral_age: int
    archetype: str
    zone_path: str
    signature: bytes
    governance_hash: str


@dataclass
class RequestVerifyResult:
    """Result of verifying a Nomotic-signed HTTP request."""

    valid: bool
    certificate_id: str
    issues: list[str] = field(default_factory=list)


def generate_headers(
    certificate: AgentCertificate,
    signing_key: SigningKey,
    request_body: bytes,
) -> dict[str, str]:
    """Generate ``X-Nomotic-*`` headers for an HTTP request.

    Signs *request_body* with the agent's signing key and includes all
    certificate metadata in the headers.
    """
    signature = signing_key.sign(request_body)
    return {
        _HDR_CERT_ID: certificate.certificate_id,
        _HDR_OWNER: certificate.owner,
        _HDR_TRUST: str(certificate.trust_score),
        _HDR_AGE: str(certificate.behavioral_age),
        _HDR_ARCHETYPE: certificate.archetype,
        _HDR_ZONE: certificate.zone_path,
        _HDR_SIGNATURE: base64.b64encode(signature).decode("ascii"),
        _HDR_GOV_HASH: certificate.governance_hash,
    }


def parse_headers(headers: dict[str, str]) -> CertificateHeaders:
    """Parse ``X-Nomotic-*`` headers into a structured object.

    Raises ``KeyError`` if any required header is missing.
    """
    return CertificateHeaders(
        certificate_id=headers[_HDR_CERT_ID],
        owner=headers[_HDR_OWNER],
        trust_score=float(headers[_HDR_TRUST]),
        behavioral_age=int(headers[_HDR_AGE]),
        archetype=headers[_HDR_ARCHETYPE],
        zone_path=headers[_HDR_ZONE],
        signature=base64.b64decode(headers[_HDR_SIGNATURE]),
        governance_hash=headers[_HDR_GOV_HASH],
    )


def verify_request(
    headers: dict[str, str],
    body: bytes,
    verifier: CertificateAuthority,
) -> RequestVerifyResult:
    """Validate an HTTP request's signature and optionally the certificate.

    1. Parses headers.
    2. Looks up the certificate by ID.
    3. Verifies the request body signature against the certificate's public key.
    4. Runs Level 2 certificate verification (issuer signature, status, expiry).
    """
    issues: list[str] = []

    try:
        parsed = parse_headers(headers)
    except (KeyError, ValueError) as exc:
        return RequestVerifyResult(
            valid=False,
            certificate_id="",
            issues=[f"header parse error: {exc}"],
        )

    cert = verifier.get(parsed.certificate_id)
    if cert is None:
        return RequestVerifyResult(
            valid=False,
            certificate_id=parsed.certificate_id,
            issues=["certificate not found"],
        )

    # Verify the agent's signature over the body
    if not verifier.verify_signature(cert, parsed.signature, body):
        issues.append("invalid request signature")

    # Verify the certificate itself
    cert_result = verifier.verify_certificate(cert)
    if not cert_result.valid:
        issues.extend(cert_result.issues)

    return RequestVerifyResult(
        valid=len(issues) == 0,
        certificate_id=parsed.certificate_id,
        issues=issues,
    )
