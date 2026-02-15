"""Gateway middleware for validating Nomotic certificates on incoming requests.

The middleware validates ``X-Nomotic-*`` headers, makes trust-based access
decisions, and exposes certificate context to downstream handlers.

Works standalone, as WSGI middleware, or through framework adapters
(FastAPI, Flask).

Usage::

    from nomotic.middleware import NomoticGateway, GatewayConfig

    gateway = NomoticGateway(config=GatewayConfig(
        require_cert=True,
        min_trust=0.6,
        verify_url="https://governance.internal/v1",
    ))

    result = gateway.check(request_headers, request_body)
    if not result.allowed:
        return Response(403, result.to_json())

Zero runtime dependencies beyond the Python standard library.
"""

from __future__ import annotations

import io
import json as _json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from nomotic.authority import CertificateAuthority
from nomotic.headers import parse_headers

__all__ = [
    "NomoticGateway",
    "GatewayConfig",
    "GatewayResult",
    "REASON_ALLOWED",
    "REASON_NO_CERTIFICATE",
    "REASON_INVALID_HEADERS",
    "REASON_INVALID_SIGNATURE",
    "REASON_CERT_NOT_FOUND",
    "REASON_CERT_INACTIVE",
    "REASON_CERT_EXPIRED",
    "REASON_TRUST_BELOW_THRESHOLD",
    "REASON_AGE_BELOW_THRESHOLD",
    "REASON_ARCHETYPE_NOT_ALLOWED",
    "REASON_ZONE_NOT_ALLOWED",
    "REASON_ORG_NOT_ALLOWED",
    "REASON_VERIFICATION_FAILED",
]

# ── Denial reasons ─────────────────────────────────────────────────────

REASON_ALLOWED = "allowed"
REASON_NO_CERTIFICATE = "no_certificate"
REASON_INVALID_HEADERS = "invalid_headers"
REASON_INVALID_SIGNATURE = "invalid_signature"
REASON_CERT_NOT_FOUND = "certificate_not_found"
REASON_CERT_INACTIVE = "certificate_inactive"
REASON_CERT_EXPIRED = "certificate_expired"
REASON_TRUST_BELOW_THRESHOLD = "trust_below_threshold"
REASON_AGE_BELOW_THRESHOLD = "age_below_threshold"
REASON_ARCHETYPE_NOT_ALLOWED = "archetype_not_allowed"
REASON_ZONE_NOT_ALLOWED = "zone_not_allowed"
REASON_ORG_NOT_ALLOWED = "org_not_allowed"
REASON_VERIFICATION_FAILED = "verification_failed"


# ── GatewayResult ──────────────────────────────────────────────────────


@dataclass
class GatewayResult:
    """Outcome of a gateway check on an incoming request."""

    allowed: bool
    reason: str
    certificate_id: str | None = None
    trust_score: float | None = None
    behavioral_age: int | None = None
    archetype: str | None = None
    organization: str | None = None
    zone_path: str | None = None
    issues: list[str] = field(default_factory=list)
    verified_level: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict for error responses."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "certificate_id": self.certificate_id,
            "trust_score": self.trust_score,
            "behavioral_age": self.behavioral_age,
            "archetype": self.archetype,
            "organization": self.organization,
            "zone_path": self.zone_path,
            "issues": self.issues,
            "verified_level": self.verified_level,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return _json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def context(self) -> dict[str, Any]:
        """Certificate context for downstream handlers.

        Only populated when the request was allowed and had valid headers.
        """
        return {
            "certificate_id": self.certificate_id,
            "trust_score": self.trust_score,
            "behavioral_age": self.behavioral_age,
            "archetype": self.archetype,
            "organization": self.organization,
            "zone_path": self.zone_path,
            "verified_level": self.verified_level,
        }


# ── GatewayConfig ──────────────────────────────────────────────────────


@dataclass
class GatewayConfig:
    """Configuration for the Nomotic gateway middleware.

    Controls how incoming requests are validated and what trust
    levels are required for access.
    """

    require_cert: bool = False
    min_trust: float = 0.0
    min_age: int = 0
    allowed_archetypes: set[str] | None = None
    allowed_zones: set[str] | None = None
    allowed_orgs: set[str] | None = None
    verify_signature: bool = True
    verify_url: str | None = None
    local_ca: CertificateAuthority | None = None
    on_deny: str = "reject"


# ── Header name constants ──────────────────────────────────────────────

_HDR_CERT_ID = "X-Nomotic-Cert-ID"


def _has_nomotic_headers(headers: dict[str, str]) -> bool:
    """Check if the request contains any X-Nomotic-* headers."""
    return any(k.lower().startswith("x-nomotic-") for k in headers)


# ── NomoticGateway ─────────────────────────────────────────────────────


class NomoticGateway:
    """Middleware that validates Nomotic certificates on incoming requests.

    Works standalone or as middleware for WSGI, FastAPI, or Flask.

    Standalone usage::

        gateway = NomoticGateway(config=GatewayConfig(
            require_cert=True,
            min_trust=0.6,
            verify_url="https://governance.internal/v1",
        ))

        result = gateway.check(request_headers, request_body)
        if not result.allowed:
            return Response(403, result.to_json())

    WSGI usage::

        app = gateway.wrap_wsgi(app)
    """

    def __init__(self, config: GatewayConfig | None = None) -> None:
        self.config = config or GatewayConfig()

    def check(
        self,
        headers: dict[str, str],
        body: bytes = b"",
    ) -> GatewayResult:
        """Validate an incoming request against the gateway configuration.

        This is the core method. Framework adapters call this.

        Steps:
            1. Look for ``X-Nomotic-*`` headers
            2. If missing and ``require_cert=True`` -> deny (no_certificate)
            3. If missing and ``require_cert=False`` -> allow (no governance context)
            4. Parse headers
            5. If ``verify_signature=True`` and ``local_ca`` provided -> verify
               body signature
            6. If ``local_ca`` provided -> Level 2 verification (issuer sig,
               status, expiry)
            7. If ``verify_url`` provided -> Level 3 verification (remote,
               current trust)
            8. Check trust >= ``min_trust``
            9. Check age >= ``min_age``
            10. Check archetype in ``allowed_archetypes`` (if set)
            11. Check zone prefix in ``allowed_zones`` (if set)
            12. Check org in ``allowed_orgs`` (if set)
            13. Return :class:`GatewayResult`
        """
        cfg = self.config

        # 1-3. Check for nomotic headers
        if not _has_nomotic_headers(headers):
            if cfg.require_cert:
                return self._make_result(
                    allowed=cfg.on_deny == "flag",
                    reason=REASON_NO_CERTIFICATE,
                    issues=["no X-Nomotic-* headers present"],
                )
            return GatewayResult(allowed=True, reason=REASON_ALLOWED)

        # 4. Parse headers
        try:
            parsed = parse_headers(headers)
        except (KeyError, ValueError) as exc:
            return self._make_result(
                allowed=cfg.on_deny == "flag",
                reason=REASON_INVALID_HEADERS,
                issues=[f"header parse error: {exc}"],
            )

        # Build result fields from parsed headers
        result_fields: dict[str, Any] = {
            "certificate_id": parsed.certificate_id,
            "trust_score": parsed.trust_score,
            "behavioral_age": parsed.behavioral_age,
            "archetype": parsed.archetype,
            "zone_path": parsed.zone_path,
        }

        verified_level = 1
        issues: list[str] = []

        # 5-6. Local CA verification (Level 2)
        if cfg.local_ca is not None:
            cert = cfg.local_ca.get(parsed.certificate_id)
            if cert is None:
                return self._make_result(
                    allowed=cfg.on_deny == "flag",
                    reason=REASON_CERT_NOT_FOUND,
                    issues=["certificate not found in local CA"],
                    **result_fields,
                )

            # Verify body signature
            if cfg.verify_signature:
                from nomotic.keys import VerifyKey

                vk = VerifyKey.from_bytes(cert.public_key)
                if not vk.verify(parsed.signature, body):
                    return self._make_result(
                        allowed=cfg.on_deny == "flag",
                        reason=REASON_INVALID_SIGNATURE,
                        issues=["request body signature is invalid"],
                        **result_fields,
                    )

            # Level 2 certificate verification
            cert_result = cfg.local_ca.verify_certificate(cert)
            if not cert_result.valid:
                # Determine specific reason
                reason = REASON_VERIFICATION_FAILED
                for issue in cert_result.issues:
                    if "status" in issue.lower():
                        reason = REASON_CERT_INACTIVE
                        break
                    if "expired" in issue.lower():
                        reason = REASON_CERT_EXPIRED
                        break
                return self._make_result(
                    allowed=cfg.on_deny == "flag",
                    reason=reason,
                    issues=cert_result.issues,
                    **result_fields,
                )

            # Update result fields from authoritative cert
            result_fields["trust_score"] = cert.trust_score
            result_fields["behavioral_age"] = cert.behavioral_age
            result_fields["organization"] = cert.organization
            verified_level = 2

        # 7. Remote verification (Level 3)
        if cfg.verify_url is not None:
            remote_result = self._remote_verify(
                parsed.certificate_id, cfg.verify_url,
            )
            if remote_result is not None:
                if not remote_result.get("valid", False):
                    reason = REASON_VERIFICATION_FAILED
                    status = remote_result.get("status", "")
                    if status in ("SUSPENDED", "REVOKED"):
                        reason = REASON_CERT_INACTIVE
                    return self._make_result(
                        allowed=cfg.on_deny == "flag",
                        reason=reason,
                        issues=remote_result.get("issues", ["remote verification failed"]),
                        **result_fields,
                    )
                # Update with remote trust data
                result_fields["trust_score"] = remote_result.get(
                    "trust_score", result_fields.get("trust_score"),
                )
                result_fields["behavioral_age"] = remote_result.get(
                    "behavioral_age", result_fields.get("behavioral_age"),
                )
                result_fields["organization"] = remote_result.get(
                    "organization", result_fields.get("organization"),
                )
                verified_level = 3
            else:
                issues.append("remote verification unreachable")

        # 8. Check trust >= min_trust
        trust = result_fields.get("trust_score", 0.0)
        if trust is not None and trust < cfg.min_trust:
            return self._make_result(
                allowed=cfg.on_deny == "flag",
                reason=REASON_TRUST_BELOW_THRESHOLD,
                issues=[
                    f"trust score {trust} is below minimum {cfg.min_trust}",
                ],
                verified_level=verified_level,
                **result_fields,
            )

        # 9. Check age >= min_age
        age = result_fields.get("behavioral_age", 0)
        if age is not None and age < cfg.min_age:
            return self._make_result(
                allowed=cfg.on_deny == "flag",
                reason=REASON_AGE_BELOW_THRESHOLD,
                issues=[
                    f"behavioral age {age} is below minimum {cfg.min_age}",
                ],
                verified_level=verified_level,
                **result_fields,
            )

        # 10. Check archetype
        if cfg.allowed_archetypes is not None:
            archetype = result_fields.get("archetype", parsed.archetype)
            if archetype not in cfg.allowed_archetypes:
                return self._make_result(
                    allowed=cfg.on_deny == "flag",
                    reason=REASON_ARCHETYPE_NOT_ALLOWED,
                    issues=[
                        f"archetype '{archetype}' is not in allowed set",
                    ],
                    verified_level=verified_level,
                    **result_fields,
                )

        # 11. Check zone prefix
        if cfg.allowed_zones is not None:
            zone = result_fields.get("zone_path", parsed.zone_path)
            if not any(
                zone == z or zone.startswith(z + "/")
                for z in cfg.allowed_zones
            ):
                return self._make_result(
                    allowed=cfg.on_deny == "flag",
                    reason=REASON_ZONE_NOT_ALLOWED,
                    issues=[
                        f"zone '{zone}' does not match any allowed zone prefix",
                    ],
                    verified_level=verified_level,
                    **result_fields,
                )

        # 12. Check organization
        if cfg.allowed_orgs is not None:
            org = result_fields.get("organization")
            if org is None:
                # Try to get from parsed headers — not available in headers,
                # only from CA verification
                issues.append(
                    "organization filter set but org not available "
                    "(requires local_ca or verify_url)"
                )
            elif org not in cfg.allowed_orgs:
                return self._make_result(
                    allowed=cfg.on_deny == "flag",
                    reason=REASON_ORG_NOT_ALLOWED,
                    issues=[
                        f"organization '{org}' is not in allowed set",
                    ],
                    verified_level=verified_level,
                    **result_fields,
                )

        # 13. Allowed
        return GatewayResult(
            allowed=True,
            reason=REASON_ALLOWED,
            certificate_id=result_fields.get("certificate_id"),
            trust_score=result_fields.get("trust_score"),
            behavioral_age=result_fields.get("behavioral_age"),
            archetype=result_fields.get("archetype", parsed.archetype),
            organization=result_fields.get("organization"),
            zone_path=result_fields.get("zone_path", parsed.zone_path),
            issues=issues,
            verified_level=verified_level,
        )

    # ── WSGI middleware ────────────────────────────────────────────────

    def wrap_wsgi(self, app: Any) -> Any:
        """Return a WSGI application that validates requests before passing through.

        Denied requests get a 401 or 403 JSON response.
        Allowed requests get the original app called with
        ``nomotic.context`` added to environ.

        If ``on_deny="flag"`` in config, always passes through but adds
        ``X-Nomotic-Warning`` response header when validation fails.
        """
        gateway = self

        def middleware(environ: dict[str, Any], start_response: Any) -> Any:
            result = gateway.check_environ(environ)

            if not result.allowed:
                status_code = "401 Unauthorized" if result.reason == REASON_NO_CERTIFICATE else "403 Forbidden"
                body = result.to_json().encode("utf-8")
                response_headers = [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(body))),
                ]
                start_response(status_code, response_headers)
                return [body]

            # Inject context into environ
            environ["nomotic.context"] = result.context

            if result.issues:
                # Add warning header via a wrapper
                def warning_start_response(
                    status: str,
                    response_headers: list[tuple[str, str]],
                    exc_info: Any = None,
                ) -> Any:
                    response_headers.append(
                        ("X-Nomotic-Warning", "; ".join(result.issues))
                    )
                    return start_response(status, response_headers, exc_info)

                return app(environ, warning_start_response)

            return app(environ, start_response)

        return middleware

    def check_environ(self, environ: dict[str, Any]) -> GatewayResult:
        """Validate from a WSGI environ dict.

        Extracts headers and body from the environ.
        """
        headers = self._extract_headers(environ)
        body = self._read_body(environ)
        return self.check(headers, body)

    # ── Internal helpers ──────────────────────────────────────────────

    def _make_result(
        self,
        allowed: bool,
        reason: str,
        issues: list[str] | None = None,
        certificate_id: str | None = None,
        trust_score: float | None = None,
        behavioral_age: int | None = None,
        archetype: str | None = None,
        organization: str | None = None,
        zone_path: str | None = None,
        verified_level: int = 0,
    ) -> GatewayResult:
        """Build a GatewayResult with consistent defaults."""
        return GatewayResult(
            allowed=allowed,
            reason=reason,
            certificate_id=certificate_id,
            trust_score=trust_score,
            behavioral_age=behavioral_age,
            archetype=archetype,
            organization=organization,
            zone_path=zone_path,
            issues=issues or [],
            verified_level=verified_level,
        )

    def _remote_verify(
        self, cert_id: str, verify_url: str,
    ) -> dict[str, Any] | None:
        """Call the remote verification endpoint."""
        url = f"{verify_url.rstrip('/')}/v1/verify/{cert_id}"
        try:
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=10)
            return _json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError):
            return None

    # Map from WSGI environ keys to the canonical header names
    _WSGI_HEADER_MAP: dict[str, str] = {
        "HTTP_X_NOMOTIC_CERT_ID": "X-Nomotic-Cert-ID",
        "HTTP_X_NOMOTIC_OWNER": "X-Nomotic-Owner",
        "HTTP_X_NOMOTIC_TRUST": "X-Nomotic-Trust",
        "HTTP_X_NOMOTIC_AGE": "X-Nomotic-Age",
        "HTTP_X_NOMOTIC_ARCHETYPE": "X-Nomotic-Archetype",
        "HTTP_X_NOMOTIC_ZONE": "X-Nomotic-Zone",
        "HTTP_X_NOMOTIC_SIGNATURE": "X-Nomotic-Signature",
        "HTTP_X_NOMOTIC_GOV_HASH": "X-Nomotic-Gov-Hash",
    }

    @staticmethod
    def _extract_headers(environ: dict[str, Any]) -> dict[str, str]:
        """Extract HTTP headers from WSGI environ.

        WSGI stores headers as ``HTTP_<NAME>`` with underscores and uppercase.
        We convert them back to the standard ``X-Nomotic-*`` form using
        an explicit mapping to preserve exact casing (e.g. ``Cert-ID``
        not ``Cert-Id``).
        """
        headers: dict[str, str] = {}
        for key, value in environ.items():
            canonical = NomoticGateway._WSGI_HEADER_MAP.get(key)
            if canonical is not None:
                headers[canonical] = value
        return headers

    @staticmethod
    def _read_body(environ: dict[str, Any]) -> bytes:
        """Read the request body from WSGI environ."""
        content_length = environ.get("CONTENT_LENGTH", "")
        if not content_length:
            return b""
        try:
            length = int(content_length)
        except (ValueError, TypeError):
            return b""
        wsgi_input = environ.get("wsgi.input")
        if wsgi_input is None:
            return b""
        body = wsgi_input.read(length)
        # Reset the stream so downstream handlers can read it too
        if hasattr(wsgi_input, "seek"):
            wsgi_input.seek(0)
        return body
