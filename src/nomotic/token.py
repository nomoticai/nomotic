"""Governance Token — JWT-based proof of governance evaluation.

A Governance Token is a signed, verifiable, time-bound artifact that proves
governance evaluation occurred. Execution environments validate tokens
before permitting actions.

Uses HMAC-SHA256 for signing, consistent with the project's zero-dependency
approach. The JWT format follows RFC 7519 with Nomotic-specific claims
prefixed ``nomo_``.

Token lifetimes:
    - single: 60 seconds (one specific action)
    - class:  15 minutes (pre-authorized action class)
    - session: 1 hour (extended session)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from nomotic.protocol import METHODS, VALID_FLOWS, VALID_SCOPES, Condition

__all__ = [
    "GovernanceToken",
    "TokenClaims",
    "TokenValidationResult",
    "TokenValidator",
]

# Default lifetimes in seconds per scope.
_DEFAULT_LIFETIMES: dict[str, int] = {
    "single": 60,
    "class": 900,    # 15 minutes
    "session": 3600,  # 1 hour
}


def _b64url_encode(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """Base64url decode with padding restoration."""
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _jwt_sign(header: dict[str, Any], payload: dict[str, Any], secret: bytes) -> str:
    """Create a signed JWT from header and payload."""
    h = _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode())
    p = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signing_input = f"{h}.{p}"
    sig = hmac.new(secret, signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(sig)}"


def _jwt_verify(token: str, secret: bytes) -> dict[str, Any] | None:
    """Verify a JWT and return the payload, or None if invalid."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    signing_input = f"{parts[0]}.{parts[1]}"
    expected_sig = hmac.new(secret, signing_input.encode("ascii"), hashlib.sha256).digest()
    try:
        actual_sig = _b64url_decode(parts[2])
    except Exception:
        return None
    if not hmac.compare_digest(expected_sig, actual_sig):
        return None
    try:
        payload = json.loads(_b64url_decode(parts[1]))
    except Exception:
        return None
    return payload


# ── Token Claims ───────────────────────────────────────────────────────


@dataclass
class TokenClaims:
    """All claims carried by a Nomotic Governance Token."""

    # Standard JWT claims
    iss: str  # Evaluator ID
    sub: str  # Agent ID
    exp: int  # Expiration (Unix timestamp)
    iat: int  # Issued at (Unix timestamp)
    jti: str  # Unique token ID

    # Nomotic claims
    nomo_verdict: str
    nomo_artifact_hash: str
    nomo_method: str
    nomo_action_target: str
    nomo_flow: str
    nomo_scope: str

    # Optional Nomotic claims
    aud: str | list[str] = ""
    nomo_conditions: list[dict[str, Any]] = field(default_factory=list)
    nomo_authority_envelope: str = ""
    nomo_config_version: str = ""
    nomo_ucs: float | None = None
    nomo_trust: float | None = None
    nomo_scope_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "iss": self.iss,
            "sub": self.sub,
            "exp": self.exp,
            "iat": self.iat,
            "jti": self.jti,
            "nomo_verdict": self.nomo_verdict,
            "nomo_artifact_hash": self.nomo_artifact_hash,
            "nomo_method": self.nomo_method,
            "nomo_action_target": self.nomo_action_target,
            "nomo_flow": self.nomo_flow,
            "nomo_scope": self.nomo_scope,
        }
        if self.aud:
            d["aud"] = self.aud
        if self.nomo_conditions:
            d["nomo_conditions"] = self.nomo_conditions
        if self.nomo_authority_envelope:
            d["nomo_authority_envelope"] = self.nomo_authority_envelope
        if self.nomo_config_version:
            d["nomo_config_version"] = self.nomo_config_version
        if self.nomo_ucs is not None:
            d["nomo_ucs"] = self.nomo_ucs
        if self.nomo_trust is not None:
            d["nomo_trust"] = self.nomo_trust
        if self.nomo_scope_constraints:
            d["nomo_scope_constraints"] = self.nomo_scope_constraints
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenClaims:
        return cls(
            iss=data["iss"],
            sub=data["sub"],
            exp=data["exp"],
            iat=data["iat"],
            jti=data["jti"],
            nomo_verdict=data["nomo_verdict"],
            nomo_artifact_hash=data["nomo_artifact_hash"],
            nomo_method=data["nomo_method"],
            nomo_action_target=data["nomo_action_target"],
            nomo_flow=data["nomo_flow"],
            nomo_scope=data["nomo_scope"],
            aud=data.get("aud", ""),
            nomo_conditions=data.get("nomo_conditions", []),
            nomo_authority_envelope=data.get("nomo_authority_envelope", ""),
            nomo_config_version=data.get("nomo_config_version", ""),
            nomo_ucs=data.get("nomo_ucs"),
            nomo_trust=data.get("nomo_trust"),
            nomo_scope_constraints=data.get("nomo_scope_constraints", {}),
        )


# ── Token Validation Result ────────────────────────────────────────────


@dataclass
class TokenValidationResult:
    """Result of token validation."""

    valid: bool
    verdict: str = ""
    conditions: list[dict[str, Any]] = field(default_factory=list)
    expires_at: str = ""
    claims: TokenClaims | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"valid": self.valid}
        if self.verdict:
            d["verdict"] = self.verdict
        if self.conditions:
            d["conditions"] = self.conditions
        if self.expires_at:
            d["expires_at"] = self.expires_at
        if self.error:
            d["error"] = self.error
        return d


# ── Governance Token ───────────────────────────────────────────────────


class GovernanceToken:
    """Creates and manages signed Governance Tokens (JWTs).

    Usage::

        token_mgr = GovernanceToken(secret=os.urandom(32), evaluator_id="eval-01")
        jwt_str = token_mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc...",
            method="approve",
            action_target="order/123",
            flow="full",
            scope="single",
        )
    """

    def __init__(
        self,
        secret: bytes,
        evaluator_id: str,
        *,
        lifetime_overrides: dict[str, int] | None = None,
    ) -> None:
        self._secret = secret
        self._evaluator_id = evaluator_id
        self._lifetimes = dict(_DEFAULT_LIFETIMES)
        if lifetime_overrides:
            self._lifetimes.update(lifetime_overrides)

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    def issue(
        self,
        agent_id: str,
        artifact_hash: str,
        method: str,
        action_target: str,
        flow: str,
        scope: str,
        *,
        verdict: str = "PROCEED",
        audience: str | list[str] = "",
        conditions: list[Condition] | None = None,
        authority_envelope: str = "",
        config_version: str = "",
        ucs: float | None = None,
        trust: float | None = None,
        scope_constraints: dict[str, Any] | None = None,
        lifetime: int | None = None,
    ) -> str:
        """Issue a signed Governance Token.

        Returns the JWT string.
        """
        now = int(time.time())
        ttl = lifetime if lifetime is not None else self._lifetimes.get(scope, 60)

        claims = TokenClaims(
            iss=self._evaluator_id,
            sub=agent_id,
            exp=now + ttl,
            iat=now,
            jti=f"ngt-{uuid.uuid4().hex[:16]}",
            nomo_verdict=verdict,
            nomo_artifact_hash=artifact_hash,
            nomo_method=method,
            nomo_action_target=action_target,
            nomo_flow=flow,
            nomo_scope=scope,
            aud=audience,
            nomo_conditions=[c.to_dict() for c in (conditions or [])],
            nomo_authority_envelope=authority_envelope,
            nomo_config_version=config_version,
            nomo_ucs=ucs,
            nomo_trust=trust,
            nomo_scope_constraints=scope_constraints or {},
        )

        header = {"alg": "HS256", "typ": "JWT"}
        return _jwt_sign(header, claims.to_dict(), self._secret)

    def decode(self, token: str) -> TokenClaims | None:
        """Decode and verify a token, returning claims or None."""
        payload = _jwt_verify(token, self._secret)
        if payload is None:
            return None
        return TokenClaims.from_dict(payload)


# ── Token Validator ────────────────────────────────────────────────────


class TokenValidator:
    """Validates Governance Tokens.

    Handles signature verification, expiration checks, and revocation.
    Execution environments use this to gate actions.
    """

    def __init__(self, secret: bytes) -> None:
        self._secret = secret
        self._revoked: set[str] = set()
        self._used: set[str] = set()  # For single-scope replay prevention

    def validate(
        self,
        token: str,
        *,
        expected_method: str | None = None,
        expected_target: str | None = None,
    ) -> TokenValidationResult:
        """Validate a Governance Token.

        Checks signature, expiration, revocation, and optionally binding.
        """
        payload = _jwt_verify(token, self._secret)
        if payload is None:
            return TokenValidationResult(valid=False, error="invalid_signature")

        claims = TokenClaims.from_dict(payload)

        # Check expiration
        now = int(time.time())
        if claims.exp <= now:
            return TokenValidationResult(valid=False, error="token_expired")

        # Check revocation
        if claims.jti in self._revoked:
            return TokenValidationResult(valid=False, error="token_revoked")

        # Check single-scope replay
        if claims.nomo_scope == "single":
            if claims.jti in self._used:
                return TokenValidationResult(valid=False, error="token_already_used")
            self._used.add(claims.jti)

        # Check method binding
        if expected_method is not None:
            if claims.nomo_scope == "class":
                # Class tokens may have allowed_methods in scope_constraints
                allowed = claims.nomo_scope_constraints.get("allowed_methods", [])
                if allowed and expected_method not in allowed:
                    return TokenValidationResult(valid=False, error="method_not_authorized")
                elif not allowed and expected_method != claims.nomo_method:
                    return TokenValidationResult(valid=False, error="method_mismatch")
            elif expected_method != claims.nomo_method:
                return TokenValidationResult(valid=False, error="method_mismatch")

        # Check target binding
        if expected_target is not None:
            if claims.nomo_scope == "class":
                allowed = claims.nomo_scope_constraints.get("allowed_targets", [])
                if allowed and expected_target not in allowed:
                    return TokenValidationResult(valid=False, error="target_not_authorized")
                elif not allowed and expected_target != claims.nomo_action_target:
                    return TokenValidationResult(valid=False, error="target_mismatch")
            elif expected_target != claims.nomo_action_target:
                return TokenValidationResult(valid=False, error="target_mismatch")

        from datetime import datetime, timezone
        expires_at = datetime.fromtimestamp(claims.exp, tz=timezone.utc).isoformat()

        return TokenValidationResult(
            valid=True,
            verdict=claims.nomo_verdict,
            conditions=claims.nomo_conditions,
            expires_at=expires_at,
            claims=claims,
        )

    def revoke(self, token_id: str) -> bool:
        """Revoke a token by its JTI. Returns True if newly revoked."""
        if token_id in self._revoked:
            return False
        self._revoked.add(token_id)
        return True

    def is_revoked(self, token_id: str) -> bool:
        return token_id in self._revoked

    def introspect(self, token: str) -> dict[str, Any] | None:
        """Full introspection of a token — returns all claims or None."""
        payload = _jwt_verify(token, self._secret)
        if payload is None:
            return None
        return payload
