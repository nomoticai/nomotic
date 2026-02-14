"""Tests for Nomotic Governance Token (JWT) generation and validation."""

import os
import time
import pytest

from nomotic.protocol import Condition
from nomotic.token import (
    GovernanceToken,
    TokenClaims,
    TokenValidationResult,
    TokenValidator,
    _b64url_decode,
    _b64url_encode,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_token_mgr(secret=None, evaluator_id="test-eval"):
    secret = secret or os.urandom(32)
    return GovernanceToken(secret=secret, evaluator_id=evaluator_id), secret


def _issue_default(mgr):
    return mgr.issue(
        agent_id="agent-1",
        artifact_hash="sha256:abc123",
        method="approve",
        action_target="order/123",
        flow="full",
        scope="single",
    )


# ── Base64url ──────────────────────────────────────────────────────────


class TestBase64url:
    def test_roundtrip(self):
        data = b"hello world"
        encoded = _b64url_encode(data)
        decoded = _b64url_decode(encoded)
        assert decoded == data

    def test_no_padding(self):
        data = b"test"
        encoded = _b64url_encode(data)
        assert "=" not in encoded


# ── GovernanceToken ────────────────────────────────────────────────────


class TestGovernanceToken:
    def test_issue_basic(self):
        mgr, _ = _make_token_mgr()
        token = _issue_default(mgr)
        assert token
        assert token.count(".") == 2  # JWT has 3 parts

    def test_decode_roundtrip(self):
        mgr, _ = _make_token_mgr()
        token = _issue_default(mgr)
        claims = mgr.decode(token)
        assert claims is not None
        assert claims.sub == "agent-1"
        assert claims.nomo_method == "approve"
        assert claims.nomo_action_target == "order/123"
        assert claims.nomo_flow == "full"
        assert claims.nomo_scope == "single"
        assert claims.nomo_verdict == "PROCEED"

    def test_decode_with_wrong_secret(self):
        mgr, _ = _make_token_mgr()
        token = _issue_default(mgr)
        other_mgr, _ = _make_token_mgr()
        claims = other_mgr.decode(token)
        assert claims is None

    def test_issue_with_conditions(self):
        mgr, _ = _make_token_mgr()
        conditions = [
            Condition(type="monitoring", description="Enhanced monitoring"),
            Condition(type="time_limit", description="Must complete within 5 min"),
        ]
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="write",
            action_target="data/records",
            flow="full",
            scope="single",
            verdict="PROCEED_WITH_CONDITIONS",
            conditions=conditions,
        )
        claims = mgr.decode(token)
        assert claims is not None
        assert claims.nomo_verdict == "PROCEED_WITH_CONDITIONS"
        assert len(claims.nomo_conditions) == 2

    def test_issue_with_all_optional_claims(self):
        mgr, _ = _make_token_mgr()
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="transfer",
            action_target="account/456",
            flow="full",
            scope="single",
            audience="execution-env-1",
            authority_envelope="env-gold",
            config_version="cfg_abc",
            ucs=0.85,
            trust=0.78,
        )
        claims = mgr.decode(token)
        assert claims is not None
        assert claims.aud == "execution-env-1"
        assert claims.nomo_authority_envelope == "env-gold"
        assert claims.nomo_ucs == 0.85
        assert claims.nomo_trust == 0.78

    def test_evaluator_id_in_issuer(self):
        mgr, _ = _make_token_mgr(evaluator_id="my-evaluator")
        token = _issue_default(mgr)
        claims = mgr.decode(token)
        assert claims.iss == "my-evaluator"

    def test_token_has_jti(self):
        mgr, _ = _make_token_mgr()
        token = _issue_default(mgr)
        claims = mgr.decode(token)
        assert claims.jti.startswith("ngt-")

    def test_token_expiration(self):
        mgr, _ = _make_token_mgr()
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/x",
            flow="full",
            scope="single",
        )
        claims = mgr.decode(token)
        assert claims.exp > claims.iat
        assert claims.exp - claims.iat == 60  # single scope default

    def test_class_scope_lifetime(self):
        mgr, _ = _make_token_mgr()
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/x",
            flow="summary",
            scope="class",
        )
        claims = mgr.decode(token)
        assert claims.exp - claims.iat == 900  # 15 minutes

    def test_session_scope_lifetime(self):
        mgr, _ = _make_token_mgr()
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/x",
            flow="summary",
            scope="session",
        )
        claims = mgr.decode(token)
        assert claims.exp - claims.iat == 3600  # 1 hour

    def test_custom_lifetime(self):
        mgr, _ = _make_token_mgr()
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/x",
            flow="full",
            scope="single",
            lifetime=30,
        )
        claims = mgr.decode(token)
        assert claims.exp - claims.iat == 30

    def test_lifetime_overrides(self):
        secret = os.urandom(32)
        mgr = GovernanceToken(
            secret=secret,
            evaluator_id="test",
            lifetime_overrides={"single": 120},
        )
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/x",
            flow="full",
            scope="single",
        )
        claims = mgr.decode(token)
        assert claims.exp - claims.iat == 120

    def test_scope_constraints(self):
        mgr, _ = _make_token_mgr()
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/records",
            flow="summary",
            scope="class",
            scope_constraints={
                "allowed_methods": ["read", "query"],
                "allowed_targets": ["data/records", "data/metadata"],
                "max_actions": 100,
            },
        )
        claims = mgr.decode(token)
        assert claims.nomo_scope_constraints["allowed_methods"] == ["read", "query"]
        assert claims.nomo_scope_constraints["max_actions"] == 100


# ── TokenValidator ─────────────────────────────────────────────────────


class TestTokenValidator:
    def test_validate_valid_token(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        token = _issue_default(mgr)
        result = validator.validate(token)
        assert result.valid is True
        assert result.verdict == "PROCEED"
        assert result.claims is not None

    def test_validate_expired_token(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/x",
            flow="full",
            scope="single",
            lifetime=-1,  # Already expired
        )
        result = validator.validate(token)
        assert result.valid is False
        assert result.error == "token_expired"

    def test_validate_wrong_secret(self):
        mgr, _ = _make_token_mgr()
        validator = TokenValidator(secret=os.urandom(32))
        token = _issue_default(mgr)
        result = validator.validate(token)
        assert result.valid is False
        assert result.error == "invalid_signature"

    def test_validate_invalid_token_format(self):
        validator = TokenValidator(secret=os.urandom(32))
        result = validator.validate("not-a-jwt")
        assert result.valid is False
        assert result.error == "invalid_signature"

    def test_revoke_token(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        token = _issue_default(mgr)
        claims = mgr.decode(token)
        assert claims is not None

        # Revoke
        assert validator.revoke(claims.jti) is True
        assert validator.is_revoked(claims.jti) is True

        # Validate — should fail
        result = validator.validate(token)
        assert result.valid is False
        assert result.error == "token_revoked"

    def test_revoke_already_revoked(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        assert validator.revoke("some-jti") is True
        assert validator.revoke("some-jti") is False

    def test_single_scope_replay_prevention(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        token = _issue_default(mgr)

        # First use — succeeds
        result1 = validator.validate(token)
        assert result1.valid is True

        # Second use — fails (replay)
        result2 = validator.validate(token)
        assert result2.valid is False
        assert result2.error == "token_already_used"

    def test_class_scope_no_replay_prevention(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/x",
            flow="summary",
            scope="class",
        )

        # Multiple uses should work for class scope
        result1 = validator.validate(token)
        assert result1.valid is True
        result2 = validator.validate(token)
        assert result2.valid is True

    def test_method_binding_mismatch(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/x",
            flow="full",
            scope="single",
        )
        result = validator.validate(token, expected_method="delete")
        assert result.valid is False
        assert result.error == "method_mismatch"

    def test_target_binding_mismatch(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/x",
            flow="full",
            scope="single",
        )
        result = validator.validate(token, expected_target="data/y")
        assert result.valid is False
        assert result.error == "target_mismatch"

    def test_method_binding_match(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/x",
            flow="full",
            scope="single",
        )
        result = validator.validate(token, expected_method="read", expected_target="data/x")
        assert result.valid is True

    def test_class_scope_allowed_methods(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        token = mgr.issue(
            agent_id="agent-1",
            artifact_hash="sha256:abc",
            method="read",
            action_target="data/records",
            flow="summary",
            scope="class",
            scope_constraints={"allowed_methods": ["read", "query"]},
        )

        # Allowed method
        result = validator.validate(token, expected_method="query")
        assert result.valid is True

        # Disallowed method
        result = validator.validate(token, expected_method="delete")
        assert result.valid is False
        assert result.error == "method_not_authorized"

    def test_introspect(self):
        mgr, secret = _make_token_mgr()
        validator = TokenValidator(secret=secret)
        token = _issue_default(mgr)
        payload = validator.introspect(token)
        assert payload is not None
        assert payload["sub"] == "agent-1"
        assert payload["nomo_method"] == "approve"

    def test_introspect_invalid(self):
        validator = TokenValidator(secret=os.urandom(32))
        payload = validator.introspect("bad-token")
        assert payload is None


# ── TokenClaims ────────────────────────────────────────────────────────


class TestTokenClaims:
    def test_to_dict_minimal(self):
        claims = TokenClaims(
            iss="eval-1", sub="agent-1", exp=1000, iat=900,
            jti="ngt-abc", nomo_verdict="PROCEED",
            nomo_artifact_hash="sha256:abc", nomo_method="read",
            nomo_action_target="data/x", nomo_flow="full", nomo_scope="single",
        )
        d = claims.to_dict()
        assert d["iss"] == "eval-1"
        assert "aud" not in d  # empty string not included
        assert "nomo_ucs" not in d  # None not included

    def test_roundtrip(self):
        claims = TokenClaims(
            iss="eval-1", sub="agent-1", exp=1000, iat=900,
            jti="ngt-abc", nomo_verdict="PROCEED",
            nomo_artifact_hash="sha256:abc", nomo_method="approve",
            nomo_action_target="order/123", nomo_flow="full", nomo_scope="single",
            aud="env-1", nomo_ucs=0.85, nomo_trust=0.78,
        )
        d = claims.to_dict()
        claims2 = TokenClaims.from_dict(d)
        assert claims2.iss == claims.iss
        assert claims2.nomo_ucs == claims.nomo_ucs
