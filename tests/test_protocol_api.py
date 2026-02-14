"""Tests for Nomotic Protocol API endpoints."""

import json
import os
import threading
import time
import urllib.request
import urllib.error
import pytest

from nomotic.api import NomoticAPIServer
from nomotic.authority import CertificateAuthority
from nomotic.evaluator import EvaluatorConfig, ProtocolEvaluator
from nomotic.keys import SigningKey
from nomotic.protocol import (
    Alternative,
    AuthorityClaim,
    Constraint,
    Factor,
    IntendedAction,
    Justification,
    ReasoningArtifact,
)
from nomotic.runtime import GovernanceRuntime
from nomotic.store import MemoryCertificateStore


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def server_with_evaluator():
    """Start a test API server with a protocol evaluator."""
    sk, _vk = SigningKey.generate()
    ca = CertificateAuthority(
        issuer_id="test-ca",
        signing_key=sk,
        store=MemoryCertificateStore(),
    )
    runtime = GovernanceRuntime()
    # Configure scope for test agents
    scope_dim = runtime.registry.get("scope_compliance")
    scope_dim.configure_agent_scope("test-agent-1", {"approve", "read", "write", "query"})
    iso_dim = runtime.registry.get("isolation_integrity")
    iso_dim.set_boundaries("test-agent-1", {"order/123", "data/records"})

    evaluator = ProtocolEvaluator(
        config=EvaluatorConfig(
            evaluator_id="test-api-evaluator",
            token_secret=os.urandom(32),
        ),
        runtime=runtime,
    )

    server = NomoticAPIServer(
        ca,
        runtime=runtime,
        evaluator=evaluator,
        host="127.0.0.1",
        port=0,  # Let OS pick a port
    )
    # Build the server to get the actual port
    httpd = server._build_server()
    # Use port 0 to get a random available port
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # Let server start

    yield base, evaluator

    httpd.shutdown()


def _post_json(base: str, path: str, data: dict) -> tuple[int, dict]:
    """POST JSON to the server and return (status, body)."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{base}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get_json(base: str, path: str) -> tuple[int, dict]:
    """GET JSON from the server and return (status, body)."""
    req = urllib.request.Request(f"{base}{path}", method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _make_artifact_dict():
    """Create a minimal valid reasoning artifact as a dict."""
    art = ReasoningArtifact(
        agent_id="test-agent-1",
        goal="Process test request",
        origin="user_request",
        constraints_identified=[
            Constraint(type="policy", description="Standard limit", source="policy://test"),
        ],
        factors=[
            Factor(id="f1", type="constraint", description="Limit check",
                   source="policy://test", assessment="OK", influence="decisive", confidence=0.95),
            Factor(id="f2", type="context", description="Context",
                   source="data://test", assessment="OK", influence="significant", confidence=0.9),
        ],
        alternatives_considered=[
            Alternative(method="deny", reason_rejected="No grounds"),
        ],
        intended_action=IntendedAction(method="approve", target="order/123"),
        justifications=[
            Justification(factor_id="f1", explanation="Constraint satisfied"),
        ],
        authority_claim=AuthorityClaim(envelope_type="standard"),
        unknowns=[],
        assumptions=[],
        overall_confidence=0.88,
    )
    return art.to_dict()


# ── POST /v1/reason ────────────────────────────────────────────────────


class TestReasonEndpoint:
    def test_full_deliberation(self, server_with_evaluator):
        base, _ = server_with_evaluator
        data = _make_artifact_dict()
        status, body = _post_json(base, "/v1/reason", data)
        assert status == 200
        assert body["verdict"] in ["PROCEED", "PROCEED_WITH_CONDITIONS"]
        assert "assessment" in body
        assert "metadata" in body
        if body["verdict"] in ["PROCEED", "PROCEED_WITH_CONDITIONS"]:
            assert "token" in body

    def test_invalid_artifact(self, server_with_evaluator):
        base, _ = server_with_evaluator
        status, body = _post_json(base, "/v1/reason", {"not": "valid"})
        assert status == 400

    def test_revise_verdict(self, server_with_evaluator):
        base, _ = server_with_evaluator
        data = _make_artifact_dict()
        data["identity"]["agent_id"] = ""  # Invalid
        data["task"]["goal"] = ""  # Invalid
        status, body = _post_json(base, "/v1/reason", data)
        assert status == 200
        assert body["verdict"] == "REVISE"
        assert "guidance" in body


# ── POST /v1/reason/summary ────────────────────────────────────────────


class TestReasonSummaryEndpoint:
    def test_summary_flow(self, server_with_evaluator):
        base, _ = server_with_evaluator
        data = _make_artifact_dict()
        status, body = _post_json(base, "/v1/reason/summary", data)
        assert status == 200
        assert body["verdict"] in ["PROCEED", "PROCEED_WITH_CONDITIONS", "REVISE", "ESCALATE"]


# ── POST /v1/reason/posthoc ───────────────────────────────────────────


class TestReasonPosthocEndpoint:
    def test_posthoc_flow(self, server_with_evaluator):
        base, _ = server_with_evaluator
        data = {
            "artifact": _make_artifact_dict(),
            "action_result": {"status": "success"},
        }
        status, body = _post_json(base, "/v1/reason/posthoc", data)
        assert status == 200
        assert "sound_reasoning" in body
        assert "would_have_approved" in body
        assert "trust_adjustment" in body

    def test_posthoc_bare_artifact(self, server_with_evaluator):
        """Post-hoc should also work with just the artifact at top level."""
        base, _ = server_with_evaluator
        data = _make_artifact_dict()
        status, body = _post_json(base, "/v1/reason/posthoc", data)
        assert status == 200
        assert "sound_reasoning" in body


# ── POST /v1/token/validate ────────────────────────────────────────────


class TestTokenValidateEndpoint:
    def test_validate_valid_token(self, server_with_evaluator):
        base, evaluator = server_with_evaluator
        # First get a token
        data = _make_artifact_dict()
        _, reason_body = _post_json(base, "/v1/reason", data)
        token = reason_body.get("token", "")
        if not token:
            pytest.skip("No token issued")

        # Validate it
        status, body = _post_json(base, "/v1/token/validate", {"token": token})
        assert status == 200
        assert body["valid"] is True

    def test_validate_missing_token(self, server_with_evaluator):
        base, _ = server_with_evaluator
        status, body = _post_json(base, "/v1/token/validate", {})
        assert status == 400

    def test_validate_bad_token(self, server_with_evaluator):
        base, _ = server_with_evaluator
        status, body = _post_json(base, "/v1/token/validate", {"token": "bad.token.here"})
        assert status == 200
        assert body["valid"] is False


# ── POST /v1/token/introspect ──────────────────────────────────────────


class TestTokenIntrospectEndpoint:
    def test_introspect_valid_token(self, server_with_evaluator):
        base, _ = server_with_evaluator
        # Get a token
        data = _make_artifact_dict()
        _, reason_body = _post_json(base, "/v1/reason", data)
        token = reason_body.get("token", "")
        if not token:
            pytest.skip("No token issued")

        status, body = _post_json(base, "/v1/token/introspect", {"token": token})
        assert status == 200
        assert "claims" in body

    def test_introspect_bad_token(self, server_with_evaluator):
        base, _ = server_with_evaluator
        status, body = _post_json(base, "/v1/token/introspect", {"token": "invalid"})
        assert status == 400


# ── POST /v1/token/revoke ──────────────────────────────────────────────


class TestTokenRevokeEndpoint:
    def test_revoke_token(self, server_with_evaluator):
        base, evaluator = server_with_evaluator
        # Get a token
        data = _make_artifact_dict()
        _, reason_body = _post_json(base, "/v1/reason", data)
        token = reason_body.get("token", "")
        if not token:
            pytest.skip("No token issued")

        # Decode to get JTI
        claims = evaluator.token_manager.decode(token)
        assert claims is not None

        # Revoke
        status, body = _post_json(base, "/v1/token/revoke", {"token_id": claims.jti})
        assert status == 200
        assert body["revoked"] is True

        # Validate — should fail
        status, body = _post_json(base, "/v1/token/validate", {"token": token})
        assert status == 200
        assert body["valid"] is False

    def test_revoke_missing_token_id(self, server_with_evaluator):
        base, _ = server_with_evaluator
        status, body = _post_json(base, "/v1/token/revoke", {})
        assert status == 400


# ── GET /v1/schema ─────────────────────────────────────────────────────


class TestSchemaEndpoints:
    def test_get_schema(self, server_with_evaluator):
        base, _ = server_with_evaluator
        status, body = _get_json(base, "/v1/schema")
        assert status == 200

    def test_get_schema_version(self, server_with_evaluator):
        base, _ = server_with_evaluator
        status, body = _get_json(base, "/v1/schema/version")
        assert status == 200
        assert "supported_versions" in body
        assert "0.1.0" in body["supported_versions"]
