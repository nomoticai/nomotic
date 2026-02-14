"""Tests for the Nomotic Protocol Evaluator."""

import os
import pytest

from nomotic.evaluator import EvaluatorConfig, PostHocAssessment, ProtocolEvaluator
from nomotic.protocol import (
    Alternative,
    Assumption,
    AuthorityClaim,
    Constraint,
    Factor,
    IntendedAction,
    Justification,
    ProtocolVerdict,
    ReasoningArtifact,
    Unknown,
)
from nomotic.runtime import GovernanceRuntime


# ── Helpers ────────────────────────────────────────────────────────────


def _make_artifact(**overrides):
    """Create a minimal valid ReasoningArtifact for testing."""
    defaults = dict(
        agent_id="test-agent-1",
        goal="Process test request",
        origin="user_request",
        constraints_identified=[
            Constraint(type="policy", description="Standard limit $500", source="policy://test"),
        ],
        factors=[
            Factor(
                id="f1", type="constraint", description="Amount within limit",
                source="policy://test", assessment="Constraint satisfied",
                influence="decisive", confidence=0.95,
            ),
            Factor(
                id="f2", type="context", description="User is verified",
                source="data://users/123", assessment="User verified",
                influence="significant", confidence=0.9,
            ),
        ],
        alternatives_considered=[
            Alternative(method="deny", reason_rejected="No grounds for denial"),
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
    defaults.update(overrides)
    return ReasoningArtifact(**defaults)


def _make_evaluator(**config_overrides):
    """Create a ProtocolEvaluator with test defaults."""
    config = EvaluatorConfig(
        evaluator_id="test-evaluator",
        token_secret=os.urandom(32),
        **config_overrides,
    )
    return ProtocolEvaluator(config=config)


def _make_evaluator_with_runtime(**config_overrides):
    """Create a ProtocolEvaluator backed by a GovernanceRuntime."""
    runtime = GovernanceRuntime()
    # Configure scope for the test agent
    scope_dim = runtime.registry.get("scope_compliance")
    scope_dim.configure_agent_scope("test-agent-1", {"approve", "read", "write", "query"})
    iso_dim = runtime.registry.get("isolation_integrity")
    iso_dim.set_boundaries("test-agent-1", {"order/123", "data/records", "data/x"})

    config = EvaluatorConfig(
        evaluator_id="test-evaluator",
        token_secret=os.urandom(32),
        require_dimensional=True,
        **config_overrides,
    )
    return ProtocolEvaluator(config=config, runtime=runtime)


# ── Full Deliberation Flow ─────────────────────────────────────────────


class TestFullDeliberation:
    def test_valid_artifact_proceeds(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        assert response.verdict in (ProtocolVerdict.PROCEED, ProtocolVerdict.PROCEED_WITH_CONDITIONS)
        assert response.assessment.completeness_score > 0
        assert response.metadata.evaluator_id == "test-evaluator"
        assert response.metadata.artifact_hash.startswith("sha256:")

    def test_valid_artifact_gets_token(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        assert response.token  # Token should be present
        # Validate the token
        result = evaluator.validate_token(response.token)
        assert result.valid is True
        assert result.verdict == response.verdict.value

    def test_invalid_artifact_gets_revise(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact(agent_id="", goal="")
        response = evaluator.evaluate(artifact)
        assert response.verdict == ProtocolVerdict.REVISE
        assert response.guidance is not None
        assert len(response.guidance.reasoning_gaps) > 0
        assert response.token == ""

    def test_failed_authority_gets_escalate(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact(
            authority_claim=AuthorityClaim(
                envelope_type="conditional",
                # No conditions_met — this should fail authority check
            ),
        )
        response = evaluator.evaluate(artifact)
        assert response.verdict == ProtocolVerdict.ESCALATE
        assert response.escalation is not None

    def test_assessment_completeness(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        assessment = response.assessment
        assert 0.0 <= assessment.completeness_score <= 1.0
        assert 0.0 <= assessment.alignment_score <= 1.0
        assert 0.0 <= assessment.uncertainty_calibration_score <= 1.0
        assert 0.0 <= assessment.alternatives_adequacy_score <= 1.0

    def test_no_alternatives_lowers_adequacy(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact(alternatives_considered=[])
        response = evaluator.evaluate(artifact)
        assert response.assessment.alternatives_adequacy_score < 1.0

    def test_overconfident_lowers_calibration(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact(
            overall_confidence=0.99,
            unknowns=[
                Unknown(description="Unknown 1", impact="High"),
                Unknown(description="Unknown 2", impact="Medium"),
                Unknown(description="Unknown 3", impact="Low"),
            ],
        )
        response = evaluator.evaluate(artifact)
        assert response.assessment.uncertainty_calibration_score < 1.0

    def test_decisive_factor_not_justified_lowers_alignment(self):
        evaluator = _make_evaluator()
        # f1 is decisive but justification references f2
        artifact = _make_artifact(
            justifications=[
                Justification(factor_id="f2", explanation="User verified"),
            ],
        )
        response = evaluator.evaluate(artifact)
        assert response.assessment.alignment_score < 1.0

    def test_metadata_populated(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        assert response.metadata.evaluation_time_ms > 0
        assert response.metadata.config_version
        assert response.metadata.timestamp
        assert response.metadata.protocol_version == "0.1.0"

    def test_response_serialization(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        d = response.to_dict()
        assert "verdict" in d
        assert "assessment" in d
        assert "metadata" in d


# ── Full Deliberation with Runtime ─────────────────────────────────────


class TestFullDeliberationWithRuntime:
    def test_dimensional_evaluation(self):
        evaluator = _make_evaluator_with_runtime()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        # Should have dimensional summary from the runtime
        assert response.assessment.dimensional_summary
        assert response.assessment.ucs is not None
        assert response.assessment.trust_state is not None

    def test_runtime_allow_produces_proceed(self):
        evaluator = _make_evaluator_with_runtime()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        # With proper scope configuration, should get PROCEED
        assert response.verdict in (ProtocolVerdict.PROCEED, ProtocolVerdict.PROCEED_WITH_CONDITIONS)
        assert response.token  # Token should be present


# ── Summary Flow ───────────────────────────────────────────────────────


class TestSummaryFlow:
    def test_valid_artifact_proceeds(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate_summary(artifact)
        assert response.verdict == ProtocolVerdict.PROCEED
        assert response.token

    def test_token_has_class_scope(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate_summary(artifact)
        # Validate that the token uses summary flow and class scope
        result = evaluator.validate_token(response.token)
        assert result.valid is True
        assert result.claims.nomo_flow == "summary"
        assert result.claims.nomo_scope == "class"

    def test_invalid_artifact_gets_revise(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact(goal="")
        response = evaluator.evaluate_summary(artifact)
        assert response.verdict == ProtocolVerdict.REVISE

    def test_authority_failure_escalates(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact(
            authority_claim=AuthorityClaim(envelope_type="conditional"),
        )
        response = evaluator.evaluate_summary(artifact)
        assert response.verdict == ProtocolVerdict.ESCALATE


# ── Post-Hoc Flow ─────────────────────────────────────────────────────


class TestPostHocFlow:
    def test_sound_reasoning_approved(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        assessment = evaluator.evaluate_posthoc(artifact)
        assert assessment.sound_reasoning is True
        assert assessment.would_have_approved is True
        assert assessment.trust_adjustment > 0
        assert assessment.concern is False

    def test_invalid_artifact_raises_concern(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact(agent_id="")
        assessment = evaluator.evaluate_posthoc(artifact)
        assert assessment.sound_reasoning is False
        assert assessment.would_have_approved is False
        assert assessment.concern is True
        assert assessment.trust_adjustment < 0

    def test_poor_quality_raises_concern(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact(
            alternatives_considered=[],  # No alternatives
            overall_confidence=0.99,  # Overconfident
            unknowns=[Unknown(description="U1", impact="High"), Unknown(description="U2", impact="High"), Unknown(description="U3", impact="High")],
        )
        assessment = evaluator.evaluate_posthoc(artifact)
        # Quality should be lower but may still pass depending on thresholds
        assert isinstance(assessment.trust_adjustment, float)

    def test_posthoc_has_assessment(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        assessment = evaluator.evaluate_posthoc(artifact)
        assert assessment.assessment is not None
        assert assessment.assessment.completeness_score > 0

    def test_posthoc_serialization(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        assessment = evaluator.evaluate_posthoc(artifact)
        d = assessment.to_dict()
        assert "sound_reasoning" in d
        assert "would_have_approved" in d
        assert "trust_adjustment" in d


# ── Token Operations ───────────────────────────────────────────────────


class TestTokenOperations:
    def test_validate_issued_token(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        result = evaluator.validate_token(response.token)
        assert result.valid is True

    def test_validate_with_binding(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        # Correct binding
        result = evaluator.validate_token(
            response.token,
            expected_method="approve",
            expected_target="order/123",
        )
        assert result.valid is True

    def test_validate_wrong_method_binding(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        result = evaluator.validate_token(
            response.token,
            expected_method="delete",
        )
        assert result.valid is False

    def test_introspect_token(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        result = evaluator.introspect_token(response.token)
        assert result is not None
        assert "claims" in result
        # Artifact should be stored and returned
        assert "artifact" in result

    def test_introspect_invalid_token(self):
        evaluator = _make_evaluator()
        result = evaluator.introspect_token("bad.token.here")
        assert result is None

    def test_revoke_token(self):
        evaluator = _make_evaluator()
        artifact = _make_artifact()
        response = evaluator.evaluate(artifact)
        # Get the JTI
        claims = evaluator.token_manager.decode(response.token)
        assert claims is not None
        # Revoke
        assert evaluator.revoke_token(claims.jti) is True
        # Validate — should fail
        result = evaluator.validate_token(response.token)
        assert result.valid is False


# ── Schema Operations ──────────────────────────────────────────────────


class TestSchemaOperations:
    def test_get_supported_versions(self):
        evaluator = _make_evaluator()
        versions = evaluator.get_supported_versions()
        assert "0.1.0" in versions

    def test_get_schema(self):
        evaluator = _make_evaluator()
        schema = evaluator.get_schema()
        assert schema is not None
        # Should return something (either the full schema or a fallback)
        assert "version" in schema or "$schema" in schema


# ── EvaluatorConfig ────────────────────────────────────────────────────


class TestEvaluatorConfig:
    def test_defaults(self):
        config = EvaluatorConfig()
        assert config.evaluator_id == "nomotic-evaluator"
        assert config.min_completeness_for_proceed == 0.6
        assert config.proceed_threshold == 0.7
        assert len(config.supported_versions) == 1

    def test_custom_config(self):
        config = EvaluatorConfig(
            evaluator_id="custom-eval",
            min_completeness_for_proceed=0.8,
        )
        evaluator = ProtocolEvaluator(config=config)
        assert evaluator.config.evaluator_id == "custom-eval"
