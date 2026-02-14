"""Tests for the ethical reasoning evaluator additions (Phase 8).

Covers EthicalReasoningAssessment scoring, integration with ProtocolEvaluator,
and edge cases around empty/maximal artifacts.
"""

import pytest

from nomotic.evaluator import (
    EthicalReasoningAssessment,
    EthicalReasoningConfig,
    ProtocolEvaluator,
)
from nomotic.protocol import (
    Alternative,
    Assumption,
    AuthorityClaim,
    Constraint,
    Factor,
    IntendedAction,
    Justification,
    ReasoningArtifact,
    Unknown,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _base_artifact(**overrides) -> ReasoningArtifact:
    """Build a minimal valid ReasoningArtifact with sensible defaults."""
    defaults = dict(
        agent_id="test-agent",
        goal="Process request",
        origin="user_request",
        constraints_identified=[
            Constraint(type="policy", description="Standard policy", source="policy://base"),
        ],
        factors=[
            Factor(id="f1", type="constraint", description="Policy constraint met",
                   source="policy://base", assessment="Satisfied", influence="decisive", confidence=0.9),
        ],
        alternatives_considered=[],
        intended_action=IntendedAction(method="approve", target="item/1"),
        justifications=[Justification(factor_id="f1", explanation="Constraint met")],
        authority_claim=AuthorityClaim(envelope_type="standard"),
        unknowns=[],
        assumptions=[],
        overall_confidence=0.85,
    )
    defaults.update(overrides)
    return ReasoningArtifact(**defaults)


def _artifact_with_stakeholders() -> ReasoningArtifact:
    """Artifact containing stakeholder-related factors and constraints."""
    return _base_artifact(
        constraints_identified=[
            Constraint(type="policy", description="Standard policy", source="policy://base"),
            Constraint(type="ethical", description="Ethical treatment of users", source="ethics://org"),
        ],
        factors=[
            Factor(id="f1", type="constraint", description="Policy constraint met",
                   source="policy://base", assessment="Satisfied", influence="decisive", confidence=0.9),
            Factor(id="f2", type="risk", description="Stakeholder impact on end users",
                   source="analysis://impact", assessment="Low risk", influence="significant", confidence=0.8),
            Factor(id="f3", type="context", description="Affected user population is small",
                   source="data://users", assessment="Manageable", influence="moderate", confidence=0.85),
        ],
        justifications=[
            Justification(factor_id="f1", explanation="Constraint met"),
            Justification(factor_id="f2", explanation="Risk acceptable"),
        ],
    )


def _artifact_with_harm() -> ReasoningArtifact:
    """Artifact with harm-awareness signals."""
    return _base_artifact(
        factors=[
            Factor(id="f1", type="constraint", description="Policy constraint met",
                   source="policy://base", assessment="Satisfied", influence="decisive", confidence=0.9),
            Factor(id="f2", type="risk", description="Potential data loss risk",
                   source="analysis://risk", assessment="Mitigated by backup", influence="significant", confidence=0.75),
        ],
        justifications=[
            Justification(factor_id="f1", explanation="Constraint met"),
            Justification(factor_id="f2", explanation="Risk mitigated"),
        ],
        unknowns=[
            Unknown(description="Risk of harm to downstream services", impact="Could cause outage"),
        ],
        alternatives_considered=[
            Alternative(method="deny", reason_rejected="Risk is manageable and safer than inaction"),
        ],
    )


def _artifact_with_fairness() -> ReasoningArtifact:
    """Artifact with fairness and equity considerations."""
    return _base_artifact(
        intended_action=IntendedAction(method="classify", target="application/99"),
        constraints_identified=[
            Constraint(type="policy", description="Non-discrimination policy", source="policy://fair"),
        ],
        factors=[
            Factor(id="f1", type="constraint", description="Fairness and equitable treatment required",
                   source="policy://fair", assessment="Constraint met", influence="decisive", confidence=0.9),
            Factor(id="f2", type="context", description="Differential impact across demographic groups",
                   source="analysis://equity", assessment="No disparate impact found", influence="significant", confidence=0.7),
        ],
        justifications=[
            Justification(factor_id="f1", explanation="Fairness constraint satisfied"),
            Justification(factor_id="f2", explanation="No differential impact"),
        ],
    )


def _artifact_with_diverse_alternatives() -> ReasoningArtifact:
    """Artifact with multiple diverse alternatives considered."""
    return _base_artifact(
        alternatives_considered=[
            Alternative(method="deny", reason_rejected="No policy basis for denial"),
            Alternative(method="escalate", reason_rejected="Population equity review passed"),
            Alternative(method="recommend", reason_rejected="Direct action more appropriate"),
        ],
    )


def _artifact_with_ethical_uncertainty() -> ReasoningArtifact:
    """Artifact acknowledging ethical uncertainty."""
    return _base_artifact(
        unknowns=[
            Unknown(description="Impact on affected stakeholder groups unknown", impact="May require follow-up"),
            Unknown(description="Potential bias in training data", impact="Could affect fairness"),
        ],
        overall_confidence=0.65,
    )


def _artifact_all_ethical() -> ReasoningArtifact:
    """Artifact with every ethical signal present for maximum scores."""
    return _base_artifact(
        intended_action=IntendedAction(method="approve", target="item/1"),
        constraints_identified=[
            Constraint(type="policy", description="Standard policy", source="policy://base"),
            Constraint(type="ethical", description="Ethical review required", source="ethics://org"),
        ],
        factors=[
            Factor(id="f1", type="constraint", description="Fairness and equitable treatment",
                   source="policy://base", assessment="Met", influence="decisive", confidence=0.9),
            Factor(id="f2", type="risk", description="Stakeholder harm from denial",
                   source="analysis://impact", assessment="Low", influence="significant", confidence=0.8),
            Factor(id="f3", type="context", description="Affected user population is diverse",
                   source="data://users", assessment="Reviewed", influence="moderate", confidence=0.7),
            Factor(id="f4", type="context", description="Differential impact on different groups",
                   source="analysis://equity", assessment="Minimal", influence="moderate", confidence=0.7),
        ],
        justifications=[
            Justification(factor_id="f1", explanation="Fairness met"),
            Justification(factor_id="f2", explanation="Risk acceptable"),
        ],
        alternatives_considered=[
            Alternative(method="deny", reason_rejected="Harmful to equity of affected population"),
            Alternative(method="escalate", reason_rejected="Demographic group review passed"),
        ],
        unknowns=[
            Unknown(description="Long-term impact on affected stakeholders unknown", impact="Monitor"),
            Unknown(description="Potential bias in downstream systems", impact="Review quarterly"),
        ],
        assumptions=[
            Assumption(description="Current data is representative", basis="Recent audit", risk_if_wrong="Biased outcome"),
        ],
        overall_confidence=0.7,
    )


def _default_config() -> EthicalReasoningConfig:
    return EthicalReasoningConfig()


def _evaluator(config: EthicalReasoningConfig | None = None) -> ProtocolEvaluator:
    return ProtocolEvaluator(ethical_reasoning_config=config or _default_config())


# ── EthicalReasoningAssessment tests (10) ──────────────────────────────


class TestEthicalReasoningAssessmentScoring:

    def test_stakeholder_factors_high_score(self):
        evaluator = _evaluator()
        result = evaluator._assess_ethical_reasoning(_artifact_with_stakeholders(), _default_config())
        assert result.stakeholder_consideration >= 0.5

    def test_no_stakeholder_factors_low_score(self):
        evaluator = _evaluator()
        artifact = _base_artifact()  # no stakeholder/impact factors
        result = evaluator._assess_ethical_reasoning(artifact, _default_config())
        assert result.stakeholder_consideration <= 0.5

    def test_harm_acknowledgment_high_score(self):
        evaluator = _evaluator()
        result = evaluator._assess_ethical_reasoning(_artifact_with_harm(), _default_config())
        assert result.harm_awareness >= 0.4

    def test_no_harm_for_decision_method_flagged(self):
        evaluator = _evaluator()
        artifact = _base_artifact(intended_action=IntendedAction(method="deny", target="item/1"))
        result = evaluator._assess_ethical_reasoning(artifact, _default_config())
        assert any("harm" in f.lower() for f in result.findings)

    def test_fairness_factors_high_score(self):
        evaluator = _evaluator()
        result = evaluator._assess_ethical_reasoning(_artifact_with_fairness(), _default_config())
        assert result.fairness_consideration >= 0.5

    def test_diverse_alternatives_high_equity(self):
        evaluator = _evaluator()
        result = evaluator._assess_ethical_reasoning(_artifact_with_diverse_alternatives(), _default_config())
        assert result.alternative_equity >= 0.3

    def test_ethical_uncertainty_acknowledged(self):
        evaluator = _evaluator()
        result = evaluator._assess_ethical_reasoning(_artifact_with_ethical_uncertainty(), _default_config())
        assert result.uncertainty_honesty >= 0.4

    def test_overall_score_weighted_correctly(self):
        evaluator = _evaluator()
        cfg = _default_config()
        result = evaluator._assess_ethical_reasoning(_artifact_with_stakeholders(), cfg)
        total_w = cfg.stakeholder_weight + cfg.harm_weight + cfg.fairness_weight + cfg.alternative_weight + cfg.uncertainty_weight
        expected = (
            result.stakeholder_consideration * cfg.stakeholder_weight
            + result.harm_awareness * cfg.harm_weight
            + result.fairness_consideration * cfg.fairness_weight
            + result.alternative_equity * cfg.alternative_weight
            + result.uncertainty_honesty * cfg.uncertainty_weight
        ) / total_w
        assert abs(result.overall_ethical_reasoning_score - expected) < 1e-6

    def test_custom_method_lists_respected(self):
        cfg = EthicalReasoningConfig(
            require_harm_acknowledgment_for_methods=["read"],
            fairness_relevant_methods=["read"],
        )
        evaluator = _evaluator(cfg)
        artifact = _base_artifact(intended_action=IntendedAction(method="read", target="doc/1"))
        result = evaluator._assess_ethical_reasoning(artifact, cfg)
        assert any("harm" in f.lower() or "fairness" in f.lower() for f in result.findings)

    def test_minimum_score_threshold_produces_findings(self):
        cfg = EthicalReasoningConfig(minimum_ethical_reasoning_score=0.95)
        evaluator = _evaluator(cfg)
        artifact = _base_artifact()
        result = evaluator._assess_ethical_reasoning(artifact, cfg)
        assert any("below" in f.lower() and "threshold" in f.lower() for f in result.findings)


# ── Integration with ProtocolEvaluator (10) ────────────────────────────


class TestEthicalReasoningIntegration:

    def test_ethical_assessment_in_response_metadata(self):
        evaluator = _evaluator()
        response = evaluator.evaluate(_artifact_with_stakeholders())
        assert "ethical_reasoning" in response.assessment.metadata

    def test_low_ethical_score_recorded(self):
        evaluator = _evaluator()
        artifact = _base_artifact()  # minimal ethical signals
        response = evaluator.evaluate(artifact)
        ethical = response.assessment.metadata["ethical_reasoning"]
        assert ethical["overall_ethical_reasoning_score"] < 0.7

    def test_high_ethical_score_does_not_harm_verdict(self):
        evaluator = _evaluator()
        response = evaluator.evaluate(_artifact_all_ethical())
        assert response.verdict.value in ("PROCEED", "PROCEED_WITH_CONDITIONS")

    def test_config_customizable_per_organization(self):
        cfg_a = EthicalReasoningConfig(stakeholder_weight=1.0, harm_weight=0.0,
                                        fairness_weight=0.0, alternative_weight=0.0,
                                        uncertainty_weight=0.0)
        cfg_b = EthicalReasoningConfig(stakeholder_weight=0.0, harm_weight=1.0,
                                        fairness_weight=0.0, alternative_weight=0.0,
                                        uncertainty_weight=0.0)
        artifact = _artifact_with_stakeholders()
        score_a = ProtocolEvaluator(ethical_reasoning_config=cfg_a)._assess_ethical_reasoning(artifact, cfg_a)
        score_b = ProtocolEvaluator(ethical_reasoning_config=cfg_b)._assess_ethical_reasoning(artifact, cfg_b)
        # Stakeholder-heavy config should give different overall than harm-heavy
        assert score_a.overall_ethical_reasoning_score != score_b.overall_ethical_reasoning_score

    def test_default_config_reasonable(self):
        cfg = _default_config()
        evaluator = _evaluator(cfg)
        result = evaluator._assess_ethical_reasoning(_artifact_with_stakeholders(), cfg)
        assert 0.0 <= result.overall_ethical_reasoning_score <= 1.0

    def test_read_method_less_demanding(self):
        cfg = _default_config()
        evaluator = _evaluator(cfg)
        read_artifact = _base_artifact(intended_action=IntendedAction(method="read", target="doc/1"))
        approve_artifact = _base_artifact(intended_action=IntendedAction(method="approve", target="item/1"))
        result_read = evaluator._assess_ethical_reasoning(read_artifact, cfg)
        result_approve = evaluator._assess_ethical_reasoning(approve_artifact, cfg)
        # Decision methods like "approve" should have more findings than "read"
        assert len(result_approve.findings) >= len(result_read.findings)

    def test_assessment_findings_actionable(self):
        evaluator = _evaluator()
        result = evaluator._assess_ethical_reasoning(_base_artifact(), _default_config())
        for finding in result.findings:
            assert isinstance(finding, str)
            assert len(finding) > 0

    def test_assessment_recommendations_specific(self):
        evaluator = _evaluator()
        result = evaluator._assess_ethical_reasoning(_base_artifact(), _default_config())
        for rec in result.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_no_ethical_config_means_no_metadata(self):
        evaluator = ProtocolEvaluator()  # no ethical_reasoning_config
        response = evaluator.evaluate(_base_artifact())
        assert "ethical_reasoning" not in response.assessment.metadata

    def test_ethical_metadata_has_all_keys(self):
        evaluator = _evaluator()
        response = evaluator.evaluate(_artifact_with_stakeholders())
        ethical = response.assessment.metadata["ethical_reasoning"]
        expected_keys = {
            "stakeholder_consideration", "harm_awareness", "fairness_consideration",
            "alternative_equity", "uncertainty_honesty", "overall_ethical_reasoning_score",
            "findings", "recommendations",
        }
        assert expected_keys == set(ethical.keys())


# ── Edge cases (5) ─────────────────────────────────────────────────────


class TestEthicalReasoningEdgeCases:

    def test_empty_artifact_lowest_scores(self):
        artifact = _base_artifact(
            factors=[
                Factor(id="f1", type="constraint", description="Minimal",
                       source="x", assessment="x", influence="minor", confidence=0.5),
            ],
            constraints_identified=[
                Constraint(type="policy", description="x", source="x"),
            ],
            alternatives_considered=[],
            unknowns=[],
            assumptions=[],
        )
        evaluator = _evaluator()
        result = evaluator._assess_ethical_reasoning(artifact, _default_config())
        assert result.overall_ethical_reasoning_score < 0.4

    def test_no_factors_handled_gracefully(self):
        # Build an artifact with the absolute minimum single constraint factor
        artifact = _base_artifact(
            factors=[
                Factor(id="f1", type="constraint", description="Bare minimum",
                       source="x", assessment="x", influence="minor", confidence=0.5),
            ],
        )
        evaluator = _evaluator()
        # Should not raise
        result = evaluator._assess_ethical_reasoning(artifact, _default_config())
        assert isinstance(result, EthicalReasoningAssessment)

    def test_all_ethical_factors_maximum_scores(self):
        evaluator = _evaluator()
        result = evaluator._assess_ethical_reasoning(_artifact_all_ethical(), _default_config())
        assert result.stakeholder_consideration >= 0.5
        assert result.harm_awareness >= 0.3
        assert result.fairness_consideration >= 0.4
        assert result.alternative_equity >= 0.5
        assert result.uncertainty_honesty >= 0.4
        assert result.overall_ethical_reasoning_score >= 0.4

    def test_all_weights_zero_neutral_score(self):
        cfg = EthicalReasoningConfig(
            stakeholder_weight=0.0,
            harm_weight=0.0,
            fairness_weight=0.0,
            alternative_weight=0.0,
            uncertainty_weight=0.0,
        )
        evaluator = _evaluator(cfg)
        result = evaluator._assess_ethical_reasoning(_base_artifact(), cfg)
        assert result.overall_ethical_reasoning_score == 0.5

    def test_to_dict_round_trip(self):
        evaluator = _evaluator()
        original = evaluator._assess_ethical_reasoning(_artifact_all_ethical(), _default_config())
        d = original.to_dict()
        restored = EthicalReasoningAssessment(
            stakeholder_consideration=d["stakeholder_consideration"],
            harm_awareness=d["harm_awareness"],
            fairness_consideration=d["fairness_consideration"],
            alternative_equity=d["alternative_equity"],
            uncertainty_honesty=d["uncertainty_honesty"],
            overall_ethical_reasoning_score=d["overall_ethical_reasoning_score"],
            findings=d["findings"],
            recommendations=d["recommendations"],
        )
        assert restored.to_dict() == d
