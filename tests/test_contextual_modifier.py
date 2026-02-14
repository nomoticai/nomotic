"""Tests for the Contextual Modifier system (Phase 7B).

Comprehensive tests covering:
- ContextModification dataclass (has_modifications, critical_signals, serialization)
- Weight adjustment rules (workflow, situational, relational, temporal, historical, input)
- Constraint rules (degraded mode, denial rates, low trust, etc.)
- Risk signal rules (unresolved deps, compound capability, adversarial input, etc.)
- Thin context assessment
- Trust modifier behavior
- Integration with GovernanceRuntime
- Modification reasoning
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from nomotic.context_profile import (
    CompletedStep,
    CompoundMethod,
    ContextProfile,
    ContextProfileManager,
    DelegationLink,
    Dependency,
    ExternalContext,
    ExternalSignal,
    FeedbackContext,
    FeedbackRecord,
    HistoricalContext,
    InputContext,
    MetaContext,
    OutcomeRecord,
    OutputContext,
    OutputRecord,
    OverrideRecord,
    PlannedStep,
    RecentVerdict,
    RelationalContext,
    SituationalContext,
    TemporalContext,
    TemporalEvent,
    WorkflowContext,
)
from nomotic.contextual_modifier import (
    ContextConstraint,
    ContextModification,
    ContextRiskSignal,
    ContextualModifier,
    ModifierConfig,
    WeightAdjustment,
)
from nomotic.runtime import GovernanceRuntime, RuntimeConfig
from nomotic.types import Action, AgentContext, GovernanceVerdict, TrustProfile, Verdict


# ── Helpers ──────────────────────────────────────────────────────────────


def _action(action_type: str = "read", target: str = "data", agent_id: str = "agent-1") -> Action:
    return Action(agent_id=agent_id, action_type=action_type, target=target)


def _context(agent_id: str = "agent-1", trust: float = 0.5, profile_id: str | None = None) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        trust_profile=TrustProfile(agent_id=agent_id, overall_trust=trust),
        context_profile_id=profile_id,
    )


def _empty_profile(agent_id: str = "agent-1") -> ContextProfile:
    return ContextProfile(
        profile_id="cp-test123",
        agent_id=agent_id,
        profile_type="workflow",
    )


def _modifier(config: ModifierConfig | None = None) -> ContextualModifier:
    return ContextualModifier(config=config)


# ── ContextModification Tests ────────────────────────────────────────────


class TestContextModification:
    """Tests for the ContextModification dataclass."""

    def test_has_modifications_false_when_empty(self):
        mod = ContextModification()
        assert mod.has_modifications() is False

    def test_has_modifications_true_with_weight_adjustments(self):
        mod = ContextModification(
            weight_adjustments=[
                WeightAdjustment("scope_compliance", 1.5, 1.8, "test"),
            ],
        )
        assert mod.has_modifications() is True

    def test_has_modifications_true_with_constraints(self):
        mod = ContextModification(
            constraints=[
                ContextConstraint("require_human_review", "test", "workflow", "required"),
            ],
        )
        assert mod.has_modifications() is True

    def test_has_modifications_true_with_trust_modifier(self):
        mod = ContextModification(trust_modifier=0.05)
        assert mod.has_modifications() is True

    def test_critical_signals_filters_correctly(self):
        mod = ContextModification(
            risk_signals=[
                ContextRiskSignal("low_signal", "desc", "src", "low", []),
                ContextRiskSignal("critical_signal", "desc", "src", "critical", ["scope_compliance"]),
                ContextRiskSignal("medium_signal", "desc", "src", "medium", []),
                ContextRiskSignal("critical_signal_2", "desc", "src", "critical", ["isolation_integrity"]),
            ],
        )
        critical = mod.critical_signals()
        assert len(critical) == 2
        assert all(s.severity == "critical" for s in critical)

    def test_modification_serialization_to_dict(self):
        mod = ContextModification(
            weight_adjustments=[
                WeightAdjustment("scope_compliance", 1.5, 1.8, "test reason"),
            ],
            constraints=[
                ContextConstraint("elevated_audit", "desc", "workflow", "recommended"),
            ],
            risk_signals=[
                ContextRiskSignal("test_signal", "desc", "src", "high", ["scope_compliance"]),
            ],
            trust_modifier=0.05,
            recommended_flow="full",
            context_completeness=0.6,
            modification_reasoning="Test reasoning",
        )
        d = mod.to_dict()
        assert len(d["weight_adjustments"]) == 1
        assert d["weight_adjustments"][0]["dimension_name"] == "scope_compliance"
        assert len(d["constraints"]) == 1
        assert len(d["risk_signals"]) == 1
        assert d["trust_modifier"] == 0.05
        assert d["recommended_flow"] == "full"
        assert d["has_modifications"] is True
        assert d["context_completeness"] == 0.6


# ── Weight Adjustment Rules Tests ────────────────────────────────────────


class TestWeightAdjustmentRules:
    """Tests for weight adjustment rules across context types."""

    def test_early_workflow_step_increases_cascading_impact(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.workflow = WorkflowContext(
            workflow_id="wf-1", workflow_type="process", total_steps=10, current_step=1,
        )
        result = modifier.modify(_action(), _context(), profile)
        cascading_adjustments = [
            a for a in result.weight_adjustments if a.dimension_name == "cascading_impact"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in cascading_adjustments)

    def test_final_workflow_step_decreases_cascading_impact(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.workflow = WorkflowContext(
            workflow_id="wf-1", workflow_type="process", total_steps=5, current_step=5,
        )
        result = modifier.modify(_action(), _context(), profile)
        cascading_adjustments = [
            a for a in result.weight_adjustments
            if a.dimension_name == "cascading_impact" and "Final" in a.reason
        ]
        assert any(a.adjusted_weight < a.original_weight for a in cascading_adjustments)

    def test_agent_initiated_origin_increases_human_override(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.situational = SituationalContext(
            origin="agent_initiated", trigger_description="self-initiated",
        )
        result = modifier.modify(_action(), _context(), profile)
        ho_adjustments = [
            a for a in result.weight_adjustments if a.dimension_name == "human_override"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in ho_adjustments)

    def test_escalation_received_adjusts_authority_and_human_override(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.situational = SituationalContext(
            origin="escalation_received", trigger_description="escalation from parent",
        )
        result = modifier.modify(_action(), _context(), profile)
        # human_override should decrease
        ho_adjustments = [
            a for a in result.weight_adjustments if a.dimension_name == "human_override"
        ]
        assert any(a.adjusted_weight < a.original_weight for a in ho_adjustments)
        # authority_verification should increase
        av_adjustments = [
            a for a in result.weight_adjustments if a.dimension_name == "authority_verification"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in av_adjustments)

    def test_falling_trust_increases_all_weights(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.historical = HistoricalContext(
            trust_current=0.4,
            trust_direction="falling",
            trust_trajectory_summary="declining",
        )
        result = modifier.modify(_action(), _context(), profile)
        # Should have adjustments for all 13 dimensions
        adjusted_dims = {a.dimension_name for a in result.weight_adjustments}
        # At least several dimensions should be adjusted
        assert len(adjusted_dims) > 5

    def test_high_drift_increases_behavioral_consistency(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.historical = HistoricalContext(
            trust_current=0.5,
            trust_direction="stable",
            trust_trajectory_summary="stable",
            behavioral_drift_status="high_drift",
        )
        result = modifier.modify(_action(), _context(), profile)
        bc_adjustments = [
            a for a in result.weight_adjustments if a.dimension_name == "behavioral_consistency"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in bc_adjustments)

    def test_compound_capability_increases_scope_and_isolation(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.relational = RelationalContext(
            compound_methods=[
                CompoundMethod(
                    method="transfer_funds",
                    agents_involved=["agent-1", "agent-2"],
                    combined_scope_description="combined fund transfer capability",
                ),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        scope_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "scope_compliance"
        ]
        iso_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "isolation_integrity"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in scope_adj)
        assert any(a.adjusted_weight > a.original_weight for a in iso_adj)

    def test_weight_adjustments_respect_max_weight_adjustment_cap(self):
        config = ModifierConfig(max_weight_adjustment=0.5)
        modifier = _modifier(config)
        profile = _empty_profile()
        # Set up scenario with many adjustments
        profile.situational = SituationalContext(
            origin="agent_initiated", trigger_description="test",
            urgency="critical",
        )
        result = modifier.modify(_action(), _context(), profile)
        # All adjustments should have delta <= 0.5
        for adj in result.weight_adjustments:
            delta = abs(adj.adjusted_weight - adj.original_weight)
            assert delta <= 0.5 + 0.001  # small epsilon for float

    def test_weight_adjustments_dont_go_below_floor(self):
        """No dimension weight should go below 0.1."""
        modifier = _modifier()
        profile = _empty_profile()
        # Temporal compliance has weight 0.8, -0.1 from incident response
        profile.situational = SituationalContext(
            origin="user_request", trigger_description="test",
            operational_mode="incident_response",
        )
        result = modifier.modify(_action(), _context(), profile)
        for adj in result.weight_adjustments:
            assert adj.adjusted_weight >= 0.1

    def test_weight_adjustments_dont_exceed_ceiling(self):
        """No dimension weight should exceed 3.0."""
        modifier = _modifier()
        profile = _empty_profile()
        profile.situational = SituationalContext(
            origin="agent_initiated", trigger_description="test",
            urgency="critical",
        )
        result = modifier.modify(_action(), _context(), profile)
        for adj in result.weight_adjustments:
            assert adj.adjusted_weight <= 3.0

    def test_no_workflow_context_no_workflow_adjustments(self):
        modifier = _modifier()
        profile = _empty_profile()
        # profile.workflow is None
        result = modifier.modify(_action(), _context(), profile)
        # No workflow-related adjustments
        workflow_reasons = [
            a for a in result.weight_adjustments
            if "workflow" in a.reason.lower() or "step" in a.reason.lower()
        ]
        assert len(workflow_reasons) == 0

    def test_no_situational_context_no_situational_adjustments(self):
        modifier = _modifier()
        profile = _empty_profile()
        result = modifier.modify(_action(), _context(), profile)
        agent_initiated_adj = [
            a for a in result.weight_adjustments
            if "agent-initiated" in a.reason.lower()
        ]
        assert len(agent_initiated_adj) == 0

    def test_disabled_modifier_categories_produce_no_adjustments(self):
        config = ModifierConfig(
            enable_workflow_modifiers=False,
            enable_situational_modifiers=False,
            enable_relational_modifiers=False,
            enable_temporal_modifiers=False,
            enable_historical_modifiers=False,
            enable_input_modifiers=False,
            enable_external_modifiers=False,
            enable_meta_modifiers=False,
            enable_feedback_modifiers=False,
        )
        modifier = _modifier(config)
        profile = _empty_profile()
        profile.situational = SituationalContext(
            origin="agent_initiated", trigger_description="test", urgency="critical",
        )
        profile.historical = HistoricalContext(
            trust_current=0.2, trust_direction="falling",
            trust_trajectory_summary="bad",
        )
        result = modifier.modify(_action(), _context(), profile)
        assert len(result.weight_adjustments) == 0
        assert len(result.constraints) == 0
        # Only thin context signals (if any) should remain
        for signal in result.risk_signals:
            assert signal.source_context == "completeness"

    def test_multiple_adjustments_to_same_dimension_accumulate(self):
        """When multiple rules affect the same dimension, their effects stack."""
        modifier = _modifier()
        profile = _empty_profile()
        # Set up scenario where cascading_impact is affected by multiple rules
        profile.workflow = WorkflowContext(
            workflow_id="wf-1", workflow_type="process", total_steps=10, current_step=1,
            rollback_points=[],  # not rollback-capable
        )
        result = modifier.modify(_action(), _context(), profile)
        cascading_adjustments = [
            a for a in result.weight_adjustments if a.dimension_name == "cascading_impact"
        ]
        # Should have multiple adjustments from early step + no rollback
        assert len(cascading_adjustments) >= 2

    def test_trust_modifier_stays_within_configured_range(self):
        config = ModifierConfig(trust_modifier_range=0.1)
        modifier = _modifier(config)
        profile = _empty_profile()
        # Multiple sources of trust modification
        profile.historical = HistoricalContext(
            trust_current=0.2, trust_direction="stable",
            trust_trajectory_summary="stable",
            behavioral_drift_status="critical_drift",
        )
        profile.meta = MetaContext(resubmission=True, revise_count=5)
        profile.feedback = FeedbackContext(
            override_history=[
                OverrideRecord(1, "ALLOW", "DENY", "owner", "bad", "2024-01-01T00:00:00Z"),
                OverrideRecord(2, "ALLOW", "DENY", "owner", "bad", "2024-01-02T00:00:00Z"),
                OverrideRecord(3, "ALLOW", "DENY", "owner", "bad", "2024-01-03T00:00:00Z"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        assert -0.1 <= result.trust_modifier <= 0.1


# ── Constraint Rules Tests ───────────────────────────────────────────────


class TestConstraintRules:
    """Tests for constraint generation rules."""

    def test_degraded_mode_adds_require_human_review(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.situational = SituationalContext(
            origin="user_request", trigger_description="test",
            operational_mode="degraded",
        )
        result = modifier.modify(_action(), _context(), profile)
        hr_constraints = [
            c for c in result.constraints if c.constraint_type == "require_human_review"
        ]
        assert len(hr_constraints) >= 1

    def test_high_workflow_denial_rate_adds_elevated_audit(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.workflow = WorkflowContext(
            workflow_id="wf-1", workflow_type="process", total_steps=10, current_step=5,
            steps_completed=[
                CompletedStep("s1", 1, "read", "data", "ALLOW", 0.8, "2024-01-01T00:00:00Z"),
                CompletedStep("s2", 2, "write", "data", "DENY", 0.3, "2024-01-01T01:00:00Z"),
                CompletedStep("s3", 3, "read", "data", "DENY", 0.2, "2024-01-01T02:00:00Z"),
                CompletedStep("s4", 4, "write", "data", "ALLOW", 0.7, "2024-01-01T03:00:00Z"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        audit_constraints = [
            c for c in result.constraints if c.constraint_type == "elevated_audit"
        ]
        assert len(audit_constraints) >= 1

    def test_low_trust_adds_require_human_review_required(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.historical = HistoricalContext(
            trust_current=0.2,
            trust_direction="falling",
            trust_trajectory_summary="declining",
        )
        result = modifier.modify(_action(), _context(), profile)
        hr_constraints = [
            c for c in result.constraints
            if c.constraint_type == "require_human_review" and c.severity == "required"
        ]
        assert len(hr_constraints) >= 1

    def test_degraded_environment_adds_reduced_authority(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.external = ExternalContext(
            environment_status="degraded",
        )
        result = modifier.modify(_action(), _context(), profile)
        ra_constraints = [
            c for c in result.constraints if c.constraint_type == "reduced_authority"
        ]
        assert len(ra_constraints) >= 1

    def test_high_denial_count_in_meta_adds_require_human_review(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.meta = MetaContext(denial_count=5)
        result = modifier.modify(_action(), _context(), profile)
        hr_constraints = [
            c for c in result.constraints
            if c.constraint_type == "require_human_review" and c.severity == "required"
        ]
        assert len(hr_constraints) >= 1

    def test_constraints_include_source_context(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.situational = SituationalContext(
            origin="user_request", trigger_description="test",
            operational_mode="degraded",
        )
        result = modifier.modify(_action(), _context(), profile)
        for constraint in result.constraints:
            assert constraint.source_context != ""

    def test_constraint_severity_levels_correct(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.historical = HistoricalContext(
            trust_current=0.2,
            trust_direction="falling",
            trust_trajectory_summary="declining",
        )
        profile.situational = SituationalContext(
            origin="user_request", trigger_description="test",
            operational_mode="degraded",
        )
        result = modifier.modify(_action(), _context(), profile)
        for constraint in result.constraints:
            assert constraint.severity in ("advisory", "recommended", "required")


# ── Risk Signal Rules Tests ──────────────────────────────────────────────


class TestRiskSignalRules:
    """Tests for risk signal generation rules."""

    def test_unresolved_dependency_produces_critical_signal(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.workflow = WorkflowContext(
            workflow_id="wf-1", workflow_type="process", total_steps=5, current_step=3,
            dependencies=[
                Dependency(from_step=2, to_step=3, dependency_type="requires", description="step 3 needs step 2"),
            ],
            steps_completed=[
                CompletedStep("s1", 1, "read", "data", "ALLOW", 0.8, "2024-01-01T00:00:00Z"),
                # Step 2 NOT completed
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        unresolved = [
            s for s in result.risk_signals if s.signal_type == "unresolved_dependency"
        ]
        assert len(unresolved) >= 1
        assert unresolved[0].severity == "critical"

    def test_compound_capability_produces_critical_signal(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.relational = RelationalContext(
            compound_methods=[
                CompoundMethod("transfer_funds", ["agent-1", "agent-2"], "combined capability"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        compound = [
            s for s in result.risk_signals if s.signal_type == "compound_capability"
        ]
        assert len(compound) >= 1
        assert compound[0].severity == "critical"

    def test_adversarial_input_produces_critical_signal(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.input_context = InputContext(
            input_type="text",
            input_summary="suspicious pattern",
            input_category="adversarial_pattern",
        )
        result = modifier.modify(_action(), _context(), profile)
        adversarial = [
            s for s in result.risk_signals if s.signal_type == "adversarial_input"
        ]
        assert len(adversarial) >= 1
        assert adversarial[0].severity == "critical"

    def test_intent_method_mismatch_detected(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.input_context = InputContext(
            input_type="text",
            input_summary="schedule a meeting",
            input_category="standard_request",
            intent_classification="schedule",
        )
        result = modifier.modify(
            _action(action_type="delete", target="meeting_room"),
            _context(),
            profile,
        )
        mismatch = [
            s for s in result.risk_signals if s.signal_type == "intent_method_mismatch"
        ]
        assert len(mismatch) >= 1

    def test_deep_delegation_chain_flagged(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.relational = RelationalContext(
            delegation_chain=[
                DelegationLink("a1", "a2", ["read"], "2024-01-01T00:00:00Z"),
                DelegationLink("a2", "a3", ["read"], "2024-01-01T01:00:00Z"),
                DelegationLink("a3", "a4", ["read"], "2024-01-01T02:00:00Z"),
                DelegationLink("a4", "a5", ["read"], "2024-01-01T03:00:00Z"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        deep_chain = [
            s for s in result.risk_signals if s.signal_type == "deep_delegation_chain"
        ]
        assert len(deep_chain) >= 1

    def test_declining_reasoning_quality_flagged(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.historical = HistoricalContext(
            trust_current=0.5,
            trust_direction="stable",
            trust_trajectory_summary="stable",
            reasoning_quality_trend="declining",
        )
        result = modifier.modify(_action(), _context(), profile)
        declining = [
            s for s in result.risk_signals if s.signal_type == "declining_reasoning_quality"
        ]
        assert len(declining) >= 1

    def test_thin_context_on_high_stakes_method_flagged(self):
        modifier = _modifier()
        profile = _empty_profile()  # completeness = 0.0
        result = modifier.modify(
            _action(action_type="transaction"),
            _context(),
            profile,
        )
        thin = [
            s for s in result.risk_signals if s.signal_type == "thin_context_high_stakes"
        ]
        assert len(thin) >= 1

    def test_stale_data_dependency_flagged(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.external = ExternalContext(
            data_freshness={"market_data": "stale - last updated 3 days ago"},
        )
        result = modifier.modify(_action(), _context(), profile)
        stale = [
            s for s in result.risk_signals if s.signal_type == "stale_data_dependency"
        ]
        assert len(stale) >= 1

    def test_deadline_pressure_flagged(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.temporal = TemporalContext(
            current_time="2024-01-01T10:00:00Z",
            time_pressure="high",
            deadline="2024-01-01T11:00:00Z",
        )
        result = modifier.modify(_action(), _context(), profile)
        deadline = [
            s for s in result.risk_signals if s.signal_type == "deadline_pressure"
        ]
        assert len(deadline) >= 1

    def test_risk_signals_include_affected_dimensions(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.relational = RelationalContext(
            compound_methods=[
                CompoundMethod("transfer", ["a1", "a2"], "combined"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        for signal in result.risk_signals:
            assert isinstance(signal.affected_dimensions, list)

    def test_no_context_produces_minimal_signals(self):
        modifier = _modifier()
        profile = _empty_profile()
        result = modifier.modify(_action(), _context(), profile)
        # Only thin context signals
        for signal in result.risk_signals:
            assert signal.source_context == "completeness"

    def test_multiple_signals_from_different_analyzers_accumulate(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.relational = RelationalContext(
            compound_methods=[
                CompoundMethod("transfer", ["a1", "a2"], "combined"),
            ],
        )
        profile.input_context = InputContext(
            input_type="text",
            input_summary="suspicious",
            input_category="adversarial_pattern",
        )
        result = modifier.modify(_action(), _context(), profile)
        sources = {s.source_context for s in result.risk_signals}
        assert "relational" in sources
        assert "input" in sources


# ── Thin Context Assessment Tests ────────────────────────────────────────


class TestThinContextAssessment:
    """Tests for thin context assessment."""

    def test_high_stakes_method_with_thin_context_produces_signal(self):
        modifier = _modifier()
        profile = _empty_profile()  # completeness = 0.0
        result = modifier.modify(
            _action(action_type="transaction"),
            _context(),
            profile,
        )
        thin = [s for s in result.risk_signals if s.signal_type == "thin_context_high_stakes"]
        assert len(thin) >= 1
        assert thin[0].severity == "high"

    def test_low_stakes_with_thin_context_no_high_stakes_signal(self):
        modifier = _modifier()
        profile = _empty_profile()
        result = modifier.modify(
            _action(action_type="read"),  # not a high-stakes method
            _context(),
            profile,
        )
        thin = [s for s in result.risk_signals if s.signal_type == "thin_context_high_stakes"]
        assert len(thin) == 0

    def test_full_context_no_thin_context_signal(self):
        modifier = _modifier()
        profile = _empty_profile()
        # Populate all 10 context types
        profile.workflow = WorkflowContext("wf-1", "process", 5, 1)
        profile.situational = SituationalContext("user_request", "test")
        profile.relational = RelationalContext()
        profile.temporal = TemporalContext("2024-01-01T10:00:00Z")
        profile.historical = HistoricalContext(0.5, "stable", "stable")
        profile.input_context = InputContext("text", "test", "standard_request")
        profile.output = OutputContext()
        profile.external = ExternalContext()
        profile.meta = MetaContext()
        profile.feedback = FeedbackContext()
        result = modifier.modify(
            _action(action_type="transaction"),
            _context(),
            profile,
        )
        thin = [s for s in result.risk_signals if s.signal_type == "thin_context_high_stakes"]
        assert len(thin) == 0

    def test_recommends_full_deliberation_for_thin_context_high_stakes(self):
        modifier = _modifier()
        profile = _empty_profile()
        result = modifier.modify(
            _action(action_type="transaction"),
            _context(),
            profile,
        )
        assert result.recommended_flow == "full"

    def test_minimal_context_always_flags(self):
        modifier = _modifier()
        profile = _empty_profile()
        result = modifier.modify(
            _action(action_type="read"),
            _context(),
            profile,
        )
        minimal = [s for s in result.risk_signals if s.signal_type == "minimal_context"]
        assert len(minimal) >= 1


# ── Trust Modifier Tests ─────────────────────────────────────────────────


class TestTrustModifier:
    """Tests for trust modifier behavior."""

    def test_positive_feedback_adds_positive_trust_modifier(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.feedback = FeedbackContext(
            feedback_received=[
                FeedbackRecord("user", "positive", "good work", "2024-01-01T00:00:00Z"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        assert result.trust_modifier > 0.0

    def test_override_to_deny_adds_negative_trust_modifier(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.feedback = FeedbackContext(
            override_history=[
                OverrideRecord(1, "ALLOW", "DENY", "owner", "too lenient", "2024-01-01T00:00:00Z"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        assert result.trust_modifier < 0.0

    def test_falling_trust_direction_adds_negative_modifier(self):
        """High drift adds negative trust modifier (from historical)."""
        modifier = _modifier()
        profile = _empty_profile()
        profile.historical = HistoricalContext(
            trust_current=0.4,
            trust_direction="stable",
            trust_trajectory_summary="stable",
            behavioral_drift_status="high_drift",
        )
        result = modifier.modify(_action(), _context(), profile)
        assert result.trust_modifier < 0.0

    def test_trust_modifier_clamped_to_range(self):
        config = ModifierConfig(trust_modifier_range=0.1)
        modifier = _modifier(config)
        profile = _empty_profile()
        # Combine many negative trust modifiers
        profile.historical = HistoricalContext(
            trust_current=0.2, trust_direction="stable",
            trust_trajectory_summary="declining",
            behavioral_drift_status="critical_drift",
        )
        profile.meta = MetaContext(resubmission=True, revise_count=5)
        profile.feedback = FeedbackContext(
            override_history=[
                OverrideRecord(i, "ALLOW", "DENY", "owner", "bad", f"2024-01-0{i}T00:00:00Z")
                for i in range(1, 6)
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        assert result.trust_modifier >= -0.1
        assert result.trust_modifier <= 0.1

    def test_multiple_trust_modifiers_accumulate_within_range(self):
        modifier = _modifier()
        profile = _empty_profile()
        # Positive feedback + override to approve should both add trust
        profile.feedback = FeedbackContext(
            feedback_received=[
                FeedbackRecord("user", "positive", "good", "2024-01-01T00:00:00Z"),
            ],
            override_history=[
                OverrideRecord(1, "DENY", "ALLOW", "owner", "too strict", "2024-01-01T00:00:00Z"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        # Both add positive trust: 0.02 (positive feedback) + 0.02 (override to approve)
        assert result.trust_modifier > 0.0


# ── Integration with Runtime Tests ───────────────────────────────────────


class TestRuntimeIntegration:
    """Tests for integration with GovernanceRuntime."""

    def test_runtime_creates_modifier_when_enabled(self):
        config = RuntimeConfig(enable_contextual_modifier=True)
        runtime = GovernanceRuntime(config=config)
        assert runtime.contextual_modifier is not None

    def test_runtime_skips_modifier_when_disabled(self):
        config = RuntimeConfig(enable_contextual_modifier=False)
        runtime = GovernanceRuntime(config=config)
        assert runtime.contextual_modifier is None

    def test_evaluate_applies_weight_adjustments_from_modifier(self):
        runtime = GovernanceRuntime()
        # Create a context profile with agent-initiated origin
        profile = runtime.context_profiles.create_profile(
            agent_id="agent-1",
            profile_type="workflow",
            situational=SituationalContext(
                origin="agent_initiated", trigger_description="self-started",
            ),
        )
        # Set up scope for the agent
        scope_dim = runtime.registry.get("scope_compliance")
        scope_dim.configure_agent_scope("agent-1", {"read"})
        iso_dim = runtime.registry.get("isolation_integrity")
        iso_dim.set_boundaries("agent-1", {"data"})

        action = Action(agent_id="agent-1", action_type="read", target="data")
        ctx = AgentContext(
            agent_id="agent-1",
            trust_profile=runtime.get_trust_profile("agent-1"),
            context_profile_id=profile.profile_id,
        )
        verdict = runtime.evaluate(action, ctx)
        # The verdict should include context modification
        assert verdict.context_modification is not None
        assert verdict.context_modification.has_modifications()

    def test_weight_adjustments_are_per_evaluation_restored_after(self):
        runtime = GovernanceRuntime()
        # Record original weight of human_override
        ho_dim = runtime.registry.get("human_override")
        original_weight = ho_dim.weight

        profile = runtime.context_profiles.create_profile(
            agent_id="agent-1",
            profile_type="workflow",
            situational=SituationalContext(
                origin="agent_initiated", trigger_description="self-started",
            ),
        )
        scope_dim = runtime.registry.get("scope_compliance")
        scope_dim.configure_agent_scope("agent-1", {"read"})
        iso_dim = runtime.registry.get("isolation_integrity")
        iso_dim.set_boundaries("agent-1", {"data"})

        action = Action(agent_id="agent-1", action_type="read", target="data")
        ctx = AgentContext(
            agent_id="agent-1",
            trust_profile=runtime.get_trust_profile("agent-1"),
            context_profile_id=profile.profile_id,
        )
        runtime.evaluate(action, ctx)

        # Weight should be restored to original
        assert ho_dim.weight == original_weight

    def test_verdict_includes_context_modification_when_profile_present(self):
        runtime = GovernanceRuntime()
        profile = runtime.context_profiles.create_profile(
            agent_id="agent-1",
            profile_type="workflow",
        )
        scope_dim = runtime.registry.get("scope_compliance")
        scope_dim.configure_agent_scope("agent-1", {"read"})
        iso_dim = runtime.registry.get("isolation_integrity")
        iso_dim.set_boundaries("agent-1", {"data"})

        action = Action(agent_id="agent-1", action_type="read", target="data")
        ctx = AgentContext(
            agent_id="agent-1",
            trust_profile=runtime.get_trust_profile("agent-1"),
            context_profile_id=profile.profile_id,
        )
        verdict = runtime.evaluate(action, ctx)
        assert verdict.context_modification is not None

    def test_no_profile_no_modification_evaluation_works(self):
        runtime = GovernanceRuntime()
        scope_dim = runtime.registry.get("scope_compliance")
        scope_dim.configure_agent_scope("agent-1", {"read"})
        iso_dim = runtime.registry.get("isolation_integrity")
        iso_dim.set_boundaries("agent-1", {"data"})

        action = Action(agent_id="agent-1", action_type="read", target="data")
        ctx = AgentContext(
            agent_id="agent-1",
            trust_profile=runtime.get_trust_profile("agent-1"),
            # No context_profile_id
        )
        verdict = runtime.evaluate(action, ctx)
        assert verdict.context_modification is None
        assert verdict.verdict is not None

    def test_all_existing_tests_pass_no_regression(self):
        """Basic sanity: runtime works without contextual modifier features."""
        runtime = GovernanceRuntime()
        scope_dim = runtime.registry.get("scope_compliance")
        scope_dim.configure_agent_scope("agent-test", {"read", "write"})
        iso_dim = runtime.registry.get("isolation_integrity")
        iso_dim.set_boundaries("agent-test", {"docs", "data"})

        action = Action(agent_id="agent-test", action_type="read", target="docs")
        ctx = AgentContext(
            agent_id="agent-test",
            trust_profile=runtime.get_trust_profile("agent-test"),
        )
        verdict = runtime.evaluate(action, ctx)
        assert verdict.verdict == Verdict.ALLOW

    def test_modifier_with_protocol_evaluator_work_together(self):
        """Protocol evaluator should pass context_profile_id through to runtime."""
        from nomotic.evaluator import EvaluatorConfig, ProtocolEvaluator
        from nomotic.protocol import (
            AuthorityClaim,
            Constraint,
            Factor,
            IntendedAction,
            Justification,
            ReasoningArtifact,
        )

        runtime = GovernanceRuntime()
        scope_dim = runtime.registry.get("scope_compliance")
        scope_dim.configure_agent_scope("agent-1", {"read"})
        iso_dim = runtime.registry.get("isolation_integrity")
        iso_dim.set_boundaries("agent-1", {"data"})

        # Create a context profile
        profile = runtime.context_profiles.create_profile(
            agent_id="agent-1",
            profile_type="workflow",
            situational=SituationalContext(
                origin="agent_initiated", trigger_description="self-started",
            ),
        )

        evaluator = ProtocolEvaluator(
            config=EvaluatorConfig(),
            runtime=runtime,
        )

        from nomotic.protocol import Alternative, Assumption, Unknown
        artifact = ReasoningArtifact(
            agent_id="agent-1",
            goal="Read data from storage",
            origin="user_request",
            constraints_identified=[
                Constraint(type="technical", description="agent has read scope", source="system"),
            ],
            factors=[
                Factor(id="f1", type="constraint", description="agent has scope for read",
                       source="system", assessment="positive", influence="decisive", confidence=0.9),
                Factor(id="f2", type="risk", description="low risk read operation",
                       source="analysis", assessment="low risk", influence="minor", confidence=0.8),
            ],
            alternatives_considered=[
                Alternative(method="query", reason_rejected="read is more direct"),
            ],
            intended_action=IntendedAction(method="read", target="data"),
            justifications=[
                Justification(factor_id="f1", explanation="agent has scope for this action"),
            ],
            authority_claim=AuthorityClaim(envelope_type="standard"),
            unknowns=[
                Unknown(description="data size unknown", impact="minimal"),
            ],
            assumptions=[
                Assumption(description="data is accessible", basis="system config", risk_if_wrong="action fails gracefully"),
            ],
            overall_confidence=0.8,
            context_profile_id=profile.profile_id,
        )
        response = evaluator.evaluate(artifact)
        # Verify the response includes context modification info
        assert response.assessment is not None
        # The context modifications should be in assessment metadata
        assert "context_modifications" in response.assessment.metadata

    def test_audit_record_captures_context_modifications(self):
        """When audit is enabled, context modifications appear in the audit record metadata."""
        config = RuntimeConfig(enable_audit=True)
        runtime = GovernanceRuntime(config=config)
        scope_dim = runtime.registry.get("scope_compliance")
        scope_dim.configure_agent_scope("agent-1", {"read"})
        iso_dim = runtime.registry.get("isolation_integrity")
        iso_dim.set_boundaries("agent-1", {"data"})

        profile = runtime.context_profiles.create_profile(
            agent_id="agent-1",
            profile_type="workflow",
            situational=SituationalContext(
                origin="agent_initiated", trigger_description="test",
            ),
        )
        action = Action(agent_id="agent-1", action_type="read", target="data")
        ctx = AgentContext(
            agent_id="agent-1",
            trust_profile=runtime.get_trust_profile("agent-1"),
            context_profile_id=profile.profile_id,
        )
        runtime.evaluate(action, ctx)

        # Check audit trail
        if runtime.audit_trail is not None:
            records = runtime.audit_trail.query(agent_id="agent-1", limit=1)
            if records:
                assert "context_modification" in records[0].metadata

    def test_preview_endpoint_returns_expected_modifications(self):
        """Test the modifications preview by directly calling the modifier."""
        runtime = GovernanceRuntime()
        profile = runtime.context_profiles.create_profile(
            agent_id="agent-1",
            profile_type="workflow",
            situational=SituationalContext(
                origin="agent_initiated", trigger_description="test",
            ),
        )
        action = Action(
            agent_id="agent-1", action_type="transaction", target="account",
        )
        ctx = AgentContext(
            agent_id="agent-1",
            trust_profile=runtime.get_trust_profile("agent-1"),
            context_profile_id=profile.profile_id,
        )
        modification = runtime.contextual_modifier.modify(action, ctx, profile)
        assert modification is not None
        assert modification.has_modifications()
        # Agent-initiated should produce human_override weight adjustment
        ho_adjustments = [
            a for a in modification.weight_adjustments
            if a.dimension_name == "human_override"
        ]
        assert len(ho_adjustments) >= 1


# ── Modification Reasoning Tests ─────────────────────────────────────────


class TestModificationReasoning:
    """Tests for human-readable modification reasoning."""

    def test_reasoning_includes_all_adjustment_sources(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.situational = SituationalContext(
            origin="agent_initiated", trigger_description="test",
        )
        profile.historical = HistoricalContext(
            trust_current=0.5,
            trust_direction="falling",
            trust_trajectory_summary="declining",
        )
        result = modifier.modify(_action(), _context(), profile)
        assert "Situational context" in result.modification_reasoning
        assert "Historical context" in result.modification_reasoning

    def test_reasoning_is_human_readable(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.situational = SituationalContext(
            origin="agent_initiated", trigger_description="test",
        )
        result = modifier.modify(_action(), _context(), profile)
        assert isinstance(result.modification_reasoning, str)
        assert len(result.modification_reasoning) > 10

    def test_empty_modification_has_appropriate_reasoning(self):
        config = ModifierConfig(
            enable_workflow_modifiers=False,
            enable_situational_modifiers=False,
            enable_relational_modifiers=False,
            enable_temporal_modifiers=False,
            enable_historical_modifiers=False,
            enable_input_modifiers=False,
            enable_external_modifiers=False,
            enable_meta_modifiers=False,
            enable_feedback_modifiers=False,
        )
        modifier = _modifier(config)
        profile = _empty_profile()
        # Populate to avoid thin context signals
        profile.workflow = WorkflowContext("wf-1", "process", 5, 1)
        profile.situational = SituationalContext("user_request", "test")
        profile.relational = RelationalContext()
        profile.temporal = TemporalContext("2024-01-01T10:00:00Z")
        profile.historical = HistoricalContext(0.5, "stable", "stable")
        profile.input_context = InputContext("text", "test", "standard_request")
        profile.output = OutputContext()
        profile.external = ExternalContext()
        profile.meta = MetaContext()
        profile.feedback = FeedbackContext()
        result = modifier.modify(_action(), _context(), profile)
        assert "No contextual modifications" in result.modification_reasoning

    def test_complex_scenario_produces_comprehensive_reasoning(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.workflow = WorkflowContext(
            workflow_id="wf-1", workflow_type="process", total_steps=10, current_step=1,
        )
        profile.situational = SituationalContext(
            origin="agent_initiated", trigger_description="self-initiated",
            urgency="critical",
        )
        profile.relational = RelationalContext(
            compound_methods=[
                CompoundMethod("transfer", ["a1", "a2"], "combined"),
            ],
        )
        profile.historical = HistoricalContext(
            trust_current=0.3,
            trust_direction="falling",
            trust_trajectory_summary="declining",
            behavioral_drift_status="high_drift",
        )
        result = modifier.modify(_action(), _context(), profile)
        # Should mention multiple context types
        assert "Workflow context" in result.modification_reasoning
        assert "Situational context" in result.modification_reasoning
        assert "Relational context" in result.modification_reasoning
        assert "Historical context" in result.modification_reasoning

    def test_context_completeness_stored_on_modification(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.workflow = WorkflowContext("wf-1", "process", 5, 1)
        profile.situational = SituationalContext("user_request", "test")
        result = modifier.modify(_action(), _context(), profile)
        # 2 out of 10 contexts populated = 0.2
        assert result.context_completeness == pytest.approx(0.2, abs=0.01)


# ── Additional Analyzer Rule Tests ───────────────────────────────────────


class TestAdditionalRules:
    """Additional tests for specific analyzer rules."""

    def test_workflow_rolling_back_elevates_security(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.workflow = WorkflowContext(
            workflow_id="wf-1", workflow_type="process", total_steps=5, current_step=3,
            status="rolling_back",
        )
        result = modifier.modify(_action(), _context(), profile)
        security_adj = [
            a for a in result.weight_adjustments
            if a.dimension_name in ("incident_detection", "isolation_integrity")
            and "rollback" in a.reason.lower()
        ]
        assert len(security_adj) >= 1

    def test_incident_response_mode_adjustments(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.situational = SituationalContext(
            origin="user_request", trigger_description="test",
            operational_mode="incident_response",
        )
        result = modifier.modify(_action(), _context(), profile)
        # Security dimensions should increase
        security_adj = [
            a for a in result.weight_adjustments
            if a.dimension_name in ("incident_detection", "isolation_integrity")
        ]
        assert len(security_adj) >= 1
        # Temporal compliance should decrease
        temporal_adj = [
            a for a in result.weight_adjustments
            if a.dimension_name == "temporal_compliance"
        ]
        assert any(a.adjusted_weight < a.original_weight for a in temporal_adj)

    def test_post_incident_operational_state(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.temporal = TemporalContext(
            current_time="2024-01-01T10:00:00Z",
            operational_state="post_incident",
        )
        result = modifier.modify(_action(), _context(), profile)
        incident_adj = [
            a for a in result.weight_adjustments
            if a.dimension_name == "incident_detection"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in incident_adj)

    def test_after_hours_agent_initiated(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.temporal = TemporalContext(
            current_time="2024-01-01T23:00:00Z",
            time_of_day_category="after_hours",
        )
        profile.situational = SituationalContext(
            origin="agent_initiated", trigger_description="self-started",
        )
        result = modifier.modify(_action(), _context(), profile)
        ho_adj = [
            a for a in result.weight_adjustments
            if a.dimension_name == "human_override"
        ]
        # Should have adjustments from both situational (agent_initiated) and temporal (after_hours)
        assert len(ho_adj) >= 2

    def test_volatile_trust_increases_behavioral_consistency(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.historical = HistoricalContext(
            trust_current=0.5,
            trust_direction="volatile",
            trust_trajectory_summary="volatile pattern",
        )
        result = modifier.modify(_action(), _context(), profile)
        bc_adj = [
            a for a in result.weight_adjustments
            if a.dimension_name == "behavioral_consistency"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in bc_adj)

    def test_scope_changes_increase_scope_and_authority(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.historical = HistoricalContext(
            trust_current=0.5,
            trust_direction="stable",
            trust_trajectory_summary="stable",
            scope_changes_recent=["added write permissions", "expanded target scope"],
        )
        result = modifier.modify(_action(), _context(), profile)
        scope_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "scope_compliance"
        ]
        auth_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "authority_verification"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in scope_adj)
        assert any(a.adjusted_weight > a.original_weight for a in auth_adj)

    def test_edge_case_input_increases_ethical_and_precedent(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.input_context = InputContext(
            input_type="text",
            input_summary="unusual request at boundary",
            input_category="edge_case",
        )
        result = modifier.modify(_action(), _context(), profile)
        ethical_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "ethical_alignment"
        ]
        precedent_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "precedent_alignment"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in ethical_adj)
        assert any(a.adjusted_weight > a.original_weight for a in precedent_adj)

    def test_ambiguous_input_increases_transparency(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.input_context = InputContext(
            input_type="text",
            input_summary="unclear request",
            input_category="ambiguous",
        )
        result = modifier.modify(_action(), _context(), profile)
        transparency_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "transparency"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in transparency_adj)

    def test_shared_workflow_agents_increases_isolation(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.relational = RelationalContext(
            shared_workflow_agents=["agent-2", "agent-3"],
        )
        result = modifier.modify(_action(), _context(), profile)
        iso_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "isolation_integrity"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in iso_adj)

    def test_child_delegations_increase_cascading_impact(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.relational = RelationalContext(
            child_agent_ids=["child-1", "child-2"],
        )
        result = modifier.modify(_action(), _context(), profile)
        cascading_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "cascading_impact"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in cascading_adj)

    def test_active_alerts_elevate_security_dimensions(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.external = ExternalContext(
            active_alerts=["security_alert_1", "system_warning_2"],
        )
        result = modifier.modify(_action(), _context(), profile)
        security_adj = [
            a for a in result.weight_adjustments
            if a.dimension_name in ("incident_detection", "isolation_integrity")
        ]
        assert len(security_adj) >= 1

    def test_downstream_failures_increase_cascading_impact(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.feedback = FeedbackContext(
            downstream_outcomes=[
                OutcomeRecord(1, "write", "success", "failed", "failed", "2024-01-01T00:00:00Z"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        cascading_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "cascading_impact"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in cascading_adj)

    def test_frequent_escalations_flagged(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.meta = MetaContext(escalation_count=4)
        result = modifier.modify(_action(), _context(), profile)
        escalation_signals = [
            s for s in result.risk_signals if s.signal_type == "frequent_escalations"
        ]
        assert len(escalation_signals) >= 1

    def test_high_governance_load_informational_only(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.meta = MetaContext(governance_load="high")
        result = modifier.modify(_action(), _context(), profile)
        load_signals = [
            s for s in result.risk_signals if s.signal_type == "high_governance_load"
        ]
        assert len(load_signals) >= 1
        assert load_signals[0].severity == "low"

    def test_resubmission_after_revise_negative_trust(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.meta = MetaContext(resubmission=True, revise_count=1)
        result = modifier.modify(_action(), _context(), profile)
        assert result.trust_modifier < 0.0

    def test_negative_feedback_increases_stakeholder_impact(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.feedback = FeedbackContext(
            feedback_received=[
                FeedbackRecord("user", "negative", "poor outcome", "2024-01-01T00:00:00Z"),
                FeedbackRecord("user", "complaint", "unresponsive", "2024-01-01T01:00:00Z"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        stakeholder_adj = [
            a for a in result.weight_adjustments if a.dimension_name == "stakeholder_impact"
        ]
        assert any(a.adjusted_weight > a.original_weight for a in stakeholder_adj)

    def test_constraining_dependency_elevates_authority(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.workflow = WorkflowContext(
            workflow_id="wf-1", workflow_type="process", total_steps=5, current_step=2,
            dependencies=[
                Dependency(from_step=2, to_step=4, dependency_type="constrains", description="step 2 constrains step 4"),
            ],
        )
        result = modifier.modify(_action(), _context(), profile)
        auth_adj = [
            a for a in result.weight_adjustments
            if a.dimension_name == "authority_verification" and "constrain" in a.reason.lower()
        ]
        assert len(auth_adj) >= 1

    def test_critical_urgency_increases_all_dimensions(self):
        modifier = _modifier()
        profile = _empty_profile()
        profile.situational = SituationalContext(
            origin="user_request", trigger_description="test",
            urgency="critical",
        )
        result = modifier.modify(_action(), _context(), profile)
        adjusted_dims = {a.dimension_name for a in result.weight_adjustments}
        # Critical urgency should affect all 13 dimensions
        assert len(adjusted_dims) == 13
