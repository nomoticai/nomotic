"""Tests for the Context Profile system (Phase 7A).

Comprehensive tests covering:
- Context type dataclasses (creation, serialization)
- ContextProfile container (completeness, risk signals, mutations)
- ContextProfileManager (CRUD, eviction, enrichment, thread safety)
- Full serialization round-trips
- Integration with AgentContext, GovernanceRuntime, and the HTTP API
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
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


# ── Helpers ────────────────────────────────────────────────────────────


def _make_workflow(**overrides: Any) -> WorkflowContext:
    defaults: dict[str, Any] = {
        "workflow_id": "wf-001",
        "workflow_type": "customer_return",
        "total_steps": 5,
        "current_step": 1,
        "started_at": "2025-01-15T10:00:00+00:00",
    }
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def _make_situational(**overrides: Any) -> SituationalContext:
    defaults: dict[str, Any] = {
        "origin": "user_request",
        "trigger_description": "Customer initiated a return",
    }
    defaults.update(overrides)
    return SituationalContext(**defaults)


def _make_completed_step(**overrides: Any) -> CompletedStep:
    defaults: dict[str, Any] = {
        "step_id": "s1",
        "step_number": 1,
        "method": "data.read",
        "target": "orders",
        "verdict": "ALLOW",
        "ucs": 0.85,
        "timestamp": "2025-01-15T10:01:00+00:00",
    }
    defaults.update(overrides)
    return CompletedStep(**defaults)


def _make_profile(**overrides: Any) -> ContextProfile:
    defaults: dict[str, Any] = {
        "profile_id": "cp-test123",
        "agent_id": "agent-1",
        "profile_type": "workflow",
    }
    defaults.update(overrides)
    return ContextProfile(**defaults)


# ═══════════════════════════════════════════════════════════════════════
# Context Type Dataclasses (~15 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestContextTypeCreation:
    """Test creation of each context type with valid data."""

    def test_workflow_context_creation(self):
        wf = _make_workflow()
        assert wf.workflow_id == "wf-001"
        assert wf.workflow_type == "customer_return"
        assert wf.total_steps == 5
        assert wf.current_step == 1
        assert wf.status == "active"

    def test_situational_context_creation(self):
        sc = _make_situational(urgency="elevated", operational_mode="incident_response")
        assert sc.origin == "user_request"
        assert sc.urgency == "elevated"
        assert sc.operational_mode == "incident_response"

    def test_relational_context_creation(self):
        rc = RelationalContext(
            parent_agent_id="parent-1",
            child_agent_ids=["child-1", "child-2"],
            shared_workflow_agents=["peer-1"],
        )
        assert rc.parent_agent_id == "parent-1"
        assert len(rc.child_agent_ids) == 2

    def test_temporal_context_creation(self):
        tc = TemporalContext(
            current_time="2025-01-15T10:00:00+00:00",
            time_of_day_category="business_hours",
            time_pressure="high",
            deadline="2025-01-15T12:00:00+00:00",
        )
        assert tc.time_pressure == "high"
        assert tc.deadline is not None

    def test_historical_context_creation(self):
        hc = HistoricalContext(
            trust_current=0.75,
            trust_direction="rising",
            trust_trajectory_summary="Improving steadily",
            behavioral_drift_status="low_drift",
            days_since_activation=30,
        )
        assert hc.trust_current == 0.75
        assert hc.days_since_activation == 30

    def test_input_context_creation(self):
        ic = InputContext(
            input_type="text",
            input_summary="Return request for order #1234",
            input_category="standard_request",
            entities_referenced=["order#1234"],
            input_hash="abc123hash",
        )
        assert ic.input_hash == "abc123hash"
        assert ic.input_type == "text"

    def test_output_context_creation(self):
        oc = OutputContext(
            cumulative_impact="moderate",
            reversible=False,
            external_effects=["email_sent"],
        )
        assert oc.cumulative_impact == "moderate"
        assert oc.reversible is False

    def test_external_context_creation(self):
        ec = ExternalContext(
            environment_status="degraded",
            active_alerts=["service-degradation"],
        )
        assert ec.environment_status == "degraded"
        assert len(ec.active_alerts) == 1

    def test_meta_context_creation(self):
        mc = MetaContext(
            evaluation_count=5,
            denial_count=1,
            resubmission=True,
            original_artifact_id="art-001",
        )
        assert mc.resubmission is True
        assert mc.denial_count == 1

    def test_feedback_context_creation(self):
        fc = FeedbackContext(
            satisfaction_signals=["thumbs_up"],
        )
        assert len(fc.feedback_received) == 0
        assert fc.satisfaction_signals == ["thumbs_up"]

    def test_workflow_with_completed_steps_and_dependencies(self):
        step = _make_completed_step()
        planned = PlannedStep(
            step_number=2,
            method="data.write",
            target="refunds",
            description="Process refund",
            estimated_risk="medium",
            depends_on=[1],
        )
        dep = Dependency(
            from_step=1,
            to_step=2,
            dependency_type="requires",
            description="Must validate order before refund",
        )
        wf = _make_workflow(
            steps_completed=[step],
            steps_remaining=[planned],
            dependencies=[dep],
        )
        assert len(wf.steps_completed) == 1
        assert len(wf.steps_remaining) == 1
        assert wf.dependencies[0].dependency_type == "requires"

    def test_completed_step_round_trip(self):
        step = _make_completed_step(output_summary="Read 5 records")
        d = step.to_dict()
        restored = CompletedStep.from_dict(d)
        assert restored.step_id == step.step_id
        assert restored.output_summary == "Read 5 records"
        assert restored.to_dict() == d

    def test_planned_step_round_trip(self):
        ps = PlannedStep(
            step_number=3,
            method="data.write",
            target="inventory",
            description="Update inventory",
            estimated_risk="low",
            depends_on=[1, 2],
        )
        d = ps.to_dict()
        restored = PlannedStep.from_dict(d)
        assert restored.depends_on == [1, 2]
        assert restored.to_dict() == d

    def test_dependency_round_trip(self):
        dep = Dependency(
            from_step=1, to_step=3,
            dependency_type="enables",
            description="Step 1 enables step 3",
        )
        d = dep.to_dict()
        restored = Dependency.from_dict(d)
        assert restored.dependency_type == "enables"
        assert restored.to_dict() == d

    def test_situational_context_all_origin_types(self):
        origins = [
            "user_request", "scheduled", "event_triggered",
            "agent_initiated", "escalation_received",
        ]
        for origin in origins:
            sc = SituationalContext(
                origin=origin,
                trigger_description=f"Triggered via {origin}",
            )
            assert sc.origin == origin

    def test_relational_context_with_delegation_chain(self):
        link1 = DelegationLink(
            from_id="parent",
            to_id="child-1",
            delegated_methods=["data.read", "data.write"],
            delegated_at="2025-01-15T10:00:00+00:00",
        )
        link2 = DelegationLink(
            from_id="child-1",
            to_id="child-2",
            delegated_methods=["data.read"],
            delegated_at="2025-01-15T10:05:00+00:00",
        )
        rc = RelationalContext(
            parent_agent_id="parent",
            child_agent_ids=["child-1", "child-2"],
            delegation_chain=[link1, link2],
        )
        assert len(rc.delegation_chain) == 2
        assert rc.delegation_chain[1].delegated_methods == ["data.read"]

    def test_input_context_preserves_hashes_not_raw_content(self):
        ic = InputContext(
            input_type="text",
            input_summary="Summarized version of request",
            input_category="standard_request",
            input_hash="sha256:e3b0c44298fc1c149afbf4c8996fb924",
        )
        d = ic.to_dict()
        assert d["input_hash"] == "sha256:e3b0c44298fc1c149afbf4c8996fb924"
        assert d["input_summary"] == "Summarized version of request"
        # No raw content field exists
        assert "raw_content" not in d
        assert "raw_input" not in d

    def test_feedback_context_with_multiple_feedback_types(self):
        fb1 = FeedbackRecord(
            source="user",
            feedback_type="positive",
            description="Great result",
            timestamp="2025-01-15T11:00:00+00:00",
        )
        fb2 = FeedbackRecord(
            source="manager",
            feedback_type="correction",
            description="Adjust the amount",
            timestamp="2025-01-15T11:05:00+00:00",
        )
        override = OverrideRecord(
            step_number=2,
            original_verdict="DENY",
            override_to="ALLOW",
            overridden_by="owner-1",
            reason="Business exception",
            timestamp="2025-01-15T11:10:00+00:00",
        )
        fc = FeedbackContext(
            feedback_received=[fb1, fb2],
            override_history=[override],
        )
        assert len(fc.feedback_received) == 2
        assert fc.feedback_received[0].feedback_type == "positive"
        assert fc.feedback_received[1].feedback_type == "correction"
        assert fc.override_history[0].overridden_by == "owner-1"


# ═══════════════════════════════════════════════════════════════════════
# ContextProfile Container (~20 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestContextProfile:
    """Test the ContextProfile container."""

    def test_create_empty_profile_all_contexts_none(self):
        profile = _make_profile()
        assert profile.workflow is None
        assert profile.situational is None
        assert profile.relational is None
        assert profile.temporal is None
        assert profile.historical is None
        assert profile.input_context is None
        assert profile.output is None
        assert profile.external is None
        assert profile.meta is None
        assert profile.feedback is None

    def test_create_with_partial_context(self):
        profile = _make_profile(
            workflow=_make_workflow(),
            situational=_make_situational(),
        )
        assert profile.workflow is not None
        assert profile.situational is not None
        assert profile.relational is None
        assert profile.temporal is None

    def test_completeness_score_zero(self):
        profile = _make_profile()
        assert profile.completeness_score() == 0.0

    def test_completeness_score_partial(self):
        profile = _make_profile(
            workflow=_make_workflow(),
            situational=_make_situational(),
        )
        assert profile.completeness_score() == pytest.approx(0.2)

    def test_completeness_score_full(self):
        profile = _make_profile(
            workflow=_make_workflow(),
            situational=_make_situational(),
            relational=RelationalContext(),
            temporal=TemporalContext(current_time="2025-01-15T10:00:00+00:00"),
            historical=HistoricalContext(
                trust_current=0.7,
                trust_direction="stable",
                trust_trajectory_summary="Stable",
            ),
            input_context=InputContext(
                input_type="text",
                input_summary="test",
                input_category="standard_request",
            ),
            output=OutputContext(),
            external=ExternalContext(),
            meta=MetaContext(),
            feedback=FeedbackContext(),
        )
        assert profile.completeness_score() == 1.0

    def test_risk_signals_agent_initiated_origin(self):
        profile = _make_profile(
            situational=_make_situational(origin="agent_initiated"),
        )
        signals = profile.risk_signals()
        assert any("Agent-initiated" in s for s in signals)

    def test_risk_signals_volatile_trust(self):
        profile = _make_profile(
            historical=HistoricalContext(
                trust_current=0.5,
                trust_direction="volatile",
                trust_trajectory_summary="Erratic",
            ),
        )
        signals = profile.risk_signals()
        assert any("Volatile trust" in s for s in signals)

    def test_risk_signals_high_drift(self):
        profile = _make_profile(
            historical=HistoricalContext(
                trust_current=0.5,
                trust_direction="stable",
                trust_trajectory_summary="Stable",
                behavioral_drift_status="high_drift",
            ),
        )
        signals = profile.risk_signals()
        assert any("high_drift" in s for s in signals)

    def test_risk_signals_active_alerts(self):
        profile = _make_profile(
            external=ExternalContext(
                active_alerts=["alert-1", "alert-2"],
            ),
        )
        signals = profile.risk_signals()
        assert any("Active external alerts" in s for s in signals)

    def test_risk_signals_resubmission_after_denial(self):
        profile = _make_profile(
            meta=MetaContext(
                resubmission=True,
                denial_count=2,
            ),
        )
        signals = profile.risk_signals()
        assert any("Resubmission after previous denial" in s for s in signals)

    def test_risk_signals_low_completeness(self):
        # Only 2 out of 10 contexts populated: 0.2 < 0.3
        profile = _make_profile(
            workflow=_make_workflow(),
            situational=_make_situational(),
        )
        signals = profile.risk_signals()
        assert any("Low context completeness" in s for s in signals)

    def test_risk_signals_returns_empty_list_when_no_risks(self):
        profile = _make_profile(
            workflow=_make_workflow(),
            situational=_make_situational(origin="user_request"),
            relational=RelationalContext(),
            temporal=TemporalContext(current_time="2025-01-15T10:00:00+00:00"),
            historical=HistoricalContext(
                trust_current=0.8,
                trust_direction="rising",
                trust_trajectory_summary="Good",
                behavioral_drift_status="normal",
            ),
            input_context=InputContext(
                input_type="text",
                input_summary="test",
                input_category="standard_request",
            ),
            output=OutputContext(),
            external=ExternalContext(),
            meta=MetaContext(),
            feedback=FeedbackContext(),
        )
        signals = profile.risk_signals()
        assert signals == []

    def test_update_workflow_step(self):
        profile = _make_profile(
            workflow=_make_workflow(current_step=1, total_steps=5),
        )
        step = _make_completed_step(step_number=1)
        profile.update_workflow_step(step)
        assert profile.workflow.current_step == 2
        assert len(profile.workflow.steps_completed) == 1

    def test_update_workflow_step_removes_from_remaining(self):
        planned = PlannedStep(
            step_number=1, method="data.read", target="orders",
            description="Read order",
        )
        profile = _make_profile(
            workflow=_make_workflow(
                current_step=1,
                total_steps=3,
                steps_remaining=[planned],
            ),
        )
        step = _make_completed_step(step_number=1)
        profile.update_workflow_step(step)
        assert len(profile.workflow.steps_remaining) == 0

    def test_add_feedback_creates_if_none(self):
        profile = _make_profile()
        assert profile.feedback is None
        fb = FeedbackRecord(
            source="user", feedback_type="positive",
            description="Good job", timestamp="2025-01-15T11:00:00+00:00",
        )
        profile.add_feedback(fb)
        assert profile.feedback is not None
        assert len(profile.feedback.feedback_received) == 1

    def test_add_feedback_appends(self):
        profile = _make_profile(feedback=FeedbackContext())
        fb1 = FeedbackRecord(
            source="user", feedback_type="positive",
            description="Good", timestamp="2025-01-15T11:00:00+00:00",
        )
        fb2 = FeedbackRecord(
            source="manager", feedback_type="negative",
            description="Needs improvement", timestamp="2025-01-15T11:05:00+00:00",
        )
        profile.add_feedback(fb1)
        profile.add_feedback(fb2)
        assert len(profile.feedback.feedback_received) == 2

    def test_add_output_creates_if_none(self):
        profile = _make_profile()
        assert profile.output is None
        out = OutputRecord(
            step_number=1, method="data.write", target="orders",
            output_type="data_modification", output_summary="Updated order status",
            reversible=True, timestamp="2025-01-15T10:01:00+00:00",
        )
        profile.add_output(out)
        assert profile.output is not None
        assert len(profile.output.outputs_produced) == 1

    def test_add_output_appends(self):
        profile = _make_profile(output=OutputContext())
        out = OutputRecord(
            step_number=1, method="data.write", target="orders",
            output_type="data_modification", output_summary="Updated",
            reversible=True, timestamp="2025-01-15T10:01:00+00:00",
        )
        profile.add_output(out)
        assert len(profile.output.outputs_produced) == 1

    def test_add_output_irreversible_updates_flag(self):
        profile = _make_profile(output=OutputContext(reversible=True))
        out = OutputRecord(
            step_number=1, method="payment.send", target="bank",
            output_type="transaction", output_summary="Sent $100",
            reversible=False, timestamp="2025-01-15T10:01:00+00:00",
        )
        profile.add_output(out)
        assert profile.output.reversible is False

    def test_add_external_signal_creates_if_none(self):
        profile = _make_profile()
        assert profile.external is None
        sig = ExternalSignal(
            source="system_monitor", signal_type="cpu_spike",
            description="CPU at 95%", severity="alert",
            received_at="2025-01-15T10:01:00+00:00",
        )
        profile.add_external_signal(sig)
        assert profile.external is not None
        assert len(profile.external.external_signals) == 1

    def test_add_external_signal_appends(self):
        profile = _make_profile(external=ExternalContext())
        sig = ExternalSignal(
            source="threat_intel", signal_type="new_vulnerability",
            description="CVE-2025-1234", severity="critical",
            received_at="2025-01-15T10:01:00+00:00",
        )
        profile.add_external_signal(sig)
        assert len(profile.external.external_signals) == 1

    def test_to_dict_only_includes_non_none_sections(self):
        profile = _make_profile(
            workflow=_make_workflow(),
        )
        d = profile.to_dict()
        assert "workflow" in d
        assert "situational" not in d
        assert "relational" not in d
        assert "profile_id" in d
        assert "agent_id" in d

    def test_from_dict_round_trip(self):
        profile = _make_profile(
            workflow=_make_workflow(),
            situational=_make_situational(),
            meta=MetaContext(evaluation_count=3, denial_count=1),
        )
        d = profile.to_dict()
        restored = ContextProfile.from_dict(d)
        assert restored.profile_id == profile.profile_id
        assert restored.agent_id == profile.agent_id
        assert restored.workflow is not None
        assert restored.workflow.workflow_id == "wf-001"
        assert restored.situational is not None
        assert restored.meta is not None
        assert restored.meta.evaluation_count == 3
        assert restored.relational is None

    def test_summary_returns_compact_representation(self):
        profile = _make_profile(
            workflow=_make_workflow(),
        )
        s = profile.summary()
        assert "profile_id" in s
        assert "agent_id" in s
        assert "completeness" in s
        assert "active" in s
        assert "workflow_status" in s
        assert "workflow_step" in s

    def test_summary_includes_risk_count(self):
        profile = _make_profile(
            situational=_make_situational(origin="agent_initiated"),
        )
        s = profile.summary()
        assert "risk_signal_count" in s
        assert s["risk_signal_count"] > 0


# ═══════════════════════════════════════════════════════════════════════
# ContextProfileManager (~15 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestContextProfileManager:
    """Test the ContextProfileManager."""

    def test_create_profile_generates_unique_id(self):
        mgr = ContextProfileManager()
        p1 = mgr.create_profile("agent-1", "workflow")
        p2 = mgr.create_profile("agent-1", "workflow")
        assert p1.profile_id != p2.profile_id
        assert p1.profile_id.startswith("cp-")
        assert p2.profile_id.startswith("cp-")

    def test_get_profile_retrieves_by_id(self):
        mgr = ContextProfileManager()
        created = mgr.create_profile("agent-1", "workflow")
        retrieved = mgr.get_profile(created.profile_id)
        assert retrieved is not None
        assert retrieved.profile_id == created.profile_id
        assert retrieved.agent_id == "agent-1"

    def test_get_profile_returns_none_for_unknown(self):
        mgr = ContextProfileManager()
        assert mgr.get_profile("cp-nonexistent") is None

    def test_get_active_profile_returns_most_recent(self):
        mgr = ContextProfileManager()
        p1 = mgr.create_profile("agent-1", "workflow")
        time.sleep(0.01)  # Ensure different created_at
        p2 = mgr.create_profile("agent-1", "workflow")
        active = mgr.get_active_profile("agent-1")
        assert active is not None
        assert active.profile_id == p2.profile_id

    def test_update_profile_modifies_specific_context(self):
        mgr = ContextProfileManager()
        profile = mgr.create_profile("agent-1", "workflow")
        updated = mgr.update_profile(
            profile.profile_id,
            workflow=_make_workflow(),
        )
        assert updated is not None
        assert updated.workflow is not None
        assert updated.workflow.workflow_id == "wf-001"

    def test_update_profile_doesnt_affect_other_contexts(self):
        mgr = ContextProfileManager()
        profile = mgr.create_profile(
            "agent-1", "workflow",
            situational=_make_situational(),
        )
        mgr.update_profile(
            profile.profile_id,
            workflow=_make_workflow(),
        )
        retrieved = mgr.get_profile(profile.profile_id)
        assert retrieved.workflow is not None
        assert retrieved.situational is not None
        assert retrieved.situational.origin == "user_request"

    def test_close_profile_marks_as_complete(self):
        mgr = ContextProfileManager()
        profile = mgr.create_profile("agent-1", "workflow")
        assert profile.active is True
        result = mgr.close_profile(profile.profile_id)
        assert result is True
        retrieved = mgr.get_profile(profile.profile_id)
        assert retrieved.active is False

    def test_get_active_profile_skips_closed(self):
        mgr = ContextProfileManager()
        p1 = mgr.create_profile("agent-1", "workflow")
        time.sleep(0.01)
        p2 = mgr.create_profile("agent-1", "workflow")
        mgr.close_profile(p2.profile_id)
        active = mgr.get_active_profile("agent-1")
        assert active is not None
        assert active.profile_id == p1.profile_id

    def test_list_profiles_filters_by_agent_id(self):
        mgr = ContextProfileManager()
        mgr.create_profile("agent-1", "workflow")
        mgr.create_profile("agent-2", "workflow")
        mgr.create_profile("agent-1", "session")
        results = mgr.list_profiles(agent_id="agent-1")
        assert len(results) == 2
        assert all(p.agent_id == "agent-1" for p in results)

    def test_list_profiles_filters_by_profile_type(self):
        mgr = ContextProfileManager()
        mgr.create_profile("agent-1", "workflow")
        mgr.create_profile("agent-1", "session")
        mgr.create_profile("agent-2", "workflow")
        results = mgr.list_profiles(profile_type="session")
        assert len(results) == 1
        assert results[0].profile_type == "session"

    def test_list_profiles_active_only(self):
        mgr = ContextProfileManager()
        p1 = mgr.create_profile("agent-1", "workflow")
        p2 = mgr.create_profile("agent-1", "workflow")
        mgr.close_profile(p1.profile_id)
        active = mgr.list_profiles(active_only=True)
        assert len(active) == 1
        assert active[0].profile_id == p2.profile_id
        all_profiles = mgr.list_profiles(active_only=False)
        assert len(all_profiles) == 2

    def test_max_profiles_evicts_oldest_closed_first(self):
        mgr = ContextProfileManager(max_profiles=3)
        p1 = mgr.create_profile("agent-1", "workflow")
        mgr.close_profile(p1.profile_id)
        time.sleep(0.01)
        p2 = mgr.create_profile("agent-1", "workflow")
        mgr.close_profile(p2.profile_id)
        time.sleep(0.01)
        p3 = mgr.create_profile("agent-1", "workflow")  # active
        # Now at capacity (3). Creating a 4th should evict oldest closed (p1)
        p4 = mgr.create_profile("agent-1", "workflow")
        assert mgr.get_profile(p1.profile_id) is None  # evicted
        assert mgr.get_profile(p2.profile_id) is not None  # still there
        assert mgr.get_profile(p3.profile_id) is not None
        assert mgr.get_profile(p4.profile_id) is not None

    def test_enrich_from_runtime(self):
        """enrich_from_runtime populates historical context from mock runtime."""

        class MockTrajectory:
            trend = "rising"
            def summary(self):
                return {"narrative": "test"}

        class MockRuntime:
            audit_trail = None

            def get_trust_profile(self, agent_id):
                from nomotic.types import TrustProfile
                return TrustProfile(agent_id=agent_id, overall_trust=0.7)

            def get_trust_trajectory(self, agent_id):
                return MockTrajectory()

            def get_drift(self, agent_id):
                return None

        mgr = ContextProfileManager()
        profile = mgr.create_profile("agent-1", "workflow")
        result = mgr.enrich_from_runtime(profile.profile_id, MockRuntime())
        assert result is True
        enriched = mgr.get_profile(profile.profile_id)
        assert enriched.historical is not None
        assert enriched.historical.trust_current == 0.7
        assert enriched.historical.trust_direction == "rising"
        assert enriched.historical.behavioral_drift_status == "normal"

    def test_enrich_from_audit(self):
        """enrich_from_audit populates meta context from mock audit trail."""

        class MockRecord:
            def __init__(self, verdict, action_type, action_target, ucs, context_code):
                self.verdict = verdict
                self.action_type = action_type
                self.action_target = action_target
                self.ucs = ucs
                self.context_code = context_code

        class MockAuditTrail:
            def query(self, agent_id=None, limit=None):
                return [
                    MockRecord("ALLOW", "data.read", "orders", 0.85, "GOV.ALLOW"),
                    MockRecord("DENY", "data.write", "secrets", 0.15, "GOV.DENY"),
                    MockRecord("ESCALATE", "admin.config", "settings", 0.4, "GOV.ESCALATE"),
                    MockRecord("ALLOW", "data.read", "users", 0.9, "GOV.REVISE_NEEDED"),
                ]

        mgr = ContextProfileManager()
        profile = mgr.create_profile("agent-1", "workflow")
        result = mgr.enrich_from_audit(profile.profile_id, MockAuditTrail())
        assert result is True
        enriched = mgr.get_profile(profile.profile_id)
        assert enriched.meta is not None
        assert enriched.meta.evaluation_count == 4
        assert enriched.meta.denial_count == 1
        assert enriched.meta.escalation_count == 1
        assert enriched.meta.revise_count == 1

    def test_thread_safety_concurrent_operations(self):
        """10 threads each doing 50 operations should not corrupt state."""
        mgr = ContextProfileManager()
        errors: list[str] = []
        created_ids: list[str] = []
        lock = threading.Lock()

        def worker(thread_id: int):
            try:
                for i in range(50):
                    op = random.choice(["create", "get", "update", "list"])
                    if op == "create":
                        p = mgr.create_profile(f"agent-{thread_id}", "workflow")
                        with lock:
                            created_ids.append(p.profile_id)
                    elif op == "get":
                        with lock:
                            if created_ids:
                                pid = random.choice(created_ids)
                            else:
                                pid = "cp-nonexistent"
                        mgr.get_profile(pid)
                    elif op == "update":
                        with lock:
                            if created_ids:
                                pid = random.choice(created_ids)
                            else:
                                pid = "cp-nonexistent"
                        mgr.update_profile(pid, meta=MetaContext())
                    elif op == "list":
                        mgr.list_profiles(agent_id=f"agent-{thread_id}")
            except Exception as e:
                with lock:
                    errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Thread errors: {errors}"
        # All created profiles should be retrievable
        for pid in created_ids:
            assert mgr.get_profile(pid) is not None


# ═══════════════════════════════════════════════════════════════════════
# Serialization (~5 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Test full serialization round-trips."""

    def test_full_profile_round_trip(self):
        profile = _make_profile(
            workflow=_make_workflow(),
            situational=_make_situational(),
            meta=MetaContext(evaluation_count=5),
            feedback=FeedbackContext(satisfaction_signals=["great"]),
        )
        d = profile.to_dict()
        restored = ContextProfile.from_dict(d)
        # Compare serialized forms
        assert restored.to_dict() == d

    def test_dependency_serialization_all_types(self):
        dep_types = ["requires", "constrains", "enables", "informs"]
        for dt in dep_types:
            dep = Dependency(
                from_step=1, to_step=2,
                dependency_type=dt,
                description=f"Dependency of type {dt}",
            )
            d = dep.to_dict()
            restored = Dependency.from_dict(d)
            assert restored.dependency_type == dt
            assert restored.to_dict() == d

    def test_compound_method_serialization(self):
        cm = CompoundMethod(
            method="payment.process",
            agents_involved=["agent-1", "agent-2", "agent-3"],
            combined_scope_description="Combined payment processing authority",
        )
        d = cm.to_dict()
        restored = CompoundMethod.from_dict(d)
        assert restored.method == "payment.process"
        assert len(restored.agents_involved) == 3
        assert restored.to_dict() == d

    def test_external_signal_serialization(self):
        sig = ExternalSignal(
            source="market_data",
            signal_type="price_change",
            description="Stock dropped 10%",
            severity="alert",
            received_at="2025-01-15T10:01:00+00:00",
        )
        d = sig.to_dict()
        restored = ExternalSignal.from_dict(d)
        assert restored.source == "market_data"
        assert restored.severity == "alert"
        assert restored.to_dict() == d

    def test_profile_with_all_ten_context_types_round_trip(self):
        profile = ContextProfile(
            profile_id="cp-full",
            agent_id="agent-full",
            profile_type="workflow",
            created_at="2025-01-15T10:00:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
            workflow=WorkflowContext(
                workflow_id="wf-full",
                workflow_type="travel_booking",
                total_steps=3,
                current_step=2,
                steps_completed=[
                    CompletedStep(
                        step_id="s1", step_number=1, method="data.read",
                        target="flights", verdict="ALLOW", ucs=0.9,
                        timestamp="2025-01-15T10:05:00+00:00",
                        output_summary="Found 5 flights",
                    ),
                ],
                steps_remaining=[
                    PlannedStep(
                        step_number=2, method="payment.charge",
                        target="credit_card", description="Charge card",
                        estimated_risk="medium",
                    ),
                ],
                dependencies=[
                    Dependency(
                        from_step=1, to_step=2,
                        dependency_type="requires",
                        description="Must find flights first",
                    ),
                ],
                rollback_points=["s1"],
                started_at="2025-01-15T10:00:00+00:00",
                status="active",
            ),
            situational=SituationalContext(
                origin="user_request",
                trigger_description="User wants to book a flight",
                operational_mode="normal",
                urgency="routine",
                origin_id="user-hash-123",
                initiating_entity="user-42",
            ),
            relational=RelationalContext(
                parent_agent_id="orchestrator",
                child_agent_ids=["booking-agent"],
                delegation_chain=[
                    DelegationLink(
                        from_id="orchestrator", to_id="booking-agent",
                        delegated_methods=["data.read", "payment.charge"],
                        delegated_at="2025-01-15T10:00:00+00:00",
                    ),
                ],
                shared_workflow_agents=["notification-agent"],
                compound_methods=[
                    CompoundMethod(
                        method="payment.charge",
                        agents_involved=["booking-agent", "payment-agent"],
                        combined_scope_description="Coordinated payment",
                    ),
                ],
            ),
            temporal=TemporalContext(
                current_time="2025-01-15T10:15:00+00:00",
                time_of_day_category="business_hours",
                operational_state="normal",
                recent_events=[
                    TemporalEvent(
                        event_type="system_update",
                        description="Scheduled maintenance completed",
                        occurred_at="2025-01-15T09:00:00+00:00",
                        relevance="System is fresh from maintenance",
                    ),
                ],
                time_pressure="moderate",
                deadline="2025-01-15T18:00:00+00:00",
            ),
            historical=HistoricalContext(
                trust_current=0.82,
                trust_direction="rising",
                trust_trajectory_summary="Consistently improving over 7 days",
                recent_verdicts=[
                    RecentVerdict(
                        method="data.read", target="flights",
                        verdict="ALLOW", ucs=0.9,
                        timestamp="2025-01-15T10:05:00+00:00",
                    ),
                ],
                scope_changes_recent=["added payment.charge"],
                reasoning_quality_trend="improving",
                behavioral_drift_status="low_drift",
                days_since_activation=14,
            ),
            input_context=InputContext(
                input_type="text",
                input_summary="User requests round-trip flight NYC-LAX",
                input_category="standard_request",
                entities_referenced=["NYC", "LAX", "round-trip"],
                constraints_expressed=["budget_under_500", "direct_flight"],
                intent_classification="travel_booking",
                input_hash="sha256:abc123def456",
            ),
            output=OutputContext(
                outputs_produced=[
                    OutputRecord(
                        step_number=1, method="data.read", target="flights",
                        output_type="report", output_summary="5 flights found",
                        reversible=True,
                        timestamp="2025-01-15T10:05:00+00:00",
                    ),
                ],
                cumulative_impact="minimal",
                external_effects=[],
                reversible=True,
            ),
            external=ExternalContext(
                external_signals=[
                    ExternalSignal(
                        source="market_data", signal_type="price_update",
                        description="Flight prices stable",
                        severity="info",
                        received_at="2025-01-15T10:10:00+00:00",
                    ),
                ],
                data_freshness={"flights": "2025-01-15T10:10:00+00:00"},
                environment_status="normal",
                active_alerts=[],
            ),
            meta=MetaContext(
                evaluation_count=2,
                revise_count=0,
                escalation_count=0,
                denial_count=0,
                governance_load="normal",
                previous_artifact_ids=["art-001"],
                resubmission=False,
            ),
            feedback=FeedbackContext(
                feedback_received=[
                    FeedbackRecord(
                        source="user", feedback_type="positive",
                        description="These options look great",
                        timestamp="2025-01-15T10:06:00+00:00",
                    ),
                ],
                override_history=[],
                downstream_outcomes=[
                    OutcomeRecord(
                        step_number=1, method="data.read",
                        expected_outcome="Find available flights",
                        actual_outcome="Found 5 matching flights",
                        outcome_quality="as_expected",
                        timestamp="2025-01-15T10:05:30+00:00",
                    ),
                ],
                satisfaction_signals=["positive_feedback"],
            ),
        )

        d = profile.to_dict()
        restored = ContextProfile.from_dict(d)
        assert restored.to_dict() == d

        # Verify a few deep fields
        assert restored.workflow.steps_completed[0].output_summary == "Found 5 flights"
        assert restored.relational.compound_methods[0].method == "payment.charge"
        assert restored.temporal.recent_events[0].event_type == "system_update"
        assert restored.historical.days_since_activation == 14
        assert restored.feedback.downstream_outcomes[0].outcome_quality == "as_expected"


# ═══════════════════════════════════════════════════════════════════════
# Integration (~5 tests)
# ═══════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests with AgentContext, GovernanceRuntime, and the API."""

    def test_agent_context_accepts_context_profile_id(self):
        from nomotic.types import AgentContext, TrustProfile
        ctx = AgentContext(
            agent_id="a1",
            trust_profile=TrustProfile(agent_id="a1"),
            context_profile_id="cp-123",
        )
        assert ctx.context_profile_id == "cp-123"

    def test_agent_context_backward_compatible_without_context_profile_id(self):
        from nomotic.types import AgentContext, TrustProfile
        ctx = AgentContext(
            agent_id="a1",
            trust_profile=TrustProfile(agent_id="a1"),
        )
        assert ctx.context_profile_id is None

    def test_runtime_creates_and_manages_profiles(self):
        from nomotic.runtime import GovernanceRuntime
        runtime = GovernanceRuntime()
        assert runtime.context_profiles is not None
        profile = runtime.context_profiles.create_profile("agent-1", "workflow")
        assert profile.profile_id.startswith("cp-")
        retrieved = runtime.context_profiles.get_profile(profile.profile_id)
        assert retrieved is not None
        assert retrieved.agent_id == "agent-1"

    def test_runtime_convenience_methods(self):
        from nomotic.runtime import GovernanceRuntime
        runtime = GovernanceRuntime()
        profile = runtime.create_context_profile("agent-1", profile_type="session")
        assert profile.profile_type == "session"
        retrieved = runtime.get_context_profile(profile.profile_id)
        assert retrieved is not None
        assert retrieved.profile_id == profile.profile_id

    def test_api_endpoint_integration(self):
        """Test context profile API endpoints: create, get, update, summary, risks."""
        import socket
        from nomotic.api import NomoticAPIServer
        from nomotic.authority import CertificateAuthority
        from nomotic.keys import SigningKey
        from nomotic.runtime import GovernanceRuntime
        from nomotic.store import MemoryCertificateStore

        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        sk, _vk = SigningKey.generate()
        ca = CertificateAuthority(
            issuer_id="test", signing_key=sk,
            store=MemoryCertificateStore(),
        )
        runtime = GovernanceRuntime()

        server = NomoticAPIServer(
            ca, runtime=runtime,
            host="127.0.0.1", port=port,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.3)

        base = f"http://127.0.0.1:{port}"

        def _post(path, data=None):
            payload = json.dumps(data or {}).encode("utf-8")
            req = urllib.request.Request(
                f"{base}{path}", data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        def _get(path):
            req = urllib.request.Request(f"{base}{path}", method="GET")
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        def _patch(path, data=None):
            payload = json.dumps(data or {}).encode("utf-8")
            req = urllib.request.Request(
                f"{base}{path}", data=payload, method="PATCH",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        try:
            # Create a context profile
            status, body = _post("/v1/context", {
                "agent_id": "api-agent",
                "profile_type": "workflow",
                "workflow": {
                    "workflow_id": "wf-api",
                    "workflow_type": "test",
                    "total_steps": 3,
                    "current_step": 1,
                },
                "situational": {
                    "origin": "user_request",
                    "trigger_description": "API test",
                },
            })
            assert status == 201
            profile_id = body["profile_id"]
            assert profile_id.startswith("cp-")

            # Get the profile
            status, body = _get(f"/v1/context/{profile_id}")
            assert status == 200
            assert body["agent_id"] == "api-agent"
            assert body["workflow"]["workflow_id"] == "wf-api"

            # Update the profile
            status, body = _patch(f"/v1/context/{profile_id}", {
                "meta": {
                    "evaluation_count": 3,
                    "denial_count": 1,
                },
            })
            assert status == 200
            assert body["meta"]["evaluation_count"] == 3

            # Get summary
            status, body = _get(f"/v1/context/{profile_id}/summary")
            assert status == 200
            assert "completeness" in body
            assert "risk_signal_count" in body

            # Get risks
            status, body = _get(f"/v1/context/{profile_id}/risks")
            assert status == 200
            assert "risk_signals" in body
            assert "count" in body
        finally:
            server.shutdown()
