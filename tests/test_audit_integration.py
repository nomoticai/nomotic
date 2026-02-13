"""Integration tests — audit trail wired into the governance runtime."""

import time

import pytest

from nomotic.runtime import GovernanceRuntime, RuntimeConfig
from nomotic.types import (
    Action,
    AgentContext,
    TrustProfile,
    UserContext,
    Verdict,
)


def _make_runtime(**kwargs) -> GovernanceRuntime:
    """Create a runtime with audit enabled and fingerprints enabled."""
    from nomotic.monitor import DriftConfig
    drift_cfg = DriftConfig(window_size=50, check_interval=10, min_observations=10)
    config = RuntimeConfig(
        enable_audit=True,
        enable_fingerprints=True,
        drift_config=drift_cfg,
        **kwargs,
    )
    return GovernanceRuntime(config=config)


def _make_action(agent_id: str = "agent-1", action_type: str = "read", target: str = "/api/data") -> Action:
    return Action(agent_id=agent_id, action_type=action_type, target=target)


def _make_context(agent_id: str = "agent-1", user_context: UserContext | None = None) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        trust_profile=TrustProfile(agent_id=agent_id),
        user_context=user_context,
    )


class TestRuntimeCreatesAuditTrail:
    """Runtime creates audit subsystems when enabled."""

    def test_audit_trail_created(self):
        runtime = _make_runtime()
        assert runtime.audit_trail is not None

    def test_provenance_log_created(self):
        runtime = _make_runtime()
        assert runtime.provenance_log is not None

    def test_owner_activity_created(self):
        runtime = _make_runtime()
        assert runtime.owner_activity is not None

    def test_user_tracker_created(self):
        runtime = _make_runtime()
        assert runtime.user_tracker is not None

    def test_audit_disabled(self):
        runtime = GovernanceRuntime(config=RuntimeConfig(enable_audit=False))
        assert runtime.audit_trail is None
        assert runtime.provenance_log is None
        assert runtime.owner_activity is None
        assert runtime.user_tracker is None


class TestEvaluateProducesAuditRecord:
    """evaluate() produces audit records."""

    def test_allow_produces_record(self):
        runtime = _make_runtime()
        action = _make_action()
        context = _make_context()
        verdict = runtime.evaluate(action, context)
        assert verdict.verdict == Verdict.ALLOW
        records = runtime.audit_trail.query(agent_id="agent-1")
        assert len(records) >= 1
        record = records[0]
        assert record.agent_id == "agent-1"
        assert record.action_type == "read"
        assert record.action_target == "/api/data"

    def test_correct_context_code_for_allow(self):
        runtime = _make_runtime()
        action = _make_action()
        context = _make_context()
        runtime.evaluate(action, context)
        records = runtime.audit_trail.query(agent_id="agent-1")
        assert records[0].context_code == "GOVERNANCE.ALLOW"
        assert records[0].severity == "info"

    def test_correct_context_code_for_deny_with_veto(self):
        runtime = _make_runtime()
        # Configure scope to only allow "read" — then request "delete"
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"read"})
        action = _make_action(action_type="delete")
        context = _make_context()
        verdict = runtime.evaluate(action, context)
        assert verdict.verdict == Verdict.DENY
        records = runtime.audit_trail.query(agent_id="agent-1", verdict="DENY")
        assert len(records) >= 1
        record = records[0]
        assert record.context_code == "GOVERNANCE.VETO"
        assert record.severity == "alert"

    def test_dimension_scores_snapshot(self):
        runtime = _make_runtime()
        action = _make_action()
        context = _make_context()
        runtime.evaluate(action, context)
        records = runtime.audit_trail.query(agent_id="agent-1")
        record = records[0]
        assert len(record.dimension_scores) > 0
        # Each dimension score has the expected shape
        for ds in record.dimension_scores:
            assert "name" in ds
            assert "score" in ds
            assert "weight" in ds
            assert "veto" in ds

    def test_trust_state_captured(self):
        runtime = _make_runtime()
        action = _make_action()
        context = _make_context()
        runtime.evaluate(action, context)
        records = runtime.audit_trail.query(agent_id="agent-1")
        record = records[0]
        assert 0.0 <= record.trust_score <= 1.0
        assert record.trust_trend in ("rising", "falling", "stable", "volatile")

    def test_justification_present(self):
        runtime = _make_runtime()
        action = _make_action()
        context = _make_context()
        runtime.evaluate(action, context)
        records = runtime.audit_trail.query(agent_id="agent-1")
        record = records[0]
        assert len(record.justification) > 0
        assert "ALLOW" in record.justification or "DENY" in record.justification

    def test_user_context_captured(self):
        runtime = _make_runtime()
        action = _make_action()
        user_ctx = UserContext(user_id="user-123", request_hash="abc123")
        context = _make_context(user_context=user_ctx)
        runtime.evaluate(action, context)
        records = runtime.audit_trail.query(agent_id="agent-1")
        record = records[0]
        assert record.user_id == "user-123"
        assert record.metadata.get("user_request_hash") == "abc123"

    def test_owner_id_from_certificate(self):
        runtime = _make_runtime()
        cert = runtime.birth(
            "agent-cert",
            archetype="customer-experience",
            organization="test-org",
            zone_path="global",
            owner="owner@test.com",
        )
        action = _make_action(agent_id="agent-cert")
        context = _make_context(agent_id="agent-cert")
        runtime.evaluate(action, context)
        records = runtime.audit_trail.query(agent_id="agent-cert")
        assert records[0].owner_id == "owner@test.com"


class TestUserTracking:
    """User activity tracked when user_context provided."""

    def test_user_activity_tracked(self):
        runtime = _make_runtime()
        action = _make_action()
        user_ctx = UserContext(user_id="user-1")
        context = _make_context(user_context=user_ctx)
        runtime.evaluate(action, context)
        stats = runtime.user_tracker.get_stats("user-1")
        assert stats is not None
        assert stats.total_interactions == 1

    def test_no_tracking_without_user_context(self):
        runtime = _make_runtime()
        action = _make_action()
        context = _make_context()
        runtime.evaluate(action, context)
        # No user ID means no user tracking
        stats = runtime.user_tracker.get_stats("")
        assert stats is None


class TestProvenanceIntegration:
    """Provenance tracking for configuration changes."""

    def test_configure_scope_produces_provenance(self):
        runtime = _make_runtime()
        runtime.configure_scope(
            "agent-1", {"read", "write"},
            actor="admin@test.com",
            reason="Setting initial scope",
        )
        records = runtime.provenance_log.query(target_type="scope")
        assert len(records) == 1
        assert records[0].actor == "admin@test.com"
        assert records[0].reason == "Setting initial scope"
        assert "read" in records[0].new_value
        assert "write" in records[0].new_value

    def test_configure_scope_modify(self):
        runtime = _make_runtime()
        runtime.configure_scope("agent-1", {"read"})
        runtime.configure_scope("agent-1", {"read", "write"})
        records = runtime.provenance_log.query(target_type="scope", target_id="agent-1")
        assert len(records) == 2
        assert records[0].change_type == "modify"  # newest first

    def test_configure_boundaries_produces_provenance(self):
        runtime = _make_runtime()
        runtime.configure_boundaries(
            "agent-1", {"/api/data", "/api/users"},
            actor="ops@test.com",
        )
        records = runtime.provenance_log.query(target_type="boundary")
        assert len(records) == 1

    def test_configure_time_window_produces_provenance(self):
        runtime = _make_runtime()
        runtime.configure_time_window("delete", 9, 17, actor="admin@test.com")
        records = runtime.provenance_log.query(target_type="time_window")
        assert len(records) == 1

    def test_configure_human_override_produces_provenance(self):
        runtime = _make_runtime()
        runtime.configure_human_override("delete", "purge", actor="admin@test.com")
        records = runtime.provenance_log.query(target_type="override")
        assert len(records) == 1

    def test_add_ethical_rule_produces_provenance(self):
        runtime = _make_runtime()

        def no_harm(action, context):
            return True, ""

        runtime.add_ethical_rule(no_harm, actor="admin@test.com", rule_name="no_harm")
        records = runtime.provenance_log.query(target_type="rule")
        assert len(records) == 1
        assert records[0].target_id == "no_harm"

    def test_set_dimension_weight_produces_provenance(self):
        runtime = _make_runtime()
        runtime.set_dimension_weight("scope_compliance", 2.0, actor="admin@test.com")
        records = runtime.provenance_log.query(target_type="weight")
        assert len(records) == 1
        assert records[0].previous_value == 1.5  # original weight
        assert records[0].new_value == 2.0

    def test_set_dimension_weight_unknown_raises(self):
        runtime = _make_runtime()
        with pytest.raises(ValueError, match="Unknown dimension"):
            runtime.set_dimension_weight("nonexistent", 1.0)

    def test_config_version_in_audit_records(self):
        runtime = _make_runtime()
        runtime.configure_scope("agent-1", {"read"}, actor="admin")
        action = _make_action()
        context = _make_context()
        runtime.evaluate(action, context)
        records = runtime.audit_trail.query(agent_id="agent-1")
        record = records[0]
        assert record.metadata.get("config_version")
        assert record.metadata["config_version"] != "0" * 12


class TestMultipleAgents:
    """Multiple agents produce independent audit records."""

    def test_independent_audit_records(self):
        runtime = _make_runtime()
        for agent_id in ("agent-a", "agent-b", "agent-c"):
            action = _make_action(agent_id=agent_id)
            context = _make_context(agent_id=agent_id)
            runtime.evaluate(action, context)
        for agent_id in ("agent-a", "agent-b", "agent-c"):
            records = runtime.audit_trail.query(agent_id=agent_id)
            assert len(records) >= 1
            assert all(r.agent_id == agent_id for r in records)


class TestDriftInAuditRecords:
    """Drift state appears in audit records when available."""

    def test_drift_captured_after_drift_builds(self):
        runtime = _make_runtime()
        # Build baseline
        for _ in range(60):
            action = _make_action(action_type="read")
            context = _make_context()
            runtime.evaluate(action, context)
        # Check if drift state is captured
        drift = runtime.get_drift("agent-1")
        if drift is not None:
            # The last records should have drift info
            records = runtime.audit_trail.query(agent_id="agent-1", limit=1)
            record = records[0]
            assert record.drift_overall is not None
            assert record.drift_severity is not None


class TestAuditSummaryIntegration:
    """Audit summary works with runtime-generated records."""

    def test_summary_after_evaluations(self):
        runtime = _make_runtime()
        for _ in range(5):
            action = _make_action()
            context = _make_context()
            runtime.evaluate(action, context)
        summary = runtime.audit_trail.summary()
        assert summary["total_records"] >= 5


class TestExistingBehaviorPreserved:
    """Existing runtime behavior still works with audit enabled."""

    def test_evaluate_returns_verdict(self):
        runtime = _make_runtime()
        action = _make_action()
        context = _make_context()
        verdict = runtime.evaluate(action, context)
        assert verdict.verdict in (Verdict.ALLOW, Verdict.DENY, Verdict.ESCALATE)

    def test_trust_still_calibrated(self):
        runtime = _make_runtime()
        action = _make_action()
        context = _make_context()
        runtime.evaluate(action, context)
        profile = runtime.get_trust_profile("agent-1")
        # Trust should have been updated
        assert profile.successful_actions > 0 or profile.violation_count > 0

    def test_fingerprints_still_work(self):
        runtime = _make_runtime()
        for _ in range(5):
            action = _make_action()
            context = _make_context()
            runtime.evaluate(action, context)
        fp = runtime.get_fingerprint("agent-1")
        assert fp is not None
        assert fp.total_observations == 5

    def test_audit_disabled_no_overhead(self):
        runtime = GovernanceRuntime(config=RuntimeConfig(enable_audit=False))
        action = _make_action()
        context = _make_context()
        verdict = runtime.evaluate(action, context)
        assert verdict.verdict in (Verdict.ALLOW, Verdict.DENY, Verdict.ESCALATE)
        assert runtime.audit_trail is None
