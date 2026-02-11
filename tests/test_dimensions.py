"""Tests for the 13 governance dimensions."""

from nomotic.types import Action, ActionRecord, AgentContext, GovernanceVerdict, TrustProfile, Verdict
from nomotic.dimensions import (
    AuthorityVerification,
    BehavioralConsistency,
    CascadingImpact,
    DimensionRegistry,
    EthicalAlignment,
    HumanOverride,
    IncidentDetection,
    IsolationIntegrity,
    PrecedentAlignment,
    ResourceBoundaries,
    ResourceLimits,
    ScopeCompliance,
    StakeholderImpact,
    TemporalCompliance,
    Transparency,
)


def _ctx(agent_id: str = "agent-1", trust: float = 0.5, history=None) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        trust_profile=TrustProfile(agent_id=agent_id, overall_trust=trust),
        action_history=history or [],
    )


def _action(action_type: str = "read", target: str = "db", **params) -> Action:
    return Action(agent_id="agent-1", action_type=action_type, target=target, parameters=params)


# --- Scope Compliance ---


class TestScopeCompliance:
    def test_in_scope_allows(self):
        dim = ScopeCompliance()
        dim.configure_agent_scope("agent-1", {"read", "write"})
        score = dim.evaluate(_action("read"), _ctx())
        assert score.score == 1.0
        assert not score.veto

    def test_out_of_scope_vetoes(self):
        dim = ScopeCompliance()
        dim.configure_agent_scope("agent-1", {"read"})
        score = dim.evaluate(_action("delete"), _ctx())
        assert score.score == 0.0
        assert score.veto

    def test_wildcard_allows_all(self):
        dim = ScopeCompliance()
        dim.configure_agent_scope("agent-1", {"*"})
        score = dim.evaluate(_action("anything"), _ctx())
        assert score.score == 1.0

    def test_no_scope_defined_caution(self):
        dim = ScopeCompliance()
        score = dim.evaluate(_action("read"), _ctx())
        assert score.score == 0.5


# --- Authority Verification ---


class TestAuthorityVerification:
    def test_no_checks_moderate_trust(self):
        dim = AuthorityVerification()
        score = dim.evaluate(_action(), _ctx())
        assert score.score == 0.7

    def test_passing_check(self):
        dim = AuthorityVerification()
        dim.add_authority_check(lambda a, c: True)
        score = dim.evaluate(_action(), _ctx())
        assert score.score == 1.0

    def test_failing_check_vetoes(self):
        dim = AuthorityVerification()
        dim.add_authority_check(lambda a, c: False)
        score = dim.evaluate(_action(), _ctx())
        assert score.veto


# --- Resource Boundaries ---


class TestResourceBoundaries:
    def test_within_limits(self):
        dim = ResourceBoundaries(limits=ResourceLimits(max_actions_per_minute=100))
        score = dim.evaluate(_action(), _ctx())
        assert score.score > 0.5
        assert not score.veto

    def test_cost_exceeded_vetoes(self):
        dim = ResourceBoundaries(limits=ResourceLimits(max_cost_per_action=10.0))
        score = dim.evaluate(_action(cost=50.0), _ctx())
        assert score.veto


# --- Behavioral Consistency ---


class TestBehavioralConsistency:
    def test_first_action_establishes_baseline(self):
        dim = BehavioralConsistency()
        score = dim.evaluate(_action("read"), _ctx())
        assert score.score == 0.7

    def test_consistent_action_high_score(self):
        dim = BehavioralConsistency()
        ctx = _ctx()
        dim.evaluate(_action("read"), ctx)
        score = dim.evaluate(_action("read"), ctx)
        assert score.score == 1.0

    def test_novel_action_lower_score(self):
        dim = BehavioralConsistency()
        ctx = _ctx()
        dim.evaluate(_action("read"), ctx)
        score = dim.evaluate(_action("delete"), ctx)
        assert score.score == 0.5


# --- Cascading Impact ---


class TestCascadingImpact:
    def test_high_impact_low_score(self):
        dim = CascadingImpact()
        score = dim.evaluate(_action("delete_all"), _ctx())
        assert score.score <= 0.3

    def test_medium_impact(self):
        dim = CascadingImpact()
        score = dim.evaluate(_action("update_record"), _ctx())
        assert 0.3 < score.score < 0.9

    def test_low_impact(self):
        dim = CascadingImpact()
        score = dim.evaluate(_action("read"), _ctx())
        assert score.score >= 0.9


# --- Stakeholder Impact ---


class TestStakeholderImpact:
    def test_sensitive_target_low_score(self):
        dim = StakeholderImpact()
        dim.mark_sensitive("production_db")
        score = dim.evaluate(_action(target="production_db"), _ctx())
        assert score.score == 0.2

    def test_external_facing_medium_score(self):
        dim = StakeholderImpact()
        score = dim.evaluate(_action(target="customer_records"), _ctx())
        assert score.score == 0.4

    def test_internal_high_score(self):
        dim = StakeholderImpact()
        score = dim.evaluate(_action(target="internal_cache"), _ctx())
        assert score.score == 0.9


# --- Incident Detection ---


class TestIncidentDetection:
    def test_no_patterns_passes(self):
        dim = IncidentDetection()
        score = dim.evaluate(_action(), _ctx())
        assert score.score == 1.0

    def test_repetitive_pattern_detected(self):
        dim = IncidentDetection()
        history = []
        for _ in range(5):
            history.append(
                ActionRecord(
                    action=_action("spam"),
                    verdict=GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8),
                )
            )
        ctx = _ctx(history=history)
        score = dim.evaluate(_action("spam"), ctx)
        assert score.score < 0.5

    def test_custom_pattern(self):
        dim = IncidentDetection()
        dim.add_pattern(lambda a, c: 0.05 if "exploit" in a.action_type else None)
        score = dim.evaluate(_action("exploit_vuln"), _ctx())
        assert score.score <= 0.05
        assert score.veto


# --- Isolation Integrity ---


class TestIsolationIntegrity:
    def test_within_boundary(self):
        dim = IsolationIntegrity()
        dim.set_boundaries("agent-1", {"db_a", "db_b"})
        score = dim.evaluate(_action(target="db_a"), _ctx())
        assert score.score == 1.0

    def test_outside_boundary_vetoes(self):
        dim = IsolationIntegrity()
        dim.set_boundaries("agent-1", {"db_a"})
        score = dim.evaluate(_action(target="db_secret"), _ctx())
        assert score.veto

    def test_no_boundaries_caution(self):
        dim = IsolationIntegrity()
        score = dim.evaluate(_action(), _ctx())
        assert score.score == 0.6


# --- Temporal Compliance ---


class TestTemporalCompliance:
    def test_no_constraints_passes(self):
        dim = TemporalCompliance()
        score = dim.evaluate(_action(), _ctx())
        assert score.score == 1.0

    def test_min_interval_violation(self):
        dim = TemporalCompliance()
        dim.set_min_interval("deploy", 3600)
        ctx = _ctx()
        # First call sets the timestamp
        dim.evaluate(_action("deploy"), ctx)
        # Second call within interval should veto
        score = dim.evaluate(_action("deploy"), ctx)
        assert score.veto


# --- Precedent Alignment ---


class TestPrecedentAlignment:
    def test_no_history(self):
        dim = PrecedentAlignment()
        score = dim.evaluate(_action(), _ctx())
        assert score.score == 0.8

    def test_consistent_with_precedent(self):
        dim = PrecedentAlignment()
        history = [
            ActionRecord(
                action=_action("read"),
                verdict=GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8),
            )
            for _ in range(5)
        ]
        score = dim.evaluate(_action("read"), _ctx(history=history))
        assert score.score >= 0.9

    def test_frequently_denied_type(self):
        dim = PrecedentAlignment()
        history = [
            ActionRecord(
                action=_action("hack"),
                verdict=GovernanceVerdict(action_id="x", verdict=Verdict.DENY, ucs=0.0),
            )
            for _ in range(5)
        ]
        score = dim.evaluate(_action("hack"), _ctx(history=history))
        assert score.score <= 0.3


# --- Transparency ---


class TestTransparency:
    def test_fully_transparent(self):
        dim = Transparency()
        score = dim.evaluate(_action(), _ctx())
        assert score.score == 1.0

    def test_missing_fields_lower_score(self):
        dim = Transparency()
        a = Action()  # Mostly empty
        score = dim.evaluate(a, _ctx())
        assert score.score < 1.0


# --- Human Override ---


class TestHumanOverride:
    def test_no_override_needed(self):
        dim = HumanOverride()
        score = dim.evaluate(_action(), _ctx())
        assert score.score == 1.0

    def test_required_action_vetoes(self):
        dim = HumanOverride()
        dim.require_human_for("deploy")
        score = dim.evaluate(_action("deploy"), _ctx())
        assert score.veto

    def test_approved_action_passes(self):
        dim = HumanOverride()
        dim.require_human_for("deploy")
        action = _action("deploy")
        dim.approve(action.id)
        score = dim.evaluate(action, _ctx())
        assert score.score == 1.0

    def test_low_trust_requires_human(self):
        dim = HumanOverride()
        score = dim.evaluate(_action(), _ctx(trust=0.1))
        assert score.veto


# --- Ethical Alignment ---


class TestEthicalAlignment:
    def test_no_rules_passes(self):
        dim = EthicalAlignment()
        score = dim.evaluate(_action(), _ctx())
        assert score.score == 0.8

    def test_passing_rule(self):
        dim = EthicalAlignment()
        dim.add_rule(lambda a, c: (True, "ok"))
        score = dim.evaluate(_action(), _ctx())
        assert score.score == 1.0

    def test_failing_rule_vetoes(self):
        dim = EthicalAlignment()
        dim.add_rule(lambda a, c: (False, "violates principle X"))
        score = dim.evaluate(_action(), _ctx())
        assert score.veto
        assert "principle X" in score.reasoning


# --- Registry ---


class TestDimensionRegistry:
    def test_default_has_13_dimensions(self):
        reg = DimensionRegistry.create_default()
        assert len(reg.dimensions) == 13

    def test_evaluate_all_returns_13_scores(self):
        reg = DimensionRegistry.create_default()
        scores = reg.evaluate_all(_action(), _ctx())
        assert len(scores) == 13

    def test_all_dimension_names_unique(self):
        reg = DimensionRegistry.create_default()
        names = [d.name for d in reg.dimensions]
        assert len(names) == len(set(names))
