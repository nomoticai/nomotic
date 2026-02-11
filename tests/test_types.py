"""Tests for core types."""

from nomotic.types import (
    Action,
    ActionRecord,
    ActionState,
    AgentContext,
    DimensionScore,
    GovernanceVerdict,
    InterruptRequest,
    Severity,
    TrustProfile,
    Verdict,
)


def test_action_has_unique_id():
    a1 = Action()
    a2 = Action()
    assert a1.id != a2.id


def test_action_frozen():
    a = Action(agent_id="test", action_type="read", target="db")
    assert a.agent_id == "test"
    assert a.action_type == "read"
    assert a.target == "db"


def test_trust_profile_violation_rate():
    p = TrustProfile(agent_id="a")
    assert p.violation_rate == 0.0

    p.violation_count = 2
    p.successful_actions = 8
    assert p.violation_rate == 0.2


def test_trust_profile_defaults():
    p = TrustProfile(agent_id="a")
    assert p.overall_trust == 0.5
    assert p.violation_count == 0
    assert p.successful_actions == 0


def test_verdict_enum():
    assert Verdict.ALLOW != Verdict.DENY
    assert Verdict.ESCALATE != Verdict.MODIFY


def test_governance_verdict_structure():
    v = GovernanceVerdict(
        action_id="abc",
        verdict=Verdict.ALLOW,
        ucs=0.85,
        tier=2,
    )
    assert v.action_id == "abc"
    assert v.ucs == 0.85
    assert v.tier == 2
    assert v.vetoed_by == []


def test_dimension_score_defaults():
    s = DimensionScore(dimension_name="test", score=0.7)
    assert s.weight == 1.0
    assert s.confidence == 1.0
    assert s.veto is False


def test_action_state_lifecycle():
    states = [
        ActionState.PENDING,
        ActionState.EVALUATING,
        ActionState.APPROVED,
        ActionState.EXECUTING,
        ActionState.COMPLETED,
    ]
    assert len(set(states)) == 5


def test_interrupt_request():
    r = InterruptRequest(
        action_id="a1",
        reason="policy violation",
        source="scope_compliance",
        severity=Severity.CRITICAL,
    )
    assert r.scope == "action"
