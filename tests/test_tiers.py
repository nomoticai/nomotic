"""Tests for the three-tier evaluation architecture."""

from nomotic.types import Action, AgentContext, DimensionScore, TrustProfile, Verdict
from nomotic.tiers import TierOneGate, TierTwoEvaluator, TierThreeDeliberator


def _ctx(trust: float = 0.5) -> AgentContext:
    return AgentContext(
        agent_id="a",
        trust_profile=TrustProfile(agent_id="a", overall_trust=trust),
    )


def _action() -> Action:
    return Action(agent_id="a", action_type="test")


class TestTierOneGate:
    def test_no_vetoes_passes_through(self):
        gate = TierOneGate()
        scores = [DimensionScore(dimension_name="d1", score=0.8)]
        result = gate.evaluate(_action(), _ctx(), scores)
        assert not result.decided

    def test_veto_denies(self):
        gate = TierOneGate()
        scores = [
            DimensionScore(dimension_name="scope_compliance", score=0.0, veto=True),
        ]
        result = gate.evaluate(_action(), _ctx(), scores)
        assert result.decided
        assert result.verdict.verdict == Verdict.DENY
        assert "scope_compliance" in result.verdict.vetoed_by

    def test_human_override_veto_escalates(self):
        gate = TierOneGate()
        scores = [
            DimensionScore(dimension_name="human_override", score=0.0, veto=True),
        ]
        result = gate.evaluate(_action(), _ctx(), scores)
        assert result.decided
        assert result.verdict.verdict == Verdict.ESCALATE

    def test_human_plus_other_veto_denies(self):
        gate = TierOneGate()
        scores = [
            DimensionScore(dimension_name="human_override", score=0.0, veto=True),
            DimensionScore(dimension_name="scope_compliance", score=0.0, veto=True),
        ]
        result = gate.evaluate(_action(), _ctx(), scores)
        assert result.decided
        assert result.verdict.verdict == Verdict.DENY


class TestTierTwoEvaluator:
    def test_high_ucs_allows(self):
        tier2 = TierTwoEvaluator(allow_threshold=0.7)
        scores = [DimensionScore(dimension_name="d1", score=0.9)]
        result = tier2.evaluate(_action(), _ctx(), scores, ucs=0.85)
        assert result.decided
        assert result.verdict.verdict == Verdict.ALLOW

    def test_low_ucs_denies(self):
        tier2 = TierTwoEvaluator(deny_threshold=0.3)
        scores = [DimensionScore(dimension_name="d1", score=0.1)]
        result = tier2.evaluate(_action(), _ctx(), scores, ucs=0.15)
        assert result.decided
        assert result.verdict.verdict == Verdict.DENY

    def test_ambiguous_ucs_passes_through(self):
        tier2 = TierTwoEvaluator(allow_threshold=0.7, deny_threshold=0.3)
        scores = [DimensionScore(dimension_name="d1", score=0.5)]
        result = tier2.evaluate(_action(), _ctx(), scores, ucs=0.5)
        assert not result.decided


class TestTierThreeDeliberator:
    def test_high_trust_borderline_allows(self):
        tier3 = TierThreeDeliberator()
        scores = [DimensionScore(dimension_name="d1", score=0.6)]
        result = tier3.evaluate(_action(), _ctx(trust=0.8), scores, ucs=0.55)
        assert result.decided
        assert result.verdict.verdict == Verdict.ALLOW

    def test_low_trust_escalates(self):
        tier3 = TierThreeDeliberator()
        scores = [DimensionScore(dimension_name="d1", score=0.5)]
        result = tier3.evaluate(_action(), _ctx(trust=0.2), scores, ucs=0.5)
        assert result.decided
        assert result.verdict.verdict == Verdict.ESCALATE

    def test_critical_low_dimension_modifies(self):
        tier3 = TierThreeDeliberator()
        scores = [
            DimensionScore(dimension_name="safe", score=0.8, weight=1.0),
            DimensionScore(dimension_name="risky", score=0.2, weight=1.5),
        ]
        result = tier3.evaluate(_action(), _ctx(trust=0.5), scores, ucs=0.5)
        assert result.decided
        assert result.verdict.verdict == Verdict.MODIFY

    def test_custom_deliberator(self):
        tier3 = TierThreeDeliberator()
        tier3.add_deliberator(lambda a, c, s, u: Verdict.DENY)
        scores = [DimensionScore(dimension_name="d1", score=0.5)]
        result = tier3.evaluate(_action(), _ctx(), scores, ucs=0.5)
        assert result.verdict.verdict == Verdict.DENY

    def test_all_tiers_set_tier_number(self):
        gate = TierOneGate()
        scores = [DimensionScore(dimension_name="d1", score=0.0, veto=True)]
        r = gate.evaluate(_action(), _ctx(), scores)
        assert r.verdict.tier == 1

        tier2 = TierTwoEvaluator()
        scores2 = [DimensionScore(dimension_name="d1", score=0.9)]
        r2 = tier2.evaluate(_action(), _ctx(), scores2, ucs=0.9)
        assert r2.verdict.tier == 2

        tier3 = TierThreeDeliberator()
        scores3 = [DimensionScore(dimension_name="d1", score=0.5)]
        r3 = tier3.evaluate(_action(), _ctx(trust=0.8), scores3, ucs=0.55)
        assert r3.verdict.tier == 3
