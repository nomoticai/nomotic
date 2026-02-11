"""Tests for the Unified Confidence Score engine."""

from nomotic.types import DimensionScore, TrustProfile
from nomotic.ucs import UCSEngine


def _trust(overall: float = 0.5) -> TrustProfile:
    return TrustProfile(agent_id="a", overall_trust=overall)


class TestUCSEngine:
    def test_empty_scores_returns_neutral(self):
        engine = UCSEngine()
        assert engine.compute([], _trust()) == 0.5

    def test_all_perfect_scores(self):
        engine = UCSEngine()
        scores = [DimensionScore(dimension_name=f"d{i}", score=1.0) for i in range(5)]
        ucs = engine.compute(scores, _trust())
        assert ucs > 0.9

    def test_all_zero_scores(self):
        engine = UCSEngine()
        scores = [DimensionScore(dimension_name=f"d{i}", score=0.0) for i in range(5)]
        ucs = engine.compute(scores, _trust())
        assert ucs < 0.1

    def test_veto_forces_zero(self):
        engine = UCSEngine()
        scores = [
            DimensionScore(dimension_name="d1", score=1.0),
            DimensionScore(dimension_name="d2", score=1.0, veto=True),
        ]
        assert engine.compute(scores, _trust()) == 0.0

    def test_high_trust_boosts_score(self):
        engine = UCSEngine(trust_influence=0.3)
        scores = [DimensionScore(dimension_name="d1", score=0.6)]
        low_trust = engine.compute(scores, _trust(0.2))
        high_trust = engine.compute(scores, _trust(0.9))
        assert high_trust > low_trust

    def test_weights_affect_outcome(self):
        engine = UCSEngine(trust_influence=0.0)
        scores_equal = [
            DimensionScore(dimension_name="good", score=1.0, weight=1.0),
            DimensionScore(dimension_name="bad", score=0.0, weight=1.0),
        ]
        scores_good_heavy = [
            DimensionScore(dimension_name="good", score=1.0, weight=3.0),
            DimensionScore(dimension_name="bad", score=0.0, weight=1.0),
        ]
        ucs_equal = engine.compute(scores_equal, _trust())
        ucs_heavy = engine.compute(scores_good_heavy, _trust())
        assert ucs_heavy > ucs_equal

    def test_low_individual_score_drags_down(self):
        engine = UCSEngine(trust_influence=0.0)
        scores_normal = [
            DimensionScore(dimension_name=f"d{i}", score=0.8) for i in range(5)
        ]
        scores_with_low = [
            DimensionScore(dimension_name=f"d{i}", score=0.8) for i in range(4)
        ] + [DimensionScore(dimension_name="d4", score=0.1)]
        ucs_normal = engine.compute(scores_normal, _trust())
        ucs_with_low = engine.compute(scores_with_low, _trust())
        assert ucs_with_low < ucs_normal

    def test_confidence_reduces_weight(self):
        engine = UCSEngine(trust_influence=0.0)
        scores_confident = [
            DimensionScore(dimension_name="d1", score=0.3, confidence=1.0),
            DimensionScore(dimension_name="d2", score=0.9, confidence=1.0),
        ]
        scores_uncertain = [
            DimensionScore(dimension_name="d1", score=0.3, confidence=0.1),
            DimensionScore(dimension_name="d2", score=0.9, confidence=1.0),
        ]
        ucs_confident = engine.compute(scores_confident, _trust())
        ucs_uncertain = engine.compute(scores_uncertain, _trust())
        # When the low score is uncertain, it has less effect
        assert ucs_uncertain > ucs_confident

    def test_result_bounded_zero_one(self):
        engine = UCSEngine(trust_influence=0.5)
        scores = [DimensionScore(dimension_name="d1", score=1.0)]
        assert 0.0 <= engine.compute(scores, _trust(1.0)) <= 1.0
        scores = [DimensionScore(dimension_name="d1", score=0.0)]
        assert 0.0 <= engine.compute(scores, _trust(0.0)) <= 1.0
