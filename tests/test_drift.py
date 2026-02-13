"""Tests for the drift calculator — _jsd, _temporal_distance, DriftScore, DriftCalculator."""

from __future__ import annotations

import math
import time
from unittest import TestCase

from nomotic.drift import DriftCalculator, DriftScore, _jsd, _temporal_distance
from nomotic.fingerprint import BehavioralFingerprint, TemporalPattern
from nomotic.priors import PriorRegistry
from nomotic.types import Action, Verdict


class TestJSD(TestCase):
    """Jensen-Shannon Divergence tests."""

    def test_identical_distributions_return_zero(self) -> None:
        p = {"a": 0.5, "b": 0.3, "c": 0.2}
        q = {"a": 0.5, "b": 0.3, "c": 0.2}
        self.assertAlmostEqual(_jsd(p, q), 0.0, places=10)

    def test_completely_disjoint_distributions_return_one(self) -> None:
        p = {"a": 1.0}
        q = {"b": 1.0}
        result = _jsd(p, q)
        # JSD of completely disjoint should be 1.0
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_partially_overlapping_distributions(self) -> None:
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.5, "c": 0.5}
        result = _jsd(p, q)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_symmetric(self) -> None:
        p = {"a": 0.7, "b": 0.2, "c": 0.1}
        q = {"a": 0.3, "b": 0.4, "c": 0.3}
        self.assertAlmostEqual(_jsd(p, q), _jsd(q, p), places=10)

    def test_both_empty_returns_zero(self) -> None:
        self.assertAlmostEqual(_jsd({}, {}), 0.0)

    def test_one_empty_returns_one(self) -> None:
        self.assertAlmostEqual(_jsd({"a": 1.0}, {}), 1.0)
        self.assertAlmostEqual(_jsd({}, {"a": 1.0}), 1.0)

    def test_single_entry_distributions(self) -> None:
        p = {"x": 1.0}
        q = {"x": 1.0}
        self.assertAlmostEqual(_jsd(p, q), 0.0, places=10)

    def test_bounded_zero_to_one(self) -> None:
        """JSD should always be in [0, 1]."""
        p = {"a": 0.1, "b": 0.9}
        q = {"a": 0.9, "b": 0.1}
        result = _jsd(p, q)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_slight_difference_small_jsd(self) -> None:
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.49, "b": 0.51}
        result = _jsd(p, q)
        self.assertLess(result, 0.01)


class TestTemporalDistance(TestCase):
    """Tests for _temporal_distance."""

    def test_identical_patterns_return_zero(self) -> None:
        tp = TemporalPattern(
            hourly_distribution={10: 0.5, 14: 0.5},
            actions_per_hour_mean=10.0,
            actions_per_hour_std=2.0,
        )
        result = _temporal_distance(tp, tp)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_completely_different_patterns_return_high(self) -> None:
        tp1 = TemporalPattern(
            hourly_distribution={0: 1.0},
            actions_per_hour_mean=10.0,
        )
        tp2 = TemporalPattern(
            hourly_distribution={12: 1.0},
            actions_per_hour_mean=100.0,
        )
        result = _temporal_distance(tp1, tp2)
        self.assertGreater(result, 0.5)

    def test_weights_hourly_and_rate(self) -> None:
        """Hourly = 0.6, rate = 0.4."""
        tp_base = TemporalPattern(
            hourly_distribution={10: 1.0},
            actions_per_hour_mean=10.0,
        )
        # Same hourly, different rate
        tp_rate_diff = TemporalPattern(
            hourly_distribution={10: 1.0},
            actions_per_hour_mean=20.0,  # 100% rate deviation
        )
        result = _temporal_distance(tp_base, tp_rate_diff)
        # Rate deviation = |20-10|/10 = 1.0, clamped to 1.0
        # Expected: 0.6*0.0 + 0.4*1.0 = 0.4
        self.assertAlmostEqual(result, 0.4, places=2)

    def test_both_empty_rates_zero(self) -> None:
        tp1 = TemporalPattern()
        tp2 = TemporalPattern()
        result = _temporal_distance(tp1, tp2)
        self.assertAlmostEqual(result, 0.0, places=5)


class TestDriftScore(TestCase):
    """Tests for DriftScore dataclass."""

    def test_severity_none(self) -> None:
        ds = DriftScore(overall=0.02, action_drift=0, target_drift=0, temporal_drift=0, outcome_drift=0, confidence=1, window_size=100, baseline_size=500)
        self.assertEqual(ds.severity, "none")

    def test_severity_low(self) -> None:
        ds = DriftScore(overall=0.10, action_drift=0, target_drift=0, temporal_drift=0, outcome_drift=0, confidence=1, window_size=100, baseline_size=500)
        self.assertEqual(ds.severity, "low")

    def test_severity_moderate(self) -> None:
        ds = DriftScore(overall=0.25, action_drift=0, target_drift=0, temporal_drift=0, outcome_drift=0, confidence=1, window_size=100, baseline_size=500)
        self.assertEqual(ds.severity, "moderate")

    def test_severity_high(self) -> None:
        ds = DriftScore(overall=0.50, action_drift=0, target_drift=0, temporal_drift=0, outcome_drift=0, confidence=1, window_size=100, baseline_size=500)
        self.assertEqual(ds.severity, "high")

    def test_severity_critical(self) -> None:
        ds = DriftScore(overall=0.75, action_drift=0, target_drift=0, temporal_drift=0, outcome_drift=0, confidence=1, window_size=100, baseline_size=500)
        self.assertEqual(ds.severity, "critical")

    def test_to_dict_serialization(self) -> None:
        ds = DriftScore(
            overall=0.25, action_drift=0.3, target_drift=0.2,
            temporal_drift=0.1, outcome_drift=0.15, confidence=0.8,
            window_size=100, baseline_size=500, detail="test detail",
        )
        d = ds.to_dict()
        self.assertEqual(d["severity"], "moderate")
        self.assertEqual(d["detail"], "test detail")
        self.assertEqual(d["window_size"], 100)
        self.assertEqual(d["baseline_size"], 500)
        self.assertIsInstance(d["overall"], float)


class TestDriftCalculator(TestCase):
    """Tests for DriftCalculator."""

    def setUp(self) -> None:
        self.calc = DriftCalculator()

    def _make_fp(self, agent_id: str, actions: list[tuple[str, str]]) -> BehavioralFingerprint:
        fp = BehavioralFingerprint(agent_id=agent_id)
        for action_type, target in actions:
            action = Action(
                agent_id=agent_id, action_type=action_type,
                target=target, timestamp=time.time(),
            )
            fp.observe(action, Verdict.ALLOW)
        return fp

    def test_identical_fingerprints_zero_drift(self) -> None:
        actions = [("read", "/data")] * 50
        fp1 = self._make_fp("a", actions)
        fp2 = self._make_fp("a", actions)
        score = self.calc.compare(fp1, fp2)
        self.assertAlmostEqual(score.overall, 0.0, places=2)
        self.assertAlmostEqual(score.action_drift, 0.0, places=2)

    def test_different_action_distributions_detect_drift(self) -> None:
        baseline = self._make_fp("a", [("read", "/data")] * 50)
        recent = self._make_fp("a", [("delete", "/data")] * 50)
        score = self.calc.compare(baseline, recent)
        self.assertGreater(score.action_drift, 0.5)
        self.assertGreater(score.overall, 0.0)

    def test_different_targets_detect_drift(self) -> None:
        baseline = self._make_fp("a", [("read", "/api/data")] * 50)
        recent = self._make_fp("a", [("read", "/api/secret")] * 50)
        score = self.calc.compare(baseline, recent)
        self.assertGreater(score.target_drift, 0.5)

    def test_drift_weights_emphasize_action(self) -> None:
        baseline = self._make_fp("a", [("read", "/data")] * 50)
        recent = self._make_fp("a", [("delete", "/data")] * 50)

        # Equal weights
        score_equal = self.calc.compare(baseline, recent, drift_weights={
            "action": 1.0, "target": 1.0, "temporal": 1.0, "outcome": 1.0,
        })
        # Heavy action weight
        score_action = self.calc.compare(baseline, recent, drift_weights={
            "action": 5.0, "target": 1.0, "temporal": 1.0, "outcome": 1.0,
        })
        # Action drift is high, so heavy action weight should push overall higher
        self.assertGreater(score_action.overall, score_equal.overall * 0.9)

    def test_both_empty_returns_zero_confidence(self) -> None:
        fp1 = BehavioralFingerprint(agent_id="a")
        fp2 = BehavioralFingerprint(agent_id="a")
        score = self.calc.compare(fp1, fp2)
        self.assertAlmostEqual(score.overall, 0.0)
        self.assertAlmostEqual(score.confidence, 0.0)

    def test_baseline_empty_recent_has_data(self) -> None:
        baseline = BehavioralFingerprint(agent_id="a")
        recent = self._make_fp("a", [("read", "/data")] * 20)
        score = self.calc.compare(baseline, recent)
        self.assertAlmostEqual(score.overall, 1.0)
        self.assertAlmostEqual(score.confidence, 0.0)

    def test_recent_empty_returns_zero(self) -> None:
        baseline = self._make_fp("a", [("read", "/data")] * 50)
        recent = BehavioralFingerprint(agent_id="a")
        score = self.calc.compare(baseline, recent)
        self.assertAlmostEqual(score.overall, 0.0)

    def test_low_observation_count_low_confidence(self) -> None:
        baseline = self._make_fp("a", [("read", "/data")] * 5)
        recent = self._make_fp("a", [("delete", "/data")] * 5)
        score = self.calc.compare(baseline, recent)
        self.assertLess(score.confidence, 0.1)

    def test_compare_against_prior(self) -> None:
        registry = PriorRegistry.with_defaults()
        prior = registry.get("customer-experience")
        self.assertIsNotNone(prior)

        # Agent behaving like a customer-experience agent
        actions = [("read", "/customer"), ("read", "/customer"), ("write", "/order")] * 20
        observed = self._make_fp("a", actions)

        score = self.calc.compare_against_prior(observed, prior)
        self.assertIsInstance(score, DriftScore)
        self.assertGreaterEqual(score.overall, 0.0)
        self.assertLessEqual(score.overall, 1.0)

    def test_detail_string_identifies_drift_source(self) -> None:
        baseline = self._make_fp("a", [("read", "/data")] * 50)
        recent = self._make_fp("a", [("delete", "/data")] * 50)
        score = self.calc.compare(baseline, recent)
        self.assertIn("Action", score.detail)
