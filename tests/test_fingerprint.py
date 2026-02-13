"""Tests for the BehavioralFingerprint and TemporalPattern data model."""

from __future__ import annotations

import math
import threading
import time

import pytest

from nomotic.fingerprint import BehavioralFingerprint, TemporalPattern
from nomotic.priors import ArchetypePrior, PriorRegistry, TemporalProfile
from nomotic.types import Action, Verdict


# ── Helpers ────────────────────────────────────────────────────────────────


def _action(action_type: str = "read", target: str = "/api/data", ts: float | None = None) -> Action:
    """Create a test action with a specific timestamp."""
    return Action(
        agent_id="test-agent",
        action_type=action_type,
        target=target,
        timestamp=ts if ts is not None else time.time(),
    )


def _simple_prior() -> ArchetypePrior:
    """Create a minimal prior for testing."""
    return ArchetypePrior(
        archetype_name="test-archetype",
        action_distribution={"read": 0.6, "write": 0.3, "delete": 0.1},
        target_categories={"data": 0.7, "config": 0.3},
        temporal_profile=TemporalProfile(
            peak_hours={10, 11, 12},
            active_hours=set(range(8, 18)),
            expected_rate_range=(50.0, 200.0),
            business_hours_only=True,
        ),
        outcome_expectations={"ALLOW": 0.9, "DENY": 0.1},
        prior_weight=50,
    )


# ── TemporalPattern Tests ─────────────────────────────────────────────────


class TestTemporalPattern:
    def test_default_empty(self):
        tp = TemporalPattern()
        assert tp.hourly_distribution == {}
        assert tp.actions_per_hour_mean == 0.0
        assert tp.actions_per_hour_std == 0.0
        assert tp.active_hours == set()

    def test_to_dict_empty(self):
        tp = TemporalPattern()
        d = tp.to_dict()
        assert d["hourly_distribution"] == {}
        assert d["active_hours"] == []
        assert d["actions_per_hour_mean"] == 0.0
        assert d["actions_per_hour_std"] == 0.0

    def test_to_dict_with_data(self):
        tp = TemporalPattern(
            hourly_distribution={10: 0.5, 14: 0.5},
            actions_per_hour_mean=100.0,
            actions_per_hour_std=20.0,
            active_hours={10, 14},
        )
        d = tp.to_dict()
        assert d["hourly_distribution"] == {"10": 0.5, "14": 0.5}
        assert d["active_hours"] == [10, 14]  # sorted

    def test_from_dict_roundtrip(self):
        tp = TemporalPattern(
            hourly_distribution={9: 0.3, 15: 0.7},
            actions_per_hour_mean=150.0,
            actions_per_hour_std=30.0,
            active_hours={9, 15},
        )
        d = tp.to_dict()
        tp2 = TemporalPattern.from_dict(d)
        assert tp2.hourly_distribution == tp.hourly_distribution
        assert tp2.actions_per_hour_mean == tp.actions_per_hour_mean
        assert tp2.actions_per_hour_std == tp.actions_per_hour_std
        assert tp2.active_hours == tp.active_hours


# ── BehavioralFingerprint Core Tests ──────────────────────────────────────


class TestBehavioralFingerprintEmpty:
    def test_zero_observations(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        assert fp.total_observations == 0
        assert fp.action_distribution == {}
        assert fp.target_distribution == {}
        assert fp.outcome_distribution == {}
        assert fp.confidence == 0.0

    def test_confidence_at_zero(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        assert fp.confidence == 0.0

    def test_to_dict_empty(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        d = fp.to_dict()
        assert d["agent_id"] == "agent-1"
        assert d["total_observations"] == 0
        assert d["confidence"] == 0.0
        assert d["action_distribution"] == {}

    def test_from_dict_empty(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        d = fp.to_dict()
        fp2 = BehavioralFingerprint.from_dict(d)
        assert fp2.agent_id == "agent-1"
        assert fp2.total_observations == 0


class TestBehavioralFingerprintObserve:
    def test_single_observation(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        action = _action("read", "/api/data")
        fp.observe(action, Verdict.ALLOW)

        assert fp.total_observations == 1
        assert fp.action_distribution == {"read": 1.0}
        assert fp.target_distribution == {"/api/data": 1.0}
        assert fp.outcome_distribution == {"ALLOW": 1.0}

    def test_two_different_actions(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        fp.observe(_action("read"), Verdict.ALLOW)
        fp.observe(_action("write"), Verdict.ALLOW)

        assert fp.total_observations == 2
        assert fp.action_distribution["read"] == pytest.approx(0.5)
        assert fp.action_distribution["write"] == pytest.approx(0.5)

    def test_hundred_mixed_actions(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        # 60 reads, 30 writes, 10 deletes
        for _ in range(60):
            fp.observe(_action("read"), Verdict.ALLOW)
        for _ in range(30):
            fp.observe(_action("write"), Verdict.ALLOW)
        for _ in range(10):
            fp.observe(_action("delete"), Verdict.DENY)

        assert fp.total_observations == 100
        assert fp.action_distribution["read"] == pytest.approx(0.6)
        assert fp.action_distribution["write"] == pytest.approx(0.3)
        assert fp.action_distribution["delete"] == pytest.approx(0.1)
        assert sum(fp.action_distribution.values()) == pytest.approx(1.0)

    def test_distributions_sum_to_one(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        for i in range(50):
            action_type = ["read", "write", "query", "send"][i % 4]
            target = ["/db", "/api", "/cache"][i % 3]
            verdict = [Verdict.ALLOW, Verdict.DENY][i % 7 == 0]
            fp.observe(_action(action_type, target), verdict)

        assert sum(fp.action_distribution.values()) == pytest.approx(1.0)
        assert sum(fp.target_distribution.values()) == pytest.approx(1.0)
        assert sum(fp.outcome_distribution.values()) == pytest.approx(1.0)

    def test_temporal_pattern_updated(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        # Create action at hour 14
        from datetime import datetime
        ts = datetime(2024, 1, 15, 14, 30, 0).timestamp()
        fp.observe(_action("read", ts=ts), Verdict.ALLOW)

        assert 14 in fp.temporal_pattern.active_hours
        assert fp.temporal_pattern.hourly_distribution.get(14) == pytest.approx(1.0)

    def test_outcome_distribution(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        for _ in range(90):
            fp.observe(_action("read"), Verdict.ALLOW)
        for _ in range(5):
            fp.observe(_action("write"), Verdict.MODIFY)
        for _ in range(3):
            fp.observe(_action("query"), Verdict.ESCALATE)
        for _ in range(2):
            fp.observe(_action("delete"), Verdict.DENY)

        assert fp.outcome_distribution["ALLOW"] == pytest.approx(0.9)
        assert fp.outcome_distribution["MODIFY"] == pytest.approx(0.05)
        assert fp.outcome_distribution["ESCALATE"] == pytest.approx(0.03)
        assert fp.outcome_distribution["DENY"] == pytest.approx(0.02)

    def test_incremental_consistency(self):
        """Calling observe one-at-a-time produces same result as batch."""
        fp1 = BehavioralFingerprint(agent_id="agent-1")
        fp2 = BehavioralFingerprint(agent_id="agent-2")
        actions_data = [
            ("read", "/api", Verdict.ALLOW),
            ("write", "/db", Verdict.ALLOW),
            ("read", "/api", Verdict.DENY),
            ("query", "/cache", Verdict.ALLOW),
            ("read", "/api", Verdict.ALLOW),
        ]
        # Feed same data to both
        for at, tgt, v in actions_data:
            a = _action(at, tgt)
            fp1.observe(a, v)
            fp2.observe(a, v)

        assert fp1.action_distribution == fp2.action_distribution
        assert fp1.target_distribution == fp2.target_distribution
        assert fp1.outcome_distribution == fp2.outcome_distribution
        assert fp1.total_observations == fp2.total_observations


class TestBehavioralFingerprintConfidence:
    def test_confidence_asymptote(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        # 0 observations
        assert fp.confidence == pytest.approx(0.0)

        # 100 observations
        fp.total_observations = 100
        assert fp.confidence == pytest.approx(1 - math.exp(-100 / 200), abs=0.01)
        assert abs(fp.confidence - 0.3935) < 0.01  # ~0.39

        # 500 observations
        fp.total_observations = 500
        assert fp.confidence == pytest.approx(1 - math.exp(-500 / 200), abs=0.01)
        assert fp.confidence > 0.9

        # 1000 observations
        fp.total_observations = 1000
        assert fp.confidence == pytest.approx(1 - math.exp(-1000 / 200), abs=0.001)
        assert fp.confidence > 0.99

    def test_confidence_monotonically_increases(self):
        prev = 0.0
        for n in range(0, 2000, 100):
            fp = BehavioralFingerprint(agent_id="x", total_observations=n)
            assert fp.confidence >= prev
            prev = fp.confidence


class TestBehavioralFingerprintPrior:
    def test_from_prior_seeds_distributions(self):
        prior = _simple_prior()
        fp = BehavioralFingerprint.from_prior("agent-1", prior)

        assert fp.agent_id == "agent-1"
        assert fp.total_observations == 0
        # Distributions should match the prior exactly at 0 observations
        assert fp.action_distribution["read"] == pytest.approx(0.6)
        assert fp.action_distribution["write"] == pytest.approx(0.3)
        assert fp.action_distribution["delete"] == pytest.approx(0.1)

    def test_prior_blending_at_zero_observations(self):
        prior = _simple_prior()
        fp = BehavioralFingerprint.from_prior("agent-1", prior)
        # At 0 real observations, distribution equals the prior
        assert fp.action_distribution == prior.action_distribution

    def test_prior_blending_fades_with_observations(self):
        prior = _simple_prior()
        fp = BehavioralFingerprint.from_prior("agent-1", prior)

        # Feed 50 "write" actions (prior_weight is 50)
        for _ in range(50):
            fp.observe(_action("write"), Verdict.ALLOW)

        # At 50 real observations (all writes), distribution is ~50% prior, ~50% observed
        # Prior says: read=0.6*50=30, write=0.3*50=15, delete=0.1*50=5
        # Observed: write=50
        # Total: 50 + 50 = 100
        # read = 30/100 = 0.30
        # write = (15+50)/100 = 0.65
        # delete = 5/100 = 0.05
        assert fp.action_distribution["read"] == pytest.approx(0.30)
        assert fp.action_distribution["write"] == pytest.approx(0.65)
        assert fp.action_distribution["delete"] == pytest.approx(0.05)

    def test_prior_diminishes_at_many_observations(self):
        prior = _simple_prior()
        fp = BehavioralFingerprint.from_prior("agent-1", prior)

        # Feed 500 "write" actions
        for _ in range(500):
            fp.observe(_action("write"), Verdict.ALLOW)

        # Prior contributes 50, observed 500, total 550
        # write = (15 + 500) / 550 ≈ 0.936
        assert fp.action_distribution["write"] > 0.9
        assert fp.action_distribution["read"] < 0.1

    def test_prior_weight_higher_persists_longer(self):
        prior_low = ArchetypePrior(
            archetype_name="low",
            action_distribution={"read": 1.0},
            target_categories={"data": 1.0},
            temporal_profile=TemporalProfile(
                peak_hours=set(), active_hours=set(),
                expected_rate_range=(0, 0), business_hours_only=False,
            ),
            outcome_expectations={"ALLOW": 1.0},
            prior_weight=10,
        )
        prior_high = ArchetypePrior(
            archetype_name="high",
            action_distribution={"read": 1.0},
            target_categories={"data": 1.0},
            temporal_profile=TemporalProfile(
                peak_hours=set(), active_hours=set(),
                expected_rate_range=(0, 0), business_hours_only=False,
            ),
            outcome_expectations={"ALLOW": 1.0},
            prior_weight=200,
        )

        fp_low = BehavioralFingerprint.from_prior("a1", prior_low)
        fp_high = BehavioralFingerprint.from_prior("a2", prior_high)

        # Feed same 50 "write" actions to both
        for _ in range(50):
            fp_low.observe(_action("write"), Verdict.ALLOW)
            fp_high.observe(_action("write"), Verdict.ALLOW)

        # Higher prior weight should keep more "read" proportion
        assert fp_high.action_distribution.get("read", 0) > fp_low.action_distribution.get("read", 0)


class TestBehavioralFingerprintSerialization:
    def test_roundtrip_with_data(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        for _ in range(10):
            fp.observe(_action("read", "/api"), Verdict.ALLOW)
        for _ in range(5):
            fp.observe(_action("write", "/db"), Verdict.DENY)

        d = fp.to_dict()
        fp2 = BehavioralFingerprint.from_dict(d)

        assert fp2.agent_id == fp.agent_id
        assert fp2.total_observations == fp.total_observations
        assert fp2.action_distribution == pytest.approx(fp.action_distribution)
        assert fp2.target_distribution == pytest.approx(fp.target_distribution)
        assert fp2.outcome_distribution == pytest.approx(fp.outcome_distribution)

    def test_roundtrip_empty(self):
        fp = BehavioralFingerprint(agent_id="empty-agent")
        d = fp.to_dict()
        fp2 = BehavioralFingerprint.from_dict(d)
        assert fp2.agent_id == "empty-agent"
        assert fp2.total_observations == 0

    def test_roundtrip_with_prior(self):
        prior = _simple_prior()
        fp = BehavioralFingerprint.from_prior("agent-1", prior)
        for _ in range(20):
            fp.observe(_action("read"), Verdict.ALLOW)

        d = fp.to_dict()
        fp2 = BehavioralFingerprint.from_dict(d)
        assert fp2._prior_weight == prior.prior_weight
        assert fp2.action_distribution == pytest.approx(fp.action_distribution)

    def test_to_dict_no_sets(self):
        """Serialized dict must be JSON-safe — no sets."""
        fp = BehavioralFingerprint(agent_id="agent-1")
        fp.observe(_action("read"), Verdict.ALLOW)
        d = fp.to_dict()

        # Check that active_hours is a list, not a set
        tp = d["temporal_pattern"]
        assert isinstance(tp["active_hours"], list)

        # Check no sets anywhere in the dict
        import json
        json.dumps(d)  # Should not raise


class TestBehavioralFingerprintThreadSafety:
    def test_concurrent_observe(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        errors = []

        def worker(action_type: str, count: int):
            try:
                for _ in range(count):
                    fp.observe(_action(action_type), Verdict.ALLOW)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("read", 100)),
            threading.Thread(target=worker, args=("write", 100)),
            threading.Thread(target=worker, args=("query", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert fp.total_observations == 300
        assert sum(fp.action_distribution.values()) == pytest.approx(1.0)


class TestBehavioralFingerprintTemporalRate:
    def test_hourly_rate_single_hour(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        ts = time.time()
        for i in range(10):
            fp.observe(_action("read", ts=ts + i), Verdict.ALLOW)

        assert fp.temporal_pattern.actions_per_hour_mean == pytest.approx(10.0)
        assert fp.temporal_pattern.actions_per_hour_std == pytest.approx(0.0)

    def test_hourly_rate_multiple_hours(self):
        fp = BehavioralFingerprint(agent_id="agent-1")
        # Two different hour slots
        base_ts = time.time()
        hour_slot_1 = (base_ts // 3600) * 3600
        hour_slot_2 = hour_slot_1 + 3600

        for i in range(10):
            fp.observe(_action("read", ts=hour_slot_1 + i), Verdict.ALLOW)
        for i in range(20):
            fp.observe(_action("read", ts=hour_slot_2 + i), Verdict.ALLOW)

        assert fp.temporal_pattern.actions_per_hour_mean == pytest.approx(15.0)
        assert fp.temporal_pattern.actions_per_hour_std > 0
