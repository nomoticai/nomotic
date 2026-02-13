"""Tests for drift-based trust adjustment and trajectory recording in TrustCalibrator."""

from nomotic.drift import DriftScore
from nomotic.trajectory import (
    SOURCE_COMPLETION_INTERRUPT,
    SOURCE_COMPLETION_SUCCESS,
    SOURCE_DRIFT_ADJUSTMENT,
    SOURCE_DRIFT_RECOVERY,
    SOURCE_TIME_DECAY,
    SOURCE_VERDICT_ALLOW,
    SOURCE_VERDICT_DENY,
)
from nomotic.trust import TrustCalibrator, TrustConfig
from nomotic.types import (
    Action,
    ActionRecord,
    ActionState,
    GovernanceVerdict,
    TrustProfile,
    Verdict,
)


def _make_drift(overall: float, confidence: float = 1.0) -> DriftScore:
    """Helper to create a DriftScore with the given overall and confidence."""
    return DriftScore(
        overall=overall,
        action_drift=overall,
        target_drift=0.0,
        temporal_drift=0.0,
        outcome_drift=0.0,
        confidence=confidence,
        window_size=50,
        baseline_size=100,
    )


class TestApplyDrift:
    def test_no_change_below_threshold(self):
        cal = TrustCalibrator()
        cal.get_profile("a")
        initial = cal.get_profile("a").overall_trust
        cal.apply_drift("a", _make_drift(0.05))
        assert cal.get_profile("a").overall_trust == initial

    def test_low_drift_erosion(self):
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        cal.apply_drift("a", _make_drift(0.15))
        expected = initial - 0.002
        assert abs(cal.get_profile("a").overall_trust - expected) < 1e-10

    def test_moderate_drift_erosion(self):
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        cal.apply_drift("a", _make_drift(0.25))
        expected = initial - 0.008
        assert abs(cal.get_profile("a").overall_trust - expected) < 1e-10

    def test_high_drift_erosion(self):
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        cal.apply_drift("a", _make_drift(0.50))
        expected = initial - 0.02
        assert abs(cal.get_profile("a").overall_trust - expected) < 1e-10

    def test_critical_drift_erosion(self):
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        cal.apply_drift("a", _make_drift(0.70))
        expected = initial - 0.04
        assert abs(cal.get_profile("a").overall_trust - expected) < 1e-10

    def test_respects_min_trust(self):
        config = TrustConfig(min_trust=0.1, baseline_trust=0.15)
        cal = TrustCalibrator(config=config)
        profile = cal.get_profile("a")
        profile.overall_trust = 0.12
        cal.apply_drift("a", _make_drift(0.70))
        assert cal.get_profile("a").overall_trust >= 0.1

    def test_respects_max_trust(self):
        cal = TrustCalibrator()
        profile = cal.get_profile("a")
        profile.overall_trust = 0.94
        # Recovery case: first set previous drift high, then apply low drift
        cal._last_drift_scores["a"] = 0.30
        cal.apply_drift("a", _make_drift(0.05))
        assert cal.get_profile("a").overall_trust <= 0.95

    def test_scales_by_confidence(self):
        cal1 = TrustCalibrator()
        initial = cal1.get_profile("a").overall_trust
        cal1.apply_drift("a", _make_drift(0.50, confidence=1.0))
        full_drop = initial - cal1.get_profile("a").overall_trust

        cal2 = TrustCalibrator()
        initial2 = cal2.get_profile("a").overall_trust
        cal2.apply_drift("a", _make_drift(0.50, confidence=0.5))
        half_drop = initial2 - cal2.get_profile("a").overall_trust

        assert abs(half_drop - full_drop * 0.5) < 1e-10

    def test_records_trajectory_event(self):
        cal = TrustCalibrator()
        cal.get_profile("a")
        cal.apply_drift("a", _make_drift(0.30))
        traj = cal.get_trajectory("a")
        assert len(traj) == 1
        event = traj.latest
        assert event is not None
        assert event.source == SOURCE_DRIFT_ADJUSTMENT
        assert "drift" in event.reason.lower()
        assert event.metadata["drift_overall"] == 0.30

    def test_recovery_when_drift_decreases(self):
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        # First: high drift
        cal.apply_drift("a", _make_drift(0.40))
        trust_after_drift = cal.get_profile("a").overall_trust
        assert trust_after_drift < initial

        # Then: drift recovers below 0.15
        cal.apply_drift("a", _make_drift(0.05))
        trust_after_recovery = cal.get_profile("a").overall_trust
        assert trust_after_recovery > trust_after_drift

    def test_recovery_records_trajectory(self):
        cal = TrustCalibrator()
        cal.get_profile("a")
        # Set previous drift high
        cal.apply_drift("a", _make_drift(0.40))
        # Recover
        cal.apply_drift("a", _make_drift(0.05))
        traj = cal.get_trajectory("a")
        recovery_events = traj.events_by_source(SOURCE_DRIFT_RECOVERY)
        assert len(recovery_events) == 1
        assert recovery_events[0].delta > 0

    def test_no_recovery_when_drift_still_high(self):
        cal = TrustCalibrator()
        cal.get_profile("a")
        # Set previous drift
        cal.apply_drift("a", _make_drift(0.40))
        trust_after = cal.get_profile("a").overall_trust
        # Drift decreases but still above 0.15
        cal.apply_drift("a", _make_drift(0.20))
        # Should still erode (moderate), not recover
        assert cal.get_profile("a").overall_trust < trust_after

    def test_sustained_drift_cumulative(self):
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        for _ in range(5):
            cal.apply_drift("a", _make_drift(0.30))
        expected = initial - (0.008 * 5)
        assert abs(cal.get_profile("a").overall_trust - expected) < 1e-10

    def test_boundary_drift_0_10(self):
        """At exactly 0.10, low erosion applies."""
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        cal.apply_drift("a", _make_drift(0.10))
        expected = initial - 0.002
        assert abs(cal.get_profile("a").overall_trust - expected) < 1e-10

    def test_boundary_drift_0_20(self):
        """At exactly 0.20, moderate erosion applies."""
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        cal.apply_drift("a", _make_drift(0.20))
        expected = initial - 0.008
        assert abs(cal.get_profile("a").overall_trust - expected) < 1e-10

    def test_boundary_drift_0_40(self):
        """At exactly 0.40, high erosion applies."""
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        cal.apply_drift("a", _make_drift(0.40))
        expected = initial - 0.02
        assert abs(cal.get_profile("a").overall_trust - expected) < 1e-10

    def test_boundary_drift_0_60(self):
        """At exactly 0.60, critical erosion applies."""
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        cal.apply_drift("a", _make_drift(0.60))
        expected = initial - 0.04
        assert abs(cal.get_profile("a").overall_trust - expected) < 1e-10


class TestExistingTrustMathUnchanged:
    """Verify that existing trust adjustment math is identical to pre-4C behavior."""

    def test_verdict_deny_math(self):
        config = TrustConfig(violation_decrement=0.05, baseline_trust=0.5)
        cal = TrustCalibrator(config=config)
        profile = cal.get_profile("a")
        assert profile.overall_trust == 0.5

        verdict = GovernanceVerdict(action_id="x", verdict=Verdict.DENY, ucs=0.0)
        cal.record_verdict("a", verdict)
        assert abs(profile.overall_trust - 0.45) < 1e-10
        assert profile.violation_count == 1

    def test_verdict_allow_math(self):
        config = TrustConfig(success_increment=0.01, baseline_trust=0.5)
        cal = TrustCalibrator(config=config)
        profile = cal.get_profile("a")

        verdict = GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8)
        cal.record_verdict("a", verdict)
        assert abs(profile.overall_trust - 0.51) < 1e-10
        assert profile.successful_actions == 1

    def test_completion_interrupt_math(self):
        config = TrustConfig(interrupt_decrement=0.03, baseline_trust=0.5)
        cal = TrustCalibrator(config=config)
        profile = cal.get_profile("a")

        record = ActionRecord(
            action=Action(agent_id="a"),
            verdict=GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8),
            state=ActionState.INTERRUPTED,
            interrupted=True,
            interrupt_reason="policy violation",
        )
        cal.record_completion("a", record)
        assert abs(profile.overall_trust - 0.47) < 1e-10

    def test_completion_success_math(self):
        config = TrustConfig(success_increment=0.01, baseline_trust=0.5)
        cal = TrustCalibrator(config=config)
        profile = cal.get_profile("a")

        record = ActionRecord(
            action=Action(agent_id="a"),
            verdict=GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8),
            state=ActionState.COMPLETED,
        )
        cal.record_completion("a", record)
        # success_increment * 0.5 = 0.005
        assert abs(profile.overall_trust - 0.505) < 1e-10


class TestTrajectoryRecording:
    """Test that existing trust methods now record trajectory events."""

    def test_verdict_deny_records_trajectory(self):
        cal = TrustCalibrator()
        verdict = GovernanceVerdict(
            action_id="x", verdict=Verdict.DENY, ucs=0.0,
            reasoning="scope violation",
        )
        cal.record_verdict("a", verdict)
        traj = cal.get_trajectory("a")
        assert len(traj) == 1
        event = traj.latest
        assert event is not None
        assert event.source == SOURCE_VERDICT_DENY
        assert "scope violation" in event.reason
        assert event.delta < 0

    def test_verdict_allow_records_trajectory(self):
        cal = TrustCalibrator()
        verdict = GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8)
        cal.record_verdict("a", verdict)
        traj = cal.get_trajectory("a")
        assert len(traj) == 1
        event = traj.latest
        assert event is not None
        assert event.source == SOURCE_VERDICT_ALLOW
        assert event.delta > 0

    def test_completion_interrupt_records_trajectory(self):
        cal = TrustCalibrator()
        record = ActionRecord(
            action=Action(agent_id="a"),
            verdict=GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8),
            state=ActionState.INTERRUPTED,
            interrupted=True,
            interrupt_reason="safety concern",
        )
        cal.record_completion("a", record)
        traj = cal.get_trajectory("a")
        assert len(traj) == 1
        event = traj.latest
        assert event is not None
        assert event.source == SOURCE_COMPLETION_INTERRUPT
        assert "safety concern" in event.reason

    def test_completion_success_records_trajectory(self):
        cal = TrustCalibrator()
        record = ActionRecord(
            action=Action(agent_id="a"),
            verdict=GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8),
            state=ActionState.COMPLETED,
        )
        cal.record_completion("a", record)
        traj = cal.get_trajectory("a")
        assert len(traj) == 1
        event = traj.latest
        assert event is not None
        assert event.source == SOURCE_COMPLETION_SUCCESS

    def test_time_decay_records_trajectory_when_significant(self):
        import time as _time

        cal = TrustCalibrator(config=TrustConfig(decay_rate=1.0))
        profile = cal.get_profile("a")
        profile.overall_trust = 0.8
        # Simulate 1 hour of inactivity
        profile.last_updated = _time.time() - 3600
        cal.apply_time_decay("a")
        traj = cal.get_trajectory("a")
        # Should record if the decay was significant
        if abs(profile.overall_trust - 0.8) > 0.001:
            assert len(traj) >= 1
            event = traj.latest
            assert event is not None
            assert event.source == SOURCE_TIME_DECAY

    def test_get_trajectory_creates_if_not_exists(self):
        cal = TrustCalibrator()
        traj = cal.get_trajectory("new-agent")
        assert traj is not None
        assert traj.agent_id == "new-agent"
        assert len(traj) == 0

    def test_same_trajectory_returned_for_same_agent(self):
        cal = TrustCalibrator()
        t1 = cal.get_trajectory("a")
        t2 = cal.get_trajectory("a")
        assert t1 is t2
