"""Tests for the trust calibration system."""

from nomotic.types import ActionRecord, ActionState, GovernanceVerdict, TrustProfile, Verdict, Action
from nomotic.trust import TrustCalibrator, TrustConfig


class TestTrustCalibrator:
    def test_new_agent_gets_baseline_trust(self):
        cal = TrustCalibrator()
        profile = cal.get_profile("new-agent")
        assert profile.overall_trust == 0.5

    def test_deny_decreases_trust(self):
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        verdict = GovernanceVerdict(action_id="x", verdict=Verdict.DENY, ucs=0.0)
        cal.record_verdict("a", verdict)
        assert cal.get_profile("a").overall_trust < initial

    def test_allow_increases_trust(self):
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        verdict = GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8)
        cal.record_verdict("a", verdict)
        assert cal.get_profile("a").overall_trust > initial

    def test_trust_loss_faster_than_gain(self):
        config = TrustConfig(success_increment=0.01, violation_decrement=0.05)
        cal = TrustCalibrator(config=config)
        cal.get_profile("a")

        # One deny
        deny = GovernanceVerdict(action_id="x", verdict=Verdict.DENY, ucs=0.0)
        cal.record_verdict("a", deny)
        after_deny = cal.get_profile("a").overall_trust

        # Five allows to try to recover
        for i in range(5):
            allow = GovernanceVerdict(action_id=f"y{i}", verdict=Verdict.ALLOW, ucs=0.8)
            cal.record_verdict("a", allow)
        after_recovery = cal.get_profile("a").overall_trust

        # 5 successes at 0.01 each = 0.05, same as the single deny
        # So after one deny + 5 allows, trust should be approximately back to baseline
        assert after_recovery >= 0.49  # approximately back

    def test_trust_bounded(self):
        config = TrustConfig(min_trust=0.1, max_trust=0.9)
        cal = TrustCalibrator(config=config)

        # Many denials
        for i in range(100):
            deny = GovernanceVerdict(action_id=f"d{i}", verdict=Verdict.DENY, ucs=0.0)
            cal.record_verdict("a", deny)
        assert cal.get_profile("a").overall_trust >= 0.1

        # Many allows
        for i in range(100):
            allow = GovernanceVerdict(action_id=f"a{i}", verdict=Verdict.ALLOW, ucs=0.8)
            cal.record_verdict("b", allow)
        assert cal.get_profile("b").overall_trust <= 0.9

    def test_interrupt_decreases_trust(self):
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        record = ActionRecord(
            action=Action(agent_id="a"),
            verdict=GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8),
            state=ActionState.INTERRUPTED,
            interrupted=True,
        )
        cal.record_completion("a", record)
        assert cal.get_profile("a").overall_trust < initial

    def test_successful_completion_increases_trust(self):
        cal = TrustCalibrator()
        initial = cal.get_profile("a").overall_trust
        record = ActionRecord(
            action=Action(agent_id="a"),
            verdict=GovernanceVerdict(action_id="x", verdict=Verdict.ALLOW, ucs=0.8),
            state=ActionState.COMPLETED,
        )
        cal.record_completion("a", record)
        assert cal.get_profile("a").overall_trust > initial

    def test_violation_count_tracked(self):
        cal = TrustCalibrator()
        for i in range(3):
            deny = GovernanceVerdict(action_id=f"d{i}", verdict=Verdict.DENY, ucs=0.0)
            cal.record_verdict("a", deny)
        assert cal.get_profile("a").violation_count == 3
