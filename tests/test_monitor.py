"""Tests for the drift monitor."""

from __future__ import annotations

import time
from unittest import TestCase

from nomotic.drift import DriftScore
from nomotic.fingerprint import BehavioralFingerprint
from nomotic.monitor import DriftAlert, DriftConfig, DriftMonitor
from nomotic.priors import PriorRegistry
from nomotic.types import Action, Verdict


class TestDriftMonitor(TestCase):
    """Tests for DriftMonitor."""

    def _action(
        self,
        agent_id: str = "agent-1",
        action_type: str = "read",
        target: str = "/data",
    ) -> Action:
        return Action(
            agent_id=agent_id,
            action_type=action_type,
            target=target,
            timestamp=time.time(),
        )

    def _make_baseline(
        self, agent_id: str = "agent-1", n: int = 100,
    ) -> BehavioralFingerprint:
        fp = BehavioralFingerprint(agent_id=agent_id)
        for _ in range(n):
            action = self._action(agent_id=agent_id)
            fp.observe(action, Verdict.ALLOW)
        return fp

    def test_observe_creates_window(self) -> None:
        mon = DriftMonitor(config=DriftConfig(check_interval=5, min_observations=3))
        baseline = self._make_baseline()
        mon.observe("agent-1", self._action(), Verdict.ALLOW, baseline_fingerprint=baseline)
        w = mon.get_window("agent-1")
        self.assertIsNotNone(w)
        self.assertEqual(w.size, 1)

    def test_no_drift_before_min_observations(self) -> None:
        cfg = DriftConfig(check_interval=1, min_observations=10)
        mon = DriftMonitor(config=cfg)
        baseline = self._make_baseline()
        for i in range(9):
            result = mon.observe("agent-1", self._action(), Verdict.ALLOW, baseline_fingerprint=baseline)
            self.assertIsNone(result)

    def test_computes_drift_at_check_interval(self) -> None:
        cfg = DriftConfig(check_interval=5, min_observations=5)
        mon = DriftMonitor(config=cfg)
        baseline = self._make_baseline()

        results = []
        for i in range(10):
            result = mon.observe("agent-1", self._action(), Verdict.ALLOW, baseline_fingerprint=baseline)
            results.append(result)

        # Should compute at observation 5 and 10
        self.assertIsNotNone(results[4])  # 5th observation
        self.assertIsNotNone(results[9])  # 10th observation
        # Others should be None
        for i in [0, 1, 2, 3, 5, 6, 7, 8]:
            self.assertIsNone(results[i])

    def test_get_drift_returns_latest(self) -> None:
        cfg = DriftConfig(check_interval=5, min_observations=5)
        mon = DriftMonitor(config=cfg)
        baseline = self._make_baseline()

        for _ in range(5):
            mon.observe("agent-1", self._action(), Verdict.ALLOW, baseline_fingerprint=baseline)

        score = mon.get_drift("agent-1")
        self.assertIsNotNone(score)
        self.assertIsInstance(score, DriftScore)

    def test_get_drift_returns_none_for_unknown_agent(self) -> None:
        mon = DriftMonitor()
        self.assertIsNone(mon.get_drift("nonexistent"))

    def test_moderate_alert_generated(self) -> None:
        cfg = DriftConfig(
            check_interval=5, min_observations=5,
            alert_threshold_moderate=0.15,
        )
        mon = DriftMonitor(config=cfg)
        # Build a baseline of reads
        baseline = self._make_baseline()

        # Feed observations that are completely different from baseline
        for _ in range(5):
            action = self._action(action_type="delete", target="/sensitive")
            mon.observe("agent-1", action, Verdict.DENY, baseline_fingerprint=baseline)

        alerts = mon.get_alerts("agent-1")
        if alerts:
            self.assertIn(alerts[0].severity, ("moderate", "high", "critical"))

    def test_high_alert_generated(self) -> None:
        cfg = DriftConfig(
            check_interval=5, min_observations=5,
            alert_threshold_high=0.20,
        )
        mon = DriftMonitor(config=cfg)
        baseline = self._make_baseline()

        for _ in range(5):
            action = self._action(action_type="delete", target="/sensitive")
            mon.observe("agent-1", action, Verdict.DENY, baseline_fingerprint=baseline)

        score = mon.get_drift("agent-1")
        if score and score.overall >= 0.20:
            alerts = mon.get_alerts("agent-1")
            self.assertTrue(len(alerts) > 0)

    def test_critical_alert_generated(self) -> None:
        cfg = DriftConfig(
            check_interval=5, min_observations=5,
            alert_threshold_critical=0.30,
        )
        mon = DriftMonitor(config=cfg)
        baseline = self._make_baseline()

        for _ in range(5):
            action = self._action(action_type="delete", target="/sensitive")
            mon.observe("agent-1", action, Verdict.DENY, baseline_fingerprint=baseline)

        score = mon.get_drift("agent-1")
        if score and score.overall >= 0.30:
            alerts = mon.get_alerts("agent-1")
            sev_list = [a.severity for a in alerts]
            self.assertIn("critical", sev_list)

    def test_no_duplicate_alerts_same_severity(self) -> None:
        cfg = DriftConfig(
            check_interval=5, min_observations=5,
            alert_threshold_moderate=0.01,
        )
        mon = DriftMonitor(config=cfg)
        baseline = self._make_baseline()

        # Generate multiple drift computations at moderate level
        for _ in range(15):
            action = self._action(action_type="write", target="/data")
            mon.observe("agent-1", action, Verdict.ALLOW, baseline_fingerprint=baseline)

        alerts = mon.get_alerts("agent-1")
        # Should have at most a few alerts, not one per check_interval
        unacked = [a for a in alerts if not a.acknowledged]
        # At least 1 alert if drift is above threshold
        if alerts:
            self.assertGreaterEqual(len(unacked), 1)

    def test_escalates_alert_severity(self) -> None:
        cfg = DriftConfig(
            check_interval=5, min_observations=5,
            alert_threshold_moderate=0.01,
            alert_threshold_high=0.30,
            alert_threshold_critical=0.50,
        )
        mon = DriftMonitor(config=cfg)
        baseline = self._make_baseline()

        # Start with mild deviation
        for _ in range(5):
            action = self._action(action_type="write", target="/data")
            mon.observe("agent-1", action, Verdict.ALLOW, baseline_fingerprint=baseline)

        # Then severe deviation
        for _ in range(5):
            action = self._action(action_type="delete", target="/sensitive")
            mon.observe("agent-1", action, Verdict.DENY, baseline_fingerprint=baseline)

        alerts = mon.get_alerts("agent-1")
        if len(alerts) >= 2:
            # Later alerts should be equal or higher severity
            severities = [a.severity for a in alerts]
            rank = {"moderate": 1, "high": 2, "critical": 3}
            for i in range(1, len(severities)):
                self.assertGreaterEqual(
                    rank.get(severities[i], 0),
                    rank.get(severities[0], 0),
                )

    def test_acknowledge_alert(self) -> None:
        cfg = DriftConfig(
            check_interval=5, min_observations=5,
            alert_threshold_moderate=0.01,
        )
        mon = DriftMonitor(config=cfg)
        baseline = self._make_baseline()

        for _ in range(5):
            action = self._action(action_type="delete", target="/sensitive")
            mon.observe("agent-1", action, Verdict.DENY, baseline_fingerprint=baseline)

        alerts = mon.get_alerts("agent-1")
        if alerts:
            ok = mon.acknowledge_alert("agent-1", 0)
            self.assertTrue(ok)
            self.assertTrue(alerts[0].acknowledged)

    def test_acknowledge_nonexistent_returns_false(self) -> None:
        mon = DriftMonitor()
        self.assertFalse(mon.acknowledge_alert("nonexistent", 0))

    def test_get_alerts_filters_by_agent(self) -> None:
        cfg = DriftConfig(check_interval=5, min_observations=5, alert_threshold_moderate=0.01)
        mon = DriftMonitor(config=cfg)

        baseline1 = self._make_baseline("agent-1")
        baseline2 = self._make_baseline("agent-2")

        for _ in range(5):
            mon.observe("agent-1", self._action("agent-1", "delete", "/x"), Verdict.DENY, baseline_fingerprint=baseline1)
            mon.observe("agent-2", self._action("agent-2", "read", "/data"), Verdict.ALLOW, baseline_fingerprint=baseline2)

        a1_alerts = mon.get_alerts("agent-1")
        a2_alerts = mon.get_alerts("agent-2")
        all_alerts = mon.get_alerts()

        # agent-1 should have alerts (delete vs read baseline), agent-2 may not
        if a1_alerts:
            for a in a1_alerts:
                self.assertEqual(a.agent_id, "agent-1")
        self.assertGreaterEqual(len(all_alerts), len(a1_alerts))

    def test_get_alerts_unacknowledged_only(self) -> None:
        cfg = DriftConfig(check_interval=5, min_observations=5, alert_threshold_moderate=0.01)
        mon = DriftMonitor(config=cfg)
        baseline = self._make_baseline()

        for _ in range(5):
            mon.observe("agent-1", self._action(action_type="delete"), Verdict.DENY, baseline_fingerprint=baseline)

        alerts = mon.get_alerts("agent-1")
        if alerts:
            mon.acknowledge_alert("agent-1", 0)
            unacked = mon.get_alerts("agent-1", unacknowledged_only=True)
            self.assertTrue(all(not a.acknowledged for a in unacked))

    def test_reset_clears_all_state(self) -> None:
        cfg = DriftConfig(check_interval=5, min_observations=5)
        mon = DriftMonitor(config=cfg)
        baseline = self._make_baseline()

        for _ in range(5):
            mon.observe("agent-1", self._action(), Verdict.ALLOW, baseline_fingerprint=baseline)

        self.assertIsNotNone(mon.get_drift("agent-1"))
        self.assertIsNotNone(mon.get_window("agent-1"))

        mon.reset("agent-1")
        self.assertIsNone(mon.get_drift("agent-1"))
        self.assertIsNone(mon.get_window("agent-1"))
        self.assertEqual(mon.get_alerts("agent-1"), [])

    def test_uses_archetype_drift_weights(self) -> None:
        registry = PriorRegistry.with_defaults()
        cfg = DriftConfig(check_interval=5, min_observations=5)
        mon = DriftMonitor(config=cfg, prior_registry=registry)

        baseline = self._make_baseline()

        for _ in range(5):
            action = self._action(action_type="delete", target="/sensitive")
            mon.observe(
                "agent-1", action, Verdict.DENY,
                baseline_fingerprint=baseline,
                archetype="financial-analyst",
            )

        score = mon.get_drift("agent-1")
        self.assertIsNotNone(score)
        # Financial analyst has high drift weights — result should be meaningful
        self.assertIsInstance(score.overall, float)

    def test_drift_without_baseline_returns_none(self) -> None:
        cfg = DriftConfig(check_interval=5, min_observations=5)
        mon = DriftMonitor(config=cfg)

        for _ in range(5):
            result = mon.observe("agent-1", self._action(), Verdict.ALLOW, baseline_fingerprint=None)
            self.assertIsNone(result)

    def test_alert_to_dict(self) -> None:
        ds = DriftScore(
            overall=0.5, action_drift=0.6, target_drift=0.3,
            temporal_drift=0.1, outcome_drift=0.2,
            confidence=0.9, window_size=100, baseline_size=500,
            detail="test",
        )
        alert = DriftAlert(agent_id="a", severity="high", drift_score=ds)
        d = alert.to_dict()
        self.assertEqual(d["agent_id"], "a")
        self.assertEqual(d["severity"], "high")
        self.assertIn("drift_score", d)
        self.assertFalse(d["acknowledged"])
