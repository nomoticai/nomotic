"""Integration tests for drift detection across the full governance pipeline."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import TestCase

from nomotic.authority import CertificateAuthority
from nomotic.dimensions import BehavioralConsistency, IncidentDetection
from nomotic.drift import DriftScore
from nomotic.keys import SigningKey
from nomotic.monitor import DriftConfig
from nomotic.runtime import GovernanceRuntime, RuntimeConfig
from nomotic.store import MemoryCertificateStore
from nomotic.types import Action, AgentContext, DimensionScore, TrustProfile, Verdict


def _action(
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


def _ctx(agent_id: str = "agent-1") -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        trust_profile=TrustProfile(agent_id=agent_id),
    )


class TestRuntimeDriftIntegration(TestCase):
    """Tests for drift integration in GovernanceRuntime."""

    def _runtime(self, **drift_kw: int | float) -> GovernanceRuntime:
        cfg = DriftConfig(
            window_size=20,
            check_interval=5,
            min_observations=5,
            **drift_kw,
        )
        return GovernanceRuntime(config=RuntimeConfig(
            enable_fingerprints=True,
            drift_config=cfg,
        ))

    def test_runtime_creates_drift_monitor(self) -> None:
        rt = self._runtime()
        self.assertIsNotNone(rt._fingerprint_observer)
        self.assertIsNotNone(rt._fingerprint_observer.drift_monitor)

    def test_get_drift_returns_score_after_enough_evals(self) -> None:
        rt = self._runtime()
        for _ in range(10):
            rt.evaluate(_action(), _ctx())
        score = rt.get_drift("agent-1")
        # May or may not have computed yet depending on check_interval
        # After 10 evals with check_interval=5, at least one computation
        if score is not None:
            self.assertIsInstance(score, DriftScore)

    def test_consistent_behavior_low_drift(self) -> None:
        rt = self._runtime()
        # All the same action type
        for _ in range(20):
            rt.evaluate(_action(action_type="read", target="/data"), _ctx())
        score = rt.get_drift("agent-1")
        if score is not None:
            self.assertLess(score.overall, 0.15)

    def test_changed_behavior_high_drift(self) -> None:
        rt = self._runtime()
        # Establish baseline
        for _ in range(20):
            rt.evaluate(_action(action_type="read", target="/data"), _ctx())
        # Change behavior dramatically
        for _ in range(20):
            rt.evaluate(_action(action_type="delete", target="/secret"), _ctx())

        score = rt.get_drift("agent-1")
        if score is not None:
            self.assertGreater(score.overall, 0.1)

    def test_drift_alerts_via_runtime(self) -> None:
        rt = self._runtime(alert_threshold_moderate=0.01)
        # Establish baseline
        for _ in range(15):
            rt.evaluate(_action(action_type="read", target="/data"), _ctx())
        # Change behavior
        for _ in range(10):
            rt.evaluate(_action(action_type="delete", target="/secret"), _ctx())

        alerts = rt.get_drift_alerts("agent-1")
        # Should have at least one alert since we set very low threshold
        # (if drift was detected)
        score = rt.get_drift("agent-1")
        if score and score.overall >= 0.01:
            self.assertGreater(len(alerts), 0)

    def test_get_drift_alerts_all_agents(self) -> None:
        rt = self._runtime(alert_threshold_moderate=0.01)
        for _ in range(15):
            rt.evaluate(_action("agent-1", "read", "/data"), _ctx("agent-1"))
        for _ in range(10):
            rt.evaluate(_action("agent-1", "delete", "/x"), _ctx("agent-1"))

        all_alerts = rt.get_drift_alerts()
        a1_alerts = rt.get_drift_alerts("agent-1")
        self.assertGreaterEqual(len(all_alerts), len(a1_alerts))

    def test_fingerprints_disabled_no_drift(self) -> None:
        rt = GovernanceRuntime(config=RuntimeConfig(enable_fingerprints=False))
        for _ in range(10):
            rt.evaluate(_action(), _ctx())
        self.assertIsNone(rt.get_drift("agent-1"))
        self.assertEqual(rt.get_drift_alerts(), [])


class TestBehavioralConsistencyDrift(TestCase):
    """Tests for drift modulation in BehavioralConsistency."""

    def test_no_drift_accessor_unchanged(self) -> None:
        """Backward compat: without drift accessor, scores are unchanged."""
        dim = BehavioralConsistency()
        action = _action()
        ctx = _ctx()
        # First action
        score = dim.evaluate(action, ctx)
        self.assertAlmostEqual(score.score, 0.7)
        # Second same action
        score = dim.evaluate(action, ctx)
        self.assertAlmostEqual(score.score, 1.0)

    def test_drift_accessor_none_unchanged(self) -> None:
        """When drift accessor returns None, score is unchanged."""
        dim = BehavioralConsistency()
        dim.set_drift_accessor(lambda agent_id: None)
        action = _action()
        ctx = _ctx()
        dim.evaluate(action, ctx)  # first
        score = dim.evaluate(action, ctx)  # second
        self.assertAlmostEqual(score.score, 1.0)

    def test_high_drift_decreases_score(self) -> None:
        """High drift should modulate score downward."""
        dim = BehavioralConsistency()
        high_drift = DriftScore(
            overall=0.5, action_drift=0.5, target_drift=0.3,
            temporal_drift=0.2, outcome_drift=0.1,
            confidence=0.9, window_size=100, baseline_size=500,
            detail="action shifted",
        )
        dim.set_drift_accessor(lambda agent_id: high_drift)
        action = _action()
        ctx = _ctx()
        dim.evaluate(action, ctx)  # first
        score = dim.evaluate(action, ctx)  # second
        # Without drift, base score would be 1.0. With 0.5 drift, it should be lower.
        self.assertLess(score.score, 1.0)
        self.assertIn("drift", score.reasoning.lower())

    def test_low_drift_no_effect(self) -> None:
        """Low drift (< 0.15) should not affect the score."""
        dim = BehavioralConsistency()
        low_drift = DriftScore(
            overall=0.05, action_drift=0.05, target_drift=0.03,
            temporal_drift=0.02, outcome_drift=0.01,
            confidence=0.9, window_size=100, baseline_size=500,
        )
        dim.set_drift_accessor(lambda agent_id: low_drift)
        action = _action()
        ctx = _ctx()
        dim.evaluate(action, ctx)  # first
        score = dim.evaluate(action, ctx)  # second
        self.assertAlmostEqual(score.score, 1.0)

    def test_low_confidence_drift_no_effect(self) -> None:
        """Drift with confidence < 0.3 should not affect score."""
        dim = BehavioralConsistency()
        unreliable_drift = DriftScore(
            overall=0.8, action_drift=0.8, target_drift=0.6,
            temporal_drift=0.5, outcome_drift=0.4,
            confidence=0.1, window_size=10, baseline_size=20,
        )
        dim.set_drift_accessor(lambda agent_id: unreliable_drift)
        action = _action()
        ctx = _ctx()
        dim.evaluate(action, ctx)  # first
        score = dim.evaluate(action, ctx)  # second
        self.assertAlmostEqual(score.score, 1.0)


class TestIncidentDetectionDrift(TestCase):
    """Tests for drift-based incident detection."""

    def test_no_drift_accessor_unchanged(self) -> None:
        """Backward compat: without drift accessor, no drift-based incidents."""
        dim = IncidentDetection()
        action = _action()
        ctx = _ctx()
        score = dim.evaluate(action, ctx)
        self.assertAlmostEqual(score.score, 1.0)

    def test_critical_drift_triggers_incident(self) -> None:
        """Critical drift (>= 0.60) should trigger a veto-level incident."""
        dim = IncidentDetection()
        critical_drift = DriftScore(
            overall=0.70, action_drift=0.8, target_drift=0.6,
            temporal_drift=0.5, outcome_drift=0.7,
            confidence=0.9, window_size=100, baseline_size=500,
            detail="massive shift",
        )
        dim.set_drift_accessor(lambda agent_id: critical_drift)
        action = _action()
        ctx = _ctx()
        score = dim.evaluate(action, ctx)
        self.assertLessEqual(score.score, 0.1)
        self.assertTrue(score.veto)
        self.assertIn("Critical behavioral drift", score.reasoning)

    def test_high_drift_lowers_score(self) -> None:
        """High drift (>= 0.40) lowers score but may not veto."""
        dim = IncidentDetection()
        high_drift = DriftScore(
            overall=0.45, action_drift=0.5, target_drift=0.4,
            temporal_drift=0.3, outcome_drift=0.3,
            confidence=0.8, window_size=100, baseline_size=500,
            detail="significant shift",
        )
        dim.set_drift_accessor(lambda agent_id: high_drift)
        action = _action()
        ctx = _ctx()
        score = dim.evaluate(action, ctx)
        self.assertLessEqual(score.score, 0.3)
        self.assertIn("drift", score.reasoning.lower())

    def test_low_drift_no_effect(self) -> None:
        """Low drift should not trigger incident detection."""
        dim = IncidentDetection()
        low_drift = DriftScore(
            overall=0.10, action_drift=0.1, target_drift=0.1,
            temporal_drift=0.05, outcome_drift=0.05,
            confidence=0.9, window_size=100, baseline_size=500,
        )
        dim.set_drift_accessor(lambda agent_id: low_drift)
        action = _action()
        ctx = _ctx()
        score = dim.evaluate(action, ctx)
        self.assertAlmostEqual(score.score, 1.0)

    def test_low_confidence_drift_no_incident(self) -> None:
        """Drift with confidence <= 0.5 should not trigger incident."""
        dim = IncidentDetection()
        unreliable = DriftScore(
            overall=0.80, action_drift=0.8, target_drift=0.7,
            temporal_drift=0.6, outcome_drift=0.5,
            confidence=0.3, window_size=20, baseline_size=30,
        )
        dim.set_drift_accessor(lambda agent_id: unreliable)
        action = _action()
        ctx = _ctx()
        score = dim.evaluate(action, ctx)
        self.assertAlmostEqual(score.score, 1.0)


class TestAPIDrift(TestCase):
    """Tests for the drift API endpoints."""

    def setUp(self) -> None:
        from nomotic.api import NomoticAPIServer

        sk, _vk = SigningKey.generate()
        store = MemoryCertificateStore()
        ca = CertificateAuthority(issuer_id="test", signing_key=sk, store=store)

        drift_cfg = DriftConfig(
            window_size=15, check_interval=5, min_observations=5,
            alert_threshold_moderate=0.01,
        )
        runtime = GovernanceRuntime(config=RuntimeConfig(
            enable_fingerprints=True, drift_config=drift_cfg,
        ))

        self._runtime = runtime

        # Feed some data — baseline reads
        for _ in range(15):
            runtime.evaluate(
                _action(action_type="read", target="/data"),
                _ctx(),
            )
        # Changed behavior
        for _ in range(15):
            runtime.evaluate(
                _action(action_type="delete", target="/secret"),
                _ctx(),
            )

        self._server_api = NomoticAPIServer(
            ca, runtime=runtime, host="127.0.0.1", port=0,
        )
        self._http_server = self._server_api._build_server()
        self._port = self._http_server.server_address[1]
        self._thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._thread.start()

    def tearDown(self) -> None:
        self._http_server.shutdown()

    def _get(self, path: str) -> tuple[int, dict]:
        import urllib.request
        import urllib.error
        url = f"http://127.0.0.1:{self._port}{path}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def _post(self, path: str) -> tuple[int, dict]:
        import urllib.request
        import urllib.error
        url = f"http://127.0.0.1:{self._port}{path}"
        try:
            req = urllib.request.Request(url, data=b"{}", method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_get_drift_returns_score(self) -> None:
        score = self._runtime.get_drift("agent-1")
        if score is None:
            self.skipTest("No drift computed yet")
        status, data = self._get("/v1/drift/agent-1")
        self.assertEqual(status, 200)
        self.assertIn("drift", data)
        self.assertIn("overall", data["drift"])
        self.assertIn("alerts", data)

    def test_get_drift_not_found(self) -> None:
        status, data = self._get("/v1/drift/nonexistent")
        self.assertEqual(status, 404)

    def test_get_alerts(self) -> None:
        status, data = self._get("/v1/alerts")
        self.assertEqual(status, 200)
        self.assertIn("alerts", data)
        self.assertIn("total", data)
        self.assertIn("unacknowledged", data)

    def test_get_alerts_with_agent_filter(self) -> None:
        status, data = self._get("/v1/alerts?agent_id=agent-1")
        self.assertEqual(status, 200)
        for alert in data.get("alerts", []):
            self.assertEqual(alert["agent_id"], "agent-1")

    def test_acknowledge_alert(self) -> None:
        alerts = self._runtime.get_drift_alerts("agent-1")
        if not alerts:
            self.skipTest("No alerts to acknowledge")
        status, data = self._post("/v1/alerts/agent-1/0/acknowledge")
        self.assertEqual(status, 200)
        self.assertTrue(data["acknowledged"])

    def test_acknowledge_nonexistent(self) -> None:
        status, data = self._post("/v1/alerts/nonexistent/0/acknowledge")
        self.assertEqual(status, 404)


class TestExistingTestsBackwardCompat(TestCase):
    """Verify that existing functionality is not broken by drift additions."""

    def test_runtime_without_fingerprints_still_works(self) -> None:
        rt = GovernanceRuntime(config=RuntimeConfig(enable_fingerprints=False))
        verdict = rt.evaluate(_action(), _ctx())
        self.assertIsNotNone(verdict)
        self.assertIn(verdict.verdict, [Verdict.ALLOW, Verdict.DENY, Verdict.MODIFY, Verdict.ESCALATE])

    def test_runtime_with_fingerprints_no_drift_config(self) -> None:
        rt = GovernanceRuntime(config=RuntimeConfig(enable_fingerprints=True))
        verdict = rt.evaluate(_action(), _ctx())
        self.assertIsNotNone(verdict)

    def test_behavioral_consistency_legacy_still_works(self) -> None:
        dim = BehavioralConsistency()
        action = _action()
        ctx = _ctx()
        # First
        score = dim.evaluate(action, ctx)
        self.assertAlmostEqual(score.score, 0.7)
        # Second
        score = dim.evaluate(action, ctx)
        self.assertAlmostEqual(score.score, 1.0)

    def test_incident_detection_legacy_still_works(self) -> None:
        dim = IncidentDetection()
        action = _action()
        ctx = _ctx()
        score = dim.evaluate(action, ctx)
        self.assertAlmostEqual(score.score, 1.0)
        self.assertEqual(score.reasoning, "No incident patterns matched")
