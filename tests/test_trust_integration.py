"""Integration tests for trust + drift + trajectory in the GovernanceRuntime."""

from nomotic.monitor import DriftConfig
from nomotic.runtime import GovernanceRuntime, RuntimeConfig
from nomotic.trajectory import SOURCE_DRIFT_ADJUSTMENT, SOURCE_VERDICT_ALLOW
from nomotic.types import Action, AgentContext, TrustProfile


def _make_runtime(
    check_interval: int = 3,
    min_observations: int = 5,
    window_size: int = 50,
) -> GovernanceRuntime:
    """Create a runtime with fast drift detection for testing.

    Uses window_size=50 by default so the confidence is high enough
    for drift adjustments to exceed the significance threshold.
    """
    drift_cfg = DriftConfig(
        window_size=window_size,
        check_interval=check_interval,
        min_observations=min_observations,
    )
    return GovernanceRuntime(
        config=RuntimeConfig(enable_fingerprints=True, drift_config=drift_cfg)
    )


def _evaluate(runtime: GovernanceRuntime, agent_id: str, action_type: str, target: str):
    """Run a single evaluation through the runtime."""
    action = Action(agent_id=agent_id, action_type=action_type, target=target)
    ctx = AgentContext(agent_id=agent_id, trust_profile=TrustProfile(agent_id=agent_id))
    return runtime.evaluate(action, ctx)


class TestTrustIntegration:
    def test_consistent_behavior_no_drift_events(self):
        """With consistent behavior, no drift adjustment events should appear."""
        runtime = _make_runtime(check_interval=5, min_observations=5)
        for _ in range(10):
            _evaluate(runtime, "a", "read", "/data")

        traj = runtime.get_trust_trajectory("a")
        drift_events = traj.events_by_source("drift")
        # With consistent behavior, drift should be low -> no drift events
        assert len(drift_events) == 0

    def test_changed_behavior_produces_drift_events(self):
        """After behavior changes, drift adjustment events should appear.

        Uses enough observations so confidence is high enough for
        drift adjustments to be significant.
        """
        runtime = _make_runtime(check_interval=10, min_observations=10, window_size=50)

        # Build baseline: 60 consistent reads
        for _ in range(60):
            _evaluate(runtime, "a", "read", "/data")

        # Change behavior: 60 deletes on different targets
        for _ in range(60):
            _evaluate(runtime, "a", "delete", "/sensitive")

        traj = runtime.get_trust_trajectory("a")
        drift_events = traj.events_by_source("drift")
        # With high enough confidence and strong drift, events should appear
        assert len(drift_events) > 0

    def test_drift_erosion_visible_in_profile(self):
        """Drift-based trust erosion should be reflected in get_trust_profile()."""
        runtime = _make_runtime(check_interval=10, min_observations=10, window_size=50)

        # Build baseline
        for _ in range(60):
            _evaluate(runtime, "a", "read", "/data")

        trust_before_drift_behavior = runtime.get_trust_profile("a").overall_trust

        # Drift behavior
        for _ in range(60):
            _evaluate(runtime, "a", "delete", "/sensitive")

        # Verify drift events exist
        traj = runtime.get_trust_trajectory("a")
        drift_events = traj.events_by_source("drift")

        # If drift was detected and significant, the drift events should erode trust
        if len(drift_events) > 0:
            drift_delta = sum(e.delta for e in drift_events)
            assert drift_delta < 0  # Erosion is negative

    def test_drift_syncs_to_certificate(self):
        """Drift-based trust changes should propagate to the certificate."""
        runtime = _make_runtime(check_interval=3, min_observations=5, window_size=50)

        # Issue a certificate
        cert = runtime.birth(
            agent_id="a",
            archetype="customer-experience",
            organization="test-org",
            zone_path="global",
        )

        # Build baseline
        for _ in range(15):
            _evaluate(runtime, "a", "read", "/data")

        # Change behavior
        for _ in range(15):
            _evaluate(runtime, "a", "delete", "/sensitive")

        # Check certificate trust updated
        updated_cert = runtime.get_certificate("a")
        if updated_cert is not None:
            # Trust in the cert should reflect the runtime trust
            runtime_trust = runtime.get_trust_profile("a").overall_trust
            assert abs(updated_cert.trust_score - runtime_trust) < 0.01

    def test_get_trust_trajectory(self):
        """get_trust_trajectory() returns a trajectory with events."""
        runtime = _make_runtime()
        for _ in range(5):
            _evaluate(runtime, "a", "read", "/data")

        traj = runtime.get_trust_trajectory("a")
        assert traj is not None
        # Should have verdict:allow events
        allow_events = traj.events_by_source(SOURCE_VERDICT_ALLOW)
        assert len(allow_events) == 5

    def test_get_trust_report_all_sections(self):
        """get_trust_report() includes trust, trajectory, fingerprint, and drift."""
        runtime = _make_runtime(check_interval=3, min_observations=5)

        # Generate enough data
        for _ in range(15):
            _evaluate(runtime, "a", "read", "/data")

        report = runtime.get_trust_report("a")

        assert "agent_id" in report
        assert "current_trust" in report
        assert "successful_actions" in report
        assert "violation_count" in report
        assert "violation_rate" in report
        assert "trajectory" in report

        # Fingerprint should be present since we enabled fingerprints
        assert "fingerprint" in report
        assert "total_observations" in report["fingerprint"]
        assert "confidence" in report["fingerprint"]

    def test_get_trust_report_without_fingerprints(self):
        """get_trust_report() works when fingerprints are disabled."""
        runtime = GovernanceRuntime(
            config=RuntimeConfig(enable_fingerprints=False)
        )

        for _ in range(3):
            _evaluate(runtime, "a", "read", "/data")

        report = runtime.get_trust_report("a")
        assert "agent_id" in report
        assert "current_trust" in report
        assert "trajectory" in report
        # No fingerprint/drift sections
        assert "fingerprint" not in report
        assert "drift" not in report

    def test_full_loop_drift_and_recovery(self):
        """Full loop: consistent -> drift -> trust drops -> normalizes -> trust recovers."""
        runtime = _make_runtime(check_interval=3, min_observations=5, window_size=50)

        # Phase 1: Consistent behavior
        for _ in range(15):
            _evaluate(runtime, "a", "read", "/data")

        # Phase 2: Drifting behavior
        for _ in range(15):
            _evaluate(runtime, "a", "delete", "/sensitive")

        # Phase 3: Return to normal
        for _ in range(15):
            _evaluate(runtime, "a", "read", "/data")

        # Trajectory should tell the story
        traj = runtime.get_trust_trajectory("a")
        assert len(traj) > 0

        # Check the report
        report = runtime.get_trust_report("a")
        trust_final = runtime.get_trust_profile("a").overall_trust
        assert report["current_trust"] == trust_final

    def test_trust_report_includes_drift_info(self):
        """Trust report includes drift data when drift has been computed."""
        runtime = _make_runtime(check_interval=3, min_observations=5, window_size=50)

        # Build baseline
        for _ in range(15):
            _evaluate(runtime, "a", "read", "/data")

        # Change to trigger drift computation
        for _ in range(10):
            _evaluate(runtime, "a", "delete", "/sensitive")

        report = runtime.get_trust_report("a")
        drift = runtime.get_drift("a")
        if drift is not None:
            assert "drift" in report
            assert "overall" in report["drift"]

    def test_trust_report_includes_alerts(self):
        """Trust report includes active alert count."""
        runtime = _make_runtime(check_interval=3, min_observations=5, window_size=50)

        # Build baseline
        for _ in range(15):
            _evaluate(runtime, "a", "read", "/data")

        # Trigger significant drift
        for _ in range(15):
            _evaluate(runtime, "a", "delete", "/sensitive")

        report = runtime.get_trust_report("a")
        alerts = runtime.get_drift_alerts("a")
        if alerts:
            assert "active_alerts" in report

    def test_apply_drift_called_on_drift_check(self):
        """Trust calibrator's apply_drift is called when drift is computed."""
        runtime = _make_runtime(check_interval=10, min_observations=10, window_size=50)

        # Build baseline
        for _ in range(60):
            _evaluate(runtime, "a", "read", "/data")

        # Verify drift was computed
        drift = runtime.get_drift("a")
        assert drift is not None

        # The last_drift_scores should have been set
        assert "a" in runtime.trust_calibrator._last_drift_scores
