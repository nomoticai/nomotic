"""Tests for human oversight drift detection.

Tests HumanInteractionProfile, HumanDriftCalculator, HumanDriftMonitor,
and HumanAuditStore.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest import TestCase

from nomotic.human_drift import (
    HumanAuditStore,
    HumanDriftCalculator,
    HumanDriftMonitor,
    HumanDriftResult,
    HumanInteractionEvent,
    HumanInteractionProfile,
)


# ── Helpers ────────────────────────────────────────────────────────────


_BASE_TIME = 1700000000.0


def _make_event(
    *,
    duration: float = 30.0,
    decision: str = "approved",
    event_type: str = "approval",
    reviewer_id: str = "reviewer1",
    agent_id: str = "agent1",
    rationale: str = "Looks good",
    context_viewed: bool = True,
    timestamp: float | None = None,
    index: int = 0,
) -> HumanInteractionEvent:
    return HumanInteractionEvent(
        timestamp=timestamp if timestamp is not None else _BASE_TIME + index * 60,
        reviewer_id=reviewer_id,
        agent_id=agent_id,
        action_id=f"action-{index}",
        event_type=event_type,
        decision=decision,
        review_duration_seconds=duration,
        rationale=rationale,
        rationale_depth=len(rationale.split()) if rationale else 0,
        context_viewed=context_viewed,
        modifications=[],
    )


def _make_profile(
    *,
    reviewer_id: str = "reviewer1",
    mean_duration: float = 30.0,
    approval_rate: float = 0.7,
    denial_rate: float = 0.2,
    modification_rate: float = 0.05,
    deferral_rate: float = 0.05,
    override_rate: float = 0.05,
    rationale_provided_rate: float = 0.8,
    context_view_rate: float = 0.9,
    mean_rationale_depth: float = 5.0,
    total: int = 200,
    interactions_per_hour: float = 10.0,
) -> HumanInteractionProfile:
    return HumanInteractionProfile(
        reviewer_id=reviewer_id,
        mean_review_duration=mean_duration,
        median_review_duration=mean_duration,
        min_review_duration=mean_duration * 0.5,
        max_review_duration=mean_duration * 2.0,
        review_duration_stddev=mean_duration * 0.2,
        approval_rate=approval_rate,
        denial_rate=denial_rate,
        modification_rate=modification_rate,
        deferral_rate=deferral_rate,
        override_rate=override_rate,
        mean_rationale_depth=mean_rationale_depth,
        rationale_provided_rate=rationale_provided_rate,
        context_view_rate=context_view_rate,
        total_interactions=total,
        interactions_per_hour=interactions_per_hour,
        window_start=_BASE_TIME,
        window_end=_BASE_TIME + 3600,
    )


# ── Profile tests ─────────────────────────────────────────────────────


class TestHumanInteractionProfile(TestCase):
    def test_from_events(self) -> None:
        events = [_make_event(duration=30, decision="approved", index=i) for i in range(10)]
        events.append(_make_event(duration=30, decision="denied", index=10))
        profile = HumanInteractionProfile.from_events("reviewer1", events)
        self.assertEqual(profile.total_interactions, 11)
        self.assertAlmostEqual(profile.approval_rate, 10 / 11, places=2)

    def test_from_empty_events(self) -> None:
        profile = HumanInteractionProfile.from_events("reviewer1", [])
        self.assertEqual(profile.total_interactions, 0)
        self.assertEqual(profile.approval_rate, 0.0)

    def test_review_duration_stats(self) -> None:
        events = [
            _make_event(duration=10, index=0),
            _make_event(duration=20, index=1),
            _make_event(duration=30, index=2),
        ]
        profile = HumanInteractionProfile.from_events("reviewer1", events)
        self.assertAlmostEqual(profile.mean_review_duration, 20.0, places=1)
        self.assertAlmostEqual(profile.median_review_duration, 20.0, places=1)
        self.assertEqual(profile.min_review_duration, 10.0)
        self.assertEqual(profile.max_review_duration, 30.0)

    def test_rationale_metrics(self) -> None:
        events = [
            _make_event(rationale="Good work here", index=0),
            _make_event(rationale="", index=1),
            _make_event(rationale="Reviewed carefully and approved", index=2),
        ]
        profile = HumanInteractionProfile.from_events("reviewer1", events)
        # 2 out of 3 have rationale
        self.assertAlmostEqual(profile.rationale_provided_rate, 2 / 3, places=2)

    def test_context_view_rate(self) -> None:
        events = [
            _make_event(context_viewed=True, index=0),
            _make_event(context_viewed=False, index=1),
            _make_event(context_viewed=True, index=2),
            _make_event(context_viewed=True, index=3),
        ]
        profile = HumanInteractionProfile.from_events("reviewer1", events)
        self.assertAlmostEqual(profile.context_view_rate, 0.75, places=2)

    def test_override_rate(self) -> None:
        events = [
            _make_event(event_type="approval", index=0),
            _make_event(event_type="override", index=1),
            _make_event(event_type="approval", index=2),
            _make_event(event_type="approval", index=3),
        ]
        profile = HumanInteractionProfile.from_events("reviewer1", events)
        self.assertAlmostEqual(profile.override_rate, 0.25, places=2)

    def test_to_dict_roundtrip(self) -> None:
        events = [_make_event(index=i) for i in range(5)]
        profile = HumanInteractionProfile.from_events("reviewer1", events)
        d = profile.to_dict()
        self.assertEqual(d["reviewer_id"], "reviewer1")
        self.assertEqual(d["total_interactions"], 5)


# ── Calculator tests ───────────────────────────────────────────────────


class TestHumanDriftCalculator(TestCase):
    def test_no_drift(self) -> None:
        baseline = _make_profile(mean_duration=30, approval_rate=0.7)
        recent = _make_profile(mean_duration=28, approval_rate=0.72)
        result = HumanDriftCalculator().calculate(baseline, recent)
        self.assertEqual(result.drift_category, "stable")
        self.assertEqual(len(result.alerts), 0)

    def test_rubber_stamping_detected(self) -> None:
        baseline = _make_profile(mean_duration=30, approval_rate=0.7, total=200)
        recent = _make_profile(
            mean_duration=2, approval_rate=0.99, total=150,
            override_rate=0.0, rationale_provided_rate=0.05,
            context_view_rate=0.1,
        )
        result = HumanDriftCalculator().calculate(baseline, recent)
        self.assertIn(result.drift_category, ("severe", "critical"))
        self.assertTrue(any("rubber-stamping" in a for a in result.alerts))

    def test_timing_drop_alert(self) -> None:
        baseline = _make_profile(mean_duration=45)
        recent = _make_profile(mean_duration=3)
        result = HumanDriftCalculator().calculate(baseline, recent)
        self.assertGreater(result.timing_drift, 0.5)
        self.assertTrue(any("duration dropped" in a.lower() for a in result.alerts))

    def test_min_review_duration_alert(self) -> None:
        baseline = _make_profile(mean_duration=45)
        recent = _make_profile(mean_duration=2)
        result = HumanDriftCalculator().calculate(baseline, recent)
        self.assertTrue(any("below" in a.lower() and "minimum" in a.lower() for a in result.alerts))

    def test_zero_overrides_alert(self) -> None:
        baseline = _make_profile(override_rate=0.05, total=200)
        recent = _make_profile(override_rate=0.0, total=150)
        result = HumanDriftCalculator().calculate(baseline, recent)
        self.assertTrue(any("zero overrides" in a.lower() for a in result.alerts))

    def test_rationale_drop(self) -> None:
        baseline = _make_profile(rationale_provided_rate=0.8)
        recent = _make_profile(rationale_provided_rate=0.1)
        result = HumanDriftCalculator().calculate(baseline, recent)
        self.assertGreater(result.engagement_drift, 0.5)

    def test_context_view_drop(self) -> None:
        baseline = _make_profile(context_view_rate=0.9)
        recent = _make_profile(context_view_rate=0.2)
        result = HumanDriftCalculator().calculate(baseline, recent)
        self.assertGreater(result.engagement_drift, 0.5)
        self.assertTrue(any("context viewing" in a.lower() for a in result.alerts))

    def test_throughput_increase_alert(self) -> None:
        baseline = _make_profile(interactions_per_hour=10)
        recent = _make_profile(interactions_per_hour=35)
        result = HumanDriftCalculator().calculate(baseline, recent)
        self.assertGreater(result.throughput_drift, 0)
        self.assertTrue(any("throughput" in a.lower() for a in result.alerts))

    def test_overall_drift_weighting(self) -> None:
        """Overall drift is a weighted combination of components."""
        baseline = _make_profile(mean_duration=30, approval_rate=0.7)
        recent = _make_profile(mean_duration=30, approval_rate=0.7)
        result = HumanDriftCalculator().calculate(baseline, recent)
        self.assertAlmostEqual(result.overall_drift, 0.0, places=2)

    def test_drift_categories(self) -> None:
        calc = HumanDriftCalculator()

        # stable: overall < 0.1
        baseline = _make_profile(mean_duration=30, approval_rate=0.7)
        recent = _make_profile(mean_duration=29, approval_rate=0.71)
        result = calc.calculate(baseline, recent)
        self.assertEqual(result.drift_category, "stable")

    def test_to_dict(self) -> None:
        baseline = _make_profile()
        recent = _make_profile(mean_duration=2, approval_rate=0.99)
        result = HumanDriftCalculator().calculate(baseline, recent)
        d = result.to_dict()
        self.assertIn("reviewer_id", d)
        self.assertIn("overall_drift", d)
        self.assertIn("drift_category", d)
        self.assertIn("alerts", d)
        self.assertIsInstance(d["alerts"], list)


# ── Monitor tests ──────────────────────────────────────────────────────


class TestHumanDriftMonitor(TestCase):
    def test_needs_sufficient_data(self) -> None:
        monitor = HumanDriftMonitor(baseline_window=10, recent_window=5)
        # Only 5 events — not enough
        for i in range(5):
            result = monitor.record_event(_make_event(index=i))
            self.assertIsNone(result)

    def test_returns_none_when_no_drift(self) -> None:
        monitor = HumanDriftMonitor(baseline_window=10, recent_window=5)
        # Mix of decisions to avoid triggering approval rate alert
        decisions = ["approved", "approved", "denied", "approved", "approved"]
        # 10 baseline + 5 recent, same pattern
        for i in range(15):
            result = monitor.record_event(
                _make_event(
                    duration=30,
                    decision=decisions[i % len(decisions)],
                    rationale="Reviewed carefully",
                    context_viewed=True,
                    index=i,
                )
            )
        # No drift detected (same pattern), so result should be None
        self.assertIsNone(result)

    def test_detects_drift_after_baseline(self) -> None:
        monitor = HumanDriftMonitor(baseline_window=20, recent_window=10)
        # 20 careful reviews (baseline)
        for i in range(20):
            monitor.record_event(
                _make_event(duration=30, decision="approved", index=i)
            )
        # 10 rushed reviews (drift)
        result = None
        for i in range(10):
            result = monitor.record_event(
                _make_event(duration=1.5, decision="approved", index=20 + i)
            )

        self.assertIsNotNone(result)
        self.assertNotEqual(result.drift_category, "stable")

    def test_get_alerts(self) -> None:
        monitor = HumanDriftMonitor(baseline_window=20, recent_window=10)
        for i in range(20):
            monitor.record_event(_make_event(duration=30, index=i))
        for i in range(10):
            monitor.record_event(_make_event(duration=1.5, index=20 + i))

        alerts = monitor.get_alerts()
        self.assertGreater(len(alerts), 0)

    def test_get_profile(self) -> None:
        monitor = HumanDriftMonitor(baseline_window=10, recent_window=5)
        for i in range(15):
            monitor.record_event(_make_event(index=i))

        profile = monitor.get_profile("reviewer1")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.total_interactions, 10)

    def test_get_recent_profile(self) -> None:
        monitor = HumanDriftMonitor(baseline_window=10, recent_window=5)
        for i in range(15):
            monitor.record_event(_make_event(index=i))

        recent = monitor.get_recent_profile("reviewer1")
        self.assertIsNotNone(recent)
        self.assertEqual(recent.total_interactions, 5)

    def test_get_recent_profile_insufficient_data(self) -> None:
        monitor = HumanDriftMonitor(baseline_window=10, recent_window=5)
        for i in range(3):
            monitor.record_event(_make_event(index=i))

        recent = monitor.get_recent_profile("reviewer1")
        self.assertIsNone(recent)

    def test_get_reviewer_ids(self) -> None:
        monitor = HumanDriftMonitor(baseline_window=5, recent_window=3)
        for i in range(5):
            monitor.record_event(_make_event(reviewer_id="alice", index=i))
        for i in range(5):
            monitor.record_event(_make_event(reviewer_id="bob", index=i))

        ids = monitor.get_reviewer_ids()
        self.assertIn("alice", ids)
        self.assertIn("bob", ids)


# ── Audit store tests ─────────────────────────────────────────────────


class TestHumanAuditStore(TestCase):
    def test_append_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HumanAuditStore(Path(tmpdir))
            event = _make_event(index=0)
            store.append(event)

            events = store.query("reviewer1")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].reviewer_id, "reviewer1")

    def test_query_all_chronological(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HumanAuditStore(Path(tmpdir))
            for i in range(5):
                store.append(_make_event(index=i))

            events = store.query_all("reviewer1")
            self.assertEqual(len(events), 5)
            # Should be chronological
            for i in range(1, len(events)):
                self.assertGreaterEqual(events[i].timestamp, events[i - 1].timestamp)

    def test_verify_chain_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HumanAuditStore(Path(tmpdir))
            for i in range(3):
                store.append(_make_event(index=i))

            is_valid, count, message = store.verify_chain("reviewer1")
            self.assertTrue(is_valid)
            self.assertEqual(count, 3)
            self.assertIn("verified", message.lower())

    def test_verify_chain_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HumanAuditStore(Path(tmpdir))
            is_valid, count, message = store.verify_chain("nonexistent")
            self.assertTrue(is_valid)
            self.assertEqual(count, 0)

    def test_list_reviewers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HumanAuditStore(Path(tmpdir))
            store.append(_make_event(reviewer_id="alice", index=0))
            store.append(_make_event(reviewer_id="bob", index=1))

            reviewers = store.list_reviewers()
            self.assertIn("alice", reviewers)
            self.assertIn("bob", reviewers)

    def test_query_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HumanAuditStore(Path(tmpdir))
            for i in range(10):
                store.append(_make_event(index=i))

            events = store.query("reviewer1", limit=3)
            self.assertEqual(len(events), 3)

    def test_query_empty_reviewer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HumanAuditStore(Path(tmpdir))
            events = store.query("nonexistent")
            self.assertEqual(len(events), 0)


# ── Event tests ────────────────────────────────────────────────────────


class TestHumanInteractionEvent(TestCase):
    def test_to_dict(self) -> None:
        event = _make_event(index=0)
        d = event.to_dict()
        self.assertEqual(d["reviewer_id"], "reviewer1")
        self.assertEqual(d["decision"], "approved")

    def test_from_dict_roundtrip(self) -> None:
        event = _make_event(index=0)
        d = event.to_dict()
        restored = HumanInteractionEvent.from_dict(d)
        self.assertEqual(restored.reviewer_id, event.reviewer_id)
        self.assertEqual(restored.decision, event.decision)
        self.assertAlmostEqual(restored.review_duration_seconds, event.review_duration_seconds)
