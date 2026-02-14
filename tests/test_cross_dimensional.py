"""Tests for the cross-dimensional signal detection module.

Covers single-action signal detection, aggregate analysis over audit trails,
and integration/lifecycle behavior of the CrossDimensionalDetector.
"""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass, field
from typing import Any

from nomotic.cross_dimensional import (
    CROSS_DIMENSIONAL_PATTERNS,
    CrossDimensionalDetector,
    CrossDimensionalReport,
    CrossDimensionalSignal,
)
from nomotic.types import DimensionScore


# ── Helpers ────────────────────────────────────────────────────────────


def _ds(name: str, score: float) -> DimensionScore:
    """Shortcut to build a DimensionScore."""
    return DimensionScore(dimension_name=name, score=score)


def _all_high(score: float = 0.9) -> list[DimensionScore]:
    """Return dimension scores where every dimension is healthy."""
    names = [
        "scope_compliance", "authority_verification", "ethical_alignment",
        "stakeholder_impact", "incident_detection", "behavioral_consistency",
        "precedent_alignment", "cascading_impact", "isolation_integrity",
        "temporal_compliance", "transparency", "human_override",
    ]
    return [_ds(n, score) for n in names]


@dataclass
class _AuditRecord:
    """Minimal audit record for testing aggregate analysis."""
    dimension_scores: list[dict[str, Any]] = field(default_factory=list)
    trust_score: float | None = None
    agent_id: str = "agent-1"
    timestamp: float = field(default_factory=time.time)


class _MockAuditTrail:
    """In-memory audit trail with a query() interface matching the detector."""

    def __init__(self, records: list[_AuditRecord] | None = None) -> None:
        self._records = list(records) if records else []

    def query(
        self,
        since: float = 0.0,
        limit: int = 100000,
        agent_id: str | None = None,
    ) -> list[_AuditRecord]:
        out = [r for r in self._records if r.timestamp >= since]
        if agent_id is not None:
            out = [r for r in out if r.agent_id == agent_id]
        return out[:limit]


def _record_with_scores(scores: dict[str, float], **kwargs: Any) -> _AuditRecord:
    """Build an audit record from a {dimension_name: score} mapping."""
    ds = [{"name": k, "score": v} for k, v in scores.items()]
    return _AuditRecord(dimension_scores=ds, **kwargs)


# ── Signal Detection Tests (15) ───────────────────────────────────────


class TestSignalDetection(unittest.TestCase):
    """Single-action signal detection."""

    def setUp(self) -> None:
        self.detector = CrossDimensionalDetector()

    def test_discriminatory_compliance_detected(self) -> None:
        scores = [
            _ds("scope_compliance", 0.95),
            _ds("authority_verification", 0.9),
            _ds("ethical_alignment", 0.3),
            _ds("stakeholder_impact", 0.8),
        ]
        signals = self.detector.detect_signals(scores)
        types = [s.signal_type for s in signals]
        self.assertIn("discriminatory_compliance", types)

    def test_empathetic_exploitation_detected(self) -> None:
        scores = [
            _ds("incident_detection", 0.3),
            _ds("ethical_alignment", 0.85),
            _ds("behavioral_consistency", 0.4),
        ]
        signals = self.detector.detect_signals(scores)
        types = [s.signal_type for s in signals]
        self.assertIn("empathetic_exploitation", types)

    def test_invisible_wall_detected(self) -> None:
        scores = [
            _ds("scope_compliance", 0.98),
            _ds("precedent_alignment", 0.3),
            _ds("stakeholder_impact", 0.2),
        ]
        signals = self.detector.detect_signals(scores)
        types = [s.signal_type for s in signals]
        self.assertIn("invisible_wall", types)

    def test_trust_authority_mismatch_detected(self) -> None:
        scores = [
            _ds("authority_verification", 0.9),
            _ds("behavioral_consistency", 0.3),
        ]
        signals = self.detector.detect_signals(scores, trust_state=0.2)
        types = [s.signal_type for s in signals]
        self.assertIn("trust_authority_mismatch", types)

    def test_cascade_without_isolation_detected(self) -> None:
        scores = [
            _ds("cascading_impact", 0.3),
            _ds("isolation_integrity", 0.5),
        ]
        signals = self.detector.detect_signals(scores)
        types = [s.signal_type for s in signals]
        self.assertIn("cascade_without_isolation", types)

    def test_temporal_security_gap_detected(self) -> None:
        scores = [
            _ds("temporal_compliance", 0.5),
            _ds("incident_detection", 0.4),
        ]
        signals = self.detector.detect_signals(scores)
        types = [s.signal_type for s in signals]
        self.assertIn("temporal_security_gap", types)

    def test_opacity_under_pressure_detected(self) -> None:
        scores = [
            _ds("transparency", 0.3),
            _ds("stakeholder_impact", 0.3),
        ]
        signals = self.detector.detect_signals(scores)
        types = [s.signal_type for s in signals]
        self.assertIn("opacity_under_pressure", types)

    def test_precedent_break_under_drift_detected(self) -> None:
        scores = [
            _ds("precedent_alignment", 0.3),
            _ds("behavioral_consistency", 0.4),
        ]
        signals = self.detector.detect_signals(scores)
        types = [s.signal_type for s in signals]
        self.assertIn("precedent_break_under_drift", types)

    def test_no_signals_when_all_dimensions_high(self) -> None:
        signals = self.detector.detect_signals(_all_high(0.9))
        self.assertEqual(signals, [])

    def test_no_signals_when_conditions_partially_met(self) -> None:
        """Scores sit right at threshold boundaries without crossing."""
        scores = [
            _ds("scope_compliance", 0.81),
            _ds("authority_verification", 0.81),
            _ds("ethical_alignment", 0.5),   # exactly 0.5, not < 0.5
            _ds("stakeholder_impact", 0.5),
        ]
        signals = self.detector.detect_signals(scores)
        types = [s.signal_type for s in signals]
        self.assertNotIn("discriminatory_compliance", types)

    def test_multiple_signals_from_same_action(self) -> None:
        """Scores crafted to trigger at least two patterns simultaneously."""
        scores = [
            _ds("scope_compliance", 0.99),
            _ds("authority_verification", 0.9),
            _ds("ethical_alignment", 0.3),
            _ds("stakeholder_impact", 0.2),
            _ds("precedent_alignment", 0.3),
            _ds("behavioral_consistency", 0.4),
            _ds("transparency", 0.3),
        ]
        signals = self.detector.detect_signals(scores)
        types = {s.signal_type for s in signals}
        self.assertTrue(
            len(types) >= 2,
            f"Expected at least 2 distinct signal types, got {types}",
        )

    def test_custom_pattern_detection(self) -> None:
        custom = {
            "name": "custom_test_pattern",
            "dimensions": ["scope_compliance"],
            "severity": "warning",
            "description": "Test pattern",
            "recommendation": "Fix it",
        }
        self.detector.add_pattern(custom)
        # Custom pattern has no evaluation logic in _evaluate_pattern, so it
        # returns False by default. Verify it appears in list but does not fire.
        signals = self.detector.detect_signals([_ds("scope_compliance", 0.1)])
        types = [s.signal_type for s in signals]
        self.assertNotIn("custom_test_pattern", types)

    def test_pattern_listing_includes_builtin_and_custom(self) -> None:
        custom = {
            "name": "my_custom",
            "dimensions": ["transparency"],
            "severity": "info",
        }
        self.detector.add_pattern(custom)
        names = [p["name"] for p in self.detector.list_patterns()]
        self.assertIn("discriminatory_compliance", names)
        self.assertIn("my_custom", names)

    def test_signal_severity_matches_pattern(self) -> None:
        scores = [
            _ds("incident_detection", 0.2),
            _ds("ethical_alignment", 0.9),
            _ds("behavioral_consistency", 0.3),
        ]
        signals = self.detector.detect_signals(scores)
        emp = [s for s in signals if s.signal_type == "empathetic_exploitation"]
        self.assertEqual(len(emp), 1)
        self.assertEqual(emp[0].severity, "critical")

    def test_signal_includes_recommendation(self) -> None:
        scores = [
            _ds("cascading_impact", 0.2),
            _ds("isolation_integrity", 0.3),
        ]
        signals = self.detector.detect_signals(scores)
        cascade = [s for s in signals if s.signal_type == "cascade_without_isolation"]
        self.assertTrue(len(cascade) > 0)
        self.assertTrue(len(cascade[0].recommendation) > 0)


# ── Aggregate Analysis Tests (10) ─────────────────────────────────────


class TestAggregateAnalysis(unittest.TestCase):
    """Aggregate analysis over an audit trail."""

    def setUp(self) -> None:
        self.detector = CrossDimensionalDetector()

    def test_aggregate_detects_repeated_patterns(self) -> None:
        scores = {"cascading_impact": 0.2, "isolation_integrity": 0.3}
        records = [_record_with_scores(scores) for _ in range(3)]
        trail = _MockAuditTrail(records)
        report = self.detector.analyze_aggregate(trail)
        types = [s.signal_type for s in report.signals]
        self.assertIn("cascade_without_isolation", types)
        self.assertEqual(report.per_pattern_counts["cascade_without_isolation"], 3)

    def test_aggregate_filters_by_agent(self) -> None:
        scores = {"cascading_impact": 0.2, "isolation_integrity": 0.3}
        r1 = _record_with_scores(scores, agent_id="alice")
        r2 = _record_with_scores(scores, agent_id="bob")
        trail = _MockAuditTrail([r1, r2])
        report = self.detector.analyze_aggregate(trail, agent_id="alice")
        # Only alice's record should contribute.
        self.assertEqual(
            report.per_pattern_counts.get("cascade_without_isolation", 0), 1,
        )

    def test_aggregate_respects_time_window(self) -> None:
        old_scores = {"cascading_impact": 0.2, "isolation_integrity": 0.3}
        old_record = _record_with_scores(old_scores)
        old_record.timestamp = time.time() - 999_999  # far in the past
        trail = _MockAuditTrail([old_record])
        report = self.detector.analyze_aggregate(trail, window_hours=1)
        self.assertEqual(report.signals, [])

    def test_trend_detection_escalates_severity(self) -> None:
        """10+ occurrences of an alert-level pattern should escalate to critical."""
        scores = {"cascading_impact": 0.2, "isolation_integrity": 0.3}
        records = [_record_with_scores(scores) for _ in range(12)]
        trail = _MockAuditTrail(records)
        report = self.detector.analyze_aggregate(trail)
        cascade = [s for s in report.signals if s.signal_type == "cascade_without_isolation"]
        self.assertEqual(len(cascade), 1)
        self.assertEqual(cascade[0].severity, "critical")

    def test_no_audit_data_returns_empty_report(self) -> None:
        trail = _MockAuditTrail([])
        report = self.detector.analyze_aggregate(trail)
        self.assertEqual(report.signals, [])
        self.assertIn("No actions", report.summary)

    def test_report_summary_human_readable(self) -> None:
        scores = {"temporal_compliance": 0.4, "incident_detection": 0.3}
        records = [_record_with_scores(scores) for _ in range(2)]
        trail = _MockAuditTrail(records)
        report = self.detector.analyze_aggregate(trail)
        self.assertIn("Analyzed", report.summary)
        self.assertIn("actions", report.summary)

    def test_report_includes_per_pattern_counts(self) -> None:
        scores = {"precedent_alignment": 0.2, "behavioral_consistency": 0.3}
        records = [_record_with_scores(scores) for _ in range(4)]
        trail = _MockAuditTrail(records)
        report = self.detector.analyze_aggregate(trail)
        self.assertIn("precedent_break_under_drift", report.per_pattern_counts)
        self.assertEqual(report.per_pattern_counts["precedent_break_under_drift"], 4)

    def test_multiple_agents_analyzed_independently(self) -> None:
        s1 = {"cascading_impact": 0.2, "isolation_integrity": 0.3}
        s2 = {"temporal_compliance": 0.4, "incident_detection": 0.3}
        r1 = _record_with_scores(s1, agent_id="agent-a")
        r2 = _record_with_scores(s2, agent_id="agent-b")
        trail = _MockAuditTrail([r1, r2])
        report_a = self.detector.analyze_aggregate(trail, agent_id="agent-a")
        report_b = self.detector.analyze_aggregate(trail, agent_id="agent-b")
        types_a = {s.signal_type for s in report_a.signals}
        types_b = {s.signal_type for s in report_b.signals}
        self.assertIn("cascade_without_isolation", types_a)
        self.assertNotIn("temporal_security_gap", types_a)
        self.assertIn("temporal_security_gap", types_b)
        self.assertNotIn("cascade_without_isolation", types_b)

    def test_report_generated_at_is_populated(self) -> None:
        trail = _MockAuditTrail([_record_with_scores({"scope_compliance": 0.9})])
        report = self.detector.analyze_aggregate(trail)
        self.assertTrue(len(report.generated_at) > 0)

    def test_aggregate_with_no_matching_patterns_clean_report(self) -> None:
        healthy = {"scope_compliance": 0.95, "authority_verification": 0.9}
        records = [_record_with_scores(healthy) for _ in range(5)]
        trail = _MockAuditTrail(records)
        report = self.detector.analyze_aggregate(trail)
        self.assertEqual(report.signals, [])
        self.assertEqual(report.per_pattern_counts, {})
        self.assertIn("No cross-dimensional signals", report.summary)


# ── Integration Tests (10) ────────────────────────────────────────────


class TestIntegration(unittest.TestCase):
    """Integration and lifecycle tests."""

    def test_detector_created_with_default_patterns(self) -> None:
        detector = CrossDimensionalDetector()
        patterns = detector.list_patterns()
        self.assertEqual(len(patterns), len(CROSS_DIMENSIONAL_PATTERNS))

    def test_detector_created_with_custom_patterns_only(self) -> None:
        custom = [{"name": "only_one", "dimensions": ["scope_compliance"], "severity": "info"}]
        detector = CrossDimensionalDetector(patterns=custom)
        patterns = detector.list_patterns()
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]["name"], "only_one")

    def test_detect_signals_populates_action_pattern(self) -> None:
        @dataclass
        class _FakeAction:
            agent_id: str = "bot-7"
            action_type: str = "deploy"
            target: str = "/prod/db"

        scores = [
            _ds("cascading_impact", 0.2),
            _ds("isolation_integrity", 0.3),
        ]
        detector = CrossDimensionalDetector()
        signals = detector.detect_signals(scores, action=_FakeAction())
        cascade = [s for s in signals if s.signal_type == "cascade_without_isolation"]
        self.assertIn("bot-7", cascade[0].action_pattern)
        self.assertIn("deploy", cascade[0].action_pattern)
        self.assertIn("/prod/db", cascade[0].action_pattern)

    def test_critical_signal_in_ambiguous_zone(self) -> None:
        """Scores that sit in an ambiguous zone should still resolve deterministically."""
        scores = [
            _ds("incident_detection", 0.49),
            _ds("ethical_alignment", 0.71),
            _ds("behavioral_consistency", 0.59),
        ]
        detector = CrossDimensionalDetector()
        signals = detector.detect_signals(scores)
        types = [s.signal_type for s in signals]
        self.assertIn("empathetic_exploitation", types)

    def test_disabled_detector_no_signals(self) -> None:
        """A detector initialized with an empty pattern list produces no signals."""
        detector = CrossDimensionalDetector(patterns=[])
        scores = [
            _ds("cascading_impact", 0.1),
            _ds("isolation_integrity", 0.1),
        ]
        signals = detector.detect_signals(scores)
        self.assertEqual(signals, [])

    def test_add_pattern_via_method(self) -> None:
        detector = CrossDimensionalDetector()
        before = len(detector.list_patterns())
        detector.add_pattern({
            "name": "added_at_runtime",
            "dimensions": ["transparency", "ethical_alignment"],
            "severity": "warning",
        })
        after = len(detector.list_patterns())
        self.assertEqual(after, before + 1)

    def test_pattern_validation_missing_required_keys(self) -> None:
        detector = CrossDimensionalDetector()
        with self.assertRaises(ValueError) as ctx:
            detector.add_pattern({"name": "incomplete"})
        self.assertIn("dimensions", str(ctx.exception))

    def test_all_builtin_patterns_have_required_fields(self) -> None:
        required = {"name", "dimensions", "severity"}
        for pattern in CROSS_DIMENSIONAL_PATTERNS:
            with self.subTest(pattern=pattern["name"]):
                self.assertTrue(
                    required.issubset(pattern.keys()),
                    f"Pattern '{pattern['name']}' missing keys: "
                    f"{required - pattern.keys()}",
                )

    def test_signal_to_dict_serialization(self) -> None:
        signal = CrossDimensionalSignal(
            signal_type="test_signal",
            dimensions_involved=["a", "b"],
            description="desc",
            severity="warning",
            action_pattern="ctx",
            recommendation="rec",
        )
        d = signal.to_dict()
        self.assertEqual(d["signal_type"], "test_signal")
        self.assertEqual(d["dimensions_involved"], ["a", "b"])
        self.assertEqual(d["severity"], "warning")
        self.assertEqual(d["recommendation"], "rec")

    def test_report_to_dict_serialization(self) -> None:
        signal = CrossDimensionalSignal(
            signal_type="inner", severity="info",
        )
        report = CrossDimensionalReport(
            report_id="rpt-001",
            analysis_type="aggregate",
            signals=[signal],
            summary="All good.",
            generated_at="2026-01-01T00:00:00+00:00",
            per_pattern_counts={"inner": 1},
        )
        d = report.to_dict()
        self.assertEqual(d["report_id"], "rpt-001")
        self.assertEqual(d["analysis_type"], "aggregate")
        self.assertEqual(len(d["signals"]), 1)
        self.assertEqual(d["signals"][0]["signal_type"], "inner")
        self.assertEqual(d["summary"], "All good.")
        self.assertEqual(d["per_pattern_counts"], {"inner": 1})


if __name__ == "__main__":
    unittest.main()
