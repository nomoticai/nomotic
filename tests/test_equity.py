"""Comprehensive tests for the nomotic.equity module.

Tests cover EquityConfig and types, EquityAnalyzer, and AnonymizationPolicy.
"""

import time
import unittest
from dataclasses import dataclass, field
from typing import Any

from nomotic.equity import (
    AnonymizationPolicy,
    AnonymizationRule,
    DisparityFinding,
    EquityAnalyzer,
    EquityConfig,
    EquityReport,
    EquityThreshold,
    GroupOutcome,
    ProtectedAttribute,
    ProxyAlert,
)


# ── Mock audit infrastructure ────────────────────────────────────────────


@dataclass
class MockAuditRecord:
    """Minimal audit record for testing."""

    agent_id: str = "agent-1"
    action_type: str = "loan_decision"
    action_target: str = "application-123"
    verdict: str = "ALLOW"
    ucs: float = 0.85
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)
    dimension_scores: dict = field(default_factory=dict)
    trust_score: float = 0.9


class MockAuditTrail:
    """Simple audit trail mock that stores records and supports query()."""

    def __init__(self, records: list[MockAuditRecord] | None = None):
        self._records = list(records) if records else []

    def add(self, record: MockAuditRecord) -> None:
        self._records.append(record)

    def query(self, **kwargs: Any) -> list[MockAuditRecord]:
        results = list(self._records)
        agent_id = kwargs.get("agent_id")
        since = kwargs.get("since")
        limit = kwargs.get("limit")

        if agent_id is not None:
            results = [r for r in results if r.agent_id == agent_id]
        if since is not None:
            results = [r for r in results if r.timestamp >= since]
        if limit is not None:
            results = results[:limit]
        return results


# ── Helper factories ──────────────────────────────────────────────────────


def _make_config(
    attributes: list[ProtectedAttribute] | None = None,
    thresholds: list[EquityThreshold] | None = None,
    minimum_sample_size: int = 5,
    enable_proxy_detection: bool = True,
) -> EquityConfig:
    if attributes is None:
        attributes = [
            ProtectedAttribute(
                name="region",
                description="Applicant region",
                attribute_source="region",
            )
        ]
    if thresholds is None:
        thresholds = [
            EquityThreshold(
                metric="denial_rate_ratio",
                warning_threshold=1.5,
                alert_threshold=2.0,
                description="Denial rate ratio between groups",
            )
        ]
    return EquityConfig(
        protected_attributes=attributes,
        thresholds=thresholds,
        minimum_sample_size=minimum_sample_size,
        enable_proxy_detection=enable_proxy_detection,
    )


def _make_records(
    group_specs: dict[str, dict[str, int]],
    attr_source: str = "region",
    agent_id: str = "agent-1",
    action_type: str = "loan_decision",
    ucs_by_group: dict[str, float] | None = None,
) -> list[MockAuditRecord]:
    """Build audit records from a spec like {"east": {"ALLOW": 40, "DENY": 10}}."""
    now = time.time()
    records = []
    for group_value, verdicts in group_specs.items():
        for verdict, count in verdicts.items():
            ucs = 0.85
            if ucs_by_group and group_value in ucs_by_group:
                ucs = ucs_by_group[group_value]
            for _ in range(count):
                records.append(
                    MockAuditRecord(
                        agent_id=agent_id,
                        action_type=action_type,
                        action_target=f"app-{len(records)}",
                        verdict=verdict,
                        ucs=ucs,
                        timestamp=now,
                        metadata={attr_source: group_value},
                    )
                )
    return records


# ══════════════════════════════════════════════════════════════════════════
# Tests: EquityConfig and types
# ══════════════════════════════════════════════════════════════════════════


class TestProtectedAttribute(unittest.TestCase):
    def test_non_proxy_creation(self):
        attr = ProtectedAttribute(
            name="region",
            description="Customer region",
            attribute_source="region",
        )
        self.assertEqual(attr.name, "region")
        self.assertFalse(attr.is_proxy)
        self.assertIsNone(attr.proxy_for)

    def test_proxy_creation(self):
        attr = ProtectedAttribute(
            name="zip_code",
            description="ZIP code as proxy for race",
            attribute_source="zip_code",
            is_proxy=True,
            proxy_for="race",
        )
        self.assertTrue(attr.is_proxy)
        self.assertEqual(attr.proxy_for, "race")


class TestEquityThreshold(unittest.TestCase):
    def test_creation(self):
        t = EquityThreshold(
            metric="denial_rate_ratio",
            warning_threshold=1.5,
            alert_threshold=2.0,
            description="Denial rate ratio",
        )
        self.assertEqual(t.metric, "denial_rate_ratio")
        self.assertAlmostEqual(t.warning_threshold, 1.5)
        self.assertAlmostEqual(t.alert_threshold, 2.0)


class TestEquityConfig(unittest.TestCase):
    def test_creation_with_attributes_and_thresholds(self):
        config = _make_config()
        self.assertEqual(len(config.protected_attributes), 1)
        self.assertEqual(len(config.thresholds), 1)
        self.assertEqual(config.minimum_sample_size, 5)

    def test_to_dict_from_dict_round_trip(self):
        config = _make_config(
            attributes=[
                ProtectedAttribute(
                    name="region",
                    description="Region",
                    attribute_source="region",
                    is_proxy=False,
                    proxy_for=None,
                ),
                ProtectedAttribute(
                    name="zip_code",
                    description="ZIP",
                    attribute_source="zip_code",
                    is_proxy=True,
                    proxy_for="race",
                ),
            ],
            thresholds=[
                EquityThreshold(
                    metric="denial_rate_ratio",
                    warning_threshold=1.5,
                    alert_threshold=2.0,
                    description="Denial ratio",
                ),
            ],
            minimum_sample_size=30,
        )
        d = config.to_dict()
        restored = EquityConfig.from_dict(d)
        self.assertEqual(len(restored.protected_attributes), 2)
        self.assertEqual(restored.protected_attributes[1].proxy_for, "race")
        self.assertEqual(restored.minimum_sample_size, 30)
        self.assertTrue(restored.enable_proxy_detection)


class TestGroupOutcome(unittest.TestCase):
    def test_approval_rate(self):
        g = GroupOutcome(
            attribute_name="region",
            attribute_value="east",
            total_actions=100,
            approval_count=80,
            denial_count=15,
            escalation_count=5,
            average_ucs=0.85,
        )
        self.assertAlmostEqual(g.approval_rate, 0.80)

    def test_denial_rate(self):
        g = GroupOutcome(
            attribute_name="region",
            attribute_value="east",
            total_actions=100,
            approval_count=80,
            denial_count=15,
            escalation_count=5,
            average_ucs=0.85,
        )
        self.assertAlmostEqual(g.denial_rate, 0.15)

    def test_rates_with_zero_actions(self):
        g = GroupOutcome(
            attribute_name="region",
            attribute_value="unknown",
            total_actions=0,
            approval_count=0,
            denial_count=0,
            escalation_count=0,
            average_ucs=0.0,
        )
        self.assertAlmostEqual(g.approval_rate, 0.0)
        self.assertAlmostEqual(g.denial_rate, 0.0)
        self.assertAlmostEqual(g.escalation_rate, 0.0)


class TestDisparityFinding(unittest.TestCase):
    def test_creation(self):
        f = DisparityFinding(
            attribute_name="region",
            group_a="east",
            group_b="west",
            metric="denial_rate_ratio",
            group_a_value=0.30,
            group_b_value=0.10,
            ratio=3.0,
            severity="alert",
            sample_size_a=100,
            sample_size_b=100,
            statistical_significance=0.001,
            description="test disparity",
        )
        self.assertEqual(f.severity, "alert")
        d = f.to_dict()
        self.assertAlmostEqual(d["ratio"], 3.0)
        self.assertEqual(d["group_a"], "east")


class TestEquityReport(unittest.TestCase):
    def test_serialization_round_trip(self):
        report = EquityReport(
            report_id="abc123",
            generated_at="2025-01-01T00:00:00+00:00",
            analysis_window_start="2024-12-01T00:00:00+00:00",
            analysis_window_end="2025-01-01T00:00:00+00:00",
            total_actions_analyzed=200,
            group_outcomes=[
                GroupOutcome(
                    attribute_name="region",
                    attribute_value="east",
                    total_actions=100,
                    approval_count=90,
                    denial_count=10,
                    escalation_count=0,
                    average_ucs=0.9,
                )
            ],
            disparities=[],
            proxy_alerts=[],
            summary="All good.",
            recommendations=[],
            config_snapshot={"minimum_sample_size": 30},
        )
        d = report.to_dict()
        self.assertEqual(d["report_id"], "abc123")
        self.assertEqual(d["total_actions_analyzed"], 200)
        self.assertEqual(len(d["group_outcomes"]), 1)
        self.assertAlmostEqual(d["group_outcomes"][0]["approval_rate"], 0.9)
        self.assertEqual(d["summary"], "All good.")


class TestAnonymizationRule(unittest.TestCase):
    def test_creation(self):
        rule = AnonymizationRule(
            attribute_name="gender",
            hide_from_reasoning=True,
            hide_from_methods=["loan_decision"],
            allow_for_methods=["medical_triage"],
            description="Hide gender from loan decisions",
        )
        self.assertEqual(rule.attribute_name, "gender")
        self.assertTrue(rule.hide_from_reasoning)
        self.assertIn("loan_decision", rule.hide_from_methods)


class TestAnonymizationPolicyShouldHide(unittest.TestCase):
    def test_hide_for_method(self):
        policy = AnonymizationPolicy(
            rules=[
                AnonymizationRule(
                    attribute_name="gender",
                    hide_from_reasoning=False,
                    hide_from_methods=["loan_decision"],
                )
            ]
        )
        self.assertTrue(policy.should_hide("gender", "loan_decision"))

    def test_allow_for_method(self):
        policy = AnonymizationPolicy(
            rules=[
                AnonymizationRule(
                    attribute_name="gender",
                    hide_from_reasoning=True,
                    allow_for_methods=["medical_triage"],
                )
            ]
        )
        self.assertFalse(policy.should_hide("gender", "medical_triage"))

    def test_default_hide(self):
        policy = AnonymizationPolicy(rules=[], default_hide=True)
        self.assertTrue(policy.should_hide("some_attr", "any_method"))

    def test_default_allow(self):
        policy = AnonymizationPolicy(rules=[], default_hide=False)
        self.assertFalse(policy.should_hide("some_attr", "any_method"))


# ══════════════════════════════════════════════════════════════════════════
# Tests: EquityAnalyzer
# ══════════════════════════════════════════════════════════════════════════


class TestEquityAnalyzerBasic(unittest.TestCase):
    def test_analyze_no_audit_data(self):
        config = _make_config()
        analyzer = EquityAnalyzer(config)
        trail = MockAuditTrail()
        report = analyzer.analyze(trail)
        self.assertEqual(report.total_actions_analyzed, 0)
        self.assertEqual(len(report.disparities), 0)
        self.assertEqual(len(report.group_outcomes), 0)
        self.assertIn("No actions", report.summary)

    def test_analyze_uniform_outcomes(self):
        config = _make_config(minimum_sample_size=5)
        analyzer = EquityAnalyzer(config)
        records = _make_records({
            "east": {"ALLOW": 50},
            "west": {"ALLOW": 50},
        })
        trail = MockAuditTrail(records)
        report = analyzer.analyze(trail)
        self.assertEqual(report.total_actions_analyzed, 100)
        self.assertEqual(len(report.disparities), 0)
        self.assertIn("No significant disparities", report.summary)

    def test_analyze_disparate_denial_rates(self):
        config = _make_config(minimum_sample_size=5)
        analyzer = EquityAnalyzer(config)
        # East: 10% denial rate, West: 50% denial rate -> ratio ~5.0
        records = _make_records({
            "east": {"ALLOW": 90, "DENY": 10},
            "west": {"ALLOW": 50, "DENY": 50},
        })
        trail = MockAuditTrail(records)
        report = analyzer.analyze(trail)
        self.assertGreater(len(report.disparities), 0)
        finding = report.disparities[0]
        self.assertEqual(finding.metric, "denial_rate_ratio")
        self.assertGreaterEqual(finding.ratio, 2.0)

    def test_analyze_respects_minimum_sample_size(self):
        config = _make_config(minimum_sample_size=100)
        analyzer = EquityAnalyzer(config)
        # Only 20 records per group - below minimum_sample_size of 100
        records = _make_records({
            "east": {"ALLOW": 18, "DENY": 2},
            "west": {"ALLOW": 2, "DENY": 18},
        })
        trail = MockAuditTrail(records)
        report = analyzer.analyze(trail)
        # Even though denial rates differ, sample is too small
        self.assertEqual(len(report.disparities), 0)

    def test_analyze_filters_by_agent_id(self):
        config = _make_config(minimum_sample_size=5)
        analyzer = EquityAnalyzer(config)
        records_a = _make_records(
            {"east": {"ALLOW": 50}, "west": {"ALLOW": 50}},
            agent_id="agent-A",
        )
        records_b = _make_records(
            {"east": {"ALLOW": 10, "DENY": 40}, "west": {"ALLOW": 50}},
            agent_id="agent-B",
        )
        trail = MockAuditTrail(records_a + records_b)

        # Analyze only agent-A: no disparities since all ALLOW
        report = analyzer.analyze(trail, agent_id="agent-A")
        self.assertEqual(report.total_actions_analyzed, 100)
        self.assertEqual(len(report.disparities), 0)

    def test_analyze_filters_by_method(self):
        config = _make_config(minimum_sample_size=5)
        analyzer = EquityAnalyzer(config)
        loan_records = _make_records(
            {"east": {"ALLOW": 10, "DENY": 40}, "west": {"ALLOW": 50}},
            action_type="loan_decision",
        )
        insurance_records = _make_records(
            {"east": {"ALLOW": 50}, "west": {"ALLOW": 50}},
            action_type="insurance_quote",
        )
        trail = MockAuditTrail(loan_records + insurance_records)

        # Only insurance_quote: no disparities
        report = analyzer.analyze(trail, method="insurance_quote")
        self.assertEqual(report.total_actions_analyzed, 100)
        self.assertEqual(len(report.disparities), 0)

    def test_analyze_respects_time_window(self):
        config = _make_config(minimum_sample_size=5)
        analyzer = EquityAnalyzer(config)
        now = time.time()
        # Old records (beyond the default 720-hour window)
        old_records = _make_records(
            {"east": {"ALLOW": 10, "DENY": 40}, "west": {"ALLOW": 50}},
        )
        for r in old_records:
            r.timestamp = now - (800 * 3600)

        # Recent records (within window)
        recent_records = _make_records(
            {"east": {"ALLOW": 50}, "west": {"ALLOW": 50}},
        )
        for r in recent_records:
            r.timestamp = now

        trail = MockAuditTrail(old_records + recent_records)
        report = analyzer.analyze(trail)
        # Only recent (uniform) records should be analyzed
        self.assertEqual(report.total_actions_analyzed, 100)
        self.assertEqual(len(report.disparities), 0)


class TestEquityAnalyzerStatistics(unittest.TestCase):
    def test_statistical_significance_filters_noise(self):
        """Small differences with small samples should not trigger findings."""
        config = _make_config(minimum_sample_size=5)
        analyzer = EquityAnalyzer(config)
        # Very slight difference in denial rates with small samples
        records = _make_records({
            "east": {"ALLOW": 8, "DENY": 2},
            "west": {"ALLOW": 7, "DENY": 3},
        })
        trail = MockAuditTrail(records)
        report = analyzer.analyze(trail)
        # Difference is too small and sample too small for significance
        self.assertEqual(len(report.disparities), 0)

    def test_large_disparity_large_samples_high_significance(self):
        """Large disparity with large samples should produce findings."""
        config = _make_config(minimum_sample_size=5)
        analyzer = EquityAnalyzer(config)
        # Stark difference: east 5% denial, west 40% denial
        records = _make_records({
            "east": {"ALLOW": 190, "DENY": 10},
            "west": {"ALLOW": 120, "DENY": 80},
        })
        trail = MockAuditTrail(records)
        report = analyzer.analyze(trail)
        self.assertGreater(len(report.disparities), 0)
        finding = report.disparities[0]
        # p-value should be very small
        self.assertLess(finding.statistical_significance, 0.01)
        self.assertIn(finding.severity, ("warning", "alert"))


class TestEquityAnalyzerMultipleAttributes(unittest.TestCase):
    def test_multiple_attributes_analyzed_independently(self):
        region_attr = ProtectedAttribute(
            name="region",
            description="Region",
            attribute_source="region",
        )
        tier_attr = ProtectedAttribute(
            name="customer_tier",
            description="Customer tier",
            attribute_source="customer_tier",
        )
        config = _make_config(
            attributes=[region_attr, tier_attr],
            minimum_sample_size=5,
        )
        analyzer = EquityAnalyzer(config)

        now = time.time()
        records = []
        # Region: no disparity
        for _ in range(50):
            records.append(MockAuditRecord(
                verdict="ALLOW", ucs=0.85, timestamp=now,
                metadata={"region": "east", "customer_tier": "gold"},
            ))
            records.append(MockAuditRecord(
                verdict="ALLOW", ucs=0.85, timestamp=now,
                metadata={"region": "west", "customer_tier": "silver"},
            ))
        # Tier: add denials only for silver to create a tier disparity
        for _ in range(50):
            records.append(MockAuditRecord(
                verdict="DENY", ucs=0.4, timestamp=now,
                metadata={"region": "east", "customer_tier": "silver"},
            ))

        trail = MockAuditTrail(records)
        report = analyzer.analyze(trail)

        # Check that group outcomes cover both attributes
        attr_names = {g.attribute_name for g in report.group_outcomes}
        self.assertIn("region", attr_names)
        self.assertIn("customer_tier", attr_names)


class TestEquityAnalyzerProxyDetection(unittest.TestCase):
    def test_non_proxy_attribute_disparity_flagged(self):
        """An attribute not marked as proxy but showing UCS correlation should be flagged."""
        attr = ProtectedAttribute(
            name="zip_code",
            description="ZIP code",
            attribute_source="zip_code",
            is_proxy=False,
        )
        config = _make_config(
            attributes=[attr],
            minimum_sample_size=5,
            enable_proxy_detection=True,
        )
        analyzer = EquityAnalyzer(config)

        now = time.time()
        records = []
        # Group A: high UCS, Group B: low UCS -> correlation with group membership
        for _ in range(60):
            records.append(MockAuditRecord(
                verdict="ALLOW", ucs=0.95, timestamp=now,
                metadata={"zip_code": "10001"},
            ))
        for _ in range(60):
            records.append(MockAuditRecord(
                verdict="DENY", ucs=0.20, timestamp=now,
                metadata={"zip_code": "90210"},
            ))

        trail = MockAuditTrail(records)
        report = analyzer.analyze(trail)
        self.assertGreater(len(report.proxy_alerts), 0)
        alert = report.proxy_alerts[0]
        self.assertEqual(alert.proxy_attribute, "zip_code")
        self.assertIn("potential proxy discrimination", alert.description)

    def test_known_proxy_attribute_correlation_flagged(self):
        """A known proxy attribute with UCS correlation should reference what it proxies."""
        attr = ProtectedAttribute(
            name="zip_code",
            description="ZIP code as proxy",
            attribute_source="zip_code",
            is_proxy=True,
            proxy_for="race",
        )
        config = _make_config(
            attributes=[attr],
            minimum_sample_size=5,
            enable_proxy_detection=True,
        )
        analyzer = EquityAnalyzer(config)

        now = time.time()
        records = []
        for _ in range(60):
            records.append(MockAuditRecord(
                verdict="ALLOW", ucs=0.95, timestamp=now,
                metadata={"zip_code": "10001"},
            ))
        for _ in range(60):
            records.append(MockAuditRecord(
                verdict="DENY", ucs=0.20, timestamp=now,
                metadata={"zip_code": "90210"},
            ))

        trail = MockAuditTrail(records)
        report = analyzer.analyze(trail)
        self.assertGreater(len(report.proxy_alerts), 0)
        alert = report.proxy_alerts[0]
        self.assertIn("Known proxy", alert.description)
        self.assertIn("race", alert.description)


class TestEquityAnalyzerReporting(unittest.TestCase):
    def test_summary_is_human_readable(self):
        config = _make_config(minimum_sample_size=5)
        analyzer = EquityAnalyzer(config)
        records = _make_records({
            "east": {"ALLOW": 190, "DENY": 10},
            "west": {"ALLOW": 120, "DENY": 80},
        })
        trail = MockAuditTrail(records)
        report = analyzer.analyze(trail)
        self.assertIn("Analyzed", report.summary)
        self.assertIn("400", report.summary)

    def test_report_includes_config_snapshot(self):
        config = _make_config(minimum_sample_size=42)
        analyzer = EquityAnalyzer(config)
        trail = MockAuditTrail(_make_records({"east": {"ALLOW": 10}}))
        report = analyzer.analyze(trail)
        self.assertIn("minimum_sample_size", report.config_snapshot)
        self.assertEqual(report.config_snapshot["minimum_sample_size"], 42)

    def test_recommendations_for_alert_level_disparities(self):
        config = _make_config(minimum_sample_size=5)
        analyzer = EquityAnalyzer(config)
        # Create a stark disparity that exceeds alert threshold (2.0)
        records = _make_records({
            "east": {"ALLOW": 190, "DENY": 10},
            "west": {"ALLOW": 100, "DENY": 100},
        })
        trail = MockAuditTrail(records)
        report = analyzer.analyze(trail)

        # Find alert-level disparities
        alert_findings = [d for d in report.disparities if d.severity == "alert"]
        if alert_findings:
            self.assertGreater(len(report.recommendations), 0)
            self.assertIn("Review", report.recommendations[0])


# ══════════════════════════════════════════════════════════════════════════
# Tests: AnonymizationPolicy
# ══════════════════════════════════════════════════════════════════════════


class TestAnonymizationPolicy(unittest.TestCase):
    def test_should_hide_returns_true_for_hidden_attribute(self):
        policy = AnonymizationPolicy(
            rules=[
                AnonymizationRule(
                    attribute_name="gender",
                    hide_from_reasoning=True,
                )
            ]
        )
        self.assertTrue(policy.should_hide("gender", "loan_decision"))

    def test_should_hide_returns_false_for_allowed_method(self):
        policy = AnonymizationPolicy(
            rules=[
                AnonymizationRule(
                    attribute_name="gender",
                    hide_from_reasoning=True,
                    allow_for_methods=["medical_triage"],
                )
            ]
        )
        self.assertFalse(policy.should_hide("gender", "medical_triage"))

    def test_apply_to_parameters_removes_hidden(self):
        policy = AnonymizationPolicy(
            rules=[
                AnonymizationRule(
                    attribute_name="gender",
                    hide_from_reasoning=True,
                ),
                AnonymizationRule(
                    attribute_name="age",
                    hide_from_reasoning=True,
                ),
            ]
        )
        params = {"gender": "female", "age": 35, "income": 50000, "credit_score": 720}
        result = policy.apply_to_parameters(params, "loan_decision")
        self.assertNotIn("gender", result)
        self.assertNotIn("age", result)

    def test_apply_to_parameters_preserves_non_hidden(self):
        policy = AnonymizationPolicy(
            rules=[
                AnonymizationRule(
                    attribute_name="gender",
                    hide_from_reasoning=True,
                )
            ]
        )
        params = {"gender": "male", "income": 50000, "credit_score": 720}
        result = policy.apply_to_parameters(params, "loan_decision")
        self.assertIn("income", result)
        self.assertEqual(result["income"], 50000)
        self.assertIn("credit_score", result)
        self.assertEqual(result["credit_score"], 720)

    def test_default_hide_mode_hides_unlisted(self):
        policy = AnonymizationPolicy(
            rules=[
                AnonymizationRule(
                    attribute_name="income",
                    hide_from_reasoning=False,
                )
            ],
            default_hide=True,
        )
        # "income" has a rule saying don't hide, so it stays
        self.assertFalse(policy.should_hide("income", "loan_decision"))
        # "gender" has no rule, default_hide=True applies
        self.assertTrue(policy.should_hide("gender", "loan_decision"))

    def test_default_allow_mode_allows_unlisted(self):
        policy = AnonymizationPolicy(
            rules=[
                AnonymizationRule(
                    attribute_name="gender",
                    hide_from_reasoning=True,
                )
            ],
            default_hide=False,
        )
        # "gender" has a rule saying hide
        self.assertTrue(policy.should_hide("gender", "loan_decision"))
        # "income" has no rule, default_hide=False applies
        self.assertFalse(policy.should_hide("income", "loan_decision"))

    def test_empty_policy_hides_nothing(self):
        policy = AnonymizationPolicy()
        self.assertFalse(policy.should_hide("gender", "loan_decision"))
        self.assertFalse(policy.should_hide("age", "insurance_quote"))
        params = {"gender": "female", "age": 30}
        result = policy.apply_to_parameters(params, "any_method")
        self.assertEqual(result, params)


if __name__ == "__main__":
    unittest.main()
