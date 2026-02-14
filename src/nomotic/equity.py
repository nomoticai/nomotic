"""Outcome Equity Analyzer and Anonymization Policy.

The Outcome Equity Analyzer examines patterns in governance decisions across
populations to detect systematic disparities.  It consumes audit trail data
and produces equity reports.

The Anonymization Policy provides a mechanism for stripping attributes from
action parameters before they reach agent reasoning for methods where those
attributes are not relevant.

Core principle: Nomotic provides transparency for ethical accountability,
not ethical judgment.  Organizations define protected attributes and
thresholds.  Nomotic evaluates against those criteria and surfaces patterns
that humans must evaluate.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AnonymizationPolicy",
    "AnonymizationRule",
    "DisparityFinding",
    "EquityAnalyzer",
    "EquityConfig",
    "EquityReport",
    "EquityThreshold",
    "GroupOutcome",
    "ProtectedAttribute",
    "ProxyAlert",
]


# ── Configuration types ────────────────────────────────────────────────


@dataclass
class ProtectedAttribute:
    """An attribute the organization wants to monitor for equitable outcomes.

    The organization defines what counts as a protected attribute.  Nomotic
    does NOT have a built-in list of protected characteristics.  What is
    protected varies by jurisdiction and context.
    """

    name: str  # e.g., "region", "customer_tier", "zip_code", "age_group"
    description: str
    attribute_source: str  # where this comes from in action parameters or context
    is_proxy: bool = False  # if True, correlates with a protected characteristic
    proxy_for: str | None = None  # what it proxies for, if known


@dataclass
class EquityThreshold:
    """Organization-defined threshold for what constitutes a concerning disparity."""

    metric: str  # "denial_rate_ratio", "approval_rate_difference", "escalation_rate_ratio"
    warning_threshold: float  # above this, flag as warning
    alert_threshold: float  # above this, flag as alert
    description: str


@dataclass
class EquityConfig:
    """Complete configuration for equity analysis.

    The organization defines the protected attributes and thresholds.
    Nomotic does NOT have defaults for these.  The organization must
    configure what they need to monitor.
    """

    protected_attributes: list[ProtectedAttribute] = field(default_factory=list)
    thresholds: list[EquityThreshold] = field(default_factory=list)
    minimum_sample_size: int = 30  # minimum observations per group before analysis
    analysis_window_hours: int = 720  # default 30 days
    enable_proxy_detection: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "protected_attributes": [
                {
                    "name": a.name,
                    "description": a.description,
                    "attribute_source": a.attribute_source,
                    "is_proxy": a.is_proxy,
                    "proxy_for": a.proxy_for,
                }
                for a in self.protected_attributes
            ],
            "thresholds": [
                {
                    "metric": t.metric,
                    "warning_threshold": t.warning_threshold,
                    "alert_threshold": t.alert_threshold,
                    "description": t.description,
                }
                for t in self.thresholds
            ],
            "minimum_sample_size": self.minimum_sample_size,
            "analysis_window_hours": self.analysis_window_hours,
            "enable_proxy_detection": self.enable_proxy_detection,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EquityConfig:
        return cls(
            protected_attributes=[
                ProtectedAttribute(**a) for a in d.get("protected_attributes", [])
            ],
            thresholds=[
                EquityThreshold(**t) for t in d.get("thresholds", [])
            ],
            minimum_sample_size=d.get("minimum_sample_size", 30),
            analysis_window_hours=d.get("analysis_window_hours", 720),
            enable_proxy_detection=d.get("enable_proxy_detection", True),
        )


# ── Analysis result types ──────────────────────────────────────────────


@dataclass
class GroupOutcome:
    """Outcomes for a specific attribute value group."""

    attribute_name: str
    attribute_value: str
    total_actions: int
    approval_count: int
    denial_count: int
    escalation_count: int
    average_ucs: float
    methods_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def approval_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.approval_count / self.total_actions

    @property
    def denial_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.denial_count / self.total_actions

    @property
    def escalation_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.escalation_count / self.total_actions

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute_name": self.attribute_name,
            "attribute_value": self.attribute_value,
            "total_actions": self.total_actions,
            "approval_count": self.approval_count,
            "denial_count": self.denial_count,
            "escalation_count": self.escalation_count,
            "average_ucs": round(self.average_ucs, 4),
            "approval_rate": round(self.approval_rate, 4),
            "denial_rate": round(self.denial_rate, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "methods_distribution": self.methods_distribution,
        }


@dataclass
class DisparityFinding:
    """A detected disparity between groups.

    Nomotic does NOT say this is 'biased' -- it says 'here is a disparity'.
    The organization determines whether the disparity is problematic.
    """

    attribute_name: str
    group_a: str  # the group with more favorable outcomes
    group_b: str  # the group with less favorable outcomes
    metric: str  # what was compared
    group_a_value: float
    group_b_value: float
    ratio: float  # group_a_value / group_b_value (or difference)
    severity: str  # "info", "warning", "alert"
    sample_size_a: int
    sample_size_b: int
    statistical_significance: float  # p-value
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute_name": self.attribute_name,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "metric": self.metric,
            "group_a_value": round(self.group_a_value, 4),
            "group_b_value": round(self.group_b_value, 4),
            "ratio": round(self.ratio, 4),
            "severity": self.severity,
            "sample_size_a": self.sample_size_a,
            "sample_size_b": self.sample_size_b,
            "statistical_significance": round(self.statistical_significance, 6),
            "description": self.description,
        }


@dataclass
class ProxyAlert:
    """Detected potential proxy discrimination.

    Nomotic flags the correlation.  The organization determines if it
    is problematic.
    """

    proxy_attribute: str
    correlated_outcome: str
    correlation_strength: float  # 0.0-1.0
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "proxy_attribute": self.proxy_attribute,
            "correlated_outcome": self.correlated_outcome,
            "correlation_strength": round(self.correlation_strength, 4),
            "description": self.description,
        }


@dataclass
class EquityReport:
    """Complete equity analysis report."""

    report_id: str
    generated_at: str
    analysis_window_start: str
    analysis_window_end: str
    total_actions_analyzed: int
    group_outcomes: list[GroupOutcome] = field(default_factory=list)
    disparities: list[DisparityFinding] = field(default_factory=list)
    proxy_alerts: list[ProxyAlert] = field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "analysis_window_start": self.analysis_window_start,
            "analysis_window_end": self.analysis_window_end,
            "total_actions_analyzed": self.total_actions_analyzed,
            "group_outcomes": [g.to_dict() for g in self.group_outcomes],
            "disparities": [d.to_dict() for d in self.disparities],
            "proxy_alerts": [p.to_dict() for p in self.proxy_alerts],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "config_snapshot": self.config_snapshot,
        }


# ── Statistical helpers (pure Python, no external dependencies) ────────


def _z_test_two_proportions(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-proportion z-test.  Returns a p-value (two-tailed).

    This is governance-grade signal detection, not publishable statistics.
    Sufficient to distinguish 'this looks concerning' from 'this is noise
    from small sample sizes'.
    """
    if n1 == 0 or n2 == 0:
        return 1.0  # No data, no significance

    # Pooled proportion
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0.0 or p_pool == 1.0:
        return 1.0  # Degenerate case

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0.0:
        return 1.0

    z = abs(p1 - p2) / se

    # Approximate p-value from z using the complementary error function
    # erfc(z / sqrt(2)) gives two-tailed p-value
    p_value = math.erfc(z / math.sqrt(2))
    return p_value


def _point_biserial_approx(values: list[float], groups: list[int]) -> float:
    """Approximate point-biserial correlation between a continuous variable
    and a binary grouping variable.  Returns correlation in [0, 1].

    Used for proxy detection: checks if outcome (e.g., UCS score) correlates
    with group membership.
    """
    if len(values) != len(groups) or len(values) < 4:
        return 0.0

    unique_groups = set(groups)
    if len(unique_groups) != 2:
        return 0.0

    g0, g1 = sorted(unique_groups)
    vals_0 = [v for v, g in zip(values, groups) if g == g0]
    vals_1 = [v for v, g in zip(values, groups) if g == g1]

    if not vals_0 or not vals_1:
        return 0.0

    mean_0 = sum(vals_0) / len(vals_0)
    mean_1 = sum(vals_1) / len(vals_1)

    n = len(values)
    n0 = len(vals_0)
    n1 = len(vals_1)

    overall_mean = sum(values) / n
    ss_total = sum((v - overall_mean) ** 2 for v in values)
    if ss_total == 0.0:
        return 0.0

    sd = math.sqrt(ss_total / n)
    if sd == 0.0:
        return 0.0

    rpb = (mean_1 - mean_0) / sd * math.sqrt(n0 * n1 / (n * n))
    return min(abs(rpb), 1.0)


# ── Equity Analyzer ────────────────────────────────────────────────────


class EquityAnalyzer:
    """Examines patterns in governance decisions across populations.

    Consumes audit trail data and produces equity reports identifying
    disparities in outcomes across organization-defined protected
    attributes.
    """

    def __init__(self, config: EquityConfig) -> None:
        self._config = config
        self._lock = threading.Lock()

    @property
    def config(self) -> EquityConfig:
        return self._config

    def analyze(
        self,
        audit_trail: Any,
        agent_id: str | None = None,
        method: str | None = None,
        window_hours: int | None = None,
    ) -> EquityReport:
        """Run equity analysis on audit trail data.

        Args:
            audit_trail: An AuditTrail instance to query.
            agent_id: Optional filter to analyze a specific agent.
            method: Optional filter to analyze a specific action type.
            window_hours: Override the default analysis window.

        Returns:
            An EquityReport with group outcomes, disparities, and proxy alerts.
        """
        window = window_hours or self._config.analysis_window_hours
        now = time.time()
        window_start = now - (window * 3600)

        # Query audit trail for actions in window
        query_kwargs: dict[str, Any] = {
            "since": window_start,
            "limit": 100000,  # High limit to get all records
        }
        if agent_id is not None:
            query_kwargs["agent_id"] = agent_id

        records = audit_trail.query(**query_kwargs)

        # Filter by method if specified
        if method is not None:
            records = [r for r in records if r.action_type == method]

        # Group by each protected attribute and compute outcomes
        all_group_outcomes: list[GroupOutcome] = []
        all_disparities: list[DisparityFinding] = []
        all_proxy_alerts: list[ProxyAlert] = []

        for attr in self._config.protected_attributes:
            groups = self._compute_group_outcomes(records, attr)
            all_group_outcomes.extend(groups)

            disparities = self._detect_disparities(groups, self._config.thresholds)
            all_disparities.extend(disparities)

        # Proxy detection
        if self._config.enable_proxy_detection:
            groups_by_attr = {}
            for attr in self._config.protected_attributes:
                groups_by_attr[attr.name] = [
                    g for g in all_group_outcomes if g.attribute_name == attr.name
                ]
            proxy_alerts = self._detect_proxies(records, groups_by_attr)
            all_proxy_alerts.extend(proxy_alerts)

        # Generate summary and recommendations
        summary = self._generate_summary(records, all_disparities, all_proxy_alerts)
        recommendations = self._generate_recommendations(all_disparities, all_proxy_alerts)

        return EquityReport(
            report_id=uuid.uuid4().hex[:12],
            generated_at=_format_time(now),
            analysis_window_start=_format_time(window_start),
            analysis_window_end=_format_time(now),
            total_actions_analyzed=len(records),
            group_outcomes=all_group_outcomes,
            disparities=all_disparities,
            proxy_alerts=all_proxy_alerts,
            summary=summary,
            recommendations=recommendations,
            config_snapshot=self._config.to_dict(),
        )

    def _compute_group_outcomes(
        self, records: list[Any], attribute: ProtectedAttribute
    ) -> list[GroupOutcome]:
        """Group audit records by attribute value and compute outcomes."""
        groups: dict[str, dict[str, Any]] = {}

        for record in records:
            # Extract attribute value from action parameters in metadata,
            # or from the record's metadata
            attr_value = self._extract_attribute(record, attribute)
            if attr_value is None:
                continue

            if attr_value not in groups:
                groups[attr_value] = {
                    "total": 0,
                    "approvals": 0,
                    "denials": 0,
                    "escalations": 0,
                    "ucs_sum": 0.0,
                    "methods": {},
                }

            g = groups[attr_value]
            g["total"] += 1
            g["ucs_sum"] += record.ucs

            if record.verdict == "ALLOW":
                g["approvals"] += 1
            elif record.verdict == "DENY":
                g["denials"] += 1
            elif record.verdict == "ESCALATE":
                g["escalations"] += 1

            method = record.action_type
            g["methods"][method] = g["methods"].get(method, 0) + 1

        results = []
        for value, data in groups.items():
            avg_ucs = data["ucs_sum"] / data["total"] if data["total"] > 0 else 0.0
            results.append(
                GroupOutcome(
                    attribute_name=attribute.name,
                    attribute_value=value,
                    total_actions=data["total"],
                    approval_count=data["approvals"],
                    denial_count=data["denials"],
                    escalation_count=data["escalations"],
                    average_ucs=avg_ucs,
                    methods_distribution=data["methods"],
                )
            )

        return results

    def _extract_attribute(
        self, record: Any, attribute: ProtectedAttribute
    ) -> str | None:
        """Extract an attribute value from an audit record.

        Looks in the record's metadata (which may contain action parameters),
        then in the record's direct fields.
        """
        source = attribute.attribute_source

        # Check metadata (which may contain the original action parameters)
        if hasattr(record, "metadata") and isinstance(record.metadata, dict):
            if source in record.metadata:
                return str(record.metadata[source])
            # Check nested parameters
            params = record.metadata.get("parameters", {})
            if isinstance(params, dict) and source in params:
                return str(params[source])

        # Check direct record fields
        if hasattr(record, source):
            val = getattr(record, source)
            if val is not None and val != "":
                return str(val)

        return None

    def _detect_disparities(
        self,
        groups: list[GroupOutcome],
        thresholds: list[EquityThreshold],
    ) -> list[DisparityFinding]:
        """Compare groups pairwise against thresholds."""
        findings: list[DisparityFinding] = []
        min_sample = self._config.minimum_sample_size

        for i, ga in enumerate(groups):
            for gb in groups[i + 1:]:
                # Skip groups below minimum sample size
                if ga.total_actions < min_sample or gb.total_actions < min_sample:
                    continue

                for threshold in thresholds:
                    finding = self._compare_groups(ga, gb, threshold)
                    if finding is not None:
                        findings.append(finding)

        return findings

    def _compare_groups(
        self,
        ga: GroupOutcome,
        gb: GroupOutcome,
        threshold: EquityThreshold,
    ) -> DisparityFinding | None:
        """Compare two groups on a single metric against a threshold."""
        metric = threshold.metric

        if metric == "denial_rate_ratio":
            val_a = ga.denial_rate
            val_b = gb.denial_rate
            # Ensure group_a has the higher rate
            if val_b > val_a:
                ga, gb = gb, ga
                val_a, val_b = val_b, val_a
            if val_b == 0.0:
                if val_a == 0.0:
                    return None
                ratio = float("inf")
            else:
                ratio = val_a / val_b
            p_value = _z_test_two_proportions(
                val_a, ga.total_actions, val_b, gb.total_actions
            )

        elif metric == "approval_rate_difference":
            val_a = ga.approval_rate
            val_b = gb.approval_rate
            # group_a has the higher approval rate (more favorable)
            if val_b > val_a:
                ga, gb = gb, ga
                val_a, val_b = val_b, val_a
            ratio = val_a - val_b  # difference, not ratio
            p_value = _z_test_two_proportions(
                val_a, ga.total_actions, val_b, gb.total_actions
            )

        elif metric == "escalation_rate_ratio":
            val_a = ga.escalation_rate
            val_b = gb.escalation_rate
            if val_b > val_a:
                ga, gb = gb, ga
                val_a, val_b = val_b, val_a
            if val_b == 0.0:
                if val_a == 0.0:
                    return None
                ratio = float("inf")
            else:
                ratio = val_a / val_b
            p_value = _z_test_two_proportions(
                val_a, ga.total_actions, val_b, gb.total_actions
            )
        else:
            return None

        # Filter out statistically insignificant results
        if p_value > 0.05:
            return None

        # Determine severity
        if ratio >= threshold.alert_threshold:
            severity = "alert"
        elif ratio >= threshold.warning_threshold:
            severity = "warning"
        else:
            return None  # Below warning threshold

        description = (
            f"{metric}: {ga.attribute_value} ({val_a:.3f}) vs "
            f"{gb.attribute_value} ({val_b:.3f}), "
            f"ratio={ratio:.2f}, p={p_value:.4f}"
        )

        return DisparityFinding(
            attribute_name=ga.attribute_name,
            group_a=ga.attribute_value,
            group_b=gb.attribute_value,
            metric=metric,
            group_a_value=val_a,
            group_b_value=val_b,
            ratio=ratio,
            severity=severity,
            sample_size_a=ga.total_actions,
            sample_size_b=gb.total_actions,
            statistical_significance=p_value,
            description=description,
        )

    def _detect_proxies(
        self,
        records: list[Any],
        groups_by_attribute: dict[str, list[GroupOutcome]],
    ) -> list[ProxyAlert]:
        """Detect potential proxy discrimination."""
        alerts: list[ProxyAlert] = []

        for attr in self._config.protected_attributes:
            groups = groups_by_attribute.get(attr.name, [])
            if len(groups) < 2:
                continue

            # For each attribute, check if UCS outcomes correlate with
            # group membership
            all_ucs: list[float] = []
            all_groups: list[int] = []
            group_map: dict[str, int] = {}
            next_id = 0

            for group in groups:
                if group.attribute_value not in group_map:
                    group_map[group.attribute_value] = next_id
                    next_id += 1

            # Re-extract per-record data for correlation
            for record in records:
                attr_val = self._extract_attribute(
                    record,
                    attr,
                )
                if attr_val is None or attr_val not in group_map:
                    continue
                all_ucs.append(record.ucs)
                all_groups.append(group_map[attr_val])

            if len(set(all_groups)) != 2 or len(all_ucs) < 10:
                continue

            correlation = _point_biserial_approx(all_ucs, all_groups)

            if correlation >= 0.3:
                # Significant correlation detected
                if attr.is_proxy:
                    desc = (
                        f"Known proxy attribute '{attr.name}' (proxy for "
                        f"'{attr.proxy_for}') shows correlation ({correlation:.2f}) "
                        f"with governance outcomes"
                    )
                else:
                    desc = (
                        f"Attribute '{attr.name}' shows correlation "
                        f"({correlation:.2f}) with governance outcomes, "
                        f"suggesting potential proxy discrimination"
                    )

                alerts.append(
                    ProxyAlert(
                        proxy_attribute=attr.name,
                        correlated_outcome="ucs_distribution",
                        correlation_strength=correlation,
                        description=desc,
                    )
                )

        return alerts

    def _generate_summary(
        self,
        records: list[Any],
        disparities: list[DisparityFinding],
        proxy_alerts: list[ProxyAlert],
    ) -> str:
        """Generate a human-readable summary."""
        if not records:
            return "No actions analyzed in the specified window."

        parts = [f"Analyzed {len(records)} governance actions."]

        alert_count = sum(1 for d in disparities if d.severity == "alert")
        warning_count = sum(1 for d in disparities if d.severity == "warning")

        if alert_count > 0:
            parts.append(f"{alert_count} alert-level disparities detected.")
        if warning_count > 0:
            parts.append(f"{warning_count} warning-level disparities detected.")
        if not disparities:
            parts.append("No significant disparities detected.")

        if proxy_alerts:
            parts.append(f"{len(proxy_alerts)} potential proxy correlation(s) flagged.")

        return " ".join(parts)

    def _generate_recommendations(
        self,
        disparities: list[DisparityFinding],
        proxy_alerts: list[ProxyAlert],
    ) -> list[str]:
        """Generate recommendations for alert-level findings."""
        recs: list[str] = []

        for d in disparities:
            if d.severity == "alert":
                recs.append(
                    f"Review {d.metric} disparity between '{d.group_a}' and "
                    f"'{d.group_b}' on attribute '{d.attribute_name}' "
                    f"(ratio: {d.ratio:.2f})"
                )

        for p in proxy_alerts:
            recs.append(
                f"Investigate correlation between '{p.proxy_attribute}' "
                f"and governance outcomes (strength: {p.correlation_strength:.2f})"
            )

        return recs

    def update_config(self, config: EquityConfig) -> None:
        """Update equity configuration."""
        with self._lock:
            self._config = config


# ── Anonymization Policy ───────────────────────────────────────────────


@dataclass
class AnonymizationRule:
    """Defines when an attribute should be hidden from agent reasoning."""

    attribute_name: str  # e.g., "gender", "age", "ethnicity"
    hide_from_reasoning: bool  # should this be stripped before agent sees it?
    hide_from_methods: list[str] = field(default_factory=list)
    allow_for_methods: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class AnonymizationPolicy:
    """Policy controlling which attributes are hidden from agent reasoning.

    The organization defines anonymization rules.  The governance system
    applies them, stripping attributes before they reach agent reasoning
    for methods where those attributes are not relevant.
    """

    rules: list[AnonymizationRule] = field(default_factory=list)
    default_hide: bool = False  # if True, attributes not in rules are hidden

    def should_hide(self, attribute_name: str, method: str) -> bool:
        """Determine if an attribute should be hidden for a given method."""
        for rule in self.rules:
            if rule.attribute_name == attribute_name:
                # If method is in the explicit allow list, don't hide
                if method in rule.allow_for_methods:
                    return False
                # If method is in the explicit hide list, hide
                if method in rule.hide_from_methods:
                    return True
                # Fall back to the rule's general setting
                return rule.hide_from_reasoning
        # No rule found — use default
        return self.default_hide

    def apply_to_parameters(self, parameters: dict[str, Any], method: str) -> dict[str, Any]:
        """Return a copy of parameters with hidden attributes removed.

        The original dict is never modified.
        """
        result = {}
        for key, value in parameters.items():
            if not self.should_hide(key, method):
                result[key] = value
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [
                {
                    "attribute_name": r.attribute_name,
                    "hide_from_reasoning": r.hide_from_reasoning,
                    "hide_from_methods": r.hide_from_methods,
                    "allow_for_methods": r.allow_for_methods,
                    "description": r.description,
                }
                for r in self.rules
            ],
            "default_hide": self.default_hide,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnonymizationPolicy:
        return cls(
            rules=[AnonymizationRule(**r) for r in d.get("rules", [])],
            default_hide=d.get("default_hide", False),
        )


# ── Helpers ────────────────────────────────────────────────────────────


def _format_time(t: float) -> str:
    """Format a timestamp as ISO 8601 string."""
    import datetime
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()
