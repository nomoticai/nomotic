"""Bias Detection Engine — structural analysis of governance rules.

The Bias Detection Engine examines the governance *rules themselves* — not
outcomes, but the configuration — for structural bias potential.  A
neutral-looking rule can produce biased outcomes when applied to a
non-neutral population.

Important limitation: the bias detector performs structural analysis, not
causal analysis.  It identifies *potential* for bias based on configuration
patterns.  It cannot determine whether a rule *actually produces* biased
outcomes — that is the equity analyzer's job using real outcome data.

The two components are complementary:
- Bias detector says: 'this rule *could* produce disparate outcomes'
- Equity analyzer says: 'this rule *does* produce disparate outcomes'
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from nomotic.equity import EquityConfig, ProtectedAttribute

__all__ = [
    "BiasDetector",
    "GovernanceBiasReport",
    "RuleBiasAssessment",
    "StructuralConcern",
]


# ── Result types ───────────────────────────────────────────────────────


@dataclass
class RuleBiasAssessment:
    """Assessment of a single governance rule's bias potential."""

    rule_type: str  # "scope", "authority_envelope", "ethical_rule", "weight_config", "resource_limit"
    rule_description: str
    bias_risk: str  # "none_detected", "low", "medium", "high"
    concerns: list[str] = field(default_factory=list)
    affected_attributes: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_type": self.rule_type,
            "rule_description": self.rule_description,
            "bias_risk": self.bias_risk,
            "concerns": self.concerns,
            "affected_attributes": self.affected_attributes,
            "recommendation": self.recommendation,
        }


@dataclass
class StructuralConcern:
    """A system-level pattern that could produce biased outcomes."""

    concern_type: str
    # Types:
    # "uniform_rule_non_uniform_population" — neutral rule, non-neutral impact
    # "proxy_variable_in_rules" — rules reference attributes correlating with protected chars
    # "missing_equity_evaluation" — high-impact rules with no equity monitoring
    # "asymmetric_authority" — different groups effectively get different authority levels
    # "threshold_cliff_effect" — binary thresholds that disproportionately affect certain groups
    description: str
    severity: str  # "low", "medium", "high"
    affected_rules: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "concern_type": self.concern_type,
            "description": self.description,
            "severity": self.severity,
            "affected_rules": self.affected_rules,
            "recommendation": self.recommendation,
        }


@dataclass
class GovernanceBiasReport:
    """Complete bias assessment of governance configuration."""

    report_id: str
    generated_at: str
    rules_assessed: int
    assessments: list[RuleBiasAssessment] = field(default_factory=list)
    structural_concerns: list[StructuralConcern] = field(default_factory=list)
    summary: str = ""
    config_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "rules_assessed": self.rules_assessed,
            "assessments": [a.to_dict() for a in self.assessments],
            "structural_concerns": [c.to_dict() for c in self.structural_concerns],
            "summary": self.summary,
            "config_version": self.config_version,
        }


# ── Bias Detector ──────────────────────────────────────────────────────


class BiasDetector:
    """Examines governance rules for structural bias potential.

    Performs static analysis of governance configuration to identify
    patterns that *could* produce disparate outcomes.
    """

    def __init__(self, equity_config: EquityConfig) -> None:
        self._config = equity_config
        self._lock = threading.Lock()

    @property
    def config(self) -> EquityConfig:
        return self._config

    def assess_configuration(self, runtime: Any) -> GovernanceBiasReport:
        """Assess the current governance configuration for bias potential.

        Examines scope configurations, authority envelopes, ethical rules,
        dimension weights, resource limits, and structural patterns.
        """
        assessments: list[RuleBiasAssessment] = []
        structural_concerns: list[StructuralConcern] = []
        protected_attrs = self._config.protected_attributes
        attr_names = {a.name for a in protected_attrs}
        proxy_names = {a.name for a in protected_attrs if a.is_proxy}

        # 1. Examine scope configurations
        scope_dim = runtime.registry.get("scope_compliance")
        if scope_dim is not None:
            assessment = self._assess_scope(scope_dim, protected_attrs)
            assessments.append(assessment)

        # 2. Examine authority envelopes via dimension checks
        auth_dim = runtime.registry.get("authority_verification")
        if auth_dim is not None:
            assessment = self._assess_authority(auth_dim, protected_attrs)
            assessments.append(assessment)

        # 3. Examine ethical rules
        ethical_dim = runtime.registry.get("ethical_alignment")
        if ethical_dim is not None:
            assessment = self._assess_ethical_rules(ethical_dim, protected_attrs)
            assessments.append(assessment)

        # 4. Examine dimension weights
        weight_assessment = self._assess_weights(runtime.registry, protected_attrs)
        assessments.append(weight_assessment)

        # 5. Examine resource limits
        resource_dim = runtime.registry.get("resource_boundaries")
        if resource_dim is not None:
            assessment = self._assess_resource_limits(resource_dim, protected_attrs)
            assessments.append(assessment)

        # 6. Examine temporal compliance
        temporal_dim = runtime.registry.get("temporal_compliance")
        if temporal_dim is not None:
            assessment = self._assess_temporal(temporal_dim, protected_attrs)
            assessments.append(assessment)

        # 7. Check for structural concerns
        structural_concerns.extend(
            self._detect_structural_concerns(runtime, assessments, protected_attrs)
        )

        # Get config version
        config_version = ""
        if hasattr(runtime, 'provenance_log') and runtime.provenance_log is not None:
            config_version = runtime.provenance_log.current_config_version()

        summary = self._generate_summary(assessments, structural_concerns)

        return GovernanceBiasReport(
            report_id=uuid.uuid4().hex[:12],
            generated_at=_format_time(time.time()),
            rules_assessed=len(assessments),
            assessments=assessments,
            structural_concerns=structural_concerns,
            summary=summary,
            config_version=config_version,
        )

    def assess_rule(
        self,
        rule_type: str,
        rule_config: dict[str, Any],
        protected_attributes: list[ProtectedAttribute],
    ) -> RuleBiasAssessment:
        """Assess a single rule for bias potential."""
        concerns: list[str] = []
        affected: list[str] = []
        attr_names = {a.name for a in protected_attributes}
        proxy_names = {a.name for a in protected_attributes if a.is_proxy}

        # Check if rule references any protected attributes directly
        rule_str = str(rule_config).lower()
        for attr in protected_attributes:
            if attr.name.lower() in rule_str:
                concerns.append(
                    f"Rule references protected attribute '{attr.name}'"
                )
                affected.append(attr.name)

        # Check if rule references proxy attributes
        for attr in protected_attributes:
            if attr.is_proxy and attr.name.lower() in rule_str:
                concerns.append(
                    f"Rule references proxy attribute '{attr.name}' "
                    f"(proxy for '{attr.proxy_for}')"
                )
                if attr.name not in affected:
                    affected.append(attr.name)

        # Check if rule creates binary thresholds (cliff effects)
        for key, value in rule_config.items():
            if isinstance(value, (int, float)) and "threshold" in key.lower():
                concerns.append(
                    f"Binary threshold '{key}={value}' may create cliff effects"
                )

        # Determine risk level
        if not concerns:
            risk = "none_detected"
            recommendation = ""
        elif any("protected attribute" in c for c in concerns):
            risk = "high"
            recommendation = (
                "Rule directly references protected attributes. "
                "Review whether this is necessary and whether it could "
                "produce disparate outcomes."
            )
        elif any("proxy" in c for c in concerns):
            risk = "medium"
            recommendation = (
                "Rule references proxy attributes. Monitor outcomes "
                "using the equity analyzer."
            )
        elif any("cliff" in c for c in concerns):
            risk = "low"
            recommendation = (
                "Rule has binary thresholds that may affect groups differently. "
                "Consider monitoring outcomes near the threshold."
            )
        else:
            risk = "low"
            recommendation = "Minor concern detected. Monitor outcomes."

        return RuleBiasAssessment(
            rule_type=rule_type,
            rule_description=f"{rule_type} configuration",
            bias_risk=risk,
            concerns=concerns,
            affected_attributes=affected,
            recommendation=recommendation,
        )

    def compare_agent_configurations(
        self,
        runtime: Any,
        agent_ids: list[str],
    ) -> list[StructuralConcern]:
        """Compare configurations across agents to detect asymmetric authority."""
        concerns: list[StructuralConcern] = []

        if len(agent_ids) < 2:
            return concerns

        scope_dim = runtime.registry.get("scope_compliance")
        if scope_dim is None:
            return concerns

        # Compare scope sizes between agents
        agent_scopes: dict[str, set[str]] = {}
        for aid in agent_ids:
            if hasattr(scope_dim, "_allowed_scopes"):
                scope = scope_dim._allowed_scopes.get(aid, set())
                agent_scopes[aid] = scope

        if len(agent_scopes) < 2:
            return concerns

        scope_sizes = {aid: len(s) for aid, s in agent_scopes.items()}
        max_scope = max(scope_sizes.values()) if scope_sizes else 0
        min_scope = min(scope_sizes.values()) if scope_sizes else 0

        if max_scope > 0 and min_scope > 0 and max_scope > min_scope * 2:
            max_agent = [a for a, s in scope_sizes.items() if s == max_scope][0]
            min_agent = [a for a, s in scope_sizes.items() if s == min_scope][0]
            concerns.append(
                StructuralConcern(
                    concern_type="asymmetric_authority",
                    description=(
                        f"Agent '{max_agent}' has significantly broader scope "
                        f"({max_scope} types) than '{min_agent}' ({min_scope} types). "
                        f"If these agents serve different populations, this could "
                        f"produce disparate access to services."
                    ),
                    severity="medium",
                    affected_rules=["scope_compliance"],
                    recommendation=(
                        "Review whether the scope difference between agents "
                        "corresponds to population differences."
                    ),
                )
            )

        return concerns

    # ── Private assessment methods ─────────────────────────────────────

    def _assess_scope(
        self,
        scope_dim: Any,
        protected_attrs: list[ProtectedAttribute],
    ) -> RuleBiasAssessment:
        """Assess scope configuration for bias potential."""
        concerns: list[str] = []
        affected: list[str] = []

        if hasattr(scope_dim, "_allowed_scopes"):
            scopes = scope_dim._allowed_scopes
            if scopes:
                # Check if different agents have very different scope sizes
                sizes = [len(s) for s in scopes.values()]
                if sizes and max(sizes) > 0 and min(sizes) > 0:
                    ratio = max(sizes) / min(sizes)
                    if ratio > 3.0:
                        concerns.append(
                            f"Scope size varies by {ratio:.1f}x across agents "
                            f"— may indicate asymmetric access"
                        )

        risk = "none_detected" if not concerns else "low"
        return RuleBiasAssessment(
            rule_type="scope",
            rule_description="Scope compliance configuration",
            bias_risk=risk,
            concerns=concerns,
            affected_attributes=affected,
            recommendation=(
                "Monitor scope differences across agents serving different populations."
                if concerns else ""
            ),
        )

    def _assess_authority(
        self,
        auth_dim: Any,
        protected_attrs: list[ProtectedAttribute],
    ) -> RuleBiasAssessment:
        """Assess authority verification for bias potential."""
        concerns: list[str] = []
        affected: list[str] = []

        if hasattr(auth_dim, "_authority_checks"):
            check_count = len(auth_dim._authority_checks)
            if check_count == 0:
                concerns.append(
                    "No authority checks configured — all claims accepted. "
                    "Cannot detect differential authority patterns."
                )

        risk = "none_detected" if not concerns else "low"
        return RuleBiasAssessment(
            rule_type="authority_envelope",
            rule_description="Authority verification configuration",
            bias_risk=risk,
            concerns=concerns,
            affected_attributes=affected,
            recommendation=(
                "Consider adding authority checks that can be monitored for equity."
                if concerns else ""
            ),
        )

    def _assess_ethical_rules(
        self,
        ethical_dim: Any,
        protected_attrs: list[ProtectedAttribute],
    ) -> RuleBiasAssessment:
        """Assess ethical rules for bias potential."""
        concerns: list[str] = []
        affected: list[str] = []

        if hasattr(ethical_dim, "_rules"):
            rule_count = len(ethical_dim._rules)
            if rule_count == 0:
                concerns.append(
                    "No ethical rules configured — ethical dimension has no criteria."
                )
            # Check for common missing patterns
            if rule_count > 0 and len(protected_attrs) > 0:
                concerns.append(
                    f"{rule_count} ethical rule(s) configured but equity monitoring "
                    f"has {len(protected_attrs)} protected attribute(s). "
                    f"Ensure ethical rules do not inadvertently affect monitored populations."
                )

        risk = "none_detected" if not concerns else "low"
        return RuleBiasAssessment(
            rule_type="ethical_rule",
            rule_description="Ethical alignment rules",
            bias_risk=risk,
            concerns=concerns,
            affected_attributes=affected,
            recommendation=(
                "Review ethical rules against equity monitoring configuration."
                if concerns else ""
            ),
        )

    def _assess_weights(
        self,
        registry: Any,
        protected_attrs: list[ProtectedAttribute],
    ) -> RuleBiasAssessment:
        """Assess dimension weight configuration for bias potential."""
        concerns: list[str] = []
        affected: list[str] = []

        # Check if ethical/stakeholder dimensions have low weight
        for dim_name in ("ethical_alignment", "stakeholder_impact"):
            dim = registry.get(dim_name)
            if dim is not None and dim.weight < 0.5:
                concerns.append(
                    f"Dimension '{dim_name}' has low weight ({dim.weight}) — "
                    f"ethical and stakeholder signals may be underweighted"
                )

        risk = "none_detected" if not concerns else "low"
        return RuleBiasAssessment(
            rule_type="weight_config",
            rule_description="Dimension weight configuration",
            bias_risk=risk,
            concerns=concerns,
            affected_attributes=affected,
            recommendation=(
                "Consider increasing weight for ethical and stakeholder dimensions."
                if concerns else ""
            ),
        )

    def _assess_resource_limits(
        self,
        resource_dim: Any,
        protected_attrs: list[ProtectedAttribute],
    ) -> RuleBiasAssessment:
        """Assess resource limits for bias potential."""
        concerns: list[str] = []
        affected: list[str] = []

        if hasattr(resource_dim, "_limits"):
            limits = resource_dim._limits
            if limits:
                # Check for very restrictive rate limits
                for agent_id, lim in limits.items():
                    if hasattr(lim, "rate_per_minute") and lim.rate_per_minute < 5:
                        concerns.append(
                            f"Agent '{agent_id}' has restrictive rate limit "
                            f"({lim.rate_per_minute}/min) — may limit service "
                            f"to high-volume populations"
                        )

        risk = "none_detected" if not concerns else "low"
        return RuleBiasAssessment(
            rule_type="resource_limit",
            rule_description="Resource boundary configuration",
            bias_risk=risk,
            concerns=concerns,
            affected_attributes=affected,
            recommendation=(
                "Review rate limits for equity across agent populations."
                if concerns else ""
            ),
        )

    def _assess_temporal(
        self,
        temporal_dim: Any,
        protected_attrs: list[ProtectedAttribute],
    ) -> RuleBiasAssessment:
        """Assess temporal compliance for bias potential."""
        concerns: list[str] = []
        affected: list[str] = []

        if hasattr(temporal_dim, "_time_windows"):
            windows = temporal_dim._time_windows
            if windows:
                for action_type, window in windows.items():
                    start, end = window if len(window) >= 2 else (0, 24)
                    window_hours = (end - start) % 24
                    if window_hours < 8:
                        concerns.append(
                            f"Action '{action_type}' restricted to {window_hours}-hour "
                            f"window — may disadvantage populations in different time zones"
                        )

        risk = "none_detected" if not concerns else "low"
        return RuleBiasAssessment(
            rule_type="temporal",
            rule_description="Temporal compliance configuration",
            bias_risk=risk,
            concerns=concerns,
            affected_attributes=affected,
            recommendation=(
                "Review temporal restrictions for impact across time zones."
                if concerns else ""
            ),
        )

    def _detect_structural_concerns(
        self,
        runtime: Any,
        assessments: list[RuleBiasAssessment],
        protected_attrs: list[ProtectedAttribute],
    ) -> list[StructuralConcern]:
        """Detect system-level patterns that could produce biased outcomes."""
        concerns: list[StructuralConcern] = []

        # Check for missing equity evaluation on high-impact rules
        high_risk_rules = [a for a in assessments if a.bias_risk in ("medium", "high")]
        if high_risk_rules and not protected_attrs:
            concerns.append(
                StructuralConcern(
                    concern_type="missing_equity_evaluation",
                    description=(
                        f"{len(high_risk_rules)} rule(s) with elevated bias risk "
                        f"but no protected attributes configured for equity monitoring"
                    ),
                    severity="high",
                    affected_rules=[r.rule_type for r in high_risk_rules],
                    recommendation=(
                        "Configure protected attributes in EquityConfig to enable "
                        "outcome monitoring for rules with bias potential."
                    ),
                )
            )

        # Check for uniform rules on potentially non-uniform populations
        scope_dim = runtime.registry.get("scope_compliance")
        if scope_dim is not None and hasattr(scope_dim, "_allowed_scopes"):
            # If all agents have identical scopes, it's a uniform rule
            scopes = list(scope_dim._allowed_scopes.values())
            if len(scopes) > 1 and all(s == scopes[0] for s in scopes):
                concerns.append(
                    StructuralConcern(
                        concern_type="uniform_rule_non_uniform_population",
                        description=(
                            "All agents have identical scope configurations. "
                            "If agents serve different populations, a uniform "
                            "rule may produce disparate outcomes."
                        ),
                        severity="low",
                        affected_rules=["scope_compliance"],
                        recommendation=(
                            "Review whether uniform scope is appropriate given "
                            "population differences across agents."
                        ),
                    )
                )

        # Check for proxy variables in configuration
        proxy_attrs = [a for a in protected_attrs if a.is_proxy]
        if proxy_attrs:
            concerns.append(
                StructuralConcern(
                    concern_type="proxy_variable_in_rules",
                    description=(
                        f"Configuration includes {len(proxy_attrs)} proxy attribute(s) "
                        f"({', '.join(a.name for a in proxy_attrs)}). "
                        f"Governance rules that reference these attributes may produce "
                        f"disparate outcomes correlated with protected characteristics."
                    ),
                    severity="medium",
                    affected_rules=["equity_config"],
                    recommendation=(
                        "Monitor outcomes using the equity analyzer to detect "
                        "actual proxy discrimination."
                    ),
                )
            )

        return concerns

    def _generate_summary(
        self,
        assessments: list[RuleBiasAssessment],
        structural_concerns: list[StructuralConcern],
    ) -> str:
        """Generate a human-readable summary."""
        parts = [f"Assessed {len(assessments)} governance rule configuration(s)."]

        high_risk = sum(1 for a in assessments if a.bias_risk == "high")
        medium_risk = sum(1 for a in assessments if a.bias_risk == "medium")
        low_risk = sum(1 for a in assessments if a.bias_risk == "low")

        if high_risk > 0:
            parts.append(f"{high_risk} high-risk finding(s).")
        if medium_risk > 0:
            parts.append(f"{medium_risk} medium-risk finding(s).")
        if low_risk > 0:
            parts.append(f"{low_risk} low-risk finding(s).")
        if high_risk == 0 and medium_risk == 0 and low_risk == 0:
            parts.append("No bias risk detected in current configuration.")

        if structural_concerns:
            parts.append(
                f"{len(structural_concerns)} structural concern(s) identified."
            )

        return " ".join(parts)

    def update_config(self, config: EquityConfig) -> None:
        """Update the equity configuration."""
        with self._lock:
            self._config = config


# ── Helpers ────────────────────────────────────────────────────────────


def _format_time(t: float) -> str:
    """Format a timestamp as ISO 8601 string."""
    import datetime
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()
