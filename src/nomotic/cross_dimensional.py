"""Cross-Dimensional Signal Detector.

The 13 Dimensions article's core insight: governance dimensions interact,
and those interactions reveal patterns that individual dimensions miss.
This component detects significant cross-dimensional signals.

Signals emerge from combinations of dimension scores that, taken
individually, might seem acceptable but in combination indicate a
governance concern.  For example, an action can be technically
compliant (high scope score) while having concerning ethical implications
(low ethical score) — the combination matters more than either alone.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CROSS_DIMENSIONAL_PATTERNS",
    "CrossDimensionalDetector",
    "CrossDimensionalReport",
    "CrossDimensionalSignal",
]


# ── Signal types ───────────────────────────────────────────────────────


@dataclass
class CrossDimensionalSignal:
    """A governance signal emerging from the interaction of multiple dimensions."""

    signal_type: str
    dimensions_involved: list[str] = field(default_factory=list)
    description: str = ""
    severity: str = "info"  # "info", "warning", "alert", "critical"
    action_pattern: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "dimensions_involved": self.dimensions_involved,
            "description": self.description,
            "severity": self.severity,
            "action_pattern": self.action_pattern,
            "recommendation": self.recommendation,
        }


@dataclass
class CrossDimensionalReport:
    """Report of cross-dimensional signals detected."""

    report_id: str
    analysis_type: str  # "single_action", "aggregate", "trend"
    signals: list[CrossDimensionalSignal] = field(default_factory=list)
    summary: str = ""
    generated_at: str = ""
    per_pattern_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "analysis_type": self.analysis_type,
            "signals": [s.to_dict() for s in self.signals],
            "summary": self.summary,
            "generated_at": self.generated_at,
            "per_pattern_counts": self.per_pattern_counts,
        }


# ── Pattern definitions ───────────────────────────────────────────────


CROSS_DIMENSIONAL_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "discriminatory_compliance",
        "description": (
            "Action is technically compliant (scope, authority clear) but "
            "ethical + stakeholder signals suggest discriminatory pattern"
        ),
        "dimensions": [
            "scope_compliance",
            "authority_verification",
            "ethical_alignment",
            "stakeholder_impact",
        ],
        "severity": "alert",
        "recommendation": (
            "Review whether compliant rules are producing inequitable outcomes"
        ),
    },
    {
        "name": "empathetic_exploitation",
        "description": (
            "Security concern combined with ethical design — adversarial "
            "manipulation of empathetic agent behavior"
        ),
        "dimensions": [
            "incident_detection",
            "ethical_alignment",
            "behavioral_consistency",
        ],
        "severity": "critical",
        "recommendation": (
            "Agent's ethical design may be exploited by adversarial input"
        ),
    },
    {
        "name": "invisible_wall",
        "description": (
            "Governance rules produce systematic exclusion — neutral rules, "
            "biased outcomes"
        ),
        "dimensions": [
            "scope_compliance",
            "precedent_alignment",
            "stakeholder_impact",
        ],
        "severity": "alert",
        "recommendation": (
            "Neutral governance rules may be producing systematic exclusion"
        ),
    },
    {
        "name": "trust_authority_mismatch",
        "description": (
            "Agent has high authority but low or declining trust — risk "
            "of capable but unreliable execution"
        ),
        "dimensions": [
            "authority_verification",
            "behavioral_consistency",
            "human_override",
        ],
        "severity": "alert",
        "recommendation": (
            "Agent authority exceeds demonstrated trustworthiness — "
            "consider scope reduction"
        ),
    },
    {
        "name": "cascade_without_isolation",
        "description": (
            "High cascading impact with insufficient isolation — action "
            "effects may propagate uncontrolled"
        ),
        "dimensions": [
            "cascading_impact",
            "isolation_integrity",
        ],
        "severity": "alert",
        "recommendation": (
            "Action has high downstream impact without adequate "
            "containment boundaries"
        ),
    },
    {
        "name": "temporal_security_gap",
        "description": (
            "Action occurring during sensitive time window with "
            "security concerns"
        ),
        "dimensions": [
            "temporal_compliance",
            "incident_detection",
        ],
        "severity": "alert",
        "recommendation": "Suspicious activity during sensitive time window",
    },
    {
        "name": "opacity_under_pressure",
        "description": (
            "Low transparency combined with high-stakes action — "
            "decisions that cannot be explained"
        ),
        "dimensions": [
            "transparency",
            "stakeholder_impact",
            "ethical_alignment",
        ],
        "severity": "alert",
        "recommendation": (
            "High-impact action with insufficient audit trail — "
            "cannot be justified after the fact"
        ),
    },
    {
        "name": "precedent_break_under_drift",
        "description": (
            "Agent breaking from precedent while also showing "
            "behavioral drift"
        ),
        "dimensions": [
            "precedent_alignment",
            "behavioral_consistency",
        ],
        "severity": "warning",
        "recommendation": (
            "Agent departing from established patterns — may indicate "
            "drift or adaptation"
        ),
    },
    {
        "name": "oversight_erosion",
        "description": (
            "Human oversight engagement is degrading — human_override "
            "dimension consistently at 1.0 (no overrides) combined with "
            "external human drift alerts"
        ),
        "dimensions": [
            "human_override",
        ],
        "severity": "high",
        "recommendation": (
            "Human reviewer may not be exercising independent judgment — "
            "verify oversight engagement and review human drift alerts"
        ),
    },
]


# ── Pattern evaluation logic ──────────────────────────────────────────


def _evaluate_pattern(
    pattern: dict[str, Any],
    score_map: dict[str, float],
    trust_state: float | None = None,
) -> bool:
    """Evaluate whether a cross-dimensional pattern's condition is met.

    Each pattern has specific conditions based on its dimension scores.
    """
    name = pattern["name"]

    def _score(dim_name: str) -> float:
        return score_map.get(dim_name, 0.8)  # Default to 0.8 (no concern)

    if name == "discriminatory_compliance":
        # scope and authority score > 0.8 AND ethical OR stakeholder score < 0.5
        scope_ok = _score("scope_compliance") > 0.8
        auth_ok = _score("authority_verification") > 0.8
        ethical_low = _score("ethical_alignment") < 0.5
        stakeholder_low = _score("stakeholder_impact") < 0.5
        return scope_ok and auth_ok and (ethical_low or stakeholder_low)

    elif name == "empathetic_exploitation":
        # incident_detection score < 0.5 AND ethical_alignment > 0.7 AND behavioral_consistency < 0.6
        return (
            _score("incident_detection") < 0.5
            and _score("ethical_alignment") > 0.7
            and _score("behavioral_consistency") < 0.6
        )

    elif name == "invisible_wall":
        # scope_compliance score = 1.0 AND precedent shows high denial rate AND stakeholder impact elevated
        return (
            _score("scope_compliance") >= 0.95
            and _score("precedent_alignment") < 0.5
            and _score("stakeholder_impact") < 0.5
        )

    elif name == "trust_authority_mismatch":
        # authority > 0.8 AND behavioral_consistency < 0.5 AND trust < 0.4
        auth_high = _score("authority_verification") > 0.8
        behavior_low = _score("behavioral_consistency") < 0.5
        trust_low = (trust_state is not None and trust_state < 0.4) or (
            trust_state is None and _score("behavioral_consistency") < 0.4
        )
        return auth_high and behavior_low and trust_low

    elif name == "cascade_without_isolation":
        # cascading_impact score < 0.5 AND isolation_integrity < 0.7
        return (
            _score("cascading_impact") < 0.5
            and _score("isolation_integrity") < 0.7
        )

    elif name == "temporal_security_gap":
        # temporal < 0.7 AND incident_detection < 0.6
        return (
            _score("temporal_compliance") < 0.7
            and _score("incident_detection") < 0.6
        )

    elif name == "opacity_under_pressure":
        # transparency < 0.5 AND stakeholder_impact < 0.5
        return (
            _score("transparency") < 0.5
            and _score("stakeholder_impact") < 0.5
        )

    elif name == "precedent_break_under_drift":
        # precedent < 0.5 AND behavioral_consistency < 0.6
        return (
            _score("precedent_alignment") < 0.5
            and _score("behavioral_consistency") < 0.6
        )

    elif name == "oversight_erosion":
        # human_override score at or near 1.0 (no overrides happening)
        return _score("human_override") >= 0.95

    return False


# ── Cross-Dimensional Detector ────────────────────────────────────────


class CrossDimensionalDetector:
    """Detects governance signals from the interaction of multiple dimensions.

    Operates in two modes:
    1. Single-action: detect signals from one evaluation's dimension scores.
    2. Aggregate: analyze audit trail for patterns over time.
    """

    def __init__(self, patterns: list[dict[str, Any]] | None = None) -> None:
        """Initialize with pattern definitions.  Uses defaults if none provided."""
        self._patterns = list(patterns) if patterns is not None else list(CROSS_DIMENSIONAL_PATTERNS)
        self._custom_patterns: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def detect_signals(
        self,
        dimension_scores: list[Any],
        trust_state: float | None = None,
        action: Any | None = None,
    ) -> list[CrossDimensionalSignal]:
        """Detect cross-dimensional signals from a set of dimension scores.

        Args:
            dimension_scores: List of DimensionScore objects from evaluation.
            trust_state: Current trust level of the agent.
            action: The action being evaluated (for context).

        Returns:
            List of triggered CrossDimensionalSignal objects.
        """
        # Build score map
        score_map: dict[str, float] = {}
        for ds in dimension_scores:
            score_map[ds.dimension_name] = ds.score

        signals: list[CrossDimensionalSignal] = []
        all_patterns = self._patterns + self._custom_patterns

        for pattern in all_patterns:
            if _evaluate_pattern(pattern, score_map, trust_state):
                action_desc = ""
                if action is not None:
                    action_desc = (
                        f"Agent '{getattr(action, 'agent_id', 'unknown')}' "
                        f"performing '{getattr(action, 'action_type', 'unknown')}' "
                        f"on '{getattr(action, 'target', 'unknown')}'"
                    )

                signals.append(
                    CrossDimensionalSignal(
                        signal_type=pattern["name"],
                        dimensions_involved=list(pattern.get("dimensions", [])),
                        description=pattern.get("description", ""),
                        severity=pattern.get("severity", "info"),
                        action_pattern=action_desc,
                        recommendation=pattern.get("recommendation", ""),
                    )
                )

        return signals

    def analyze_aggregate(
        self,
        audit_trail: Any,
        agent_id: str | None = None,
        window_hours: int = 168,
    ) -> CrossDimensionalReport:
        """Analyze audit trail for aggregate cross-dimensional patterns.

        Looks at patterns across many actions to detect repeated signals,
        growing or declining trends, and concentration on specific agents
        or methods.
        """
        now = time.time()
        since = now - (window_hours * 3600)

        query_kwargs: dict[str, Any] = {"since": since, "limit": 100000}
        if agent_id is not None:
            query_kwargs["agent_id"] = agent_id

        records = audit_trail.query(**query_kwargs)

        if not records:
            return CrossDimensionalReport(
                report_id=uuid.uuid4().hex[:12],
                analysis_type="aggregate",
                signals=[],
                summary="No actions in the analysis window.",
                generated_at=_format_time(now),
            )

        # For each record, reconstruct dimension scores and check signals
        all_signals: list[CrossDimensionalSignal] = []
        pattern_counts: dict[str, int] = {}

        for record in records:
            if not record.dimension_scores:
                continue

            # Reconstruct DimensionScore-like objects
            score_map: dict[str, float] = {}
            for ds in record.dimension_scores:
                if isinstance(ds, dict):
                    score_map[ds.get("name", "")] = ds.get("score", 0.8)
                else:
                    score_map[ds.dimension_name] = ds.score

            all_patterns = self._patterns + self._custom_patterns
            for pattern in all_patterns:
                if _evaluate_pattern(pattern, score_map, record.trust_score):
                    name = pattern["name"]
                    pattern_counts[name] = pattern_counts.get(name, 0) + 1

        # Create signals for patterns that appear multiple times
        for pattern in self._patterns + self._custom_patterns:
            name = pattern["name"]
            count = pattern_counts.get(name, 0)
            if count > 0:
                # Severity escalates with frequency
                base_severity = pattern.get("severity", "info")
                if count >= 10:
                    severity = "critical" if base_severity in ("alert", "critical") else "alert"
                elif count >= 5:
                    severity = "alert" if base_severity in ("warning", "alert", "critical") else "warning"
                else:
                    severity = base_severity

                all_signals.append(
                    CrossDimensionalSignal(
                        signal_type=name,
                        dimensions_involved=list(pattern.get("dimensions", [])),
                        description=(
                            f"{pattern.get('description', '')} "
                            f"(detected {count} time(s) in {window_hours}h window)"
                        ),
                        severity=severity,
                        action_pattern=f"Aggregate: {count} occurrences",
                        recommendation=pattern.get("recommendation", ""),
                    )
                )

        summary = self._generate_aggregate_summary(
            records, all_signals, pattern_counts
        )

        return CrossDimensionalReport(
            report_id=uuid.uuid4().hex[:12],
            analysis_type="aggregate",
            signals=all_signals,
            summary=summary,
            generated_at=_format_time(now),
            per_pattern_counts=pattern_counts,
        )

    def add_pattern(self, pattern: dict[str, Any]) -> None:
        """Add a custom cross-dimensional pattern.

        Pattern must include at minimum: name, dimensions, severity.
        """
        required = {"name", "dimensions", "severity"}
        if not required.issubset(pattern.keys()):
            raise ValueError(
                f"Pattern must include keys: {required}. "
                f"Got: {set(pattern.keys())}"
            )
        with self._lock:
            self._custom_patterns.append(pattern)

    def list_patterns(self) -> list[dict[str, Any]]:
        """Return all active patterns (built-in + custom)."""
        with self._lock:
            return list(self._patterns) + list(self._custom_patterns)

    def _generate_aggregate_summary(
        self,
        records: list[Any],
        signals: list[CrossDimensionalSignal],
        pattern_counts: dict[str, int],
    ) -> str:
        """Generate a human-readable summary for aggregate analysis."""
        parts = [f"Analyzed {len(records)} actions for cross-dimensional patterns."]

        if not signals:
            parts.append("No cross-dimensional signals detected.")
        else:
            critical = sum(1 for s in signals if s.severity == "critical")
            alert = sum(1 for s in signals if s.severity == "alert")
            warning = sum(1 for s in signals if s.severity == "warning")
            if critical > 0:
                parts.append(f"{critical} critical signal(s).")
            if alert > 0:
                parts.append(f"{alert} alert signal(s).")
            if warning > 0:
                parts.append(f"{warning} warning signal(s).")

            # Top pattern
            if pattern_counts:
                top = max(pattern_counts, key=pattern_counts.get)  # type: ignore[arg-type]
                parts.append(
                    f"Most frequent: '{top}' ({pattern_counts[top]} occurrences)."
                )

        return " ".join(parts)


# ── Helpers ────────────────────────────────────────────────────────────


def _format_time(t: float) -> str:
    """Format a timestamp as ISO 8601 string."""
    import datetime
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()
