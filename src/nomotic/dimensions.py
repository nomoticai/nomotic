"""The 13 governance dimensions evaluated simultaneously for every action.

Each dimension assesses one aspect of whether an action should proceed.
Dimensions can score, veto, or modify actions independently. The UCS engine
combines their signals into a unified decision.

The 13 dimensions:
 1. Scope Compliance       - Is the action within the agent's authorized scope?
 2. Authority Verification - Does the agent have authority for this action?
 3. Resource Boundaries    - Are resource limits respected?
 4. Behavioral Consistency - Does this match expected behavior patterns?
 5. Cascading Impact       - What are the downstream consequences?
 6. Stakeholder Impact     - Who is affected and how?
 7. Incident Detection     - Does this match known failure patterns?
 8. Isolation Integrity    - Are containment boundaries maintained?
 9. Temporal Compliance    - Is the timing appropriate?
10. Precedent Alignment    - Is this consistent with past governance decisions?
11. Transparency           - Is the action auditable and explainable?
12. Human Override         - Is human intervention required or requested?
13. Ethical Alignment      - Does the action meet ethical constraints?
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from nomotic.types import Action, AgentContext, DimensionScore

__all__ = [
    "AuthorityVerification",
    "BehavioralConsistency",
    "CascadingImpact",
    "DimensionRegistry",
    "EthicalAlignment",
    "GovernanceDimension",
    "HumanOverride",
    "IncidentDetection",
    "IsolationIntegrity",
    "PrecedentAlignment",
    "ResourceBoundaries",
    "ResourceLimits",
    "ScopeCompliance",
    "StakeholderImpact",
    "TemporalCompliance",
    "Transparency",
]


class GovernanceDimension(ABC):
    """Base class for all governance dimensions.

    Each dimension evaluates actions independently and returns a score.
    A score of 0.0 means maximum governance concern (likely deny).
    A score of 1.0 means no governance concern (allow).
    A veto from any dimension overrides all other scores.
    """

    name: str
    weight: float = 1.0
    can_veto: bool = False

    @abstractmethod
    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        """Evaluate an action and return a governance score."""
        ...


# --- Dimension 1: Scope Compliance ---


class ScopeCompliance(GovernanceDimension):
    """Is the action within the agent's authorized scope?

    Checks action type, target, and parameters against the agent's
    defined permissions. Hard boundary — out-of-scope actions are vetoed.
    """

    name = "scope_compliance"
    weight = 1.5
    can_veto = True

    def __init__(self, allowed_scopes: dict[str, set[str]] | None = None):
        self._allowed_scopes: dict[str, set[str]] = allowed_scopes or {}

    def configure_agent_scope(self, agent_id: str, allowed_actions: set[str]) -> None:
        self._allowed_scopes[agent_id] = allowed_actions

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        allowed = self._allowed_scopes.get(context.agent_id, set())
        if not allowed:
            return DimensionScore(
                dimension_name=self.name,
                score=0.5,
                weight=self.weight,
                reasoning="No scope defined for agent; default caution",
            )
        if action.action_type in allowed or "*" in allowed:
            return DimensionScore(
                dimension_name=self.name,
                score=1.0,
                weight=self.weight,
                reasoning=f"Action '{action.action_type}' within scope",
            )
        return DimensionScore(
            dimension_name=self.name,
            score=0.0,
            weight=self.weight,
            veto=True,
            reasoning=f"Action '{action.action_type}' outside agent scope",
        )


# --- Dimension 2: Authority Verification ---


class AuthorityVerification(GovernanceDimension):
    """Does the agent have authority for this specific action?

    Scope says what types of actions are allowed. Authority says whether
    this specific action on this specific target is permitted.
    """

    name = "authority_verification"
    weight = 1.5
    can_veto = True

    def __init__(self) -> None:
        self._authority_checks: list[Callable[[Action, AgentContext], bool]] = []

    def add_authority_check(
        self, check: Callable[[Action, AgentContext], bool]
    ) -> None:
        self._authority_checks.append(check)

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        if not self._authority_checks:
            return DimensionScore(
                dimension_name=self.name,
                score=0.7,
                weight=self.weight,
                reasoning="No authority checks configured; default moderate trust",
            )
        for check in self._authority_checks:
            if not check(action, context):
                return DimensionScore(
                    dimension_name=self.name,
                    score=0.0,
                    weight=self.weight,
                    veto=True,
                    reasoning="Authority check failed",
                )
        return DimensionScore(
            dimension_name=self.name,
            score=1.0,
            weight=self.weight,
            reasoning="All authority checks passed",
        )


# --- Dimension 3: Resource Boundaries ---


@dataclass
class ResourceLimits:
    max_actions_per_minute: int = 60
    max_concurrent_actions: int = 10
    max_cost_per_action: float = float("inf")
    max_total_cost: float = float("inf")


class ResourceBoundaries(GovernanceDimension):
    """Are resource limits respected?

    Tracks rate, concurrency, and cost against configured limits.
    Degrades gracefully — approaching limits lowers the score before
    hitting them triggers a veto.
    """

    name = "resource_boundaries"
    weight = 1.2
    can_veto = True

    def __init__(self, limits: ResourceLimits | None = None):
        self._limits = limits or ResourceLimits()
        self._action_timestamps: list[float] = []
        self._active_count: int = 0
        self._total_cost: float = 0.0
        self._lock = threading.Lock()

    def record_action_start(self, cost: float = 0.0) -> None:
        import time

        with self._lock:
            self._action_timestamps.append(time.time())
            self._active_count += 1
            self._total_cost += cost

    def record_action_end(self) -> None:
        with self._lock:
            self._active_count = max(0, self._active_count - 1)

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        import time

        now = time.time()
        with self._lock:
            # Clean old timestamps
            cutoff = now - 60
            self._action_timestamps = [
                t for t in self._action_timestamps if t > cutoff
            ]
            rate = len(self._action_timestamps)
            active_count = self._active_count
            total_cost = self._total_cost

        # Rate check
        rate_ratio = rate / self._limits.max_actions_per_minute
        if rate_ratio >= 1.0:
            return DimensionScore(
                dimension_name=self.name,
                score=0.0,
                weight=self.weight,
                veto=True,
                reasoning=f"Rate limit exceeded: {rate}/{self._limits.max_actions_per_minute} per minute",
            )

        # Concurrency check
        conc_ratio = active_count / self._limits.max_concurrent_actions
        if conc_ratio >= 1.0:
            return DimensionScore(
                dimension_name=self.name,
                score=0.0,
                weight=self.weight,
                veto=True,
                reasoning=f"Concurrency limit: {active_count}/{self._limits.max_concurrent_actions}",
            )

        # Cost check
        action_cost = action.parameters.get("cost", 0.0)
        if action_cost > self._limits.max_cost_per_action:
            return DimensionScore(
                dimension_name=self.name,
                score=0.0,
                weight=self.weight,
                veto=True,
                reasoning=f"Action cost {action_cost} exceeds limit {self._limits.max_cost_per_action}",
            )
        if total_cost + action_cost > self._limits.max_total_cost:
            return DimensionScore(
                dimension_name=self.name,
                score=0.0,
                weight=self.weight,
                veto=True,
                reasoning=f"Total cost would exceed limit",
            )

        # Graduated score — closer to limits = lower score
        score = 1.0 - max(rate_ratio, conc_ratio) * 0.5
        return DimensionScore(
            dimension_name=self.name,
            score=score,
            weight=self.weight,
            reasoning=f"Within limits (rate={rate}, concurrent={active_count})",
        )


# --- Dimension 4: Behavioral Consistency ---


class BehavioralConsistency(GovernanceDimension):
    """Does this action match expected behavior patterns?

    Compares the action against the agent's established behavioral profile.
    Sudden deviations — new action types, unusual targets, parameter
    anomalies — lower the score.

    When a behavioral fingerprint is available (from the fingerprint
    observer), the evaluation checks against the fingerprint's action
    distribution for richer scoring.  Without a fingerprint, falls back
    to the legacy "have I seen this action type before?" check.
    """

    name = "behavioral_consistency"
    weight = 1.0
    can_veto = False

    def __init__(self) -> None:
        self._known_patterns: dict[str, set[str]] = {}  # agent_id -> action_types seen
        self._fingerprint_accessor: Callable[[str], Any] | None = None
        self._lock = threading.Lock()

    def set_fingerprint_accessor(
        self, accessor: Callable[[str], Any],
    ) -> None:
        """Set a function that retrieves fingerprints for agents.

        Called by the runtime during initialization to wire up
        fingerprint access without circular imports.
        """
        self._fingerprint_accessor = accessor

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        with self._lock:
            seen = self._known_patterns.setdefault(context.agent_id, set())

            # Legacy behavior: check if action type has been seen
            if not seen:
                seen.add(action.action_type)
                return DimensionScore(
                    dimension_name=self.name,
                    score=0.7,
                    weight=self.weight,
                    reasoning="First action; establishing baseline",
                )

            is_novel = action.action_type not in seen
            seen.add(action.action_type)

            # If no fingerprint accessor, fall back to legacy behavior
            if self._fingerprint_accessor is None:
                if is_novel:
                    return DimensionScore(
                        dimension_name=self.name,
                        score=0.5,
                        weight=self.weight,
                        reasoning=f"Novel action type '{action.action_type}' for this agent",
                    )
                return DimensionScore(
                    dimension_name=self.name,
                    score=1.0,
                    weight=self.weight,
                    reasoning="Action type consistent with history",
                )

            # Enhanced behavior: check against fingerprint distribution
            fp = self._fingerprint_accessor(context.agent_id)
            if fp is None or fp.total_observations < 10:
                # Not enough data — fall back to legacy
                if is_novel:
                    return DimensionScore(
                        dimension_name=self.name,
                        score=0.5,
                        weight=self.weight,
                        reasoning=f"Novel action type '{action.action_type}' (insufficient fingerprint data)",
                    )
                return DimensionScore(
                    dimension_name=self.name,
                    score=1.0,
                    weight=self.weight,
                    reasoning="Action type consistent with history",
                )

            # Check if this action type exists in the fingerprint's distribution
            expected_freq = fp.action_distribution.get(action.action_type, 0.0)

            if expected_freq >= 0.05:
                # This is a normal action type (>= 5% of typical activity)
                return DimensionScore(
                    dimension_name=self.name,
                    score=1.0,
                    weight=self.weight,
                    confidence=fp.confidence,
                    reasoning=f"Action type '{action.action_type}' is {expected_freq:.0%} of typical activity",
                )
            elif expected_freq > 0:
                # Rare but observed — mild concern
                return DimensionScore(
                    dimension_name=self.name,
                    score=0.7,
                    weight=self.weight,
                    confidence=fp.confidence,
                    reasoning=f"Action type '{action.action_type}' is rare ({expected_freq:.1%} of typical activity)",
                )
            else:
                # Never seen in fingerprint — moderate concern
                return DimensionScore(
                    dimension_name=self.name,
                    score=0.4,
                    weight=self.weight,
                    confidence=fp.confidence,
                    reasoning=f"Action type '{action.action_type}' has no precedent in behavioral fingerprint",
                )


# --- Dimension 5: Cascading Impact ---


class CascadingImpact(GovernanceDimension):
    """What are the downstream consequences of this action?

    Evaluates whether the action could trigger chains of effects.
    Actions that modify shared state, trigger other agents, or affect
    external systems score lower.
    """

    name = "cascading_impact"
    weight = 1.3
    can_veto = False

    HIGH_IMPACT_TYPES = {"delete", "deploy", "publish", "broadcast", "transfer"}
    MEDIUM_IMPACT_TYPES = {"update", "modify", "write", "send", "create"}

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        action_lower = action.action_type.lower()
        for t in self.HIGH_IMPACT_TYPES:
            if t in action_lower:
                return DimensionScore(
                    dimension_name=self.name,
                    score=0.3,
                    weight=self.weight,
                    reasoning=f"High cascading impact: '{action.action_type}'",
                )
        for t in self.MEDIUM_IMPACT_TYPES:
            if t in action_lower:
                return DimensionScore(
                    dimension_name=self.name,
                    score=0.6,
                    weight=self.weight,
                    reasoning=f"Medium cascading impact: '{action.action_type}'",
                )
        return DimensionScore(
            dimension_name=self.name,
            score=0.9,
            weight=self.weight,
            reasoning="Low cascading impact",
        )


# --- Dimension 6: Stakeholder Impact ---


class StakeholderImpact(GovernanceDimension):
    """Who is affected by this action and how?

    Actions that affect more stakeholders, or more sensitive stakeholders,
    score lower. External-facing actions score lower than internal ones.
    """

    name = "stakeholder_impact"
    weight = 1.2
    can_veto = False

    def __init__(self) -> None:
        self._sensitive_targets: set[str] = set()

    def mark_sensitive(self, *targets: str) -> None:
        self._sensitive_targets.update(targets)

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        if action.target in self._sensitive_targets:
            return DimensionScore(
                dimension_name=self.name,
                score=0.2,
                weight=self.weight,
                reasoning=f"Sensitive target: '{action.target}'",
            )
        # Check for indicators of external impact
        external_indicators = {"customer", "user", "external", "public", "client"}
        if any(ind in action.target.lower() for ind in external_indicators):
            return DimensionScore(
                dimension_name=self.name,
                score=0.4,
                weight=self.weight,
                reasoning="Action targets external stakeholders",
            )
        return DimensionScore(
            dimension_name=self.name,
            score=0.9,
            weight=self.weight,
            reasoning="Internal action with limited stakeholder impact",
        )


# --- Dimension 7: Incident Detection ---


class IncidentDetection(GovernanceDimension):
    """Does this action match known failure or attack patterns?

    Cross-action pattern monitoring. Detects sequences that look like
    known incidents — rapid repeated failures, privilege escalation
    attempts, data exfiltration patterns.
    """

    name = "incident_detection"
    weight = 1.5
    can_veto = True

    def __init__(self) -> None:
        self._patterns: list[Callable[[Action, AgentContext], float | None]] = []

    def add_pattern(
        self, detector: Callable[[Action, AgentContext], float | None]
    ) -> None:
        """Add a pattern detector. Returns a score (0-1) if pattern matches, None otherwise."""
        self._patterns.append(detector)

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        # Check for rapid repeated actions (built-in pattern)
        recent = context.action_history[-20:] if context.action_history else []
        if len(recent) >= 5:
            recent_types = [r.action.action_type for r in recent[-5:]]
            if len(set(recent_types)) == 1 and recent_types[0] == action.action_type:
                return DimensionScore(
                    dimension_name=self.name,
                    score=0.2,
                    weight=self.weight,
                    reasoning=f"Repetitive pattern detected: 5+ identical actions",
                )

        # Run custom pattern detectors
        worst_score = 1.0
        worst_reason = "No incident patterns matched"
        for detector in self._patterns:
            result = detector(action, context)
            if result is not None and result < worst_score:
                worst_score = result
                worst_reason = "Custom incident pattern matched"

        veto = worst_score <= 0.1
        return DimensionScore(
            dimension_name=self.name,
            score=worst_score,
            weight=self.weight,
            veto=veto,
            reasoning=worst_reason,
        )


# --- Dimension 8: Isolation Integrity ---


class IsolationIntegrity(GovernanceDimension):
    """Are containment boundaries maintained?

    Agents should operate within defined boundaries. Actions that cross
    isolation boundaries — accessing other agents' resources, modifying
    shared state without coordination — score lower or get vetoed.
    """

    name = "isolation_integrity"
    weight = 1.4
    can_veto = True

    def __init__(self) -> None:
        self._boundaries: dict[str, set[str]] = {}  # agent_id -> allowed targets

    def set_boundaries(self, agent_id: str, allowed_targets: set[str]) -> None:
        self._boundaries[agent_id] = allowed_targets

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        boundaries = self._boundaries.get(context.agent_id)
        if boundaries is None:
            return DimensionScore(
                dimension_name=self.name,
                score=0.6,
                weight=self.weight,
                reasoning="No isolation boundaries defined",
            )
        if not action.target or action.target in boundaries or "*" in boundaries:
            return DimensionScore(
                dimension_name=self.name,
                score=1.0,
                weight=self.weight,
                reasoning="Within isolation boundaries",
            )
        return DimensionScore(
            dimension_name=self.name,
            score=0.0,
            weight=self.weight,
            veto=True,
            reasoning=f"Target '{action.target}' outside isolation boundaries",
        )


# --- Dimension 9: Temporal Compliance ---


class TemporalCompliance(GovernanceDimension):
    """Is the timing of this action appropriate?

    Some actions should only happen during specific windows. Others have
    minimum intervals between executions. Rate patterns matter.
    """

    name = "temporal_compliance"
    weight = 0.8
    can_veto = True

    def __init__(self) -> None:
        self._time_windows: dict[str, tuple[int, int]] = {}  # action_type -> (start_hour, end_hour)
        self._min_intervals: dict[str, float] = {}  # action_type -> min seconds between
        self._last_execution: dict[str, float] = {}  # (agent_id, action_type) -> timestamp

    def set_time_window(self, action_type: str, start_hour: int, end_hour: int) -> None:
        if not (0 <= start_hour <= 23):
            raise ValueError(f"start_hour must be 0-23, got {start_hour}")
        if not (0 <= end_hour <= 23):
            raise ValueError(f"end_hour must be 0-23, got {end_hour}")
        self._time_windows[action_type] = (start_hour, end_hour)

    def set_min_interval(self, action_type: str, seconds: float) -> None:
        self._min_intervals[action_type] = seconds

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        import time
        from datetime import datetime

        now = datetime.now()

        # Check time window
        window = self._time_windows.get(action.action_type)
        if window:
            start, end = window
            if start <= end:
                in_window = start <= now.hour < end
            else:
                in_window = now.hour >= start or now.hour < end
            if not in_window:
                return DimensionScore(
                    dimension_name=self.name,
                    score=0.0,
                    weight=self.weight,
                    veto=True,
                    reasoning=f"Outside time window ({start}:00-{end}:00)",
                )

        # Check minimum interval
        interval = self._min_intervals.get(action.action_type)
        if interval:
            key = f"{context.agent_id}:{action.action_type}"
            last = self._last_execution.get(key)
            if last and (time.time() - last) < interval:
                elapsed = time.time() - last
                return DimensionScore(
                    dimension_name=self.name,
                    score=0.1,
                    weight=self.weight,
                    veto=True,
                    reasoning=f"Min interval not met ({elapsed:.1f}s < {interval}s)",
                )
            self._last_execution[key] = time.time()

        return DimensionScore(
            dimension_name=self.name,
            score=1.0,
            weight=self.weight,
            reasoning="Temporal constraints satisfied",
        )


# --- Dimension 10: Precedent Alignment ---


class PrecedentAlignment(GovernanceDimension):
    """Is this consistent with past governance decisions?

    Similar actions should get similar governance outcomes. Sudden changes
    in how governance treats an action type indicate something unusual.
    """

    name = "precedent_alignment"
    weight = 0.7
    can_veto = False

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        if not context.action_history:
            return DimensionScore(
                dimension_name=self.name,
                score=0.8,
                weight=self.weight,
                reasoning="No precedent history available",
            )
        # Find similar past actions
        similar = [
            r
            for r in context.action_history
            if r.action.action_type == action.action_type
        ]
        if not similar:
            return DimensionScore(
                dimension_name=self.name,
                score=0.7,
                weight=self.weight,
                reasoning="No precedent for this action type",
            )
        # Check what past verdicts looked like
        from nomotic.types import Verdict

        denied = sum(1 for r in similar if r.verdict.verdict == Verdict.DENY)
        if denied > len(similar) * 0.5:
            return DimensionScore(
                dimension_name=self.name,
                score=0.3,
                weight=self.weight,
                reasoning=f"Similar actions frequently denied ({denied}/{len(similar)})",
            )
        return DimensionScore(
            dimension_name=self.name,
            score=0.9,
            weight=self.weight,
            reasoning=f"Consistent with precedent ({len(similar)} similar actions)",
        )


# --- Dimension 11: Transparency ---


class Transparency(GovernanceDimension):
    """Is the action auditable and explainable?

    Actions must carry enough context to be understood later. Missing
    metadata, opaque parameters, or unexplainable targets lower the score.
    """

    name = "transparency"
    weight = 0.6
    can_veto = False

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        score = 1.0
        reasons = []
        if not action.action_type:
            score -= 0.3
            reasons.append("Missing action_type")
        if not action.target:
            score -= 0.2
            reasons.append("Missing target")
        if not action.agent_id:
            score -= 0.2
            reasons.append("Missing agent_id")
        if not action.parameters and action.action_type not in {"read", "list", "query"}:
            score -= 0.1
            reasons.append("No parameters for non-read action")

        return DimensionScore(
            dimension_name=self.name,
            score=max(0.0, score),
            weight=self.weight,
            reasoning="; ".join(reasons) if reasons else "Fully transparent",
        )


# --- Dimension 12: Human Override ---


class HumanOverride(GovernanceDimension):
    """Is human intervention required or requested?

    Some actions always require human approval. Others require it based on
    context (high value, sensitive target, low trust). This dimension
    enforces human-in-the-loop requirements.
    """

    name = "human_override"
    weight = 2.0  # Highest weight — human override is authoritative
    can_veto = True

    def __init__(self) -> None:
        self._require_human: set[str] = set()  # action types that always need human
        self._trust_threshold: float = 0.3  # below this, require human approval
        self._pending_approvals: set[str] = set()  # action IDs awaiting approval
        self._approved: set[str] = set()  # action IDs that have been approved

    def require_human_for(self, *action_types: str) -> None:
        self._require_human.update(action_types)

    def approve(self, action_id: str) -> None:
        self._approved.add(action_id)
        self._pending_approvals.discard(action_id)

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        # Already approved by human
        if action.id in self._approved:
            return DimensionScore(
                dimension_name=self.name,
                score=1.0,
                weight=self.weight,
                reasoning="Human-approved",
            )

        # Action type requires human approval
        if action.action_type in self._require_human:
            self._pending_approvals.add(action.id)
            return DimensionScore(
                dimension_name=self.name,
                score=0.0,
                weight=self.weight,
                veto=True,
                reasoning=f"Human approval required for '{action.action_type}'",
            )

        # Low trust triggers human requirement
        if context.trust_profile.overall_trust < self._trust_threshold:
            self._pending_approvals.add(action.id)
            return DimensionScore(
                dimension_name=self.name,
                score=0.0,
                weight=self.weight,
                veto=True,
                reasoning=f"Low trust ({context.trust_profile.overall_trust:.2f}) requires human approval",
            )

        return DimensionScore(
            dimension_name=self.name,
            score=1.0,
            weight=self.weight,
            reasoning="No human override required",
        )


# --- Dimension 13: Ethical Alignment ---


class EthicalAlignment(GovernanceDimension):
    """Does the action meet ethical constraints?

    Evaluates against configured ethical rules. These are hard constraints
    that cannot be overridden by high trust or other dimension scores.
    """

    name = "ethical_alignment"
    weight = 2.0
    can_veto = True

    def __init__(self) -> None:
        self._rules: list[Callable[[Action, AgentContext], tuple[bool, str]]] = []

    def add_rule(
        self, rule: Callable[[Action, AgentContext], tuple[bool, str]]
    ) -> None:
        """Add an ethical rule. Returns (passes, reason)."""
        self._rules.append(rule)

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        if not self._rules:
            return DimensionScore(
                dimension_name=self.name,
                score=0.8,
                weight=self.weight,
                reasoning="No ethical rules configured",
            )
        for rule in self._rules:
            passes, reason = rule(action, context)
            if not passes:
                return DimensionScore(
                    dimension_name=self.name,
                    score=0.0,
                    weight=self.weight,
                    veto=True,
                    reasoning=f"Ethical violation: {reason}",
                )
        return DimensionScore(
            dimension_name=self.name,
            score=1.0,
            weight=self.weight,
            reasoning="All ethical rules satisfied",
        )


# --- Registry ---


class DimensionRegistry:
    """Manages all governance dimensions and evaluates actions against them.

    Dimensions are evaluated in parallel (logically — all at once). Each
    produces an independent score. The registry collects these for the
    UCS engine to combine.
    """

    def __init__(self) -> None:
        self._dimensions: dict[str, GovernanceDimension] = {}

    def register(self, dimension: GovernanceDimension) -> None:
        self._dimensions[dimension.name] = dimension

    def get(self, name: str) -> GovernanceDimension | None:
        return self._dimensions.get(name)

    @property
    def dimensions(self) -> list[GovernanceDimension]:
        return list(self._dimensions.values())

    def evaluate_all(
        self, action: Action, context: AgentContext
    ) -> list[DimensionScore]:
        """Evaluate all dimensions simultaneously and return their scores."""
        return [dim.evaluate(action, context) for dim in self._dimensions.values()]

    @classmethod
    def create_default(cls) -> DimensionRegistry:
        """Create a registry with all 13 dimensions using default configuration."""
        registry = cls()
        registry.register(ScopeCompliance())
        registry.register(AuthorityVerification())
        registry.register(ResourceBoundaries())
        registry.register(BehavioralConsistency())
        registry.register(CascadingImpact())
        registry.register(StakeholderImpact())
        registry.register(IncidentDetection())
        registry.register(IsolationIntegrity())
        registry.register(TemporalCompliance())
        registry.register(PrecedentAlignment())
        registry.register(Transparency())
        registry.register(HumanOverride())
        registry.register(EthicalAlignment())
        return registry
