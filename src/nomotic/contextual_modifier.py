"""Contextual Modifier — adjusts governance posture based on situational factors.

Sits between context profiles and dimensional evaluation. Reads context profile
data and produces weight adjustments, additional constraints, and governance
signals that change how the 13 dimensions evaluate an action — without rewriting
the dimensions themselves.

The same action by the same agent gets different governance treatment depending
on the situation. Think of it as the difference between a referee applying the
same rulebook in a regular season game versus a championship final. The rules
don't change. The scrutiny does.

The modifier does NOT make governance decisions. It adjusts the landscape in
which decisions are made: dimension weights, evaluation constraints, and risk
flags — all before the dimensions see the action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nomotic.context_profile import (
    ContextProfile,
    ExternalContext,
    FeedbackContext,
    HistoricalContext,
    InputContext,
    MetaContext,
    RelationalContext,
    SituationalContext,
    TemporalContext,
    WorkflowContext,
)
from nomotic.types import Action, AgentContext

__all__ = [
    "ContextConstraint",
    "ContextModification",
    "ContextRiskSignal",
    "ContextualModifier",
    "ModifierConfig",
    "WeightAdjustment",
]


# ── Data structures produced by the modifier ────────────────────────────


@dataclass
class WeightAdjustment:
    """A single weight change for one governance dimension."""

    dimension_name: str
    original_weight: float
    adjusted_weight: float
    reason: str  # why this weight changed

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension_name": self.dimension_name,
            "original_weight": self.original_weight,
            "adjusted_weight": self.adjusted_weight,
            "reason": self.reason,
        }


@dataclass
class ContextConstraint:
    """An additional constraint imposed by context analysis."""

    constraint_type: str  # "require_human_review", "elevated_audit", "scope_restriction", "confirmation_required", "reduced_authority"
    description: str
    source_context: str  # which context type produced this constraint
    severity: str  # "advisory", "recommended", "required"

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "description": self.description,
            "source_context": self.source_context,
            "severity": self.severity,
        }


@dataclass
class ContextRiskSignal:
    """A risk signal derived from context analysis."""

    signal_type: str  # e.g., "compound_capability", "unresolved_dependency", "trust_trajectory_concern", "input_anomaly"
    description: str
    source_context: str
    severity: str  # "low", "medium", "high", "critical"
    affected_dimensions: list[str]  # which dimensions this signal is relevant to

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "description": self.description,
            "source_context": self.source_context,
            "severity": self.severity,
            "affected_dimensions": self.affected_dimensions,
        }


@dataclass
class ContextModification:
    """The complete set of modifications the Contextual Modifier produces for a single evaluation."""

    weight_adjustments: list[WeightAdjustment] = field(default_factory=list)
    constraints: list[ContextConstraint] = field(default_factory=list)
    risk_signals: list[ContextRiskSignal] = field(default_factory=list)
    trust_modifier: float = 0.0  # additional trust shift, clamped to configured range
    recommended_flow: str | None = None  # "full", "summary", "posthoc"
    context_completeness: float = 0.0  # 0.0-1.0, from the profile
    modification_reasoning: str = ""  # human-readable explanation

    def has_modifications(self) -> bool:
        """Whether any modifications were produced."""
        return bool(
            self.weight_adjustments
            or self.constraints
            or self.risk_signals
            or self.trust_modifier != 0.0
        )

    def critical_signals(self) -> list[ContextRiskSignal]:
        """Risk signals at critical severity."""
        return [s for s in self.risk_signals if s.severity == "critical"]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "weight_adjustments": [w.to_dict() for w in self.weight_adjustments],
            "constraints": [c.to_dict() for c in self.constraints],
            "risk_signals": [s.to_dict() for s in self.risk_signals],
            "trust_modifier": self.trust_modifier,
            "context_completeness": self.context_completeness,
            "modification_reasoning": self.modification_reasoning,
            "has_modifications": self.has_modifications(),
        }
        if self.recommended_flow is not None:
            d["recommended_flow"] = self.recommended_flow
        return d


# ── Configuration ───────────────────────────────────────────────────────


@dataclass
class ModifierConfig:
    """Tunable parameters for the contextual modifier."""

    enable_workflow_modifiers: bool = True
    enable_situational_modifiers: bool = True
    enable_relational_modifiers: bool = True
    enable_temporal_modifiers: bool = True
    enable_historical_modifiers: bool = True
    enable_input_modifiers: bool = True
    enable_external_modifiers: bool = True
    enable_meta_modifiers: bool = True
    enable_feedback_modifiers: bool = True
    thin_context_threshold: float = 0.3  # below this completeness, flag as risk
    max_weight_adjustment: float = 0.5  # cap on how much any single rule can change a weight
    trust_modifier_range: float = 0.1  # max trust adjustment in either direction


# ── Dimension name constants ────────────────────────────────────────────

_SECURITY_DIMENSIONS = [
    "incident_detection",
    "isolation_integrity",
]

_ALL_DIMENSION_NAMES = [
    "scope_compliance",
    "authority_verification",
    "resource_boundaries",
    "behavioral_consistency",
    "cascading_impact",
    "stakeholder_impact",
    "incident_detection",
    "isolation_integrity",
    "temporal_compliance",
    "precedent_alignment",
    "transparency",
    "human_override",
    "ethical_alignment",
]

# Methods/categories considered high-governance for thin context assessment
_HIGH_GOVERNANCE_METHODS = {
    "transaction", "decision", "security", "delete", "execute",
    "transfer", "deploy", "terminate", "modify_permissions",
}


# ── The Engine ──────────────────────────────────────────────────────────


class ContextualModifier:
    """Analyzes context profiles and produces governance modifications.

    Thread-safe: each call to modify() is independent. No mutable shared
    state is modified during analysis — all results are returned as new
    objects.
    """

    def __init__(self, config: ModifierConfig | None = None) -> None:
        self.config = config or ModifierConfig()

    def modify(
        self,
        action: Action,
        context: AgentContext,
        profile: ContextProfile,
    ) -> ContextModification:
        """Analyze the context profile and produce governance modifications.

        This is the main entry point. Runs all enabled modifier rules
        against the context profile and aggregates results.
        """
        all_adjustments: list[WeightAdjustment] = []
        all_constraints: list[ContextConstraint] = []
        all_signals: list[ContextRiskSignal] = []
        trust_mod = 0.0
        reasoning_parts: list[str] = []

        # Run each context-type analyzer if enabled and context is present
        if self.config.enable_workflow_modifiers and profile.workflow is not None:
            adj, con, sig = self._analyze_workflow(action, context, profile.workflow)
            all_adjustments.extend(adj)
            all_constraints.extend(con)
            all_signals.extend(sig)
            if adj or con or sig:
                reasoning_parts.append(f"Workflow context: {len(adj)} weight adjustments, {len(con)} constraints, {len(sig)} risk signals")

        if self.config.enable_situational_modifiers and profile.situational is not None:
            adj, con, sig = self._analyze_situational(action, context, profile.situational)
            all_adjustments.extend(adj)
            all_constraints.extend(con)
            all_signals.extend(sig)
            if adj or con or sig:
                reasoning_parts.append(f"Situational context: {len(adj)} weight adjustments, {len(con)} constraints, {len(sig)} risk signals")

        if self.config.enable_relational_modifiers and profile.relational is not None:
            adj, con, sig = self._analyze_relational(action, context, profile.relational)
            all_adjustments.extend(adj)
            all_constraints.extend(con)
            all_signals.extend(sig)
            if adj or con or sig:
                reasoning_parts.append(f"Relational context: {len(adj)} weight adjustments, {len(con)} constraints, {len(sig)} risk signals")

        if self.config.enable_temporal_modifiers and profile.temporal is not None:
            adj, con, sig = self._analyze_temporal(action, context, profile.temporal, profile.situational)
            all_adjustments.extend(adj)
            all_constraints.extend(con)
            all_signals.extend(sig)
            if adj or con or sig:
                reasoning_parts.append(f"Temporal context: {len(adj)} weight adjustments, {len(con)} constraints, {len(sig)} risk signals")

        if self.config.enable_historical_modifiers and profile.historical is not None:
            adj, con, sig, t_mod = self._analyze_historical(action, context, profile.historical)
            all_adjustments.extend(adj)
            all_constraints.extend(con)
            all_signals.extend(sig)
            trust_mod += t_mod
            if adj or con or sig or t_mod != 0.0:
                reasoning_parts.append(f"Historical context: {len(adj)} weight adjustments, {len(con)} constraints, {len(sig)} risk signals, trust_modifier={t_mod:+.3f}")

        if self.config.enable_input_modifiers and profile.input_context is not None:
            adj, con, sig = self._analyze_input(action, context, profile.input_context)
            all_adjustments.extend(adj)
            all_constraints.extend(con)
            all_signals.extend(sig)
            if adj or con or sig:
                reasoning_parts.append(f"Input context: {len(adj)} weight adjustments, {len(con)} constraints, {len(sig)} risk signals")

        if self.config.enable_external_modifiers and profile.external is not None:
            adj, con, sig = self._analyze_external(action, context, profile.external)
            all_adjustments.extend(adj)
            all_constraints.extend(con)
            all_signals.extend(sig)
            if adj or con or sig:
                reasoning_parts.append(f"External context: {len(adj)} weight adjustments, {len(con)} constraints, {len(sig)} risk signals")

        if self.config.enable_meta_modifiers and profile.meta is not None:
            adj, con, sig, t_mod = self._analyze_meta(action, context, profile.meta)
            all_adjustments.extend(adj)
            all_constraints.extend(con)
            all_signals.extend(sig)
            trust_mod += t_mod
            if adj or con or sig or t_mod != 0.0:
                reasoning_parts.append(f"Meta context: {len(adj)} weight adjustments, {len(con)} constraints, {len(sig)} risk signals, trust_modifier={t_mod:+.3f}")

        if self.config.enable_feedback_modifiers and profile.feedback is not None:
            adj, con, sig, t_mod = self._analyze_feedback(action, context, profile.feedback)
            all_adjustments.extend(adj)
            all_constraints.extend(con)
            all_signals.extend(sig)
            trust_mod += t_mod
            if adj or con or sig or t_mod != 0.0:
                reasoning_parts.append(f"Feedback context: {len(adj)} weight adjustments, {len(con)} constraints, {len(sig)} risk signals, trust_modifier={t_mod:+.3f}")

        # Thin context assessment
        completeness = profile.completeness_score()
        thin_signals = self._assess_context_completeness(action, profile)
        all_signals.extend(thin_signals)
        if thin_signals:
            reasoning_parts.append(f"Thin context: {len(thin_signals)} risk signals (completeness={completeness:.2f})")

        # Clamp trust modifier to configured range
        trust_mod = max(-self.config.trust_modifier_range, min(self.config.trust_modifier_range, trust_mod))

        # Determine recommended flow
        recommended_flow = self._recommend_flow(action, profile, all_signals, completeness)

        # Build reasoning
        if not reasoning_parts:
            reasoning = "No contextual modifications applied."
        else:
            reasoning = "Contextual modifications applied: " + "; ".join(reasoning_parts) + "."

        return ContextModification(
            weight_adjustments=all_adjustments,
            constraints=all_constraints,
            risk_signals=all_signals,
            trust_modifier=trust_mod,
            recommended_flow=recommended_flow,
            context_completeness=completeness,
            modification_reasoning=reasoning,
        )

    # ── Context-type analyzers ──────────────────────────────────────────

    def _analyze_workflow(
        self,
        action: Action,
        context: AgentContext,
        workflow: WorkflowContext,
    ) -> tuple[list[WeightAdjustment], list[ContextConstraint], list[ContextRiskSignal]]:
        adjustments: list[WeightAdjustment] = []
        constraints: list[ContextConstraint] = []
        signals: list[ContextRiskSignal] = []

        total = workflow.total_steps
        current = workflow.current_step

        # Early steps get +0.3 to cascading_impact weight
        if total > 2 and current <= 2:
            adjustments.append(WeightAdjustment(
                dimension_name="cascading_impact",
                original_weight=1.3,
                adjusted_weight=min(1.3 + 0.3, 3.0),
                reason=f"Early workflow step ({current}/{total}): decisions constrain future steps",
            ))

        # Final steps get -0.2 to cascading_impact
        if total > 2 and current >= total:
            adjustments.append(WeightAdjustment(
                dimension_name="cascading_impact",
                original_weight=1.3,
                adjusted_weight=max(1.3 - 0.2, 0.1),
                reason=f"Final workflow step ({current}/{total}): limited downstream effect",
            ))

        # Unresolved dependencies: current step depends on uncompleted steps
        completed_step_numbers = {s.step_number for s in workflow.steps_completed}
        for dep in workflow.dependencies:
            if dep.to_step == current and dep.dependency_type == "requires":
                if dep.from_step not in completed_step_numbers:
                    signals.append(ContextRiskSignal(
                        signal_type="unresolved_dependency",
                        description=f"Step {current} requires step {dep.from_step} which has not completed",
                        source_context="workflow",
                        severity="critical",
                        affected_dimensions=["cascading_impact", "scope_compliance"],
                    ))

        # Steps with "constrains" dependencies get elevated authority verification
        for dep in workflow.dependencies:
            if dep.from_step == current and dep.dependency_type == "constrains":
                adjustments.append(WeightAdjustment(
                    dimension_name="authority_verification",
                    original_weight=1.5,
                    adjusted_weight=min(1.5 + 0.2, 3.0),
                    reason=f"Step {current} constrains step {dep.to_step}: commitment narrows future options",
                ))
                break  # only add once

        # Rollback capability affects risk posture
        current_step_id = str(current)
        steps_remaining = total - current
        if current_step_id not in workflow.rollback_points and steps_remaining > 2:
            adjustments.append(WeightAdjustment(
                dimension_name="cascading_impact",
                original_weight=1.3,
                adjusted_weight=min(1.3 + 0.2, 3.0),
                reason=f"Step {current} is not rollback-capable with {steps_remaining} steps remaining",
            ))

        # High denial rate in workflow
        if workflow.steps_completed:
            denied = sum(1 for s in workflow.steps_completed if s.verdict.upper() == "DENY")
            rate = denied / len(workflow.steps_completed)
            if rate > 0.3:
                constraints.append(ContextConstraint(
                    constraint_type="elevated_audit",
                    description=f"High denial rate ({rate:.0%}) in workflow {workflow.workflow_id}",
                    source_context="workflow",
                    severity="recommended",
                ))

        # Workflow in "rolling_back" status
        if workflow.status == "rolling_back":
            for dim_name in _SECURITY_DIMENSIONS:
                adjustments.append(WeightAdjustment(
                    dimension_name=dim_name,
                    original_weight=1.5 if dim_name == "incident_detection" else 1.4,
                    adjusted_weight=min((1.5 if dim_name == "incident_detection" else 1.4) + 0.3, 3.0),
                    reason="Workflow in rollback status: heightened security scrutiny",
                ))

        return adjustments, constraints, signals

    def _analyze_situational(
        self,
        action: Action,
        context: AgentContext,
        situational: SituationalContext,
    ) -> tuple[list[WeightAdjustment], list[ContextConstraint], list[ContextRiskSignal]]:
        adjustments: list[WeightAdjustment] = []
        constraints: list[ContextConstraint] = []
        signals: list[ContextRiskSignal] = []

        # Agent-initiated origin
        if situational.origin == "agent_initiated":
            adjustments.append(WeightAdjustment(
                dimension_name="human_override",
                original_weight=2.0,
                adjusted_weight=min(2.0 + 0.3, 3.0),
                reason="Agent-initiated action: governance should be more attentive",
            ))

        # Escalation received
        if situational.origin == "escalation_received":
            adjustments.append(WeightAdjustment(
                dimension_name="human_override",
                original_weight=2.0,
                adjusted_weight=max(2.0 - 0.1, 0.1),
                reason="Escalation received: already reviewed by escalating authority",
            ))
            adjustments.append(WeightAdjustment(
                dimension_name="authority_verification",
                original_weight=1.5,
                adjusted_weight=min(1.5 + 0.2, 3.0),
                reason="Escalation received: verify the escalation is legitimate",
            ))

        # Incident response mode
        if situational.operational_mode == "incident_response":
            for dim_name in _SECURITY_DIMENSIONS:
                adjustments.append(WeightAdjustment(
                    dimension_name=dim_name,
                    original_weight=1.5 if dim_name == "incident_detection" else 1.4,
                    adjusted_weight=min((1.5 if dim_name == "incident_detection" else 1.4) + 0.2, 3.0),
                    reason="Incident response mode: elevated security scrutiny",
                ))
            adjustments.append(WeightAdjustment(
                dimension_name="temporal_compliance",
                original_weight=0.8,
                adjusted_weight=max(0.8 - 0.1, 0.1),
                reason="Incident response mode: relaxed timing constraints",
            ))

        # Critical urgency
        if situational.urgency == "critical":
            for dim_name in _ALL_DIMENSION_NAMES:
                adjustments.append(WeightAdjustment(
                    dimension_name=dim_name,
                    original_weight=1.0,  # placeholder — actual weight applied during clamping
                    adjusted_weight=1.1,  # +0.1 delta applied during aggregation
                    reason="Critical urgency: heightened scrutiny across all dimensions",
                ))

        # Degraded mode
        if situational.operational_mode == "degraded":
            constraints.append(ContextConstraint(
                constraint_type="require_human_review",
                description="System in degraded mode: automated governance has reduced certainty",
                source_context="situational",
                severity="recommended",
            ))

        return adjustments, constraints, signals

    def _analyze_relational(
        self,
        action: Action,
        context: AgentContext,
        relational: RelationalContext,
    ) -> tuple[list[WeightAdjustment], list[ContextConstraint], list[ContextRiskSignal]]:
        adjustments: list[WeightAdjustment] = []
        constraints: list[ContextConstraint] = []
        signals: list[ContextRiskSignal] = []

        # Long delegation chain (>3 hops)
        if len(relational.delegation_chain) > 3:
            signals.append(ContextRiskSignal(
                signal_type="deep_delegation_chain",
                description=f"Delegation chain depth: {len(relational.delegation_chain)} hops. Authority dilutes with each hop.",
                source_context="relational",
                severity="medium",
                affected_dimensions=["authority_verification"],
            ))
            adjustments.append(WeightAdjustment(
                dimension_name="authority_verification",
                original_weight=1.5,
                adjusted_weight=min(1.5 + 0.2, 3.0),
                reason=f"Deep delegation chain ({len(relational.delegation_chain)} hops): increased authority verification",
            ))

        # Compound capability detected
        if relational.compound_methods:
            signals.append(ContextRiskSignal(
                signal_type="compound_capability",
                description=f"Compound capability detected: {len(relational.compound_methods)} combined methods across agents",
                source_context="relational",
                severity="critical",
                affected_dimensions=["scope_compliance", "isolation_integrity"],
            ))
            adjustments.append(WeightAdjustment(
                dimension_name="scope_compliance",
                original_weight=1.5,
                adjusted_weight=min(1.5 + 0.3, 3.0),
                reason="Compound capability detected: elevated scope compliance",
            ))
            adjustments.append(WeightAdjustment(
                dimension_name="isolation_integrity",
                original_weight=1.4,
                adjusted_weight=min(1.4 + 0.3, 3.0),
                reason="Compound capability detected: elevated isolation integrity",
            ))

        # Multiple agents in shared workflow
        if len(relational.shared_workflow_agents) > 1:
            adjustments.append(WeightAdjustment(
                dimension_name="isolation_integrity",
                original_weight=1.4,
                adjusted_weight=min(1.4 + 0.1, 3.0),
                reason=f"Shared workflow with {len(relational.shared_workflow_agents)} agents: increased coordination risk",
            ))

        # Agent has active child delegations
        if relational.child_agent_ids:
            adjustments.append(WeightAdjustment(
                dimension_name="cascading_impact",
                original_weight=1.3,
                adjusted_weight=min(1.3 + 0.1, 3.0),
                reason=f"Agent has {len(relational.child_agent_ids)} active child delegations",
            ))

        return adjustments, constraints, signals

    def _analyze_temporal(
        self,
        action: Action,
        context: AgentContext,
        temporal: TemporalContext,
        situational: SituationalContext | None = None,
    ) -> tuple[list[WeightAdjustment], list[ContextConstraint], list[ContextRiskSignal]]:
        adjustments: list[WeightAdjustment] = []
        constraints: list[ContextConstraint] = []
        signals: list[ContextRiskSignal] = []

        # Post-incident operational state
        if temporal.operational_state == "post_incident":
            adjustments.append(WeightAdjustment(
                dimension_name="incident_detection",
                original_weight=1.5,
                adjusted_weight=min(1.5 + 0.2, 3.0),
                reason="Post-incident operational state: heightened alertness",
            ))
            for dim_name in _SECURITY_DIMENSIONS:
                if dim_name != "incident_detection":
                    adjustments.append(WeightAdjustment(
                        dimension_name=dim_name,
                        original_weight=1.4,
                        adjusted_weight=min(1.4 + 0.1, 3.0),
                        reason="Post-incident operational state: elevated security scrutiny",
                    ))

        # Active critical external events
        critical_events = [
            e for e in temporal.recent_events
            if e.event_type in ("critical", "security_incident", "system_failure")
        ]
        if critical_events:
            for dim_name in _ALL_DIMENSION_NAMES:
                adjustments.append(WeightAdjustment(
                    dimension_name=dim_name,
                    original_weight=1.0,
                    adjusted_weight=1.1,
                    reason=f"Active critical events ({len(critical_events)}): heightened scrutiny",
                ))

        # Time pressure high with deadline approaching
        if temporal.time_pressure == "high" and temporal.deadline:
            signals.append(ContextRiskSignal(
                signal_type="deadline_pressure",
                description=f"High time pressure with deadline: {temporal.deadline}. Governance maintains full evaluation.",
                source_context="temporal",
                severity="medium",
                affected_dimensions=["transparency", "ethical_alignment"],
            ))

        # After hours + agent-initiated
        is_agent_initiated = (
            situational is not None and situational.origin == "agent_initiated"
        )
        if temporal.time_of_day_category == "after_hours" and is_agent_initiated:
            adjustments.append(WeightAdjustment(
                dimension_name="human_override",
                original_weight=2.0,
                adjusted_weight=min(2.0 + 0.2, 3.0),
                reason="Agent acting on its own outside business hours: warrants attention",
            ))

        return adjustments, constraints, signals

    def _analyze_historical(
        self,
        action: Action,
        context: AgentContext,
        historical: HistoricalContext,
    ) -> tuple[list[WeightAdjustment], list[ContextConstraint], list[ContextRiskSignal], float]:
        adjustments: list[WeightAdjustment] = []
        constraints: list[ContextConstraint] = []
        signals: list[ContextRiskSignal] = []
        trust_mod = 0.0

        # Trust direction "falling"
        if historical.trust_direction == "falling":
            for dim_name in _ALL_DIMENSION_NAMES:
                adjustments.append(WeightAdjustment(
                    dimension_name=dim_name,
                    original_weight=1.0,
                    adjusted_weight=1.1,
                    reason="Falling trust direction: increased scrutiny across all dimensions",
                ))

        # Trust direction "volatile"
        if historical.trust_direction == "volatile":
            adjustments.append(WeightAdjustment(
                dimension_name="behavioral_consistency",
                original_weight=1.0,
                adjusted_weight=min(1.0 + 0.2, 3.0),
                reason="Volatile trust trajectory: increased behavioral consistency monitoring",
            ))

        # Recent scope expansion
        if historical.scope_changes_recent:
            adjustments.append(WeightAdjustment(
                dimension_name="scope_compliance",
                original_weight=1.5,
                adjusted_weight=min(1.5 + 0.1, 3.0),
                reason=f"Recent scope changes ({len(historical.scope_changes_recent)}): closer attention to scope compliance",
            ))
            adjustments.append(WeightAdjustment(
                dimension_name="authority_verification",
                original_weight=1.5,
                adjusted_weight=min(1.5 + 0.1, 3.0),
                reason=f"Recent scope changes ({len(historical.scope_changes_recent)}): verify newly granted authority",
            ))

        # Reasoning quality trend declining
        if historical.reasoning_quality_trend == "declining":
            signals.append(ContextRiskSignal(
                signal_type="declining_reasoning_quality",
                description="Agent's reasoning quality trend is declining",
                source_context="historical",
                severity="medium",
                affected_dimensions=["transparency"],
            ))
            adjustments.append(WeightAdjustment(
                dimension_name="transparency",
                original_weight=0.6,
                adjusted_weight=min(0.6 + 0.1, 3.0),
                reason="Declining reasoning quality: increased transparency weight",
            ))

        # High drift status
        if historical.behavioral_drift_status in ("high_drift", "critical_drift"):
            adjustments.append(WeightAdjustment(
                dimension_name="behavioral_consistency",
                original_weight=1.0,
                adjusted_weight=min(1.0 + 0.2, 3.0),
                reason=f"Behavioral drift status: {historical.behavioral_drift_status}",
            ))
            trust_mod -= 0.05

        # Low trust (<0.3)
        if historical.trust_current < 0.3:
            constraints.append(ContextConstraint(
                constraint_type="require_human_review",
                description=f"Agent trust is very low ({historical.trust_current:.2f}): human review required",
                source_context="historical",
                severity="required",
            ))

        return adjustments, constraints, signals, trust_mod

    def _analyze_input(
        self,
        action: Action,
        context: AgentContext,
        input_ctx: InputContext,
    ) -> tuple[list[WeightAdjustment], list[ContextConstraint], list[ContextRiskSignal]]:
        adjustments: list[WeightAdjustment] = []
        constraints: list[ContextConstraint] = []
        signals: list[ContextRiskSignal] = []

        # Adversarial input pattern
        if input_ctx.input_category == "adversarial_pattern":
            signals.append(ContextRiskSignal(
                signal_type="adversarial_input",
                description="Input classified as adversarial pattern",
                source_context="input",
                severity="critical",
                affected_dimensions=["incident_detection", "isolation_integrity", "scope_compliance"],
            ))
            adjustments.append(WeightAdjustment(
                dimension_name="incident_detection",
                original_weight=1.5,
                adjusted_weight=min(1.5 + 0.3, 3.0),
                reason="Adversarial input pattern detected",
            ))
            for dim_name in _SECURITY_DIMENSIONS:
                if dim_name != "incident_detection":
                    adjustments.append(WeightAdjustment(
                        dimension_name=dim_name,
                        original_weight=1.4,
                        adjusted_weight=min(1.4 + 0.2, 3.0),
                        reason="Adversarial input pattern: elevated security dimensions",
                    ))

        # Ambiguous input
        if input_ctx.input_category == "ambiguous":
            adjustments.append(WeightAdjustment(
                dimension_name="transparency",
                original_weight=0.6,
                adjusted_weight=min(0.6 + 0.1, 3.0),
                reason="Ambiguous input: governance should verify agent interpretation",
            ))

        # Edge case
        if input_ctx.input_category == "edge_case":
            adjustments.append(WeightAdjustment(
                dimension_name="ethical_alignment",
                original_weight=2.0,
                adjusted_weight=min(2.0 + 0.1, 3.0),
                reason="Edge case input: ethical issues may be present",
            ))
            adjustments.append(WeightAdjustment(
                dimension_name="precedent_alignment",
                original_weight=0.7,
                adjusted_weight=min(0.7 + 0.1, 3.0),
                reason="Edge case input: precedent matters most at boundaries",
            ))

        # Intent-method mismatch
        if input_ctx.intent_classification and action.action_type:
            intent = input_ctx.intent_classification.lower()
            method = action.action_type.lower()
            # Detect obvious mismatches
            _MISMATCH_PAIRS = [
                ({"schedule", "plan", "query", "read", "view"}, {"delete", "terminate", "destroy", "remove"}),
                ({"read", "view", "inspect"}, {"write", "modify", "update", "execute"}),
            ]
            for intent_set, method_set in _MISMATCH_PAIRS:
                if any(i in intent for i in intent_set) and any(m in method for m in method_set):
                    signals.append(ContextRiskSignal(
                        signal_type="intent_method_mismatch",
                        description=f"Input intent '{input_ctx.intent_classification}' doesn't match action method '{action.action_type}'",
                        source_context="input",
                        severity="high",
                        affected_dimensions=["scope_compliance", "transparency", "ethical_alignment"],
                    ))
                    break

        return adjustments, constraints, signals

    def _analyze_external(
        self,
        action: Action,
        context: AgentContext,
        external: ExternalContext,
    ) -> tuple[list[WeightAdjustment], list[ContextConstraint], list[ContextRiskSignal]]:
        adjustments: list[WeightAdjustment] = []
        constraints: list[ContextConstraint] = []
        signals: list[ContextRiskSignal] = []

        # Active critical alerts
        if external.active_alerts:
            for alert in external.active_alerts:
                signals.append(ContextRiskSignal(
                    signal_type="active_alert",
                    description=f"Active external alert: {alert}",
                    source_context="external",
                    severity="high",
                    affected_dimensions=["incident_detection", "isolation_integrity"],
                ))
            # Elevate security dimension weights
            for dim_name in _SECURITY_DIMENSIONS:
                adjustments.append(WeightAdjustment(
                    dimension_name=dim_name,
                    original_weight=1.5 if dim_name == "incident_detection" else 1.4,
                    adjusted_weight=min((1.5 if dim_name == "incident_detection" else 1.4) + 0.2, 3.0),
                    reason=f"Active external alerts ({len(external.active_alerts)}): elevated security scrutiny",
                ))

        # Environment degraded or outage
        if external.environment_status in ("degraded", "outage"):
            constraints.append(ContextConstraint(
                constraint_type="reduced_authority",
                description=f"Environment status is {external.environment_status}: agent authority should contract",
                source_context="external",
                severity="recommended",
            ))

        # Data freshness concerns — flag stale dependencies
        if external.data_freshness:
            for source_name, last_updated in external.data_freshness.items():
                # If a source is explicitly flagged as stale via the value
                if "stale" in last_updated.lower() or "unknown" in last_updated.lower():
                    signals.append(ContextRiskSignal(
                        signal_type="stale_data_dependency",
                        description=f"Data source '{source_name}' has stale or unknown freshness: {last_updated}",
                        source_context="external",
                        severity="medium",
                        affected_dimensions=["cascading_impact", "stakeholder_impact"],
                    ))

        return adjustments, constraints, signals

    def _analyze_meta(
        self,
        action: Action,
        context: AgentContext,
        meta: MetaContext,
    ) -> tuple[list[WeightAdjustment], list[ContextConstraint], list[ContextRiskSignal], float]:
        adjustments: list[WeightAdjustment] = []
        constraints: list[ContextConstraint] = []
        signals: list[ContextRiskSignal] = []
        trust_mod = 0.0

        # Resubmission after REVISE
        if meta.resubmission:
            trust_mod -= 0.05
            # If revise_count is high, flag as unresponsive to guidance
            if meta.revise_count > 1:
                signals.append(ContextRiskSignal(
                    signal_type="unresponsive_to_guidance",
                    description=f"Agent has been revised {meta.revise_count} times and is resubmitting",
                    source_context="meta",
                    severity="medium",
                    affected_dimensions=["transparency", "behavioral_consistency"],
                ))

        # High denial count
        if meta.denial_count > 3:
            constraints.append(ContextConstraint(
                constraint_type="require_human_review",
                description=f"Repeated denials ({meta.denial_count}): agent persistently attempting actions outside authority",
                source_context="meta",
                severity="required",
            ))

        # High escalation count
        if meta.escalation_count > 2:
            signals.append(ContextRiskSignal(
                signal_type="frequent_escalations",
                description=f"Frequent escalations ({meta.escalation_count}) in this workflow",
                source_context="meta",
                severity="low",
                affected_dimensions=["authority_verification"],
            ))

        # Governance load high
        if meta.governance_load == "high":
            signals.append(ContextRiskSignal(
                signal_type="high_governance_load",
                description="Governance load is high — informational only, governance does not relax under load",
                source_context="meta",
                severity="low",
                affected_dimensions=[],
            ))

        return adjustments, constraints, signals, trust_mod

    def _analyze_feedback(
        self,
        action: Action,
        context: AgentContext,
        feedback: FeedbackContext,
    ) -> tuple[list[WeightAdjustment], list[ContextConstraint], list[ContextRiskSignal], float]:
        adjustments: list[WeightAdjustment] = []
        constraints: list[ContextConstraint] = []
        signals: list[ContextRiskSignal] = []
        trust_mod = 0.0

        # Recent negative feedback
        negative_feedback = [
            f for f in feedback.feedback_received
            if f.feedback_type in ("negative", "complaint", "correction")
        ]
        if negative_feedback:
            adjustments.append(WeightAdjustment(
                dimension_name="stakeholder_impact",
                original_weight=1.2,
                adjusted_weight=min(1.2 + 0.1, 3.0),
                reason=f"Recent negative feedback ({len(negative_feedback)}): elevated stakeholder impact weight",
            ))

        # Human override history
        if feedback.override_history:
            signals.append(ContextRiskSignal(
                signal_type="governance_override_pattern",
                description=f"Human overrides in workflow: {len(feedback.override_history)}",
                source_context="feedback",
                severity="medium",
                affected_dimensions=["human_override", "authority_verification"],
            ))
            # Track whether overrides were to approve or deny
            for override in feedback.override_history:
                if override.override_to.upper() in ("ALLOW", "APPROVE", "PROCEED"):
                    trust_mod += 0.02  # governance was too strict
                elif override.override_to.upper() in ("DENY", "REJECT", "BLOCK"):
                    trust_mod -= 0.03  # governance was too lenient

        # Downstream failure outcomes
        failure_outcomes = [
            o for o in feedback.downstream_outcomes
            if o.outcome_quality in ("worse", "failed")
        ]
        if failure_outcomes:
            adjustments.append(WeightAdjustment(
                dimension_name="cascading_impact",
                original_weight=1.3,
                adjusted_weight=min(1.3 + 0.2, 3.0),
                reason=f"Downstream failures ({len(failure_outcomes)}): previous actions have not gone well",
            ))

        # Positive feedback
        positive_feedback = [
            f for f in feedback.feedback_received
            if f.feedback_type == "positive"
        ]
        if positive_feedback:
            trust_mod += 0.02

        return adjustments, constraints, signals, trust_mod

    # ── Thin context handler ────────────────────────────────────────────

    def _assess_context_completeness(
        self,
        action: Action,
        profile: ContextProfile,
    ) -> list[ContextRiskSignal]:
        signals: list[ContextRiskSignal] = []
        completeness = profile.completeness_score()

        # Determine if action involves a high-governance method
        method = action.action_type.lower() if action.action_type else ""
        is_high_stakes = any(hg in method for hg in _HIGH_GOVERNANCE_METHODS)

        if completeness < self.config.thin_context_threshold and is_high_stakes:
            signals.append(ContextRiskSignal(
                signal_type="thin_context_high_stakes",
                description=f"Context completeness ({completeness:.2f}) below threshold ({self.config.thin_context_threshold}) for high-stakes method '{action.action_type}'",
                source_context="completeness",
                severity="high",
                affected_dimensions=_ALL_DIMENSION_NAMES,
            ))

        if completeness < 0.1 and action.action_type:
            signals.append(ContextRiskSignal(
                signal_type="minimal_context",
                description=f"Minimal context ({completeness:.2f}) for action '{action.action_type}'",
                source_context="completeness",
                severity="medium",
                affected_dimensions=["transparency", "cascading_impact"],
            ))

        return signals

    # ── Flow recommendation ─────────────────────────────────────────────

    def _recommend_flow(
        self,
        action: Action,
        profile: ContextProfile,
        signals: list[ContextRiskSignal],
        completeness: float,
    ) -> str | None:
        """Recommend protocol flow based on context analysis."""
        # Thin context + high stakes → full deliberation
        method = action.action_type.lower() if action.action_type else ""
        is_high_stakes = any(hg in method for hg in _HIGH_GOVERNANCE_METHODS)

        if completeness < self.config.thin_context_threshold and is_high_stakes:
            return "full"

        # Any critical signals → full deliberation
        if any(s.severity == "critical" for s in signals):
            return "full"

        return None
