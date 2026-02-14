"""Context Profile — structured, accumulating situational context for governance.

A Context Profile is a living document that captures ten distinct types of context
surrounding an agent's actions. It updates as a workflow progresses, carrying the
data foundation that governance uses to make informed decisions.

The profile is NOT owned by any single reasoning artifact. It spans a workflow
(or a session, or an agent's operational period). Multiple reasoning artifacts
reference the same context profile. The profile updates between steps.

Ten context types:
1. WorkflowContext — where in the workflow are we?
2. SituationalContext — what triggered this and under what conditions?
3. RelationalContext — who else is involved?
4. TemporalContext — when is this happening and what time pressures exist?
5. HistoricalContext — what is this agent's track record?
6. InputContext — what was requested (structured, not raw)?
7. OutputContext — what has been produced so far?
8. ExternalContext — what external signals are active?
9. MetaContext — what is the governance evaluation history?
10. FeedbackContext — what feedback/overrides have occurred?
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CompletedStep",
    "CompoundMethod",
    "ContextProfile",
    "ContextProfileManager",
    "DelegationLink",
    "Dependency",
    "ExternalContext",
    "ExternalSignal",
    "FeedbackContext",
    "FeedbackRecord",
    "HistoricalContext",
    "InputContext",
    "MetaContext",
    "OutcomeRecord",
    "OutputContext",
    "OutputRecord",
    "OverrideRecord",
    "PlannedStep",
    "RecentVerdict",
    "RelationalContext",
    "SituationalContext",
    "TemporalContext",
    "TemporalEvent",
    "WorkflowContext",
]


# ── Helper sub-structures ─────────────────────────────────────────────


@dataclass
class CompletedStep:
    """A workflow step that has been completed."""

    step_id: str
    step_number: int
    method: str
    target: str
    verdict: str  # governance verdict for this step
    ucs: float
    timestamp: str  # ISO timestamp
    output_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "method": self.method,
            "target": self.target,
            "verdict": self.verdict,
            "ucs": self.ucs,
            "timestamp": self.timestamp,
        }
        if self.output_summary is not None:
            d["output_summary"] = self.output_summary
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletedStep:
        return cls(
            step_id=data["step_id"],
            step_number=data["step_number"],
            method=data["method"],
            target=data["target"],
            verdict=data["verdict"],
            ucs=data["ucs"],
            timestamp=data["timestamp"],
            output_summary=data.get("output_summary"),
        )


@dataclass
class PlannedStep:
    """A workflow step that is planned but not yet executed."""

    step_number: int
    method: str
    target: str
    description: str
    estimated_risk: str | None = None  # "low", "medium", "high"
    depends_on: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_number": self.step_number,
            "method": self.method,
            "target": self.target,
            "description": self.description,
        }
        if self.estimated_risk is not None:
            d["estimated_risk"] = self.estimated_risk
        if self.depends_on:
            d["depends_on"] = self.depends_on
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannedStep:
        return cls(
            step_number=data["step_number"],
            method=data["method"],
            target=data["target"],
            description=data["description"],
            estimated_risk=data.get("estimated_risk"),
            depends_on=data.get("depends_on", []),
        )


@dataclass
class Dependency:
    """A dependency between workflow steps."""

    from_step: int
    to_step: int
    dependency_type: str  # "requires", "constrains", "enables", "informs"
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_step": self.from_step,
            "to_step": self.to_step,
            "dependency_type": self.dependency_type,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dependency:
        return cls(
            from_step=data["from_step"],
            to_step=data["to_step"],
            dependency_type=data["dependency_type"],
            description=data["description"],
        )


@dataclass
class DelegationLink:
    """A link in the delegation chain."""

    from_id: str
    to_id: str
    delegated_methods: list[str]
    delegated_at: str  # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "delegated_methods": self.delegated_methods,
            "delegated_at": self.delegated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DelegationLink:
        return cls(
            from_id=data["from_id"],
            to_id=data["to_id"],
            delegated_methods=data["delegated_methods"],
            delegated_at=data["delegated_at"],
        )


@dataclass
class CompoundMethod:
    """A method that multiple agents collectively access."""

    method: str
    agents_involved: list[str]
    combined_scope_description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "agents_involved": self.agents_involved,
            "combined_scope_description": self.combined_scope_description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompoundMethod:
        return cls(
            method=data["method"],
            agents_involved=data["agents_involved"],
            combined_scope_description=data["combined_scope_description"],
        )


@dataclass
class TemporalEvent:
    """A significant event in the environment."""

    event_type: str
    description: str
    occurred_at: str  # ISO timestamp
    relevance: str  # why this matters to governance

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "description": self.description,
            "occurred_at": self.occurred_at,
            "relevance": self.relevance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalEvent:
        return cls(
            event_type=data["event_type"],
            description=data["description"],
            occurred_at=data["occurred_at"],
            relevance=data["relevance"],
        )


@dataclass
class RecentVerdict:
    """A recent governance verdict for an agent."""

    method: str
    target: str
    verdict: str
    ucs: float
    timestamp: str  # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "target": self.target,
            "verdict": self.verdict,
            "ucs": self.ucs,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecentVerdict:
        return cls(
            method=data["method"],
            target=data["target"],
            verdict=data["verdict"],
            ucs=data["ucs"],
            timestamp=data["timestamp"],
        )


@dataclass
class OutputRecord:
    """A record of an output produced during a workflow."""

    step_number: int
    method: str
    target: str
    output_type: str  # "data_modification", "communication", "transaction", "configuration", "report"
    output_summary: str
    reversible: bool
    timestamp: str  # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "method": self.method,
            "target": self.target,
            "output_type": self.output_type,
            "output_summary": self.output_summary,
            "reversible": self.reversible,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputRecord:
        return cls(
            step_number=data["step_number"],
            method=data["method"],
            target=data["target"],
            output_type=data["output_type"],
            output_summary=data["output_summary"],
            reversible=data["reversible"],
            timestamp=data["timestamp"],
        )


@dataclass
class ExternalSignal:
    """An external signal received from outside the system."""

    source: str  # "market_data", "threat_intel", "regulatory_alert", "system_monitor", "custom"
    signal_type: str
    description: str
    severity: str  # "info", "warning", "alert", "critical"
    received_at: str  # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "signal_type": self.signal_type,
            "description": self.description,
            "severity": self.severity,
            "received_at": self.received_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalSignal:
        return cls(
            source=data["source"],
            signal_type=data["signal_type"],
            description=data["description"],
            severity=data["severity"],
            received_at=data["received_at"],
        )


@dataclass
class FeedbackRecord:
    """A record of feedback received during a workflow."""

    source: str  # "user", "manager", "system", "downstream_service"
    feedback_type: str  # "positive", "negative", "correction", "complaint", "override"
    description: str
    timestamp: str  # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "feedback_type": self.feedback_type,
            "description": self.description,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackRecord:
        return cls(
            source=data["source"],
            feedback_type=data["feedback_type"],
            description=data["description"],
            timestamp=data["timestamp"],
        )


@dataclass
class OverrideRecord:
    """A record of a human override of a governance decision."""

    step_number: int
    original_verdict: str
    override_to: str
    overridden_by: str  # owner/human ID
    reason: str
    timestamp: str  # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "original_verdict": self.original_verdict,
            "override_to": self.override_to,
            "overridden_by": self.overridden_by,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OverrideRecord:
        return cls(
            step_number=data["step_number"],
            original_verdict=data["original_verdict"],
            override_to=data["override_to"],
            overridden_by=data["overridden_by"],
            reason=data["reason"],
            timestamp=data["timestamp"],
        )


@dataclass
class OutcomeRecord:
    """A record of what happened after an action was taken."""

    step_number: int
    method: str
    expected_outcome: str
    actual_outcome: str
    outcome_quality: str  # "as_expected", "better", "worse", "failed"
    timestamp: str  # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "method": self.method,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "outcome_quality": self.outcome_quality,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeRecord:
        return cls(
            step_number=data["step_number"],
            method=data["method"],
            expected_outcome=data["expected_outcome"],
            actual_outcome=data["actual_outcome"],
            outcome_quality=data["outcome_quality"],
            timestamp=data["timestamp"],
        )


# ── Ten Context Types ──────────────────────────────────────────────────


@dataclass
class WorkflowContext:
    """Where in the workflow are we?"""

    workflow_id: str
    workflow_type: str  # e.g., "customer_return", "travel_booking", "incident_response"
    total_steps: int
    current_step: int
    steps_completed: list[CompletedStep] = field(default_factory=list)
    steps_remaining: list[PlannedStep] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    rollback_points: list[str] = field(default_factory=list)  # step IDs
    started_at: str = ""  # ISO timestamp
    status: str = "active"  # "active", "paused", "completing", "rolling_back"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "steps_completed": [s.to_dict() for s in self.steps_completed],
            "steps_remaining": [s.to_dict() for s in self.steps_remaining],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "rollback_points": self.rollback_points,
            "started_at": self.started_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowContext:
        return cls(
            workflow_id=data["workflow_id"],
            workflow_type=data["workflow_type"],
            total_steps=data["total_steps"],
            current_step=data["current_step"],
            steps_completed=[CompletedStep.from_dict(s) for s in data.get("steps_completed", [])],
            steps_remaining=[PlannedStep.from_dict(s) for s in data.get("steps_remaining", [])],
            dependencies=[Dependency.from_dict(d) for d in data.get("dependencies", [])],
            rollback_points=data.get("rollback_points", []),
            started_at=data.get("started_at", ""),
            status=data.get("status", "active"),
        )


@dataclass
class SituationalContext:
    """What triggered this and under what conditions?"""

    origin: str  # "user_request", "scheduled", "event_triggered", "agent_initiated", "escalation_received"
    trigger_description: str
    operational_mode: str = "normal"  # "normal", "incident_response", "maintenance", "degraded", "high_volume"
    urgency: str = "routine"  # "routine", "elevated", "urgent", "critical"
    origin_id: str | None = None  # hashed identifier
    initiating_entity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "origin": self.origin,
            "trigger_description": self.trigger_description,
            "operational_mode": self.operational_mode,
            "urgency": self.urgency,
        }
        if self.origin_id is not None:
            d["origin_id"] = self.origin_id
        if self.initiating_entity is not None:
            d["initiating_entity"] = self.initiating_entity
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SituationalContext:
        return cls(
            origin=data["origin"],
            trigger_description=data["trigger_description"],
            operational_mode=data.get("operational_mode", "normal"),
            urgency=data.get("urgency", "routine"),
            origin_id=data.get("origin_id"),
            initiating_entity=data.get("initiating_entity"),
        )


@dataclass
class RelationalContext:
    """Who else is involved?"""

    parent_agent_id: str | None = None
    child_agent_ids: list[str] = field(default_factory=list)
    delegation_chain: list[DelegationLink] = field(default_factory=list)
    shared_workflow_agents: list[str] = field(default_factory=list)
    compound_methods: list[CompoundMethod] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "child_agent_ids": self.child_agent_ids,
            "delegation_chain": [link.to_dict() for link in self.delegation_chain],
            "shared_workflow_agents": self.shared_workflow_agents,
            "compound_methods": [m.to_dict() for m in self.compound_methods],
        }
        if self.parent_agent_id is not None:
            d["parent_agent_id"] = self.parent_agent_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationalContext:
        return cls(
            parent_agent_id=data.get("parent_agent_id"),
            child_agent_ids=data.get("child_agent_ids", []),
            delegation_chain=[DelegationLink.from_dict(l) for l in data.get("delegation_chain", [])],
            shared_workflow_agents=data.get("shared_workflow_agents", []),
            compound_methods=[CompoundMethod.from_dict(m) for m in data.get("compound_methods", [])],
        )


@dataclass
class TemporalContext:
    """When is this happening and what time pressures exist?"""

    current_time: str  # ISO timestamp
    time_of_day_category: str = "business_hours"  # "business_hours", "after_hours", "weekend", "holiday"
    operational_state: str = "normal"  # "normal", "peak", "maintenance_window", "post_incident"
    recent_events: list[TemporalEvent] = field(default_factory=list)
    time_pressure: str | None = None  # "none", "moderate", "high"
    deadline: str | None = None  # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "current_time": self.current_time,
            "time_of_day_category": self.time_of_day_category,
            "operational_state": self.operational_state,
            "recent_events": [e.to_dict() for e in self.recent_events],
        }
        if self.time_pressure is not None:
            d["time_pressure"] = self.time_pressure
        if self.deadline is not None:
            d["deadline"] = self.deadline
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalContext:
        return cls(
            current_time=data["current_time"],
            time_of_day_category=data.get("time_of_day_category", "business_hours"),
            operational_state=data.get("operational_state", "normal"),
            recent_events=[TemporalEvent.from_dict(e) for e in data.get("recent_events", [])],
            time_pressure=data.get("time_pressure"),
            deadline=data.get("deadline"),
        )


@dataclass
class HistoricalContext:
    """What is this agent's track record?"""

    trust_current: float
    trust_direction: str  # "rising", "falling", "stable", "volatile"
    trust_trajectory_summary: str
    recent_verdicts: list[RecentVerdict] = field(default_factory=list)  # last 20 max
    scope_changes_recent: list[str] = field(default_factory=list)
    reasoning_quality_trend: str | None = None  # "improving", "stable", "declining"
    behavioral_drift_status: str = "normal"  # "normal", "low_drift", "moderate_drift", "high_drift", "critical_drift"
    days_since_activation: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "trust_current": self.trust_current,
            "trust_direction": self.trust_direction,
            "trust_trajectory_summary": self.trust_trajectory_summary,
            "recent_verdicts": [v.to_dict() for v in self.recent_verdicts],
            "scope_changes_recent": self.scope_changes_recent,
            "behavioral_drift_status": self.behavioral_drift_status,
        }
        if self.reasoning_quality_trend is not None:
            d["reasoning_quality_trend"] = self.reasoning_quality_trend
        if self.days_since_activation is not None:
            d["days_since_activation"] = self.days_since_activation
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoricalContext:
        return cls(
            trust_current=data["trust_current"],
            trust_direction=data["trust_direction"],
            trust_trajectory_summary=data["trust_trajectory_summary"],
            recent_verdicts=[RecentVerdict.from_dict(v) for v in data.get("recent_verdicts", [])],
            scope_changes_recent=data.get("scope_changes_recent", []),
            reasoning_quality_trend=data.get("reasoning_quality_trend"),
            behavioral_drift_status=data.get("behavioral_drift_status", "normal"),
            days_since_activation=data.get("days_since_activation"),
        )


@dataclass
class InputContext:
    """What was requested (structured, not raw)?

    Privacy by design: never store raw user input. Store structured summaries,
    categories, and hashes.
    """

    input_type: str  # "text", "structured", "api_call", "event", "delegation"
    input_summary: str  # structured summary — NOT raw text
    input_category: str  # "standard_request", "edge_case", "ambiguous", "complex", "adversarial_pattern"
    entities_referenced: list[str] = field(default_factory=list)
    constraints_expressed: list[str] = field(default_factory=list)
    intent_classification: str = ""
    input_hash: str = ""  # hash of original input for traceability

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_type": self.input_type,
            "input_summary": self.input_summary,
            "input_category": self.input_category,
            "entities_referenced": self.entities_referenced,
            "constraints_expressed": self.constraints_expressed,
            "intent_classification": self.intent_classification,
            "input_hash": self.input_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputContext:
        return cls(
            input_type=data["input_type"],
            input_summary=data["input_summary"],
            input_category=data["input_category"],
            entities_referenced=data.get("entities_referenced", []),
            constraints_expressed=data.get("constraints_expressed", []),
            intent_classification=data.get("intent_classification", ""),
            input_hash=data.get("input_hash", ""),
        )


@dataclass
class OutputContext:
    """What has been produced so far?"""

    outputs_produced: list[OutputRecord] = field(default_factory=list)
    cumulative_impact: str = "minimal"  # "minimal", "moderate", "significant", "major"
    external_effects: list[str] = field(default_factory=list)
    reversible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs_produced": [o.to_dict() for o in self.outputs_produced],
            "cumulative_impact": self.cumulative_impact,
            "external_effects": self.external_effects,
            "reversible": self.reversible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputContext:
        return cls(
            outputs_produced=[OutputRecord.from_dict(o) for o in data.get("outputs_produced", [])],
            cumulative_impact=data.get("cumulative_impact", "minimal"),
            external_effects=data.get("external_effects", []),
            reversible=data.get("reversible", True),
        )


@dataclass
class ExternalContext:
    """What external signals are active?"""

    external_signals: list[ExternalSignal] = field(default_factory=list)
    data_freshness: dict[str, str] = field(default_factory=dict)  # source_name -> last_updated
    environment_status: str = "normal"  # "normal", "degraded", "outage", "recovery"
    active_alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "external_signals": [s.to_dict() for s in self.external_signals],
            "data_freshness": self.data_freshness,
            "environment_status": self.environment_status,
            "active_alerts": self.active_alerts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalContext:
        return cls(
            external_signals=[ExternalSignal.from_dict(s) for s in data.get("external_signals", [])],
            data_freshness=data.get("data_freshness", {}),
            environment_status=data.get("environment_status", "normal"),
            active_alerts=data.get("active_alerts", []),
        )


@dataclass
class MetaContext:
    """What is the governance evaluation history?"""

    evaluation_count: int = 0
    revise_count: int = 0
    escalation_count: int = 0
    denial_count: int = 0
    governance_load: str = "normal"  # "normal", "elevated", "high"
    previous_artifact_ids: list[str] = field(default_factory=list)
    resubmission: bool = False
    original_artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "evaluation_count": self.evaluation_count,
            "revise_count": self.revise_count,
            "escalation_count": self.escalation_count,
            "denial_count": self.denial_count,
            "governance_load": self.governance_load,
            "previous_artifact_ids": self.previous_artifact_ids,
            "resubmission": self.resubmission,
        }
        if self.original_artifact_id is not None:
            d["original_artifact_id"] = self.original_artifact_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetaContext:
        return cls(
            evaluation_count=data.get("evaluation_count", 0),
            revise_count=data.get("revise_count", 0),
            escalation_count=data.get("escalation_count", 0),
            denial_count=data.get("denial_count", 0),
            governance_load=data.get("governance_load", "normal"),
            previous_artifact_ids=data.get("previous_artifact_ids", []),
            resubmission=data.get("resubmission", False),
            original_artifact_id=data.get("original_artifact_id"),
        )


@dataclass
class FeedbackContext:
    """What feedback/overrides have occurred?"""

    feedback_received: list[FeedbackRecord] = field(default_factory=list)
    override_history: list[OverrideRecord] = field(default_factory=list)
    downstream_outcomes: list[OutcomeRecord] = field(default_factory=list)
    satisfaction_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_received": [f.to_dict() for f in self.feedback_received],
            "override_history": [o.to_dict() for o in self.override_history],
            "downstream_outcomes": [o.to_dict() for o in self.downstream_outcomes],
            "satisfaction_signals": self.satisfaction_signals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackContext:
        return cls(
            feedback_received=[FeedbackRecord.from_dict(f) for f in data.get("feedback_received", [])],
            override_history=[OverrideRecord.from_dict(o) for o in data.get("override_history", [])],
            downstream_outcomes=[OutcomeRecord.from_dict(o) for o in data.get("downstream_outcomes", [])],
            satisfaction_signals=data.get("satisfaction_signals", []),
        )


# ── Context Profile Container ─────────────────────────────────────────


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ContextProfile:
    """The container for all ten context types.

    A living object that updates as a workflow progresses. All context types
    are optional — sparse context early in a workflow is expected.
    """

    profile_id: str
    agent_id: str
    profile_type: str  # "workflow", "session", "operational_period"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    # Ten context types — all optional
    workflow: WorkflowContext | None = None
    situational: SituationalContext | None = None
    relational: RelationalContext | None = None
    temporal: TemporalContext | None = None
    historical: HistoricalContext | None = None
    input_context: InputContext | None = None
    output: OutputContext | None = None
    external: ExternalContext | None = None
    meta: MetaContext | None = None
    feedback: FeedbackContext | None = None

    # Lifecycle
    active: bool = True

    _CONTEXT_FIELDS = (
        "workflow", "situational", "relational", "temporal", "historical",
        "input_context", "output", "external", "meta", "feedback",
    )

    def completeness_score(self) -> float:
        """What fraction of context types are populated? Returns 0.0-1.0."""
        populated = sum(1 for f in self._CONTEXT_FIELDS if getattr(self, f) is not None)
        return populated / len(self._CONTEXT_FIELDS)

    def risk_signals(self) -> list[str]:
        """Scan all context types for elevated risk indicators.

        Returns a list of human-readable risk signal descriptions.
        """
        signals: list[str] = []

        # Situational: agent-initiated origin
        if self.situational is not None and self.situational.origin == "agent_initiated":
            signals.append("Agent-initiated action (no external trigger)")

        # Situational: critical urgency
        if self.situational is not None and self.situational.urgency == "critical":
            signals.append("Critical urgency level")

        # Historical: volatile trust
        if self.historical is not None and self.historical.trust_direction == "volatile":
            signals.append("Volatile trust trajectory")

        # Historical: high/critical drift
        if self.historical is not None and self.historical.behavioral_drift_status in (
            "high_drift", "critical_drift",
        ):
            signals.append(f"Behavioral drift: {self.historical.behavioral_drift_status}")

        # External: active alerts
        if self.external is not None and self.external.active_alerts:
            signals.append(f"Active external alerts: {len(self.external.active_alerts)}")

        # External: degraded environment
        if self.external is not None and self.external.environment_status in ("degraded", "outage"):
            signals.append(f"Environment status: {self.external.environment_status}")

        # Meta: resubmission after denial
        if self.meta is not None and self.meta.resubmission and self.meta.denial_count > 0:
            signals.append("Resubmission after previous denial")

        # Meta: high escalation count
        if self.meta is not None and self.meta.escalation_count >= 3:
            signals.append(f"Multiple escalations in workflow: {self.meta.escalation_count}")

        # Feedback: overrides present
        if self.feedback is not None and self.feedback.override_history:
            signals.append(f"Human overrides in workflow: {len(self.feedback.override_history)}")

        # Low completeness on what we can infer
        if self.completeness_score() < 0.3:
            signals.append(f"Low context completeness: {self.completeness_score():.1%}")

        return signals

    def update_workflow_step(self, step: CompletedStep) -> None:
        """Add a completed step to workflow context, update current_step."""
        if self.workflow is None:
            return
        self.workflow.steps_completed.append(step)
        self.workflow.current_step = step.step_number + 1
        # Remove from remaining if present
        self.workflow.steps_remaining = [
            s for s in self.workflow.steps_remaining
            if s.step_number != step.step_number
        ]
        # Update status
        if self.workflow.current_step >= self.workflow.total_steps:
            self.workflow.status = "completing"
        self.updated_at = _now_iso()

    def add_feedback(self, feedback: FeedbackRecord) -> None:
        """Add feedback to feedback context, creating it if None."""
        if self.feedback is None:
            self.feedback = FeedbackContext()
        self.feedback.feedback_received.append(feedback)
        self.updated_at = _now_iso()

    def add_external_signal(self, signal: ExternalSignal) -> None:
        """Add external signal, creating context if None."""
        if self.external is None:
            self.external = ExternalContext()
        self.external.external_signals.append(signal)
        self.updated_at = _now_iso()

    def add_output(self, output: OutputRecord) -> None:
        """Add output record, creating context if None."""
        if self.output is None:
            self.output = OutputContext()
        self.output.outputs_produced.append(output)
        # Update reversibility
        if not output.reversible:
            self.output.reversible = False
        self.updated_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        """Full serialization. Only includes populated context types."""
        d: dict[str, Any] = {
            "profile_id": self.profile_id,
            "agent_id": self.agent_id,
            "profile_type": self.profile_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active": self.active,
        }
        if self.workflow is not None:
            d["workflow"] = self.workflow.to_dict()
        if self.situational is not None:
            d["situational"] = self.situational.to_dict()
        if self.relational is not None:
            d["relational"] = self.relational.to_dict()
        if self.temporal is not None:
            d["temporal"] = self.temporal.to_dict()
        if self.historical is not None:
            d["historical"] = self.historical.to_dict()
        if self.input_context is not None:
            d["input_context"] = self.input_context.to_dict()
        if self.output is not None:
            d["output"] = self.output.to_dict()
        if self.external is not None:
            d["external"] = self.external.to_dict()
        if self.meta is not None:
            d["meta"] = self.meta.to_dict()
        if self.feedback is not None:
            d["feedback"] = self.feedback.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextProfile:
        """Deserialize from dict. Missing sections become None."""
        profile = cls(
            profile_id=data["profile_id"],
            agent_id=data["agent_id"],
            profile_type=data["profile_type"],
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
            active=data.get("active", True),
        )
        if "workflow" in data:
            profile.workflow = WorkflowContext.from_dict(data["workflow"])
        if "situational" in data:
            profile.situational = SituationalContext.from_dict(data["situational"])
        if "relational" in data:
            profile.relational = RelationalContext.from_dict(data["relational"])
        if "temporal" in data:
            profile.temporal = TemporalContext.from_dict(data["temporal"])
        if "historical" in data:
            profile.historical = HistoricalContext.from_dict(data["historical"])
        if "input_context" in data:
            profile.input_context = InputContext.from_dict(data["input_context"])
        if "output" in data:
            profile.output = OutputContext.from_dict(data["output"])
        if "external" in data:
            profile.external = ExternalContext.from_dict(data["external"])
        if "meta" in data:
            profile.meta = MetaContext.from_dict(data["meta"])
        if "feedback" in data:
            profile.feedback = FeedbackContext.from_dict(data["feedback"])
        return profile

    def summary(self) -> dict[str, Any]:
        """Compact representation for logging/debugging."""
        s: dict[str, Any] = {
            "profile_id": self.profile_id,
            "agent_id": self.agent_id,
            "completeness": round(self.completeness_score(), 2),
            "risk_signal_count": len(self.risk_signals()),
            "active": self.active,
        }
        if self.workflow is not None:
            s["workflow_status"] = self.workflow.status
            s["workflow_step"] = f"{self.workflow.current_step}/{self.workflow.total_steps}"
        if self.historical is not None:
            s["trust_direction"] = self.historical.trust_direction
        return s


# ── Context Profile Manager ───────────────────────────────────────────


class ContextProfileManager:
    """Manages context profiles across workflows and sessions.

    Thread-safe. In-memory storage with configurable max profiles
    (default 1000, oldest closed profiles evicted first).
    """

    def __init__(self, max_profiles: int = 1000) -> None:
        self._profiles: dict[str, ContextProfile] = {}
        self._max_profiles = max_profiles
        self._lock = threading.Lock()

    def create_profile(
        self,
        agent_id: str,
        profile_type: str,
        *,
        workflow: WorkflowContext | None = None,
        situational: SituationalContext | None = None,
        relational: RelationalContext | None = None,
        temporal: TemporalContext | None = None,
        historical: HistoricalContext | None = None,
        input_context: InputContext | None = None,
        output: OutputContext | None = None,
        external: ExternalContext | None = None,
        meta: MetaContext | None = None,
        feedback: FeedbackContext | None = None,
    ) -> ContextProfile:
        """Create and register a new context profile."""
        profile_id = f"cp-{uuid.uuid4().hex[:12]}"
        profile = ContextProfile(
            profile_id=profile_id,
            agent_id=agent_id,
            profile_type=profile_type,
            workflow=workflow,
            situational=situational,
            relational=relational,
            temporal=temporal,
            historical=historical,
            input_context=input_context,
            output=output,
            external=external,
            meta=meta,
            feedback=feedback,
        )
        with self._lock:
            self._evict_if_needed()
            self._profiles[profile_id] = profile
        return profile

    def get_profile(self, profile_id: str) -> ContextProfile | None:
        """Retrieve a profile by ID."""
        with self._lock:
            return self._profiles.get(profile_id)

    def get_active_profile(self, agent_id: str) -> ContextProfile | None:
        """Retrieve the most recent active profile for an agent."""
        with self._lock:
            candidates = [
                p for p in self._profiles.values()
                if p.agent_id == agent_id and p.active
            ]
        if not candidates:
            return None
        # Most recent by created_at
        return max(candidates, key=lambda p: p.created_at)

    def update_profile(self, profile_id: str, **kwargs: Any) -> ContextProfile | None:
        """Update specific context types on a profile.

        Accepts any of the ten context type dataclasses as keyword arguments.
        """
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                return None
            for key, value in kwargs.items():
                if key in ContextProfile._CONTEXT_FIELDS and value is not None:
                    setattr(profile, key, value)
            profile.updated_at = _now_iso()
            return profile

    def close_profile(self, profile_id: str) -> bool:
        """Mark a profile as complete (workflow finished, session ended)."""
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                return False
            profile.active = False
            profile.updated_at = _now_iso()
            return True

    def list_profiles(
        self,
        agent_id: str | None = None,
        profile_type: str | None = None,
        active_only: bool = True,
    ) -> list[ContextProfile]:
        """List profiles with filters."""
        with self._lock:
            profiles = list(self._profiles.values())
        result = []
        for p in profiles:
            if agent_id is not None and p.agent_id != agent_id:
                continue
            if profile_type is not None and p.profile_type != profile_type:
                continue
            if active_only and not p.active:
                continue
            result.append(p)
        return result

    def enrich_from_runtime(self, profile_id: str, runtime: Any) -> bool:
        """Auto-populate historical context from the governance runtime.

        Pulls trust state, drift status, and recent verdicts.
        """
        with self._lock:
            profile = self._profiles.get(profile_id)
        if profile is None:
            return False

        agent_id = profile.agent_id

        # Trust state
        trust_profile = runtime.get_trust_profile(agent_id)
        trajectory = runtime.get_trust_trajectory(agent_id)

        # Drift state
        drift = runtime.get_drift(agent_id)
        drift_status = "normal"
        if drift is not None:
            severity = drift.severity if hasattr(drift, "severity") else "normal"
            drift_map = {
                "low": "low_drift",
                "moderate": "moderate_drift",
                "high": "high_drift",
                "critical": "critical_drift",
            }
            drift_status = drift_map.get(severity, "normal")

        # Recent verdicts from audit trail
        recent_verdicts: list[RecentVerdict] = []
        if runtime.audit_trail is not None:
            records = runtime.audit_trail.query(agent_id=agent_id, limit=20)
            for r in records:
                recent_verdicts.append(RecentVerdict(
                    method=r.action_type,
                    target=r.action_target,
                    verdict=r.verdict,
                    ucs=r.ucs,
                    timestamp=str(r.timestamp),
                ))

        historical = HistoricalContext(
            trust_current=trust_profile.overall_trust,
            trust_direction=trajectory.trend if hasattr(trajectory, "trend") else "stable",
            trust_trajectory_summary=trajectory.summary().get("narrative", "") if hasattr(trajectory, "summary") else "",
            recent_verdicts=recent_verdicts,
            behavioral_drift_status=drift_status,
        )

        with self._lock:
            profile.historical = historical
            profile.updated_at = _now_iso()
        return True

    def enrich_from_audit(self, profile_id: str, audit_trail: Any) -> bool:
        """Auto-populate meta context from the audit trail."""
        with self._lock:
            profile = self._profiles.get(profile_id)
        if profile is None:
            return False

        agent_id = profile.agent_id
        records = audit_trail.query(agent_id=agent_id, limit=100)

        evaluation_count = len(records)
        revise_count = 0
        escalation_count = 0
        denial_count = 0

        for r in records:
            verdict = r.verdict if isinstance(r.verdict, str) else r.verdict.name
            if verdict == "ESCALATE":
                escalation_count += 1
            elif verdict == "DENY":
                denial_count += 1
            # Check context code for REVISE-like patterns
            if hasattr(r, "context_code") and "REVISE" in str(r.context_code):
                revise_count += 1

        meta = MetaContext(
            evaluation_count=evaluation_count,
            revise_count=revise_count,
            escalation_count=escalation_count,
            denial_count=denial_count,
        )

        with self._lock:
            profile.meta = meta
            profile.updated_at = _now_iso()
        return True

    def _evict_if_needed(self) -> None:
        """Evict oldest closed profiles if at capacity. Must hold lock."""
        if len(self._profiles) < self._max_profiles:
            return
        # Find closed profiles sorted by created_at (oldest first)
        closed = [
            (pid, p) for pid, p in self._profiles.items()
            if not p.active
        ]
        closed.sort(key=lambda x: x[1].created_at)
        # Evict oldest closed profiles
        to_remove = len(self._profiles) - self._max_profiles + 1
        for i in range(min(to_remove, len(closed))):
            del self._profiles[closed[i][0]]
