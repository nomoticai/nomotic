"""Nomotic Protocol types — Reasoning Artifacts, Governance Responses, and supporting structures.

The Nomotic Protocol makes agent reasoning visible, structured, and governable.
This module defines the Python data structures that correspond to the protocol's
JSON schemas, along with serialization/deserialization, validation, and the
Method taxonomy.

All types are plain dataclasses with no external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "METHODS",
    "METHOD_CATEGORIES",
    "Alternative",
    "Assumption",
    "Assessment",
    "AuthorityClaim",
    "Condition",
    "Constraint",
    "Denial",
    "Escalation",
    "Factor",
    "GovernanceResponseData",
    "Guidance",
    "IntendedAction",
    "Justification",
    "Plan",
    "ProtocolVerdict",
    "ReasoningArtifact",
    "ResponseMetadata",
    "Unknown",
]

PROTOCOL_VERSION = "0.1.0"

# ── Method Taxonomy ────────────────────────────────────────────────────


class ProtocolVerdict(Enum):
    """Governance verdict for the Nomotic Protocol."""

    PROCEED = "PROCEED"
    PROCEED_WITH_CONDITIONS = "PROCEED_WITH_CONDITIONS"
    REVISE = "REVISE"
    ESCALATE = "ESCALATE"
    DENY = "DENY"


# Method taxonomy — every action in the protocol is identified by a single word.
METHOD_CATEGORIES: dict[str, list[str]] = {
    "data": ["query", "read", "write", "update", "delete", "archive", "restore", "export", "import"],
    "retrieval": ["fetch", "search", "find", "scan", "filter", "extract", "pull"],
    "decision": ["approve", "deny", "escalate", "recommend", "classify", "prioritize", "evaluate", "validate", "check", "rank", "predict"],
    "communication": ["notify", "request", "respond", "reply", "broadcast", "subscribe", "publish", "send", "call"],
    "orchestration": ["schedule", "assign", "delegate", "invoke", "retry", "cancel", "pause", "resume", "route", "run", "start", "open"],
    "transaction": ["transfer", "refund", "charge", "reserve", "release", "reconcile", "purchase"],
    "security": ["authenticate", "authorize", "revoke", "elevate", "sign", "register"],
    "system": ["configure", "deploy", "monitor", "report", "log", "audit", "sync"],
    "generation": ["generate", "create", "summarize", "transform", "translate", "normalize", "merge", "link", "map", "make"],
    "control": ["set", "take", "show", "turn", "break", "submit"],
}

# Flat set of all valid methods, and reverse lookup to category.
METHODS: set[str] = set()
_METHOD_TO_CATEGORY: dict[str, str] = {}
for _cat, _methods in METHOD_CATEGORIES.items():
    for _m in _methods:
        METHODS.add(_m)
        _METHOD_TO_CATEGORY[_m] = _cat


def method_category(method: str) -> str | None:
    """Return the category for a method, or None if unrecognised."""
    return _METHOD_TO_CATEGORY.get(method)


# Governance weight profiles per category.
CATEGORY_WEIGHTS: dict[str, str] = {
    "data": "standard",
    "retrieval": "lower",
    "decision": "elevated",
    "communication": "moderate",
    "orchestration": "cascading",
    "transaction": "highest",
    "security": "critical",
    "system": "operational",
    "generation": "moderate",
    "control": "context_dependent",
}

# Valid origins for a task.
VALID_ORIGINS = {"user_request", "scheduled", "event_triggered", "agent_initiated", "escalation_received"}

# Valid constraint types.
VALID_CONSTRAINT_TYPES = {"policy", "regulatory", "authority", "ethical", "resource", "temporal", "technical", "organizational"}

# Valid factor types.
VALID_FACTOR_TYPES = {"constraint", "context", "precedent", "evidence", "inference", "uncertainty", "alternative", "risk"}

# Valid influence levels.
VALID_INFLUENCES = {"decisive", "significant", "moderate", "minor", "noted"}

# Valid envelope types.
VALID_ENVELOPE_TYPES = {"standard", "conditional", "delegated", "escalated", "pre_authorized"}

# Valid condition types.
VALID_CONDITION_TYPES = {"audit_level", "scope_reduction", "confirmation_required", "time_limit", "monitoring", "notification", "custom"}

# Valid flow types.
VALID_FLOWS = {"full", "summary", "posthoc"}

# Valid scope types.
VALID_SCOPES = {"single", "class", "session"}


# ── Reasoning Artifact structures ──────────────────────────────────────


@dataclass
class Constraint:
    """A constraint the agent identified as relevant to the task."""

    type: str
    description: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "description": self.description, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Constraint:
        return cls(type=data["type"], description=data["description"], source=data["source"])


@dataclass
class Factor:
    """A consideration the agent evaluated during reasoning."""

    id: str
    type: str
    description: str
    source: str
    assessment: str
    influence: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "source": self.source,
            "assessment": self.assessment,
            "influence": self.influence,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Factor:
        return cls(
            id=data["id"],
            type=data["type"],
            description=data["description"],
            source=data["source"],
            assessment=data["assessment"],
            influence=data["influence"],
            confidence=data["confidence"],
        )


@dataclass
class Alternative:
    """An alternative action the agent considered and rejected."""

    method: str
    reason_rejected: str
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"method": self.method, "reason_rejected": self.reason_rejected}
        if self.context:
            d["context"] = self.context
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Alternative:
        return cls(
            method=data["method"],
            reason_rejected=data["reason_rejected"],
            context=data.get("context", ""),
        )


@dataclass
class IntendedAction:
    """The action the agent will take if governance approves."""

    method: str
    target: str
    context: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"method": self.method, "target": self.target}
        if self.context:
            d["context"] = self.context
        if self.parameters:
            d["parameters"] = self.parameters
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntendedAction:
        return cls(
            method=data["method"],
            target=data["target"],
            context=data.get("context", ""),
            parameters=data.get("parameters", {}),
        )


@dataclass
class Justification:
    """Links a decision to a specific reasoning factor."""

    factor_id: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {"factor_id": self.factor_id, "explanation": self.explanation}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Justification:
        return cls(factor_id=data["factor_id"], explanation=data["explanation"])


@dataclass
class AuthorityClaim:
    """The agent's claim of what authority it believes it is operating under."""

    envelope_type: str
    conditions_met: list[str] = field(default_factory=list)
    limit_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"envelope_type": self.envelope_type}
        if self.conditions_met:
            d["conditions_met"] = self.conditions_met
        if self.limit_reference:
            d["limit_reference"] = self.limit_reference
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthorityClaim:
        return cls(
            envelope_type=data["envelope_type"],
            conditions_met=data.get("conditions_met", []),
            limit_reference=data.get("limit_reference", ""),
        )


@dataclass
class Unknown:
    """Information the agent identified as relevant but unavailable."""

    description: str
    impact: str

    def to_dict(self) -> dict[str, Any]:
        return {"description": self.description, "impact": self.impact}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Unknown:
        return cls(description=data["description"], impact=data["impact"])


@dataclass
class Assumption:
    """An assumption the agent is making to proceed despite incomplete information."""

    description: str
    basis: str
    risk_if_wrong: str

    def to_dict(self) -> dict[str, Any]:
        return {"description": self.description, "basis": self.basis, "risk_if_wrong": self.risk_if_wrong}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assumption:
        return cls(
            description=data["description"],
            basis=data["basis"],
            risk_if_wrong=data["risk_if_wrong"],
        )


@dataclass
class Plan:
    """For multi-step workflows, context about where this reasoning fits."""

    workflow_id: str
    total_steps: int
    current_step: int
    step_description: str
    dependencies: list[str] = field(default_factory=list)
    remaining_steps: list[dict[str, Any]] = field(default_factory=list)
    rollback_capability: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "step_description": self.step_description,
        }
        if self.dependencies:
            d["dependencies"] = self.dependencies
        if self.remaining_steps:
            d["remaining_steps"] = self.remaining_steps
        if self.rollback_capability:
            d["rollback_capability"] = self.rollback_capability
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        return cls(
            workflow_id=data["workflow_id"],
            total_steps=data["total_steps"],
            current_step=data["current_step"],
            step_description=data["step_description"],
            dependencies=data.get("dependencies", []),
            remaining_steps=data.get("remaining_steps", []),
            rollback_capability=data.get("rollback_capability", False),
        )


# ── The Reasoning Artifact ─────────────────────────────────────────────


@dataclass
class ReasoningArtifact:
    """A structured representation of an AI agent's reasoning process.

    This is the core protocol object. It externalizes agent deliberation
    in a format that governance systems can evaluate before action occurs.
    """

    # Identity
    agent_id: str
    # Task
    goal: str
    origin: str
    constraints_identified: list[Constraint]
    # Reasoning
    factors: list[Factor]
    alternatives_considered: list[Alternative]
    # Decision
    intended_action: IntendedAction
    justifications: list[Justification]
    authority_claim: AuthorityClaim
    # Uncertainty
    unknowns: list[Unknown]
    assumptions: list[Assumption]
    overall_confidence: float

    # Optional identity fields
    certificate_id: str = ""
    envelope_id: str = ""
    session_id: str = ""
    origin_id: str = ""

    # Optional reasoning narrative
    narrative: str = ""

    # Optional plan
    plan: Plan | None = None

    # Auto-generated
    protocol_version: str = PROTOCOL_VERSION
    artifact_id: str = field(default_factory=lambda: f"ra-{uuid.uuid4().hex[:12]}")
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            from datetime import datetime, timezone
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary matching the JSON schema."""
        identity: dict[str, Any] = {"agent_id": self.agent_id}
        if self.certificate_id:
            identity["certificate_id"] = self.certificate_id
        if self.envelope_id:
            identity["envelope_id"] = self.envelope_id
        if self.session_id:
            identity["session_id"] = self.session_id

        task: dict[str, Any] = {
            "goal": self.goal,
            "origin": self.origin,
            "constraints_identified": [c.to_dict() for c in self.constraints_identified],
        }
        if self.origin_id:
            task["origin_id"] = self.origin_id

        reasoning: dict[str, Any] = {
            "factors": [f.to_dict() for f in self.factors],
            "alternatives_considered": [a.to_dict() for a in self.alternatives_considered],
        }
        if self.narrative:
            reasoning["narrative"] = self.narrative

        decision: dict[str, Any] = {
            "intended_action": self.intended_action.to_dict(),
            "justifications": [j.to_dict() for j in self.justifications],
            "authority_claim": self.authority_claim.to_dict(),
        }

        uncertainty: dict[str, Any] = {
            "unknowns": [u.to_dict() for u in self.unknowns],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "overall_confidence": self.overall_confidence,
        }

        d: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "artifact_id": self.artifact_id,
            "timestamp": self.timestamp,
            "identity": identity,
            "task": task,
            "reasoning": reasoning,
            "decision": decision,
            "uncertainty": uncertainty,
        }

        if self.plan is not None:
            d["plan"] = self.plan.to_dict()

        return d

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def hash(self) -> str:
        """Compute SHA-256 hash of the canonical JSON representation."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningArtifact:
        """Deserialize from a dictionary."""
        identity = data["identity"]
        task = data["task"]
        reasoning = data["reasoning"]
        decision = data["decision"]
        uncertainty = data["uncertainty"]

        plan = None
        if "plan" in data:
            plan = Plan.from_dict(data["plan"])

        return cls(
            protocol_version=data.get("protocol_version", PROTOCOL_VERSION),
            artifact_id=data.get("artifact_id", f"ra-{uuid.uuid4().hex[:12]}"),
            timestamp=data.get("timestamp", ""),
            agent_id=identity["agent_id"],
            certificate_id=identity.get("certificate_id", ""),
            envelope_id=identity.get("envelope_id", ""),
            session_id=identity.get("session_id", ""),
            goal=task["goal"],
            origin=task["origin"],
            origin_id=task.get("origin_id", ""),
            constraints_identified=[Constraint.from_dict(c) for c in task["constraints_identified"]],
            factors=[Factor.from_dict(f) for f in reasoning["factors"]],
            alternatives_considered=[Alternative.from_dict(a) for a in reasoning["alternatives_considered"]],
            narrative=reasoning.get("narrative", ""),
            intended_action=IntendedAction.from_dict(decision["intended_action"]),
            justifications=[Justification.from_dict(j) for j in decision["justifications"]],
            authority_claim=AuthorityClaim.from_dict(decision["authority_claim"]),
            unknowns=[Unknown.from_dict(u) for u in uncertainty["unknowns"]],
            assumptions=[Assumption.from_dict(a) for a in uncertainty["assumptions"]],
            overall_confidence=uncertainty["overall_confidence"],
            plan=plan,
        )

    @classmethod
    def from_json(cls, raw: str) -> ReasoningArtifact:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(raw))


# ── Governance Response structures ─────────────────────────────────────


@dataclass
class Assessment:
    """Structured evaluation of a reasoning artifact."""

    completeness_score: float
    completeness_detail: str = ""
    missing_constraints: list[str] = field(default_factory=list)

    authority_verified: bool = True
    authority_detail: str = ""
    actual_envelope: str = ""

    alignment_score: float = 1.0
    alignment_detail: str = ""
    misalignments: list[str] = field(default_factory=list)

    uncertainty_calibration_score: float = 1.0
    uncertainty_calibration_detail: str = ""

    alternatives_adequacy_score: float = 1.0
    alternatives_adequacy_detail: str = ""

    dimensional_summary: dict[str, float] = field(default_factory=dict)
    ucs: float | None = None
    trust_state: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "completeness": {"score": self.completeness_score},
            "authority_verified": {"verified": self.authority_verified},
            "alignment": {"score": self.alignment_score},
            "uncertainty_calibration": {"score": self.uncertainty_calibration_score},
            "alternatives_adequacy": {"score": self.alternatives_adequacy_score},
        }
        if self.completeness_detail:
            d["completeness"]["detail"] = self.completeness_detail
        if self.missing_constraints:
            d["completeness"]["missing_constraints"] = self.missing_constraints
        if self.authority_detail:
            d["authority_verified"]["detail"] = self.authority_detail
        if self.actual_envelope:
            d["authority_verified"]["actual_envelope"] = self.actual_envelope
        if self.alignment_detail:
            d["alignment"]["detail"] = self.alignment_detail
        if self.misalignments:
            d["alignment"]["misalignments"] = self.misalignments
        if self.uncertainty_calibration_detail:
            d["uncertainty_calibration"]["detail"] = self.uncertainty_calibration_detail
        if self.alternatives_adequacy_detail:
            d["alternatives_adequacy"]["detail"] = self.alternatives_adequacy_detail
        if self.dimensional_summary:
            d["dimensional_summary"] = self.dimensional_summary
        if self.ucs is not None:
            d["ucs"] = self.ucs
        if self.trust_state is not None:
            d["trust_state"] = self.trust_state
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assessment:
        comp = data["completeness"]
        auth = data["authority_verified"]
        align = data["alignment"]
        unc = data["uncertainty_calibration"]
        alt = data["alternatives_adequacy"]
        return cls(
            completeness_score=comp["score"],
            completeness_detail=comp.get("detail", ""),
            missing_constraints=comp.get("missing_constraints", []),
            authority_verified=auth["verified"],
            authority_detail=auth.get("detail", ""),
            actual_envelope=auth.get("actual_envelope", ""),
            alignment_score=align["score"],
            alignment_detail=align.get("detail", ""),
            misalignments=align.get("misalignments", []),
            uncertainty_calibration_score=unc["score"],
            uncertainty_calibration_detail=unc.get("detail", ""),
            alternatives_adequacy_score=alt["score"],
            alternatives_adequacy_detail=alt.get("detail", ""),
            dimensional_summary=data.get("dimensional_summary", {}),
            ucs=data.get("ucs"),
            trust_state=data.get("trust_state"),
        )


@dataclass
class Condition:
    """A condition that must be honored during execution."""

    type: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.parameters:
            d["parameters"] = self.parameters
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Condition:
        return cls(
            type=data["type"],
            description=data["description"],
            parameters=data.get("parameters", {}),
        )


@dataclass
class Guidance:
    """Direction for improving reasoning (present when verdict is REVISE)."""

    missing_constraints: list[str] = field(default_factory=list)
    authority_correction: str = ""
    reasoning_gaps: list[str] = field(default_factory=list)
    recommended_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.missing_constraints:
            d["missing_constraints"] = self.missing_constraints
        if self.authority_correction:
            d["authority_correction"] = self.authority_correction
        if self.reasoning_gaps:
            d["reasoning_gaps"] = self.reasoning_gaps
        if self.recommended_factors:
            d["recommended_factors"] = self.recommended_factors
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Guidance:
        return cls(
            missing_constraints=data.get("missing_constraints", []),
            authority_correction=data.get("authority_correction", ""),
            reasoning_gaps=data.get("reasoning_gaps", []),
            recommended_factors=data.get("recommended_factors", []),
        )


@dataclass
class Escalation:
    """Routing information for higher authority (present when verdict is ESCALATE)."""

    escalation_target: str = ""
    authority_required: str = ""
    context_package: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.escalation_target:
            d["escalation_target"] = self.escalation_target
        if self.authority_required:
            d["authority_required"] = self.authority_required
        if self.context_package:
            d["context_package"] = self.context_package
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Escalation:
        return cls(
            escalation_target=data.get("escalation_target", ""),
            authority_required=data.get("authority_required", ""),
            context_package=data.get("context_package", ""),
        )


@dataclass
class Denial:
    """Grounds for denial (present when verdict is DENY)."""

    grounds: list[str] = field(default_factory=list)
    veto_dimensions: list[str] = field(default_factory=list)
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.grounds:
            d["grounds"] = self.grounds
        if self.veto_dimensions:
            d["veto_dimensions"] = self.veto_dimensions
        if self.remediation:
            d["remediation"] = self.remediation
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Denial:
        return cls(
            grounds=data.get("grounds", []),
            veto_dimensions=data.get("veto_dimensions", []),
            remediation=data.get("remediation", ""),
        )


@dataclass
class ResponseMetadata:
    """Evaluation context for audit and traceability."""

    evaluator_id: str
    evaluation_time_ms: float
    config_version: str
    timestamp: str
    artifact_hash: str
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "evaluation_time_ms": self.evaluation_time_ms,
            "config_version": self.config_version,
            "timestamp": self.timestamp,
            "artifact_hash": self.artifact_hash,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResponseMetadata:
        return cls(
            evaluator_id=data["evaluator_id"],
            evaluation_time_ms=data["evaluation_time_ms"],
            config_version=data["config_version"],
            timestamp=data["timestamp"],
            artifact_hash=data["artifact_hash"],
            protocol_version=data.get("protocol_version", PROTOCOL_VERSION),
        )


@dataclass
class GovernanceResponseData:
    """The structured response from a Governance Evaluator.

    Named GovernanceResponseData to avoid collision with HTTP response types.
    Contains the verdict, assessment, optional token, and verdict-specific sections.
    """

    verdict: ProtocolVerdict
    assessment: Assessment
    metadata: ResponseMetadata

    response_id: str = field(default_factory=lambda: f"gr-{uuid.uuid4().hex[:12]}")
    protocol_version: str = PROTOCOL_VERSION

    # Conditional sections
    token: str = ""
    conditions: list[Condition] = field(default_factory=list)
    guidance: Guidance | None = None
    escalation: Escalation | None = None
    denial: Denial | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "response_id": self.response_id,
            "verdict": self.verdict.value,
            "assessment": self.assessment.to_dict(),
            "metadata": self.metadata.to_dict(),
        }
        if self.token:
            d["token"] = self.token
        if self.conditions:
            d["conditions"] = [c.to_dict() for c in self.conditions]
        if self.guidance is not None:
            d["guidance"] = self.guidance.to_dict()
        if self.escalation is not None:
            d["escalation"] = self.escalation.to_dict()
        if self.denial is not None:
            d["denial"] = self.denial.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GovernanceResponseData:
        guidance = None
        if "guidance" in data:
            guidance = Guidance.from_dict(data["guidance"])
        escalation = None
        if "escalation" in data:
            escalation = Escalation.from_dict(data["escalation"])
        denial = None
        if "denial" in data:
            denial = Denial.from_dict(data["denial"])
        conditions = [Condition.from_dict(c) for c in data.get("conditions", [])]

        return cls(
            protocol_version=data.get("protocol_version", PROTOCOL_VERSION),
            response_id=data.get("response_id", f"gr-{uuid.uuid4().hex[:12]}"),
            verdict=ProtocolVerdict(data["verdict"]),
            assessment=Assessment.from_dict(data["assessment"]),
            metadata=ResponseMetadata.from_dict(data["metadata"]),
            token=data.get("token", ""),
            conditions=conditions,
            guidance=guidance,
            escalation=escalation,
            denial=denial,
        )


# ── Artifact validation ────────────────────────────────────────────────


def validate_artifact(artifact: ReasoningArtifact) -> list[str]:
    """Validate a ReasoningArtifact for structural correctness.

    Returns a list of validation errors. An empty list means the artifact
    is structurally valid.
    """
    errors: list[str] = []

    # Protocol version
    if not artifact.protocol_version:
        errors.append("protocol_version is required")

    # Identity
    if not artifact.agent_id:
        errors.append("identity.agent_id is required")

    # Task
    if not artifact.goal:
        errors.append("task.goal is required")
    if artifact.origin not in VALID_ORIGINS:
        errors.append(f"task.origin must be one of {VALID_ORIGINS}, got '{artifact.origin}'")

    for i, c in enumerate(artifact.constraints_identified):
        if c.type not in VALID_CONSTRAINT_TYPES:
            errors.append(f"task.constraints_identified[{i}].type must be one of {VALID_CONSTRAINT_TYPES}, got '{c.type}'")

    # Reasoning — factors
    if not artifact.factors:
        errors.append("reasoning.factors must contain at least one factor")
    else:
        has_constraint_factor = False
        factor_ids: set[str] = set()
        for i, f in enumerate(artifact.factors):
            if not f.id:
                errors.append(f"reasoning.factors[{i}].id is required")
            if f.id in factor_ids:
                errors.append(f"reasoning.factors[{i}].id '{f.id}' is duplicate")
            factor_ids.add(f.id)
            if f.type not in VALID_FACTOR_TYPES:
                errors.append(f"reasoning.factors[{i}].type must be one of {VALID_FACTOR_TYPES}, got '{f.type}'")
            if f.type == "constraint":
                has_constraint_factor = True
            if f.influence not in VALID_INFLUENCES:
                errors.append(f"reasoning.factors[{i}].influence must be one of {VALID_INFLUENCES}, got '{f.influence}'")
            if not (0.0 <= f.confidence <= 1.0):
                errors.append(f"reasoning.factors[{i}].confidence must be 0.0-1.0, got {f.confidence}")

        if not has_constraint_factor:
            errors.append("reasoning.factors must contain at least one factor of type 'constraint'")

        # Alternatives
        for i, a in enumerate(artifact.alternatives_considered):
            if a.method not in METHODS:
                errors.append(f"reasoning.alternatives_considered[{i}].method '{a.method}' is not a valid method")

        # Decision
        if artifact.intended_action.method not in METHODS:
            errors.append(f"decision.intended_action.method '{artifact.intended_action.method}' is not a valid method")
        if not artifact.intended_action.target:
            errors.append("decision.intended_action.target is required")

        if not artifact.justifications:
            errors.append("decision.justifications must contain at least one justification")
        for i, j in enumerate(artifact.justifications):
            if j.factor_id not in factor_ids:
                errors.append(f"decision.justifications[{i}].factor_id '{j.factor_id}' does not reference a valid factor")

        if artifact.authority_claim.envelope_type not in VALID_ENVELOPE_TYPES:
            errors.append(
                f"decision.authority_claim.envelope_type must be one of {VALID_ENVELOPE_TYPES}, "
                f"got '{artifact.authority_claim.envelope_type}'"
            )

    # Uncertainty
    if not (0.0 <= artifact.overall_confidence <= 1.0):
        errors.append(f"uncertainty.overall_confidence must be 0.0-1.0, got {artifact.overall_confidence}")

    # Plan (optional)
    if artifact.plan is not None:
        if artifact.plan.total_steps < 1:
            errors.append("plan.total_steps must be >= 1")
        if artifact.plan.current_step < 1:
            errors.append("plan.current_step must be >= 1")
        if artifact.plan.current_step > artifact.plan.total_steps:
            errors.append("plan.current_step must be <= plan.total_steps")

    return errors
