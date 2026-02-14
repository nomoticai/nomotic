"""Core types for the Nomotic governance framework.

Every component speaks this vocabulary. Actions describe what agents want to do.
Contexts describe who is doing it and under what circumstances. Verdicts carry
governance decisions. Trust profiles track calibrated confidence in agents.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

__all__ = [
    "Action",
    "ActionRecord",
    "ActionState",
    "AgentContext",
    "DimensionScore",
    "GovernanceVerdict",
    "InterruptRequest",
    "Severity",
    "TrustProfile",
    "UserContext",
    "Verdict",
]


class Verdict(Enum):
    """The outcome of a governance evaluation."""

    ALLOW = auto()
    DENY = auto()
    MODIFY = auto()
    ESCALATE = auto()
    SUSPEND = auto()


class Severity(Enum):
    """How serious a governance violation is."""

    INFO = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class ActionState(Enum):
    """Lifecycle state of an action under governance."""

    PENDING = auto()
    EVALUATING = auto()
    APPROVED = auto()
    EXECUTING = auto()
    INTERRUPTED = auto()
    COMPLETED = auto()
    DENIED = auto()
    ROLLED_BACK = auto()


@dataclass(frozen=True)
class Action:
    """An action an agent wants to perform.

    This is the unit of governance. Every action passes through the governance
    runtime before, during, and after execution.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_id: str = ""
    action_type: str = ""
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    parent_action_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserContext:
    """Information about the user who triggered an agent action.

    Optional — not all agent actions are user-triggered.
    Batch processing, scheduled tasks, and autonomous workflows
    may have no user context.
    """

    user_id: str = ""           # Identifier for the user (email, ID, session)
    session_id: str = ""        # User's session (for tracking repeated interactions)
    request_hash: str = ""      # Hash of user's input (NOT the input itself — privacy)
    classification: str = ""    # "normal", "out_of_scope", "suspicious", "manipulation"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    """Everything governance knows about the agent requesting an action.

    This includes identity, current trust level, session history, and
    any active constraints.
    """

    agent_id: str
    trust_profile: TrustProfile
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    action_history: list[ActionRecord] = field(default_factory=list)
    active_constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    user_context: UserContext | None = None
    context_profile_id: str | None = None  # reference to active ContextProfile


@dataclass
class TrustProfile:
    """Calibrated trust in an agent, updated continuously based on behavior.

    Trust is not binary. It is a set of scores across dimensions, with an
    overall level that determines what the agent is permitted to do.
    """

    agent_id: str
    overall_trust: float = 0.5  # 0.0 = no trust, 1.0 = full trust
    dimension_trust: dict[str, float] = field(default_factory=dict)
    violation_count: int = 0
    successful_actions: int = 0
    last_violation_time: float | None = None
    last_updated: float = field(default_factory=time.time)

    @property
    def violation_rate(self) -> float:
        total = self.violation_count + self.successful_actions
        if total == 0:
            return 0.0
        return self.violation_count / total


@dataclass
class DimensionScore:
    """A single governance dimension's assessment of an action."""

    dimension_name: str
    score: float  # 0.0 = maximum concern, 1.0 = no concern
    weight: float = 1.0
    confidence: float = 1.0
    veto: bool = False
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceVerdict:
    """The complete governance decision for an action.

    Includes the verdict, the unified confidence score, individual dimension
    scores, which tier made the decision, and whether any dimension vetoed.
    """

    action_id: str
    verdict: Verdict
    ucs: float  # Unified Confidence Score: 0.0 = deny, 1.0 = full confidence
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    tier: int = 1  # Which tier (1, 2, or 3) made the final decision
    vetoed_by: list[str] = field(default_factory=list)
    modifications: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)
    evaluation_time_ms: float = 0.0
    context_modification: Any = None  # ContextModification from Phase 7B, if present


@dataclass
class ActionRecord:
    """A completed action with its governance verdict and outcome."""

    action: Action
    verdict: GovernanceVerdict
    state: ActionState = ActionState.COMPLETED
    outcome: dict[str, Any] = field(default_factory=dict)
    interrupted: bool = False
    interrupt_reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class InterruptRequest:
    """A request to interrupt an action in progress."""

    action_id: str
    reason: str
    source: str  # Which dimension or system triggered the interrupt
    severity: Severity = Severity.HIGH
    scope: str = "action"  # "action", "agent", or "workflow"
    timestamp: float = field(default_factory=time.time)
