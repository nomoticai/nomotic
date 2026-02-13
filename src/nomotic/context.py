"""Context codes — structured vocabulary for governance events.

Context codes categorize WHY governance events happen.  They make audit
trails queryable: "show me all SECURITY.INJECTION_ATTEMPT events this week"
rather than grep-ing through free-text reasoning.

Codes are hierarchical: CATEGORY.SPECIFIC_EVENT

Categories:
    GOVERNANCE  — routine governance decisions (allow, deny, escalate)
    SCOPE       — scope and authority boundary events
    TRUST       — trust changes (earned, lost, decayed)
    DRIFT       — behavioral drift events
    SECURITY    — security-relevant events (injection, anomaly)
    USER        — user-triggered events (requests, manipulation attempts)
    CONFIG      — configuration changes (rules, weights, boundaries)
    SYSTEM      — system-level events (startup, shutdown, errors)
    OVERRIDE    — human override events
    ETHICAL     — ethical constraint events
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ContextCode",
    "CODES",
]


@dataclass(frozen=True)
class ContextCode:
    """A single context code with metadata.

    Context codes are immutable identifiers. They carry a code string,
    a human-readable description, and a severity level that indicates
    how significant the event is for audit purposes.
    """

    code: str
    description: str
    severity: str = "info"  # "info", "warning", "alert", "critical"
    category: str = ""      # Auto-derived from code prefix

    def __post_init__(self) -> None:
        if not self.category:
            object.__setattr__(self, "category", self.code.split(".")[0])

    def __str__(self) -> str:
        return self.code

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.code == other
        if isinstance(other, ContextCode):
            return self.code == other.code
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.code)


class CODES:
    """The complete context code catalog.

    Usage::

        from nomotic.context import CODES

        record.context_code = CODES.GOVERNANCE_ALLOW
        record.context_code = CODES.SECURITY_INJECTION_ATTEMPT
    """

    # ── GOVERNANCE: routine governance decisions ──────────────────

    GOVERNANCE_ALLOW = ContextCode(
        "GOVERNANCE.ALLOW",
        "Action evaluated and permitted",
    )
    GOVERNANCE_DENY = ContextCode(
        "GOVERNANCE.DENY",
        "Action evaluated and denied",
        severity="warning",
    )
    GOVERNANCE_MODIFY = ContextCode(
        "GOVERNANCE.MODIFY",
        "Action permitted with modifications",
        severity="warning",
    )
    GOVERNANCE_ESCALATE = ContextCode(
        "GOVERNANCE.ESCALATE",
        "Action escalated to human review",
        severity="warning",
    )
    GOVERNANCE_VETO = ContextCode(
        "GOVERNANCE.VETO",
        "Action vetoed by one or more dimensions",
        severity="alert",
    )

    # ── SCOPE: authority boundary events ──────────────────────────

    SCOPE_WITHIN = ContextCode(
        "SCOPE.WITHIN",
        "Action within agent's authorized scope",
    )
    SCOPE_VIOLATION = ContextCode(
        "SCOPE.VIOLATION",
        "Action outside agent's authorized scope",
        severity="alert",
    )
    SCOPE_NEAR_BOUNDARY = ContextCode(
        "SCOPE.NEAR_BOUNDARY",
        "Action near the edge of authorized scope",
        severity="warning",
    )

    # ── TRUST: trust change events ────────────────────────────────

    TRUST_EARNED = ContextCode(
        "TRUST.EARNED",
        "Trust increased from successful action",
    )
    TRUST_LOST_VIOLATION = ContextCode(
        "TRUST.LOST_VIOLATION",
        "Trust decreased from governance violation",
        severity="warning",
    )
    TRUST_LOST_INTERRUPT = ContextCode(
        "TRUST.LOST_INTERRUPT",
        "Trust decreased from interrupted action",
        severity="warning",
    )
    TRUST_LOST_DRIFT = ContextCode(
        "TRUST.LOST_DRIFT",
        "Trust decreased from behavioral drift",
        severity="warning",
    )
    TRUST_RECOVERED = ContextCode(
        "TRUST.RECOVERED",
        "Trust recovered after drift normalized",
    )
    TRUST_DECAYED = ContextCode(
        "TRUST.DECAYED",
        "Trust decayed toward baseline during inactivity",
    )
    TRUST_THRESHOLD_LOW = ContextCode(
        "TRUST.THRESHOLD_LOW",
        "Trust dropped below low-trust threshold (0.3)",
        severity="alert",
    )

    # ── DRIFT: behavioral drift events ────────────────────────────

    DRIFT_DETECTED_LOW = ContextCode(
        "DRIFT.DETECTED_LOW",
        "Low behavioral drift detected",
    )
    DRIFT_DETECTED_MODERATE = ContextCode(
        "DRIFT.DETECTED_MODERATE",
        "Moderate behavioral drift detected",
        severity="warning",
    )
    DRIFT_DETECTED_HIGH = ContextCode(
        "DRIFT.DETECTED_HIGH",
        "High behavioral drift detected",
        severity="alert",
    )
    DRIFT_DETECTED_CRITICAL = ContextCode(
        "DRIFT.DETECTED_CRITICAL",
        "Critical behavioral drift detected",
        severity="critical",
    )
    DRIFT_NORMALIZED = ContextCode(
        "DRIFT.NORMALIZED",
        "Behavioral drift returned to normal levels",
    )

    # ── SECURITY: security-relevant events ────────────────────────

    SECURITY_INJECTION_ATTEMPT = ContextCode(
        "SECURITY.INJECTION_ATTEMPT",
        "Possible prompt injection or manipulation detected",
        severity="critical",
    )
    SECURITY_ANOMALOUS_PATTERN = ContextCode(
        "SECURITY.ANOMALOUS_PATTERN",
        "Anomalous action pattern detected by incident detection",
        severity="alert",
    )
    SECURITY_BOUNDARY_BREACH = ContextCode(
        "SECURITY.BOUNDARY_BREACH",
        "Agent attempted to access resources outside isolation boundary",
        severity="critical",
    )
    SECURITY_RATE_EXCEEDED = ContextCode(
        "SECURITY.RATE_EXCEEDED",
        "Agent exceeded rate or resource limits",
        severity="alert",
    )

    # ── USER: user-triggered events ───────────────────────────────

    USER_REQUEST_NORMAL = ContextCode(
        "USER.REQUEST_NORMAL",
        "User request within normal parameters",
    )
    USER_REQUEST_OUT_OF_SCOPE = ContextCode(
        "USER.REQUEST_OUT_OF_SCOPE",
        "User requested action outside agent's scope",
        severity="warning",
    )
    USER_MANIPULATION_SUSPECTED = ContextCode(
        "USER.MANIPULATION_SUSPECTED",
        "User input shows signs of manipulation or social engineering",
        severity="alert",
    )
    USER_REPEATED_BOUNDARY_TESTING = ContextCode(
        "USER.REPEATED_BOUNDARY_TESTING",
        "User has repeatedly tested agent's boundaries",
        severity="alert",
    )
    USER_ESCALATION_REQUESTED = ContextCode(
        "USER.ESCALATION_REQUESTED",
        "User explicitly requested escalation to human",
    )

    # ── CONFIG: configuration change events ───────────────────────

    CONFIG_SCOPE_CHANGED = ContextCode(
        "CONFIG.SCOPE_CHANGED",
        "Agent scope configuration modified",
        severity="warning",
    )
    CONFIG_RULE_ADDED = ContextCode(
        "CONFIG.RULE_ADDED",
        "Governance rule added",
        severity="warning",
    )
    CONFIG_RULE_REMOVED = ContextCode(
        "CONFIG.RULE_REMOVED",
        "Governance rule removed",
        severity="alert",
    )
    CONFIG_WEIGHT_CHANGED = ContextCode(
        "CONFIG.WEIGHT_CHANGED",
        "Dimension weight modified",
        severity="warning",
    )
    CONFIG_THRESHOLD_CHANGED = ContextCode(
        "CONFIG.THRESHOLD_CHANGED",
        "Governance threshold modified",
        severity="warning",
    )
    CONFIG_BOUNDARY_CHANGED = ContextCode(
        "CONFIG.BOUNDARY_CHANGED",
        "Isolation boundary modified",
        severity="warning",
    )

    # ── SYSTEM: system-level events ───────────────────────────────

    SYSTEM_STARTUP = ContextCode(
        "SYSTEM.STARTUP",
        "Governance runtime initialized",
    )
    SYSTEM_SHUTDOWN = ContextCode(
        "SYSTEM.SHUTDOWN",
        "Governance runtime shutting down",
    )
    SYSTEM_ERROR = ContextCode(
        "SYSTEM.ERROR",
        "Internal governance error",
        severity="critical",
    )

    # ── OVERRIDE: human override events ───────────────────────────

    OVERRIDE_REQUIRED = ContextCode(
        "OVERRIDE.REQUIRED",
        "Action requires human approval",
        severity="warning",
    )
    OVERRIDE_APPROVED = ContextCode(
        "OVERRIDE.APPROVED",
        "Human approved the action",
    )
    OVERRIDE_REJECTED = ContextCode(
        "OVERRIDE.REJECTED",
        "Human rejected the action",
        severity="warning",
    )
    OVERRIDE_INTERRUPT = ContextCode(
        "OVERRIDE.INTERRUPT",
        "Human interrupted an in-progress action",
        severity="alert",
    )

    # ── ETHICAL: ethical constraint events ─────────────────────────

    ETHICAL_CONSTRAINT_PASSED = ContextCode(
        "ETHICAL.CONSTRAINT_PASSED",
        "Action passed all ethical checks",
    )
    ETHICAL_CONSTRAINT_FAILED = ContextCode(
        "ETHICAL.CONSTRAINT_FAILED",
        "Action failed an ethical constraint",
        severity="alert",
    )
    ETHICAL_REVIEW_RECOMMENDED = ContextCode(
        "ETHICAL.REVIEW_RECOMMENDED",
        "Action passed but ethical review recommended",
        severity="warning",
    )

    @classmethod
    def all_codes(cls) -> list[ContextCode]:
        """Return all defined context codes."""
        return [
            v for k, v in vars(cls).items()
            if isinstance(v, ContextCode)
        ]

    @classmethod
    def by_category(cls, category: str) -> list[ContextCode]:
        """Return all codes in a category."""
        return [c for c in cls.all_codes() if c.category == category]

    @classmethod
    def by_severity(cls, severity: str) -> list[ContextCode]:
        """Return all codes at a given severity."""
        return [c for c in cls.all_codes() if c.severity == severity]

    @classmethod
    def lookup(cls, code_str: str) -> ContextCode | None:
        """Look up a context code by its string value."""
        for c in cls.all_codes():
            if c.code == code_str:
                return c
        return None
