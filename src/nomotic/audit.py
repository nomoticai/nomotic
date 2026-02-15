"""Audit trail — structured, append-only log of governance events.

The audit trail records every governance decision with enough detail to
reconstruct decisions after the fact.  It is the evidence base for
transparency and accountability.

In-memory implementation with a pluggable backend interface.  The trail
accumulates records and can be queried by agent, by time, by context
code, or by severity.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from nomotic.context import CODES, ContextCode

__all__ = [
    "AuditRecord",
    "AuditTrail",
    "build_justification",
    "verify_chain",
]


# ── AuditRecord ───────────────────────────────────────────────────────────


@dataclass
class AuditRecord:
    """A single audit record — one governance event fully documented.

    This is the atomic unit of the audit trail.  It captures everything
    needed to reconstruct a governance decision after the fact.
    """

    # Identity
    record_id: str
    timestamp: float

    # Context code
    context_code: str  # e.g., "GOVERNANCE.ALLOW"
    severity: str      # Copied from the ContextCode for fast filtering

    # The three actors
    agent_id: str
    owner_id: str        # From the certificate's owner field
    user_id: str         # Who triggered this action (if known)

    # The action
    action_id: str
    action_type: str
    action_target: str

    # The governance decision
    verdict: str         # Verdict name: "ALLOW", "DENY", etc.
    ucs: float           # Unified Confidence Score
    tier: int            # Which tier decided (1, 2, 3)

    # Dimension scores snapshot
    dimension_scores: list[dict[str, Any]] = field(default_factory=list)

    # Trust state at time of decision
    trust_score: float = 0.5
    trust_trend: str = "stable"

    # Drift state at time of decision (if available)
    drift_overall: float | None = None
    drift_severity: str | None = None

    # Contextual justification — the narrative explanation
    justification: str = ""

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Hash chain fields
    previous_hash: str = ""      # SHA-256 hash of the previous record
    record_hash: str = ""        # SHA-256 hash of this record (computed on creation)

    def compute_hash(self) -> str:
        """Compute the SHA-256 hash of this record.

        Serializes the record (without record_hash) to canonical JSON,
        prepends the previous_hash, and computes SHA-256.
        """
        d = self._hashable_dict()
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        payload = self.previous_hash + canonical
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hashable_dict(self) -> dict[str, Any]:
        """Dict of all fields except record_hash, for hashing."""
        d: dict[str, Any] = {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "context_code": self.context_code,
            "severity": self.severity,
            "agent_id": self.agent_id,
            "owner_id": self.owner_id,
            "user_id": self.user_id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_target": self.action_target,
            "verdict": self.verdict,
            "ucs": round(self.ucs, 4),
            "tier": self.tier,
            "dimension_scores": self.dimension_scores,
            "trust_score": round(self.trust_score, 4),
            "trust_trend": self.trust_trend,
            "drift_overall": round(self.drift_overall, 4) if self.drift_overall is not None else None,
            "drift_severity": self.drift_severity,
            "justification": self.justification,
            "previous_hash": self.previous_hash,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        d: dict[str, Any] = {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "context_code": self.context_code,
            "severity": self.severity,
            "agent_id": self.agent_id,
            "owner_id": self.owner_id,
            "user_id": self.user_id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_target": self.action_target,
            "verdict": self.verdict,
            "ucs": round(self.ucs, 4),
            "tier": self.tier,
            "dimension_scores": self.dimension_scores,
            "trust_score": round(self.trust_score, 4),
            "trust_trend": self.trust_trend,
            "drift_overall": round(self.drift_overall, 4) if self.drift_overall is not None else None,
            "drift_severity": self.drift_severity,
            "justification": self.justification,
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditRecord:
        """Deserialize from dict."""
        return cls(
            record_id=d["record_id"],
            timestamp=d["timestamp"],
            context_code=d["context_code"],
            severity=d["severity"],
            agent_id=d["agent_id"],
            owner_id=d["owner_id"],
            user_id=d["user_id"],
            action_id=d["action_id"],
            action_type=d["action_type"],
            action_target=d["action_target"],
            verdict=d["verdict"],
            ucs=d["ucs"],
            tier=d["tier"],
            dimension_scores=d.get("dimension_scores", []),
            trust_score=d.get("trust_score", 0.5),
            trust_trend=d.get("trust_trend", "stable"),
            drift_overall=d.get("drift_overall"),
            drift_severity=d.get("drift_severity"),
            justification=d.get("justification", ""),
            metadata=d.get("metadata", {}),
            previous_hash=d.get("previous_hash", ""),
            record_hash=d.get("record_hash", ""),
        )


# ── Justification builder ─────────────────────────────────────────────────


def build_justification(
    verdict: Any,
    action: Any,
    context: Any,
    drift_score: Any = None,
) -> str:
    """Build a contextual justification for a governance decision.

    Synthesizes dimension scores, trust state, and drift into a
    narrative that a regulator, auditor, or board member can understand.
    """
    from nomotic.types import Verdict

    parts: list[str] = []

    # Verdict summary
    verdict_name = verdict.verdict.name if hasattr(verdict.verdict, "name") else str(verdict.verdict)
    tier_str = f"Tier {verdict.tier}"

    if verdict.vetoed_by:
        parts.append(
            f"Action {verdict_name} (UCS: {verdict.ucs:.2f}, {tier_str}, veto)."
        )
    else:
        parts.append(
            f"Action {verdict_name} (UCS: {verdict.ucs:.2f}, {tier_str})."
        )

    # Agent and trust context
    trust = context.trust_profile.overall_trust if hasattr(context, "trust_profile") else 0.5
    parts.append(
        f"Agent (trust: {trust:.2f}) requested '{action.action_type}' on '{action.target}'."
    )

    # Dimension scores detail
    if verdict.vetoed_by:
        # Lead with the veto reasons
        for score in verdict.dimension_scores:
            if score.veto:
                parts.append(
                    f"VETOED by {score.dimension_name} ({score.score:.2f}) — {score.reasoning}."
                )

    # Brief mention of other dimensions
    for score in verdict.dimension_scores:
        if score.veto:
            continue  # Already covered above
        if score.score >= 0.8:
            # Brief mention
            name = score.dimension_name.replace("_", " ").title()
            parts.append(f"{name} ok ({score.score:.2f}).")
        elif score.score >= 0.4:
            name = score.dimension_name.replace("_", " ").title()
            parts.append(f"{name} elevated ({score.score:.2f}) — {score.reasoning}.")
        else:
            name = score.dimension_name.replace("_", " ").title()
            parts.append(f"{name} concern ({score.score:.2f}) — {score.reasoning}.")

    # Drift info
    if drift_score is not None:
        severity = drift_score.severity if hasattr(drift_score, "severity") else "unknown"
        overall = drift_score.overall if hasattr(drift_score, "overall") else 0.0
        if severity in ("high", "critical"):
            parts.append(f"Drift {severity.upper()} ({overall:.2f}).")
        elif severity == "moderate":
            parts.append(f"Drift moderate ({overall:.2f}).")
        elif severity == "none":
            parts.append("No drift detected.")

    # Cap length for normal cases
    result = " ".join(parts)
    if verdict.verdict == Verdict.ALLOW and len(result) > 500:
        result = result[:497] + "..."

    return result


# ── AuditTrail ─────────────────────────────────────────────────────────────


def verify_chain(records: list[AuditRecord]) -> tuple[bool, str]:
    """Verify the hash chain integrity.

    Records should be in chronological order (oldest first).
    Returns (valid, error_message).
    """
    if not records:
        return True, ""

    for i, record in enumerate(records):
        # Skip records without hash chain data (pre-upgrade)
        if not record.record_hash:
            continue

        # Verify the previous_hash links
        if i > 0 and records[i - 1].record_hash:
            if record.previous_hash != records[i - 1].record_hash:
                return False, (
                    f"Chain broken at record {i} ({record.record_id}): "
                    f"previous_hash mismatch"
                )

        # Verify the record's own hash
        expected = record.compute_hash()
        if record.record_hash != expected:
            return False, (
                f"Hash mismatch at record {i} ({record.record_id}): "
                f"expected {expected[:16]}..., got {record.record_hash[:16]}..."
            )

    return True, ""


class AuditTrail:
    """Append-only structured log of governance events.

    The audit trail records every governance decision with enough detail
    to reconstruct decisions after the fact.  It is the evidence base for
    transparency and accountability.

    In-memory implementation with a pluggable backend interface.
    The trail accumulates records and can be queried by agent, by time,
    by context code, or by severity.

    The trail is capped at *max_records* to prevent unbounded memory
    growth.  When the cap is hit, oldest records are dropped.

    Records are hash-chained: each record includes the SHA-256 hash of
    the previous record, creating an immutable audit trail.
    """

    def __init__(self, max_records: int = 10000) -> None:
        self._records: list[AuditRecord] = []
        self._max_records = max_records
        self._lock = threading.Lock()
        self._last_hash: str = ""

    def append(self, record: AuditRecord) -> None:
        """Add a record to the trail.  Thread-safe.

        Computes hash chain: sets previous_hash from the last record
        and computes record_hash for the new record.
        """
        with self._lock:
            # Set hash chain fields
            record.previous_hash = self._last_hash
            record.record_hash = record.compute_hash()
            self._last_hash = record.record_hash

            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

    def query(
        self,
        *,
        agent_id: str | None = None,
        owner_id: str | None = None,
        user_id: str | None = None,
        context_code: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        verdict: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        """Query audit records with filters.  All filters are AND-combined.

        Returns records in reverse chronological order (newest first).
        """
        with self._lock:
            records = list(self._records)

        # Apply filters
        if agent_id is not None:
            records = [r for r in records if r.agent_id == agent_id]
        if owner_id is not None:
            records = [r for r in records if r.owner_id == owner_id]
        if user_id is not None:
            records = [r for r in records if r.user_id == user_id]
        if context_code is not None:
            code_str = str(context_code)
            records = [r for r in records if r.context_code == code_str]
        if category is not None:
            records = [r for r in records if r.context_code.split(".")[0] == category]
        if severity is not None:
            records = [r for r in records if r.severity == severity]
        if verdict is not None:
            records = [r for r in records if r.verdict == verdict]
        if since is not None:
            records = [r for r in records if r.timestamp >= since]
        if until is not None:
            records = [r for r in records if r.timestamp <= until]

        # Reverse chronological, limited
        records.reverse()
        return records[:limit]

    def count(self, **filters: Any) -> int:
        """Count records matching filters (same args as query)."""
        return len(self.query(**filters, limit=self._max_records))

    def summary(
        self,
        agent_id: str | None = None,
        since: float | None = None,
    ) -> dict[str, Any]:
        """Summary statistics for the audit trail.

        Returns:
            {
                "total_records": 1247,
                "by_verdict": {"ALLOW": 1180, "DENY": 45, ...},
                "by_category": {"GOVERNANCE": 1225, "SECURITY": 12, ...},
                "by_severity": {"info": 1100, "warning": 120, ...},
                "recent_alerts": [last 5 records with severity >= "alert"],
            }
        """
        with self._lock:
            records = list(self._records)

        if agent_id is not None:
            records = [r for r in records if r.agent_id == agent_id]
        if since is not None:
            records = [r for r in records if r.timestamp >= since]

        by_verdict: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        alerts: list[AuditRecord] = []

        for r in records:
            by_verdict[r.verdict] = by_verdict.get(r.verdict, 0) + 1
            cat = r.context_code.split(".")[0]
            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[r.severity] = by_severity.get(r.severity, 0) + 1
            if r.severity in ("alert", "critical"):
                alerts.append(r)

        return {
            "total_records": len(records),
            "by_verdict": by_verdict,
            "by_category": by_category,
            "by_severity": by_severity,
            "recent_alerts": [a.to_dict() for a in alerts[-5:]],
        }

    @property
    def records(self) -> list[AuditRecord]:
        """All records, newest first.  Returns a copy."""
        with self._lock:
            return list(reversed(self._records))

    def export(self, format: str = "json") -> str | list[dict[str, Any]]:
        """Export the audit trail.

        Args:
            format: ``"json"`` returns a JSON string.
                    ``"dicts"`` returns a list of dicts.
        """
        with self._lock:
            dicts = [r.to_dict() for r in self._records]
        if format == "dicts":
            return dicts
        return json.dumps(dicts, indent=2)

    def clear(self) -> None:
        """Clear the audit trail.  Use with caution."""
        with self._lock:
            self._records.clear()
