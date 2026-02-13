"""Configuration provenance — Git for governance rules.

Every configuration change produces a record: what changed, who changed
it, when, and why.  This is the evidence that configuration was
intentional and authorized.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ProvenanceLog",
    "ProvenanceRecord",
]


# ── ProvenanceRecord ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProvenanceRecord:
    """A single configuration change with attribution.

    Records WHO changed WHAT, WHEN, and WHY.
    """

    record_id: str
    timestamp: float

    # Who made the change
    actor: str           # Who changed it — human email, system ID, or "system"
    actor_type: str      # "human", "system", "api"

    # What changed
    target_type: str     # "scope", "rule", "weight", "boundary", "threshold", "override", "trust"
    target_id: str       # What was changed (dimension name, agent_id, etc.)
    change_type: str     # "add", "remove", "modify"

    # The change details
    previous_value: Any = None  # What it was before (None for adds)
    new_value: Any = None       # What it is now (None for removes)

    # Why
    reason: str = ""     # Human-provided reason for the change
    ticket: str = ""     # Optional reference (e.g., "OPS-4521", "JIRA-123")

    # Context code
    context_code: str = "CONFIG.SCOPE_CHANGED"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "actor_type": self.actor_type,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "change_type": self.change_type,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "ticket": self.ticket,
            "context_code": self.context_code,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProvenanceRecord:
        """Deserialize from dict."""
        return cls(
            record_id=d["record_id"],
            timestamp=d["timestamp"],
            actor=d["actor"],
            actor_type=d["actor_type"],
            target_type=d["target_type"],
            target_id=d["target_id"],
            change_type=d["change_type"],
            previous_value=d.get("previous_value"),
            new_value=d.get("new_value"),
            reason=d.get("reason", ""),
            ticket=d.get("ticket", ""),
            context_code=d.get("context_code", "CONFIG.SCOPE_CHANGED"),
        )


# ── ProvenanceLog ──────────────────────────────────────────────────────────


class ProvenanceLog:
    """Append-only log of configuration changes.

    Every call to configure_agent_scope, add_rule, set_boundaries,
    set_time_window, require_human_for, add_authority_check, etc.
    should produce a provenance record.

    The log answers: "Who authorized this configuration? When was it
    changed? What was it before?"
    """

    def __init__(self, max_records: int = 5000) -> None:
        self._records: list[ProvenanceRecord] = []
        self._max_records = max_records
        self._lock = threading.Lock()

    def record(
        self,
        actor: str,
        target_type: str,
        target_id: str,
        change_type: str,
        *,
        actor_type: str = "human",
        previous_value: Any = None,
        new_value: Any = None,
        reason: str = "",
        ticket: str = "",
        context_code: str = "CONFIG.SCOPE_CHANGED",
    ) -> ProvenanceRecord:
        """Record a configuration change."""
        rec = ProvenanceRecord(
            record_id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            actor=actor,
            actor_type=actor_type,
            target_type=target_type,
            target_id=target_id,
            change_type=change_type,
            previous_value=previous_value,
            new_value=new_value,
            reason=reason,
            ticket=ticket,
            context_code=context_code,
        )
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
        return rec

    def query(
        self,
        *,
        actor: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        change_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[ProvenanceRecord]:
        """Query provenance records.  Filters are AND-combined.

        Returns records in reverse chronological order (newest first).
        """
        with self._lock:
            records = list(self._records)

        if actor is not None:
            records = [r for r in records if r.actor == actor]
        if target_type is not None:
            records = [r for r in records if r.target_type == target_type]
        if target_id is not None:
            records = [r for r in records if r.target_id == target_id]
        if change_type is not None:
            records = [r for r in records if r.change_type == change_type]
        if since is not None:
            records = [r for r in records if r.timestamp >= since]
        if until is not None:
            records = [r for r in records if r.timestamp <= until]

        records.reverse()
        return records[:limit]

    def history(self, target_type: str, target_id: str) -> list[ProvenanceRecord]:
        """Get full change history for a specific configuration target.

        e.g., history("scope", "agent-1") returns all scope changes for
        agent-1, in chronological order (oldest first).
        """
        with self._lock:
            return [
                r for r in self._records
                if r.target_type == target_type and r.target_id == target_id
            ]

    def current_config_version(self) -> str:
        """A hash of the current configuration state.

        Computed from the latest provenance records.  Changes when any
        configuration changes.  Stored in audit records to link governance
        decisions to the configuration that was active when they were made.
        """
        with self._lock:
            if not self._records:
                return "0" * 12
            # Hash all record IDs for a stable version identifier
            content = "|".join(r.record_id for r in self._records)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    @property
    def records(self) -> list[ProvenanceRecord]:
        """All records, newest first.  Returns a copy."""
        with self._lock:
            return list(reversed(self._records))

    def clear(self) -> None:
        """Clear the provenance log.  Use with caution."""
        with self._lock:
            self._records.clear()
