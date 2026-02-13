"""Accountability — owner engagement and user activity tracking.

Owner Activity:
    Track what certificate owners actually DO — not just that they exist.
    An owner who has never acknowledged an alert is a nominal owner.
    This makes accountability quality visible.

User Activity:
    Track user interaction patterns with agents.  Detect when users
    repeatedly test boundaries, attempt manipulation, or request actions
    outside agent scope.  The tracker does NOT store user input content.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OwnerActivity",
    "OwnerActivityLog",
    "UserActivityTracker",
    "UserStats",
]


# ── OwnerActivity ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OwnerActivity:
    """A single owner activity record.

    Activity types:
        ``"alert_acknowledged"`` — owner acknowledged a drift/governance alert
        ``"override_approved"``  — owner approved a human override
        ``"override_rejected"``  — owner rejected a human override
        ``"config_changed"``     — owner changed agent configuration
        ``"report_reviewed"``    — owner viewed a trust/drift/audit report
        ``"agent_suspended"``    — owner suspended an agent
        ``"agent_revoked"``      — owner revoked an agent's certificate
        ``"scope_modified"``     — owner changed agent scope
        ``"rule_modified"``      — owner added/removed governance rules
    """

    timestamp: float
    owner_id: str
    activity_type: str
    target_agent_id: str = ""
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "owner_id": self.owner_id,
            "activity_type": self.activity_type,
        }
        if self.target_agent_id:
            d["target_agent_id"] = self.target_agent_id
        if self.detail:
            d["detail"] = self.detail
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ── OwnerActivityLog ───────────────────────────────────────────────────────


class OwnerActivityLog:
    """Tracks owner engagement with governance.

    Records what certificate owners do: acknowledge alerts, approve
    overrides, change configuration, review reports.  This makes the
    quality of human oversight visible.

    An owner with high engagement is actively governing.  An owner with
    zero engagement is a name on a certificate.
    """

    def __init__(self, max_records_per_owner: int = 1000) -> None:
        self._records: dict[str, list[OwnerActivity]] = {}
        self._max_records_per_owner = max_records_per_owner
        self._lock = threading.Lock()

    def record(
        self,
        owner_id: str,
        activity_type: str,
        target_agent_id: str = "",
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> OwnerActivity:
        """Record an owner activity."""
        activity = OwnerActivity(
            timestamp=time.time(),
            owner_id=owner_id,
            activity_type=activity_type,
            target_agent_id=target_agent_id,
            detail=detail,
            metadata=metadata or {},
        )
        with self._lock:
            records = self._records.setdefault(owner_id, [])
            records.append(activity)
            if len(records) > self._max_records_per_owner:
                self._records[owner_id] = records[-self._max_records_per_owner:]
        return activity

    def get_activities(
        self,
        owner_id: str,
        *,
        activity_type: str | None = None,
        since: float | None = None,
        limit: int = 50,
    ) -> list[OwnerActivity]:
        """Get activities for an owner.  Newest first."""
        with self._lock:
            records = list(self._records.get(owner_id, []))
        if activity_type is not None:
            records = [r for r in records if r.activity_type == activity_type]
        if since is not None:
            records = [r for r in records if r.timestamp >= since]
        records.reverse()
        return records[:limit]

    def engagement_score(self, owner_id: str, window_days: int = 30) -> dict[str, Any]:
        """Compute an engagement score for an owner.

        Returns:
            {
                "owner_id": "clark@acme.com",
                "total_activities": 47,
                "window_days": 30,
                "activities_in_window": 12,
                "alert_response_rate": 0.85,
                "engagement_level": "active",
                "last_activity": 1707849600.0,
            }
        """
        with self._lock:
            all_records = list(self._records.get(owner_id, []))

        cutoff = time.time() - (window_days * 86400)
        in_window = [r for r in all_records if r.timestamp >= cutoff]

        # Count alert-related activities
        alert_acks = sum(
            1 for r in all_records if r.activity_type == "alert_acknowledged"
        )

        # Engagement level
        window_count = len(in_window)
        if window_count >= 10:
            level = "active"
        elif window_count >= 1:
            level = "passive"
        else:
            level = "absent"

        last_activity = all_records[-1].timestamp if all_records else 0.0

        return {
            "owner_id": owner_id,
            "total_activities": len(all_records),
            "window_days": window_days,
            "activities_in_window": window_count,
            "alert_response_rate": round(alert_acks / max(len(all_records), 1), 4),
            "engagement_level": level,
            "last_activity": last_activity,
        }

    def summary(self, owner_id: str) -> dict[str, Any]:
        """Summary of owner activities."""
        with self._lock:
            all_records = list(self._records.get(owner_id, []))

        by_type: dict[str, int] = {}
        for r in all_records:
            by_type[r.activity_type] = by_type.get(r.activity_type, 0) + 1

        return {
            "owner_id": owner_id,
            "total_activities": len(all_records),
            "by_type": by_type,
            "last_activity": all_records[-1].timestamp if all_records else None,
            "recent": [r.to_dict() for r in all_records[-5:]],
        }


# ── UserStats ──────────────────────────────────────────────────────────────


@dataclass
class UserStats:
    """Aggregated interaction statistics for a user."""

    user_id: str
    total_interactions: int = 0
    allow_count: int = 0
    deny_count: int = 0
    escalate_count: int = 0
    out_of_scope_count: int = 0
    last_interaction: float = 0.0
    agents_interacted: set[str] = field(default_factory=set)

    @property
    def denial_rate(self) -> float:
        if self.total_interactions == 0:
            return 0.0
        return self.deny_count / self.total_interactions

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "total_interactions": self.total_interactions,
            "allow_count": self.allow_count,
            "deny_count": self.deny_count,
            "escalate_count": self.escalate_count,
            "out_of_scope_count": self.out_of_scope_count,
            "denial_rate": round(self.denial_rate, 4),
            "last_interaction": self.last_interaction,
            "agents_interacted": sorted(self.agents_interacted),
        }


# ── UserActivityTracker ────────────────────────────────────────────────────


class UserActivityTracker:
    """Tracks user interaction patterns with agents.

    Detects when users repeatedly test boundaries, attempt manipulation,
    or request actions outside agent scope.  This information feeds into
    audit records and can trigger security context codes.

    The tracker does NOT store user input content.  It tracks patterns:
    how often a user triggers out-of-scope requests, how many times
    they've been denied, whether their request patterns are anomalous.
    """

    def __init__(self, max_users: int = 1000) -> None:
        self._user_stats: dict[str, UserStats] = {}
        self._max_users = max_users
        self._lock = threading.Lock()

    def record_interaction(
        self,
        user_id: str,
        agent_id: str,
        verdict: str,
        context_code: str,
    ) -> UserStats:
        """Record a user interaction.  Returns updated stats."""
        with self._lock:
            if user_id not in self._user_stats:
                # Evict oldest if at capacity
                if len(self._user_stats) >= self._max_users:
                    oldest_id = min(
                        self._user_stats,
                        key=lambda k: self._user_stats[k].last_interaction,
                    )
                    del self._user_stats[oldest_id]
                self._user_stats[user_id] = UserStats(user_id=user_id)

            stats = self._user_stats[user_id]
            stats.total_interactions += 1
            stats.last_interaction = time.time()
            stats.agents_interacted.add(agent_id)

            if verdict == "ALLOW":
                stats.allow_count += 1
            elif verdict == "DENY":
                stats.deny_count += 1
            elif verdict == "ESCALATE":
                stats.escalate_count += 1

            if context_code.startswith("SCOPE.VIOLATION") or context_code == "GOVERNANCE.VETO":
                stats.out_of_scope_count += 1

            return stats

    def get_stats(self, user_id: str) -> UserStats | None:
        """Get interaction statistics for a user."""
        with self._lock:
            stats = self._user_stats.get(user_id)
            return stats

    def classify_request(
        self,
        user_id: str,
        agent_id: str,
        verdict: str,
    ) -> str:
        """Classify a user request based on patterns.

        Returns:
            ``"normal"``           — typical user request
            ``"out_of_scope"``     — user asked for something the agent can't do
            ``"boundary_testing"`` — user has repeatedly hit boundaries
            ``"suspicious"``       — pattern suggests manipulation attempt
        """
        with self._lock:
            stats = self._user_stats.get(user_id)

        if stats is None:
            return "normal"

        # Suspicious: high denial rate with many interactions
        if stats.total_interactions >= 10 and stats.denial_rate > 0.5:
            return "suspicious"

        # Boundary testing: repeated denials
        if stats.deny_count >= 5:
            return "boundary_testing"

        # Out of scope verdict
        if verdict == "DENY":
            return "out_of_scope"

        return "normal"

    def get_flagged_users(self, threshold: int = 5) -> list[str]:
        """Get users whose denial count exceeds a threshold."""
        with self._lock:
            return [
                uid for uid, stats in self._user_stats.items()
                if stats.deny_count >= threshold
            ]
