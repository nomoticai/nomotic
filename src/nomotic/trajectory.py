"""Trust trajectory — timestamped history of trust changes with reasons.

Trust trajectory answers: "Why is this agent's trust at 0.72?" with a
sequence of trust change events, each carrying a timestamp, delta, source,
and human-readable reason.

This is the evidence base for verifiable trust.  Rather than a single
opaque float, trajectory makes trust *explainable*.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TrustEvent",
    "TrustTrajectory",
    "SOURCE_VERDICT_ALLOW",
    "SOURCE_VERDICT_DENY",
    "SOURCE_COMPLETION_SUCCESS",
    "SOURCE_COMPLETION_INTERRUPT",
    "SOURCE_TIME_DECAY",
    "SOURCE_DRIFT_ADJUSTMENT",
    "SOURCE_DRIFT_RECOVERY",
    "SOURCE_MANUAL_UPDATE",
    "SOURCE_CERTIFICATE_SYNC",
]

# ── Trust event sources ────────────────────────────────────────────────

SOURCE_VERDICT_ALLOW = "verdict:allow"
SOURCE_VERDICT_DENY = "verdict:deny"
SOURCE_COMPLETION_SUCCESS = "completion:success"
SOURCE_COMPLETION_INTERRUPT = "completion:interrupt"
SOURCE_TIME_DECAY = "time_decay"
SOURCE_DRIFT_ADJUSTMENT = "drift:adjustment"
SOURCE_DRIFT_RECOVERY = "drift:recovery"
SOURCE_MANUAL_UPDATE = "manual:update"
SOURCE_CERTIFICATE_SYNC = "certificate:sync"


# ── TrustEvent ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrustEvent:
    """A single trust change with its cause.

    The trajectory is a sequence of these events, oldest to newest.
    """

    timestamp: float
    trust_before: float
    trust_after: float
    delta: float
    source: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        """'up', 'down', or 'unchanged'."""
        if self.delta > 0.001:
            return "up"
        if self.delta < -0.001:
            return "down"
        return "unchanged"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "trust_before": round(self.trust_before, 6),
            "trust_after": round(self.trust_after, 6),
            "delta": round(self.delta, 6),
            "source": self.source,
            "reason": self.reason,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ── TrustTrajectory ────────────────────────────────────────────────────


class TrustTrajectory:
    """History of trust changes for an agent.

    Records every significant trust change with timestamp, delta,
    source, and reason.  Provides analysis methods for understanding
    trust trends.

    Capped at *max_events* to prevent unbounded growth.
    """

    def __init__(self, agent_id: str, max_events: int = 500) -> None:
        self._agent_id = agent_id
        self._max_events = max_events
        self._events: list[TrustEvent] = []
        self._lock = threading.Lock()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def record(
        self,
        trust_before: float,
        trust_after: float,
        source: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> TrustEvent | None:
        """Record a trust change.

        Only records if the delta is significant (|delta| > 0.001).
        Returns the event if recorded, None if the change was too small.
        """
        delta = trust_after - trust_before
        if abs(delta) <= 0.001:
            return None

        event = TrustEvent(
            timestamp=time.time(),
            trust_before=trust_before,
            trust_after=trust_after,
            delta=delta,
            source=source,
            reason=reason,
            metadata=metadata or {},
        )

        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

        return event

    @property
    def events(self) -> list[TrustEvent]:
        """All recorded events, oldest first.  Returns a copy."""
        with self._lock:
            return list(self._events)

    @property
    def latest(self) -> TrustEvent | None:
        """Most recent event, or None if empty."""
        with self._lock:
            return self._events[-1] if self._events else None

    def events_since(self, timestamp: float) -> list[TrustEvent]:
        """Events after a given timestamp."""
        with self._lock:
            return [e for e in self._events if e.timestamp > timestamp]

    def events_by_source(self, source_prefix: str) -> list[TrustEvent]:
        """Events matching a source prefix.

        e.g., events_by_source("drift") returns all drift-related events.
        e.g., events_by_source("verdict:deny") returns all denial events.
        """
        with self._lock:
            return [e for e in self._events if e.source.startswith(source_prefix)]

    @property
    def net_delta(self) -> float:
        """Net trust change across all recorded events."""
        with self._lock:
            return sum(e.delta for e in self._events)

    @property
    def trend(self) -> str:
        """Recent trust trend based on last 20 events.

        Returns 'rising', 'falling', 'stable', or 'volatile'.

        - 'rising': majority of recent deltas are positive
        - 'falling': majority of recent deltas are negative
        - 'stable': most recent deltas are near zero
        - 'volatile': mix of significant positive and negative deltas
        """
        with self._lock:
            recent = self._events[-20:]
        if not recent:
            return "stable"

        positive = 0
        negative = 0
        near_zero = 0

        for e in recent:
            if e.delta > 0.001:
                positive += 1
            elif e.delta < -0.001:
                negative += 1
            else:
                near_zero += 1

        total = len(recent)

        # Stable: majority near zero (shouldn't happen often since we
        # filter insignificant deltas, but handles edge cases)
        if near_zero > total * 0.6:
            return "stable"

        significant = positive + negative
        if significant == 0:
            return "stable"

        # Volatile: both positive and negative are well-represented
        if positive > 0 and negative > 0:
            minor = min(positive, negative)
            if minor / significant >= 0.3:
                return "volatile"

        # Rising or falling: clear majority one direction
        if positive > negative:
            return "rising"
        if negative > positive:
            return "falling"

        return "stable"

    def summary(self) -> dict[str, Any]:
        """Summary statistics for the trajectory.

        Returns:
            {
                "agent_id": "agent-1",
                "total_events": 247,
                "net_delta": -0.13,
                "trend": "falling",
                "current_trust": 0.72,
                "sources": {
                    "verdict:allow": {"count": 180, "net_delta": +0.18},
                    ...
                },
                "recent_events": [last 5 events as dicts],
            }
        """
        with self._lock:
            events = list(self._events)

        # Group by source
        sources: dict[str, dict[str, Any]] = {}
        for e in events:
            if e.source not in sources:
                sources[e.source] = {"count": 0, "net_delta": 0.0}
            sources[e.source]["count"] += 1
            sources[e.source]["net_delta"] += e.delta

        # Round net_delta values
        for info in sources.values():
            info["net_delta"] = round(info["net_delta"], 6)

        current_trust = events[-1].trust_after if events else None

        return {
            "agent_id": self._agent_id,
            "total_events": len(events),
            "net_delta": round(sum(e.delta for e in events), 6),
            "trend": self.trend,
            "current_trust": current_trust,
            "sources": sources,
            "recent_events": [e.to_dict() for e in events[-5:]],
        }

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize all events to a list of dicts."""
        with self._lock:
            return [e.to_dict() for e in self._events]

    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)
