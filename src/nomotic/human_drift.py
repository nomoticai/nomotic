"""Human oversight drift detection.

Monitors human interaction patterns to detect when oversight is degrading.
Tracks approval times, override rates, rationale depth, and engagement
patterns per reviewer. Uses the same drift detection approach as agent
behavioral monitoring.

The insight: governance fails not when agents misbehave, but when humans
stop paying attention. A 93% approval rate with 2-second review times
across 500 decisions is not oversight — it's rubber-stamping.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "HumanAuditStore",
    "HumanDriftCalculator",
    "HumanDriftMonitor",
    "HumanDriftResult",
    "HumanInteractionEvent",
    "HumanInteractionProfile",
]


# ── Data types ─────────────────────────────────────────────────────────


@dataclass
class HumanInteractionEvent:
    """A single human interaction with the governance system."""

    timestamp: float
    reviewer_id: str
    agent_id: str
    action_id: str  # links to the governance evaluation
    event_type: str  # "approval", "denial", "override", "escalation_response", "review"
    decision: str  # "approved", "denied", "modified", "deferred"
    review_duration_seconds: float  # time between event presented and decision made
    rationale: str  # explanation provided (empty string if none)
    rationale_depth: int  # word count of rationale (0 = no explanation given)
    context_viewed: bool  # did reviewer expand/view full context?
    modifications: list[str] = field(default_factory=list)  # what was changed if "modified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HumanInteractionEvent:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Interaction profile ────────────────────────────────────────────────


@dataclass
class HumanInteractionProfile:
    """Behavioral profile of a human reviewer's oversight patterns.

    Captures what "engaged oversight" looks like for this reviewer,
    establishing a baseline against which drift is measured.
    """

    reviewer_id: str

    # Review timing
    mean_review_duration: float = 0.0  # average seconds per review
    median_review_duration: float = 0.0
    min_review_duration: float = 0.0
    max_review_duration: float = 0.0
    review_duration_stddev: float = 0.0

    # Decision patterns
    approval_rate: float = 0.0  # 0.0 to 1.0
    denial_rate: float = 0.0
    modification_rate: float = 0.0
    deferral_rate: float = 0.0
    override_rate: float = 0.0  # how often they override agent recommendations

    # Engagement quality
    mean_rationale_depth: float = 0.0  # average word count
    rationale_provided_rate: float = 0.0  # % of decisions with rationale
    context_view_rate: float = 0.0  # % of reviews where full context was viewed

    # Volume
    total_interactions: int = 0
    interactions_per_hour: float = 0.0  # throughput

    # Window
    window_start: float = 0.0
    window_end: float = 0.0

    @classmethod
    def from_events(
        cls, reviewer_id: str, events: list[HumanInteractionEvent]
    ) -> HumanInteractionProfile:
        """Build a profile from a list of interaction events."""
        if not events:
            return cls(reviewer_id=reviewer_id)

        durations = [e.review_duration_seconds for e in events]
        decisions = [e.decision for e in events]
        rationale_depths = [e.rationale_depth for e in events]

        total = len(events)
        time_span = max(1.0, events[-1].timestamp - events[0].timestamp)

        return cls(
            reviewer_id=reviewer_id,
            mean_review_duration=statistics.mean(durations),
            median_review_duration=statistics.median(durations),
            min_review_duration=min(durations),
            max_review_duration=max(durations),
            review_duration_stddev=(
                statistics.stdev(durations) if len(durations) > 1 else 0.0
            ),
            approval_rate=decisions.count("approved") / total,
            denial_rate=decisions.count("denied") / total,
            modification_rate=decisions.count("modified") / total,
            deferral_rate=decisions.count("deferred") / total,
            override_rate=sum(1 for e in events if e.event_type == "override") / total,
            mean_rationale_depth=statistics.mean(rationale_depths),
            rationale_provided_rate=sum(1 for d in rationale_depths if d > 0) / total,
            context_view_rate=sum(1 for e in events if e.context_viewed) / total,
            total_interactions=total,
            interactions_per_hour=total / (time_span / 3600) if time_span > 0 else 0,
            window_start=events[0].timestamp,
            window_end=events[-1].timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Drift result ───────────────────────────────────────────────────────


@dataclass
class HumanDriftResult:
    """Result of human oversight drift calculation."""

    reviewer_id: str
    overall_drift: float  # 0.0 (no drift) to 1.0 (complete drift)
    drift_category: str  # "stable", "mild", "moderate", "severe", "critical"

    # Component drift scores
    timing_drift: float  # change in review duration
    decision_drift: float  # change in approval/denial pattern
    engagement_drift: float  # change in rationale depth / context viewing
    throughput_drift: float  # change in reviews per hour

    # Specific alerts
    alerts: list[str]  # human-readable alert messages

    # Raw comparison
    baseline_profile: HumanInteractionProfile
    recent_profile: HumanInteractionProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "overall_drift": round(self.overall_drift, 3),
            "drift_category": self.drift_category,
            "timing_drift": round(self.timing_drift, 3),
            "decision_drift": round(self.decision_drift, 3),
            "engagement_drift": round(self.engagement_drift, 3),
            "throughput_drift": round(self.throughput_drift, 3),
            "alerts": self.alerts,
        }


# ── Drift calculator ──────────────────────────────────────────────────


class HumanDriftCalculator:
    """Calculates drift in human oversight patterns.

    Compares a baseline interaction profile against a recent window
    and produces drift scores with specific alerts.
    """

    # Thresholds for alerts
    MIN_REVIEW_DURATION_ALERT = 3.0  # seconds — reviews faster than this are suspicious
    APPROVAL_RATE_ALERT = 0.95  # 95%+ approval suggests rubber-stamping
    RATIONALE_DROP_ALERT = 0.5  # 50% drop in rationale provision
    TIMING_DROP_ALERT = 0.7  # 70% drop in review duration
    ZERO_OVERRIDE_WINDOW = 100  # 100+ decisions with zero overrides

    def calculate(
        self,
        baseline: HumanInteractionProfile,
        recent: HumanInteractionProfile,
    ) -> HumanDriftResult:
        """Calculate human oversight drift."""
        alerts: list[str] = []

        # Timing drift — is the reviewer getting faster (less careful)?
        timing_drift = 0.0
        if baseline.mean_review_duration > 0:
            duration_ratio = recent.mean_review_duration / baseline.mean_review_duration
            if duration_ratio < 1.0:
                timing_drift = 1.0 - duration_ratio  # 0 = same, 1 = instant

            if duration_ratio < (1.0 - self.TIMING_DROP_ALERT):
                pct = round((1.0 - duration_ratio) * 100)
                alerts.append(
                    f"Review duration dropped {pct}%: "
                    f"{baseline.mean_review_duration:.1f}s → {recent.mean_review_duration:.1f}s"
                )

        if recent.mean_review_duration < self.MIN_REVIEW_DURATION_ALERT:
            alerts.append(
                f"Average review time is {recent.mean_review_duration:.1f}s — "
                f"below {self.MIN_REVIEW_DURATION_ALERT}s minimum for meaningful review"
            )

        # Decision drift — is the reviewer approving everything?
        decision_drift = 0.0
        if recent.approval_rate > self.APPROVAL_RATE_ALERT:
            decision_drift = (recent.approval_rate - self.APPROVAL_RATE_ALERT) / (
                1.0 - self.APPROVAL_RATE_ALERT
            )
            alerts.append(
                f"Approval rate is {recent.approval_rate:.0%} across "
                f"{recent.total_interactions} decisions — possible rubber-stamping"
            )

        if (
            recent.override_rate == 0
            and recent.total_interactions >= self.ZERO_OVERRIDE_WINDOW
        ):
            decision_drift = max(decision_drift, 0.5)
            alerts.append(
                f"Zero overrides across {recent.total_interactions} decisions — "
                f"reviewer may not be exercising independent judgment"
            )

        # Engagement drift — is the reviewer providing less explanation?
        engagement_drift = 0.0
        if baseline.rationale_provided_rate > 0:
            rationale_ratio = (
                recent.rationale_provided_rate / baseline.rationale_provided_rate
            )
            if rationale_ratio < 1.0:
                engagement_drift = 1.0 - rationale_ratio

            if rationale_ratio < (1.0 - self.RATIONALE_DROP_ALERT):
                alerts.append(
                    f"Rationale provision dropped from {baseline.rationale_provided_rate:.0%} "
                    f"to {recent.rationale_provided_rate:.0%}"
                )

        if baseline.context_view_rate > 0:
            context_ratio = recent.context_view_rate / max(
                baseline.context_view_rate, 0.01
            )
            if context_ratio < 0.5:
                engagement_drift = max(engagement_drift, 0.6)
                alerts.append(
                    f"Context viewing dropped from {baseline.context_view_rate:.0%} "
                    f"to {recent.context_view_rate:.0%} — reviewer may not be reading details"
                )

        # Throughput drift — is the reviewer processing faster than humanly careful?
        throughput_drift = 0.0
        if baseline.interactions_per_hour > 0:
            throughput_ratio = (
                recent.interactions_per_hour / baseline.interactions_per_hour
            )
            if throughput_ratio > 2.0:  # 2x throughput = suspicious
                throughput_drift = min(1.0, (throughput_ratio - 1.0) / 3.0)
                alerts.append(
                    f"Review throughput increased from {baseline.interactions_per_hour:.1f}/hr "
                    f"to {recent.interactions_per_hour:.1f}/hr"
                )

        # Overall drift — weighted combination
        overall = (
            timing_drift * 0.30
            + decision_drift * 0.30
            + engagement_drift * 0.25
            + throughput_drift * 0.15
        )

        # Categorize
        if overall < 0.1:
            category = "stable"
        elif overall < 0.25:
            category = "mild"
        elif overall < 0.5:
            category = "moderate"
        elif overall < 0.75:
            category = "severe"
        else:
            category = "critical"

        return HumanDriftResult(
            reviewer_id=baseline.reviewer_id,
            overall_drift=overall,
            drift_category=category,
            timing_drift=timing_drift,
            decision_drift=decision_drift,
            engagement_drift=engagement_drift,
            throughput_drift=throughput_drift,
            alerts=alerts,
            baseline_profile=baseline,
            recent_profile=recent,
        )


# ── Drift monitor ─────────────────────────────────────────────────────


class HumanDriftMonitor:
    """Orchestrates human oversight drift detection.

    Manages sliding windows of interaction events per reviewer,
    maintains baseline profiles, and runs periodic drift checks.
    """

    def __init__(
        self,
        baseline_window: int = 200,  # events for baseline
        recent_window: int = 50,  # events for recent comparison
    ) -> None:
        self._baseline_window = baseline_window
        self._recent_window = recent_window
        self._calculator = HumanDriftCalculator()

        # Per-reviewer event buffers
        self._events: dict[str, list[HumanInteractionEvent]] = {}
        self._baselines: dict[str, HumanInteractionProfile] = {}
        self._alerts: list[HumanDriftResult] = []

    def record_event(
        self, event: HumanInteractionEvent
    ) -> HumanDriftResult | None:
        """Record a human interaction event and check for drift.

        Returns a HumanDriftResult if drift is detected, None otherwise.
        """
        reviewer = event.reviewer_id
        if reviewer not in self._events:
            self._events[reviewer] = []
        self._events[reviewer].append(event)

        events = self._events[reviewer]

        # Build baseline from first N events
        if len(events) < self._baseline_window + self._recent_window:
            return None  # not enough data yet

        # Establish or refresh baseline
        if reviewer not in self._baselines:
            baseline_events = events[: self._baseline_window]
            self._baselines[reviewer] = HumanInteractionProfile.from_events(
                reviewer, baseline_events
            )

        # Compare recent window against baseline
        recent_events = events[-self._recent_window :]
        recent_profile = HumanInteractionProfile.from_events(
            reviewer, recent_events
        )

        result = self._calculator.calculate(self._baselines[reviewer], recent_profile)

        if result.alerts:
            self._alerts.append(result)
            return result

        return None

    def get_alerts(self) -> list[HumanDriftResult]:
        """Get all unacknowledged drift alerts."""
        return list(self._alerts)

    def get_profile(self, reviewer_id: str) -> HumanInteractionProfile | None:
        """Get the current baseline profile for a reviewer."""
        return self._baselines.get(reviewer_id)

    def get_recent_profile(
        self, reviewer_id: str
    ) -> HumanInteractionProfile | None:
        """Get the most recent interaction profile for a reviewer."""
        events = self._events.get(reviewer_id, [])
        if len(events) < self._recent_window:
            return None
        recent = events[-self._recent_window :]
        return HumanInteractionProfile.from_events(reviewer_id, recent)

    def get_reviewer_ids(self) -> list[str]:
        """Get all tracked reviewer IDs."""
        return list(self._events.keys())


# ── Human audit store ──────────────────────────────────────────────────


class HumanAuditStore:
    """Persistent append-only storage for human interaction events.

    Stores events in ~/.nomotic/audit/human/<reviewer_id>.jsonl
    with SHA-256 hash chaining for tamper evidence.
    """

    def __init__(self, base_dir: Path) -> None:
        self._dir = base_dir / "audit" / "human"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _reviewer_file(self, reviewer_id: str) -> Path:
        safe_name = reviewer_id.lower().replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe_name}.jsonl"

    def get_last_hash(self, reviewer_id: str) -> str:
        """Get the hash of the most recent record for chain linking."""
        path = self._reviewer_file(reviewer_id)
        if not path.exists():
            return ""
        last_line = ""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if not last_line:
            return ""
        try:
            record = json.loads(last_line)
            return record.get("record_hash", "")
        except json.JSONDecodeError:
            return ""

    def _compute_hash(self, record_dict: dict[str, Any], previous_hash: str) -> str:
        hashable = dict(record_dict)
        hashable.pop("record_hash", None)
        hashable["previous_hash"] = previous_hash
        canonical = json.dumps(hashable, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(self, event: HumanInteractionEvent) -> None:
        """Append an event to the reviewer's log file with hash chaining."""
        path = self._reviewer_file(event.reviewer_id)
        previous_hash = self.get_last_hash(event.reviewer_id)
        record = event.to_dict()
        record["previous_hash"] = previous_hash
        record["record_hash"] = self._compute_hash(record, previous_hash)
        line = json.dumps(record, separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def query(
        self, reviewer_id: str, limit: int = 50
    ) -> list[HumanInteractionEvent]:
        """Read events for a reviewer, most recent first."""
        path = self._reviewer_file(reviewer_id)
        if not path.exists():
            return []
        events: list[HumanInteractionEvent] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    events.append(HumanInteractionEvent.from_dict(data))
                except (json.JSONDecodeError, TypeError):
                    continue
        return list(reversed(events[-limit:]))

    def query_all(self, reviewer_id: str) -> list[HumanInteractionEvent]:
        """Read ALL events for a reviewer in chronological order."""
        path = self._reviewer_file(reviewer_id)
        if not path.exists():
            return []
        events: list[HumanInteractionEvent] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(HumanInteractionEvent.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return events

    def verify_chain(self, reviewer_id: str) -> tuple[bool, int, str]:
        """Verify the hash chain integrity.

        Returns (is_valid, record_count, error_message).
        """
        path = self._reviewer_file(reviewer_id)
        if not path.exists():
            return True, 0, "No records found"

        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not records:
            return True, 0, "No records found"

        previous_hash = ""
        for i, record in enumerate(records):
            expected = self._compute_hash(record, previous_hash)
            if record.get("record_hash") != expected:
                return False, len(records), (
                    f"TAMPERING DETECTED at record #{i + 1}. "
                    f"Expected hash: {expected[:24]}... "
                    f"Actual hash: {record.get('record_hash', '')[:24]}..."
                )
            if record.get("previous_hash") != previous_hash:
                return False, len(records), (
                    f"CHAIN BREAK at record #{i + 1}. Previous hash mismatch."
                )
            previous_hash = record["record_hash"]

        return True, len(records), "All records verified"

    def list_reviewers(self) -> list[str]:
        """List all reviewer IDs that have log files."""
        return [p.stem for p in self._dir.glob("*.jsonl")]
