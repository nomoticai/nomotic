"""Behavioral fingerprint — operational signature of an agent's behavior.

A fingerprint captures what "normal" looks like for an agent across four
distributions: what actions it performs (action distribution), where it
operates (target distribution), when it acts (temporal pattern), and how
governance evaluates it (outcome distribution).

Fingerprints form automatically from governance telemetry.  No configuration
is required beyond picking an archetype.  They start from an archetype prior
and are refined as real observations accumulate.

This is NOT personality profiling.  These are operational patterns: what
actions, against what targets, at what frequency, with what outcomes.
Observable, measurable, automatic.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nomotic.types import Action, Verdict

if TYPE_CHECKING:
    from nomotic.priors import ArchetypePrior

__all__ = [
    "BehavioralFingerprint",
    "TemporalPattern",
]

# Rolling window for hourly rate statistics: 168 hours = 1 week.
_ROLLING_WINDOW_HOURS = 168


@dataclass
class TemporalPattern:
    """Temporal activity pattern for an agent.

    Tracks when the agent operates: activity by hour of day,
    actions per time unit statistics, and which hours are
    typically active.
    """

    hourly_distribution: dict[int, float] = field(default_factory=dict)
    # Maps hour (0-23) to fraction of total activity in that hour.
    # Values sum to 1.0 (or 0.0 if no data).

    actions_per_hour_mean: float = 0.0
    actions_per_hour_std: float = 0.0

    active_hours: set[int] = field(default_factory=set)
    # Hours where the agent has been observed active.
    # Not a constraint — just an observation for drift detection.

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "hourly_distribution": {str(k): v for k, v in sorted(self.hourly_distribution.items())},
            "actions_per_hour_mean": self.actions_per_hour_mean,
            "actions_per_hour_std": self.actions_per_hour_std,
            "active_hours": sorted(self.active_hours),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalPattern:
        """Deserialize from dict."""
        return cls(
            hourly_distribution={int(k): v for k, v in data.get("hourly_distribution", {}).items()},
            actions_per_hour_mean=data.get("actions_per_hour_mean", 0.0),
            actions_per_hour_std=data.get("actions_per_hour_std", 0.0),
            active_hours=set(data.get("active_hours", [])),
        )


@dataclass
class BehavioralFingerprint:
    """Operational signature of an agent's behavior.

    Built automatically from governance telemetry. The fingerprint
    captures what "normal" looks like across four distributions:
    what the agent does, where it operates, when it acts, and
    how governance evaluates it.

    Fingerprints start from an archetype prior and are refined
    as the agent operates. After sufficient observations, the
    agent's own fingerprint becomes the authority.
    """

    agent_id: str

    # The four distributions
    action_distribution: dict[str, float] = field(default_factory=dict)
    target_distribution: dict[str, float] = field(default_factory=dict)
    temporal_pattern: TemporalPattern = field(default_factory=TemporalPattern)
    outcome_distribution: dict[str, float] = field(default_factory=dict)

    # Observation counts (for incremental updates and confidence)
    total_observations: int = 0
    observation_window_start: float = field(default_factory=time.time)

    # Raw counts backing the distributions (needed for incremental updates)
    _action_counts: dict[str, int] = field(default_factory=dict)
    _target_counts: dict[str, int] = field(default_factory=dict)
    _hourly_counts: dict[int, int] = field(default_factory=dict)
    _outcome_counts: dict[str, int] = field(default_factory=dict)

    # Per-hour action counts for rate statistics (ring buffer of recent hours)
    # List of (hour_timestamp, count) tuples — hour_timestamp is the start
    # of the hour slot (floored to the hour).
    _hourly_action_totals: list[tuple[float, int]] = field(default_factory=list)

    # Prior blending support
    _prior_weight: int = 0
    _prior_action_counts: dict[str, float] = field(default_factory=dict)
    _prior_target_counts: dict[str, float] = field(default_factory=dict)
    _prior_outcome_counts: dict[str, float] = field(default_factory=dict)

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def confidence(self) -> float:
        """How confident the fingerprint is, based on observation count.

        0.0 at zero observations. Approaches 1.0 asymptotically.
        At 100 observations: ~0.63. At 500: ~0.93. At 1000: ~0.99.

        Uses: 1 - e^(-observations/200)
        """
        return 1.0 - math.exp(-self.total_observations / 200.0)

    def observe(self, action: Action, verdict: Verdict) -> None:
        """Record an observation from a governance evaluation.

        Updates all four distributions incrementally.
        This is called by the runtime after every evaluate().
        """
        with self._lock:
            self.total_observations += 1

            # 1. Action distribution
            self._action_counts[action.action_type] = (
                self._action_counts.get(action.action_type, 0) + 1
            )
            self._recompute_action_distribution()

            # 2. Target distribution
            self._target_counts[action.target] = (
                self._target_counts.get(action.target, 0) + 1
            )
            self._recompute_target_distribution()

            # 3. Temporal pattern
            from datetime import datetime
            dt = datetime.fromtimestamp(action.timestamp)
            hour = dt.hour
            self._hourly_counts[hour] = self._hourly_counts.get(hour, 0) + 1
            self.temporal_pattern.active_hours.add(hour)
            self._recompute_hourly_distribution()
            self._update_hourly_rate(action.timestamp)

            # 4. Outcome distribution
            verdict_name = verdict.name
            self._outcome_counts[verdict_name] = (
                self._outcome_counts.get(verdict_name, 0) + 1
            )
            self._recompute_outcome_distribution()

    def _recompute_action_distribution(self) -> None:
        """Recompute action distribution from raw counts, blending with prior."""
        effective_total = self._prior_weight + self.total_observations
        if effective_total == 0:
            self.action_distribution = {}
            return
        # Collect all keys from both prior and observed
        all_keys = set(self._action_counts) | set(self._prior_action_counts)
        dist: dict[str, float] = {}
        for k in all_keys:
            prior_count = self._prior_action_counts.get(k, 0.0)
            observed_count = self._action_counts.get(k, 0)
            dist[k] = (prior_count + observed_count) / effective_total
        self.action_distribution = dist

    def _recompute_target_distribution(self) -> None:
        """Recompute target distribution from raw counts, blending with prior."""
        effective_total = self._prior_weight + self.total_observations
        if effective_total == 0:
            self.target_distribution = {}
            return
        all_keys = set(self._target_counts) | set(self._prior_target_counts)
        dist: dict[str, float] = {}
        for k in all_keys:
            prior_count = self._prior_target_counts.get(k, 0.0)
            observed_count = self._target_counts.get(k, 0)
            dist[k] = (prior_count + observed_count) / effective_total
        self.target_distribution = dist

    def _recompute_outcome_distribution(self) -> None:
        """Recompute outcome distribution from raw counts, blending with prior."""
        effective_total = self._prior_weight + self.total_observations
        if effective_total == 0:
            self.outcome_distribution = {}
            return
        all_keys = set(self._outcome_counts) | set(self._prior_outcome_counts)
        dist: dict[str, float] = {}
        for k in all_keys:
            prior_count = self._prior_outcome_counts.get(k, 0.0)
            observed_count = self._outcome_counts.get(k, 0)
            dist[k] = (prior_count + observed_count) / effective_total
        self.outcome_distribution = dist

    def _recompute_hourly_distribution(self) -> None:
        """Recompute hourly distribution from raw counts."""
        total = sum(self._hourly_counts.values())
        if total == 0:
            self.temporal_pattern.hourly_distribution = {}
            return
        self.temporal_pattern.hourly_distribution = {
            h: c / total for h, c in self._hourly_counts.items()
        }

    def _update_hourly_rate(self, timestamp: float) -> None:
        """Update the rolling hourly rate statistics.

        Maintains a list of (hour_slot_start, count) entries for the
        last 168 hours (1 week). Computes mean and std from the buffer.
        """
        # Floor timestamp to the hour
        hour_slot = (timestamp // 3600) * 3600
        cutoff = hour_slot - _ROLLING_WINDOW_HOURS * 3600

        # Prune old entries
        self._hourly_action_totals = [
            (ts, c) for ts, c in self._hourly_action_totals if ts > cutoff
        ]

        # Find or create the current hour slot entry
        found = False
        for i, (ts, c) in enumerate(self._hourly_action_totals):
            if ts == hour_slot:
                self._hourly_action_totals[i] = (ts, c + 1)
                found = True
                break
        if not found:
            self._hourly_action_totals.append((hour_slot, 1))

        # Compute mean and std from the buffer
        if self._hourly_action_totals:
            counts = [c for _, c in self._hourly_action_totals]
            n = len(counts)
            mean = sum(counts) / n
            self.temporal_pattern.actions_per_hour_mean = mean
            if n > 1:
                variance = sum((c - mean) ** 2 for c in counts) / (n - 1)
                self.temporal_pattern.actions_per_hour_std = math.sqrt(variance)
            else:
                self.temporal_pattern.actions_per_hour_std = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        with self._lock:
            return {
                "agent_id": self.agent_id,
                "action_distribution": dict(sorted(self.action_distribution.items())),
                "target_distribution": dict(sorted(self.target_distribution.items())),
                "temporal_pattern": self.temporal_pattern.to_dict(),
                "outcome_distribution": dict(sorted(self.outcome_distribution.items())),
                "total_observations": self.total_observations,
                "observation_window_start": self.observation_window_start,
                "confidence": self.confidence,
                "_action_counts": dict(sorted(self._action_counts.items())),
                "_target_counts": dict(sorted(self._target_counts.items())),
                "_hourly_counts": {str(k): v for k, v in sorted(self._hourly_counts.items())},
                "_outcome_counts": dict(sorted(self._outcome_counts.items())),
                "_hourly_action_totals": [
                    [ts, c] for ts, c in self._hourly_action_totals
                ],
                "_prior_weight": self._prior_weight,
                "_prior_action_counts": dict(sorted(self._prior_action_counts.items())),
                "_prior_target_counts": dict(sorted(self._prior_target_counts.items())),
                "_prior_outcome_counts": dict(sorted(self._prior_outcome_counts.items())),
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehavioralFingerprint:
        """Deserialize from dict."""
        fp = cls(
            agent_id=data["agent_id"],
            action_distribution=data.get("action_distribution", {}),
            target_distribution=data.get("target_distribution", {}),
            temporal_pattern=TemporalPattern.from_dict(data.get("temporal_pattern", {})),
            outcome_distribution=data.get("outcome_distribution", {}),
            total_observations=data.get("total_observations", 0),
            observation_window_start=data.get("observation_window_start", 0.0),
            _action_counts=data.get("_action_counts", {}),
            _target_counts=data.get("_target_counts", {}),
            _hourly_counts={int(k): v for k, v in data.get("_hourly_counts", {}).items()},
            _outcome_counts=data.get("_outcome_counts", {}),
            _hourly_action_totals=[
                (ts, c) for ts, c in data.get("_hourly_action_totals", [])
            ],
            _prior_weight=data.get("_prior_weight", 0),
            _prior_action_counts=data.get("_prior_action_counts", {}),
            _prior_target_counts=data.get("_prior_target_counts", {}),
            _prior_outcome_counts=data.get("_prior_outcome_counts", {}),
        )
        return fp

    @classmethod
    def from_prior(cls, agent_id: str, prior: ArchetypePrior) -> BehavioralFingerprint:
        """Create a fingerprint seeded with an archetype prior.

        The prior's distributions are installed as the starting point.
        As real observations accumulate, they gradually replace the prior
        (weighted by observation count vs prior weight).
        """
        pw = prior.prior_weight

        # Convert prior distributions into synthetic count equivalents
        prior_action_counts = {
            k: v * pw for k, v in prior.action_distribution.items()
        }
        prior_target_counts = {
            k: v * pw for k, v in prior.target_categories.items()
        }
        prior_outcome_counts = {
            k: v * pw for k, v in prior.outcome_expectations.items()
        }

        fp = cls(
            agent_id=agent_id,
            _prior_weight=pw,
            _prior_action_counts=prior_action_counts,
            _prior_target_counts=prior_target_counts,
            _prior_outcome_counts=prior_outcome_counts,
        )

        # Initialize distributions from prior (0 real observations)
        fp.action_distribution = dict(prior.action_distribution)
        fp.target_distribution = dict(prior.target_categories)
        fp.outcome_distribution = dict(prior.outcome_expectations)

        # Seed temporal from prior profile
        if prior.temporal_profile.active_hours:
            fp.temporal_pattern.active_hours = set(prior.temporal_profile.active_hours)

        return fp
