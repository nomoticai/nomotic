"""Drift monitor — orchestrates drift detection for all agents.

Owns sliding windows, runs the drift calculator on configurable
intervals, and produces alerts when drift exceeds thresholds.

This is the integration layer that the runtime and dimensions talk to.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from nomotic.drift import DriftCalculator, DriftScore
from nomotic.fingerprint import BehavioralFingerprint
from nomotic.priors import PriorRegistry
from nomotic.types import Action, Verdict
from nomotic.window import SlidingWindow

__all__ = [
    "DriftAlert",
    "DriftConfig",
    "DriftMonitor",
]


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    window_size: int = 100
    """Number of recent observations in the sliding window.
    Smaller = faster detection, noisier.  Larger = slower, more stable."""

    check_interval: int = 10
    """Recompute drift every *N* observations.  Avoids computing on
    every single action."""

    min_observations: int = 20
    """Minimum observations in the window before drift is computed.
    Below this, the sample is too small to be meaningful."""

    alert_threshold_moderate: float = 0.20
    """Drift score above this triggers a moderate alert."""

    alert_threshold_high: float = 0.40
    """Drift score above this triggers a high alert."""

    alert_threshold_critical: float = 0.60
    """Drift score above this triggers a critical alert."""


@dataclass
class DriftAlert:
    """Alert generated when behavioural drift exceeds a threshold."""

    agent_id: str
    severity: str  # "moderate", "high", "critical"
    drift_score: DriftScore
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "agent_id": self.agent_id,
            "severity": self.severity,
            "drift_score": self.drift_score.to_dict(),
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
        }


_SEVERITY_RANK = {"moderate": 1, "high": 2, "critical": 3}


class DriftMonitor:
    """Orchestrates drift detection for all agents.

    Owns sliding windows, runs the drift calculator on configurable
    intervals, and produces alerts when drift exceeds thresholds.

    Wired into the :class:`FingerprintObserver` — when the observer
    records an observation, it also feeds the drift monitor's sliding
    window.

    Usage::

        monitor = DriftMonitor(config=DriftConfig(window_size=100))

        # Called by the observer after every evaluation:
        monitor.observe(agent_id, action, verdict, archetype="customer-experience")

        # Check current drift for an agent:
        score = monitor.get_drift(agent_id)

        # Get active alerts:
        alerts = monitor.get_alerts(agent_id)
    """

    def __init__(
        self,
        config: DriftConfig | None = None,
        prior_registry: PriorRegistry | None = None,
    ) -> None:
        self._config = config or DriftConfig()
        self._prior_registry = prior_registry
        self._calculator = DriftCalculator()
        self._windows: dict[str, SlidingWindow] = {}
        self._latest_drift: dict[str, DriftScore] = {}
        self._alerts: dict[str, list[DriftAlert]] = {}
        self._observation_counters: dict[str, int] = {}
        self._agent_archetypes: dict[str, str] = {}
        self._lock = threading.Lock()

    def observe(
        self,
        agent_id: str,
        action: Action,
        verdict: Verdict,
        baseline_fingerprint: BehavioralFingerprint | None = None,
        archetype: str | None = None,
    ) -> DriftScore | None:
        """Record an observation and optionally recompute drift.

        Drift is recomputed every *check_interval* observations.
        Returns the :class:`DriftScore` if recomputed, ``None`` otherwise.

        Args:
            agent_id: The agent being observed.
            action: The action that was evaluated.
            verdict: The governance verdict.
            baseline_fingerprint: The agent's full behavioural fingerprint
                (from :class:`FingerprintObserver`).  Required for drift
                computation.
            archetype: The agent's archetype (for drift_weights lookup).
        """
        with self._lock:
            # Ensure window exists
            if agent_id not in self._windows:
                self._windows[agent_id] = SlidingWindow(
                    agent_id=agent_id,
                    window_size=self._config.window_size,
                )
            if archetype:
                self._agent_archetypes[agent_id] = archetype

            # Add to window
            self._windows[agent_id].observe(action, verdict)

            # Increment counter
            count = self._observation_counters.get(agent_id, 0) + 1
            self._observation_counters[agent_id] = count

            # Check if we should recompute drift
            if count % self._config.check_interval != 0:
                return None

            window = self._windows[agent_id]
            if window.size < self._config.min_observations:
                return None

            if baseline_fingerprint is None:
                return None

            # Get drift weights from archetype prior
            drift_weights: dict[str, float] | None = None
            arch = self._agent_archetypes.get(agent_id)
            if arch and self._prior_registry:
                prior = self._prior_registry.get_for_agent(arch)
                if prior:
                    drift_weights = prior.drift_weights

            # Compute drift
            score = self._calculator.compare(
                baseline=baseline_fingerprint,
                recent=window.fingerprint,
                drift_weights=drift_weights,
            )
            self._latest_drift[agent_id] = score

            # Check alert thresholds
            self._check_alerts(agent_id, score)

            return score

    def get_drift(self, agent_id: str) -> DriftScore | None:
        """Get the latest drift score for an agent.

        Returns ``None`` if drift has never been computed for this agent.
        """
        with self._lock:
            return self._latest_drift.get(agent_id)

    def get_alerts(
        self,
        agent_id: str | None = None,
        *,
        unacknowledged_only: bool = False,
    ) -> list[DriftAlert]:
        """Get drift alerts.

        Args:
            agent_id: If provided, only alerts for this agent.
                If ``None``, all alerts across all agents.
            unacknowledged_only: If ``True``, only return unacknowledged alerts.
        """
        with self._lock:
            if agent_id is not None:
                alerts = list(self._alerts.get(agent_id, []))
            else:
                alerts = [a for lst in self._alerts.values() for a in lst]
            if unacknowledged_only:
                alerts = [a for a in alerts if not a.acknowledged]
            return alerts

    def acknowledge_alert(self, agent_id: str, index: int) -> bool:
        """Mark an alert as acknowledged.  Returns ``True`` if found."""
        with self._lock:
            alerts = self._alerts.get(agent_id, [])
            if 0 <= index < len(alerts):
                alerts[index].acknowledged = True
                return True
            return False

    def get_window(self, agent_id: str) -> SlidingWindow | None:
        """Get the sliding window for an agent (for inspection/debugging)."""
        with self._lock:
            return self._windows.get(agent_id)

    def reset(self, agent_id: str) -> None:
        """Reset all drift state for an agent."""
        with self._lock:
            self._windows.pop(agent_id, None)
            self._latest_drift.pop(agent_id, None)
            self._alerts.pop(agent_id, None)
            self._observation_counters.pop(agent_id, None)
            self._agent_archetypes.pop(agent_id, None)

    def _check_alerts(self, agent_id: str, score: DriftScore) -> None:
        """Check drift score against thresholds and generate alerts.

        Only generates an alert if the severity has increased since
        the last unacknowledged alert (avoids spamming the same severity
        repeatedly).
        """
        cfg = self._config
        severity: str | None = None

        if score.overall >= cfg.alert_threshold_critical:
            severity = "critical"
        elif score.overall >= cfg.alert_threshold_high:
            severity = "high"
        elif score.overall >= cfg.alert_threshold_moderate:
            severity = "moderate"

        if severity is None:
            return

        # Check if we already have an unacknowledged alert at this severity or higher
        existing = self._alerts.get(agent_id, [])
        new_rank = _SEVERITY_RANK[severity]

        for alert in reversed(existing):
            if not alert.acknowledged:
                existing_rank = _SEVERITY_RANK.get(alert.severity, 0)
                if existing_rank >= new_rank:
                    return

        alert = DriftAlert(
            agent_id=agent_id,
            severity=severity,
            drift_score=score,
        )
        self._alerts.setdefault(agent_id, []).append(alert)

        # Cap alert history per agent
        if len(self._alerts[agent_id]) > 50:
            self._alerts[agent_id] = self._alerts[agent_id][-50:]
