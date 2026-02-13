"""Fingerprint Observer — bridges governance telemetry to behavioral fingerprints.

The observer sits in the governance runtime and automatically updates agent
fingerprints from governance telemetry.  It is the bridge between the
runtime and the fingerprint system.

Usage::

    runtime = GovernanceRuntime()
    observer = FingerprintObserver(prior_registry=PriorRegistry.with_defaults())
    runtime.add_verdict_listener(observer.on_verdict)

Or more conveniently, the runtime can be configured to auto-attach::

    runtime = GovernanceRuntime(config=RuntimeConfig(enable_fingerprints=True))
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from nomotic.fingerprint import BehavioralFingerprint
from nomotic.priors import PriorRegistry
from nomotic.types import Action, GovernanceVerdict, Verdict

if TYPE_CHECKING:
    from nomotic.drift import DriftScore
    from nomotic.monitor import DriftAlert, DriftConfig, DriftMonitor

__all__ = ["FingerprintObserver"]


class FingerprintObserver:
    """Observes governance evaluations and maintains behavioral fingerprints.

    Attached to the GovernanceRuntime, the observer is called after every
    evaluate() and automatically updates the agent's fingerprint. No
    configuration required — fingerprints form from telemetry.

    When drift detection is enabled (the default), the observer also
    feeds a :class:`DriftMonitor` that tracks sliding-window drift
    and generates alerts.
    """

    def __init__(
        self,
        prior_registry: PriorRegistry | None = None,
        drift_config: DriftConfig | None = None,
    ) -> None:
        self._prior_registry = prior_registry or PriorRegistry.with_defaults()
        self._fingerprints: dict[str, BehavioralFingerprint] = {}
        self._lock = threading.Lock()

        # Drift monitoring (Phase 4B)
        from nomotic.monitor import DriftMonitor as _DriftMonitor
        self._drift_monitor: _DriftMonitor = _DriftMonitor(
            config=drift_config,
            prior_registry=self._prior_registry,
        )

    def get_fingerprint(self, agent_id: str) -> BehavioralFingerprint | None:
        """Get the current fingerprint for an agent."""
        with self._lock:
            return self._fingerprints.get(agent_id)

    def get_or_create(
        self, agent_id: str, archetype: str | None = None,
    ) -> BehavioralFingerprint:
        """Get or create a fingerprint, optionally seeded from an archetype prior."""
        with self._lock:
            if agent_id not in self._fingerprints:
                prior = None
                if archetype:
                    prior = self._prior_registry.get_for_agent(archetype)
                if prior:
                    self._fingerprints[agent_id] = BehavioralFingerprint.from_prior(agent_id, prior)
                else:
                    self._fingerprints[agent_id] = BehavioralFingerprint(agent_id=agent_id)
            return self._fingerprints[agent_id]

    def observe(
        self,
        agent_id: str,
        action: Action,
        verdict: Verdict,
        archetype: str | None = None,
    ) -> BehavioralFingerprint:
        """Record an observation from a governance evaluation.

        Called after every runtime.evaluate(). Updates both the main
        fingerprint and the drift monitor.
        """
        fp = self.get_or_create(agent_id, archetype)
        fp.observe(action, verdict)

        # Feed drift monitor
        self._drift_monitor.observe(
            agent_id=agent_id,
            action=action,
            verdict=verdict,
            baseline_fingerprint=fp,
            archetype=archetype,
        )
        return fp

    def on_verdict(self, verdict_data: GovernanceVerdict) -> None:
        """Verdict listener compatible with runtime.add_verdict_listener().

        Note: This receives the GovernanceVerdict but needs the Action and
        agent_id. The runtime integration handles this by calling observe()
        directly rather than through this listener.
        """
        # Simplified hook — the full integration goes through
        # runtime._record_verdict which has access to both action and context.
        pass

    def all_fingerprints(self) -> dict[str, BehavioralFingerprint]:
        """Get all current fingerprints. Returns a copy."""
        with self._lock:
            return dict(self._fingerprints)

    def reset(self, agent_id: str) -> None:
        """Remove an agent's fingerprint. It will be recreated on next observation."""
        with self._lock:
            self._fingerprints.pop(agent_id, None)
        self._drift_monitor.reset(agent_id)

    # ── Drift accessors (Phase 4B) ──────────────────────────────────

    def get_drift(self, agent_id: str) -> DriftScore | None:
        """Get the latest drift score for an agent."""
        return self._drift_monitor.get_drift(agent_id)

    def get_alerts(
        self,
        agent_id: str | None = None,
        **kwargs: bool,
    ) -> list[DriftAlert]:
        """Get drift alerts."""
        return self._drift_monitor.get_alerts(agent_id, **kwargs)

    @property
    def drift_monitor(self) -> DriftMonitor:
        """Direct access to the drift monitor (for configuration/inspection)."""
        return self._drift_monitor
