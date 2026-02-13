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

from nomotic.fingerprint import BehavioralFingerprint
from nomotic.priors import PriorRegistry
from nomotic.types import Action, GovernanceVerdict, Verdict

__all__ = ["FingerprintObserver"]


class FingerprintObserver:
    """Observes governance evaluations and maintains behavioral fingerprints.

    Attached to the GovernanceRuntime, the observer is called after every
    evaluate() and automatically updates the agent's fingerprint. No
    configuration required — fingerprints form from telemetry.
    """

    def __init__(
        self,
        prior_registry: PriorRegistry | None = None,
    ) -> None:
        self._prior_registry = prior_registry or PriorRegistry.with_defaults()
        self._fingerprints: dict[str, BehavioralFingerprint] = {}
        self._lock = threading.Lock()

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

        Called after every runtime.evaluate(). If the agent has no
        fingerprint, one is created (seeded from archetype prior if available).
        """
        fp = self.get_or_create(agent_id, archetype)
        fp.observe(action, verdict)
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
