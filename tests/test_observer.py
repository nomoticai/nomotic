"""Tests for the FingerprintObserver."""

from __future__ import annotations

import threading
import time

import pytest

from nomotic.fingerprint import BehavioralFingerprint
from nomotic.observer import FingerprintObserver
from nomotic.priors import PriorRegistry
from nomotic.types import Action, Verdict


def _action(action_type: str = "read", target: str = "/api/data") -> Action:
    return Action(
        agent_id="test-agent",
        action_type=action_type,
        target=target,
        timestamp=time.time(),
    )


class TestFingerprintObserverBasic:
    def test_observe_creates_fingerprint(self):
        observer = FingerprintObserver()
        fp = observer.observe("agent-1", _action("read"), Verdict.ALLOW)
        assert fp is not None
        assert fp.agent_id == "agent-1"
        assert fp.total_observations == 1

    def test_observe_with_archetype_seeds_from_prior(self):
        observer = FingerprintObserver(prior_registry=PriorRegistry.with_defaults())
        fp = observer.observe(
            "agent-1", _action("read"), Verdict.ALLOW,
            archetype="customer-experience",
        )
        assert fp is not None
        # Should have the prior's distributions (blended with 1 observation)
        assert "read" in fp.action_distribution
        # Prior weight of 50 + 1 observation
        assert fp.total_observations == 1

    def test_observe_without_archetype_creates_empty(self):
        observer = FingerprintObserver()
        fp = observer.observe("agent-1", _action("read"), Verdict.ALLOW)
        assert fp._prior_weight == 0

    def test_get_fingerprint_returns_none_for_unknown(self):
        observer = FingerprintObserver()
        assert observer.get_fingerprint("nonexistent") is None

    def test_get_or_create_returns_same_instance(self):
        observer = FingerprintObserver()
        fp1 = observer.get_or_create("agent-1")
        fp2 = observer.get_or_create("agent-1")
        assert fp1 is fp2

    def test_get_or_create_with_archetype(self):
        observer = FingerprintObserver(prior_registry=PriorRegistry.with_defaults())
        fp = observer.get_or_create("agent-1", archetype="customer-experience")
        assert fp._prior_weight == 50
        assert "read" in fp.action_distribution

    def test_reset_removes_fingerprint(self):
        observer = FingerprintObserver()
        observer.observe("agent-1", _action("read"), Verdict.ALLOW)
        assert observer.get_fingerprint("agent-1") is not None

        observer.reset("agent-1")
        assert observer.get_fingerprint("agent-1") is None

    def test_all_fingerprints_returns_copy(self):
        observer = FingerprintObserver()
        observer.observe("agent-1", _action("read"), Verdict.ALLOW)
        observer.observe("agent-2", _action("write"), Verdict.DENY)

        all_fps = observer.all_fingerprints()
        assert len(all_fps) == 2
        assert "agent-1" in all_fps
        assert "agent-2" in all_fps

        # Modifying the copy should not affect the observer
        all_fps.pop("agent-1")
        assert observer.get_fingerprint("agent-1") is not None

    def test_multiple_agents(self):
        observer = FingerprintObserver()
        observer.observe("agent-1", _action("read"), Verdict.ALLOW)
        observer.observe("agent-2", _action("write"), Verdict.DENY)
        observer.observe("agent-1", _action("read"), Verdict.ALLOW)

        fp1 = observer.get_fingerprint("agent-1")
        fp2 = observer.get_fingerprint("agent-2")
        assert fp1 is not None
        assert fp2 is not None
        assert fp1.total_observations == 2
        assert fp2.total_observations == 1

    def test_on_verdict_is_no_op(self):
        """on_verdict is a simplified hook that does nothing by itself."""
        from nomotic.types import GovernanceVerdict
        observer = FingerprintObserver()
        verdict = GovernanceVerdict(action_id="123", verdict=Verdict.ALLOW, ucs=0.9)
        observer.on_verdict(verdict)
        # Should not raise, and should not create any fingerprints
        assert len(observer.all_fingerprints()) == 0


class TestFingerprintObserverThreadSafety:
    def test_concurrent_observe_multiple_agents(self):
        observer = FingerprintObserver()
        errors = []

        def worker(agent_id: str, count: int):
            try:
                for _ in range(count):
                    observer.observe(agent_id, _action("read"), Verdict.ALLOW)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("agent-1", 50)),
            threading.Thread(target=worker, args=("agent-2", 50)),
            threading.Thread(target=worker, args=("agent-3", 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert observer.get_fingerprint("agent-1").total_observations == 50
        assert observer.get_fingerprint("agent-2").total_observations == 50
        assert observer.get_fingerprint("agent-3").total_observations == 50
