"""Tests for the sliding window."""

from __future__ import annotations

import threading
import time
from unittest import TestCase

from nomotic.types import Action, Verdict
from nomotic.window import SlidingWindow


class TestSlidingWindow(TestCase):
    """Tests for SlidingWindow."""

    def _action(self, action_type: str = "read", target: str = "/data") -> Action:
        return Action(
            agent_id="test-agent",
            action_type=action_type,
            target=target,
            timestamp=time.time(),
        )

    def test_empty_window_fingerprint_is_empty(self) -> None:
        w = SlidingWindow(agent_id="a", window_size=10)
        fp = w.fingerprint
        self.assertEqual(fp.total_observations, 0)
        self.assertEqual(fp.action_distribution, {})

    def test_observe_adds_to_window(self) -> None:
        w = SlidingWindow(agent_id="a", window_size=10)
        w.observe(self._action("read"), Verdict.ALLOW)
        self.assertEqual(w.size, 1)
        fp = w.fingerprint
        self.assertEqual(fp.total_observations, 1)
        self.assertIn("read", fp.action_distribution)

    def test_lazy_rebuild(self) -> None:
        """Fingerprint is rebuilt only on access, not on every observe."""
        w = SlidingWindow(agent_id="a", window_size=100)
        # Add several observations
        for _ in range(10):
            w.observe(self._action("read"), Verdict.ALLOW)
        # Access fingerprint once — should rebuild
        fp = w.fingerprint
        self.assertEqual(fp.total_observations, 10)

    def test_window_at_capacity_drops_oldest(self) -> None:
        w = SlidingWindow(agent_id="a", window_size=5)
        for i in range(5):
            w.observe(self._action("read"), Verdict.ALLOW)
        self.assertEqual(w.size, 5)
        self.assertTrue(w.is_full)

        # Add one more — oldest should drop
        w.observe(self._action("write"), Verdict.ALLOW)
        self.assertEqual(w.size, 5)

        # The fingerprint should reflect the new observation
        fp = w.fingerprint
        self.assertEqual(fp.total_observations, 5)
        self.assertIn("write", fp.action_distribution)

    def test_fingerprint_after_drop_reflects_current_contents(self) -> None:
        w = SlidingWindow(agent_id="a", window_size=3)
        # Add 3 reads
        for _ in range(3):
            w.observe(self._action("read"), Verdict.ALLOW)
        fp = w.fingerprint
        self.assertAlmostEqual(fp.action_distribution.get("read", 0), 1.0, places=5)

        # Now add 3 writes, pushing out all reads
        for _ in range(3):
            w.observe(self._action("write"), Verdict.ALLOW)
        fp = w.fingerprint
        self.assertAlmostEqual(fp.action_distribution.get("write", 0), 1.0, places=5)
        self.assertAlmostEqual(fp.action_distribution.get("read", 0), 0.0, places=5)

    def test_is_full(self) -> None:
        w = SlidingWindow(agent_id="a", window_size=3)
        self.assertFalse(w.is_full)
        w.observe(self._action(), Verdict.ALLOW)
        w.observe(self._action(), Verdict.ALLOW)
        self.assertFalse(w.is_full)
        w.observe(self._action(), Verdict.ALLOW)
        self.assertTrue(w.is_full)

    def test_size_tracks_current_count(self) -> None:
        w = SlidingWindow(agent_id="a", window_size=100)
        self.assertEqual(w.size, 0)
        for i in range(5):
            w.observe(self._action(), Verdict.ALLOW)
            self.assertEqual(w.size, i + 1)

    def test_thread_safety_concurrent_observe(self) -> None:
        w = SlidingWindow(agent_id="a", window_size=1000)
        errors: list[Exception] = []

        def worker(n: int) -> None:
            try:
                for _ in range(100):
                    w.observe(self._action(f"type-{n}"), Verdict.ALLOW)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(w.size, 500)
        fp = w.fingerprint
        self.assertEqual(fp.total_observations, 500)
