"""Sliding window — maintains a rolling fingerprint of recent behaviour.

The window keeps the last *N* observations and lazily rebuilds a
:class:`BehavioralFingerprint` from only those observations.  As new
observations arrive the oldest are dropped.  This recent-window
fingerprint is compared against the full baseline to detect drift.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from nomotic.fingerprint import BehavioralFingerprint
from nomotic.types import Action, Verdict

__all__ = [
    "SlidingWindow",
    "WindowObservation",
]


@dataclass(frozen=True)
class WindowObservation:
    """A single observation stored in the sliding window.

    Lightweight — only the fields needed for fingerprint reconstruction.
    """

    action_type: str
    target: str
    timestamp: float
    verdict: str  # Verdict name (string for serialisation)


class SlidingWindow:
    """Maintains a rolling window of recent observations for drift detection.

    Keeps the last *N* observations and maintains a fingerprint built
    from only those observations.  As new observations arrive, the oldest
    are dropped and the fingerprint is recomputed lazily on next access.

    The window size is configurable.  Smaller windows detect drift faster
    but are noisier.  Larger windows are more stable but slower to respond.
    """

    def __init__(
        self,
        agent_id: str,
        window_size: int = 100,
    ) -> None:
        """
        Args:
            agent_id: The agent this window tracks.
            window_size: Number of recent observations to keep.
                Default 100.  Recommended range: 50-500.
        """
        self._agent_id = agent_id
        self._window_size = window_size
        self._observations: deque[WindowObservation] = deque(maxlen=window_size)
        self._fingerprint: BehavioralFingerprint = BehavioralFingerprint(agent_id=agent_id)
        self._dirty: bool = False
        self._lock = threading.Lock()

    def observe(self, action: Action, verdict: Verdict) -> None:
        """Add an observation to the window.

        If the window is full, the oldest observation is dropped.
        The fingerprint is rebuilt lazily on next access.
        """
        with self._lock:
            self._observations.append(WindowObservation(
                action_type=action.action_type,
                target=action.target,
                timestamp=action.timestamp,
                verdict=verdict.name,
            ))
            self._dirty = True

    @property
    def fingerprint(self) -> BehavioralFingerprint:
        """The fingerprint built from the current window contents.

        Rebuilt lazily when the window has been modified since last access.
        """
        with self._lock:
            if self._dirty:
                self._rebuild()
                self._dirty = False
            return self._fingerprint

    @property
    def size(self) -> int:
        """Current number of observations in the window."""
        with self._lock:
            return len(self._observations)

    @property
    def is_full(self) -> bool:
        """Whether the window has reached its maximum size."""
        with self._lock:
            return len(self._observations) >= self._window_size

    def _rebuild(self) -> None:
        """Rebuild the fingerprint from current window contents.

        Called lazily when the fingerprint is accessed after new
        observations have been added.  Creates a fresh
        :class:`BehavioralFingerprint` and replays all observations.
        """
        fp = BehavioralFingerprint(agent_id=self._agent_id)
        for obs in self._observations:
            action = Action(
                action_type=obs.action_type,
                target=obs.target,
                timestamp=obs.timestamp,
                agent_id=self._agent_id,
            )
            fp.observe(action, Verdict[obs.verdict])
        self._fingerprint = fp
