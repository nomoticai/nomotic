"""Interruption rights — the mechanical authority to stop actions mid-execution.

This is the governance layer nobody builds. Not because it is unimportant.
Because it is hard.

Interruption rights are not policy objections. They are not escalation
procedures. They are the mechanical authority to halt an AI's action while
it is in progress.

The InterruptAuthority operates synchronously with execution. It has:
- Real-time visibility into what the agent is doing
- Technical capability to intervene, not just observe
- Authority to override, not just recommend
- Granular control: halt specific actions, agents, or workflows

Interruption without recovery creates problems it doesn't solve. So the
system manages transactional boundaries and rollback state.
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from nomotic.types import (
    Action,
    ActionState,
    AgentContext,
    InterruptRequest,
    Severity,
)

__all__ = [
    "ExecutionHandle",
    "InterruptAuthority",
    "InterruptRecord",
    "InterruptScope",
]


class InterruptScope(Enum):
    """Granularity of an interrupt."""

    ACTION = auto()     # Stop this specific action
    AGENT = auto()      # Stop all actions by this agent
    WORKFLOW = auto()    # Stop this workflow (action and its children)
    GLOBAL = auto()     # Stop everything (emergency kill switch)


@dataclass
class ExecutionHandle:
    """A handle to a running action that the governance layer can interrupt.

    This is the mechanical link between governance and execution.
    The execution layer creates a handle when it starts an action.
    The governance layer holds the handle and can pull it at any time.
    """

    action: Action
    agent_id: str
    workflow_id: str | None = None
    state: ActionState = ActionState.EXECUTING
    started_at: float = field(default_factory=time.time)
    rollback: Callable[[], None] | None = None
    checkpoint: dict[str, Any] = field(default_factory=dict)
    _interrupted: threading.Event = field(default_factory=threading.Event)

    @property
    def is_interrupted(self) -> bool:
        return self._interrupted.is_set()

    def check_interrupt(self) -> bool:
        """Called by execution code at safe interrupt points.

        The execution layer must cooperate by calling this periodically.
        This is the point where governance authority becomes real.
        """
        return self._interrupted.is_set()

    def _signal_interrupt(self) -> None:
        self._interrupted.set()
        self.state = ActionState.INTERRUPTED


@dataclass
class InterruptRecord:
    """Record of an interrupt that was issued."""

    request: InterruptRequest
    handle: ExecutionHandle
    executed_at: float = field(default_factory=time.time)
    rollback_succeeded: bool | None = None


class InterruptAuthority:
    """The governance layer's authority to stop actions mid-execution.

    This is the core of nomotic enforcement. Without this, governance
    is commentary. With this, governance has teeth.

    Usage:
        authority = InterruptAuthority()

        # Execution layer registers a running action
        handle = authority.register_execution(action, agent_id)

        # During execution, check for interrupts
        if handle.check_interrupt():
            # Governance has interrupted this action — stop gracefully

        # Governance layer can interrupt at any time
        authority.interrupt(action.id, reason="Policy violation detected")

        # When execution completes normally
        authority.complete_execution(action.id)
    """

    def __init__(self) -> None:
        self._active: dict[str, ExecutionHandle] = {}
        self._agent_handles: dict[str, set[str]] = defaultdict(set)
        self._workflow_handles: dict[str, set[str]] = defaultdict(set)
        self._history: list[InterruptRecord] = []
        self._monitors: list[Callable[[ExecutionHandle], InterruptRequest | None]] = []
        self._lock = threading.Lock()

    def register_execution(
        self,
        action: Action,
        agent_id: str,
        workflow_id: str | None = None,
        rollback: Callable[[], None] | None = None,
    ) -> ExecutionHandle:
        """Register a new action execution with the interrupt authority.

        Returns a handle that the execution layer uses to check for interrupts
        and that the governance layer uses to issue interrupts.
        """
        handle = ExecutionHandle(
            action=action,
            agent_id=agent_id,
            workflow_id=workflow_id,
            rollback=rollback,
        )
        with self._lock:
            self._active[action.id] = handle
            self._agent_handles[agent_id].add(action.id)
            if workflow_id:
                self._workflow_handles[workflow_id].add(action.id)
        return handle

    def interrupt(
        self,
        action_id: str,
        reason: str,
        source: str = "governance",
        severity: Severity = Severity.HIGH,
        scope: InterruptScope = InterruptScope.ACTION,
    ) -> list[InterruptRecord]:
        """Interrupt one or more actions.

        The scope determines how broad the interruption is:
        - ACTION: just this action
        - AGENT: all actions by the same agent
        - WORKFLOW: all actions in the same workflow
        - GLOBAL: everything

        Returns records of all interrupts issued.
        """
        request = InterruptRequest(
            action_id=action_id,
            reason=reason,
            source=source,
            severity=severity,
            scope=scope.name.lower(),
        )
        with self._lock:
            targets = self._resolve_targets(action_id, scope)
            records = []
            for target_id in targets:
                handle = self._active.get(target_id)
                if handle and not handle.is_interrupted:
                    handle._signal_interrupt()
                    record = InterruptRecord(request=request, handle=handle)
                    # Attempt rollback if available
                    if handle.rollback:
                        try:
                            handle.rollback()
                            record.rollback_succeeded = True
                        except Exception:
                            record.rollback_succeeded = False
                    records.append(record)
                    self._history.append(record)
            return records

    def complete_execution(self, action_id: str) -> None:
        """Mark an action as completed and remove from active tracking."""
        with self._lock:
            handle = self._active.pop(action_id, None)
            if handle:
                handle.state = ActionState.COMPLETED
                self._agent_handles[handle.agent_id].discard(action_id)
                if handle.workflow_id:
                    self._workflow_handles[handle.workflow_id].discard(action_id)

    def check_monitors(self, handle: ExecutionHandle) -> InterruptRequest | None:
        """Run all registered monitors against an active execution.

        Monitors are continuous governance checks that run during execution.
        If any monitor returns an InterruptRequest, the action is interrupted.
        """
        for monitor in self._monitors:
            request = monitor(handle)
            if request is not None:
                return request
        return None

    def add_monitor(
        self, monitor: Callable[[ExecutionHandle], InterruptRequest | None]
    ) -> None:
        """Register a continuous governance monitor."""
        self._monitors.append(monitor)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def active_handles(self) -> dict[str, ExecutionHandle]:
        return dict(self._active)

    @property
    def interrupt_history(self) -> list[InterruptRecord]:
        return list(self._history)

    def _resolve_targets(
        self, action_id: str, scope: InterruptScope
    ) -> set[str]:
        """Resolve which action IDs should be interrupted based on scope."""
        if scope == InterruptScope.ACTION:
            return {action_id} if action_id in self._active else set()

        if scope == InterruptScope.AGENT:
            handle = self._active.get(action_id)
            if handle:
                return set(self._agent_handles.get(handle.agent_id, set()))
            return set()

        if scope == InterruptScope.WORKFLOW:
            handle = self._active.get(action_id)
            if handle and handle.workflow_id:
                return set(
                    self._workflow_handles.get(handle.workflow_id, set())
                )
            return {action_id} if action_id in self._active else set()

        if scope == InterruptScope.GLOBAL:
            return set(self._active.keys())

        return set()
