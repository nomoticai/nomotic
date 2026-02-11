"""Tests for the interruption rights system."""

from nomotic.types import Action, Severity
from nomotic.interrupt import ExecutionHandle, InterruptAuthority, InterruptScope


def _action(id: str = "a1") -> Action:
    return Action(id=id, agent_id="agent-1", action_type="process")


class TestExecutionHandle:
    def test_initially_not_interrupted(self):
        handle = ExecutionHandle(action=_action(), agent_id="agent-1")
        assert not handle.is_interrupted
        assert not handle.check_interrupt()

    def test_signal_interrupt_sets_flag(self):
        handle = ExecutionHandle(action=_action(), agent_id="agent-1")
        handle._signal_interrupt()
        assert handle.is_interrupted
        assert handle.check_interrupt()


class TestInterruptAuthority:
    def test_register_and_complete(self):
        auth = InterruptAuthority()
        action = _action()
        handle = auth.register_execution(action, "agent-1")
        assert auth.active_count == 1
        auth.complete_execution(action.id)
        assert auth.active_count == 0

    def test_interrupt_action(self):
        auth = InterruptAuthority()
        action = _action()
        handle = auth.register_execution(action, "agent-1")
        records = auth.interrupt(action.id, reason="test")
        assert len(records) == 1
        assert handle.is_interrupted

    def test_interrupt_with_rollback(self):
        rolled_back = {"called": False}

        def rollback():
            rolled_back["called"] = True

        auth = InterruptAuthority()
        action = _action()
        auth.register_execution(action, "agent-1", rollback=rollback)
        records = auth.interrupt(action.id, reason="rollback test")
        assert rolled_back["called"]
        assert records[0].rollback_succeeded

    def test_interrupt_agent_scope(self):
        auth = InterruptAuthority()
        a1 = Action(id="a1", agent_id="agent-1", action_type="x")
        a2 = Action(id="a2", agent_id="agent-1", action_type="y")
        a3 = Action(id="a3", agent_id="agent-2", action_type="z")
        h1 = auth.register_execution(a1, "agent-1")
        h2 = auth.register_execution(a2, "agent-1")
        h3 = auth.register_execution(a3, "agent-2")

        records = auth.interrupt("a1", reason="agent halt", scope=InterruptScope.AGENT)
        assert len(records) == 2
        assert h1.is_interrupted
        assert h2.is_interrupted
        assert not h3.is_interrupted

    def test_interrupt_workflow_scope(self):
        auth = InterruptAuthority()
        a1 = Action(id="a1", agent_id="agent-1", action_type="x")
        a2 = Action(id="a2", agent_id="agent-1", action_type="y")
        a3 = Action(id="a3", agent_id="agent-1", action_type="z")
        auth.register_execution(a1, "agent-1", workflow_id="wf1")
        auth.register_execution(a2, "agent-1", workflow_id="wf1")
        h3 = auth.register_execution(a3, "agent-1", workflow_id="wf2")

        records = auth.interrupt("a1", reason="workflow halt", scope=InterruptScope.WORKFLOW)
        assert len(records) == 2
        assert not h3.is_interrupted

    def test_interrupt_global_scope(self):
        auth = InterruptAuthority()
        for i in range(5):
            auth.register_execution(
                Action(id=f"a{i}", agent_id=f"agent-{i}", action_type="x"),
                f"agent-{i}",
            )
        records = auth.interrupt("a0", reason="emergency", scope=InterruptScope.GLOBAL)
        assert len(records) == 5

    def test_double_interrupt_ignored(self):
        auth = InterruptAuthority()
        action = _action()
        auth.register_execution(action, "agent-1")
        r1 = auth.interrupt(action.id, reason="first")
        r2 = auth.interrupt(action.id, reason="second")
        assert len(r1) == 1
        assert len(r2) == 0  # Already interrupted

    def test_monitor_integration(self):
        auth = InterruptAuthority()
        action = _action()
        handle = auth.register_execution(action, "agent-1")

        from nomotic.types import InterruptRequest

        auth.add_monitor(
            lambda h: InterruptRequest(
                action_id=h.action.id,
                reason="monitor triggered",
                source="test_monitor",
            )
            if h.action.action_type == "process"
            else None
        )
        result = auth.check_monitors(handle)
        assert result is not None
        assert result.reason == "monitor triggered"

    def test_interrupt_history(self):
        auth = InterruptAuthority()
        action = _action()
        auth.register_execution(action, "agent-1")
        auth.interrupt(action.id, reason="test")
        assert len(auth.interrupt_history) == 1
        assert auth.interrupt_history[0].request.reason == "test"
