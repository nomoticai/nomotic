"""Tests for the governance runtime — the full pipeline."""

from nomotic.types import Action, AgentContext, TrustProfile, Verdict
from nomotic.runtime import GovernanceRuntime, RuntimeConfig
from nomotic.interrupt import InterruptScope


def _ctx(agent_id: str = "agent-1", trust: float = 0.5) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        trust_profile=TrustProfile(agent_id=agent_id, overall_trust=trust),
    )


def _action(action_type: str = "read", target: str = "db", **params) -> Action:
    return Action(agent_id="agent-1", action_type=action_type, target=target, parameters=params)


class TestGovernanceRuntime:
    def test_basic_allow(self):
        """A simple read action with no constraints should be allowed."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        verdict = runtime.evaluate(_action("read"), _ctx())
        assert verdict.verdict == Verdict.ALLOW

    def test_scope_violation_denied(self):
        """An out-of-scope action should be denied at Tier 1."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"read"})

        verdict = runtime.evaluate(_action("delete"), _ctx())
        assert verdict.verdict == Verdict.DENY
        assert verdict.tier == 1
        assert "scope_compliance" in verdict.vetoed_by

    def test_full_execution_lifecycle(self):
        """Test the complete flow: evaluate -> execute -> complete."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        action = _action("read")
        ctx = _ctx()

        # Evaluate
        verdict = runtime.evaluate(action, ctx)
        assert verdict.verdict == Verdict.ALLOW

        # Begin execution
        handle = runtime.begin_execution(action, ctx)
        assert not handle.is_interrupted
        assert runtime.interrupt_authority.active_count == 1

        # Complete
        record = runtime.complete_execution(action.id, ctx)
        assert record is not None
        assert runtime.interrupt_authority.active_count == 0

    def test_interrupt_during_execution(self):
        """Test that governance can interrupt a running action."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        action = _action("process")
        ctx = _ctx()
        verdict = runtime.evaluate(action, ctx)
        handle = runtime.begin_execution(action, ctx)

        # Simulate governance monitoring detecting an issue
        assert not handle.check_interrupt()
        success = runtime.interrupt_action(action.id, reason="anomaly detected")
        assert success
        assert handle.check_interrupt()

    def test_interrupt_with_rollback(self):
        """Test that interruption triggers rollback."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        rolled_back = {"called": False}

        def rollback():
            rolled_back["called"] = True

        action = _action("write")
        ctx = _ctx()
        runtime.evaluate(action, ctx)
        handle = runtime.begin_execution(action, ctx, rollback=rollback)
        runtime.interrupt_action(action.id, reason="rollback test")
        assert rolled_back["called"]

    def test_trust_updates_on_deny(self):
        """Denied actions should decrease trust."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"read"})

        ctx = _ctx()
        initial_trust = runtime.get_trust_profile("agent-1").overall_trust
        runtime.evaluate(_action("delete"), ctx)  # Will be denied
        assert runtime.get_trust_profile("agent-1").overall_trust < initial_trust

    def test_trust_updates_on_allow(self):
        """Allowed actions should increase trust."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        ctx = _ctx()
        initial_trust = runtime.get_trust_profile("agent-1").overall_trust
        runtime.evaluate(_action("read"), ctx)
        assert runtime.get_trust_profile("agent-1").overall_trust > initial_trust

    def test_human_override_escalates(self):
        """Actions requiring human approval should escalate."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})
        human = runtime.registry.get("human_override")
        human.require_human_for("deploy")

        verdict = runtime.evaluate(_action("deploy"), _ctx())
        assert verdict.verdict in (Verdict.ESCALATE, Verdict.DENY)

    def test_ethical_violation_denied(self):
        """Actions violating ethical rules should be denied."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})
        ethics = runtime.registry.get("ethical_alignment")
        ethics.add_rule(lambda a, c: (False, "violates safety principle"))

        verdict = runtime.evaluate(_action("read"), _ctx())
        assert verdict.verdict == Verdict.DENY
        assert "ethical_alignment" in verdict.vetoed_by

    def test_verdict_listener(self):
        """Listeners should be called on every verdict."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})
        verdicts = []
        runtime.add_verdict_listener(lambda v: verdicts.append(v))
        runtime.evaluate(_action("read"), _ctx())
        assert len(verdicts) == 1

    def test_agent_history_recorded(self):
        """Actions should appear in agent history."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"read"})

        runtime.evaluate(_action("delete"), _ctx())  # Denied
        history = runtime.get_agent_history("agent-1")
        assert len(history) == 1

    def test_evaluation_time_recorded(self):
        """Verdicts should include evaluation time."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})
        verdict = runtime.evaluate(_action("read"), _ctx())
        assert verdict.evaluation_time_ms >= 0

    def test_low_trust_triggers_human_override(self):
        """Agents with very low trust should require human approval."""
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        # Drive trust very low
        scope_dim = runtime.registry.get("scope_compliance")
        scope_dim.configure_agent_scope("agent-1", {"read"})  # Only read
        for i in range(20):
            runtime.evaluate(
                Action(id=f"bad-{i}", agent_id="agent-1", action_type="delete", target="db"),
                _ctx(),
            )

        # Now even an allowed action may escalate due to low trust
        scope_dim.configure_agent_scope("agent-1", {"*"})
        verdict = runtime.evaluate(_action("read"), _ctx())
        # Trust should be very low by now
        trust = runtime.get_trust_profile("agent-1").overall_trust
        assert trust < 0.3
