"""Integration tests: certificates + governance runtime."""

from nomotic.authority import CertificateAuthority
from nomotic.certificate import CertStatus
from nomotic.keys import SigningKey
from nomotic.runtime import GovernanceRuntime
from nomotic.types import Action, AgentContext, TrustProfile, Verdict


def _ctx(agent_id: str = "agent-1", trust: float = 0.5) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        trust_profile=TrustProfile(agent_id=agent_id, overall_trust=trust),
    )


def _action(agent_id: str = "agent-1", action_type: str = "read", target: str = "db") -> Action:
    return Action(agent_id=agent_id, action_type=action_type, target=target)


class TestRuntimeBirth:
    def test_birth_issues_certificate(self):
        runtime = GovernanceRuntime()
        cert = runtime.birth("agent-1", "customer-experience", "acme", "global/us")
        assert cert.certificate_id.startswith("nmc-")
        assert cert.agent_id == "agent-1"
        assert cert.archetype == "customer-experience"
        assert cert.status == CertStatus.ACTIVE

    def test_birth_baseline_trust(self):
        runtime = GovernanceRuntime()
        cert = runtime.birth("agent-1", "arch", "org", "zone")
        assert cert.trust_score == 0.50
        assert cert.behavioral_age == 0

    def test_get_certificate_after_birth(self):
        runtime = GovernanceRuntime()
        cert = runtime.birth("agent-1", "arch", "org", "zone")
        retrieved = runtime.get_certificate("agent-1")
        assert retrieved is not None
        assert retrieved.certificate_id == cert.certificate_id

    def test_get_certificate_no_birth(self):
        runtime = GovernanceRuntime()
        assert runtime.get_certificate("ghost") is None

    def test_birth_syncs_trust_profile(self):
        runtime = GovernanceRuntime()
        cert = runtime.birth("agent-1", "arch", "org", "zone")
        profile = runtime.get_trust_profile("agent-1")
        assert profile.overall_trust == cert.trust_score


class TestRuntimeEvaluateWithCert:
    def test_evaluate_with_cert_allows(self):
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        cert = runtime.birth("agent-1", "arch", "org", "zone")
        ctx = _ctx()
        verdict = runtime.evaluate_with_cert(_action(), ctx)
        assert verdict.verdict == Verdict.ALLOW

    def test_evaluate_increments_behavioral_age(self):
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        cert = runtime.birth("agent-1", "arch", "org", "zone")
        ctx = _ctx()
        runtime.evaluate_with_cert(_action(), ctx)

        updated = runtime.get_certificate("agent-1")
        assert updated.behavioral_age == 1

    def test_evaluate_updates_trust_on_cert(self):
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        cert = runtime.birth("agent-1", "arch", "org", "zone")
        initial_trust = cert.trust_score
        ctx = _ctx()
        runtime.evaluate_with_cert(_action(), ctx)

        updated = runtime.get_certificate("agent-1")
        # Trust should increase after allow
        assert updated.trust_score > initial_trust

    def test_evaluate_deny_decreases_cert_trust(self):
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"read"})

        cert = runtime.birth("agent-1", "arch", "org", "zone")
        initial_trust = cert.trust_score
        ctx = _ctx()
        runtime.evaluate_with_cert(
            Action(agent_id="agent-1", action_type="delete", target="db"),
            ctx,
        )

        updated = runtime.get_certificate("agent-1")
        assert updated.trust_score < initial_trust

    def test_suspended_cert_denied(self):
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        cert = runtime.birth("agent-1", "arch", "org", "zone")
        ca = runtime._ensure_ca()
        ca.suspend(cert.certificate_id, "test")

        ctx = _ctx()
        verdict = runtime.evaluate_with_cert(_action(), ctx)
        assert verdict.verdict == Verdict.DENY
        assert "SUSPENDED" in verdict.reasoning

    def test_behavioral_age_monotonic(self):
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        cert = runtime.birth("agent-1", "arch", "org", "zone")
        for _ in range(5):
            runtime.evaluate_with_cert(_action(), _ctx())

        updated = runtime.get_certificate("agent-1")
        assert updated.behavioral_age == 5

    def test_explicit_certificate_id(self):
        runtime = GovernanceRuntime()
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        cert = runtime.birth("agent-1", "arch", "org", "zone")
        ctx = _ctx()
        verdict = runtime.evaluate_with_cert(
            _action(), ctx, certificate_id=cert.certificate_id
        )
        assert verdict.verdict == Verdict.ALLOW


class TestExternalCA:
    def test_set_certificate_authority(self):
        runtime = GovernanceRuntime()
        sk, _ = SigningKey.generate()
        ca = CertificateAuthority(issuer_id="external", signing_key=sk)
        runtime.set_certificate_authority(ca)

        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"*"})

        cert = runtime.birth("agent-1", "arch", "org", "zone")
        assert cert.certificate_id.startswith("nmc-")
        # Verify it went through the external CA
        assert ca.get(cert.certificate_id) is not None
