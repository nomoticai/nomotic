"""Governance Runtime — the orchestrator.

This is where everything comes together. The runtime is the single entry
point for all governance. Every action passes through it. Every governance
decision flows from it.

The pipeline for every action:
1. Receive action + agent context
2. Apply time decay to trust profile
3. Evaluate all 13 dimensions simultaneously
4. Pass through Tier 1 (deterministic gate — vetoes checked)
5. If Tier 1 doesn't decide, compute UCS and pass through Tier 2
6. If Tier 2 doesn't decide, pass through Tier 3 (deliberation)
7. Record the verdict in trust calibration
8. If ALLOW, register execution with interrupt authority
9. During execution, governance monitors run continuously
10. On completion, update trust based on outcome

Governance is not something that happens before execution. It is something
that happens throughout execution. The runtime ensures this.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from nomotic.types import (
    Action,
    ActionRecord,
    ActionState,
    AgentContext,
    GovernanceVerdict,
    InterruptRequest,
    Severity,
    TrustProfile,
    Verdict,
)
from nomotic.dimensions import DimensionRegistry
from nomotic.ucs import UCSEngine
from nomotic.tiers import TierOneGate, TierTwoEvaluator, TierThreeDeliberator
from nomotic.interrupt import ExecutionHandle, InterruptAuthority, InterruptScope
from nomotic.trust import TrustCalibrator, TrustConfig
from nomotic.certificate import AgentCertificate, CertStatus
from nomotic.keys import SigningKey
from nomotic.authority import CertificateAuthority
from nomotic.store import MemoryCertificateStore
from nomotic.registry import ArchetypeRegistry, OrganizationRegistry, ZoneValidator

__all__ = ["GovernanceRuntime", "RuntimeConfig"]


@dataclass
class RuntimeConfig:
    """Configuration for the governance runtime."""

    allow_threshold: float = 0.7
    deny_threshold: float = 0.3
    trust_influence: float = 0.2
    trust_config: TrustConfig = field(default_factory=TrustConfig)
    max_history_per_agent: int = 1000
    enable_fingerprints: bool = True


class GovernanceRuntime:
    """The complete nomotic governance runtime.

    This is the system. Every action goes through evaluate(). Every
    execution is monitored through the interrupt authority. Trust is
    calibrated continuously. Governance is not advisory — it is
    authoritative.

    Usage:
        runtime = GovernanceRuntime()

        # Configure dimensions
        scope = runtime.registry.get("scope_compliance")
        scope.configure_agent_scope("agent-1", {"read", "write"})

        # Evaluate an action
        verdict = runtime.evaluate(action, context)

        if verdict.verdict == Verdict.ALLOW:
            # Execute with governance oversight
            handle = runtime.begin_execution(action, context)

            # Execution code checks for interrupts
            for step in workflow:
                if handle.check_interrupt():
                    break
                do_work(step)

            runtime.complete_execution(action.id, context)
    """

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig()
        if self.config.allow_threshold <= self.config.deny_threshold:
            raise ValueError(
                f"allow_threshold must be greater than deny_threshold, "
                f"got allow={self.config.allow_threshold}, deny={self.config.deny_threshold}"
            )
        self.registry = DimensionRegistry.create_default()
        self.ucs_engine = UCSEngine(trust_influence=self.config.trust_influence)
        self.tier_one = TierOneGate()
        self.tier_two = TierTwoEvaluator(
            allow_threshold=self.config.allow_threshold,
            deny_threshold=self.config.deny_threshold,
        )
        self.tier_three = TierThreeDeliberator()
        self.interrupt_authority = InterruptAuthority()
        self.trust_calibrator = TrustCalibrator(config=self.config.trust_config)
        self._action_history: dict[str, list[ActionRecord]] = {}
        self._verdicts: dict[str, GovernanceVerdict] = {}
        self._listeners: list[Callable[[GovernanceVerdict], None]] = []

        # Behavioral fingerprint observer (opt-in, default: enabled)
        if self.config.enable_fingerprints:
            from nomotic.fingerprint import BehavioralFingerprint
            from nomotic.observer import FingerprintObserver
            from nomotic.priors import PriorRegistry
            self._fingerprint_observer: FingerprintObserver | None = FingerprintObserver(
                prior_registry=PriorRegistry.with_defaults(),
            )
        else:
            self._fingerprint_observer = None

        # Wire fingerprint accessor into BehavioralConsistency dimension
        if self._fingerprint_observer is not None:
            behavioral = self.registry.get("behavioral_consistency")
            if behavioral is not None:
                behavioral.set_fingerprint_accessor(self._fingerprint_observer.get_fingerprint)

        # Certificate authority — initialized lazily or explicitly
        self._ca: CertificateAuthority | None = None
        self._cert_map: dict[str, str] = {}  # agent_id -> certificate_id

        # Registries — initialized lazily
        self._archetype_registry: ArchetypeRegistry | None = None
        self._zone_validator: ZoneValidator | None = None
        self._org_registry: OrganizationRegistry | None = None

    def evaluate(self, action: Action, context: AgentContext) -> GovernanceVerdict:
        """Evaluate an action through the full governance pipeline.

        This is the primary entry point. Returns a GovernanceVerdict
        that tells the caller what to do.
        """
        start = time.time()

        # Step 1: Apply time decay to trust
        self.trust_calibrator.apply_time_decay(context.agent_id)
        context.trust_profile = self.trust_calibrator.get_profile(context.agent_id)

        # Step 2: Evaluate all 13 dimensions simultaneously
        scores = self.registry.evaluate_all(action, context)

        # Step 3: Tier 1 — deterministic gate
        tier1_result = self.tier_one.evaluate(action, context, scores)
        if tier1_result.decided:
            verdict = tier1_result.verdict
            assert verdict is not None
            verdict.evaluation_time_ms = (time.time() - start) * 1000
            self._record_verdict(action, context, verdict)
            return verdict

        # Step 4: Compute UCS for Tier 2
        ucs = self.ucs_engine.compute(scores, context.trust_profile)

        # Step 5: Tier 2 — weighted evaluation
        tier2_result = self.tier_two.evaluate(action, context, scores, ucs)
        if tier2_result.decided:
            verdict = tier2_result.verdict
            assert verdict is not None
            verdict.evaluation_time_ms = (time.time() - start) * 1000
            self._record_verdict(action, context, verdict)
            return verdict

        # Step 6: Tier 3 — deliberative review
        tier3_result = self.tier_three.evaluate(action, context, scores, ucs)
        verdict = tier3_result.verdict
        assert verdict is not None
        verdict.evaluation_time_ms = (time.time() - start) * 1000
        self._record_verdict(action, context, verdict)
        return verdict

    def begin_execution(
        self,
        action: Action,
        context: AgentContext,
        rollback: Callable[[], None] | None = None,
        workflow_id: str | None = None,
    ) -> ExecutionHandle:
        """Register an approved action for execution with governance oversight.

        Returns an ExecutionHandle that the execution layer uses to check
        for interrupts. The governance layer can interrupt through the
        interrupt_authority at any time.
        """
        return self.interrupt_authority.register_execution(
            action=action,
            agent_id=context.agent_id,
            workflow_id=workflow_id,
            rollback=rollback,
        )

    def complete_execution(
        self,
        action_id: str,
        context: AgentContext,
        outcome: dict[str, Any] | None = None,
    ) -> ActionRecord | None:
        """Record successful completion of an action.

        Updates trust calibration and action history.
        """
        verdict = self._verdicts.get(action_id)
        if not verdict:
            return None

        self.interrupt_authority.complete_execution(action_id)

        record = ActionRecord(
            action=Action(id=action_id, agent_id=context.agent_id),
            verdict=verdict,
            state=ActionState.COMPLETED,
            outcome=outcome or {},
        )

        self.trust_calibrator.record_completion(context.agent_id, record)
        self._append_history(context.agent_id, record)
        return record

    def interrupt_action(
        self,
        action_id: str,
        reason: str,
        source: str = "governance",
        scope: InterruptScope = InterruptScope.ACTION,
    ) -> bool:
        """Interrupt a running action.

        This is governance with teeth. Returns True if the interrupt
        was issued, False if the action wasn't found.
        """
        records = self.interrupt_authority.interrupt(
            action_id=action_id,
            reason=reason,
            source=source,
            scope=scope,
        )
        # Update trust for interrupted agent(s)
        for record in records:
            agent_id = record.handle.agent_id
            action_record = ActionRecord(
                action=record.handle.action,
                verdict=self._verdicts.get(record.handle.action.id, GovernanceVerdict(
                    action_id=record.handle.action.id,
                    verdict=Verdict.SUSPEND,
                    ucs=0.0,
                )),
                state=ActionState.INTERRUPTED,
                interrupted=True,
                interrupt_reason=reason,
            )
            self.trust_calibrator.record_completion(agent_id, action_record)
            self._append_history(agent_id, action_record)
        return len(records) > 0

    def add_verdict_listener(
        self, listener: Callable[[GovernanceVerdict], None]
    ) -> None:
        """Register a listener called after every governance verdict."""
        self._listeners.append(listener)

    def get_agent_history(self, agent_id: str) -> list[ActionRecord]:
        return list(self._action_history.get(agent_id, []))

    def get_trust_profile(self, agent_id: str) -> TrustProfile:
        return self.trust_calibrator.get_profile(agent_id)

    # ── Registry accessors ────────────────────────────────────────────

    @property
    def archetype_registry(self) -> ArchetypeRegistry:
        """Lazily initialized archetype registry with defaults."""
        if self._archetype_registry is None:
            self._archetype_registry = ArchetypeRegistry.with_defaults()
        return self._archetype_registry

    @property
    def zone_validator(self) -> ZoneValidator:
        """Lazily initialized zone validator."""
        if self._zone_validator is None:
            self._zone_validator = ZoneValidator()
        return self._zone_validator

    @property
    def org_registry(self) -> OrganizationRegistry:
        """Lazily initialized organization registry."""
        if self._org_registry is None:
            self._org_registry = OrganizationRegistry()
        return self._org_registry

    # ── Certificate integration ──────────────────────────────────────

    def _ensure_ca(self) -> CertificateAuthority:
        """Lazily initialize the certificate authority.

        The auto-created CA does not attach registries — it is a bare
        authority for programmatic use.  Attach registries explicitly via
        :meth:`set_certificate_authority` or by passing a pre-configured
        CA.
        """
        if self._ca is None:
            sk, _vk = SigningKey.generate()
            self._ca = CertificateAuthority(
                issuer_id="runtime-auto",
                signing_key=sk,
                store=MemoryCertificateStore(),
            )
        return self._ca

    def set_certificate_authority(self, ca: CertificateAuthority) -> None:
        """Attach an external certificate authority to this runtime."""
        self._ca = ca

    def birth(
        self,
        agent_id: str,
        archetype: str,
        organization: str,
        zone_path: str,
        *,
        owner: str = "",
        **opts: Any,
    ) -> AgentCertificate:
        """Issue a birth certificate for an agent through the runtime.

        Delegates to CertificateAuthority.issue() and maps the agent_id
        to the new certificate.
        """
        ca = self._ensure_ca()
        cert, _agent_sk = ca.issue(
            agent_id=agent_id,
            archetype=archetype,
            organization=organization,
            zone_path=zone_path,
            owner=owner,
            **opts,
        )
        self._cert_map[agent_id] = cert.certificate_id
        # Sync trust: seed the trust calibrator with baseline
        profile = self.trust_calibrator.get_profile(agent_id)
        profile.overall_trust = cert.trust_score
        return cert

    def get_certificate(self, agent_id: str) -> AgentCertificate | None:
        """Get the current certificate for an agent."""
        ca = self._ensure_ca()
        cert_id = self._cert_map.get(agent_id)
        if cert_id is None:
            return None
        return ca.get(cert_id)

    def evaluate_with_cert(
        self,
        action: Action,
        context: AgentContext,
        certificate_id: str | None = None,
    ) -> GovernanceVerdict:
        """Evaluate an action, integrating with the certificate system.

        If a certificate_id is provided (or the agent has one mapped),
        the certificate's trust score is used, its status is checked,
        and its behavioral_age and trust_score are updated afterward.
        """
        ca = self._ensure_ca()

        # Resolve certificate
        cid = certificate_id or self._cert_map.get(context.agent_id)
        cert: AgentCertificate | None = None
        if cid:
            cert = ca.get(cid)

        if cert is not None:
            # Verify certificate is ACTIVE
            if cert.status != CertStatus.ACTIVE:
                return GovernanceVerdict(
                    action_id=action.id,
                    verdict=Verdict.DENY,
                    ucs=0.0,
                    reasoning=f"certificate status is {cert.status.name}",
                    tier=1,
                )

            # Use certificate's trust as the authoritative source
            context.trust_profile.overall_trust = cert.trust_score

        # Run the normal evaluation pipeline
        verdict = self.evaluate(action, context)

        # Update certificate post-evaluation (single store write)
        if cert is not None:
            new_trust = self.trust_calibrator.get_profile(context.agent_id).overall_trust
            ca.record_action(cert.certificate_id, new_trust)

        return verdict

    def get_fingerprint(self, agent_id: str) -> Any:
        """Get the behavioral fingerprint for an agent.

        Returns None if fingerprints are disabled or if no fingerprint
        exists for the agent.
        """
        if self._fingerprint_observer is None:
            return None
        return self._fingerprint_observer.get_fingerprint(agent_id)

    def _record_verdict(
        self, action: Action, context: AgentContext, verdict: GovernanceVerdict
    ) -> None:
        """Record a verdict and update trust."""
        self._verdicts[action.id] = verdict
        self.trust_calibrator.record_verdict(context.agent_id, verdict)

        # Update behavioral fingerprint
        if self._fingerprint_observer is not None:
            archetype = None
            cert = self.get_certificate(action.agent_id)
            if cert:
                archetype = cert.archetype
            self._fingerprint_observer.observe(
                agent_id=context.agent_id,
                action=action,
                verdict=verdict.verdict,
                archetype=archetype,
            )

        # Update context history for future evaluations
        if verdict.verdict == Verdict.DENY:
            record = ActionRecord(
                action=action,
                verdict=verdict,
                state=ActionState.DENIED,
            )
            self._append_history(context.agent_id, record)
            context.action_history.append(record)

        for listener in self._listeners:
            listener(verdict)

    def _append_history(self, agent_id: str, record: ActionRecord) -> None:
        history = self._action_history.setdefault(agent_id, [])
        history.append(record)
        if len(history) > self.config.max_history_per_agent:
            self._action_history[agent_id] = history[-self.config.max_history_per_agent :]
