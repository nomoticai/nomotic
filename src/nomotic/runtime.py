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
import uuid
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
    drift_config: Any = None
    """Optional :class:`DriftConfig`.  If ``None``, uses DriftConfig defaults.
    Passed to :class:`FingerprintObserver`."""
    enable_audit: bool = True
    audit_max_records: int = 10000
    provenance_max_records: int = 5000


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
                drift_config=self.config.drift_config,
            )
        else:
            self._fingerprint_observer = None

        # Wire fingerprint and drift accessors into dimensions
        if self._fingerprint_observer is not None:
            behavioral = self.registry.get("behavioral_consistency")
            if behavioral is not None:
                behavioral.set_fingerprint_accessor(self._fingerprint_observer.get_fingerprint)
                behavioral.set_drift_accessor(self._fingerprint_observer.get_drift)
            incident = self.registry.get("incident_detection")
            if incident is not None:
                incident.set_drift_accessor(self._fingerprint_observer.get_drift)

        # Audit trail (Phase 5)
        if self.config.enable_audit:
            from nomotic.audit import AuditTrail
            from nomotic.provenance import ProvenanceLog
            from nomotic.accountability import OwnerActivityLog, UserActivityTracker
            self._audit_trail: AuditTrail | None = AuditTrail(
                max_records=self.config.audit_max_records,
            )
            self._provenance_log: ProvenanceLog | None = ProvenanceLog(
                max_records=self.config.provenance_max_records,
            )
            self._owner_activity: OwnerActivityLog | None = OwnerActivityLog()
            self._user_tracker: UserActivityTracker | None = UserActivityTracker()
        else:
            self._audit_trail = None
            self._provenance_log = None
            self._owner_activity = None
            self._user_tracker = None

        # Context profiles (Phase 7A)
        from nomotic.context_profile import ContextProfileManager
        self.context_profiles = ContextProfileManager()

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

    def get_drift(self, agent_id: str) -> Any:
        """Get the latest behavioral drift score for an agent.

        Returns None if fingerprints/drift are disabled or if drift
        has never been computed for this agent.
        """
        if self._fingerprint_observer is None:
            return None
        return self._fingerprint_observer.get_drift(agent_id)

    def get_drift_alerts(
        self, agent_id: str | None = None, **kwargs: bool,
    ) -> list[Any]:
        """Get drift alerts, optionally filtered by agent."""
        if self._fingerprint_observer is None:
            return []
        return self._fingerprint_observer.get_alerts(agent_id, **kwargs)

    def get_trust_trajectory(self, agent_id: str) -> Any:
        """Get the trust trajectory for an agent."""
        return self.trust_calibrator.get_trajectory(agent_id)

    def get_trust_report(self, agent_id: str) -> dict[str, Any]:
        """Get a comprehensive trust report for an agent.

        Combines current trust, trajectory summary, fingerprint status,
        and drift status into a single report.
        """
        profile = self.get_trust_profile(agent_id)
        trajectory = self.trust_calibrator.get_trajectory(agent_id)

        report: dict[str, Any] = {
            "agent_id": agent_id,
            "current_trust": profile.overall_trust,
            "successful_actions": profile.successful_actions,
            "violation_count": profile.violation_count,
            "violation_rate": profile.violation_rate,
            "trajectory": trajectory.summary(),
        }

        if self._fingerprint_observer is not None:
            fp = self._fingerprint_observer.get_fingerprint(agent_id)
            if fp is not None:
                report["fingerprint"] = {
                    "total_observations": fp.total_observations,
                    "confidence": fp.confidence,
                }

            drift = self._fingerprint_observer.get_drift(agent_id)
            if drift is not None:
                report["drift"] = drift.to_dict()

            alerts = self._fingerprint_observer.get_alerts(agent_id)
            if alerts:
                report["active_alerts"] = len(
                    [a for a in alerts if not a.acknowledged]
                )

        return report

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

            # Drift-based trust adjustment (Phase 4C)
            drift = self._fingerprint_observer.get_drift(context.agent_id)
            if drift is not None:
                self.trust_calibrator.apply_drift(context.agent_id, drift)

                # Sync trust back to certificate
                if cert is not None:
                    new_trust = self.trust_calibrator.get_profile(
                        context.agent_id
                    ).overall_trust
                    ca = self._ensure_ca()
                    ca.update_trust(cert.certificate_id, new_trust)

        # Audit trail (Phase 5)
        if self._audit_trail is not None:
            self._record_audit(action, context, verdict)

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

    # ── Audit trail integration (Phase 5) ────────────────────────────

    def _record_audit(
        self, action: Action, context: AgentContext, verdict: GovernanceVerdict,
    ) -> None:
        """Create an audit record for a governance decision."""
        from nomotic.audit import AuditRecord, build_justification
        from nomotic.context import CODES

        # Determine context code
        if verdict.verdict == Verdict.ALLOW:
            code = CODES.GOVERNANCE_ALLOW
        elif verdict.verdict == Verdict.DENY:
            if verdict.vetoed_by:
                code = CODES.GOVERNANCE_VETO
            else:
                code = CODES.GOVERNANCE_DENY
        elif verdict.verdict == Verdict.ESCALATE:
            code = CODES.GOVERNANCE_ESCALATE
        elif verdict.verdict == Verdict.MODIFY:
            code = CODES.GOVERNANCE_MODIFY
        else:
            code = CODES.GOVERNANCE_ALLOW

        # Resolve owner from certificate
        owner_id = ""
        cert = self.get_certificate(action.agent_id)
        if cert:
            owner_id = cert.owner

        # Resolve user from context
        user_id = ""
        if context.user_context is not None:
            user_id = context.user_context.user_id

        # Get trust state
        profile = self.trust_calibrator.get_profile(context.agent_id)
        trajectory = self.trust_calibrator.get_trajectory(context.agent_id)

        # Get drift state
        drift = self.get_drift(context.agent_id)

        # Build justification
        justification = build_justification(verdict, action, context, drift)

        record = AuditRecord(
            record_id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            context_code=code.code,
            severity=code.severity,
            agent_id=context.agent_id,
            owner_id=owner_id,
            user_id=user_id,
            action_id=action.id,
            action_type=action.action_type,
            action_target=action.target,
            verdict=verdict.verdict.name,
            ucs=verdict.ucs,
            tier=verdict.tier,
            dimension_scores=[
                {
                    "name": s.dimension_name,
                    "score": round(s.score, 4),
                    "weight": s.weight,
                    "veto": s.veto,
                    "reasoning": s.reasoning,
                }
                for s in verdict.dimension_scores
            ],
            trust_score=profile.overall_trust,
            trust_trend=trajectory.trend,
            drift_overall=drift.overall if drift else None,
            drift_severity=drift.severity if drift else None,
            justification=justification,
            metadata={
                "session_id": context.session_id,
                "config_version": (
                    self._provenance_log.current_config_version()
                    if self._provenance_log else ""
                ),
                "user_request_hash": (
                    context.user_context.request_hash
                    if context.user_context else ""
                ),
            },
        )
        assert self._audit_trail is not None
        self._audit_trail.append(record)

        # Track user activity
        if (
            context.user_context is not None
            and context.user_context.user_id
            and self._user_tracker is not None
        ):
            self._user_tracker.record_interaction(
                user_id=context.user_context.user_id,
                agent_id=context.agent_id,
                verdict=verdict.verdict.name,
                context_code=code.code,
            )

    def _record_provenance(
        self,
        actor: str,
        target_type: str,
        target_id: str,
        change_type: str,
        **kwargs: Any,
    ) -> None:
        """Record a configuration change in the provenance log."""
        if self._provenance_log is not None:
            self._provenance_log.record(
                actor=actor,
                target_type=target_type,
                target_id=target_id,
                change_type=change_type,
                **kwargs,
            )

    # ── Provenance-tracked configuration wrappers ─────────────────

    def configure_scope(
        self,
        agent_id: str,
        scope: set[str],
        *,
        actor: str = "system",
        reason: str = "",
        ticket: str = "",
    ) -> None:
        """Configure agent scope with provenance tracking."""
        dim = self.registry.get("scope_compliance")
        previous = dim._allowed_scopes.get(agent_id)
        dim.configure_agent_scope(agent_id, scope)
        self._record_provenance(
            actor=actor,
            target_type="scope",
            target_id=agent_id,
            change_type="modify" if previous else "add",
            previous_value=sorted(previous) if previous else None,
            new_value=sorted(scope),
            reason=reason,
            ticket=ticket,
            context_code="CONFIG.SCOPE_CHANGED",
        )

    def configure_boundaries(
        self,
        agent_id: str,
        allowed_targets: set[str],
        *,
        actor: str = "system",
        reason: str = "",
        ticket: str = "",
    ) -> None:
        """Configure isolation boundaries with provenance tracking."""
        dim = self.registry.get("isolation_integrity")
        previous = dim._boundaries.get(agent_id)
        dim.set_boundaries(agent_id, allowed_targets)
        self._record_provenance(
            actor=actor,
            target_type="boundary",
            target_id=agent_id,
            change_type="modify" if previous else "add",
            previous_value=sorted(previous) if previous else None,
            new_value=sorted(allowed_targets),
            reason=reason,
            ticket=ticket,
            context_code="CONFIG.BOUNDARY_CHANGED",
        )

    def configure_time_window(
        self,
        action_type: str,
        start_hour: int,
        end_hour: int,
        *,
        actor: str = "system",
        reason: str = "",
        ticket: str = "",
    ) -> None:
        """Configure temporal compliance window with provenance tracking."""
        dim = self.registry.get("temporal_compliance")
        previous = dim._time_windows.get(action_type)
        dim.set_time_window(action_type, start_hour, end_hour)
        self._record_provenance(
            actor=actor,
            target_type="time_window",
            target_id=action_type,
            change_type="modify" if previous else "add",
            previous_value=list(previous) if previous else None,
            new_value=[start_hour, end_hour],
            reason=reason,
            ticket=ticket,
            context_code="CONFIG.THRESHOLD_CHANGED",
        )

    def configure_human_override(
        self,
        *action_types: str,
        actor: str = "system",
        reason: str = "",
        ticket: str = "",
    ) -> None:
        """Configure human override requirements with provenance tracking."""
        dim = self.registry.get("human_override")
        previous = set(dim._require_human)
        dim.require_human_for(*action_types)
        self._record_provenance(
            actor=actor,
            target_type="override",
            target_id=",".join(action_types),
            change_type="add",
            previous_value=sorted(previous) if previous else None,
            new_value=sorted(dim._require_human),
            reason=reason,
            ticket=ticket,
            context_code="CONFIG.THRESHOLD_CHANGED",
        )

    def add_ethical_rule(
        self,
        rule: Callable,
        *,
        actor: str = "system",
        reason: str = "",
        ticket: str = "",
        rule_name: str = "",
    ) -> None:
        """Add an ethical rule with provenance tracking."""
        dim = self.registry.get("ethical_alignment")
        dim.add_rule(rule)
        self._record_provenance(
            actor=actor,
            target_type="rule",
            target_id=rule_name or "ethical_rule",
            change_type="add",
            new_value=rule_name or repr(rule),
            reason=reason,
            ticket=ticket,
            context_code="CONFIG.RULE_ADDED",
        )

    def set_dimension_weight(
        self,
        dimension_name: str,
        weight: float,
        *,
        actor: str = "system",
        reason: str = "",
        ticket: str = "",
    ) -> None:
        """Set a dimension weight with provenance tracking."""
        dim = self.registry.get(dimension_name)
        if dim is None:
            raise ValueError(f"Unknown dimension: {dimension_name}")
        previous = dim.weight
        dim.weight = weight
        self._record_provenance(
            actor=actor,
            target_type="weight",
            target_id=dimension_name,
            change_type="modify",
            previous_value=previous,
            new_value=weight,
            reason=reason,
            ticket=ticket,
            context_code="CONFIG.WEIGHT_CHANGED",
        )

    # ── Phase 5 property accessors ────────────────────────────────

    @property
    def audit_trail(self) -> Any:
        """The audit trail, or None if auditing is disabled."""
        return self._audit_trail

    @property
    def provenance_log(self) -> Any:
        """The provenance log, or None if auditing is disabled."""
        return self._provenance_log

    @property
    def owner_activity(self) -> Any:
        """The owner activity log, or None if auditing is disabled."""
        return self._owner_activity

    @property
    def user_tracker(self) -> Any:
        """The user activity tracker, or None if auditing is disabled."""
        return self._user_tracker

    # ── Context Profile convenience methods (Phase 7A) ────────────

    def get_context_profile(self, profile_id: str) -> Any:
        """Retrieve a context profile by ID."""
        return self.context_profiles.get_profile(profile_id)

    def create_context_profile(self, agent_id: str, **kwargs: Any) -> Any:
        """Create a new context profile for an agent."""
        from nomotic.context_profile import ContextProfile
        return self.context_profiles.create_profile(agent_id, **kwargs)

    def _append_history(self, agent_id: str, record: ActionRecord) -> None:
        history = self._action_history.setdefault(agent_id, [])
        history.append(record)
        if len(history) > self.config.max_history_per_agent:
            self._action_history[agent_id] = history[-self.config.max_history_per_agent :]
