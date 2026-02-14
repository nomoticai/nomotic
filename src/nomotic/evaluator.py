"""Protocol Evaluator — bridges Reasoning Artifacts to the governance runtime.

The evaluator receives Reasoning Artifacts, performs structural evaluation,
delegates dimensional evaluation to the existing GovernanceRuntime, produces
Governance Responses, and issues Governance Tokens when approved.

Supports all three protocol flows:
    - Full Deliberation: thorough evaluation before action
    - Summary: streamlined evaluation for routine actions
    - Post-Hoc: retroactive evaluation after action

The evaluator performs two layers of evaluation:

1. **Structural evaluation** (required) — schema conformance, completeness,
   internal consistency, authority verification.

2. **Dimensional evaluation** (optional, requires runtime) — delegates to
   the runtime's 13 governance dimensions, computes UCS, applies trust.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nomotic.protocol import (
    METHODS,
    PROTOCOL_VERSION,
    Assessment,
    Condition,
    Denial,
    Escalation,
    GovernanceResponseData,
    Guidance,
    ProtocolVerdict,
    ReasoningArtifact,
    ResponseMetadata,
    method_category,
    validate_artifact,
)
from nomotic.token import GovernanceToken, TokenValidator
from nomotic.types import Action, AgentContext, TrustProfile, Verdict

__all__ = [
    "EthicalReasoningAssessment",
    "EthicalReasoningConfig",
    "EvaluatorConfig",
    "PostHocAssessment",
    "ProtocolEvaluator",
]


# ── Ethical Reasoning types ────────────────────────────────────────────


@dataclass
class EthicalReasoningConfig:
    """Organization-defined criteria for ethical reasoning evaluation.

    The organization defines which methods require ethical reasoning,
    what the minimum score is, and how the components are weighted.
    Nomotic evaluates against those criteria.
    """

    require_stakeholder_factors: bool = True
    require_harm_acknowledgment_for_methods: list[str] = field(
        default_factory=lambda: [
            "approve", "deny", "classify", "prioritize", "rank",
            "transfer", "delete", "revoke",
        ]
    )
    fairness_relevant_methods: list[str] = field(
        default_factory=lambda: [
            "approve", "deny", "classify", "prioritize", "rank", "recommend",
        ]
    )
    minimum_ethical_reasoning_score: float = 0.5
    stakeholder_weight: float = 0.25
    harm_weight: float = 0.25
    fairness_weight: float = 0.25
    alternative_weight: float = 0.15
    uncertainty_weight: float = 0.10


@dataclass
class EthicalReasoningAssessment:
    """Assessment of the ethical quality of an agent's reasoning.

    All evaluation is structural and statistical, not semantic.
    No LLM-in-the-loop for ethical judgment.
    """

    stakeholder_consideration: float = 0.0  # 0.0-1.0
    harm_awareness: float = 0.0  # 0.0-1.0
    fairness_consideration: float = 0.0  # 0.0-1.0
    alternative_equity: float = 0.0  # 0.0-1.0
    uncertainty_honesty: float = 0.0  # 0.0-1.0
    overall_ethical_reasoning_score: float = 0.0
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stakeholder_consideration": round(self.stakeholder_consideration, 4),
            "harm_awareness": round(self.harm_awareness, 4),
            "fairness_consideration": round(self.fairness_consideration, 4),
            "alternative_equity": round(self.alternative_equity, 4),
            "uncertainty_honesty": round(self.uncertainty_honesty, 4),
            "overall_ethical_reasoning_score": round(self.overall_ethical_reasoning_score, 4),
            "findings": self.findings,
            "recommendations": self.recommendations,
        }


@dataclass
class EvaluatorConfig:
    """Configuration for the protocol evaluator."""

    evaluator_id: str = "nomotic-evaluator"
    # Structural evaluation thresholds
    min_completeness_for_proceed: float = 0.6
    min_alignment_for_proceed: float = 0.5
    # Whether to require dimensional evaluation (requires runtime)
    require_dimensional: bool = False
    # UCS threshold for approval
    proceed_threshold: float = 0.7
    deny_threshold: float = 0.3
    # Token secret (auto-generated if not provided)
    token_secret: bytes = field(default_factory=lambda: os.urandom(32))
    # Supported schema versions
    supported_versions: list[str] = field(default_factory=lambda: ["0.1.0"])


@dataclass
class PostHocAssessment:
    """Assessment from a post-hoc evaluation."""

    sound_reasoning: bool
    would_have_approved: bool
    trust_adjustment: float  # positive = trust increase, negative = decrease
    concern: bool  # if True, future actions should use Full Deliberation
    detail: str = ""
    assessment: Assessment | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "sound_reasoning": self.sound_reasoning,
            "would_have_approved": self.would_have_approved,
            "trust_adjustment": self.trust_adjustment,
            "concern": self.concern,
        }
        if self.detail:
            d["detail"] = self.detail
        if self.assessment is not None:
            d["assessment"] = self.assessment.to_dict()
        return d


class ProtocolEvaluator:
    """Evaluates Reasoning Artifacts and produces Governance Responses.

    Usage::

        evaluator = ProtocolEvaluator(runtime=my_runtime)
        response = evaluator.evaluate(artifact)

        if response.verdict == ProtocolVerdict.PROCEED:
            # Token is in response.token
            execute_with_token(response.token)
    """

    def __init__(
        self,
        config: EvaluatorConfig | None = None,
        runtime: Any = None,  # GovernanceRuntime, imported lazily
        ethical_reasoning_config: EthicalReasoningConfig | None = None,
    ) -> None:
        self.config = config or EvaluatorConfig()
        self._runtime = runtime
        self._ethical_reasoning_config = ethical_reasoning_config
        self._token_mgr = GovernanceToken(
            secret=self.config.token_secret,
            evaluator_id=self.config.evaluator_id,
        )
        self._validator = TokenValidator(secret=self.config.token_secret)
        self._config_version = hashlib.sha256(
            f"{self.config.evaluator_id}:{PROTOCOL_VERSION}".encode()
        ).hexdigest()[:12]
        # Store artifacts by hash for introspection
        self._artifact_store: dict[str, ReasoningArtifact] = {}
        self._response_store: dict[str, GovernanceResponseData] = {}

    @property
    def token_validator(self) -> TokenValidator:
        """Access the token validator for external use."""
        return self._validator

    @property
    def token_manager(self) -> GovernanceToken:
        """Access the token manager for external use."""
        return self._token_mgr

    # ── Full Deliberation Flow ────────────────────────────────────────

    def evaluate(self, artifact: ReasoningArtifact) -> GovernanceResponseData:
        """Full Deliberation Flow — evaluate a Reasoning Artifact.

        This is the most thorough governance engagement. The agent's
        reasoning is evaluated before any action occurs.
        """
        start = time.time()
        artifact_hash = artifact.hash()

        # Step 1: Structural validation
        validation_errors = validate_artifact(artifact)
        if validation_errors:
            return self._build_revise_response(
                artifact_hash=artifact_hash,
                start=start,
                reasoning_gaps=validation_errors,
            )

        # Step 2: Structural evaluation
        assessment = self._structural_evaluation(artifact)

        # Step 3: Dimensional evaluation (if runtime is available)
        dim_summary: dict[str, float] = {}
        ucs_score: float | None = None
        trust_state: float | None = None
        runtime_verdict: Verdict | None = None
        runtime_vetoed_by: list[str] = []

        if self._runtime is not None:
            dim_result = self._dimensional_evaluation(artifact)
            dim_summary = dim_result["dimensional_summary"]
            ucs_score = dim_result["ucs"]
            trust_state = dim_result["trust_state"]
            runtime_verdict = dim_result["verdict"]
            runtime_vetoed_by = dim_result["vetoed_by"]
            assessment.dimensional_summary = dim_summary
            assessment.ucs = ucs_score
            assessment.trust_state = trust_state
            # Capture context modifications (Phase 7B)
            if "context_modification" in dim_result:
                assessment.metadata["context_modifications"] = dim_result["context_modification"].to_dict()

        # Step 3b: Ethical reasoning assessment (Phase 8)
        ethical_assessment: EthicalReasoningAssessment | None = None
        if self._ethical_reasoning_config is not None:
            ethical_assessment = self._assess_ethical_reasoning(
                artifact, self._ethical_reasoning_config,
            )
            assessment.metadata["ethical_reasoning"] = ethical_assessment.to_dict()

        # Step 4: Determine verdict
        verdict, conditions, guidance, escalation, denial = self._determine_verdict(
            artifact=artifact,
            assessment=assessment,
            ucs_score=ucs_score,
            runtime_verdict=runtime_verdict,
            runtime_vetoed_by=runtime_vetoed_by,
        )

        # Step 5: Issue token if approved
        token_str = ""
        if verdict in (ProtocolVerdict.PROCEED, ProtocolVerdict.PROCEED_WITH_CONDITIONS):
            token_str = self._token_mgr.issue(
                agent_id=artifact.agent_id,
                artifact_hash=artifact_hash,
                method=artifact.intended_action.method,
                action_target=artifact.intended_action.target,
                flow="full",
                scope="single",
                verdict=verdict.value,
                conditions=conditions if conditions else None,
                authority_envelope=artifact.envelope_id,
                config_version=self._config_version,
                ucs=ucs_score,
                trust=trust_state,
            )

        elapsed_ms = (time.time() - start) * 1000

        response = GovernanceResponseData(
            verdict=verdict,
            assessment=assessment,
            metadata=ResponseMetadata(
                evaluator_id=self.config.evaluator_id,
                evaluation_time_ms=round(elapsed_ms, 2),
                config_version=self._config_version,
                timestamp=datetime.now(timezone.utc).isoformat(),
                artifact_hash=artifact_hash,
            ),
            token=token_str,
            conditions=conditions or [],
            guidance=guidance,
            escalation=escalation,
            denial=denial,
        )

        # Store for introspection
        self._artifact_store[artifact_hash] = artifact
        self._response_store[response.response_id] = response

        return response

    # ── Summary Flow ──────────────────────────────────────────────────

    def evaluate_summary(self, artifact: ReasoningArtifact) -> GovernanceResponseData:
        """Summary Flow — streamlined evaluation for routine actions.

        Performs structural validation and a lighter evaluation pass.
        Suitable for actions within established authority.
        """
        start = time.time()
        artifact_hash = artifact.hash()

        validation_errors = validate_artifact(artifact)
        if validation_errors:
            return self._build_revise_response(
                artifact_hash=artifact_hash,
                start=start,
                reasoning_gaps=validation_errors,
            )

        assessment = self._structural_evaluation(artifact)

        # Summary flow uses structural assessment only (no dimensional)
        # unless runtime is available and config requires it
        ucs_score: float | None = None
        trust_state: float | None = None
        if self._runtime is not None and self.config.require_dimensional:
            dim_result = self._dimensional_evaluation(artifact)
            assessment.dimensional_summary = dim_result["dimensional_summary"]
            ucs_score = dim_result["ucs"]
            trust_state = dim_result["trust_state"]
            assessment.ucs = ucs_score
            assessment.trust_state = trust_state

        # Summary flow: approve if structural assessment is adequate
        avg_score = (
            assessment.completeness_score
            + assessment.alignment_score
            + assessment.uncertainty_calibration_score
            + assessment.alternatives_adequacy_score
        ) / 4.0

        if not assessment.authority_verified:
            verdict = ProtocolVerdict.ESCALATE
        elif avg_score >= self.config.min_completeness_for_proceed:
            verdict = ProtocolVerdict.PROCEED
        else:
            verdict = ProtocolVerdict.REVISE

        token_str = ""
        if verdict == ProtocolVerdict.PROCEED:
            token_str = self._token_mgr.issue(
                agent_id=artifact.agent_id,
                artifact_hash=artifact_hash,
                method=artifact.intended_action.method,
                action_target=artifact.intended_action.target,
                flow="summary",
                scope="class",
                verdict=verdict.value,
                authority_envelope=artifact.envelope_id,
                config_version=self._config_version,
                ucs=ucs_score,
                trust=trust_state,
            )

        elapsed_ms = (time.time() - start) * 1000

        escalation = None
        guidance = None
        if verdict == ProtocolVerdict.ESCALATE:
            escalation = Escalation(
                escalation_target="human_review",
                authority_required="Authority verification failed",
                context_package=f"Agent {artifact.agent_id} claimed {artifact.authority_claim.envelope_type} authority",
            )
        elif verdict == ProtocolVerdict.REVISE:
            guidance = Guidance(
                reasoning_gaps=["Summary evaluation found insufficient reasoning quality"],
            )

        response = GovernanceResponseData(
            verdict=verdict,
            assessment=assessment,
            metadata=ResponseMetadata(
                evaluator_id=self.config.evaluator_id,
                evaluation_time_ms=round(elapsed_ms, 2),
                config_version=self._config_version,
                timestamp=datetime.now(timezone.utc).isoformat(),
                artifact_hash=artifact_hash,
            ),
            token=token_str,
            guidance=guidance,
            escalation=escalation,
        )

        self._artifact_store[artifact_hash] = artifact
        self._response_store[response.response_id] = response

        return response

    # ── Post-Hoc Flow ─────────────────────────────────────────────────

    def evaluate_posthoc(
        self,
        artifact: ReasoningArtifact,
        action_result: dict[str, Any] | None = None,
    ) -> PostHocAssessment:
        """Post-Hoc Flow — retroactive evaluation after action.

        Evaluates whether the reasoning was sound and whether the action
        would have been approved. No token is issued.
        """
        validation_errors = validate_artifact(artifact)
        if validation_errors:
            return PostHocAssessment(
                sound_reasoning=False,
                would_have_approved=False,
                trust_adjustment=-0.02,
                concern=True,
                detail=f"Structural validation failed: {'; '.join(validation_errors)}",
            )

        assessment = self._structural_evaluation(artifact)

        avg_score = (
            assessment.completeness_score
            + assessment.alignment_score
            + assessment.uncertainty_calibration_score
            + assessment.alternatives_adequacy_score
        ) / 4.0

        sound = avg_score >= 0.6 and assessment.authority_verified
        would_approve = avg_score >= self.config.min_completeness_for_proceed and assessment.authority_verified

        if sound and would_approve:
            trust_adj = 0.01
            concern = False
        elif sound:
            trust_adj = 0.0
            concern = False
        elif would_approve:
            trust_adj = -0.01
            concern = True
        else:
            trust_adj = -0.03
            concern = True

        return PostHocAssessment(
            sound_reasoning=sound,
            would_have_approved=would_approve,
            trust_adjustment=trust_adj,
            concern=concern,
            detail=f"Average structural score: {avg_score:.2f}",
            assessment=assessment,
        )

    # ── Token operations ──────────────────────────────────────────────

    def validate_token(
        self,
        token: str,
        *,
        expected_method: str | None = None,
        expected_target: str | None = None,
    ) -> Any:
        """Validate a governance token."""
        return self._validator.validate(
            token,
            expected_method=expected_method,
            expected_target=expected_target,
        )

    def introspect_token(self, token: str) -> dict[str, Any] | None:
        """Full introspection of a token with governance context."""
        payload = self._validator.introspect(token)
        if payload is None:
            return None

        result: dict[str, Any] = {"claims": payload}

        # Look up the artifact and response if available
        artifact_hash = payload.get("nomo_artifact_hash", "")
        artifact = self._artifact_store.get(artifact_hash)
        if artifact is not None:
            result["artifact"] = artifact.to_dict()

        return result

    def revoke_token(self, token_id: str) -> bool:
        """Revoke a token by its JTI."""
        return self._validator.revoke(token_id)

    # ── Schema operations ─────────────────────────────────────────────

    def get_schema(self) -> dict[str, Any]:
        """Return the current Reasoning Artifact JSON Schema."""
        import json
        import importlib.resources as resources

        # Try to load from the schemas directory
        try:
            schema_path = resources.files("nomotic").parent.parent / "schemas" / "reasoning-artifact.v0.1.0.schema.json"
            return json.loads(schema_path.read_text())
        except Exception:
            return {"schema": "reasoning-artifact", "version": PROTOCOL_VERSION}

    def get_supported_versions(self) -> list[str]:
        """Return supported schema versions."""
        return list(self.config.supported_versions)

    # ── Internal evaluation methods ───────────────────────────────────

    def _structural_evaluation(self, artifact: ReasoningArtifact) -> Assessment:
        """Perform structural evaluation of a reasoning artifact."""

        # 1. Completeness — did the agent identify relevant constraints?
        completeness_score = 1.0
        missing: list[str] = []

        if not artifact.constraints_identified:
            completeness_score = 0.3
            missing.append("No constraints identified")
        else:
            # Check that constraint factors exist for each constraint type identified
            constraint_types_in_factors = {
                f.description for f in artifact.factors if f.type == "constraint"
            }
            if len(artifact.constraints_identified) > len(constraint_types_in_factors):
                completeness_score = max(0.5, completeness_score - 0.1)

        if not artifact.unknowns and not artifact.assumptions:
            # Agent declared no uncertainty at all — suspicious for non-trivial tasks
            if len(artifact.factors) > 2:
                completeness_score = max(0.5, completeness_score - 0.15)
                missing.append("No unknowns or assumptions declared for a multi-factor decision")

        completeness_detail = "All relevant constraints identified." if not missing else "; ".join(missing)

        # 2. Authority verification — is the authority claim plausible?
        auth_verified = True
        auth_detail = "Authority claim accepted."

        if artifact.authority_claim.envelope_type == "conditional":
            if not artifact.authority_claim.conditions_met:
                auth_verified = False
                auth_detail = "Conditional authority claimed but no conditions specified."
        elif artifact.authority_claim.envelope_type == "delegated":
            if not artifact.envelope_id:
                auth_detail = "Delegated authority claimed but no envelope_id provided. Accepted with caveat."

        # If runtime is available with certificates, verify against cert
        if self._runtime is not None and artifact.certificate_id:
            cert = self._runtime.get_certificate(artifact.agent_id)
            if cert is not None:
                from nomotic.certificate import CertStatus
                if cert.status != CertStatus.ACTIVE:
                    auth_verified = False
                    auth_detail = f"Agent certificate status is {cert.status.name}"

        # 3. Alignment — does the action follow from reasoning?
        alignment_score = 1.0
        alignment_detail = "Intended action follows from stated reasoning."
        misalignments: list[str] = []

        # Check that justifications reference relevant factors
        justified_factor_ids = {j.factor_id for j in artifact.justifications}
        decisive_factors = {f.id for f in artifact.factors if f.influence == "decisive"}

        # Decisive factors should be referenced in justifications
        for fid in decisive_factors:
            if fid not in justified_factor_ids:
                alignment_score -= 0.15
                misalignments.append(f"Decisive factor '{fid}' not referenced in justifications")

        alignment_score = max(0.0, alignment_score)
        if misalignments:
            alignment_detail = "; ".join(misalignments)

        # 4. Uncertainty calibration
        calibration_score = 1.0
        calibration_detail = "Confidence is well-calibrated."

        num_unknowns = len(artifact.unknowns)
        num_assumptions = len(artifact.assumptions)
        high_uncertainty_factors = sum(1 for f in artifact.factors if f.confidence < 0.5)

        # Check if confidence is consistent with uncertainties
        if artifact.overall_confidence > 0.9 and (num_unknowns > 2 or high_uncertainty_factors > 2):
            calibration_score = 0.5
            calibration_detail = (
                f"Overall confidence {artifact.overall_confidence} seems high given "
                f"{num_unknowns} unknowns and {high_uncertainty_factors} low-confidence factors."
            )
        elif artifact.overall_confidence > 0.95 and num_unknowns > 0:
            calibration_score = 0.7
            calibration_detail = (
                f"Overall confidence {artifact.overall_confidence} may be slightly high "
                f"given {num_unknowns} declared unknowns."
            )
        elif artifact.overall_confidence < 0.3 and num_unknowns == 0 and high_uncertainty_factors == 0:
            calibration_score = 0.6
            calibration_detail = (
                f"Overall confidence {artifact.overall_confidence} seems low given "
                "no declared unknowns or low-confidence factors."
            )

        # 5. Alternatives adequacy
        alt_score = 1.0
        alt_detail = "Alternatives considered adequately."

        num_alts = len(artifact.alternatives_considered)
        if num_alts == 0:
            # For non-trivial decisions, no alternatives is concerning
            num_factors = len(artifact.factors)
            if num_factors > 3:
                alt_score = 0.4
                alt_detail = "No alternatives considered for a complex decision."
            elif num_factors > 1:
                alt_score = 0.6
                alt_detail = "No alternatives considered."
            else:
                alt_score = 0.8
                alt_detail = "No alternatives considered, but decision appears straightforward."
        elif num_alts == 1:
            alt_score = 0.85
            alt_detail = "One alternative considered."

        return Assessment(
            completeness_score=round(completeness_score, 2),
            completeness_detail=completeness_detail,
            missing_constraints=missing,
            authority_verified=auth_verified,
            authority_detail=auth_detail,
            alignment_score=round(alignment_score, 2),
            alignment_detail=alignment_detail,
            misalignments=misalignments,
            uncertainty_calibration_score=round(calibration_score, 2),
            uncertainty_calibration_detail=calibration_detail,
            alternatives_adequacy_score=round(alt_score, 2),
            alternatives_adequacy_detail=alt_detail,
        )

    def _dimensional_evaluation(self, artifact: ReasoningArtifact) -> dict[str, Any]:
        """Delegate to the governance runtime for dimensional evaluation.

        Translates the Reasoning Artifact into the runtime's Action/Context
        types, runs the evaluation, and returns the results.
        """
        runtime = self._runtime

        # Build an Action from the artifact's intended action
        action = Action(
            agent_id=artifact.agent_id,
            action_type=artifact.intended_action.method,
            target=artifact.intended_action.target,
            parameters=artifact.intended_action.parameters,
            metadata={
                "protocol_flow": "full",
                "artifact_id": artifact.artifact_id,
                "method_category": method_category(artifact.intended_action.method) or "",
            },
        )

        # Build agent context
        trust_profile = runtime.get_trust_profile(artifact.agent_id)
        context = AgentContext(
            agent_id=artifact.agent_id,
            trust_profile=trust_profile,
            session_id=artifact.session_id or uuid.uuid4().hex[:12],
            context_profile_id=getattr(artifact, "context_profile_id", None),
        )

        # Run the governance evaluation
        gov_verdict = runtime.evaluate(action, context)

        # Extract dimension scores
        dim_summary: dict[str, float] = {}
        for score in gov_verdict.dimension_scores:
            # Normalize dimension name to snake_case for consistency
            dim_summary[score.dimension_name] = round(score.score, 4)

        result: dict[str, Any] = {
            "dimensional_summary": dim_summary,
            "ucs": round(gov_verdict.ucs, 4),
            "trust_state": round(trust_profile.overall_trust, 4),
            "verdict": gov_verdict.verdict,
            "vetoed_by": gov_verdict.vetoed_by,
            "tier": gov_verdict.tier,
        }
        # Include context modifications if present (Phase 7B)
        if gov_verdict.context_modification is not None:
            result["context_modification"] = gov_verdict.context_modification
        return result

    def _determine_verdict(
        self,
        artifact: ReasoningArtifact,
        assessment: Assessment,
        ucs_score: float | None,
        runtime_verdict: Verdict | None,
        runtime_vetoed_by: list[str],
    ) -> tuple[
        ProtocolVerdict,
        list[Condition] | None,
        Guidance | None,
        Escalation | None,
        Denial | None,
    ]:
        """Determine the protocol verdict from evaluation results."""
        conditions: list[Condition] | None = None
        guidance: Guidance | None = None
        escalation: Escalation | None = None
        denial: Denial | None = None

        # If runtime issued a DENY with vetoes, respect that
        if runtime_verdict == Verdict.DENY and runtime_vetoed_by:
            denial = Denial(
                grounds=[f"Dimension veto: {dim}" for dim in runtime_vetoed_by],
                veto_dimensions=runtime_vetoed_by,
                remediation="Address the veto dimensions and resubmit.",
            )
            return ProtocolVerdict.DENY, conditions, guidance, escalation, denial

        # If authority is not verified, escalate
        if not assessment.authority_verified:
            escalation = Escalation(
                escalation_target="human_review",
                authority_required="Authority verification failed",
                context_package=f"Agent {artifact.agent_id} claimed {artifact.authority_claim.envelope_type} authority. {assessment.authority_detail}",
            )
            return ProtocolVerdict.ESCALATE, conditions, guidance, escalation, denial

        # Compute structural average
        struct_avg = (
            assessment.completeness_score
            + assessment.alignment_score
            + assessment.uncertainty_calibration_score
            + assessment.alternatives_adequacy_score
        ) / 4.0

        # If we have dimensional evaluation, use UCS as primary signal
        if ucs_score is not None:
            if runtime_verdict == Verdict.DENY:
                denial = Denial(
                    grounds=[f"UCS {ucs_score:.3f} below threshold {self.config.deny_threshold}"],
                    remediation="Improve reasoning quality and constraint coverage.",
                )
                return ProtocolVerdict.DENY, conditions, guidance, escalation, denial

            if runtime_verdict == Verdict.ESCALATE:
                escalation = Escalation(
                    escalation_target="human_review",
                    authority_required="Tier 3 deliberation resulted in escalation",
                )
                return ProtocolVerdict.ESCALATE, conditions, guidance, escalation, denial

            if runtime_verdict == Verdict.MODIFY:
                conditions = [
                    Condition(
                        type="monitoring",
                        description="Action approved with enhanced monitoring due to governance modifications.",
                    ),
                ]
                return ProtocolVerdict.PROCEED_WITH_CONDITIONS, conditions, guidance, escalation, denial

            if runtime_verdict == Verdict.ALLOW:
                # Check structural quality — if below threshold, add conditions
                if struct_avg < 0.7:
                    conditions = [
                        Condition(
                            type="audit_level",
                            description="Enhanced audit logging due to structural reasoning quality.",
                            parameters={"structural_score": round(struct_avg, 2)},
                        ),
                    ]
                    return ProtocolVerdict.PROCEED_WITH_CONDITIONS, conditions, guidance, escalation, denial
                return ProtocolVerdict.PROCEED, conditions, guidance, escalation, denial

        # Structural-only evaluation (no runtime)
        if struct_avg < self.config.min_completeness_for_proceed:
            guidance = Guidance(
                reasoning_gaps=[
                    f"Overall structural quality ({struct_avg:.2f}) below threshold ({self.config.min_completeness_for_proceed})",
                ],
                missing_constraints=assessment.missing_constraints,
                recommended_factors=self._recommend_factors(artifact),
            )
            return ProtocolVerdict.REVISE, conditions, guidance, escalation, denial

        if struct_avg < 0.7:
            conditions = [
                Condition(
                    type="audit_level",
                    description="Enhanced audit logging due to moderate reasoning quality.",
                    parameters={"structural_score": round(struct_avg, 2)},
                ),
            ]
            return ProtocolVerdict.PROCEED_WITH_CONDITIONS, conditions, guidance, escalation, denial

        return ProtocolVerdict.PROCEED, conditions, guidance, escalation, denial

    def _recommend_factors(self, artifact: ReasoningArtifact) -> list[str]:
        """Recommend factor types the agent should consider."""
        existing_types = {f.type for f in artifact.factors}
        recommendations: list[str] = []
        if "risk" not in existing_types:
            recommendations.append("risk")
        if "precedent" not in existing_types:
            recommendations.append("precedent")
        if "uncertainty" not in existing_types and artifact.unknowns:
            recommendations.append("uncertainty")
        return recommendations

    def _build_revise_response(
        self,
        artifact_hash: str,
        start: float,
        reasoning_gaps: list[str],
    ) -> GovernanceResponseData:
        """Build a REVISE response for validation failures."""
        elapsed_ms = (time.time() - start) * 1000
        return GovernanceResponseData(
            verdict=ProtocolVerdict.REVISE,
            assessment=Assessment(
                completeness_score=0.0,
                completeness_detail="Artifact failed structural validation.",
            ),
            metadata=ResponseMetadata(
                evaluator_id=self.config.evaluator_id,
                evaluation_time_ms=round(elapsed_ms, 2),
                config_version=self._config_version,
                timestamp=datetime.now(timezone.utc).isoformat(),
                artifact_hash=artifact_hash,
            ),
            guidance=Guidance(reasoning_gaps=reasoning_gaps),
        )

    # ── Ethical Reasoning Assessment (Phase 8) ─────────────────────────

    def _assess_ethical_reasoning(
        self,
        artifact: ReasoningArtifact,
        config: EthicalReasoningConfig,
    ) -> EthicalReasoningAssessment:
        """Assess the ethical quality of an agent's reasoning.

        All evaluation is structural — no semantic AI evaluation.
        """
        findings: list[str] = []
        recommendations: list[str] = []
        method = artifact.intended_action.method

        # ── 1. Stakeholder consideration ──
        stakeholder_score = 0.0
        has_risk_factors_with_stakeholders = any(
            f.type == "risk" and "stakeholder" in f.description.lower()
            for f in artifact.factors
        )
        has_ethical_constraint = any(
            "ethic" in c.type.lower() or "ethic" in c.description.lower()
            for c in artifact.constraints_identified
        )
        has_affected_parties_factor = any(
            "affected" in f.description.lower()
            or "stakeholder" in f.description.lower()
            or "user" in f.description.lower()
            or "impact" in f.description.lower()
            for f in artifact.factors
        )

        if has_risk_factors_with_stakeholders:
            stakeholder_score += 0.3
        if has_ethical_constraint:
            stakeholder_score += 0.2
        if has_affected_parties_factor:
            stakeholder_score += 0.3
        # Minimum: if the action is non-trivial and no stakeholders considered
        if not has_risk_factors_with_stakeholders and not has_affected_parties_factor:
            if len(artifact.factors) > 1:
                stakeholder_score = max(stakeholder_score, 0.2)
                findings.append("No stakeholder factors identified in reasoning.")
                recommendations.append(
                    "Consider adding factors that address affected parties."
                )
            else:
                stakeholder_score = max(stakeholder_score, 0.5)

        stakeholder_score = min(1.0, stakeholder_score)

        # ── 2. Harm awareness ──
        harm_score = 0.0
        has_harm_in_uncertainty = any(
            "harm" in u.description.lower() or "risk" in u.description.lower()
            or "damage" in u.description.lower()
            for u in artifact.unknowns
        )
        has_significant_risk_factor = any(
            f.type == "risk" and f.influence in ("significant", "decisive")
            for f in artifact.factors
        )
        has_harm_reducing_alternatives = any(
            "harm" in a.reason_rejected.lower() or "risk" in a.reason_rejected.lower()
            or "safe" in a.reason_rejected.lower()
            for a in artifact.alternatives_considered
        )
        has_any_risk_factor = any(f.type == "risk" for f in artifact.factors)

        if has_harm_in_uncertainty:
            harm_score += 0.4
        if has_significant_risk_factor:
            harm_score += 0.3
        if has_harm_reducing_alternatives:
            harm_score += 0.3

        # If it's a decision method and no risk factors at all, flag
        if (
            method in config.require_harm_acknowledgment_for_methods
            and not has_any_risk_factor
        ):
            harm_score = max(harm_score, 0.2)
            findings.append(
                f"Method '{method}' should include harm awareness in reasoning."
            )
            recommendations.append("Add risk-type factors acknowledging potential harm.")

        harm_score = min(1.0, harm_score)

        # ── 3. Fairness consideration ──
        fairness_score = 0.0
        has_fairness_factor = any(
            any(
                keyword in f.description.lower()
                for keyword in ("equit", "fair", "equal", "non-discriminat", "bias")
            )
            for f in artifact.factors
        )
        has_fairness_constraint = any(
            any(
                keyword in c.description.lower()
                for keyword in ("equit", "fair", "equal", "non-discriminat", "bias")
            )
            for c in artifact.constraints_identified
        )
        has_differential_impact_factor = any(
            "differential" in f.description.lower()
            or "different population" in f.description.lower()
            or "different group" in f.description.lower()
            for f in artifact.factors
        )

        if has_fairness_factor or has_fairness_constraint:
            fairness_score += 0.4
        if has_differential_impact_factor:
            fairness_score += 0.3

        # For decision methods, check more strictly
        if method in config.fairness_relevant_methods:
            if not has_fairness_factor and not has_fairness_constraint:
                fairness_score = max(fairness_score, 0.2)
                findings.append(
                    f"Method '{method}' has no fairness considerations in reasoning."
                )
                recommendations.append(
                    "Consider adding factors addressing equitable treatment."
                )
            else:
                fairness_score = max(fairness_score, 0.5)

        fairness_score = min(1.0, fairness_score)

        # ── 4. Alternative equity ──
        alt_score = 0.0
        num_alts = len(artifact.alternatives_considered)
        if num_alts == 0:
            alt_score = 0.1
            findings.append("No alternatives considered.")
        elif num_alts == 1:
            alt_score = 0.3
        else:
            # Check if alternatives are substantively different
            methods = {a.method for a in artifact.alternatives_considered}
            if len(methods) > 1:
                alt_score = 0.5
            else:
                alt_score = 0.3  # All same method, different parameters

            # Check if any alternative considers different populations
            has_population_alt = any(
                any(
                    keyword in a.reason_rejected.lower()
                    for keyword in ("population", "group", "demographic", "equity")
                )
                for a in artifact.alternatives_considered
            )
            if has_population_alt:
                alt_score += 0.3

        alt_score = min(1.0, alt_score)

        # ── 5. Uncertainty honesty (ethical dimension) ──
        uncertainty_score = 0.0
        has_impact_unknowns = any(
            any(
                keyword in u.description.lower()
                for keyword in ("impact", "affect", "people", "user", "harm")
            )
            for u in artifact.unknowns
        )

        if has_impact_unknowns:
            uncertainty_score += 0.4

        # Is confidence appropriately lower when ethical dimensions uncertain?
        ethical_uncertainties = sum(
            1 for u in artifact.unknowns
            if any(
                keyword in u.description.lower()
                for keyword in ("ethic", "fair", "harm", "impact", "bias")
            )
        )
        if ethical_uncertainties > 0 and artifact.overall_confidence < 0.8:
            uncertainty_score += 0.3
        elif ethical_uncertainties > 0 and artifact.overall_confidence >= 0.8:
            findings.append(
                "High confidence despite ethical uncertainties."
            )

        # Acknowledges what it doesn't know about affected parties
        if any(
            "affected" in u.description.lower() or "stakeholder" in u.description.lower()
            for u in artifact.unknowns
        ):
            uncertainty_score += 0.3

        # If no unknowns at all but non-trivial action
        if not artifact.unknowns and len(artifact.factors) > 2:
            uncertainty_score = max(uncertainty_score, 0.2)

        uncertainty_score = min(1.0, uncertainty_score)

        # ── Overall composite ──
        total_weight = (
            config.stakeholder_weight
            + config.harm_weight
            + config.fairness_weight
            + config.alternative_weight
            + config.uncertainty_weight
        )

        if total_weight > 0:
            overall = (
                stakeholder_score * config.stakeholder_weight
                + harm_score * config.harm_weight
                + fairness_score * config.fairness_weight
                + alt_score * config.alternative_weight
                + uncertainty_score * config.uncertainty_weight
            ) / total_weight
        else:
            overall = 0.5  # Neutral if all weights zero

        # Check minimum threshold
        if overall < config.minimum_ethical_reasoning_score:
            findings.append(
                f"Overall ethical reasoning score ({overall:.2f}) below "
                f"minimum threshold ({config.minimum_ethical_reasoning_score})."
            )

        return EthicalReasoningAssessment(
            stakeholder_consideration=stakeholder_score,
            harm_awareness=harm_score,
            fairness_consideration=fairness_score,
            alternative_equity=alt_score,
            uncertainty_honesty=uncertainty_score,
            overall_ethical_reasoning_score=overall,
            findings=findings,
            recommendations=recommendations,
        )
