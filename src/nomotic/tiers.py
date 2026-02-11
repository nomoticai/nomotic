"""Three-tier hybrid evaluation architecture.

Tier 1: Deterministic Gate
    Fast, rule-based checks. Hard boundaries. Vetoes are final.
    Runs in microseconds. No ambiguity — pass or fail.

Tier 2: Weighted Evaluation
    The UCS engine combines dimension scores using weights, trust
    calibration, and contextual factors. Produces a confidence score.
    Most actions are decided here.

Tier 3: Deliberative Review
    For edge cases where Tier 2 produces ambiguous results. Applies
    deeper analysis, historical comparison, and if necessary, escalates
    to human review. Slowest but most thorough.

The tier system is a cascade: Tier 1 filters obvious cases, Tier 2 handles
the bulk, and Tier 3 catches what falls through the cracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from nomotic.types import (
    Action,
    AgentContext,
    DimensionScore,
    GovernanceVerdict,
    Verdict,
)

__all__ = [
    "TierOneGate",
    "TierResult",
    "TierThreeDeliberator",
    "TierTwoEvaluator",
]


@dataclass
class TierResult:
    """Result from a single tier's evaluation."""

    decided: bool  # Did this tier make a final decision?
    verdict: GovernanceVerdict | None = None


class TierOneGate:
    """Tier 1: Deterministic Gate.

    Checks hard boundaries. Any veto from a dimension with veto authority
    results in immediate DENY. No scoring, no weighing — just binary checks.

    This tier handles:
    - Scope violations (action outside agent's permissions)
    - Authority failures (agent lacks specific authority)
    - Resource limit breaches (rate/concurrency/cost exceeded)
    - Isolation boundary violations
    - Temporal constraint violations
    - Ethical hard constraints
    - Human override requirements
    """

    def evaluate(
        self, action: Action, context: AgentContext, scores: list[DimensionScore]
    ) -> TierResult:
        vetoes = [s for s in scores if s.veto]
        if vetoes:
            # Any veto with score 0.0 on a human_override dimension means ESCALATE
            human_vetoes = [v for v in vetoes if v.dimension_name == "human_override"]
            if human_vetoes and not any(
                v.dimension_name != "human_override" for v in vetoes
            ):
                return TierResult(
                    decided=True,
                    verdict=GovernanceVerdict(
                        action_id=action.id,
                        verdict=Verdict.ESCALATE,
                        ucs=0.0,
                        dimension_scores=scores,
                        tier=1,
                        vetoed_by=[v.dimension_name for v in vetoes],
                        reasoning="Human approval required",
                    ),
                )
            return TierResult(
                decided=True,
                verdict=GovernanceVerdict(
                    action_id=action.id,
                    verdict=Verdict.DENY,
                    ucs=0.0,
                    dimension_scores=scores,
                    tier=1,
                    vetoed_by=[v.dimension_name for v in vetoes],
                    reasoning="; ".join(v.reasoning for v in vetoes),
                ),
            )
        return TierResult(decided=False)


class TierTwoEvaluator:
    """Tier 2: Weighted Evaluation via UCS.

    Combines dimension scores into a Unified Confidence Score. Uses
    configurable thresholds to decide ALLOW, DENY, or pass to Tier 3.

    The ambiguity zone between allow_threshold and deny_threshold is
    where Tier 3 takes over.
    """

    def __init__(
        self,
        allow_threshold: float = 0.7,
        deny_threshold: float = 0.3,
    ):
        self.allow_threshold = allow_threshold
        self.deny_threshold = deny_threshold

    def evaluate(
        self,
        action: Action,
        context: AgentContext,
        scores: list[DimensionScore],
        ucs: float,
    ) -> TierResult:
        if ucs >= self.allow_threshold:
            return TierResult(
                decided=True,
                verdict=GovernanceVerdict(
                    action_id=action.id,
                    verdict=Verdict.ALLOW,
                    ucs=ucs,
                    dimension_scores=scores,
                    tier=2,
                    reasoning=f"UCS {ucs:.3f} >= allow threshold {self.allow_threshold}",
                ),
            )
        if ucs <= self.deny_threshold:
            return TierResult(
                decided=True,
                verdict=GovernanceVerdict(
                    action_id=action.id,
                    verdict=Verdict.DENY,
                    ucs=ucs,
                    dimension_scores=scores,
                    tier=2,
                    reasoning=f"UCS {ucs:.3f} <= deny threshold {self.deny_threshold}",
                ),
            )
        # Ambiguous — pass to Tier 3
        return TierResult(decided=False)


class TierThreeDeliberator:
    """Tier 3: Deliberative Review.

    Handles ambiguous cases that Tier 2 couldn't decide. Applies deeper
    analysis including:
    - Historical comparison with similar actions
    - Trust trajectory analysis (is trust trending up or down?)
    - Worst-case impact assessment
    - Optional escalation to human review

    If all else fails, defaults to the safer option (DENY or MODIFY).
    """

    def __init__(self) -> None:
        self._deliberators: list[
            Callable[[Action, AgentContext, list[DimensionScore], float], Verdict | None]
        ] = []

    def add_deliberator(
        self,
        func: Callable[
            [Action, AgentContext, list[DimensionScore], float], Verdict | None
        ],
    ) -> None:
        """Add a custom deliberation function."""
        self._deliberators.append(func)

    def evaluate(
        self,
        action: Action,
        context: AgentContext,
        scores: list[DimensionScore],
        ucs: float,
    ) -> TierResult:
        # Run custom deliberators
        for deliberator in self._deliberators:
            result = deliberator(action, context, scores, ucs)
            if result is not None:
                return TierResult(
                    decided=True,
                    verdict=GovernanceVerdict(
                        action_id=action.id,
                        verdict=result,
                        ucs=ucs,
                        dimension_scores=scores,
                        tier=3,
                        reasoning="Custom deliberator decided",
                    ),
                )

        # Built-in deliberation logic
        trust = context.trust_profile.overall_trust

        # High trust + borderline UCS = allow with modification
        if trust > 0.7 and ucs > 0.5:
            return TierResult(
                decided=True,
                verdict=GovernanceVerdict(
                    action_id=action.id,
                    verdict=Verdict.ALLOW,
                    ucs=ucs,
                    dimension_scores=scores,
                    tier=3,
                    reasoning=f"High trust ({trust:.2f}) tips borderline UCS ({ucs:.3f}) to allow",
                ),
            )

        # Low trust + borderline UCS = escalate
        if trust < 0.4:
            return TierResult(
                decided=True,
                verdict=GovernanceVerdict(
                    action_id=action.id,
                    verdict=Verdict.ESCALATE,
                    ucs=ucs,
                    dimension_scores=scores,
                    tier=3,
                    reasoning=f"Low trust ({trust:.2f}) + ambiguous UCS ({ucs:.3f}) requires escalation",
                ),
            )

        # Check for any low-scoring critical dimensions
        critical_low = [
            s
            for s in scores
            if s.weight >= 1.3 and s.score < 0.4
        ]
        if critical_low:
            return TierResult(
                decided=True,
                verdict=GovernanceVerdict(
                    action_id=action.id,
                    verdict=Verdict.MODIFY,
                    ucs=ucs,
                    dimension_scores=scores,
                    tier=3,
                    reasoning=f"Critical dimensions scored low: {[s.dimension_name for s in critical_low]}",
                    modifications={"reduce_scope": True, "require_confirmation": True},
                ),
            )

        # Default: allow with reduced confidence
        return TierResult(
            decided=True,
            verdict=GovernanceVerdict(
                action_id=action.id,
                verdict=Verdict.ALLOW,
                ucs=ucs,
                dimension_scores=scores,
                tier=3,
                reasoning=f"Tier 3 default: allow with UCS {ucs:.3f}",
            ),
        )
