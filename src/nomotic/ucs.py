"""Unified Confidence Score (UCS) engine.

Combines all dimension scores into a single governance confidence value
between 0.0 (deny) and 1.0 (full confidence to allow).

The UCS is not a simple weighted average. It incorporates:
- Weighted dimension scores (each dimension has a configured weight)
- Confidence adjustments (dimensions can express uncertainty)
- Trust modulation (agent trust level influences the final score)
- Veto override (any veto forces UCS to 0.0)

The UCS is the primary input to Tier 2 evaluation.
"""

from __future__ import annotations

from nomotic.types import AgentContext, DimensionScore, TrustProfile


class UCSEngine:
    """Computes the Unified Confidence Score from dimension evaluations."""

    def __init__(self, trust_influence: float = 0.2):
        """
        Args:
            trust_influence: How much agent trust level affects the UCS.
                0.0 = trust has no effect, 1.0 = trust dominates.
                Default 0.2 means trust can shift the score by ±20%.
        """
        self.trust_influence = trust_influence

    def compute(
        self,
        scores: list[DimensionScore],
        trust: TrustProfile,
    ) -> float:
        """Compute the Unified Confidence Score.

        Returns a float between 0.0 and 1.0.
        """
        if not scores:
            return 0.5  # No data — neutral

        # Any veto forces UCS to 0
        if any(s.veto for s in scores):
            return 0.0

        # Weighted average of dimension scores, adjusted by confidence
        total_weight = 0.0
        weighted_sum = 0.0
        for s in scores:
            effective_weight = s.weight * s.confidence
            weighted_sum += s.score * effective_weight
            total_weight += effective_weight

        if total_weight == 0:
            return 0.5

        base_score = weighted_sum / total_weight

        # Trust modulation — shifts the score toward trust level
        trust_delta = (trust.overall_trust - 0.5) * self.trust_influence
        modulated = base_score + trust_delta

        # Floor and ceiling effects from extreme dimension scores
        min_score = min(s.score for s in scores)
        if min_score < 0.2:
            # A very low individual score drags down the overall UCS
            drag = (0.2 - min_score) * 0.3
            modulated -= drag

        return max(0.0, min(1.0, modulated))
