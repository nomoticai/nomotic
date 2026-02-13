"""Trust calibration — continuously updated confidence in agents.

Trust is not binary. It is calibrated continuously based on observed behavior.
Interruption rights operationalize that calibration: when trust is high,
intervention is rare. When behavior deviates, intervention capability exists.

The TrustCalibrator updates trust profiles after every action based on:
- Whether the action was approved or denied
- Whether the action completed successfully or was interrupted
- How closely the action matched governance expectations
- Trend analysis — is trust trending up or down?

Trust feeds back into governance. Lower trust means:
- Lower UCS scores (trust modulation in the UCS engine)
- More likely to trigger Tier 3 deliberation
- More likely to require human override
- More likely to get interrupted
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nomotic.types import (
    ActionRecord,
    ActionState,
    GovernanceVerdict,
    TrustProfile,
    Verdict,
)
from nomotic.trajectory import (
    TrustTrajectory,
    SOURCE_COMPLETION_INTERRUPT,
    SOURCE_COMPLETION_SUCCESS,
    SOURCE_DRIFT_ADJUSTMENT,
    SOURCE_DRIFT_RECOVERY,
    SOURCE_TIME_DECAY,
    SOURCE_VERDICT_ALLOW,
    SOURCE_VERDICT_DENY,
)

if TYPE_CHECKING:
    from nomotic.drift import DriftScore

__all__ = ["TrustCalibrator", "TrustConfig"]


@dataclass
class TrustConfig:
    """Configuration for trust calibration behavior."""

    # How much a successful action increases trust
    success_increment: float = 0.01
    # How much a violation decreases trust
    violation_decrement: float = 0.05
    # How much an interruption decreases trust
    interrupt_decrement: float = 0.03
    # Trust decays toward this baseline over time
    baseline_trust: float = 0.5
    # How fast trust decays toward baseline (per hour)
    decay_rate: float = 0.01
    # Minimum trust floor
    min_trust: float = 0.05
    # Maximum trust ceiling
    max_trust: float = 0.95
    # Number of successful actions to reach full trust from baseline
    ramp_up_actions: int = 100


class TrustCalibrator:
    """Updates trust profiles based on observed agent behavior.

    Called after every governance decision and action completion.
    Trust changes are small and incremental — building trust is slow,
    losing it is fast. This is by design.
    """

    def __init__(self, config: TrustConfig | None = None):
        self.config = config or TrustConfig()
        self._validate_config(self.config)
        self._profiles: dict[str, TrustProfile] = {}
        self._trajectories: dict[str, TrustTrajectory] = {}
        self._last_drift_scores: dict[str, float] = {}

    @staticmethod
    def _validate_config(config: TrustConfig) -> None:
        if not 0.0 <= config.min_trust <= config.baseline_trust <= config.max_trust <= 1.0:
            raise ValueError(
                f"Trust bounds must satisfy 0 <= min_trust <= baseline_trust <= max_trust <= 1, "
                f"got min={config.min_trust}, baseline={config.baseline_trust}, max={config.max_trust}"
            )
        if config.success_increment < 0:
            raise ValueError(f"success_increment must be non-negative, got {config.success_increment}")
        if config.violation_decrement < 0:
            raise ValueError(f"violation_decrement must be non-negative, got {config.violation_decrement}")
        if config.interrupt_decrement < 0:
            raise ValueError(f"interrupt_decrement must be non-negative, got {config.interrupt_decrement}")
        if config.decay_rate < 0:
            raise ValueError(f"decay_rate must be non-negative, got {config.decay_rate}")

    def get_profile(self, agent_id: str) -> TrustProfile:
        """Get or create a trust profile for an agent."""
        if agent_id not in self._profiles:
            self._profiles[agent_id] = TrustProfile(
                agent_id=agent_id,
                overall_trust=self.config.baseline_trust,
            )
        return self._profiles[agent_id]

    def get_trajectory(self, agent_id: str) -> TrustTrajectory:
        """Get or create the trust trajectory for an agent."""
        if agent_id not in self._trajectories:
            self._trajectories[agent_id] = TrustTrajectory(agent_id)
        return self._trajectories[agent_id]

    def record_verdict(
        self, agent_id: str, verdict: GovernanceVerdict
    ) -> TrustProfile:
        """Update trust based on a governance verdict.

        Called after the governance decision is made but before execution.
        A denied action is a signal that the agent attempted something
        that governance rejected.
        """
        profile = self.get_profile(agent_id)
        trajectory = self.get_trajectory(agent_id)
        trust_before = profile.overall_trust

        if verdict.verdict == Verdict.DENY:
            profile.violation_count += 1
            profile.last_violation_time = time.time()
            profile.overall_trust = max(
                self.config.min_trust,
                profile.overall_trust - self.config.violation_decrement,
            )
            # Update per-dimension trust
            for score in verdict.dimension_scores:
                if score.veto or score.score < 0.3:
                    dim_trust = profile.dimension_trust.get(
                        score.dimension_name, self.config.baseline_trust
                    )
                    profile.dimension_trust[score.dimension_name] = max(
                        self.config.min_trust,
                        dim_trust - self.config.violation_decrement,
                    )

            trajectory.record(
                trust_before=trust_before,
                trust_after=profile.overall_trust,
                source=SOURCE_VERDICT_DENY,
                reason=f"Action denied: {verdict.reasoning[:100]}" if verdict.reasoning else "Action denied",
                metadata={"action_id": verdict.action_id, "tier": verdict.tier},
            )

        elif verdict.verdict == Verdict.ALLOW:
            profile.successful_actions += 1
            profile.overall_trust = min(
                self.config.max_trust,
                profile.overall_trust + self.config.success_increment,
            )
            # Gradually restore per-dimension trust
            for score in verdict.dimension_scores:
                if score.score > 0.7:
                    dim_trust = profile.dimension_trust.get(
                        score.dimension_name, self.config.baseline_trust
                    )
                    profile.dimension_trust[score.dimension_name] = min(
                        self.config.max_trust,
                        dim_trust + self.config.success_increment * 0.5,
                    )

            trajectory.record(
                trust_before=trust_before,
                trust_after=profile.overall_trust,
                source=SOURCE_VERDICT_ALLOW,
                reason="Action allowed",
                metadata={"action_id": verdict.action_id},
            )

        profile.last_updated = time.time()
        return profile

    def record_completion(
        self, agent_id: str, record: ActionRecord
    ) -> TrustProfile:
        """Update trust based on action completion.

        Called after an action finishes executing (or is interrupted).
        Successful completion reinforces trust. Interruption reduces it.
        """
        profile = self.get_profile(agent_id)
        trajectory = self.get_trajectory(agent_id)
        trust_before = profile.overall_trust

        if record.interrupted:
            profile.overall_trust = max(
                self.config.min_trust,
                profile.overall_trust - self.config.interrupt_decrement,
            )
            trajectory.record(
                trust_before=trust_before,
                trust_after=profile.overall_trust,
                source=SOURCE_COMPLETION_INTERRUPT,
                reason=f"Action interrupted: {record.interrupt_reason[:100]}" if record.interrupt_reason else "Action interrupted",
            )
        elif record.state == ActionState.COMPLETED:
            profile.overall_trust = min(
                self.config.max_trust,
                profile.overall_trust + self.config.success_increment * 0.5,
            )
            trajectory.record(
                trust_before=trust_before,
                trust_after=profile.overall_trust,
                source=SOURCE_COMPLETION_SUCCESS,
                reason="Action completed successfully",
            )

        profile.last_updated = time.time()
        return profile

    def apply_time_decay(self, agent_id: str) -> TrustProfile:
        """Apply time-based trust decay toward baseline.

        Trust that isn't actively reinforced drifts back toward the
        baseline. This prevents stale high-trust profiles from persisting
        after an agent has been idle.
        """
        profile = self.get_profile(agent_id)
        trust_before = profile.overall_trust
        elapsed_hours = (time.time() - profile.last_updated) / 3600
        if elapsed_hours < 0.01:
            return profile

        decay = self.config.decay_rate * elapsed_hours
        if profile.overall_trust > self.config.baseline_trust:
            profile.overall_trust = max(
                self.config.baseline_trust,
                profile.overall_trust - decay,
            )
        elif profile.overall_trust < self.config.baseline_trust:
            profile.overall_trust = min(
                self.config.baseline_trust,
                profile.overall_trust + decay,
            )

        if abs(profile.overall_trust - trust_before) > 0.001:
            trajectory = self.get_trajectory(agent_id)
            trajectory.record(
                trust_before=trust_before,
                trust_after=profile.overall_trust,
                source=SOURCE_TIME_DECAY,
                reason="Trust decayed toward baseline during inactivity",
            )

        profile.last_updated = time.time()
        return profile

    def apply_drift(
        self,
        agent_id: str,
        drift_score: DriftScore,
    ) -> TrustProfile:
        """Adjust trust based on behavioral drift.

        Called by the runtime after drift is computed (every check_interval
        observations).  This is the bridge between drift detection and trust.

        Drift adjustment rules:
        - drift.overall < 0.10: no adjustment (within normal variance)
        - drift.overall 0.10-0.20: very small erosion (-0.002 per check)
        - drift.overall 0.20-0.40: moderate erosion (-0.008 per check)
        - drift.overall 0.40-0.60: significant erosion (-0.02 per check)
        - drift.overall >= 0.60: heavy erosion (-0.04 per check)

        Recovery: if drift decreases from the previous check back below
        0.15, apply a small recovery (+0.003 per check).

        All adjustments are scaled by drift.confidence.
        """
        profile = self.get_profile(agent_id)
        trajectory = self.get_trajectory(agent_id)

        trust_before = profile.overall_trust

        confidence = drift_score.confidence

        drift = drift_score.overall
        previous_drift = self._last_drift_scores.get(agent_id, 0.0)
        self._last_drift_scores[agent_id] = drift

        adjustment = 0.0
        reason = ""

        if drift >= 0.60:
            adjustment = -0.04 * confidence
            reason = f"Critical behavioral drift ({drift:.2f})"
        elif drift >= 0.40:
            adjustment = -0.02 * confidence
            reason = f"High behavioral drift ({drift:.2f})"
        elif drift >= 0.20:
            adjustment = -0.008 * confidence
            reason = f"Moderate behavioral drift ({drift:.2f})"
        elif drift >= 0.10:
            adjustment = -0.002 * confidence
            reason = f"Low behavioral drift ({drift:.2f})"
        elif drift < 0.15 and previous_drift >= 0.15:
            adjustment = 0.003 * confidence
            reason = f"Behavioral drift recovered ({previous_drift:.2f} -> {drift:.2f})"

        if abs(adjustment) < 0.0001:
            return profile

        profile.overall_trust = max(
            self.config.min_trust,
            min(self.config.max_trust, profile.overall_trust + adjustment),
        )

        source = SOURCE_DRIFT_RECOVERY if adjustment > 0 else SOURCE_DRIFT_ADJUSTMENT
        trajectory.record(
            trust_before=trust_before,
            trust_after=profile.overall_trust,
            source=source,
            reason=reason,
            metadata={
                "drift_overall": drift,
                "drift_confidence": confidence,
                "drift_severity": drift_score.severity,
                "adjustment": adjustment,
            },
        )

        profile.last_updated = time.time()
        return profile
