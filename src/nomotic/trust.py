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

from nomotic.types import (
    ActionRecord,
    ActionState,
    GovernanceVerdict,
    TrustProfile,
    Verdict,
)


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
        self._profiles: dict[str, TrustProfile] = {}

    def get_profile(self, agent_id: str) -> TrustProfile:
        """Get or create a trust profile for an agent."""
        if agent_id not in self._profiles:
            self._profiles[agent_id] = TrustProfile(
                agent_id=agent_id,
                overall_trust=self.config.baseline_trust,
            )
        return self._profiles[agent_id]

    def record_verdict(
        self, agent_id: str, verdict: GovernanceVerdict
    ) -> TrustProfile:
        """Update trust based on a governance verdict.

        Called after the governance decision is made but before execution.
        A denied action is a signal that the agent attempted something
        that governance rejected.
        """
        profile = self.get_profile(agent_id)

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

        if record.interrupted:
            profile.overall_trust = max(
                self.config.min_trust,
                profile.overall_trust - self.config.interrupt_decrement,
            )
        elif record.state == ActionState.COMPLETED:
            profile.overall_trust = min(
                self.config.max_trust,
                profile.overall_trust + self.config.success_increment * 0.5,
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

        profile.last_updated = time.time()
        return profile
