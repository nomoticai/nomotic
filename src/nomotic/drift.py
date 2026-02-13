"""Drift calculator — measures behavioral deviation from baseline.

Compares two fingerprints (baseline and recent window) and produces a
drift score.  Pure computation — no side effects, no state mutation.

The core metric is Jensen-Shannon Divergence (JSD), a symmetric,
bounded distance between probability distributions.  JSD is computed
with basic math (:mod:`math`) — no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nomotic.fingerprint import BehavioralFingerprint, TemporalPattern

if TYPE_CHECKING:
    from nomotic.priors import ArchetypePrior

__all__ = [
    "DriftCalculator",
    "DriftScore",
]

# Tiny smoothing constant to avoid log(0).
_EPS = 1e-12


# ── Distance metrics ────────────────────────────────────────────────────


def _kld(a: dict[str, float], b: dict[str, float], all_keys: set[str]) -> float:
    """Kullback-Leibler divergence KLD(a || b) over *all_keys*.

    Zero entries in *a* contribute 0 (0 * log(0/x) = 0 by convention).
    Zero entries in *b* are smoothed with ``_EPS``.
    """
    total = 0.0
    for k in all_keys:
        ak = a.get(k, 0.0)
        if ak <= 0.0:
            continue
        bk = b.get(k, 0.0) + _EPS
        total += ak * math.log2(ak / bk)
    return total


def _jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon Divergence between two distributions.

    Both *p* and *q* map categories to probabilities.  Categories
    present in one but not the other are treated as 0.0.

    Returns a value in [0.0, 1.0].

    JSD(p, q) = 0.5 * KLD(p || m) + 0.5 * KLD(q || m)
    where m = 0.5 * (p + q).

    Special cases:

    * Both empty → 0.0
    * One empty, other non-empty → 1.0
    * Identical → 0.0
    """
    if not p and not q:
        return 0.0
    if not p or not q:
        return 1.0

    all_keys = set(p) | set(q)

    # Build midpoint distribution m = 0.5*(p + q)
    m: dict[str, float] = {}
    for k in all_keys:
        m[k] = 0.5 * (p.get(k, 0.0) + q.get(k, 0.0))

    jsd_val = 0.5 * _kld(p, m, all_keys) + 0.5 * _kld(q, m, all_keys)
    # Clamp to [0, 1] for numerical safety
    return max(0.0, min(1.0, jsd_val))


def _temporal_distance(
    baseline: TemporalPattern,
    recent: TemporalPattern,
) -> float:
    """Distance between two temporal patterns.

    Combines:

    * JSD of hourly distributions (when the agent is active): weight 0.6
    * Normalised rate deviation (how much activity changed): weight 0.4

    Rate deviation = |recent_mean - baseline_mean| / max(baseline_mean, 1.0),
    clamped to [0.0, 1.0].

    Returns a value in [0.0, 1.0].
    """
    # Hourly distribution distance
    # Convert int keys to str for _jsd compatibility
    p = {str(k): v for k, v in baseline.hourly_distribution.items()}
    q = {str(k): v for k, v in recent.hourly_distribution.items()}
    hourly_jsd = _jsd(p, q)

    # Rate deviation
    baseline_mean = baseline.actions_per_hour_mean
    recent_mean = recent.actions_per_hour_mean
    if baseline_mean <= 0.0 and recent_mean <= 0.0:
        rate_dev = 0.0
    else:
        rate_dev = abs(recent_mean - baseline_mean) / max(baseline_mean, 1.0)
        rate_dev = min(rate_dev, 1.0)

    return 0.6 * hourly_jsd + 0.4 * rate_dev


# ── DriftScore ──────────────────────────────────────────────────────────


@dataclass
class DriftScore:
    """Result of comparing recent behaviour against baseline.

    The overall drift is a weighted combination of per-distribution
    drift scores, where weights come from the archetype's drift_weights.
    """

    overall: float
    """Weighted composite drift.  0.0 = identical to baseline, 1.0 = completely different."""

    action_drift: float
    """How much the action-type distribution has changed."""

    target_drift: float
    """How much the target distribution has changed."""

    temporal_drift: float
    """How much the temporal pattern has changed."""

    outcome_drift: float
    """How much the governance-outcome distribution has changed."""

    confidence: float
    """How confident the drift measurement is, based on observation counts.
    Low confidence means the score is unreliable."""

    window_size: int
    """Number of observations in the recent window."""

    baseline_size: int
    """Number of observations in the baseline."""

    detail: str = ""
    """Human-readable summary of what drifted most."""

    @property
    def severity(self) -> str:
        """Categorise drift severity.

        Returns one of: ``"none"``, ``"low"``, ``"moderate"``, ``"high"``, ``"critical"``.
        """
        if self.overall < 0.05:
            return "none"
        if self.overall < 0.15:
            return "low"
        if self.overall < 0.35:
            return "moderate"
        if self.overall < 0.60:
            return "high"
        return "critical"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "overall": round(self.overall, 4),
            "action_drift": round(self.action_drift, 4),
            "target_drift": round(self.target_drift, 4),
            "temporal_drift": round(self.temporal_drift, 4),
            "outcome_drift": round(self.outcome_drift, 4),
            "confidence": round(self.confidence, 4),
            "window_size": self.window_size,
            "baseline_size": self.baseline_size,
            "severity": self.severity,
            "detail": self.detail,
        }


# ── DriftCalculator ─────────────────────────────────────────────────────


def _build_detail(
    baseline: BehavioralFingerprint,
    recent: BehavioralFingerprint,
    drifts: dict[str, float],
) -> str:
    """Build the human-readable detail string.

    Identifies the distribution that contributed most to drift and
    reports the two categories with the largest absolute change.
    """
    # Find the distribution with the largest drift
    dist_name = max(drifts, key=lambda k: drifts[k])
    dist_drift = drifts[dist_name]
    if dist_drift < 0.01:
        return "No significant drift detected"

    # Get the two distributions to compare
    dist_map = {
        "action": (baseline.action_distribution, recent.action_distribution),
        "target": (baseline.target_distribution, recent.target_distribution),
        "outcome": (baseline.outcome_distribution, recent.outcome_distribution),
    }

    if dist_name == "temporal":
        return f"Temporal pattern shifted (drift={dist_drift:.2f})"

    base_dist, recent_dist = dist_map.get(dist_name, ({}, {}))
    if not base_dist and not recent_dist:
        return f"{dist_name.title()} distribution shifted (drift={dist_drift:.2f})"

    # Compute deltas for all keys
    all_keys = set(base_dist) | set(recent_dist)
    deltas: list[tuple[str, float, float, float]] = []
    for k in all_keys:
        bv = base_dist.get(k, 0.0)
        rv = recent_dist.get(k, 0.0)
        deltas.append((k, bv, rv, rv - bv))

    # Sort by absolute delta descending
    deltas.sort(key=lambda x: abs(x[3]), reverse=True)

    parts: list[str] = []
    for cat, bv, rv, delta in deltas[:2]:
        direction = "increased" if delta > 0 else "decreased"
        parts.append(f"'{cat}' {direction} from {bv:.0%} to {rv:.0%}")

    label = dist_name.title()
    return f"{label} distribution shifted: {', '.join(parts)}"


class DriftCalculator:
    """Compares recent agent behaviour against baseline fingerprint.

    Pure computation — no side effects, no state mutation.  Takes two
    fingerprints (baseline and recent) and produces a :class:`DriftScore`.

    Usage::

        calculator = DriftCalculator()
        score = calculator.compare(baseline_fp, recent_fp, drift_weights)
    """

    def compare(
        self,
        baseline: BehavioralFingerprint,
        recent: BehavioralFingerprint,
        drift_weights: dict[str, float] | None = None,
    ) -> DriftScore:
        """Compare recent behaviour against baseline.

        Args:
            baseline: The agent's full behavioural fingerprint (all history).
            recent: A fingerprint built from only the last *N* actions.
            drift_weights: Optional per-distribution importance weights.
                Keys: ``"action"``, ``"target"``, ``"temporal"``, ``"outcome"``.
                Values: multipliers (1.0 = normal importance).
                If ``None``, all distributions weighted equally.

        Returns:
            :class:`DriftScore` with per-distribution and overall drift.
        """
        weights = drift_weights or {
            "action": 1.0,
            "target": 1.0,
            "temporal": 1.0,
            "outcome": 1.0,
        }

        # Edge cases
        if baseline.total_observations == 0 and recent.total_observations == 0:
            return DriftScore(
                overall=0.0,
                action_drift=0.0,
                target_drift=0.0,
                temporal_drift=0.0,
                outcome_drift=0.0,
                confidence=0.0,
                window_size=recent.total_observations,
                baseline_size=baseline.total_observations,
                detail="No observations in either fingerprint",
            )

        if baseline.total_observations == 0:
            return DriftScore(
                overall=1.0,
                action_drift=1.0,
                target_drift=1.0,
                temporal_drift=1.0,
                outcome_drift=1.0,
                confidence=0.0,
                window_size=recent.total_observations,
                baseline_size=0,
                detail="No baseline data to compare against",
            )

        if recent.total_observations == 0:
            return DriftScore(
                overall=0.0,
                action_drift=0.0,
                target_drift=0.0,
                temporal_drift=0.0,
                outcome_drift=0.0,
                confidence=0.0,
                window_size=0,
                baseline_size=baseline.total_observations,
                detail="No recent observations",
            )

        # Per-distribution drift
        action_drift = _jsd(baseline.action_distribution, recent.action_distribution)
        target_drift = _jsd(baseline.target_distribution, recent.target_distribution)
        temporal_drift = _temporal_distance(baseline.temporal_pattern, recent.temporal_pattern)
        outcome_drift = _jsd(baseline.outcome_distribution, recent.outcome_distribution)

        # Weighted overall
        drifts = {
            "action": action_drift,
            "target": target_drift,
            "temporal": temporal_drift,
            "outcome": outcome_drift,
        }
        weighted_sum = sum(drifts[k] * weights.get(k, 1.0) for k in drifts)
        weight_total = sum(weights.get(k, 1.0) for k in drifts)
        overall = weighted_sum / weight_total if weight_total > 0 else 0.0
        overall = max(0.0, min(1.0, overall))

        # Confidence — based on observation counts
        min_obs = min(baseline.total_observations, recent.total_observations)
        confidence = min(baseline.confidence, recent.confidence)
        # Reduce further for very small sample sizes
        if min_obs < 50:
            confidence *= min_obs / 50.0

        detail = _build_detail(baseline, recent, drifts)

        return DriftScore(
            overall=overall,
            action_drift=action_drift,
            target_drift=target_drift,
            temporal_drift=temporal_drift,
            outcome_drift=outcome_drift,
            confidence=confidence,
            window_size=recent.total_observations,
            baseline_size=baseline.total_observations,
            detail=detail,
        )

    def compare_against_prior(
        self,
        observed: BehavioralFingerprint,
        prior: ArchetypePrior,
    ) -> DriftScore:
        """Compare observed behaviour against the archetype prior.

        Useful when the agent has no established baseline yet —
        compare against what the archetype says is normal.

        Constructs a synthetic baseline fingerprint from the prior's
        distributions and compares using the prior's drift_weights.
        """
        synthetic = BehavioralFingerprint(agent_id=f"_prior_{prior.archetype_name}")
        synthetic.action_distribution = dict(prior.action_distribution)
        synthetic.target_distribution = dict(prior.target_categories)
        synthetic.outcome_distribution = dict(prior.outcome_expectations)
        # Give the synthetic baseline a nominal observation count so
        # confidence works and we don't hit the "empty baseline" edge case.
        synthetic.total_observations = prior.prior_weight

        return self.compare(
            baseline=synthetic,
            recent=observed,
            drift_weights=prior.drift_weights,
        )
