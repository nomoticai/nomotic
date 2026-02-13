"""Archetype behavioral priors — expected operational signatures.

Each archetype ships with a behavioral prior: the expected operational
signature for that category of agent.  Priors serve as the starting
fingerprint before the agent has any history.  As real observations
accumulate, they gradually replace the prior.

Ten archetypes have full behavioral priors.  The remaining six built-in
archetype names map to the closest prior (or to ``None`` for
general-purpose).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ArchetypePrior",
    "ARCHETYPE_PRIOR_MAP",
    "PriorRegistry",
    "TemporalProfile",
]


@dataclass
class TemporalProfile:
    """Expected temporal behavior for an archetype."""

    peak_hours: set[int]  # Hours of highest expected activity
    active_hours: set[int]  # All hours where activity is expected
    expected_rate_range: tuple[float, float]  # (min, max) actions per hour
    business_hours_only: bool  # If True, overnight activity is anomalous

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "peak_hours": sorted(self.peak_hours),
            "active_hours": sorted(self.active_hours),
            "expected_rate_range": list(self.expected_rate_range),
            "business_hours_only": self.business_hours_only,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalProfile:
        """Deserialize from dict."""
        return cls(
            peak_hours=set(data["peak_hours"]),
            active_hours=set(data["active_hours"]),
            expected_rate_range=tuple(data["expected_rate_range"]),
            business_hours_only=data["business_hours_only"],
        )


@dataclass
class ArchetypePrior:
    """Expected behavioral signature for an archetype.

    These are first-principles estimates of what agents of this type
    typically do. They serve as the starting fingerprint and the
    baseline for drift detection until the agent builds its own history.
    """

    archetype_name: str

    action_distribution: dict[str, float]
    # Expected distribution of action types.
    # Keys are action type strings, values are fractions summing to 1.0.

    target_categories: dict[str, float]
    # Expected distribution of target categories.
    # Keys are target category patterns (used for prefix matching),
    # values are fractions summing to 1.0.

    temporal_profile: TemporalProfile
    # Expected temporal behavior.

    outcome_expectations: dict[str, float]
    # Expected governance outcome distribution.

    # Which distributions are "load-bearing" for this archetype.
    # Drift in load-bearing distributions matters more.
    # Keys are distribution names, values are weight multipliers (1.0 = normal).
    drift_weights: dict[str, float] = field(default_factory=lambda: {
        "action": 1.0,
        "target": 1.0,
        "temporal": 1.0,
        "outcome": 1.0,
    })

    prior_weight: int = 50
    # How many synthetic observations the prior represents.
    # Higher = prior takes longer to fade. Lower = faster adaptation.


# ── Built-in archetype priors ────────────────────────────────────────────

_BUSINESS_HOURS = set(range(8, 18))
_ALL_HOURS = set(range(24))


def _build_customer_experience() -> ArchetypePrior:
    return ArchetypePrior(
        archetype_name="customer-experience",
        action_distribution={
            "read": 0.55, "write": 0.20, "send": 0.12,
            "query": 0.08, "escalate": 0.05,
        },
        target_categories={
            "customer": 0.50, "order": 0.25,
            "product": 0.15, "communication": 0.10,
        },
        temporal_profile=TemporalProfile(
            peak_hours={10, 11, 12, 13, 14},
            active_hours=_BUSINESS_HOURS,
            expected_rate_range=(150.0, 400.0),
            business_hours_only=True,
        ),
        outcome_expectations={
            "ALLOW": 0.92, "MODIFY": 0.04,
            "ESCALATE": 0.03, "DENY": 0.01,
        },
        drift_weights={
            "action": 1.2, "target": 1.0,
            "temporal": 0.8, "outcome": 1.5,
        },
    )


def _build_operations_coordinator() -> ArchetypePrior:
    return ArchetypePrior(
        archetype_name="operations-coordinator",
        action_distribution={
            "query": 0.30, "write": 0.25, "send": 0.20,
            "read": 0.15, "create": 0.10,
        },
        target_categories={
            "workflow": 0.30, "agent": 0.25, "status": 0.20,
            "report": 0.15, "config": 0.10,
        },
        temporal_profile=TemporalProfile(
            peak_hours={9, 10, 11, 14, 15, 16},
            active_hours=_BUSINESS_HOURS,
            expected_rate_range=(100.0, 300.0),
            business_hours_only=True,
        ),
        outcome_expectations={
            "ALLOW": 0.95, "MODIFY": 0.03,
            "ESCALATE": 0.01, "DENY": 0.01,
        },
        drift_weights={
            "action": 1.0, "target": 1.3,
            "temporal": 1.0, "outcome": 1.0,
        },
    )


def _build_financial_analyst() -> ArchetypePrior:
    return ArchetypePrior(
        archetype_name="financial-analyst",
        action_distribution={
            "read": 0.45, "query": 0.30, "write": 0.15,
            "escalate": 0.07, "send": 0.03,
        },
        target_categories={
            "financial": 0.40, "market": 0.25, "report": 0.20,
            "compliance": 0.10, "client": 0.05,
        },
        temporal_profile=TemporalProfile(
            peak_hours={8, 9, 10},
            active_hours=_BUSINESS_HOURS,
            expected_rate_range=(50.0, 200.0),
            business_hours_only=True,
        ),
        outcome_expectations={
            "ALLOW": 0.88, "MODIFY": 0.05,
            "ESCALATE": 0.05, "DENY": 0.02,
        },
        drift_weights={
            "action": 1.5, "target": 1.5,
            "temporal": 1.2, "outcome": 1.3,
        },
    )


def _build_data_processor() -> ArchetypePrior:
    return ArchetypePrior(
        archetype_name="data-processor",
        action_distribution={
            "read": 0.40, "write": 0.35, "query": 0.15,
            "create": 0.08, "delete": 0.02,
        },
        target_categories={
            "data": 0.50, "storage": 0.25,
            "pipeline": 0.15, "cache": 0.10,
        },
        temporal_profile=TemporalProfile(
            peak_hours=set(),
            active_hours=_ALL_HOURS,
            expected_rate_range=(500.0, 5000.0),
            business_hours_only=False,
        ),
        outcome_expectations={
            "ALLOW": 0.97, "MODIFY": 0.01,
            "ESCALATE": 0.01, "DENY": 0.01,
        },
        drift_weights={
            "action": 1.0, "target": 1.2,
            "temporal": 0.5, "outcome": 1.5,
        },
    )


def _build_content_creator() -> ArchetypePrior:
    return ArchetypePrior(
        archetype_name="content-creator",
        action_distribution={
            "create": 0.35, "write": 0.25, "read": 0.20,
            "query": 0.15, "send": 0.05,
        },
        target_categories={
            "content": 0.40, "media": 0.20, "template": 0.15,
            "brand": 0.15, "publish": 0.10,
        },
        temporal_profile=TemporalProfile(
            peak_hours={10, 11, 12, 13, 14, 15, 16},
            active_hours=_BUSINESS_HOURS,
            expected_rate_range=(30.0, 150.0),
            business_hours_only=True,
        ),
        outcome_expectations={
            "ALLOW": 0.90, "MODIFY": 0.06,
            "ESCALATE": 0.02, "DENY": 0.02,
        },
        drift_weights={
            "action": 0.8, "target": 1.0,
            "temporal": 0.6, "outcome": 1.0,
        },
    )


def _build_security_monitor() -> ArchetypePrior:
    return ArchetypePrior(
        archetype_name="security-monitor",
        action_distribution={
            "read": 0.50, "query": 0.30, "escalate": 0.10,
            "write": 0.08, "send": 0.02,
        },
        target_categories={
            "logs": 0.35, "network": 0.25, "auth": 0.20,
            "incident": 0.15, "config": 0.05,
        },
        temporal_profile=TemporalProfile(
            peak_hours=set(),
            active_hours=_ALL_HOURS,
            expected_rate_range=(200.0, 1000.0),
            business_hours_only=False,
        ),
        outcome_expectations={
            "ALLOW": 0.96, "MODIFY": 0.01,
            "ESCALATE": 0.02, "DENY": 0.01,
        },
        drift_weights={
            "action": 1.3, "target": 1.5,
            "temporal": 0.8, "outcome": 1.2,
        },
    )


def _build_research_analyst() -> ArchetypePrior:
    return ArchetypePrior(
        archetype_name="research-analyst",
        action_distribution={
            "read": 0.40, "query": 0.35, "write": 0.12,
            "create": 0.08, "send": 0.05,
        },
        target_categories={
            "research": 0.30, "external": 0.25, "data": 0.20,
            "document": 0.15, "reference": 0.10,
        },
        temporal_profile=TemporalProfile(
            peak_hours={10, 11, 12, 13, 14, 15, 16},
            active_hours=_BUSINESS_HOURS | {18, 19},
            expected_rate_range=(50.0, 200.0),
            business_hours_only=False,
        ),
        outcome_expectations={
            "ALLOW": 0.93, "MODIFY": 0.03,
            "ESCALATE": 0.02, "DENY": 0.02,
        },
        drift_weights={
            "action": 0.8, "target": 0.8,
            "temporal": 0.6, "outcome": 1.0,
        },
    )


def _build_sales_agent() -> ArchetypePrior:
    return ArchetypePrior(
        archetype_name="sales-agent",
        action_distribution={
            "read": 0.30, "send": 0.25, "write": 0.20,
            "query": 0.15, "create": 0.10,
        },
        target_categories={
            "prospect": 0.30, "client": 0.25, "product": 0.20,
            "communication": 0.15, "crm": 0.10,
        },
        temporal_profile=TemporalProfile(
            peak_hours={10, 11, 12, 14, 15, 16},
            active_hours=_BUSINESS_HOURS,
            expected_rate_range=(100.0, 300.0),
            business_hours_only=True,
        ),
        outcome_expectations={
            "ALLOW": 0.93, "MODIFY": 0.03,
            "ESCALATE": 0.02, "DENY": 0.02,
        },
        drift_weights={
            "action": 1.0, "target": 1.2,
            "temporal": 0.8, "outcome": 1.0,
        },
    )


def _build_executive_assistant() -> ArchetypePrior:
    return ArchetypePrior(
        archetype_name="executive-assistant",
        action_distribution={
            "read": 0.30, "write": 0.25, "send": 0.20,
            "create": 0.15, "query": 0.10,
        },
        target_categories={
            "calendar": 0.30, "communication": 0.25, "document": 0.20,
            "travel": 0.15, "contact": 0.10,
        },
        temporal_profile=TemporalProfile(
            peak_hours={8, 9, 10},
            active_hours=_BUSINESS_HOURS,
            expected_rate_range=(80.0, 250.0),
            business_hours_only=True,
        ),
        outcome_expectations={
            "ALLOW": 0.95, "MODIFY": 0.02,
            "ESCALATE": 0.02, "DENY": 0.01,
        },
        drift_weights={
            "action": 0.8, "target": 1.0,
            "temporal": 1.0, "outcome": 1.0,
        },
    )


def _build_healthcare_agent() -> ArchetypePrior:
    return ArchetypePrior(
        archetype_name="healthcare-agent",
        action_distribution={
            "read": 0.40, "query": 0.25, "write": 0.15,
            "escalate": 0.12, "send": 0.08,
        },
        target_categories={
            "patient": 0.35, "clinical": 0.25, "schedule": 0.20,
            "pharmacy": 0.10, "insurance": 0.10,
        },
        temporal_profile=TemporalProfile(
            peak_hours={8, 9, 10, 11, 12},
            active_hours=_ALL_HOURS,
            expected_rate_range=(100.0, 400.0),
            business_hours_only=False,
        ),
        outcome_expectations={
            "ALLOW": 0.87, "MODIFY": 0.05,
            "ESCALATE": 0.06, "DENY": 0.02,
        },
        drift_weights={
            "action": 1.3, "target": 1.5,
            "temporal": 0.8, "outcome": 1.5,
        },
    )


# ── Archetype-to-prior mapping ──────────────────────────────────────────

ARCHETYPE_PRIOR_MAP: dict[str, str | None] = {
    # Direct mappings (archetype name == prior name or close variant)
    "customer-experience": "customer-experience",
    "data-processing": "data-processor",
    "security-monitoring": "security-monitor",
    "content-generation": "content-creator",
    "clinical-support": "healthcare-agent",
    "research-assistant": "research-analyst",

    # Mapped to nearest prior
    "sales-assistant": "sales-agent",
    "analytics": "research-analyst",
    "financial-transactions": "financial-analyst",
    "underwriting": "financial-analyst",
    "system-administration": "operations-coordinator",
    "workflow-orchestration": "operations-coordinator",
    "supply-chain": "operations-coordinator",
    "content-moderation": "security-monitor",
    "compliance-audit": "security-monitor",
    "general-purpose": None,  # No prior — starts with empty fingerprint
}


# ── Prior Registry ──────────────────────────────────────────────────────


_BUILTIN_PRIOR_BUILDERS = [
    _build_customer_experience,
    _build_operations_coordinator,
    _build_financial_analyst,
    _build_data_processor,
    _build_content_creator,
    _build_security_monitor,
    _build_research_analyst,
    _build_sales_agent,
    _build_executive_assistant,
    _build_healthcare_agent,
]


class PriorRegistry:
    """Registry of archetype behavioral priors.

    Ships with priors for 10 archetypes. Custom priors can be registered.
    Archetypes without explicit priors map to the nearest built-in prior
    via ARCHETYPE_PRIOR_MAP.
    """

    def __init__(self) -> None:
        self._priors: dict[str, ArchetypePrior] = {}

    @classmethod
    def with_defaults(cls) -> PriorRegistry:
        """Create registry with all 10 built-in priors."""
        registry = cls()
        for builder in _BUILTIN_PRIOR_BUILDERS:
            prior = builder()
            registry._priors[prior.archetype_name] = prior
        return registry

    def register(self, prior: ArchetypePrior) -> None:
        """Register a custom archetype prior."""
        self._priors[prior.archetype_name] = prior

    def get(self, archetype_name: str) -> ArchetypePrior | None:
        """Get the prior for an archetype.

        Checks direct registration first, then ARCHETYPE_PRIOR_MAP.
        Returns None for "general-purpose" or unmapped archetypes.
        """
        # Direct match
        if archetype_name in self._priors:
            return self._priors[archetype_name]

        # Follow the mapping
        mapped = ARCHETYPE_PRIOR_MAP.get(archetype_name)
        if mapped is None:
            return None
        return self._priors.get(mapped)

    def get_for_agent(self, archetype_name: str) -> ArchetypePrior | None:
        """Same as get() but follows the archetype mapping."""
        return self.get(archetype_name)

    @property
    def priors(self) -> dict[str, ArchetypePrior]:
        """All registered priors."""
        return dict(self._priors)
