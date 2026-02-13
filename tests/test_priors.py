"""Tests for archetype behavioral priors and PriorRegistry."""

from __future__ import annotations

import pytest

from nomotic.priors import (
    ARCHETYPE_PRIOR_MAP,
    ArchetypePrior,
    PriorRegistry,
    TemporalProfile,
)
from nomotic.registry import BUILT_IN_ARCHETYPES


class TestArchetypePriorDistributions:
    """Verify that all 10 built-in priors have valid distributions."""

    @pytest.fixture
    def registry(self):
        return PriorRegistry.with_defaults()

    def test_registry_has_10_priors(self, registry):
        assert len(registry.priors) == 10

    @pytest.mark.parametrize("name", [
        "customer-experience",
        "operations-coordinator",
        "financial-analyst",
        "data-processor",
        "content-creator",
        "security-monitor",
        "research-analyst",
        "sales-agent",
        "executive-assistant",
        "healthcare-agent",
    ])
    def test_action_distribution_sums_to_one(self, registry, name):
        prior = registry.get(name)
        assert prior is not None, f"Prior '{name}' not found"
        total = sum(prior.action_distribution.values())
        assert total == pytest.approx(1.0), f"{name}: action_distribution sums to {total}"

    @pytest.mark.parametrize("name", [
        "customer-experience",
        "operations-coordinator",
        "financial-analyst",
        "data-processor",
        "content-creator",
        "security-monitor",
        "research-analyst",
        "sales-agent",
        "executive-assistant",
        "healthcare-agent",
    ])
    def test_target_categories_sums_to_one(self, registry, name):
        prior = registry.get(name)
        assert prior is not None
        total = sum(prior.target_categories.values())
        assert total == pytest.approx(1.0), f"{name}: target_categories sums to {total}"

    @pytest.mark.parametrize("name", [
        "customer-experience",
        "operations-coordinator",
        "financial-analyst",
        "data-processor",
        "content-creator",
        "security-monitor",
        "research-analyst",
        "sales-agent",
        "executive-assistant",
        "healthcare-agent",
    ])
    def test_outcome_expectations_sums_to_one(self, registry, name):
        prior = registry.get(name)
        assert prior is not None
        total = sum(prior.outcome_expectations.values())
        assert total == pytest.approx(1.0), f"{name}: outcome_expectations sums to {total}"

    @pytest.mark.parametrize("name", [
        "customer-experience",
        "operations-coordinator",
        "financial-analyst",
        "data-processor",
        "content-creator",
        "security-monitor",
        "research-analyst",
        "sales-agent",
        "executive-assistant",
        "healthcare-agent",
    ])
    def test_drift_weights_present(self, registry, name):
        prior = registry.get(name)
        assert prior is not None
        assert "action" in prior.drift_weights
        assert "target" in prior.drift_weights
        assert "temporal" in prior.drift_weights
        assert "outcome" in prior.drift_weights

    @pytest.mark.parametrize("name", [
        "customer-experience",
        "operations-coordinator",
        "financial-analyst",
        "data-processor",
        "content-creator",
        "security-monitor",
        "research-analyst",
        "sales-agent",
        "executive-assistant",
        "healthcare-agent",
    ])
    def test_temporal_profile_valid(self, registry, name):
        prior = registry.get(name)
        assert prior is not None
        tp = prior.temporal_profile
        assert isinstance(tp, TemporalProfile)
        assert isinstance(tp.peak_hours, set)
        assert isinstance(tp.active_hours, set)
        assert len(tp.expected_rate_range) == 2
        assert tp.expected_rate_range[0] <= tp.expected_rate_range[1]


class TestPriorRegistry:
    def test_with_defaults_loads_all_10(self):
        registry = PriorRegistry.with_defaults()
        assert len(registry.priors) == 10

    def test_get_direct_name(self):
        registry = PriorRegistry.with_defaults()
        prior = registry.get("customer-experience")
        assert prior is not None
        assert prior.archetype_name == "customer-experience"

    def test_get_for_agent_follows_mapping(self):
        registry = PriorRegistry.with_defaults()
        # "data-processing" maps to "data-processor"
        prior = registry.get_for_agent("data-processing")
        assert prior is not None
        assert prior.archetype_name == "data-processor"

    def test_get_for_agent_general_purpose_returns_none(self):
        registry = PriorRegistry.with_defaults()
        assert registry.get_for_agent("general-purpose") is None

    def test_get_unknown_archetype_returns_none(self):
        registry = PriorRegistry.with_defaults()
        assert registry.get("unknown-archetype") is None

    def test_custom_prior_registration(self):
        registry = PriorRegistry.with_defaults()
        custom = ArchetypePrior(
            archetype_name="custom-type",
            action_distribution={"read": 0.5, "write": 0.5},
            target_categories={"data": 1.0},
            temporal_profile=TemporalProfile(
                peak_hours={10}, active_hours={10},
                expected_rate_range=(10, 100),
                business_hours_only=True,
            ),
            outcome_expectations={"ALLOW": 1.0},
        )
        registry.register(custom)
        assert registry.get("custom-type") is not None
        assert registry.get("custom-type").archetype_name == "custom-type"

    def test_custom_prior_overrides_builtin(self):
        registry = PriorRegistry.with_defaults()
        original = registry.get("customer-experience")
        assert original is not None

        custom = ArchetypePrior(
            archetype_name="customer-experience",
            action_distribution={"read": 1.0},
            target_categories={"data": 1.0},
            temporal_profile=TemporalProfile(
                peak_hours=set(), active_hours=set(),
                expected_rate_range=(0, 0),
                business_hours_only=False,
            ),
            outcome_expectations={"ALLOW": 1.0},
        )
        registry.register(custom)
        updated = registry.get("customer-experience")
        assert updated is not None
        assert updated.action_distribution == {"read": 1.0}


class TestArchetypePriorMap:
    def test_covers_all_16_builtin_archetypes(self):
        for archetype_name in BUILT_IN_ARCHETYPES:
            assert archetype_name in ARCHETYPE_PRIOR_MAP, (
                f"Archetype '{archetype_name}' missing from ARCHETYPE_PRIOR_MAP"
            )

    def test_all_mapped_priors_exist(self):
        registry = PriorRegistry.with_defaults()
        for archetype_name, prior_name in ARCHETYPE_PRIOR_MAP.items():
            if prior_name is None:
                continue
            prior = registry.get(prior_name)
            assert prior is not None, (
                f"Archetype '{archetype_name}' maps to '{prior_name}' which doesn't exist"
            )

    def test_specific_mappings(self):
        assert ARCHETYPE_PRIOR_MAP["customer-experience"] == "customer-experience"
        assert ARCHETYPE_PRIOR_MAP["data-processing"] == "data-processor"
        assert ARCHETYPE_PRIOR_MAP["security-monitoring"] == "security-monitor"
        assert ARCHETYPE_PRIOR_MAP["content-generation"] == "content-creator"
        assert ARCHETYPE_PRIOR_MAP["clinical-support"] == "healthcare-agent"
        assert ARCHETYPE_PRIOR_MAP["research-assistant"] == "research-analyst"
        assert ARCHETYPE_PRIOR_MAP["sales-assistant"] == "sales-agent"
        assert ARCHETYPE_PRIOR_MAP["analytics"] == "research-analyst"
        assert ARCHETYPE_PRIOR_MAP["financial-transactions"] == "financial-analyst"
        assert ARCHETYPE_PRIOR_MAP["underwriting"] == "financial-analyst"
        assert ARCHETYPE_PRIOR_MAP["system-administration"] == "operations-coordinator"
        assert ARCHETYPE_PRIOR_MAP["workflow-orchestration"] == "operations-coordinator"
        assert ARCHETYPE_PRIOR_MAP["supply-chain"] == "operations-coordinator"
        assert ARCHETYPE_PRIOR_MAP["content-moderation"] == "security-monitor"
        assert ARCHETYPE_PRIOR_MAP["compliance-audit"] == "security-monitor"
        assert ARCHETYPE_PRIOR_MAP["general-purpose"] is None
