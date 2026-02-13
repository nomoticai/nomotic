"""Integration tests for behavioral fingerprints with GovernanceRuntime.

Tests that fingerprints form automatically from governance telemetry,
BehavioralConsistency uses them when available, and backward
compatibility is maintained.
"""

from __future__ import annotations

import time

import pytest

from nomotic.dimensions import BehavioralConsistency
from nomotic.fingerprint import BehavioralFingerprint
from nomotic.registry import ArchetypeDefinition, ArchetypeRegistry, BUILT_IN_ARCHETYPES
from nomotic.runtime import GovernanceRuntime, RuntimeConfig
from nomotic.types import Action, AgentContext, TrustProfile, Verdict


# ── Helpers ────────────────────────────────────────────────────────────────


def _action(
    agent_id: str = "agent-1",
    action_type: str = "read",
    target: str = "/api/data",
) -> Action:
    return Action(
        agent_id=agent_id,
        action_type=action_type,
        target=target,
        timestamp=time.time(),
    )


def _context(agent_id: str = "agent-1") -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        trust_profile=TrustProfile(agent_id=agent_id),
    )


# ── Runtime Integration Tests ─────────────────────────────────────────────


class TestRuntimeFingerprintIntegration:
    def test_fingerprints_enabled_by_default(self):
        runtime = GovernanceRuntime()
        assert runtime._fingerprint_observer is not None

    def test_fingerprints_disabled(self):
        config = RuntimeConfig(enable_fingerprints=False)
        runtime = GovernanceRuntime(config=config)
        assert runtime._fingerprint_observer is None

    def test_get_fingerprint_returns_none_when_disabled(self):
        config = RuntimeConfig(enable_fingerprints=False)
        runtime = GovernanceRuntime(config=config)
        assert runtime.get_fingerprint("agent-1") is None

    def test_evaluate_creates_fingerprint(self):
        runtime = GovernanceRuntime()
        action = _action()
        context = _context()
        runtime.evaluate(action, context)

        fp = runtime.get_fingerprint("agent-1")
        assert fp is not None
        assert fp.total_observations == 1
        assert fp.agent_id == "agent-1"

    def test_fingerprint_accumulates_over_evaluations(self):
        runtime = GovernanceRuntime()
        for i in range(50):
            action_type = "read" if i % 3 != 0 else "write"
            action = _action(action_type=action_type, target=f"/api/resource-{i % 5}")
            context = _context()
            runtime.evaluate(action, context)

        fp = runtime.get_fingerprint("agent-1")
        assert fp is not None
        assert fp.total_observations == 50
        assert "read" in fp.action_distribution
        assert "write" in fp.action_distribution
        assert sum(fp.action_distribution.values()) == pytest.approx(1.0)

    def test_fingerprint_has_correct_distributions(self):
        runtime = GovernanceRuntime()
        # 8 reads, 2 writes
        for _ in range(8):
            runtime.evaluate(_action(action_type="read"), _context())
        for _ in range(2):
            runtime.evaluate(_action(action_type="write"), _context())

        fp = runtime.get_fingerprint("agent-1")
        assert fp is not None
        assert fp.action_distribution["read"] == pytest.approx(0.8)
        assert fp.action_distribution["write"] == pytest.approx(0.2)

    def test_fingerprint_accessible_via_get_fingerprint(self):
        runtime = GovernanceRuntime()
        runtime.evaluate(_action(), _context())
        fp = runtime.get_fingerprint("agent-1")
        assert isinstance(fp, BehavioralFingerprint)

    def test_multiple_agents_get_separate_fingerprints(self):
        runtime = GovernanceRuntime()
        runtime.evaluate(_action(agent_id="agent-1"), _context("agent-1"))
        runtime.evaluate(_action(agent_id="agent-2", action_type="write"), _context("agent-2"))

        fp1 = runtime.get_fingerprint("agent-1")
        fp2 = runtime.get_fingerprint("agent-2")
        assert fp1 is not None
        assert fp2 is not None
        assert fp1.agent_id == "agent-1"
        assert fp2.agent_id == "agent-2"
        assert fp1.action_distribution != fp2.action_distribution


# ── BehavioralConsistency Enhancement Tests ───────────────────────────────


class TestBehavioralConsistencyLegacy:
    """Tests that BehavioralConsistency without fingerprints behaves identically."""

    def test_first_action_scores_0_7(self):
        dim = BehavioralConsistency()
        score = dim.evaluate(_action(), _context())
        assert score.score == 0.7
        assert "First action" in score.reasoning

    def test_seen_action_scores_1_0(self):
        dim = BehavioralConsistency()
        dim.evaluate(_action(action_type="read"), _context())
        score = dim.evaluate(_action(action_type="read"), _context())
        assert score.score == 1.0

    def test_novel_action_scores_0_5(self):
        dim = BehavioralConsistency()
        dim.evaluate(_action(action_type="read"), _context())
        score = dim.evaluate(_action(action_type="write"), _context())
        assert score.score == 0.5

    def test_no_fingerprint_accessor_uses_legacy(self):
        dim = BehavioralConsistency()
        # No set_fingerprint_accessor called
        dim.evaluate(_action(action_type="read"), _context())
        score = dim.evaluate(_action(action_type="read"), _context())
        assert score.score == 1.0
        assert "consistent with history" in score.reasoning


class TestBehavioralConsistencyEnhanced:
    """Tests BehavioralConsistency with fingerprint data."""

    def test_falls_back_when_insufficient_data(self):
        """With < 10 observations, should use legacy behavior."""
        runtime = GovernanceRuntime()
        # Do 5 evaluations (below threshold of 10)
        for _ in range(5):
            runtime.evaluate(_action(action_type="read"), _context())

        dim = runtime.registry.get("behavioral_consistency")
        score = dim.evaluate(_action(action_type="read"), _context())
        # Should still use legacy (seen action)
        assert score.score == 1.0

    def test_high_score_for_frequent_action(self):
        """Common action type (>= 5%) should score 1.0."""
        runtime = GovernanceRuntime()
        # Build up enough observations
        for _ in range(20):
            runtime.evaluate(_action(action_type="read"), _context())

        dim = runtime.registry.get("behavioral_consistency")
        score = dim.evaluate(_action(action_type="read"), _context())
        assert score.score == 1.0
        assert "typical activity" in score.reasoning

    def test_low_score_for_never_seen_action(self):
        """Action type with no precedent should score 0.4."""
        runtime = GovernanceRuntime()
        # Build up fingerprint with reads
        for _ in range(20):
            runtime.evaluate(_action(action_type="read"), _context())

        dim = runtime.registry.get("behavioral_consistency")
        # "delete" was never seen in any context
        score = dim.evaluate(_action(action_type="delete"), _context())
        assert score.score == 0.4
        assert "no precedent" in score.reasoning

    def test_fingerprint_disabled_uses_legacy(self):
        """With enable_fingerprints=False, should behave exactly like legacy."""
        config = RuntimeConfig(enable_fingerprints=False)
        runtime = GovernanceRuntime(config=config)

        dim = runtime.registry.get("behavioral_consistency")
        # First action
        score1 = dim.evaluate(_action(action_type="read"), _context())
        assert score1.score == 0.7

        # Same action
        score2 = dim.evaluate(_action(action_type="read"), _context())
        assert score2.score == 1.0

        # Novel action
        score3 = dim.evaluate(_action(action_type="write"), _context())
        assert score3.score == 0.5


# ── ArchetypeDefinition Extension Tests ───────────────────────────────────


class TestArchetypeDefinitionPriorName:
    def test_prior_name_field_exists(self):
        defn = ArchetypeDefinition(
            name="test", description="test", category="test",
            builtin=False, prior_name="test-prior",
        )
        assert defn.prior_name == "test-prior"

    def test_prior_name_defaults_to_none(self):
        defn = ArchetypeDefinition(
            name="test", description="test", category="test", builtin=False,
        )
        assert defn.prior_name is None

    def test_builtin_archetypes_have_prior_name(self):
        registry = ArchetypeRegistry.with_defaults()
        # customer-experience should map to itself
        ce = registry.get("customer-experience")
        assert ce is not None
        assert ce.prior_name == "customer-experience"

        # data-processing maps to data-processor
        dp = registry.get("data-processing")
        assert dp is not None
        assert dp.prior_name == "data-processor"

        # general-purpose has no prior
        gp = registry.get("general-purpose")
        assert gp is not None
        assert gp.prior_name is None

    def test_all_16_builtins_have_prior_name_set(self):
        registry = ArchetypeRegistry.with_defaults()
        for name in BUILT_IN_ARCHETYPES:
            defn = registry.get(name)
            assert defn is not None, f"Built-in archetype '{name}' not found"
            # prior_name should be set (even if None for general-purpose)
            expected = BUILT_IN_ARCHETYPES[name].get("prior_name")
            assert defn.prior_name == expected, (
                f"Archetype '{name}': expected prior_name={expected}, got {defn.prior_name}"
            )


# ── API Fingerprint Endpoint Tests ────────────────────────────────────────


class TestFingerprintAPI:
    """Test the GET /v1/fingerprint/{agent_id} endpoint."""

    @pytest.fixture
    def server_with_runtime(self):
        """Create a test server with a runtime that has fingerprint data."""
        import json
        import threading
        from http.server import HTTPServer

        from nomotic.api import NomoticAPIServer, _Handler, _ServerContext, NomoticHTTPServer
        from nomotic.authority import CertificateAuthority
        from nomotic.keys import SigningKey
        from nomotic.store import MemoryCertificateStore

        sk, _vk = SigningKey.generate()
        ca = CertificateAuthority(
            issuer_id="test-issuer",
            signing_key=sk,
            store=MemoryCertificateStore(),
        )
        runtime = GovernanceRuntime()

        # Feed some actions to build a fingerprint
        for _ in range(5):
            runtime.evaluate(_action(), _context())

        server = NomoticHTTPServer(("127.0.0.1", 0), _Handler)
        server.ctx = _ServerContext(
            ca=ca,
            archetype_registry=ArchetypeRegistry.with_defaults(),
            zone_validator=__import__("nomotic.registry", fromlist=["ZoneValidator"]).ZoneValidator(),
            org_registry=__import__("nomotic.registry", fromlist=["OrganizationRegistry"]).OrganizationRegistry(),
            runtime=runtime,
        )

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield server
        server.shutdown()

    def test_fingerprint_endpoint_returns_data(self, server_with_runtime):
        import json
        import urllib.request

        host, port = server_with_runtime.server_address
        url = f"http://{host}:{port}/v1/fingerprint/agent-1"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        assert data["agent_id"] == "agent-1"
        assert data["total_observations"] == 5
        assert "action_distribution" in data
        assert "target_distribution" in data
        assert "temporal_pattern" in data
        assert "outcome_distribution" in data
        assert "confidence" in data

    def test_fingerprint_endpoint_404_for_unknown(self, server_with_runtime):
        import urllib.request
        import urllib.error

        host, port = server_with_runtime.server_address
        url = f"http://{host}:{port}/v1/fingerprint/unknown-agent"
        req = urllib.request.Request(url)
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 404

    def test_fingerprint_endpoint_without_runtime(self):
        """Server without a runtime should return 404 for fingerprint requests."""
        import json
        import threading
        import urllib.request
        import urllib.error

        from nomotic.api import _Handler, _ServerContext, NomoticHTTPServer
        from nomotic.authority import CertificateAuthority
        from nomotic.keys import SigningKey
        from nomotic.store import MemoryCertificateStore

        sk, _vk = SigningKey.generate()
        ca = CertificateAuthority(
            issuer_id="test-issuer",
            signing_key=sk,
            store=MemoryCertificateStore(),
        )

        server = NomoticHTTPServer(("127.0.0.1", 0), _Handler)
        server.ctx = _ServerContext(
            ca=ca,
            archetype_registry=ArchetypeRegistry.with_defaults(),
            zone_validator=__import__("nomotic.registry", fromlist=["ZoneValidator"]).ZoneValidator(),
            org_registry=__import__("nomotic.registry", fromlist=["OrganizationRegistry"]).OrganizationRegistry(),
            runtime=None,
        )

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            url = f"http://{host}:{port}/v1/fingerprint/agent-1"
            req = urllib.request.Request(url)
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)
            assert exc_info.value.code == 404
        finally:
            server.shutdown()


# ── Backward Compatibility Tests ─────────────────────────────────────────


class TestBackwardCompatibility:
    def test_runtime_without_fingerprints_works(self):
        """GovernanceRuntime with fingerprints disabled should work identically."""
        config = RuntimeConfig(enable_fingerprints=False)
        runtime = GovernanceRuntime(config=config)
        action = _action()
        context = _context()
        verdict = runtime.evaluate(action, context)
        assert verdict.verdict in (Verdict.ALLOW, Verdict.DENY, Verdict.MODIFY, Verdict.ESCALATE)

    def test_behavioral_consistency_without_fingerprint_unchanged(self):
        """BehavioralConsistency without fingerprint accessor returns same scores."""
        dim = BehavioralConsistency()
        ctx = _context()

        # First action → 0.7
        s1 = dim.evaluate(_action(action_type="read"), ctx)
        assert s1.score == 0.7

        # Seen action → 1.0
        s2 = dim.evaluate(_action(action_type="read"), ctx)
        assert s2.score == 1.0

        # Novel action → 0.5
        s3 = dim.evaluate(_action(action_type="write"), ctx)
        assert s3.score == 0.5

    def test_archetype_definition_backward_compatible(self):
        """ArchetypeDefinition with prior_name=None should work like before."""
        defn = ArchetypeDefinition(
            name="test", description="test", category="test", builtin=False,
        )
        assert defn.prior_name is None
        # Can still use all existing fields
        assert defn.name == "test"
        assert defn.description == "test"
        assert defn.category == "test"
        assert defn.builtin is False
