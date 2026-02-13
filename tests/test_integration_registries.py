"""Integration tests for CertificateAuthority with validation registries."""

from __future__ import annotations

import warnings

import pytest

from nomotic.authority import CertificateAuthority
from nomotic.keys import SigningKey
from nomotic.registry import (
    ArchetypeRegistry,
    OrganizationRegistry,
    OrgStatus,
    ZoneValidator,
)
from nomotic.runtime import GovernanceRuntime
from nomotic.store import MemoryCertificateStore


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_sk() -> tuple[SigningKey, str]:
    """Generate a signing key and return (sk, issuer_fingerprint)."""
    sk, vk = SigningKey.generate()
    return sk, vk.fingerprint()


# ═══════════════════════════════════════════════════════════════════════
# CertificateAuthority + ArchetypeRegistry
# ═══════════════════════════════════════════════════════════════════════


class TestCAWithArchetypeRegistry:
    def test_issue_known_archetype_passes(self) -> None:
        sk, fp = _make_sk()
        reg = ArchetypeRegistry.with_defaults()
        ca = CertificateAuthority(
            "test", sk, archetype_registry=reg,
        )
        cert, _ = ca.issue("a1", "customer-experience", "org", "global")
        assert cert.archetype == "customer-experience"

    def test_issue_unknown_archetype_nonstrict_warns(self) -> None:
        sk, fp = _make_sk()
        reg = ArchetypeRegistry.with_defaults(strict=False)
        ca = CertificateAuthority(
            "test", sk, archetype_registry=reg,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cert, _ = ca.issue("a1", "my-custom-agent", "org", "global")
            assert len(w) >= 1
            assert "not registered" in str(w[0].message)
        assert cert.archetype == "my-custom-agent"

    def test_issue_unknown_archetype_strict_raises(self) -> None:
        sk, fp = _make_sk()
        reg = ArchetypeRegistry.with_defaults(strict=True)
        ca = CertificateAuthority(
            "test", sk, archetype_registry=reg,
        )
        with pytest.raises(ValueError, match="unknown archetype"):
            ca.issue("a1", "my-custom-agent", "org", "global")

    def test_issue_invalid_archetype_format_raises(self) -> None:
        sk, fp = _make_sk()
        reg = ArchetypeRegistry.with_defaults()
        ca = CertificateAuthority(
            "test", sk, archetype_registry=reg,
        )
        with pytest.raises(ValueError, match="Invalid archetype"):
            ca.issue("a1", "Bad Name!", "org", "global")

    def test_issue_with_suggestion_in_error(self) -> None:
        sk, fp = _make_sk()
        reg = ArchetypeRegistry.with_defaults(strict=True)
        ca = CertificateAuthority(
            "test", sk, archetype_registry=reg,
        )
        with pytest.raises(ValueError, match="did you mean"):
            ca.issue("a1", "customer-experince", "org", "global")


# ═══════════════════════════════════════════════════════════════════════
# CertificateAuthority + ZoneValidator
# ═══════════════════════════════════════════════════════════════════════


class TestCAWithZoneValidator:
    def test_issue_valid_zone_passes(self) -> None:
        sk, fp = _make_sk()
        zv = ZoneValidator()
        ca = CertificateAuthority("test", sk, zone_validator=zv)
        cert, _ = ca.issue("a1", "arch", "org", "global/us/prod")
        assert cert.zone_path == "global/us/prod"

    def test_issue_invalid_zone_raises(self) -> None:
        sk, fp = _make_sk()
        zv = ZoneValidator()
        ca = CertificateAuthority("test", sk, zone_validator=zv)
        with pytest.raises(ValueError, match="Invalid zone path"):
            ca.issue("a1", "arch", "org", "/bad/path/")

    def test_issue_double_slash_zone_raises(self) -> None:
        sk, fp = _make_sk()
        zv = ZoneValidator()
        ca = CertificateAuthority("test", sk, zone_validator=zv)
        with pytest.raises(ValueError, match="Invalid zone path"):
            ca.issue("a1", "arch", "org", "global//us")

    def test_issue_empty_zone_raises(self) -> None:
        sk, fp = _make_sk()
        zv = ZoneValidator()
        ca = CertificateAuthority("test", sk, zone_validator=zv)
        with pytest.raises(ValueError, match="Invalid zone path"):
            ca.issue("a1", "arch", "org", "")


# ═══════════════════════════════════════════════════════════════════════
# CertificateAuthority + OrganizationRegistry
# ═══════════════════════════════════════════════════════════════════════


class TestCAWithOrgRegistry:
    def test_issue_registered_org_passes(self) -> None:
        sk, fp = _make_sk()
        org_reg = OrganizationRegistry()
        org_reg.register("acme-corp", fp)
        ca = CertificateAuthority("test", sk, org_registry=org_reg)
        cert, _ = ca.issue("a1", "arch", "acme-corp", "global")
        assert cert.organization == "acme-corp"

    def test_issue_unregistered_org_raises(self) -> None:
        sk, fp = _make_sk()
        org_reg = OrganizationRegistry()
        ca = CertificateAuthority("test", sk, org_registry=org_reg)
        with pytest.raises(ValueError, match="not registered"):
            ca.issue("a1", "arch", "unknown-org", "global")

    def test_issue_wrong_issuer_raises(self) -> None:
        sk, fp = _make_sk()
        org_reg = OrganizationRegistry()
        org_reg.register("acme-corp", "SHA256:wrong-fingerprint")
        ca = CertificateAuthority("test", sk, org_registry=org_reg)
        with pytest.raises(ValueError, match="not authorized"):
            ca.issue("a1", "arch", "acme-corp", "global")

    def test_issue_suspended_org_raises(self) -> None:
        sk, fp = _make_sk()
        org_reg = OrganizationRegistry()
        org_reg.register("acme-corp", fp)
        org_reg.suspend("acme-corp", "testing")
        ca = CertificateAuthority("test", sk, org_registry=org_reg)
        with pytest.raises(ValueError, match="SUSPENDED"):
            ca.issue("a1", "arch", "acme-corp", "global")


# ═══════════════════════════════════════════════════════════════════════
# Backward compatibility — no registries
# ═══════════════════════════════════════════════════════════════════════


class TestCAWithoutRegistries:
    def test_issue_without_registries_no_validation(self) -> None:
        sk, fp = _make_sk()
        ca = CertificateAuthority("test", sk)
        cert, _ = ca.issue("a1", "anything-goes", "any-org", "any/path")
        assert cert.archetype == "anything-goes"
        assert cert.organization == "any-org"

    def test_issue_invalid_names_pass_without_registries(self) -> None:
        sk, fp = _make_sk()
        ca = CertificateAuthority("test", sk)
        cert, _ = ca.issue("a1", "Bad Name!", "Bad Org!", "/bad//path/")
        assert cert.archetype == "Bad Name!"

    def test_existing_tests_unaffected(self) -> None:
        """Ensure the default CA constructor is backward compatible."""
        sk, _ = SigningKey.generate()
        ca = CertificateAuthority("test", sk)
        cert, agent_sk = ca.issue("agent-1", "arch", "org", "zone")
        assert cert.status.name == "ACTIVE"
        result = ca.verify_certificate(cert)
        assert result.valid is True


# ═══════════════════════════════════════════════════════════════════════
# Combined registries
# ═══════════════════════════════════════════════════════════════════════


class TestCAWithAllRegistries:
    def test_all_registries_valid(self) -> None:
        sk, fp = _make_sk()
        arch_reg = ArchetypeRegistry.with_defaults()
        zv = ZoneValidator()
        org_reg = OrganizationRegistry()
        org_reg.register("acme-corp", fp)

        ca = CertificateAuthority(
            "test", sk,
            archetype_registry=arch_reg,
            zone_validator=zv,
            org_registry=org_reg,
        )
        cert, _ = ca.issue("a1", "analytics", "acme-corp", "global/us")
        assert cert.archetype == "analytics"
        assert cert.organization == "acme-corp"
        assert cert.zone_path == "global/us"

    def test_archetype_checked_before_zone(self) -> None:
        """If archetype fails, zone validation shouldn't even matter."""
        sk, fp = _make_sk()
        arch_reg = ArchetypeRegistry.with_defaults(strict=True)
        zv = ZoneValidator()

        ca = CertificateAuthority(
            "test", sk,
            archetype_registry=arch_reg,
            zone_validator=zv,
        )
        with pytest.raises(ValueError, match="unknown archetype"):
            ca.issue("a1", "nonexistent-archetype", "org", "global")


# ═══════════════════════════════════════════════════════════════════════
# Runtime integration
# ═══════════════════════════════════════════════════════════════════════


class TestRuntimeRegistries:
    def test_runtime_has_archetype_registry(self) -> None:
        rt = GovernanceRuntime()
        reg = rt.archetype_registry
        assert reg is not None
        assert reg.get("customer-experience") is not None

    def test_runtime_has_zone_validator(self) -> None:
        rt = GovernanceRuntime()
        zv = rt.zone_validator
        assert zv is not None
        result = zv.validate("global/us")
        assert result.valid is True

    def test_runtime_has_org_registry(self) -> None:
        rt = GovernanceRuntime()
        org_reg = rt.org_registry
        assert org_reg is not None

    def test_runtime_registries_are_lazy_singletons(self) -> None:
        rt = GovernanceRuntime()
        assert rt.archetype_registry is rt.archetype_registry
        assert rt.zone_validator is rt.zone_validator
        assert rt.org_registry is rt.org_registry

    def test_runtime_birth_without_registries_on_ca(self) -> None:
        """Runtime.birth() should work without registries on the auto CA."""
        rt = GovernanceRuntime()
        cert = rt.birth("a1", "any-arch", "any-org", "global")
        assert cert.agent_id == "a1"
