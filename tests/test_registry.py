"""Tests for the validation registries: archetypes, zones, organizations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nomotic.registry import (
    BUILT_IN_ARCHETYPES,
    ArchetypeDefinition,
    ArchetypeRegistry,
    FileOrgStore,
    MemoryOrgStore,
    OrganizationRegistry,
    OrgRegistration,
    OrgStatus,
    ValidationResult,
    ZoneValidator,
    _normalize_org_name,
    _validate_name_format,
)


# ═══════════════════════════════════════════════════════════════════════
# Name format validation
# ═══════════════════════════════════════════════════════════════════════


class TestNameFormatValidation:
    """Shared name-validation rules used by archetypes, zones, and orgs."""

    def test_valid_names(self) -> None:
        for name in [
            "abc",
            "customer-experience",
            "data-processing",
            "a1b",
            "x" * 64,
            "a-b",
            "ab1",
        ]:
            assert _validate_name_format(name) == [], f"{name!r} should be valid"

    def test_empty_name(self) -> None:
        errors = _validate_name_format("")
        assert any("empty" in e for e in errors)

    def test_too_short(self) -> None:
        errors = _validate_name_format("ab")
        assert any("at least 3" in e for e in errors)

    def test_too_long(self) -> None:
        errors = _validate_name_format("a" * 65)
        assert any("at most 64" in e for e in errors)

    def test_uppercase_rejected(self) -> None:
        errors = _validate_name_format("Customer")
        assert any("lowercase" in e for e in errors)

    def test_spaces_rejected(self) -> None:
        errors = _validate_name_format("my name")
        assert any("alphanumeric" in e or "lowercase" in e for e in errors)

    def test_underscores_rejected(self) -> None:
        errors = _validate_name_format("my_name")
        assert any("alphanumeric" in e or "hyphens" in e for e in errors)

    def test_start_with_hyphen_rejected(self) -> None:
        errors = _validate_name_format("-abc")
        assert any("start" in e for e in errors)

    def test_end_with_hyphen_rejected(self) -> None:
        errors = _validate_name_format("abc-")
        assert any("end" in e for e in errors)

    def test_double_hyphen_rejected(self) -> None:
        errors = _validate_name_format("abc--def")
        assert any("double" in e for e in errors)

    def test_dots_rejected(self) -> None:
        errors = _validate_name_format("my.name")
        assert len(errors) > 0

    def test_minimum_length_three(self) -> None:
        assert _validate_name_format("abc") == []

    def test_exact_max_length(self) -> None:
        name = "a" + "-b" * 31 + "c"  # 64 chars
        # This might have double hyphens so let's use a simple name
        name = "a" * 64
        assert _validate_name_format(name) == []


# ═══════════════════════════════════════════════════════════════════════
# ArchetypeRegistry
# ═══════════════════════════════════════════════════════════════════════


class TestArchetypeDefinition:
    def test_frozen(self) -> None:
        defn = ArchetypeDefinition("test", "desc", "cat", True)
        with pytest.raises(AttributeError):
            defn.name = "other"  # type: ignore[misc]


class TestArchetypeRegistry:
    def test_empty_registry(self) -> None:
        reg = ArchetypeRegistry()
        assert reg.list() == []

    def test_with_defaults_has_all_builtins(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        for name in BUILT_IN_ARCHETYPES:
            defn = reg.get(name)
            assert defn is not None
            assert defn.builtin is True
            assert defn.name == name

    def test_builtin_count(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        assert len(reg.list()) == len(BUILT_IN_ARCHETYPES)

    def test_all_builtins_valid_format(self) -> None:
        for name in BUILT_IN_ARCHETYPES:
            assert _validate_name_format(name) == [], f"Built-in '{name}' has invalid format"

    def test_register_custom(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        defn = reg.register("my-custom-agent", "A custom agent", "custom")
        assert defn.name == "my-custom-agent"
        assert defn.builtin is False
        assert reg.get("my-custom-agent") is not None

    def test_register_duplicate_raises(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        with pytest.raises(ValueError, match="already registered"):
            reg.register("customer-experience", "dup", "dup")

    def test_register_invalid_name_raises(self) -> None:
        reg = ArchetypeRegistry()
        with pytest.raises(ValueError, match="Invalid"):
            reg.register("Bad Name", "desc", "cat")

    def test_get_missing_returns_none(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        assert reg.get("nonexistent-archetype") is None


class TestArchetypeValidation:
    def test_validate_known_archetype(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        result = reg.validate("customer-experience")
        assert result.valid is True
        assert result.warnings == []
        assert result.errors == []

    def test_validate_unknown_nonstrict(self) -> None:
        reg = ArchetypeRegistry.with_defaults(strict=False)
        result = reg.validate("my-new-archetype")
        assert result.valid is True
        assert len(result.warnings) > 0
        assert "not registered" in result.warnings[0]

    def test_validate_unknown_strict(self) -> None:
        reg = ArchetypeRegistry.with_defaults(strict=True)
        result = reg.validate("my-new-archetype")
        assert result.valid is False
        assert len(result.errors) > 0
        assert "unknown" in result.errors[0]

    def test_validate_invalid_format(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        result = reg.validate("Bad Name!")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_too_short(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        result = reg.validate("ab")
        assert result.valid is False

    def test_validate_double_hyphen(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        result = reg.validate("my--archetype")
        assert result.valid is False


class TestArchetypeFuzzy:
    def test_suggestion_for_typo(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        result = reg.validate("customer-experince")
        assert result.suggestion == "customer-experience"

    def test_suggestion_for_close_match(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        result = reg.validate("data-procesing")
        assert result.suggestion == "data-processing"

    def test_no_suggestion_for_unrelated(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        result = reg.validate("zzzzzzzzzzz")
        assert result.suggestion is None


class TestArchetypeCategories:
    def test_list_by_category(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        financial = reg.list(category="financial")
        assert len(financial) >= 2
        assert all(a.category == "financial" for a in financial)

    def test_list_all_categories(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        cats = reg.categories()
        assert "financial" in cats
        assert "operations" in cats
        assert "customer-facing" in cats

    def test_list_empty_category(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        assert reg.list(category="nonexistent") == []

    def test_list_sorted_by_name(self) -> None:
        reg = ArchetypeRegistry.with_defaults()
        names = [a.name for a in reg.list()]
        assert names == sorted(names)


# ═══════════════════════════════════════════════════════════════════════
# ZoneValidator
# ═══════════════════════════════════════════════════════════════════════


class TestZoneValidator:
    def setup_method(self) -> None:
        self.v = ZoneValidator()

    def test_valid_single_segment(self) -> None:
        result = self.v.validate("global")
        assert result.valid is True

    def test_valid_multi_segment(self) -> None:
        result = self.v.validate("global/us/production")
        assert result.valid is True

    def test_valid_two_segments(self) -> None:
        result = self.v.validate("global/us")
        assert result.valid is True

    def test_empty_path(self) -> None:
        result = self.v.validate("")
        assert result.valid is False

    def test_leading_slash(self) -> None:
        result = self.v.validate("/global/us")
        assert result.valid is False
        assert any("start" in e for e in result.errors)

    def test_trailing_slash(self) -> None:
        result = self.v.validate("global/us/")
        assert result.valid is False
        assert any("end" in e for e in result.errors)

    def test_double_slash(self) -> None:
        result = self.v.validate("global//us")
        assert result.valid is False
        assert any("//" in e for e in result.errors)

    def test_uppercase_segment(self) -> None:
        result = self.v.validate("global/US/production")
        assert result.valid is False

    def test_max_depth(self) -> None:
        path = "/".join(f"s{i}" for i in range(10))
        result = self.v.validate(path)
        assert result.valid is True

    def test_exceeds_max_depth(self) -> None:
        path = "/".join(f"s{i}" for i in range(11))
        result = self.v.validate(path)
        assert result.valid is False
        assert any("at most" in e for e in result.errors)

    def test_segment_with_hyphen(self) -> None:
        result = self.v.validate("global/us-east/prod")
        assert result.valid is True

    def test_segment_with_numbers(self) -> None:
        result = self.v.validate("region1/zone2")
        assert result.valid is True


class TestZoneParseSegments:
    def setup_method(self) -> None:
        self.v = ZoneValidator()

    def test_parse_valid(self) -> None:
        segs = self.v.parse_segments("global/us/production")
        assert segs == ["global", "us", "production"]

    def test_parse_single(self) -> None:
        segs = self.v.parse_segments("global")
        assert segs == ["global"]

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            self.v.parse_segments("global//us")


class TestZoneHierarchy:
    def setup_method(self) -> None:
        self.v = ZoneValidator()

    def test_is_parent_of(self) -> None:
        assert self.v.is_parent_of("global", "global/us") is True
        assert self.v.is_parent_of("global", "global/us/prod") is True
        assert self.v.is_parent_of("global/us", "global/us/prod") is True

    def test_is_not_parent_of_self(self) -> None:
        assert self.v.is_parent_of("global", "global") is False

    def test_is_not_parent_of_sibling(self) -> None:
        assert self.v.is_parent_of("global/us", "global/eu") is False

    def test_common_ancestor(self) -> None:
        assert self.v.common_ancestor("global/us/prod", "global/us/staging") == "global/us"

    def test_common_ancestor_same(self) -> None:
        assert self.v.common_ancestor("global/us", "global/us") == "global/us"

    def test_common_ancestor_root(self) -> None:
        assert self.v.common_ancestor("global/us", "global/eu") == "global"

    def test_common_ancestor_none(self) -> None:
        assert self.v.common_ancestor("us/prod", "eu/prod") is None


# ═══════════════════════════════════════════════════════════════════════
# Organization name normalization
# ═══════════════════════════════════════════════════════════════════════


class TestOrgNameNormalization:
    def test_simple(self) -> None:
        assert _normalize_org_name("Acme Corp") == "acme-corp"

    def test_strip_whitespace(self) -> None:
        assert _normalize_org_name("  My Company  ") == "my-company"

    def test_uppercase(self) -> None:
        assert _normalize_org_name("ACME CORP") == "acme-corp"

    def test_special_chars_removed(self) -> None:
        assert _normalize_org_name("ACME_Corp!") == "acme-corp"

    def test_collapse_hyphens(self) -> None:
        assert _normalize_org_name("acme---corp") == "acme-corp"

    def test_already_normalized(self) -> None:
        assert _normalize_org_name("acme-corp") == "acme-corp"

    def test_trailing_period_stripped(self) -> None:
        assert _normalize_org_name("Acme Co.") == "acme-co"

    def test_trailing_hyphens_stripped(self) -> None:
        assert _normalize_org_name("acme-") == "acme"


# ═══════════════════════════════════════════════════════════════════════
# MemoryOrgStore
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryOrgStore:
    def _make_org(self, name: str = "acme-corp") -> OrgRegistration:
        from datetime import datetime, timezone
        return OrgRegistration(
            name=name,
            display_name="Acme Corp",
            registered_at=datetime.now(timezone.utc),
            issuer_fingerprint="SHA256:abc123",
        )

    def test_save_and_get(self) -> None:
        store = MemoryOrgStore()
        org = self._make_org()
        store.save(org)
        assert store.get("acme-corp") is not None

    def test_get_missing(self) -> None:
        store = MemoryOrgStore()
        assert store.get("nonexistent") is None

    def test_list_all(self) -> None:
        store = MemoryOrgStore()
        store.save(self._make_org("aaa"))
        store.save(self._make_org("bbb"))
        assert len(store.list()) == 2

    def test_list_by_status(self) -> None:
        store = MemoryOrgStore()
        org1 = self._make_org("aaa")
        org2 = self._make_org("bbb")
        org2.status = OrgStatus.SUSPENDED
        store.save(org1)
        store.save(org2)
        active = store.list(status=OrgStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].name == "aaa"

    def test_update(self) -> None:
        store = MemoryOrgStore()
        org = self._make_org()
        store.save(org)
        org.status = OrgStatus.SUSPENDED
        store.update(org)
        retrieved = store.get("acme-corp")
        assert retrieved is not None
        assert retrieved.status == OrgStatus.SUSPENDED


# ═══════════════════════════════════════════════════════════════════════
# FileOrgStore
# ═══════════════════════════════════════════════════════════════════════


class TestFileOrgStore:
    def _make_org(self, name: str = "acme-corp") -> OrgRegistration:
        from datetime import datetime, timezone
        return OrgRegistration(
            name=name,
            display_name="Acme Corp",
            registered_at=datetime.now(timezone.utc),
            issuer_fingerprint="SHA256:abc123",
        )

    def test_save_and_get(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = FileOrgStore(td)
            org = self._make_org()
            store.save(org)
            retrieved = store.get("acme-corp")
            assert retrieved is not None
            assert retrieved.name == "acme-corp"

    def test_get_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = FileOrgStore(td)
            assert store.get("nonexistent") is None

    def test_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = FileOrgStore(td)
            store.save(self._make_org("aaa"))
            store.save(self._make_org("bbb"))
            assert len(store.list()) == 2

    def test_update(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = FileOrgStore(td)
            org = self._make_org()
            store.save(org)
            org.status = OrgStatus.SUSPENDED
            store.update(org)
            retrieved = store.get("acme-corp")
            assert retrieved is not None
            assert retrieved.status == OrgStatus.SUSPENDED


# ═══════════════════════════════════════════════════════════════════════
# OrganizationRegistry
# ═══════════════════════════════════════════════════════════════════════


class TestOrganizationRegistry:
    def test_register_and_get(self) -> None:
        reg = OrganizationRegistry()
        org = reg.register("Acme Corp", "SHA256:abc123")
        assert org.name == "acme-corp"
        assert org.display_name == "Acme Corp"
        assert reg.get("acme-corp") is not None

    def test_register_normalized_lookup(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc123")
        assert reg.get("Acme Corp") is not None
        assert reg.get("ACME CORP") is not None

    def test_register_duplicate_raises(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc123")
        with pytest.raises(ValueError, match="already registered"):
            reg.register("acme-corp", "SHA256:other")

    def test_register_duplicate_different_casing(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc123")
        with pytest.raises(ValueError, match="already registered"):
            reg.register("ACME CORP", "SHA256:other")

    def test_register_invalid_name(self) -> None:
        reg = OrganizationRegistry()
        with pytest.raises(ValueError, match="Invalid"):
            reg.register("ab", "SHA256:abc123")  # too short after normalization

    def test_register_with_email(self) -> None:
        reg = OrganizationRegistry()
        org = reg.register("My Org", "SHA256:abc", contact_email="info@myorg.com")
        assert org.contact_email == "info@myorg.com"

    def test_validate_available(self) -> None:
        reg = OrganizationRegistry()
        result = reg.validate("acme-corp")
        assert result.valid is True

    def test_validate_taken(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc123")
        result = reg.validate("acme-corp")
        assert result.valid is False
        assert any("already registered" in e for e in result.errors)

    def test_validate_invalid_format(self) -> None:
        reg = OrganizationRegistry()
        result = reg.validate("a")  # too short
        assert result.valid is False

    def test_verify_issuer_match(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc123")
        assert reg.verify_issuer("acme-corp", "SHA256:abc123") is True

    def test_verify_issuer_mismatch(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc123")
        assert reg.verify_issuer("acme-corp", "SHA256:wrong") is False

    def test_verify_issuer_missing_org(self) -> None:
        reg = OrganizationRegistry()
        assert reg.verify_issuer("nonexistent", "SHA256:abc") is False

    def test_list_all(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Org A", "SHA256:a")
        reg.register("Org B", "SHA256:b")
        assert len(reg.list()) == 2

    def test_list_by_status(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Org A", "SHA256:a")
        reg.register("Org B", "SHA256:b")
        reg.suspend("org-b", "testing")
        active = reg.list(status=OrgStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].name == "org-a"


class TestOrgLifecycle:
    def test_suspend(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc")
        org = reg.suspend("acme-corp", "testing")
        assert org.status == OrgStatus.SUSPENDED

    def test_suspend_nonexistent_raises(self) -> None:
        reg = OrganizationRegistry()
        with pytest.raises(KeyError):
            reg.suspend("nonexistent", "testing")

    def test_suspend_non_active_raises(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc")
        reg.suspend("acme-corp", "testing")
        with pytest.raises(ValueError, match="Cannot suspend"):
            reg.suspend("acme-corp", "again")

    def test_revoke(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc")
        org = reg.revoke("acme-corp", "testing")
        assert org.status == OrgStatus.REVOKED

    def test_revoke_nonexistent_raises(self) -> None:
        reg = OrganizationRegistry()
        with pytest.raises(KeyError):
            reg.revoke("nonexistent", "testing")

    def test_revoke_already_revoked_raises(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc")
        reg.revoke("acme-corp", "testing")
        with pytest.raises(ValueError, match="already revoked"):
            reg.revoke("acme-corp", "again")

    def test_revoke_suspended_org(self) -> None:
        reg = OrganizationRegistry()
        reg.register("Acme Corp", "SHA256:abc")
        reg.suspend("acme-corp", "testing")
        org = reg.revoke("acme-corp", "permanent")
        assert org.status == OrgStatus.REVOKED


# ═══════════════════════════════════════════════════════════════════════
# OrgRegistration serialization
# ═══════════════════════════════════════════════════════════════════════


class TestOrgRegistrationSerialization:
    def test_round_trip(self) -> None:
        from datetime import datetime, timezone
        org = OrgRegistration(
            name="acme-corp",
            display_name="Acme Corp",
            registered_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            issuer_fingerprint="SHA256:abc123",
            contact_email="info@acme.com",
            status=OrgStatus.ACTIVE,
        )
        d = org.to_dict()
        restored = OrgRegistration.from_dict(d)
        assert restored.name == org.name
        assert restored.display_name == org.display_name
        assert restored.issuer_fingerprint == org.issuer_fingerprint
        assert restored.contact_email == org.contact_email
        assert restored.status == org.status


# ═══════════════════════════════════════════════════════════════════════
# ValidationResult
# ═══════════════════════════════════════════════════════════════════════


class TestValidationResult:
    def test_valid_result(self) -> None:
        r = ValidationResult(valid=True, name="test")
        assert r.valid is True
        assert r.warnings == []
        assert r.errors == []
        assert r.suggestion is None

    def test_invalid_with_errors(self) -> None:
        r = ValidationResult(valid=False, name="test", errors=["bad"])
        assert r.valid is False
        assert r.errors == ["bad"]

    def test_valid_with_suggestion(self) -> None:
        r = ValidationResult(valid=True, name="test", suggestion="other")
        assert r.suggestion == "other"
