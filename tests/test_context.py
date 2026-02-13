"""Tests for context codes — the structured vocabulary for governance events."""

import pytest

from nomotic.context import CODES, ContextCode


class TestContextCode:
    """ContextCode dataclass tests."""

    def test_auto_category(self):
        code = ContextCode("GOVERNANCE.ALLOW", "test")
        assert code.category == "GOVERNANCE"

    def test_explicit_category(self):
        code = ContextCode("GOVERNANCE.ALLOW", "test", category="CUSTOM")
        assert code.category == "CUSTOM"

    def test_str(self):
        code = ContextCode("GOVERNANCE.ALLOW", "test")
        assert str(code) == "GOVERNANCE.ALLOW"

    def test_eq_string(self):
        code = ContextCode("GOVERNANCE.ALLOW", "test")
        assert code == "GOVERNANCE.ALLOW"
        assert code != "GOVERNANCE.DENY"

    def test_eq_context_code(self):
        a = ContextCode("GOVERNANCE.ALLOW", "test a")
        b = ContextCode("GOVERNANCE.ALLOW", "test b")
        assert a == b

    def test_neq_different_code(self):
        a = ContextCode("GOVERNANCE.ALLOW", "test")
        b = ContextCode("GOVERNANCE.DENY", "test")
        assert a != b

    def test_eq_other_type(self):
        code = ContextCode("GOVERNANCE.ALLOW", "test")
        assert code.__eq__(42) is NotImplemented

    def test_hash(self):
        a = ContextCode("GOVERNANCE.ALLOW", "test a")
        b = ContextCode("GOVERNANCE.ALLOW", "test b")
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1

    def test_frozen(self):
        code = ContextCode("GOVERNANCE.ALLOW", "test")
        with pytest.raises(AttributeError):
            code.code = "CHANGED"  # type: ignore[misc]

    def test_default_severity(self):
        code = ContextCode("TEST.CODE", "test")
        assert code.severity == "info"


class TestCODES:
    """CODES catalog tests."""

    def test_all_codes_returns_list(self):
        codes = CODES.all_codes()
        assert isinstance(codes, list)
        assert len(codes) > 0

    def test_all_codes_unique(self):
        codes = CODES.all_codes()
        code_strings = [c.code for c in codes]
        assert len(code_strings) == len(set(code_strings)), "Duplicate code strings found"

    def test_all_codes_have_valid_categories(self):
        valid_categories = {
            "GOVERNANCE", "SCOPE", "TRUST", "DRIFT", "SECURITY",
            "USER", "CONFIG", "SYSTEM", "OVERRIDE", "ETHICAL",
        }
        for code in CODES.all_codes():
            assert code.category in valid_categories, f"Invalid category: {code.category} for {code.code}"

    def test_all_codes_have_valid_severities(self):
        valid_severities = {"info", "warning", "alert", "critical"}
        for code in CODES.all_codes():
            assert code.severity in valid_severities, f"Invalid severity: {code.severity} for {code.code}"

    def test_by_category_governance(self):
        gov = CODES.by_category("GOVERNANCE")
        assert len(gov) >= 5
        assert all(c.category == "GOVERNANCE" for c in gov)

    def test_by_category_security(self):
        sec = CODES.by_category("SECURITY")
        assert len(sec) >= 4
        assert all(c.category == "SECURITY" for c in sec)

    def test_by_category_empty(self):
        result = CODES.by_category("NONEXISTENT")
        assert result == []

    def test_by_severity_critical(self):
        crit = CODES.by_severity("critical")
        assert len(crit) >= 3
        assert all(c.severity == "critical" for c in crit)

    def test_by_severity_info(self):
        info = CODES.by_severity("info")
        assert len(info) >= 5
        assert all(c.severity == "info" for c in info)

    def test_lookup_found(self):
        code = CODES.lookup("GOVERNANCE.ALLOW")
        assert code is not None
        assert code.code == "GOVERNANCE.ALLOW"
        assert code == CODES.GOVERNANCE_ALLOW

    def test_lookup_not_found(self):
        code = CODES.lookup("NONEXISTENT.CODE")
        assert code is None

    def test_specific_codes_exist(self):
        """Verify key codes from each category exist."""
        assert CODES.GOVERNANCE_ALLOW.code == "GOVERNANCE.ALLOW"
        assert CODES.GOVERNANCE_DENY.severity == "warning"
        assert CODES.GOVERNANCE_VETO.severity == "alert"
        assert CODES.SCOPE_VIOLATION.severity == "alert"
        assert CODES.TRUST_EARNED.severity == "info"
        assert CODES.TRUST_LOST_DRIFT.severity == "warning"
        assert CODES.DRIFT_DETECTED_CRITICAL.severity == "critical"
        assert CODES.SECURITY_INJECTION_ATTEMPT.severity == "critical"
        assert CODES.USER_REQUEST_NORMAL.severity == "info"
        assert CODES.CONFIG_SCOPE_CHANGED.severity == "warning"
        assert CODES.SYSTEM_ERROR.severity == "critical"
        assert CODES.OVERRIDE_APPROVED.severity == "info"
        assert CODES.ETHICAL_CONSTRAINT_FAILED.severity == "alert"
