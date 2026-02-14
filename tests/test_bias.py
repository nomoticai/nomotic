"""Tests for bias detection — structural analysis of governance rules."""

from nomotic.bias import BiasDetector, GovernanceBiasReport, RuleBiasAssessment, StructuralConcern
from nomotic.equity import EquityConfig, ProtectedAttribute

# ── Mock helpers ──────────────────────────────────────────────────────

class _Dim:
    def __init__(self, *, scopes=None, authority_checks=None, rules=None,
                 limits=None, time_windows=None, weight=1.0):
        self._allowed_scopes = scopes or {}
        self._authority_checks = authority_checks or []
        self._rules = rules or []
        self._limits = limits or {}
        self._time_windows = time_windows or {}
        self.weight = weight

class _Prov:
    def __init__(self, v="v1.0.0"): self._v = v
    def current_config_version(self): return self._v

class _Reg:
    def __init__(self, d=None): self._d = d or {}
    def get(self, name): return self._d.get(name)

class _RT:
    def __init__(self, dims=None, version="v1.0.0"):
        self.registry = _Reg(dims or {})
        self.provenance_log = _Prov(version)

def _neutral_rt():
    return _RT(dims={
        "scope_compliance": _Dim(scopes={"a": {"read", "write"}, "b": {"read", "write"}}),
        "authority_verification": _Dim(authority_checks=["c1"]),
        "ethical_alignment": _Dim(rules=["r1"], weight=0.8),
        "stakeholder_impact": _Dim(weight=0.7),
        "resource_boundaries": _Dim(), "temporal_compliance": _Dim(),
    })

def _proxy():
    return ProtectedAttribute(name="zip_code", description="Postal code",
                              attribute_source="p.zip", is_proxy=True, proxy_for="race")

def _plain():
    return ProtectedAttribute(name="region", description="Region",
                              attribute_source="p.region")

def _det(attrs=None):
    return BiasDetector(EquityConfig(protected_attributes=attrs or [_plain()]))

# ── BiasDetector.assess_configuration ─────────────────────────────────

class TestAssessConfiguration:
    def test_neutral_config_no_high_risk(self):
        report = _det().assess_configuration(_neutral_rt())
        assert not [a for a in report.assessments if a.bias_risk == "high"]

    def test_detects_proxy_variable(self):
        report = _det([_proxy()]).assess_configuration(_neutral_rt())
        assert "proxy_variable_in_rules" in [c.concern_type for c in report.structural_concerns]

    def test_detects_uniform_rule(self):
        rt = _RT(dims={"scope_compliance": _Dim(scopes={"a": {"read"}, "b": {"read"}}),
                        "authority_verification": _Dim(authority_checks=["c"]),
                        "ethical_alignment": _Dim(rules=["r"], weight=0.8),
                        "stakeholder_impact": _Dim(weight=0.7)})
        report = _det().assess_configuration(rt)
        assert "uniform_rule_non_uniform_population" in [c.concern_type for c in report.structural_concerns]

    def test_missing_equity_evaluation_needs_high_risk(self):
        rt = _RT(dims={"scope_compliance": _Dim(scopes={"a": {"r"}}),
                        "authority_verification": _Dim(authority_checks=["c"]),
                        "ethical_alignment": _Dim(rules=["r"], weight=0.8),
                        "stakeholder_impact": _Dim(weight=0.7)})
        report = _det([]).assess_configuration(rt)
        assert "missing_equity_evaluation" not in [c.concern_type for c in report.structural_concerns]

    def test_asymmetric_authority_via_scope_ratio(self):
        rt = _RT(dims={"scope_compliance": _Dim(scopes={"a": set("abcdefg"), "b": {"a"}}),
                        "authority_verification": _Dim(authority_checks=["c"]),
                        "ethical_alignment": _Dim(rules=["r"], weight=0.8),
                        "stakeholder_impact": _Dim(weight=0.7)})
        report = _det().assess_configuration(rt)
        scope = [a for a in report.assessments if a.rule_type == "scope"][0]
        assert scope.bias_risk != "none_detected"

    def test_threshold_cliff_effects(self):
        a = _det().assess_rule("scope", {"threshold_score": 0.75}, [_plain()])
        assert any("cliff" in c for c in a.concerns)

    def test_assess_rule_scope(self):
        a = _det().assess_rule("scope", {"allowed": ["read"]}, [_plain()])
        assert a.rule_type == "scope" and a.bias_risk == "none_detected"

    def test_assess_rule_authority_envelope(self):
        a = _det().assess_rule("authority_envelope", {"level": "admin"}, [_plain()])
        assert a.rule_type == "authority_envelope"

    def test_assess_rule_ethical(self):
        a = _det().assess_rule("ethical_rule", {"check": "fairness"}, [_plain()])
        assert a.rule_type == "ethical_rule" and a.bias_risk == "none_detected"

    def test_assess_rule_protected_attr_reference(self):
        a = _det().assess_rule("scope", {"filter_by": "region"}, [_plain()])
        assert a.bias_risk == "high" and "region" in a.affected_attributes

    def test_assess_rule_proxy_reference(self):
        a = _det([_proxy()]).assess_rule("scope", {"f": "zip_code"}, [_proxy()])
        assert a.bias_risk in ("medium", "high") and "zip_code" in a.affected_attributes

    def test_severity_levels(self):
        d = _det()
        assert d.assess_rule("scope", {"f": "region"}, [_plain()]).bias_risk == "high"
        assert d.assess_rule("scope", {"threshold_limit": 5}, [_plain()]).bias_risk == "low"

    def test_affected_attributes_populated(self):
        a = _det().assess_rule("scope", {"v": "region"}, [_plain()])
        assert "region" in a.affected_attributes

    def test_recommendation_populated(self):
        a = _det().assess_rule("scope", {"v": "region"}, [_plain()])
        assert a.recommendation != ""

    def test_structural_concerns_have_affected_rules(self):
        report = _det([_proxy()]).assess_configuration(_neutral_rt())
        pc = [c for c in report.structural_concerns if c.concern_type == "proxy_variable_in_rules"]
        assert pc[0].affected_rules

# ── compare_agent_configurations ──────────────────────────────────────

class TestCompareAgents:
    def test_disparate_config(self):
        rt = _RT(dims={"scope_compliance": _Dim(scopes={"a": set("abcde"), "b": {"a"}})})
        cs = _det().compare_agent_configurations(rt, ["a", "b"])
        assert any(c.concern_type == "asymmetric_authority" for c in cs)

    def test_identical_configs_no_asymmetry(self):
        rt = _RT(dims={"scope_compliance": _Dim(scopes={"a": {"r", "w"}, "b": {"r", "w"}})})
        cs = _det().compare_agent_configurations(rt, ["a", "b"])
        assert not any(c.concern_type == "asymmetric_authority" for c in cs)

# ── GovernanceBiasReport ──────────────────────────────────────────────

class TestGovernanceBiasReport:
    def test_serialization(self):
        r = GovernanceBiasReport(
            report_id="abc", generated_at="2025-01-01T00:00:00+00:00",
            rules_assessed=2,
            assessments=[RuleBiasAssessment("scope", "t", "low", concerns=["c"])],
            structural_concerns=[StructuralConcern("proxy_variable_in_rules", "d", "medium", ["scope"])],
            summary="s", config_version="v2.0")
        d = r.to_dict()
        assert d["report_id"] == "abc" and len(d["assessments"]) == 1
        assert d["structural_concerns"][0]["concern_type"] == "proxy_variable_in_rules"
        assert d["config_version"] == "v2.0"

    def test_summary_generation(self):
        report = _det().assess_configuration(_neutral_rt())
        assert "Assessed" in report.summary

    def test_config_version_captured(self):
        rt = _RT(dims={"scope_compliance": _Dim(scopes={"a": {"r"}}),
                        "authority_verification": _Dim(authority_checks=["c"]),
                        "ethical_alignment": _Dim(rules=["r"], weight=0.8),
                        "stakeholder_impact": _Dim(weight=0.7)}, version="v3.1.4")
        assert _det().assess_configuration(rt).config_version == "v3.1.4"

    def test_rules_assessed_count(self):
        report = _det().assess_configuration(_neutral_rt())
        assert report.rules_assessed == len(report.assessments) > 0

    def test_empty_runtime_clean_report(self):
        report = _det([]).assess_configuration(_RT())
        assert report.report_id and report.generated_at
        assert report.rules_assessed == 1 and report.structural_concerns == []

# ── Bias + Equity complementarity ─────────────────────────────────────

class TestComplementarity:
    def test_proxy_cross_reference(self):
        report = _det([_proxy()]).assess_configuration(_neutral_rt())
        pc = [c for c in report.structural_concerns if c.concern_type == "proxy_variable_in_rules"]
        assert len(pc) >= 1 and "zip_code" in pc[0].description

    def test_no_structural_concern_still_has_assessments(self):
        rt = _RT(dims={"scope_compliance": _Dim(scopes={"a": {"r", "w"}}),
                        "authority_verification": _Dim(authority_checks=["c"]),
                        "ethical_alignment": _Dim(rules=[], weight=0.8),
                        "stakeholder_impact": _Dim(weight=0.7)})
        report = _det().assess_configuration(rt)
        assert report.rules_assessed > 0 and isinstance(report.summary, str)

    def test_independent_reports(self):
        cfg = EquityConfig(protected_attributes=[_plain(), _proxy()], minimum_sample_size=10)
        det = BiasDetector(cfg)
        assert det.config is cfg
        assert isinstance(det.assess_configuration(_neutral_rt()), GovernanceBiasReport)
        assert len(EquityConfig.from_dict(cfg.to_dict()).protected_attributes) == 2

    def test_update_config_changes_behaviour(self):
        det = BiasDetector(EquityConfig(protected_attributes=[]))
        r1 = det.assess_configuration(_neutral_rt())
        assert not [c for c in r1.structural_concerns if c.concern_type == "proxy_variable_in_rules"]
        det.update_config(EquityConfig(protected_attributes=[_proxy()]))
        r2 = det.assess_configuration(_neutral_rt())
        assert [c for c in r2.structural_concerns if c.concern_type == "proxy_variable_in_rules"]

    def test_attribute_names_shared(self):
        attr = _proxy()
        report = _det([attr]).assess_configuration(_neutral_rt())
        pc = [c for c in report.structural_concerns if c.concern_type == "proxy_variable_in_rules"]
        assert attr.name in pc[0].description
