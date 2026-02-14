"""Tests for Nomotic Protocol types and validation."""

import json
import pytest

from nomotic.protocol import (
    METHODS,
    METHOD_CATEGORIES,
    PROTOCOL_VERSION,
    Alternative,
    Assumption,
    Assessment,
    AuthorityClaim,
    Condition,
    Constraint,
    Denial,
    Escalation,
    Factor,
    GovernanceResponseData,
    Guidance,
    IntendedAction,
    Justification,
    Plan,
    ProtocolVerdict,
    ReasoningArtifact,
    ResponseMetadata,
    Unknown,
    method_category,
    validate_artifact,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_artifact(**overrides):
    """Create a minimal valid ReasoningArtifact for testing."""
    defaults = dict(
        agent_id="test-agent-1",
        goal="Process test request",
        origin="user_request",
        constraints_identified=[
            Constraint(type="policy", description="Standard limit $500", source="policy://test"),
        ],
        factors=[
            Factor(
                id="f1", type="constraint", description="Amount within limit",
                source="policy://test", assessment="Constraint satisfied",
                influence="decisive", confidence=0.95,
            ),
            Factor(
                id="f2", type="context", description="User is verified",
                source="data://users/123", assessment="User verified",
                influence="significant", confidence=0.9,
            ),
        ],
        alternatives_considered=[
            Alternative(method="deny", reason_rejected="No grounds for denial"),
        ],
        intended_action=IntendedAction(method="approve", target="order/123"),
        justifications=[
            Justification(factor_id="f1", explanation="Constraint satisfied"),
        ],
        authority_claim=AuthorityClaim(envelope_type="standard"),
        unknowns=[],
        assumptions=[],
        overall_confidence=0.88,
    )
    defaults.update(overrides)
    return ReasoningArtifact(**defaults)


# ── Method Taxonomy ────────────────────────────────────────────────────


class TestMethodTaxonomy:
    def test_all_categories_have_methods(self):
        for cat, methods in METHOD_CATEGORIES.items():
            assert len(methods) > 0, f"Category {cat} has no methods"

    def test_all_methods_in_flat_set(self):
        for cat, methods in METHOD_CATEGORIES.items():
            for m in methods:
                assert m in METHODS

    def test_method_category_lookup(self):
        assert method_category("query") == "data"
        assert method_category("approve") == "decision"
        assert method_category("transfer") == "transaction"
        assert method_category("authenticate") == "security"
        assert method_category("nonexistent") is None

    def test_known_method_count(self):
        # Verify we have a substantial taxonomy
        assert len(METHODS) > 70


# ── Constraint ─────────────────────────────────────────────────────────


class TestConstraint:
    def test_roundtrip(self):
        c = Constraint(type="policy", description="Limit $500", source="policy://test")
        d = c.to_dict()
        c2 = Constraint.from_dict(d)
        assert c2.type == "policy"
        assert c2.description == "Limit $500"
        assert c2.source == "policy://test"


# ── Factor ─────────────────────────────────────────────────────────────


class TestFactor:
    def test_roundtrip(self):
        f = Factor(
            id="f1", type="constraint", description="Test",
            source="test", assessment="OK", influence="decisive", confidence=0.9,
        )
        d = f.to_dict()
        f2 = Factor.from_dict(d)
        assert f2.id == "f1"
        assert f2.confidence == 0.9


# ── Alternative ────────────────────────────────────────────────────────


class TestAlternative:
    def test_roundtrip_with_context(self):
        a = Alternative(method="deny", reason_rejected="No grounds", context="Full denial")
        d = a.to_dict()
        assert "context" in d
        a2 = Alternative.from_dict(d)
        assert a2.context == "Full denial"

    def test_roundtrip_without_context(self):
        a = Alternative(method="deny", reason_rejected="No grounds")
        d = a.to_dict()
        assert "context" not in d
        a2 = Alternative.from_dict(d)
        assert a2.context == ""


# ── IntendedAction ─────────────────────────────────────────────────────


class TestIntendedAction:
    def test_roundtrip(self):
        ia = IntendedAction(
            method="approve", target="order/123",
            context="Gold-tier exception",
            parameters={"amount": 800},
        )
        d = ia.to_dict()
        ia2 = IntendedAction.from_dict(d)
        assert ia2.method == "approve"
        assert ia2.parameters == {"amount": 800}

    def test_minimal(self):
        ia = IntendedAction(method="read", target="data/records")
        d = ia.to_dict()
        assert "context" not in d
        assert "parameters" not in d


# ── AuthorityClaim ─────────────────────────────────────────────────────


class TestAuthorityClaim:
    def test_conditional(self):
        ac = AuthorityClaim(
            envelope_type="conditional",
            conditions_met=["Gold tier", "Within range"],
            limit_reference="$1500",
        )
        d = ac.to_dict()
        ac2 = AuthorityClaim.from_dict(d)
        assert ac2.envelope_type == "conditional"
        assert len(ac2.conditions_met) == 2

    def test_standard(self):
        ac = AuthorityClaim(envelope_type="standard")
        d = ac.to_dict()
        assert "conditions_met" not in d


# ── Plan ───────────────────────────────────────────────────────────────


class TestPlan:
    def test_roundtrip(self):
        p = Plan(
            workflow_id="wf-1", total_steps=3, current_step=2,
            step_description="Process payment",
            dependencies=["ra-step1"],
            remaining_steps=[{"step_number": 3, "description": "Confirm"}],
            rollback_capability=True,
        )
        d = p.to_dict()
        p2 = Plan.from_dict(d)
        assert p2.workflow_id == "wf-1"
        assert p2.rollback_capability is True
        assert len(p2.remaining_steps) == 1


# ── ReasoningArtifact ──────────────────────────────────────────────────


class TestReasoningArtifact:
    def test_create_minimal(self):
        art = _make_artifact()
        assert art.protocol_version == PROTOCOL_VERSION
        assert art.artifact_id.startswith("ra-")
        assert art.timestamp  # auto-generated

    def test_to_dict_roundtrip(self):
        art = _make_artifact()
        d = art.to_dict()
        art2 = ReasoningArtifact.from_dict(d)
        assert art2.agent_id == art.agent_id
        assert art2.goal == art.goal
        assert art2.overall_confidence == art.overall_confidence
        assert len(art2.factors) == len(art.factors)

    def test_json_roundtrip(self):
        art = _make_artifact()
        j = art.to_json()
        art2 = ReasoningArtifact.from_json(j)
        assert art2.artifact_id == art.artifact_id

    def test_hash_deterministic(self):
        art = _make_artifact()
        h1 = art.hash()
        h2 = art.hash()
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_hash_changes_with_content(self):
        art1 = _make_artifact(goal="Goal A")
        art2 = _make_artifact(goal="Goal B")
        assert art1.hash() != art2.hash()

    def test_with_plan(self):
        plan = Plan(workflow_id="wf-1", total_steps=2, current_step=1, step_description="Step 1")
        art = _make_artifact(plan=plan)
        d = art.to_dict()
        assert "plan" in d
        art2 = ReasoningArtifact.from_dict(d)
        assert art2.plan is not None
        assert art2.plan.workflow_id == "wf-1"

    def test_with_optional_fields(self):
        art = _make_artifact(
            certificate_id="cert-1",
            envelope_id="env-1",
            session_id="sess-1",
            origin_id="usr_hash",
            narrative="Test narrative",
        )
        d = art.to_dict()
        assert d["identity"]["certificate_id"] == "cert-1"
        assert d["task"]["origin_id"] == "usr_hash"
        assert d["reasoning"]["narrative"] == "Test narrative"

    def test_example_from_spec(self):
        """Test that the example from the protocol spec can be parsed."""
        example = {
            "protocol_version": "0.1.0",
            "artifact_id": "ra-20260213-a7b3c",
            "timestamp": "2026-02-13T22:15:00Z",
            "identity": {
                "agent_id": "cs-agent-47",
                "certificate_id": "cert-acme-cs-47",
                "envelope_id": "env-returns-gold",
                "session_id": "sess-9f8e7d",
            },
            "task": {
                "goal": "Process return request for order #ORD-88421",
                "origin": "user_request",
                "origin_id": "usr_hash_3a9f2b",
                "constraints_identified": [
                    {"type": "policy", "description": "Standard return limit is $500", "source": "policy://returns/standard-limit"},
                    {"type": "authority", "description": "Gold-tier conditional authority extends to $1500", "source": "envelope://env-returns-gold"},
                    {"type": "temporal", "description": "Return must be within 30-day purchase window", "source": "policy://returns/time-limit"},
                ],
            },
            "reasoning": {
                "factors": [
                    {"id": "f1", "type": "constraint", "description": "Return amount ($800) exceeds standard $500 limit", "source": "policy://returns/standard-limit", "assessment": "Standard authority insufficient.", "influence": "decisive", "confidence": 1.0},
                    {"id": "f2", "type": "context", "description": "Customer is Gold tier", "source": "data://customer/cust-44291/profile", "assessment": "Qualifies for Gold-tier", "influence": "decisive", "confidence": 0.99},
                    {"id": "f3", "type": "evidence", "description": "Order placed 12 days ago", "source": "data://orders/ORD-88421", "assessment": "Within window", "influence": "significant", "confidence": 1.0},
                ],
                "alternatives_considered": [
                    {"method": "deny", "context": "Return and cite standard $500 limit", "reason_rejected": "Customer qualifies for exception."},
                    {"method": "escalate", "context": "Entire request to human reviewer", "reason_rejected": "Amount within conditional authority envelope."},
                ],
                "narrative": "Customer requests $800 return.",
            },
            "decision": {
                "intended_action": {
                    "method": "approve",
                    "target": "order/ORD-88421",
                    "context": "Return request, gold-tier conditional authority exception",
                    "parameters": {"amount": 800.00, "reason": "gold_tier_exception"},
                },
                "justifications": [
                    {"factor_id": "f1", "explanation": "Amount exceeds standard limit"},
                    {"factor_id": "f2", "explanation": "Gold tier activates conditional authority"},
                ],
                "authority_claim": {
                    "envelope_type": "conditional",
                    "conditions_met": ["Customer loyalty tier: Gold", "Amount within range"],
                    "limit_reference": "Gold-tier return authority: up to $1500",
                },
            },
            "uncertainty": {
                "unknowns": [{"description": "Physical condition of returned product", "impact": "May not meet quality criteria"}],
                "assumptions": [{"description": "Customer's loyalty tier data is current", "basis": "Real-time system", "risk_if_wrong": "Conditional authority may not apply"}],
                "overall_confidence": 0.88,
            },
        }
        art = ReasoningArtifact.from_dict(example)
        assert art.agent_id == "cs-agent-47"
        assert len(art.constraints_identified) == 3
        assert len(art.factors) == 3
        assert len(art.alternatives_considered) == 2
        assert art.intended_action.method == "approve"
        assert art.overall_confidence == 0.88


# ── Validation ─────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_artifact(self):
        art = _make_artifact()
        errors = validate_artifact(art)
        assert errors == []

    def test_missing_agent_id(self):
        art = _make_artifact(agent_id="")
        errors = validate_artifact(art)
        assert any("agent_id" in e for e in errors)

    def test_missing_goal(self):
        art = _make_artifact(goal="")
        errors = validate_artifact(art)
        assert any("goal" in e for e in errors)

    def test_invalid_origin(self):
        art = _make_artifact(origin="invalid_origin")
        errors = validate_artifact(art)
        assert any("origin" in e for e in errors)

    def test_no_factors(self):
        art = _make_artifact(factors=[])
        errors = validate_artifact(art)
        assert any("at least one factor" in e for e in errors)

    def test_no_constraint_factor(self):
        art = _make_artifact(
            factors=[
                Factor(id="f1", type="context", description="X", source="test",
                       assessment="OK", influence="decisive", confidence=0.9),
            ],
        )
        errors = validate_artifact(art)
        assert any("constraint" in e.lower() for e in errors)

    def test_duplicate_factor_ids(self):
        art = _make_artifact(
            factors=[
                Factor(id="f1", type="constraint", description="A", source="test",
                       assessment="OK", influence="decisive", confidence=0.9),
                Factor(id="f1", type="context", description="B", source="test",
                       assessment="OK", influence="minor", confidence=0.8),
            ],
        )
        errors = validate_artifact(art)
        assert any("duplicate" in e for e in errors)

    def test_invalid_method(self):
        art = _make_artifact(
            intended_action=IntendedAction(method="nonexistent_method", target="x"),
        )
        errors = validate_artifact(art)
        assert any("method" in e for e in errors)

    def test_missing_target(self):
        art = _make_artifact(
            intended_action=IntendedAction(method="approve", target=""),
        )
        errors = validate_artifact(art)
        assert any("target" in e for e in errors)

    def test_invalid_justification_reference(self):
        art = _make_artifact(
            justifications=[
                Justification(factor_id="nonexistent", explanation="Bad ref"),
            ],
        )
        errors = validate_artifact(art)
        assert any("nonexistent" in e for e in errors)

    def test_invalid_envelope_type(self):
        art = _make_artifact(
            authority_claim=AuthorityClaim(envelope_type="invalid_type"),
        )
        errors = validate_artifact(art)
        assert any("envelope_type" in e for e in errors)

    def test_confidence_out_of_range(self):
        art = _make_artifact(overall_confidence=1.5)
        errors = validate_artifact(art)
        assert any("overall_confidence" in e for e in errors)

    def test_invalid_plan(self):
        art = _make_artifact(
            plan=Plan(workflow_id="wf-1", total_steps=2, current_step=3, step_description="Bad"),
        )
        errors = validate_artifact(art)
        assert any("current_step" in e for e in errors)


# ── Assessment ─────────────────────────────────────────────────────────


class TestAssessment:
    def test_roundtrip(self):
        a = Assessment(
            completeness_score=0.95,
            completeness_detail="Good",
            authority_verified=True,
            authority_detail="Verified",
            alignment_score=0.98,
            uncertainty_calibration_score=0.9,
            alternatives_adequacy_score=0.85,
            dimensional_summary={"scope_compliance": 1.0},
            ucs=0.847,
            trust_state=0.82,
        )
        d = a.to_dict()
        a2 = Assessment.from_dict(d)
        assert a2.completeness_score == 0.95
        assert a2.ucs == 0.847


# ── GovernanceResponseData ─────────────────────────────────────────────


class TestGovernanceResponseData:
    def test_proceed_response(self):
        r = GovernanceResponseData(
            verdict=ProtocolVerdict.PROCEED,
            assessment=Assessment(completeness_score=0.95),
            metadata=ResponseMetadata(
                evaluator_id="test", evaluation_time_ms=12.4,
                config_version="cfg_1", timestamp="2026-02-13T22:15:00Z",
                artifact_hash="sha256:abc",
            ),
            token="jwt.token.here",
        )
        d = r.to_dict()
        assert d["verdict"] == "PROCEED"
        assert d["token"] == "jwt.token.here"

    def test_revise_response(self):
        r = GovernanceResponseData(
            verdict=ProtocolVerdict.REVISE,
            assessment=Assessment(completeness_score=0.3),
            metadata=ResponseMetadata(
                evaluator_id="test", evaluation_time_ms=5.0,
                config_version="cfg_1", timestamp="2026-02-13T22:15:00Z",
                artifact_hash="sha256:abc",
            ),
            guidance=Guidance(
                reasoning_gaps=["Missing risk assessment"],
                recommended_factors=["risk"],
            ),
        )
        d = r.to_dict()
        assert d["verdict"] == "REVISE"
        assert "guidance" in d

    def test_deny_response(self):
        r = GovernanceResponseData(
            verdict=ProtocolVerdict.DENY,
            assessment=Assessment(completeness_score=0.5),
            metadata=ResponseMetadata(
                evaluator_id="test", evaluation_time_ms=8.0,
                config_version="cfg_1", timestamp="2026-02-13T22:15:00Z",
                artifact_hash="sha256:abc",
            ),
            denial=Denial(
                grounds=["Authority exceeded"],
                veto_dimensions=["scope_compliance"],
            ),
        )
        d = r.to_dict()
        assert d["verdict"] == "DENY"
        assert "denial" in d

    def test_roundtrip(self):
        r = GovernanceResponseData(
            verdict=ProtocolVerdict.PROCEED_WITH_CONDITIONS,
            assessment=Assessment(completeness_score=0.8),
            metadata=ResponseMetadata(
                evaluator_id="test", evaluation_time_ms=10.0,
                config_version="cfg_1", timestamp="2026-02-13T22:15:00Z",
                artifact_hash="sha256:abc",
            ),
            conditions=[
                Condition(type="monitoring", description="Enhanced monitoring"),
            ],
        )
        d = r.to_dict()
        r2 = GovernanceResponseData.from_dict(d)
        assert r2.verdict == ProtocolVerdict.PROCEED_WITH_CONDITIONS
        assert len(r2.conditions) == 1


# ── ProtocolVerdict ────────────────────────────────────────────────────


class TestProtocolVerdict:
    def test_all_verdicts(self):
        assert ProtocolVerdict.PROCEED.value == "PROCEED"
        assert ProtocolVerdict.PROCEED_WITH_CONDITIONS.value == "PROCEED_WITH_CONDITIONS"
        assert ProtocolVerdict.REVISE.value == "REVISE"
        assert ProtocolVerdict.ESCALATE.value == "ESCALATE"
        assert ProtocolVerdict.DENY.value == "DENY"

    def test_from_string(self):
        assert ProtocolVerdict("PROCEED") == ProtocolVerdict.PROCEED
        assert ProtocolVerdict("DENY") == ProtocolVerdict.DENY
