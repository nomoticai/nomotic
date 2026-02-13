"""Tests for audit trail — structured governance event logging."""

import json
import threading
import time

import pytest

from nomotic.audit import AuditRecord, AuditTrail, build_justification
from nomotic.context import CODES
from nomotic.types import (
    Action,
    AgentContext,
    DimensionScore,
    GovernanceVerdict,
    TrustProfile,
    Verdict,
)


def _make_record(
    *,
    agent_id: str = "agent-1",
    context_code: str = "GOVERNANCE.ALLOW",
    severity: str = "info",
    verdict: str = "ALLOW",
    ucs: float = 0.8,
    tier: int = 2,
    trust_score: float = 0.7,
    timestamp: float | None = None,
    user_id: str = "",
    owner_id: str = "",
    action_type: str = "read",
    action_target: str = "/api/data",
) -> AuditRecord:
    return AuditRecord(
        record_id=f"rec-{time.time_ns() % 100000:05d}",
        timestamp=timestamp or time.time(),
        context_code=context_code,
        severity=severity,
        agent_id=agent_id,
        owner_id=owner_id,
        user_id=user_id,
        action_id=f"act-{time.time_ns() % 100000:05d}",
        action_type=action_type,
        action_target=action_target,
        verdict=verdict,
        ucs=ucs,
        tier=tier,
        trust_score=trust_score,
        trust_trend="stable",
    )


class TestAuditRecord:
    """AuditRecord serialization tests."""

    def test_to_dict(self):
        record = _make_record()
        d = record.to_dict()
        assert d["agent_id"] == "agent-1"
        assert d["context_code"] == "GOVERNANCE.ALLOW"
        assert d["verdict"] == "ALLOW"
        assert isinstance(d["ucs"], float)

    def test_from_dict_roundtrip(self):
        record = _make_record(
            agent_id="agent-x",
            verdict="DENY",
            ucs=0.15,
            tier=1,
            context_code="GOVERNANCE.VETO",
            severity="alert",
        )
        record.drift_overall = 0.45
        record.drift_severity = "high"
        d = record.to_dict()
        restored = AuditRecord.from_dict(d)
        assert restored.agent_id == "agent-x"
        assert restored.verdict == "DENY"
        assert restored.ucs == pytest.approx(0.15, abs=0.001)
        assert restored.drift_severity == "high"
        assert restored.drift_overall == pytest.approx(0.45, abs=0.001)

    def test_metadata_in_dict(self):
        record = _make_record()
        record.metadata = {"session_id": "sess-123"}
        d = record.to_dict()
        assert d["metadata"]["session_id"] == "sess-123"


class TestAuditTrail:
    """AuditTrail append, query, summary tests."""

    def test_append_and_query(self):
        trail = AuditTrail()
        record = _make_record()
        trail.append(record)
        results = trail.query()
        assert len(results) == 1
        assert results[0].agent_id == "agent-1"

    def test_query_no_filters_returns_all(self):
        trail = AuditTrail()
        for i in range(5):
            trail.append(_make_record(agent_id=f"agent-{i}"))
        results = trail.query()
        assert len(results) == 5

    def test_query_filter_agent_id(self):
        trail = AuditTrail()
        trail.append(_make_record(agent_id="agent-a"))
        trail.append(_make_record(agent_id="agent-b"))
        trail.append(_make_record(agent_id="agent-a"))
        results = trail.query(agent_id="agent-a")
        assert len(results) == 2
        assert all(r.agent_id == "agent-a" for r in results)

    def test_query_filter_context_code(self):
        trail = AuditTrail()
        trail.append(_make_record(context_code="GOVERNANCE.ALLOW", severity="info"))
        trail.append(_make_record(context_code="GOVERNANCE.DENY", severity="warning"))
        results = trail.query(context_code="GOVERNANCE.DENY")
        assert len(results) == 1

    def test_query_filter_category(self):
        trail = AuditTrail()
        trail.append(_make_record(context_code="GOVERNANCE.ALLOW", severity="info"))
        trail.append(_make_record(context_code="SECURITY.INJECTION_ATTEMPT", severity="critical"))
        results = trail.query(category="SECURITY")
        assert len(results) == 1

    def test_query_filter_severity(self):
        trail = AuditTrail()
        trail.append(_make_record(severity="info"))
        trail.append(_make_record(severity="warning"))
        trail.append(_make_record(severity="critical"))
        results = trail.query(severity="critical")
        assert len(results) == 1

    def test_query_filter_verdict(self):
        trail = AuditTrail()
        trail.append(_make_record(verdict="ALLOW"))
        trail.append(_make_record(verdict="DENY"))
        results = trail.query(verdict="DENY")
        assert len(results) == 1

    def test_query_filter_time_range(self):
        trail = AuditTrail()
        t1 = time.time() - 100
        t2 = time.time() - 50
        t3 = time.time()
        trail.append(_make_record(timestamp=t1))
        trail.append(_make_record(timestamp=t2))
        trail.append(_make_record(timestamp=t3))
        results = trail.query(since=t2 - 1, until=t2 + 1)
        assert len(results) == 1

    def test_query_combined_filters(self):
        trail = AuditTrail()
        trail.append(_make_record(agent_id="a", verdict="ALLOW"))
        trail.append(_make_record(agent_id="a", verdict="DENY"))
        trail.append(_make_record(agent_id="b", verdict="DENY"))
        results = trail.query(agent_id="a", verdict="DENY")
        assert len(results) == 1

    def test_query_newest_first(self):
        trail = AuditTrail()
        t1 = time.time() - 2
        t2 = time.time() - 1
        t3 = time.time()
        trail.append(_make_record(timestamp=t1))
        trail.append(_make_record(timestamp=t2))
        trail.append(_make_record(timestamp=t3))
        results = trail.query()
        assert results[0].timestamp >= results[1].timestamp
        assert results[1].timestamp >= results[2].timestamp

    def test_query_limit(self):
        trail = AuditTrail()
        for _ in range(10):
            trail.append(_make_record())
        results = trail.query(limit=3)
        assert len(results) == 3

    def test_max_records_cap(self):
        trail = AuditTrail(max_records=5)
        for i in range(10):
            trail.append(_make_record(agent_id=f"agent-{i}"))
        assert len(trail.records) == 5

    def test_count(self):
        trail = AuditTrail()
        trail.append(_make_record(verdict="ALLOW"))
        trail.append(_make_record(verdict="DENY"))
        trail.append(_make_record(verdict="ALLOW"))
        assert trail.count(verdict="ALLOW") == 2
        assert trail.count(verdict="DENY") == 1

    def test_summary(self):
        trail = AuditTrail()
        trail.append(_make_record(verdict="ALLOW", severity="info", context_code="GOVERNANCE.ALLOW"))
        trail.append(_make_record(verdict="DENY", severity="warning", context_code="GOVERNANCE.DENY"))
        trail.append(_make_record(verdict="DENY", severity="alert", context_code="GOVERNANCE.VETO"))
        summary = trail.summary()
        assert summary["total_records"] == 3
        assert summary["by_verdict"]["ALLOW"] == 1
        assert summary["by_verdict"]["DENY"] == 2
        assert summary["by_severity"]["info"] == 1
        assert "GOVERNANCE" in summary["by_category"]
        assert len(summary["recent_alerts"]) == 1

    def test_summary_with_agent_filter(self):
        trail = AuditTrail()
        trail.append(_make_record(agent_id="a"))
        trail.append(_make_record(agent_id="b"))
        summary = trail.summary(agent_id="a")
        assert summary["total_records"] == 1

    def test_export_json(self):
        trail = AuditTrail()
        trail.append(_make_record())
        exported = trail.export("json")
        assert isinstance(exported, str)
        parsed = json.loads(exported)
        assert len(parsed) == 1

    def test_export_dicts(self):
        trail = AuditTrail()
        trail.append(_make_record())
        exported = trail.export("dicts")
        assert isinstance(exported, list)
        assert len(exported) == 1
        assert isinstance(exported[0], dict)

    def test_clear(self):
        trail = AuditTrail()
        trail.append(_make_record())
        trail.clear()
        assert len(trail.records) == 0

    def test_records_property_newest_first(self):
        trail = AuditTrail()
        trail.append(_make_record(timestamp=1.0))
        trail.append(_make_record(timestamp=2.0))
        records = trail.records
        assert records[0].timestamp == 2.0

    def test_thread_safety(self):
        trail = AuditTrail()
        errors = []

        def append_records(n):
            try:
                for _ in range(n):
                    trail.append(_make_record())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=append_records, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(trail.records) == 200


class TestBuildJustification:
    """build_justification tests."""

    def test_allow_verdict(self):
        verdict = GovernanceVerdict(
            action_id="act-1",
            verdict=Verdict.ALLOW,
            ucs=0.82,
            tier=2,
            dimension_scores=[
                DimensionScore("scope_compliance", 1.0, 1.5, reasoning="Within scope"),
                DimensionScore("behavioral_consistency", 0.95, 1.0, reasoning="Consistent"),
            ],
        )
        action = Action(agent_id="agent-1", action_type="read", target="/api/data")
        context = AgentContext(agent_id="agent-1", trust_profile=TrustProfile(agent_id="agent-1", overall_trust=0.78))
        result = build_justification(verdict, action, context)
        assert "ALLOW" in result
        assert "0.82" in result
        assert "Tier 2" in result
        assert "read" in result

    def test_deny_verdict_with_veto(self):
        verdict = GovernanceVerdict(
            action_id="act-1",
            verdict=Verdict.DENY,
            ucs=0.15,
            tier=1,
            vetoed_by=["scope_compliance"],
            dimension_scores=[
                DimensionScore("scope_compliance", 0.0, 1.5, veto=True, reasoning="Out of scope"),
            ],
        )
        action = Action(agent_id="agent-1", action_type="delete", target="/api/data")
        context = AgentContext(agent_id="agent-1", trust_profile=TrustProfile(agent_id="agent-1", overall_trust=0.42))
        result = build_justification(verdict, action, context)
        assert "DENY" in result
        assert "veto" in result
        assert "VETOED" in result
        assert "scope_compliance" in result

    def test_includes_drift_info(self):
        verdict = GovernanceVerdict(
            action_id="act-1",
            verdict=Verdict.DENY,
            ucs=0.2,
            tier=2,
            dimension_scores=[],
        )
        action = Action(agent_id="agent-1", action_type="read", target="/api/data")
        context = AgentContext(agent_id="agent-1", trust_profile=TrustProfile(agent_id="agent-1"))

        class FakeDrift:
            overall = 0.45
            severity = "high"

        result = build_justification(verdict, action, context, FakeDrift())
        assert "HIGH" in result or "high" in result.lower()

    def test_no_drift(self):
        verdict = GovernanceVerdict(
            action_id="act-1",
            verdict=Verdict.ALLOW,
            ucs=0.9,
            tier=2,
            dimension_scores=[],
        )
        action = Action(agent_id="agent-1", action_type="read", target="/api/data")
        context = AgentContext(agent_id="agent-1", trust_profile=TrustProfile(agent_id="agent-1"))
        result = build_justification(verdict, action, context, None)
        assert "ALLOW" in result

    def test_handles_missing_optional_data(self):
        verdict = GovernanceVerdict(
            action_id="act-1",
            verdict=Verdict.ALLOW,
            ucs=0.9,
            tier=2,
            dimension_scores=[],
        )
        action = Action(agent_id="", action_type="", target="")
        context = AgentContext(agent_id="", trust_profile=TrustProfile(agent_id=""))
        result = build_justification(verdict, action, context)
        assert isinstance(result, str)
        assert len(result) > 0
