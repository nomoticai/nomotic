"""Tests for configuration provenance — Git for governance rules."""

import time

import pytest

from nomotic.provenance import ProvenanceLog, ProvenanceRecord


class TestProvenanceRecord:
    """ProvenanceRecord serialization tests."""

    def test_to_dict(self):
        rec = ProvenanceRecord(
            record_id="rec-001",
            timestamp=1000.0,
            actor="alice@acme.com",
            actor_type="human",
            target_type="scope",
            target_id="agent-1",
            change_type="modify",
            previous_value=["read"],
            new_value=["read", "write"],
            reason="Adding write access",
            ticket="OPS-123",
            context_code="CONFIG.SCOPE_CHANGED",
        )
        d = rec.to_dict()
        assert d["actor"] == "alice@acme.com"
        assert d["target_type"] == "scope"
        assert d["change_type"] == "modify"
        assert d["new_value"] == ["read", "write"]
        assert d["ticket"] == "OPS-123"

    def test_from_dict_roundtrip(self):
        rec = ProvenanceRecord(
            record_id="rec-002",
            timestamp=2000.0,
            actor="system",
            actor_type="system",
            target_type="rule",
            target_id="ethical_rule",
            change_type="add",
            new_value="no_harm_rule",
            reason="Adding safety rule",
        )
        d = rec.to_dict()
        restored = ProvenanceRecord.from_dict(d)
        assert restored.actor == "system"
        assert restored.target_type == "rule"
        assert restored.new_value == "no_harm_rule"

    def test_frozen(self):
        rec = ProvenanceRecord(
            record_id="rec-003",
            timestamp=3000.0,
            actor="test",
            actor_type="human",
            target_type="scope",
            target_id="agent-1",
            change_type="add",
        )
        with pytest.raises(AttributeError):
            rec.actor = "changed"  # type: ignore[misc]


class TestProvenanceLog:
    """ProvenanceLog recording and querying tests."""

    def test_record_creates_record(self):
        log = ProvenanceLog()
        rec = log.record(
            actor="alice@acme.com",
            target_type="scope",
            target_id="agent-1",
            change_type="add",
            new_value=["read", "write"],
        )
        assert rec.actor == "alice@acme.com"
        assert rec.record_id  # not empty
        assert rec.timestamp > 0

    def test_query_all(self):
        log = ProvenanceLog()
        log.record("alice", "scope", "agent-1", "add")
        log.record("bob", "rule", "ethical_rule", "add")
        results = log.query()
        assert len(results) == 2

    def test_query_filter_actor(self):
        log = ProvenanceLog()
        log.record("alice", "scope", "agent-1", "add")
        log.record("bob", "scope", "agent-2", "add")
        results = log.query(actor="alice")
        assert len(results) == 1
        assert results[0].actor == "alice"

    def test_query_filter_target_type(self):
        log = ProvenanceLog()
        log.record("alice", "scope", "agent-1", "add")
        log.record("alice", "rule", "rule-1", "add")
        results = log.query(target_type="scope")
        assert len(results) == 1

    def test_query_filter_target_id(self):
        log = ProvenanceLog()
        log.record("alice", "scope", "agent-1", "add")
        log.record("alice", "scope", "agent-2", "add")
        results = log.query(target_id="agent-1")
        assert len(results) == 1

    def test_query_filter_change_type(self):
        log = ProvenanceLog()
        log.record("alice", "scope", "agent-1", "add")
        log.record("alice", "scope", "agent-1", "modify")
        results = log.query(change_type="modify")
        assert len(results) == 1

    def test_query_newest_first(self):
        log = ProvenanceLog()
        log.record("alice", "scope", "agent-1", "add")
        time.sleep(0.01)
        log.record("bob", "scope", "agent-2", "add")
        results = log.query()
        assert results[0].actor == "bob"

    def test_history(self):
        log = ProvenanceLog()
        log.record("alice", "scope", "agent-1", "add", new_value=["read"])
        log.record("bob", "scope", "agent-2", "add", new_value=["read"])
        log.record("alice", "scope", "agent-1", "modify", new_value=["read", "write"])
        history = log.history("scope", "agent-1")
        assert len(history) == 2
        assert history[0].change_type == "add"
        assert history[1].change_type == "modify"

    def test_current_config_version_empty(self):
        log = ProvenanceLog()
        v = log.current_config_version()
        assert v == "0" * 12

    def test_current_config_version_changes(self):
        log = ProvenanceLog()
        v1 = log.current_config_version()
        log.record("alice", "scope", "agent-1", "add")
        v2 = log.current_config_version()
        assert v1 != v2
        log.record("bob", "rule", "rule-1", "add")
        v3 = log.current_config_version()
        assert v2 != v3

    def test_max_records_cap(self):
        log = ProvenanceLog(max_records=3)
        for i in range(5):
            log.record("alice", "scope", f"agent-{i}", "add")
        assert len(log.records) == 3

    def test_clear(self):
        log = ProvenanceLog()
        log.record("alice", "scope", "agent-1", "add")
        log.clear()
        assert len(log.records) == 0

    def test_records_property_newest_first(self):
        log = ProvenanceLog()
        log.record("alice", "scope", "agent-1", "add")
        time.sleep(0.01)
        log.record("bob", "scope", "agent-2", "add")
        records = log.records
        assert records[0].actor == "bob"

    def test_record_with_reason_and_ticket(self):
        log = ProvenanceLog()
        rec = log.record(
            "alice", "scope", "agent-1", "modify",
            reason="Security review",
            ticket="SEC-456",
        )
        assert rec.reason == "Security review"
        assert rec.ticket == "SEC-456"
