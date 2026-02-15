"""Tests for audit record hash chaining and chain verification."""

import time

import pytest

from nomotic.audit import AuditRecord, AuditTrail, verify_chain


def _make_record(
    *,
    agent_id: str = "agent-1",
    verdict: str = "ALLOW",
    timestamp: float | None = None,
) -> AuditRecord:
    return AuditRecord(
        record_id=f"rec-{time.time_ns() % 100000:05d}",
        timestamp=timestamp or time.time(),
        context_code="GOVERNANCE.ALLOW",
        severity="info",
        agent_id=agent_id,
        owner_id="owner",
        user_id="user",
        action_id=f"act-{time.time_ns() % 100000:05d}",
        action_type="read",
        action_target="/api/data",
        verdict=verdict,
        ucs=0.8,
        tier=2,
        trust_score=0.7,
        trust_trend="stable",
    )


class TestHashChainFields:
    """Test hash chain fields on AuditRecord."""

    def test_compute_hash_deterministic(self):
        record = _make_record(timestamp=1000.0)
        record.record_id = "fixed-id"
        record.action_id = "fixed-action"
        h1 = record.compute_hash()
        h2 = record.compute_hash()
        assert h1 == h2

    def test_compute_hash_changes_with_content(self):
        r1 = _make_record(timestamp=1000.0)
        r1.record_id = "id-1"
        r1.action_id = "act-1"

        r2 = _make_record(timestamp=1000.0)
        r2.record_id = "id-1"
        r2.action_id = "act-1"
        r2.verdict = "DENY"

        assert r1.compute_hash() != r2.compute_hash()

    def test_compute_hash_includes_previous_hash(self):
        record = _make_record(timestamp=1000.0)
        record.record_id = "id-1"
        record.action_id = "act-1"

        record.previous_hash = ""
        h1 = record.compute_hash()

        record.previous_hash = "abc123"
        h2 = record.compute_hash()

        assert h1 != h2

    def test_hash_fields_in_to_dict(self):
        record = _make_record()
        record.previous_hash = "prev-hash"
        record.record_hash = "rec-hash"
        d = record.to_dict()
        assert d["previous_hash"] == "prev-hash"
        assert d["record_hash"] == "rec-hash"

    def test_hash_fields_from_dict(self):
        record = _make_record()
        record.previous_hash = "prev-hash"
        record.record_hash = "rec-hash"
        d = record.to_dict()
        restored = AuditRecord.from_dict(d)
        assert restored.previous_hash == "prev-hash"
        assert restored.record_hash == "rec-hash"

    def test_from_dict_defaults_hash_fields(self):
        """Old records without hash fields should deserialize with empty strings."""
        d = {
            "record_id": "r1",
            "timestamp": 1000.0,
            "context_code": "GOVERNANCE.ALLOW",
            "severity": "info",
            "agent_id": "a1",
            "owner_id": "",
            "user_id": "",
            "action_id": "act1",
            "action_type": "read",
            "action_target": "/",
            "verdict": "ALLOW",
            "ucs": 0.8,
            "tier": 2,
        }
        record = AuditRecord.from_dict(d)
        assert record.previous_hash == ""
        assert record.record_hash == ""


class TestAuditTrailHashChaining:
    """Test that AuditTrail correctly chains hashes on append."""

    def test_first_record_has_empty_previous_hash(self):
        trail = AuditTrail()
        record = _make_record()
        trail.append(record)
        assert record.previous_hash == ""
        assert record.record_hash != ""

    def test_second_record_links_to_first(self):
        trail = AuditTrail()
        r1 = _make_record(timestamp=1.0)
        r2 = _make_record(timestamp=2.0)
        trail.append(r1)
        trail.append(r2)
        assert r2.previous_hash == r1.record_hash
        assert r2.record_hash != r1.record_hash

    def test_chain_of_three(self):
        trail = AuditTrail()
        records = [_make_record(timestamp=float(i)) for i in range(3)]
        for r in records:
            trail.append(r)

        assert records[0].previous_hash == ""
        assert records[1].previous_hash == records[0].record_hash
        assert records[2].previous_hash == records[1].record_hash

    def test_hash_is_verifiable(self):
        trail = AuditTrail()
        record = _make_record()
        trail.append(record)
        assert record.record_hash == record.compute_hash()


class TestVerifyChain:
    """Test the verify_chain function."""

    def test_empty_chain_valid(self):
        valid, msg = verify_chain([])
        assert valid is True
        assert msg == ""

    def test_single_record_valid(self):
        trail = AuditTrail()
        r = _make_record()
        trail.append(r)
        valid, msg = verify_chain([r])
        assert valid is True

    def test_valid_chain(self):
        trail = AuditTrail()
        records = [_make_record(timestamp=float(i)) for i in range(5)]
        for r in records:
            trail.append(r)
        valid, msg = verify_chain(records)
        assert valid is True
        assert msg == ""

    def test_tampered_record_detected(self):
        trail = AuditTrail()
        records = [_make_record(timestamp=float(i)) for i in range(3)]
        for r in records:
            trail.append(r)

        # Tamper with the second record
        records[1].verdict = "DENY"
        valid, msg = verify_chain(records)
        assert valid is False
        assert "Hash mismatch" in msg

    def test_broken_chain_detected(self):
        trail = AuditTrail()
        records = [_make_record(timestamp=float(i)) for i in range(3)]
        for r in records:
            trail.append(r)

        # Break the chain by modifying previous_hash
        records[2].previous_hash = "wrong-hash"
        # Also need to recompute record_hash with wrong previous_hash
        records[2].record_hash = records[2].compute_hash()
        valid, msg = verify_chain(records)
        assert valid is False
        assert "previous_hash mismatch" in msg

    def test_pre_upgrade_records_skipped(self):
        """Records without hash chain data (pre-upgrade) should be skipped."""
        records = [_make_record(timestamp=float(i)) for i in range(3)]
        # Leave hash fields empty (as if from old version)
        valid, msg = verify_chain(records)
        assert valid is True
