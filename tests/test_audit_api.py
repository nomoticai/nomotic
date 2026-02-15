"""Tests for persistent audit API endpoints."""

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from nomotic.api import NomoticAPIServer
from nomotic.audit_store import AuditStore, PersistentAuditRecord
from nomotic.authority import CertificateAuthority
from nomotic.keys import SigningKey
from nomotic.registry import ArchetypeRegistry, OrganizationRegistry, ZoneValidator
from nomotic.store import MemoryCertificateStore


# ── Helpers ──────────────────────────────────────────────────────────────


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _seed_audit_records(
    base_dir: Path,
    agent_id: str,
    count: int = 5,
    *,
    start_time: float | None = None,
) -> list[PersistentAuditRecord]:
    """Create test audit records with proper hash chaining."""
    store = AuditStore(base_dir)
    prev_hash = ""
    records: list[PersistentAuditRecord] = []
    ts = start_time or time.time()
    for i in range(count):
        data: dict[str, Any] = {
            "record_id": f"rec{i:04d}",
            "timestamp": ts + i,
            "agent_id": agent_id,
            "action_type": "read" if i % 2 == 0 else "write",
            "action_target": "test_db",
            "verdict": "ALLOW" if i < count - 1 else "DENY",
            "ucs": 0.9 if i < count - 1 else 0.0,
            "tier": 2 if i < count - 1 else 1,
            "trust_score": 0.5 + (i * 0.01),
            "trust_delta": 0.01 if i < count - 1 else -0.05,
            "trust_trend": "rising" if i < count - 1 else "falling",
            "severity": "info" if i < count - 1 else "alert",
            "justification": f"Test record {i}",
            "vetoed_by": [] if i < count - 1 else ["scope_compliance"],
            "dimension_scores": {"scope_compliance": 1.0},
            "parameters": {},
            "source": "",
            "previous_hash": prev_hash,
            "record_hash": "",
        }
        data["record_hash"] = store.compute_hash(data, prev_hash)
        prev_hash = data["record_hash"]
        rec = PersistentAuditRecord(**data)
        store.append(rec)
        records.append(rec)
    return records


class _APIClient:
    """Minimal HTTP client for testing."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def get(self, path: str) -> tuple[int, bytes]:
        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def get_json(self, path: str) -> tuple[int, dict[str, Any]]:
        status, body = self.get(path)
        return status, json.loads(body)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def audit_env(tmp_path):
    """Start a test server with a temp base_dir and seeded audit data."""
    sk, _vk = SigningKey.generate()
    store = MemoryCertificateStore()
    ca = CertificateAuthority(issuer_id="test-issuer", signing_key=sk, store=store)

    port = _find_free_port()
    server = NomoticAPIServer(
        ca,
        archetype_registry=ArchetypeRegistry.with_defaults(),
        zone_validator=ZoneValidator(),
        org_registry=OrganizationRegistry(),
        base_dir=tmp_path,
        host="127.0.0.1",
        port=port,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)

    client = _APIClient(f"http://127.0.0.1:{port}")
    yield client, tmp_path, ca
    server.shutdown()


# ── Store-level tests (no server needed) ─────────────────────────────────


class TestAuditStoreDirectly:
    """Validate audit store operations used by the API handlers."""

    def test_records_query(self, tmp_path):
        _seed_audit_records(tmp_path, "TestBot", 5)
        store = AuditStore(tmp_path)
        records = store.query("TestBot")
        assert len(records) == 5

    def test_verify_valid_chain(self, tmp_path):
        _seed_audit_records(tmp_path, "TestBot", 10)
        store = AuditStore(tmp_path)
        valid, count, msg = store.verify_chain("TestBot")
        assert valid is True
        assert count == 10

    def test_verify_tampered_chain(self, tmp_path):
        _seed_audit_records(tmp_path, "TestBot", 5)
        audit_file = tmp_path / "audit" / "testbot.jsonl"
        lines = audit_file.read_text().strip().split("\n")
        record = json.loads(lines[2])
        record["verdict"] = "TAMPERED"
        lines[2] = json.dumps(record, separators=(",", ":"))
        audit_file.write_text("\n".join(lines) + "\n")

        store = AuditStore(tmp_path)
        valid, count, msg = store.verify_chain("TestBot")
        assert valid is False
        assert "TAMPERING" in msg

    def test_case_insensitive_lookup(self, tmp_path):
        _seed_audit_records(tmp_path, "TestBot", 3)
        store = AuditStore(tmp_path)
        assert len(store.query("testbot")) == 3
        assert len(store.query("TESTBOT")) == 3

    def test_summary(self, tmp_path):
        _seed_audit_records(tmp_path, "TestBot", 5)
        store = AuditStore(tmp_path)
        summary = store.summary("TestBot")
        assert summary["total"] == 5
        assert "ALLOW" in summary["by_verdict"]

    def test_seal_consistent(self, tmp_path):
        _seed_audit_records(tmp_path, "TestBot", 5)
        store = AuditStore(tmp_path)
        seal = store.seal("TestBot")
        assert seal.startswith("sha256:")
        assert store.seal("TestBot") == seal


# ── API endpoint tests ───────────────────────────────────────────────────


class TestAuditRecordsEndpoint:

    def test_returns_records(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "TestBot", 5)
        status, data = client.get_json("/v1/audit/TestBot/records")
        assert status == 200
        assert data["agent_id"] == "TestBot"
        assert data["total"] == 5
        assert data["chain_fields_included"] is True
        assert len(data["records"]) == 5

    def test_limit_param(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "TestBot", 10)
        status, data = client.get_json("/v1/audit/TestBot/records?limit=3")
        assert status == 200
        assert data["total"] == 3

    def test_severity_filter(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "TestBot", 5)
        status, data = client.get_json("/v1/audit/TestBot/records?severity=alert")
        assert status == 200
        # Last record has severity=alert
        assert data["total"] == 1
        assert data["records"][0]["severity"] == "alert"

    def test_since_filter(self, audit_env):
        client, tmp_path, _ca = audit_env
        base_ts = 1000000.0
        _seed_audit_records(tmp_path, "TestBot", 5, start_time=base_ts)
        status, data = client.get_json(f"/v1/audit/TestBot/records?since={base_ts + 2.5}")
        assert status == 200
        # Records at ts+3 and ts+4 should match
        assert data["total"] == 2

    def test_jsonl_format(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "TestBot", 3)
        status, body = client.get("/v1/audit/TestBot/records?format=jsonl")
        assert status == 200
        lines = body.decode("utf-8").strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "record_id" in parsed

    def test_empty_agent_returns_empty(self, audit_env):
        client, _tmp_path, _ca = audit_env
        status, data = client.get_json("/v1/audit/NoSuchAgent/records")
        assert status == 200
        assert data["total"] == 0

    def test_case_insensitive_agent(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "TestBot", 3)
        status, data = client.get_json("/v1/audit/testbot/records")
        assert status == 200
        assert data["total"] == 3


class TestAuditSummaryEndpoint:

    def test_returns_summary(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "SumBot", 5)
        status, data = client.get_json("/v1/audit/SumBot/summary")
        assert status == 200
        assert data["agent_id"] == "SumBot"
        assert data["total"] == 5
        assert "by_verdict" in data
        assert "by_severity" in data

    def test_404_for_missing_agent(self, audit_env):
        client, _tmp_path, _ca = audit_env
        status, data = client.get_json("/v1/audit/Ghost/summary")
        assert status == 404


class TestAuditVerifyEndpoint:

    def test_valid_chain(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "VerifyBot", 10)
        status, data = client.get_json("/v1/audit/VerifyBot/verify")
        assert status == 200
        assert data["valid"] is True
        assert data["record_count"] == 10
        assert "chain_head" in data
        assert "verified_at" in data

    def test_tampered_chain(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "TamperBot", 5)
        audit_file = tmp_path / "audit" / "tamperbot.jsonl"
        lines = audit_file.read_text().strip().split("\n")
        record = json.loads(lines[1])
        record["verdict"] = "TAMPERED"
        lines[1] = json.dumps(record, separators=(",", ":"))
        audit_file.write_text("\n".join(lines) + "\n")

        status, data = client.get_json("/v1/audit/TamperBot/verify")
        assert status == 200
        assert data["valid"] is False
        assert "TAMPERING" in data["message"]

    def test_404_for_missing_agent(self, audit_env):
        client, _tmp_path, _ca = audit_env
        status, data = client.get_json("/v1/audit/NoBot/verify")
        assert status == 404


class TestAuditExportEndpoint:

    def test_export_returns_jsonl(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "ExportBot", 5)
        status, body = client.get("/v1/audit/ExportBot/export")
        assert status == 200
        lines = body.decode("utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_404_for_missing_agent(self, audit_env):
        client, _tmp_path, _ca = audit_env
        status, _body = client.get("/v1/audit/NoBot/export")
        assert status == 404


class TestAuditSealEndpoint:

    def test_returns_seal_for_revoked_agent(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "RevokedBot", 5)

        # Create a revocation file
        rev_dir = tmp_path / "revocations"
        rev_dir.mkdir(parents=True, exist_ok=True)
        rev_data = {
            "agent_name": "RevokedBot",
            "certificate_id": "nmc-test-1234",
            "revoked_at": "2026-02-14T21:00:00Z",
            "reason": "Decommissioned",
            "seal_hash": "sha256:abc123",
            "total_evaluations": 5,
            "final_trust": 0.05,
        }
        (rev_dir / "nmc-test-1234.json").write_text(
            json.dumps(rev_data), encoding="utf-8"
        )

        status, data = client.get_json("/v1/audit/RevokedBot/seal")
        assert status == 200
        assert data["agent_name"] == "RevokedBot"
        assert "chain_valid" in data

    def test_404_for_non_revoked_agent(self, audit_env):
        client, _tmp_path, _ca = audit_env
        status, data = client.get_json("/v1/audit/ActiveBot/seal")
        assert status == 404

    def test_seal_case_insensitive(self, audit_env):
        client, tmp_path, _ca = audit_env
        _seed_audit_records(tmp_path, "CaseBot", 3)
        rev_dir = tmp_path / "revocations"
        rev_dir.mkdir(parents=True, exist_ok=True)
        rev_data = {"agent_name": "CaseBot", "certificate_id": "nmc-case"}
        (rev_dir / "nmc-case.json").write_text(
            json.dumps(rev_data), encoding="utf-8"
        )
        status, data = client.get_json("/v1/audit/casebot/seal")
        assert status == 200


class TestAuditCertIdResolution:

    def test_resolve_cert_id_to_agent(self, audit_env):
        client, tmp_path, ca = audit_env
        # Issue a cert so we can resolve by cert-id
        cert, _sk = ca.issue("CertAgent", "assistant", "org", "global")
        _seed_audit_records(tmp_path, "CertAgent", 3)
        status, data = client.get_json(f"/v1/audit/{cert.certificate_id}/records")
        assert status == 200
        assert data["total"] == 3
