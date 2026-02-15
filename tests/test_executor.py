"""Tests for GovernedToolExecutor — local governance wrapper."""

import json
import time

import pytest

from nomotic.authority import CertificateAuthority
from nomotic.audit_store import LogStore
from nomotic.executor import ExecutionResult, GovernedToolExecutor
from nomotic.keys import SigningKey
from nomotic.sandbox import AgentConfig, save_agent_config
from nomotic.store import FileCertificateStore


def _setup_agent(
    tmp_path,
    agent_id: str = "TestBot",
    actions: list[str] | None = None,
    boundaries: list[str] | None = None,
) -> str:
    """Create a certificate and config for an agent in tmp_path.

    Returns the certificate ID.
    """
    # Create issuer infrastructure
    sk, _vk = SigningKey.generate()
    store = FileCertificateStore(tmp_path)
    ca = CertificateAuthority(issuer_id="test-issuer", signing_key=sk, store=store)

    # Issue certificate
    cert, _agent_sk = ca.issue(
        agent_id=agent_id,
        archetype="general",
        organization="test-org",
        zone_path="global",
        owner="test-owner",
    )

    # Save agent config
    if actions is None:
        actions = ["read", "write", "query"]
    config = AgentConfig(
        agent_id=agent_id,
        actions=actions,
        boundaries=boundaries or [],
    )
    save_agent_config(tmp_path, config)

    return cert.certificate_id


class TestConnect:
    """Test executor creation and agent resolution."""

    def test_connect_by_name(self, tmp_path):
        """Create an agent and connect executor by name."""
        _setup_agent(tmp_path, "TestBot")
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        assert executor.agent_id == "TestBot"

    def test_connect_returns_executor(self, tmp_path):
        """connect() returns a GovernedToolExecutor instance."""
        _setup_agent(tmp_path, "TestBot")
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        assert isinstance(executor, GovernedToolExecutor)

    def test_case_insensitive_connect(self, tmp_path):
        """Connect works regardless of name casing."""
        cert_id = _setup_agent(tmp_path, "TestBot")

        e1 = GovernedToolExecutor.connect("testbot", base_dir=tmp_path)
        e2 = GovernedToolExecutor.connect("TESTBOT", base_dir=tmp_path)
        assert e1.certificate_id == cert_id
        assert e2.certificate_id == cert_id

    def test_connect_without_certificate(self, tmp_path):
        """Connect works even without a certificate (no cert on disk)."""
        executor = GovernedToolExecutor.connect("NoCertBot", base_dir=tmp_path)
        assert executor.agent_id == "NoCertBot"
        assert executor.certificate_id == ""

    def test_connect_test_mode(self, tmp_path):
        """Connect with test_mode=True."""
        _setup_agent(tmp_path, "TestBot")
        executor = GovernedToolExecutor.connect(
            "TestBot", base_dir=tmp_path, test_mode=True
        )
        assert executor.is_test_mode is True

    def test_initial_trust(self, tmp_path):
        """Trust starts at certificate's trust score (0.5 baseline)."""
        _setup_agent(tmp_path, "TestBot")
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        assert executor.trust == pytest.approx(0.5, abs=0.01)


class TestExecute:
    """Test execute() — the core governance + execution method."""

    def test_allowed_action_executes(self, tmp_path):
        """Tool function runs when governance approves."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.execute(
            action="read",
            target="test_db",
            tool_fn=lambda: {"rows": [1, 2, 3]},
        )
        assert result.allowed is True
        assert result.verdict == "ALLOW"
        assert result.data == {"rows": [1, 2, 3]}
        assert result.ucs > 0.0
        assert result.tier in (1, 2, 3)
        assert result.action_id != ""
        assert result.duration_ms > 0.0

    def test_denied_action_skips_execution(self, tmp_path):
        """Tool function does NOT run when governance denies."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        executed = []
        result = executor.execute(
            action="delete",
            target="test_db",
            tool_fn=lambda: executed.append(True),
        )
        assert result.allowed is False
        assert result.verdict == "DENY"
        assert len(executed) == 0  # tool never ran
        assert result.data is None

    def test_execute_without_tool_fn(self, tmp_path):
        """Execute with no tool_fn is evaluation-only."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.execute(action="read", target="test_db")
        assert result.allowed is True
        assert result.data is None

    def test_execution_error_handling(self, tmp_path):
        """Tool exceptions are caught, verdict is still ALLOW."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        def bad_tool():
            raise ValueError("db connection failed")

        result = executor.execute(action="read", target="test_db", tool_fn=bad_tool)
        assert result.allowed is True  # governance approved it
        assert "EXECUTION ERROR" in str(result.data)
        assert "db connection failed" in str(result.data)

    def test_execute_with_params(self, tmp_path):
        """Params are passed through to governance evaluation."""
        _setup_agent(tmp_path, "TestBot", actions=["query"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.execute(
            action="query",
            target="customers",
            params={"sql": "SELECT * FROM customers"},
            tool_fn=lambda: [{"id": 1}],
        )
        assert result.allowed is True
        assert result.data == [{"id": 1}]

    def test_action_count_increments(self, tmp_path):
        """Action count increments with each execute call."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        assert executor.action_count == 0
        executor.execute("read", "test_db", tool_fn=lambda: "ok")
        assert executor.action_count == 1
        executor.execute("read", "test_db", tool_fn=lambda: "ok")
        assert executor.action_count == 2

    def test_dimension_scores_populated(self, tmp_path):
        """Dimension scores are populated in the result."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.execute("read", "test_db", tool_fn=lambda: "ok")
        assert isinstance(result.dimension_scores, dict)
        assert len(result.dimension_scores) > 0
        # All scores should be between 0 and 1
        for score in result.dimension_scores.values():
            assert 0.0 <= score <= 1.0

    def test_denied_populates_vetoed_by(self, tmp_path):
        """Denied actions include veto information."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.execute("delete", "test_db")
        assert result.allowed is False
        assert len(result.vetoed_by) > 0


class TestCheck:
    """Test check() — governance evaluation without execution."""

    def test_check_allowed(self, tmp_path):
        """check() returns ALLOW for in-scope actions."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.check("read", "test_db")
        assert result.allowed is True
        assert result.data is None  # no tool_fn provided

    def test_check_denied(self, tmp_path):
        """check() returns DENY for out-of-scope actions."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.check("delete", "test_db")
        assert result.allowed is False
        assert result.data is None


class TestTrust:
    """Test trust tracking and persistence."""

    def test_trust_increases_on_allow(self, tmp_path):
        """Trust increases after an allowed action."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.execute("read", "test_db", tool_fn=lambda: "ok")
        assert result.trust_after >= result.trust_before

    def test_trust_decreases_on_deny(self, tmp_path):
        """Trust decreases after a denied action."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.execute("delete", "test_db")
        assert result.trust_after <= result.trust_before

    def test_trust_persists_across_calls(self, tmp_path):
        """Trust changes persist between execute() calls."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        r1 = executor.execute("read", "test_db", tool_fn=lambda: "ok")
        r2 = executor.execute("read", "test_db", tool_fn=lambda: "ok")

        assert r2.trust_before == r1.trust_after

    def test_trust_property_reflects_current(self, tmp_path):
        """The trust property reflects the current trust score."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        initial = executor.trust
        executor.execute("read", "test_db", tool_fn=lambda: "ok")
        assert executor.trust >= initial


class TestAuditLog:
    """Test audit/testlog record writing."""

    def test_audit_records_written(self, tmp_path):
        """Each execution writes a persistent audit record."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        executor.execute("read", "test_db", tool_fn=lambda: "ok")
        executor.execute("read", "test_db", tool_fn=lambda: "ok")

        store = LogStore(tmp_path, "audit")
        records = store.query("TestBot")
        assert len(records) == 2

    def test_audit_record_source(self, tmp_path):
        """Audit records have source='executor'."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        executor.execute("read", "test_db", tool_fn=lambda: "ok")

        store = LogStore(tmp_path, "audit")
        records = store.query("TestBot")
        assert records[0].source == "executor"

    def test_hash_chain_integrity(self, tmp_path):
        """Audit chain is valid after multiple executions."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        for _ in range(10):
            executor.execute("read", "test_db", tool_fn=lambda: "ok")

        valid, count, msg = executor.verify_chain()
        assert valid is True
        assert count == 10

    def test_audit_summary(self, tmp_path):
        """get_audit_summary returns correct summary."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        executor.execute("read", "test_db", tool_fn=lambda: "ok")
        executor.execute("read", "test_db", tool_fn=lambda: "ok")
        executor.execute("delete", "test_db")  # denied

        summary = executor.get_audit_summary()
        assert summary["total"] == 3
        assert "ALLOW" in summary["by_verdict"]
        assert "DENY" in summary["by_verdict"]

    def test_denied_records_have_alert_severity(self, tmp_path):
        """Denied actions are recorded with severity='alert'."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        executor.execute("delete", "test_db")

        store = LogStore(tmp_path, "audit")
        records = store.query("TestBot")
        assert records[0].severity == "alert"


class TestTestMode:
    """Test test_mode behavior — separate from production."""

    def test_test_mode_uses_testlog(self, tmp_path):
        """Test mode writes to testlog, not audit."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect(
            "TestBot", base_dir=tmp_path, test_mode=True
        )
        executor.execute("read", "test_db", tool_fn=lambda: "ok")

        audit = LogStore(tmp_path, "audit")
        testlog = LogStore(tmp_path, "testlog")

        assert len(audit.query("TestBot")) == 0
        assert len(testlog.query("TestBot")) == 1

    def test_test_mode_source_field(self, tmp_path):
        """Test mode records have source='executor-test'."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect(
            "TestBot", base_dir=tmp_path, test_mode=True
        )
        executor.execute("read", "test_db", tool_fn=lambda: "ok")

        testlog = LogStore(tmp_path, "testlog")
        records = testlog.query("TestBot")
        assert records[0].source == "executor-test"

    def test_test_mode_simulated_trust(self, tmp_path):
        """Test mode doesn't change production trust."""
        cert_id = _setup_agent(tmp_path, "TestBot", actions=["read"])
        store = FileCertificateStore(tmp_path)
        original_trust = store.get(cert_id).trust_score

        executor = GovernedToolExecutor.connect(
            "TestBot", base_dir=tmp_path, test_mode=True
        )
        # Run some denials to drop simulated trust
        for _ in range(5):
            executor.execute("delete", "test_db", tool_fn=lambda: "bad")

        # Production cert trust should be unchanged
        cert = store.get(cert_id)
        assert cert.trust_score == original_trust

    def test_test_mode_hash_chain(self, tmp_path):
        """Test mode hash chain is valid."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect(
            "TestBot", base_dir=tmp_path, test_mode=True
        )
        for _ in range(5):
            executor.execute("read", "test_db", tool_fn=lambda: "ok")

        valid, count, msg = executor.verify_chain()
        assert valid is True
        assert count == 5


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_to_dict(self, tmp_path):
        """to_dict returns all governance fields (excludes data)."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.execute("read", "test_db", tool_fn=lambda: "secret")
        d = result.to_dict()

        assert "allowed" in d
        assert "verdict" in d
        assert "ucs" in d
        assert "tier" in d
        assert "trust_before" in d
        assert "trust_after" in d
        assert "trust_delta" in d
        assert "dimension_scores" in d
        assert "vetoed_by" in d
        assert "action_id" in d
        assert "duration_ms" in d
        assert "reason" in d
        # data is NOT in to_dict (sensitive tool output)
        assert "data" not in d

    def test_to_dict_serializable(self, tmp_path):
        """to_dict output is JSON-serializable."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = executor.execute("read", "test_db", tool_fn=lambda: "ok")
        d = result.to_dict()
        # Should not raise
        json.dumps(d)


class TestTimeout:
    """Test execution timeout."""

    def test_timeout_raises(self, tmp_path):
        """Timeout raises when tool takes too long."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        def slow_tool():
            time.sleep(5)
            return "done"

        result = executor.execute(
            action="read",
            target="test_db",
            tool_fn=slow_tool,
            timeout=0.1,
        )
        # The timeout exception is caught and wrapped
        assert result.allowed is True
        assert "EXECUTION ERROR" in str(result.data)


class TestImports:
    """Test that the executor is importable from the package."""

    def test_import_from_nomotic(self):
        """GovernedToolExecutor is importable from nomotic."""
        from nomotic import GovernedToolExecutor as GTE
        from nomotic import ExecutionResult as ER

        assert GTE is GovernedToolExecutor
        assert ER is ExecutionResult
