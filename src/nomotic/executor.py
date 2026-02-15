"""GovernedToolExecutor — governance wrapper for local tool execution.

The integration point between any AI agent and Nomotic governance.
Wraps local function calls in pre-execution governance evaluation.

Usage:
    from nomotic import GovernedToolExecutor

    executor = GovernedToolExecutor.connect("claims-bot")

    result = executor.execute(
        action="query_db",
        target="customers",
        tool_fn=lambda: db.execute("SELECT * FROM customers"),
    )

    if result.allowed:
        print(result.data)    # tool output
    else:
        print(result.reason)  # denial explanation
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nomotic.audit_store import LogStore, PersistentLogRecord
from nomotic.certificate import AgentCertificate
from nomotic.runtime import GovernanceRuntime
from nomotic.sandbox import (
    AgentConfig,
    build_sandbox_runtime,
    find_agent_cert_id,
    load_agent_config,
)
from nomotic.store import FileCertificateStore
from nomotic.types import Action, AgentContext, TrustProfile

__all__ = [
    "ExecutionResult",
    "GovernedToolExecutor",
]


@dataclass
class ExecutionResult:
    """Result of a governed tool execution."""

    allowed: bool
    verdict: str  # "ALLOW", "DENY", "ESCALATE"
    reason: str  # human-readable explanation
    data: Any  # tool return value (None if denied)
    ucs: float  # unified compliance score
    tier: int  # evaluation tier (1=veto, 2=threshold, 3=full)
    trust_before: float  # trust score before this action
    trust_after: float  # trust score after this action
    trust_delta: float  # change in trust
    dimension_scores: dict[str, float]  # per-dimension scores
    vetoed_by: list[str]  # dimensions that vetoed (empty if allowed)
    action_id: str  # unique ID for this evaluation
    duration_ms: float  # evaluation time in milliseconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "verdict": self.verdict,
            "reason": self.reason,
            "ucs": self.ucs,
            "tier": self.tier,
            "trust_before": self.trust_before,
            "trust_after": self.trust_after,
            "trust_delta": self.trust_delta,
            "dimension_scores": self.dimension_scores,
            "vetoed_by": self.vetoed_by,
            "action_id": self.action_id,
            "duration_ms": self.duration_ms,
        }


class GovernedToolExecutor:
    """Governance wrapper for local tool execution.

    Connects to an agent's Nomotic identity (certificate, scope, trust)
    and evaluates every tool call through the 13-dimension governance pipeline
    before execution.
    """

    def __init__(
        self,
        agent_id: str,
        base_dir: Path | None = None,
        test_mode: bool = False,
    ):
        """
        Args:
            agent_id: Agent name, numeric ID, or certificate ID
            base_dir: Nomotic config directory (default ~/.nomotic)
            test_mode: If True, writes to testlog and uses simulated trust
        """
        self._base_dir = base_dir or Path.home() / ".nomotic"
        self._test_mode = test_mode
        self._agent_id = agent_id

        # Resolve agent identity
        self._cert = self._load_certificate(agent_id)
        self._config = self._load_config(agent_id)

        # Build governance runtime
        self._runtime = self._build_runtime()

        # Connect to persistent log store
        log_type = "testlog" if test_mode else "audit"
        self._log_store = LogStore(self._base_dir, log_type)

        # Track simulated trust separately in test mode
        if test_mode:
            self._simulated_trust = self._load_simulated_trust()

        # Action counter for behavioral fingerprinting
        self._action_count = 0

    @classmethod
    def connect(cls, agent_id: str, **kwargs: Any) -> GovernedToolExecutor:
        """Connect to an agent's governance identity.

        Convenience constructor — the primary way to create an executor.

        Usage:
            executor = GovernedToolExecutor.connect("claims-bot")
            executor = GovernedToolExecutor.connect("claims-bot", test_mode=True)
        """
        return cls(agent_id, **kwargs)

    def execute(
        self,
        action: str,
        target: str = "",
        params: dict[str, Any] | None = None,
        tool_fn: Callable[..., Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Evaluate governance and conditionally execute a tool.

        1. Evaluates the action through the 13-dimension governance pipeline
        2. If ALLOW: executes tool_fn and returns the result
        3. If DENY: skips execution and returns denial reason
        4. Updates trust score (production or simulated)
        5. Writes to persistent log (audit or testlog)

        Args:
            action: The action type (e.g., "read_file", "query_db", "send_email")
            target: The target resource (e.g., "payroll.csv", "customers")
            params: Optional parameters (e.g., {"sql": "SELECT * ..."})
            tool_fn: Callable to execute if governance approves. If None, evaluation only.
            timeout: Optional execution timeout in seconds.

        Returns:
            ExecutionResult with verdict, trust changes, and tool output (if allowed)
        """
        params = params or {}
        start_time = time.monotonic()

        # Build the action for evaluation
        action_obj = Action(
            agent_id=self._agent_id,
            action_type=action,
            target=target,
            parameters=params,
        )

        # Get current trust (simulated in test mode, real in production)
        trust_before = self._get_current_trust()

        # Build context with current trust
        trust_profile = self._runtime.get_trust_profile(self._agent_id)
        trust_profile.overall_trust = trust_before
        ctx = AgentContext(agent_id=self._agent_id, trust_profile=trust_profile)

        # Run governance evaluation
        verdict = self._runtime.evaluate(action_obj, ctx)

        # Get the trust after evaluation from the runtime
        trust_after = self._runtime.get_trust_profile(self._agent_id).overall_trust
        trust_delta = trust_after - trust_before
        trust_after = round(max(0.0, min(1.0, trust_after)), 3)

        # Execute the tool if allowed and tool_fn provided
        tool_output = None
        if verdict.verdict.name == "ALLOW" and tool_fn is not None:
            try:
                if timeout:
                    tool_output = self._execute_with_timeout(tool_fn, timeout)
                else:
                    tool_output = tool_fn()
            except Exception as e:
                tool_output = f"EXECUTION ERROR: {e}"

        duration_ms = (time.monotonic() - start_time) * 1000

        # Update trust
        self._update_trust(trust_after)

        # Write to persistent log
        self._write_log_record(
            action=action,
            target=target,
            params=params,
            verdict=verdict,
            trust_before=trust_before,
            trust_after=trust_after,
            trust_delta=trust_delta,
        )

        # Update action count for fingerprinting
        self._action_count += 1

        # Build dimension scores dict
        dim_scores: dict[str, float] = {}
        vetoed_by: list[str] = []
        if verdict.dimension_scores:
            for ds in verdict.dimension_scores:
                dim_scores[ds.dimension_name] = ds.score
        if verdict.vetoed_by:
            vetoed_by = list(verdict.vetoed_by)

        return ExecutionResult(
            allowed=verdict.verdict.name == "ALLOW",
            verdict=verdict.verdict.name,
            reason=verdict.reasoning or "",
            data=tool_output,
            ucs=verdict.ucs,
            tier=verdict.tier,
            trust_before=trust_before,
            trust_after=trust_after,
            trust_delta=trust_delta,
            dimension_scores=dim_scores,
            vetoed_by=vetoed_by,
            action_id=verdict.action_id,
            duration_ms=round(duration_ms, 2),
        )

    def check(
        self, action: str, target: str = "", params: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """Evaluate governance WITHOUT executing anything.

        Useful for pre-flight checks: "would this action be allowed?"
        Still updates trust and writes to log.
        """
        return self.execute(action=action, target=target, params=params, tool_fn=None)

    @property
    def trust(self) -> float:
        """Current trust score (simulated in test mode, production otherwise)."""
        return self._get_current_trust()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def certificate_id(self) -> str:
        return self._cert.certificate_id if self._cert else ""

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def is_test_mode(self) -> bool:
        return self._test_mode

    def get_audit_summary(self) -> dict[str, Any]:
        """Get summary of this agent's log (audit or testlog)."""
        return self._log_store.summary(self._agent_id)

    def verify_chain(self) -> tuple[bool, int, str]:
        """Verify the integrity of this agent's log chain."""
        return self._log_store.verify_chain(self._agent_id)

    # ── Internal methods ──────────────────────────────────────────────

    def _load_certificate(self, agent_id: str) -> AgentCertificate | None:
        """Find and load the agent's certificate from disk.

        Uses the same case-insensitive resolution as sandbox.py's
        find_agent_cert_id().
        """
        cert_id = find_agent_cert_id(self._base_dir, agent_id)
        if cert_id is None:
            return None
        store = FileCertificateStore(self._base_dir)
        return store.get(cert_id)

    def _load_config(self, agent_id: str) -> AgentConfig:
        """Load agent governance config, or create a default."""
        config = load_agent_config(self._base_dir, agent_id)
        if config is None:
            config = AgentConfig(agent_id=agent_id)
        return config

    def _build_runtime(self) -> GovernanceRuntime:
        """Build governance runtime from agent config using shared builder."""
        return build_sandbox_runtime(
            agent_config=self._config,
            agent_id=self._agent_id,
            base_dir=self._base_dir,
        )

    def _get_current_trust(self) -> float:
        """Return current trust score.

        In test mode: return simulated trust.
        In production: return certificate trust or runtime trust profile.
        """
        if self._test_mode:
            return self._simulated_trust
        if self._cert is not None:
            return self._cert.trust_score
        return self._runtime.get_trust_profile(self._agent_id).overall_trust

    def _update_trust(self, new_trust: float) -> None:
        """Persist the updated trust score.

        In test mode: update simulated trust only.
        In production: update the certificate on disk.
        """
        if self._test_mode:
            self._simulated_trust = new_trust
            self._save_simulated_trust(new_trust)
        elif self._cert is not None:
            self._cert.trust_score = new_trust
            store = FileCertificateStore(self._base_dir)
            store.save(self._cert)

    def _load_simulated_trust(self) -> float:
        """Load simulated trust from testlog directory, or fall back to cert/default."""
        production_trust = self._cert.trust_score if self._cert else 0.5
        trust_file = (
            self._base_dir / "testlog" / f"{self._agent_id.lower()}.trust.json"
        )
        if trust_file.exists():
            try:
                data = json.loads(trust_file.read_text(encoding="utf-8"))
                age = time.time() - data.get("last_updated", 0)
                if age < 3600:  # 1 hour
                    return data.get("simulated_trust", production_trust)
            except (json.JSONDecodeError, KeyError):
                pass
        return production_trust

    def _save_simulated_trust(self, simulated: float) -> None:
        """Persist simulated trust to testlog directory."""
        production_trust = self._cert.trust_score if self._cert else 0.5
        testlog_dir = self._base_dir / "testlog"
        testlog_dir.mkdir(parents=True, exist_ok=True)
        trust_file = testlog_dir / f"{self._agent_id.lower()}.trust.json"
        trust_file.write_text(
            json.dumps(
                {
                    "simulated_trust": simulated,
                    "production_trust_at_start": production_trust,
                    "last_updated": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_log_record(
        self,
        action: str,
        target: str,
        params: dict[str, Any],
        verdict: Any,
        trust_before: float,
        trust_after: float,
        trust_delta: float,
    ) -> None:
        """Create a PersistentLogRecord with hash chain linking and append."""
        # Determine trend
        if trust_delta > 0.001:
            trend = "rising"
        elif trust_delta < -0.001:
            trend = "falling"
        else:
            trend = "stable"

        # Determine severity
        if verdict.verdict.name == "DENY":
            severity = "alert"
        elif verdict.verdict.name == "ESCALATE":
            severity = "warning"
        else:
            severity = "info"

        # Build dimension scores dict for the record
        dim_scores: dict[str, float] = {}
        if verdict.dimension_scores:
            for ds in verdict.dimension_scores:
                dim_scores[ds.dimension_name] = ds.score

        vetoed_by = list(verdict.vetoed_by) if verdict.vetoed_by else []

        source = "executor-test" if self._test_mode else "executor"

        record_data = {
            "record_id": uuid.uuid4().hex[:12],
            "timestamp": time.time(),
            "agent_id": self._agent_id,
            "action_type": action,
            "action_target": target,
            "verdict": verdict.verdict.name,
            "ucs": verdict.ucs,
            "tier": verdict.tier,
            "trust_score": trust_after,
            "trust_delta": trust_delta,
            "trust_trend": trend,
            "severity": severity,
            "justification": verdict.reasoning or "",
            "vetoed_by": vetoed_by,
            "dimension_scores": dim_scores,
            "parameters": params,
            "source": source,
            "previous_hash": "",
            "record_hash": "",
        }

        # Hash chain linking
        previous_hash = self._log_store.get_last_hash(self._agent_id)
        record_data["previous_hash"] = previous_hash
        record_data["record_hash"] = self._log_store.compute_hash(
            record_data, previous_hash
        )

        record = PersistentLogRecord(**record_data)
        self._log_store.append(record)

    def _execute_with_timeout(self, fn: Callable[..., Any], timeout: float) -> Any:
        """Run a callable with a timeout using threading."""
        result: list[Any] = []
        exception: list[BaseException] = []

        def _target() -> None:
            try:
                result.append(fn())
            except BaseException as e:
                exception.append(e)

        thread = threading.Thread(target=_target)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Tool execution exceeded {timeout}s timeout"
            )
        if exception:
            raise exception[0]
        return result[0] if result else None
