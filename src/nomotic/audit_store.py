"""Persistent append-only log storage with hash chaining.

Used for both production audit trails and test logs.
Each agent gets a JSONL file. Each record includes a SHA-256 hash
of the previous record, forming a tamper-evident chain.

Production audit: base_dir / "audit" / "<agent>.jsonl"
Test log:         base_dir / "testlog" / "<agent>.jsonl"
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AuditStore",
    "LogStore",
    "PersistentAuditRecord",
    "PersistentLogRecord",
]


@dataclass
class PersistentLogRecord:
    """A single log record with hash chain fields."""

    record_id: str
    timestamp: float
    agent_id: str
    action_type: str
    action_target: str
    verdict: str  # "ALLOW", "DENY", "ESCALATE"
    ucs: float
    tier: int
    trust_score: float  # trust after this action
    trust_delta: float
    trust_trend: str
    severity: str  # "info", "alert", "warning"
    justification: str
    vetoed_by: list[str] = field(default_factory=list)
    dimension_scores: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str = ""  # "gateway", "sdk", "cli-test"
    previous_hash: str = ""
    record_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PersistentLogRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Backward-compatible alias
PersistentAuditRecord = PersistentLogRecord


class LogStore:
    """Append-only log storage with hash chaining.

    Usage:
        audit_store = LogStore(base_dir, "audit")      # production
        test_store  = LogStore(base_dir, "testlog")     # test
    """

    def __init__(self, base_dir: Path, log_type: str = "audit") -> None:
        """
        Args:
            base_dir: Root directory (e.g. ~/.nomotic)
            log_type: Subdirectory name — "audit" or "testlog"
        """
        self._dir = base_dir / log_type
        self._dir.mkdir(parents=True, exist_ok=True)
        self._log_type = log_type

    def _agent_file(self, agent_id: str) -> Path:
        """Get the log file path for an agent (lowercase normalized)."""
        safe_name = agent_id.lower().replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe_name}.jsonl"

    def get_last_hash(self, agent_id: str) -> str:
        """Get the hash of the most recent record for chain linking."""
        path = self._agent_file(agent_id)
        if not path.exists():
            return ""
        # Read last non-empty line
        last_line = ""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if not last_line:
            return ""
        try:
            record = json.loads(last_line)
            return record.get("record_hash", "")
        except json.JSONDecodeError:
            return ""

    def compute_hash(self, record_dict: dict[str, Any], previous_hash: str) -> str:
        """Compute SHA-256 hash for a record."""
        # Create canonical representation (without record_hash field)
        hashable = dict(record_dict)
        hashable.pop("record_hash", None)
        hashable["previous_hash"] = previous_hash
        canonical = json.dumps(hashable, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(self, record: PersistentLogRecord) -> None:
        """Append a record to the agent's log file."""
        path = self._agent_file(record.agent_id)
        line = json.dumps(record.to_dict(), separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def query(
        self,
        agent_id: str,
        limit: int = 20,
        severity: str | None = None,
    ) -> list[PersistentLogRecord]:
        """Read records for an agent, most recent first."""
        path = self._agent_file(agent_id)
        if not path.exists():
            return []
        records: list[PersistentLogRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if severity and data.get("severity") != severity:
                        continue
                    records.append(PersistentLogRecord.from_dict(data))
                except (json.JSONDecodeError, TypeError):
                    continue
        # Return most recent first, limited
        return list(reversed(records[-limit:]))

    def query_all(self, agent_id: str) -> list[PersistentLogRecord]:
        """Read ALL records for an agent in chronological order."""
        path = self._agent_file(agent_id)
        if not path.exists():
            return []
        records: list[PersistentLogRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(PersistentLogRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return records

    def verify_chain(self, agent_id: str) -> tuple[bool, int, str]:
        """Verify the hash chain integrity.

        Returns (is_valid, record_count, error_message).
        """
        records = self.query_all(agent_id)
        if not records:
            return True, 0, "No records found"

        previous_hash = ""
        for i, record in enumerate(records):
            # Recompute the hash
            expected = self.compute_hash(record.to_dict(), previous_hash)
            if record.record_hash != expected:
                return False, len(records), (
                    f"TAMPERING DETECTED at record #{i + 1} ({record.record_id}). "
                    f"Expected hash: {expected[:24]}... "
                    f"Actual hash: {record.record_hash[:24]}..."
                )
            if record.previous_hash != previous_hash:
                return False, len(records), (
                    f"CHAIN BREAK at record #{i + 1} ({record.record_id}). "
                    f"Previous hash mismatch."
                )
            previous_hash = record.record_hash

        return True, len(records), "All records verified"

    def summary(self, agent_id: str) -> dict[str, Any]:
        """Generate summary for an agent."""
        records = self.query_all(agent_id)
        if not records:
            return {"total": 0}

        by_verdict: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for r in records:
            by_verdict[r.verdict] = by_verdict.get(r.verdict, 0) + 1
            by_severity[r.severity] = by_severity.get(r.severity, 0) + 1

        return {
            "total": len(records),
            "by_verdict": by_verdict,
            "by_severity": by_severity,
            "first_timestamp": records[0].timestamp,
            "last_timestamp": records[-1].timestamp,
            "trust_start": records[0].trust_score - records[0].trust_delta,
            "trust_end": records[-1].trust_score,
        }

    def seal(self, agent_id: str) -> str:
        """Compute a seal hash over the entire log history.

        Used during revocation to create an immutable snapshot.
        Returns the seal hash.
        """
        records = self.query_all(agent_id)
        if not records:
            return "sha256:empty"
        # The seal is the hash of the last record's hash (which chains everything)
        last_hash = records[-1].record_hash
        seal_input = f"SEAL:{agent_id}:{len(records)}:{last_hash}"
        return "sha256:" + hashlib.sha256(seal_input.encode("utf-8")).hexdigest()

    def list_agents(self) -> list[str]:
        """List all agent IDs that have log files."""
        agents: list[str] = []
        for path in self._dir.glob("*.jsonl"):
            agents.append(path.stem)
        return agents


class AuditStore(LogStore):
    """Backward-compatible alias for LogStore with log_type='audit'."""

    def __init__(self, base_dir: Path) -> None:
        super().__init__(base_dir, "audit")
