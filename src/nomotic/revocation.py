"""Revocation sealing — immutable final record for revoked agents.

When an agent is revoked, a seal is computed over its complete audit
history. The seal is stored alongside summary statistics so that the
full lifecycle of the agent can be reconstructed at any time.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nomotic.certificate import AgentCertificate

__all__ = ["create_revocation_seal", "load_revocation_seal"]


def create_revocation_seal(
    base_dir: Path,
    *,
    agent_id: int,
    agent_name: str,
    certificate_id: str,
    reason: str,
    cert: AgentCertificate | None = None,
    audit_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a revocation seal for an agent.

    Computes a SHA-256 seal hash over the complete ordered audit history
    and writes a revocation record to ~/.nomotic/revocations/<agent-id>.json.
    """
    now = datetime.now(timezone.utc)

    # Compute audit seal hash
    audit_seal = ""
    total_evaluations = 0
    total_violations = 0
    chain_records = 0

    if audit_records:
        # Compute seal over all records
        canonical = json.dumps(audit_records, sort_keys=True, separators=(",", ":"))
        audit_seal = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
        total_evaluations = len(audit_records)
        total_violations = sum(
            1 for r in audit_records if r.get("verdict") in ("DENY", "ESCALATE")
        )
        chain_records = total_evaluations

    # Compute trust stats from certificate
    final_trust = cert.trust_score if cert else 0.05
    peak_trust = max(0.50, final_trust)  # At minimum, trust started at 0.50

    # Estimate evaluation counts from behavioral age if no audit records
    if cert and total_evaluations == 0:
        total_evaluations = cert.behavioral_age
        if cert.trust_score < 0.50 and total_evaluations > 0:
            delta = 0.50 - cert.trust_score
            est_violations = max(0, int((total_evaluations * 0.002 - delta) / 0.012))
            total_violations = est_violations

    seal = {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "certificate_id": certificate_id,
        "revoked_at": now.isoformat(),
        "revoked_by": "cli-user",
        "reason": reason,
        "audit_seal": audit_seal,
        "total_evaluations": total_evaluations,
        "final_trust": round(final_trust, 3),
        "peak_trust": round(peak_trust, 3),
        "total_violations": total_violations,
        "chain_records": chain_records,
    }

    # Write to revocations directory
    revocations_dir = Path(base_dir) / "revocations"
    revocations_dir.mkdir(parents=True, exist_ok=True)
    seal_path = revocations_dir / f"{agent_id}.json"
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return seal


def load_revocation_seal(
    base_dir: Path,
    agent_id: int,
) -> dict[str, Any] | None:
    """Load a revocation seal for an agent, or None if not found."""
    seal_path = Path(base_dir) / "revocations" / f"{agent_id}.json"
    if not seal_path.exists():
        return None
    return json.loads(seal_path.read_text(encoding="utf-8"))
