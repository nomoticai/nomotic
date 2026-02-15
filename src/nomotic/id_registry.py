"""Sequential agent ID registry.

Agent IDs are unique integers starting at 1000. They are never reused.
The registry is backed by a JSON file at ~/.nomotic/id_registry.json.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["AgentIdRegistry"]

_START_ID = 1000

# Pattern for name+ID combo like "claims-bot1000"
_COMBO_RE = re.compile(r"^(.+?)(\d{4,})$")


class AgentIdRegistry:
    """Manages sequential agent IDs backed by a JSON file.

    The registry file uses atomic write with rename to prevent corruption
    from concurrent CLI invocations.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)
        self._path = self._base / "id_registry.json"
        self._data: dict[str, Any] = self._load()

    # ── Public API ────────────────────────────────────────────────────

    def next_id(self) -> int:
        """Allocate and return the next sequential agent ID."""
        nid = self._data["next_id"]
        self._data["next_id"] = nid + 1
        self._save()
        return nid

    def register(
        self,
        agent_id: int,
        name: str,
        certificate_id: str,
        organization: str,
    ) -> None:
        """Register a new agent. Called during birth."""
        now = datetime.now(timezone.utc).isoformat()
        self._data["agents"][str(agent_id)] = {
            "name": name,
            "certificate_id": certificate_id,
            "organization": organization,
            "created_at": now,
            "status": "ACTIVE",
            "revoked_at": None,
        }
        self._save()

    def update_status(
        self,
        agent_id: int,
        status: str,
        revoked_at: str | None = None,
    ) -> None:
        """Update agent status (e.g., on revocation)."""
        key = str(agent_id)
        if key not in self._data["agents"]:
            raise KeyError(f"Agent ID {agent_id} not found in registry")
        self._data["agents"][key]["status"] = status
        if revoked_at is not None:
            self._data["agents"][key]["revoked_at"] = revoked_at
        self._save()

    def resolve(self, identifier: str) -> list[dict[str, Any]]:
        """Resolve an identifier to agent records.

        Accepts:
        - Integer string ("1000") -- exact ID match
        - Name string ("claims-bot") -- all matching names
        - Combo string ("claims-bot1000") -- name+ID combo
        - Cert-id string ("nmc-...") -- exact cert match

        Returns list of matching agent records (may be multiple for name lookups).
        Each returned dict includes an ``"agent_id"`` key with the numeric ID.
        """
        agents = self._data["agents"]

        # Cert-id match
        if identifier.startswith("nmc-"):
            for aid, entry in agents.items():
                if entry["certificate_id"] == identifier:
                    return [{"agent_id": int(aid), **entry}]
            return []

        # Exact numeric ID match
        if identifier.isdigit():
            entry = agents.get(identifier)
            if entry is not None:
                return [{"agent_id": int(identifier), **entry}]
            return []

        # Combo match: name + numeric id
        m = _COMBO_RE.match(identifier)
        if m:
            name_part, id_part = m.group(1), m.group(2)
            entry = agents.get(id_part)
            if entry is not None and entry["name"] == name_part:
                return [{"agent_id": int(id_part), **entry}]
            # Fall through to name search if combo didn't match

        # Name match (may return multiple)
        results: list[dict[str, Any]] = []
        for aid, entry in agents.items():
            if entry["name"] == identifier:
                results.append({"agent_id": int(aid), **entry})
        return results

    def resolve_single(self, identifier: str) -> dict[str, Any]:
        """Like resolve() but raises ValueError if not exactly one match."""
        matches = self.resolve(identifier)
        if len(matches) == 0:
            raise ValueError(f"No agent found for identifier: {identifier}")
        if len(matches) > 1:
            ids = ", ".join(str(m["agent_id"]) for m in matches)
            name = matches[0]["name"]
            raise ValueError(
                f"Ambiguous identifier '{identifier}' matches agents: {ids}\n"
                f"  Use numeric ID or '{name}<ID>' to disambiguate."
            )
        return matches[0]

    def list_all(
        self,
        status: str | None = None,
        org: str | None = None,
    ) -> list[dict[str, Any]]:
        """List agents with optional filters."""
        results: list[dict[str, Any]] = []
        for aid, entry in self._data["agents"].items():
            if status is not None and entry["status"] != status:
                continue
            if org is not None and entry["organization"] != org:
                continue
            results.append({"agent_id": int(aid), **entry})
        return results

    def get(self, agent_id: int) -> dict[str, Any] | None:
        """Get a single agent entry by numeric ID, or None."""
        entry = self._data["agents"].get(str(agent_id))
        if entry is None:
            return None
        return {"agent_id": agent_id, **entry}

    # ── Migration ─────────────────────────────────────────────────────

    def migrate_from_certs(self, certs_dir: Path) -> int:
        """Scan existing certificate files and assign sequential IDs.

        Returns the number of agents migrated.
        """
        if not certs_dir.exists():
            return 0

        # Collect existing certs ordered by issued_at
        existing: list[dict[str, Any]] = []
        for cert_file in certs_dir.glob("nmc-*.json"):
            try:
                data = json.loads(cert_file.read_text(encoding="utf-8"))
                existing.append(data)
            except (json.JSONDecodeError, KeyError):
                continue

        # Also check revoked dir
        revoked_dir = certs_dir.parent / "revoked"
        if revoked_dir.exists():
            for cert_file in revoked_dir.glob("nmc-*.json"):
                try:
                    data = json.loads(cert_file.read_text(encoding="utf-8"))
                    existing.append(data)
                except (json.JSONDecodeError, KeyError):
                    continue

        # Sort by issued_at
        existing.sort(key=lambda d: d.get("issued_at", ""))

        # Skip certs already in registry
        known_cert_ids = {
            e["certificate_id"] for e in self._data["agents"].values()
        }

        count = 0
        for cert_data in existing:
            cert_id = cert_data.get("certificate_id", "")
            if cert_id in known_cert_ids:
                continue

            agent_name = cert_data.get("agent_id", "unknown")
            org = cert_data.get("organization", "")
            status = cert_data.get("status", "ACTIVE")
            created_at = cert_data.get("issued_at", "")

            nid = self.next_id()
            self._data["agents"][str(nid)] = {
                "name": agent_name,
                "certificate_id": cert_id,
                "organization": org,
                "created_at": created_at,
                "status": status,
                "revoked_at": None,
            }
            count += 1

        if count > 0:
            self._save()
        return count

    # ── Internal ──────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        """Load registry from disk, creating it if needed."""
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            # Ensure required keys exist
            data.setdefault("next_id", _START_ID)
            data.setdefault("agents", {})
            return data

        # First run: create empty registry and attempt migration
        self._base.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"next_id": _START_ID, "agents": {}}
        return data

    def _save(self) -> None:
        """Atomically write registry to disk."""
        self._base.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self._data, indent=2, sort_keys=False)
        # Atomic write: write to temp file, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._base), suffix=".tmp", prefix="id_registry_"
        )
        try:
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            os.replace(tmp_path, str(self._path))
        except Exception:
            os.close(fd) if not os.get_inheritable(fd) else None
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
