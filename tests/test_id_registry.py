"""Tests for AgentIdRegistry — sequential ID allocation and resolution."""

import json
import tempfile
from pathlib import Path

import pytest

from nomotic.id_registry import AgentIdRegistry


class TestAgentIdRegistry:
    """Basic registry operations."""

    def test_next_id_starts_at_1000(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = AgentIdRegistry(Path(tmp))
            assert reg.next_id() == 1000

    def test_next_id_increments(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = AgentIdRegistry(Path(tmp))
            id1 = reg.next_id()
            id2 = reg.next_id()
            id3 = reg.next_id()
            assert id1 == 1000
            assert id2 == 1001
            assert id3 == 1002

    def test_register_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = AgentIdRegistry(Path(tmp))
            nid = reg.next_id()
            reg.register(nid, "claims-bot", "nmc-abc123", "acme-insurance")
            entry = reg.get(nid)
            assert entry is not None
            assert entry["name"] == "claims-bot"
            assert entry["certificate_id"] == "nmc-abc123"
            assert entry["organization"] == "acme-insurance"
            assert entry["status"] == "ACTIVE"
            assert entry["revoked_at"] is None

    def test_update_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = AgentIdRegistry(Path(tmp))
            nid = reg.next_id()
            reg.register(nid, "agent-1", "nmc-111", "acme")
            reg.update_status(nid, "REVOKED", revoked_at="2026-02-14T21:00:00Z")
            entry = reg.get(nid)
            assert entry["status"] == "REVOKED"
            assert entry["revoked_at"] == "2026-02-14T21:00:00Z"

    def test_update_status_unknown_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = AgentIdRegistry(Path(tmp))
            with pytest.raises(KeyError, match="Agent ID 9999"):
                reg.update_status(9999, "REVOKED")

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            reg1 = AgentIdRegistry(base)
            nid = reg1.next_id()
            reg1.register(nid, "persistent-bot", "nmc-persist", "org-1")

            # Create new instance — should load from file
            reg2 = AgentIdRegistry(base)
            assert reg2.next_id() == 1001
            entry = reg2.get(1000)
            assert entry is not None
            assert entry["name"] == "persistent-bot"


class TestAgentIdResolution:
    """Resolution by ID, name, combo, and cert-id."""

    def _setup_registry(self, base: Path) -> AgentIdRegistry:
        reg = AgentIdRegistry(base)
        reg.register(reg.next_id(), "claims-bot", "nmc-aaa", "acme")
        reg.register(reg.next_id(), "data-agent", "nmc-bbb", "acme")
        reg.register(reg.next_id(), "claims-bot", "nmc-ccc", "other-co")
        return reg

    def test_resolve_by_numeric_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._setup_registry(Path(tmp))
            results = reg.resolve("1000")
            assert len(results) == 1
            assert results[0]["name"] == "claims-bot"
            assert results[0]["agent_id"] == 1000

    def test_resolve_by_cert_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._setup_registry(Path(tmp))
            results = reg.resolve("nmc-bbb")
            assert len(results) == 1
            assert results[0]["name"] == "data-agent"

    def test_resolve_by_name_multiple(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._setup_registry(Path(tmp))
            results = reg.resolve("claims-bot")
            assert len(results) == 2  # Two agents with same name in different orgs

    def test_resolve_by_combo(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._setup_registry(Path(tmp))
            results = reg.resolve("claims-bot1000")
            assert len(results) == 1
            assert results[0]["agent_id"] == 1000
            assert results[0]["organization"] == "acme"

    def test_resolve_by_combo_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._setup_registry(Path(tmp))
            results = reg.resolve("claims-bot1002")
            assert len(results) == 1
            assert results[0]["organization"] == "other-co"

    def test_resolve_no_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._setup_registry(Path(tmp))
            results = reg.resolve("nonexistent")
            assert results == []

    def test_resolve_single_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._setup_registry(Path(tmp))
            result = reg.resolve_single("1001")
            assert result["name"] == "data-agent"

    def test_resolve_single_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._setup_registry(Path(tmp))
            with pytest.raises(ValueError, match="Ambiguous"):
                reg.resolve_single("claims-bot")

    def test_resolve_single_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._setup_registry(Path(tmp))
            with pytest.raises(ValueError, match="No agent found"):
                reg.resolve_single("ghost")


class TestAgentIdListAll:
    """List and filter agents."""

    def test_list_all_no_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = AgentIdRegistry(Path(tmp))
            reg.register(reg.next_id(), "a", "nmc-1", "org1")
            reg.register(reg.next_id(), "b", "nmc-2", "org2")
            results = reg.list_all()
            assert len(results) == 2

    def test_list_all_filter_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = AgentIdRegistry(Path(tmp))
            nid1 = reg.next_id()
            reg.register(nid1, "a", "nmc-1", "org1")
            nid2 = reg.next_id()
            reg.register(nid2, "b", "nmc-2", "org1")
            reg.update_status(nid2, "REVOKED", revoked_at="2026-01-01T00:00:00Z")
            active = reg.list_all(status="ACTIVE")
            assert len(active) == 1
            assert active[0]["name"] == "a"

    def test_list_all_filter_org(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = AgentIdRegistry(Path(tmp))
            reg.register(reg.next_id(), "a", "nmc-1", "org1")
            reg.register(reg.next_id(), "b", "nmc-2", "org2")
            results = reg.list_all(org="org2")
            assert len(results) == 1
            assert results[0]["name"] == "b"


class TestMigrationFromCerts:
    """Migration from existing certificate files."""

    def test_migrate_from_certs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            certs_dir = base / "certs"
            certs_dir.mkdir(parents=True)

            # Create two old-format cert files
            cert1 = {
                "certificate_id": "nmc-old-1",
                "agent_id": "legacy-bot",
                "organization": "old-corp",
                "issued_at": "2026-01-01T00:00:00+00:00",
                "status": "ACTIVE",
            }
            cert2 = {
                "certificate_id": "nmc-old-2",
                "agent_id": "legacy-agent",
                "organization": "old-corp",
                "issued_at": "2026-01-02T00:00:00+00:00",
                "status": "ACTIVE",
            }
            (certs_dir / "nmc-old-1.json").write_text(json.dumps(cert1))
            (certs_dir / "nmc-old-2.json").write_text(json.dumps(cert2))

            reg = AgentIdRegistry(base)
            count = reg.migrate_from_certs(certs_dir)
            assert count == 2

            # Should have assigned sequential IDs starting at 1000
            results = reg.list_all()
            assert len(results) == 2
            names = {r["name"] for r in results}
            assert "legacy-bot" in names
            assert "legacy-agent" in names

    def test_migrate_skips_already_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            certs_dir = base / "certs"
            certs_dir.mkdir(parents=True)

            cert1 = {
                "certificate_id": "nmc-existing",
                "agent_id": "existing-bot",
                "organization": "corp",
                "issued_at": "2026-01-01T00:00:00+00:00",
                "status": "ACTIVE",
            }
            (certs_dir / "nmc-existing.json").write_text(json.dumps(cert1))

            reg = AgentIdRegistry(base)
            # Register one first
            nid = reg.next_id()
            reg.register(nid, "existing-bot", "nmc-existing", "corp")

            # Migration should skip the already-registered cert
            count = reg.migrate_from_certs(certs_dir)
            assert count == 0


class TestDuplicateNameHandling:
    """Duplicate names are allowed across different orgs."""

    def test_duplicate_names_different_orgs(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = AgentIdRegistry(Path(tmp))
            nid1 = reg.next_id()
            reg.register(nid1, "bot", "nmc-1", "org-a")
            nid2 = reg.next_id()
            reg.register(nid2, "bot", "nmc-2", "org-b")

            results = reg.resolve("bot")
            assert len(results) == 2

            # Can disambiguate with combo
            r1 = reg.resolve(f"bot{nid1}")
            assert len(r1) == 1
            assert r1[0]["organization"] == "org-a"
