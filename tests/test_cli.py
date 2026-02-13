"""Tests for the CLI argument parsing and commands."""

import json
import tempfile
from pathlib import Path

from nomotic.cli import build_parser, main


class TestCLIArgParsing:
    def test_birth_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "birth",
            "--agent-id", "agent-1",
            "--archetype", "customer-experience",
            "--org", "acme",
            "--zone", "global/us",
        ])
        assert args.command == "birth"
        assert args.agent_id == "agent-1"
        assert args.archetype == "customer-experience"
        assert args.org == "acme"
        assert args.zone == "global/us"

    def test_birth_args_with_owner(self):
        parser = build_parser()
        args = parser.parse_args([
            "birth",
            "--agent-id", "agent-1",
            "--owner", "ops@acme.com",
            "--archetype", "arch",
            "--org", "org",
        ])
        assert args.owner == "ops@acme.com"

    def test_birth_args_no_zone(self):
        parser = build_parser()
        args = parser.parse_args([
            "birth",
            "--agent-id", "agent-1",
            "--archetype", "arch",
            "--org", "org",
        ])
        assert args.zone is None
        assert args.owner is None

    def test_verify_args(self):
        parser = build_parser()
        args = parser.parse_args(["verify", "nmc-1234"])
        assert args.command == "verify"
        assert args.cert_id == "nmc-1234"

    def test_inspect_args(self):
        parser = build_parser()
        args = parser.parse_args(["inspect", "nmc-1234"])
        assert args.command == "inspect"
        assert args.cert_id == "nmc-1234"

    def test_suspend_args(self):
        parser = build_parser()
        args = parser.parse_args(["suspend", "nmc-1234", "--reason", "policy"])
        assert args.command == "suspend"
        assert args.cert_id == "nmc-1234"
        assert args.reason == "policy"

    def test_reactivate_args(self):
        parser = build_parser()
        args = parser.parse_args(["reactivate", "nmc-1234"])
        assert args.command == "reactivate"
        assert args.cert_id == "nmc-1234"

    def test_revoke_args(self):
        parser = build_parser()
        args = parser.parse_args(["revoke", "nmc-1234", "--reason", "done"])
        assert args.command == "revoke"
        assert args.cert_id == "nmc-1234"
        assert args.reason == "done"

    def test_renew_args(self):
        parser = build_parser()
        args = parser.parse_args(["renew", "nmc-1234"])
        assert args.command == "renew"
        assert args.cert_id == "nmc-1234"

    def test_list_args(self):
        parser = build_parser()
        args = parser.parse_args(["list", "--status", "ACTIVE", "--org", "acme"])
        assert args.command == "list"
        assert args.status == "ACTIVE"
        assert args.org == "acme"

    def test_reputation_args(self):
        parser = build_parser()
        args = parser.parse_args(["reputation", "nmc-1234"])
        assert args.command == "reputation"
        assert args.cert_id == "nmc-1234"

    def test_export_args(self):
        parser = build_parser()
        args = parser.parse_args(["export", "nmc-1234"])
        assert args.command == "export"
        assert args.cert_id == "nmc-1234"

    def test_base_dir_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--base-dir", "/tmp/test", "list"])
        assert args.base_dir == Path("/tmp/test")


class TestCLICommands:
    def test_birth_command(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            main([
                "--base-dir", tmp,
                "birth",
                "--agent-id", "agent-1",
                "--owner", "ops@acme.com",
                "--archetype", "customer-experience",
                "--org", "acme",
            ])
            captured = capsys.readouterr()
            assert "Certificate issued:" in captured.out
            assert "agent-1" in captured.out
            assert "ops@acme.com" in captured.out
            assert "customer-experience" in captured.out

    def test_birth_then_verify(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            main([
                "--base-dir", tmp,
                "birth",
                "--agent-id", "agent-1",
                "--archetype", "arch",
                "--org", "org",
            ])
            captured = capsys.readouterr()
            # Extract cert ID from output
            cert_id = captured.out.split("Certificate issued: ")[1].split("\n")[0].strip()

            main(["--base-dir", tmp, "verify", cert_id])
            captured = capsys.readouterr()
            assert "VALID" in captured.out

    def test_birth_then_inspect(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            main([
                "--base-dir", tmp,
                "birth",
                "--agent-id", "agent-1",
                "--archetype", "arch",
                "--org", "org",
            ])
            captured = capsys.readouterr()
            cert_id = captured.out.split("Certificate issued: ")[1].split("\n")[0].strip()

            main(["--base-dir", tmp, "inspect", cert_id])
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["agent_id"] == "agent-1"

    def test_birth_suspend_reactivate(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            main([
                "--base-dir", tmp,
                "birth",
                "--agent-id", "agent-1",
                "--archetype", "arch",
                "--org", "org",
            ])
            captured = capsys.readouterr()
            cert_id = captured.out.split("Certificate issued: ")[1].split("\n")[0].strip()

            main(["--base-dir", tmp, "suspend", cert_id, "--reason", "test"])
            captured = capsys.readouterr()
            assert "Suspended" in captured.out

            main(["--base-dir", tmp, "reactivate", cert_id])
            captured = capsys.readouterr()
            assert "Reactivated" in captured.out

    def test_birth_then_revoke(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            main([
                "--base-dir", tmp,
                "birth",
                "--agent-id", "agent-1",
                "--archetype", "arch",
                "--org", "org",
            ])
            captured = capsys.readouterr()
            cert_id = captured.out.split("Certificate issued: ")[1].split("\n")[0].strip()

            main(["--base-dir", tmp, "revoke", cert_id, "--reason", "done"])
            captured = capsys.readouterr()
            assert "Revoked" in captured.out

    def test_birth_then_renew(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            main([
                "--base-dir", tmp,
                "birth",
                "--agent-id", "agent-1",
                "--archetype", "arch",
                "--org", "org",
            ])
            captured = capsys.readouterr()
            cert_id = captured.out.split("Certificate issued: ")[1].split("\n")[0].strip()

            main(["--base-dir", tmp, "renew", cert_id])
            captured = capsys.readouterr()
            assert "Renewed" in captured.out
            assert "Lineage:" in captured.out

    def test_list_command(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            main([
                "--base-dir", tmp,
                "birth",
                "--agent-id", "agent-1",
                "--archetype", "arch",
                "--org", "org",
            ])
            main([
                "--base-dir", tmp,
                "birth",
                "--agent-id", "agent-2",
                "--archetype", "arch",
                "--org", "org",
            ])
            capsys.readouterr()  # Clear output

            main(["--base-dir", tmp, "list"])
            captured = capsys.readouterr()
            assert "agent-1" in captured.out
            assert "agent-2" in captured.out

    def test_reputation_command(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            main([
                "--base-dir", tmp,
                "birth",
                "--agent-id", "agent-1",
                "--archetype", "arch",
                "--org", "org",
            ])
            captured = capsys.readouterr()
            cert_id = captured.out.split("Certificate issued: ")[1].split("\n")[0].strip()

            main(["--base-dir", tmp, "reputation", cert_id])
            captured = capsys.readouterr()
            assert "Trust Score:" in captured.out
            assert "Behavioral Age:" in captured.out

    def test_no_command_shows_help(self, capsys):
        import pytest
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(SystemExit) as exc_info:
                main(["--base-dir", tmp])
            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            assert "nomotic" in captured.out.lower() or "usage" in captured.out.lower()
