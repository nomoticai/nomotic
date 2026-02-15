"""Tests for the ``nomotic hello`` CLI command."""

import subprocess
import sys
from pathlib import Path

import pytest

from nomotic.cli import main


class TestHelloCommand:
    def test_hello_runs_without_error(self, capsys):
        """The hello command should redirect to the tutorial."""
        # Run the hello command via the CLI main()
        try:
            main(["hello"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "Starting the Nomotic tutorial..." in captured.out
        assert "Tutorial complete" in captured.out

    def test_hello_does_not_create_files(self, tmp_path, monkeypatch):
        """The hello command should not create any files in ~/.nomotic/."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        try:
            main(["hello"])
        except SystemExit:
            pass

        nomotic_dir = fake_home / ".nomotic"
        # The hello command should not create ~/.nomotic/
        assert not nomotic_dir.exists()

    def test_hello_produces_stdout(self, capsys):
        """The hello command produces output to stdout."""
        try:
            main(["hello"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert len(captured.out) > 0
