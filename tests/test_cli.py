"""
tests/test_cli.py
=================
Tests for the CLI commands in verity/cli.py.

Structure:
  TestCLIInit   — init command: dry-run, key creation, already-signed guard,
                  --force re-initialization, enterprise ceremony message.
  TestCLIStatus — status command: unsigned / signed / tampered detection,
                  Python version output, graph backend output.

Design notes:
  - All tests use monkeypatch to redirect PRINCIPLES_PATH and _key_dir so
    they never touch the real principles.yaml or ~/.verity/keys/.
  - Commands are called directly (cmd_init / cmd_status) rather than via
    subprocess so tests run fast and stay in-process.
  - The real principles.yaml (signature: null) is copied to tmp_path for
    each test that needs it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pytest
import yaml

# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_init_args(
    profile: str = "personal",
    dry_run: bool = False,
    force: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(profile=profile, dry_run=dry_run, force=force)


def _make_status_args() -> argparse.Namespace:
    return argparse.Namespace()


def _copy_principles(tmp_path: Path) -> Path:
    """Copy the real principles.yaml (unsigned) to *tmp_path*."""
    import verity.cli as cli  # import after any monkeypatching

    src = cli._REPO_ROOT / "principles.yaml"
    dst = tmp_path / "principles.yaml"
    shutil.copy(src, dst)
    return dst


# ── TestCLIInit ───────────────────────────────────────────────────────────────

class TestCLIInit:
    """Tests for the ``verity init`` command."""

    def test_dry_run_prints_without_writing_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--dry-run prints what would happen and creates no files."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        key_dir = tmp_path / "keys"
        monkeypatch.setattr(cli, "_key_dir", lambda: key_dir)

        rc = cli.cmd_init(_make_init_args(dry_run=True))

        captured = capsys.readouterr()
        assert rc == 0
        assert "dry-run" in captured.out
        assert "would" in captured.out
        # No key file should have been created
        assert not key_dir.exists()
        # principles.yaml must still be unsigned
        data = yaml.safe_load(tmp_principles.read_text())
        assert data.get("signature") is None

    def test_init_personal_creates_key_file_at_expected_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """init --profile personal writes a private key at the expected path."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        key_dir = tmp_path / ".verity" / "keys"
        monkeypatch.setattr(cli, "_key_dir", lambda: key_dir)

        rc = cli.cmd_init(_make_init_args(profile="personal"))

        captured = capsys.readouterr()
        assert rc == 0
        assert "initialized" in captured.out.lower()
        assert "personal" in captured.out

        key_file = key_dir / "verity_personal.pem"
        assert key_file.exists(), "private key file should have been created"
        assert key_file.read_bytes().startswith(b"-----BEGIN PRIVATE KEY-----")

        # Permissions must be 0o600
        file_mode = key_file.stat().st_mode & 0o777
        assert file_mode == 0o600, f"expected 0o600, got {oct(file_mode)}"

    def test_init_personal_writes_signature_to_principles_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """After init, principles.yaml has non-null signed_by and signature fields."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        monkeypatch.setattr(cli, "_key_dir", lambda: tmp_path / "keys")

        cli.cmd_init(_make_init_args())

        data = yaml.safe_load(tmp_principles.read_text())
        assert data.get("signed_by") is not None
        assert data.get("signature") is not None
        # Public key should be a PEM public key
        assert str(data["signed_by"]).startswith("-----BEGIN PUBLIC KEY-----")

    def test_init_developer_uses_different_key_filename(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Developer profile stores the key as verity_dev.pem."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        key_dir = tmp_path / "keys"
        monkeypatch.setattr(cli, "_key_dir", lambda: key_dir)

        rc = cli.cmd_init(_make_init_args(profile="developer"))

        assert rc == 0
        assert (key_dir / "verity_dev.pem").exists()

    def test_init_already_signed_exits_0_with_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Second init without --force exits 0 and prints 'already initialized'."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        monkeypatch.setattr(cli, "_key_dir", lambda: tmp_path / "keys")

        cli.cmd_init(_make_init_args())       # first init
        capsys.readouterr()                   # clear output

        rc = cli.cmd_init(_make_init_args())  # second init (no --force)
        captured = capsys.readouterr()

        assert rc == 0
        assert "already initialized" in captured.out.lower()

    def test_init_already_signed_does_not_change_principles(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Second init without --force leaves principles.yaml unchanged."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        monkeypatch.setattr(cli, "_key_dir", lambda: tmp_path / "keys")

        cli.cmd_init(_make_init_args())
        content_after_first = tmp_principles.read_text()

        cli.cmd_init(_make_init_args())
        content_after_second = tmp_principles.read_text()

        assert content_after_first == content_after_second

    def test_init_force_reinitializes_with_new_signature(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--force re-runs init and writes a new signature (new keypair each time)."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        monkeypatch.setattr(cli, "_key_dir", lambda: tmp_path / "keys")

        cli.cmd_init(_make_init_args())
        sig1 = yaml.safe_load(tmp_principles.read_text()).get("signature")

        cli.cmd_init(_make_init_args(force=True))
        sig2 = yaml.safe_load(tmp_principles.read_text()).get("signature")

        assert sig1 != sig2, "re-initialization should produce a different signature"

    def test_init_enterprise_prints_ceremony_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Enterprise profile prints the ceremony message without touching files."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        monkeypatch.setattr(cli, "_key_dir", lambda: tmp_path / "keys")

        rc = cli.cmd_init(_make_init_args(profile="enterprise"))
        captured = capsys.readouterr()

        assert rc == 0
        assert "ceremony" in captured.out.lower()
        # principles.yaml must remain unsigned
        data = yaml.safe_load(tmp_principles.read_text())
        assert data.get("signature") is None

    def test_init_professional_prints_ceremony_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Professional profile also prints the ceremony message."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        monkeypatch.setattr(cli, "_key_dir", lambda: tmp_path / "keys")

        rc = cli.cmd_init(_make_init_args(profile="professional"))
        captured = capsys.readouterr()

        assert rc == 0
        assert "ceremony" in captured.out.lower()


# ── TestCLIStatus ─────────────────────────────────────────────────────────────

class TestCLIStatus:
    """Tests for the ``verity status`` command."""

    def test_status_unsigned_principles_reports_unsigned(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Status on unsigned principles.yaml prints 'unsigned'."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)

        rc = cli.cmd_status(_make_status_args())
        captured = capsys.readouterr()

        assert rc == 0
        assert "unsigned" in captured.out

    def test_status_signed_principles_reports_signed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Status on correctly signed principles.yaml prints 'signed'."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        monkeypatch.setattr(cli, "_key_dir", lambda: tmp_path / "keys")

        cli.cmd_init(_make_init_args())  # sign it
        capsys.readouterr()             # clear init output

        rc = cli.cmd_status(_make_status_args())
        captured = capsys.readouterr()

        assert rc == 0
        assert "signed" in captured.out
        assert "tampered" not in captured.out

    def test_status_tampered_principles_reports_tampered_and_exits_1(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Status on tampered principles.yaml prints 'tampered' and exits 1."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)
        monkeypatch.setattr(cli, "_key_dir", lambda: tmp_path / "keys")

        cli.cmd_init(_make_init_args())  # sign it

        # Tamper: append a space to the timestamp field
        content = tmp_principles.read_text()
        tampered = content.replace(
            'timestamp: "2025-01-01T00:00:00Z"',
            'timestamp: "2025-01-01T00:00:01Z"',
        )
        tmp_principles.write_text(tampered)

        capsys.readouterr()  # clear init output
        rc = cli.cmd_status(_make_status_args())
        captured = capsys.readouterr()

        assert rc == 1
        assert "tampered" in captured.out

    def test_status_prints_python_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Status output includes the current Python version."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)

        cli.cmd_status(_make_status_args())
        captured = capsys.readouterr()

        python_version = sys.version.split()[0]
        assert python_version in captured.out

    def test_status_prints_graph_backend(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Status output includes graph backend information."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)

        cli.cmd_status(_make_status_args())
        captured = capsys.readouterr()

        assert "graph backend" in captured.out.lower()
        # The backend line mentions rdflib (the default)
        assert "rdflib" in captured.out

    def test_status_prints_profile_default_personal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Status shows 'personal' as the default profile."""
        import verity.cli as cli

        tmp_principles = _copy_principles(tmp_path)
        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_principles)

        cli.cmd_status(_make_status_args())
        captured = capsys.readouterr()

        assert "personal" in captured.out

    def test_status_missing_principles_returns_exit_1(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Status exits 1 (via error message) when principles.yaml is missing."""
        import verity.cli as cli

        monkeypatch.setattr(cli, "PRINCIPLES_PATH", tmp_path / "nonexistent.yaml")

        rc = cli.cmd_status(_make_status_args())
        captured = capsys.readouterr()

        assert rc == 1
        assert "not found" in captured.out.lower()
