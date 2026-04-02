"""Tests for pipeline/ci/load_image_config.py — GITHUB_OUTPUT writer for image config.

Tests are organized around the PUBLIC API:
- main(): parses CLI args, loads config, writes key=value lines to GITHUB_OUTPUT or stdout
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.ci.load_image_config import main

VALID_SHA = "a" * 40
VALID_ISSUE = "311"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_COMPLETE_YAML = """\
dockerfile: docker/ubuntu22_04/Dockerfile
image: tinaudio/perm
base_image: "ubuntu@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751"
base_image_tag: ubuntu22_04
build_mode: prebuilt
target_platform: linux/amd64
torch_index_url: "https://download.pytorch.org/whl/cu128"
r2_endpoint: "https://example.r2.cloudflarestorage.com"
r2_bucket: test-bucket
"""

_EXPECTED_LINES = [
    "dockerfile=docker/ubuntu22_04/Dockerfile",
    "image=tinaudio/perm",
    "base_image=ubuntu@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751",
    "base_image_tag=ubuntu22_04",
    "build_mode=prebuilt",
    "target_platform=linux/amd64",
    "torch_index_url=https://download.pytorch.org/whl/cu128",
    "r2_endpoint=https://example.r2.cloudflarestorage.com",
    "r2_bucket=test-bucket",
    f"github_sha={VALID_SHA}",
    "issue_number=311",
    "image_config_id=dev-snapshot",
]


def _write_config(tmp_path: Path) -> Path:
    """Write a complete YAML config and return its path."""
    config_path = tmp_path / "dev-snapshot.yaml"
    config_path.write_text(_COMPLETE_YAML)
    return config_path


def _set_argv(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    """Override sys.argv with the given argument list."""
    monkeypatch.setattr(sys, "argv", ["load_image_config"] + args)


# ---------------------------------------------------------------------------
# main — GITHUB_OUTPUT file writing
# ---------------------------------------------------------------------------


class TestMainGithubOutput:
    """Main() writes key=value lines to the GITHUB_OUTPUT file."""

    def test_main_writes_all_fields_to_github_output(
        # plumb:req-13e3f802
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All ImageConfig fields are written as key=value lines to GITHUB_OUTPUT file."""
        config_path = _write_config(tmp_path)
        output_file = tmp_path / "github_output.txt"

        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        _set_argv(
            monkeypatch,
            [
                "--config",
                str(config_path),
                "--github-sha",
                VALID_SHA,
                "--issue-number",
                VALID_ISSUE,
            ],
        )

        main()

        written = output_file.read_text()
        lines = written.strip().splitlines()

        assert lines == _EXPECTED_LINES

    def test_main_appends_to_existing_github_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Main() appends to GITHUB_OUTPUT rather than overwriting existing content."""
        config_path = _write_config(tmp_path)
        output_file = tmp_path / "github_output.txt"
        output_file.write_text("existing=value\n")

        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        _set_argv(
            monkeypatch,
            [
                "--config",
                str(config_path),
                "--github-sha",
                VALID_SHA,
                "--issue-number",
                VALID_ISSUE,
            ],
        )

        main()

        written = output_file.read_text()
        assert written.startswith("existing=value\n")
        assert "dockerfile=docker/ubuntu22_04/Dockerfile\n" in written


# ---------------------------------------------------------------------------
# main — stdout fallback
# ---------------------------------------------------------------------------


class TestMainStdoutFallback:
    """Main() writes to stdout when GITHUB_OUTPUT is not set."""

    def test_main_writes_to_stdout_when_no_github_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When GITHUB_OUTPUT is unset, key=value lines go to stdout."""
        config_path = _write_config(tmp_path)

        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            [
                "--config",
                str(config_path),
                "--github-sha",
                VALID_SHA,
                "--issue-number",
                VALID_ISSUE,
            ],
        )

        main()

        captured = capsys.readouterr().out
        lines = captured.strip().splitlines()

        assert lines == _EXPECTED_LINES


# ---------------------------------------------------------------------------
# main — missing required args
# ---------------------------------------------------------------------------


class TestMainMissingArgs:
    """Main() exits when required CLI arguments are missing."""

    def test_main_missing_config_arg_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing --config causes SystemExit."""
        _set_argv(monkeypatch, ["--github-sha", VALID_SHA, "--issue-number", VALID_ISSUE])

        with pytest.raises(SystemExit):
            main()

    def test_main_missing_github_sha_arg_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing --github-sha causes SystemExit."""
        config_path = _write_config(tmp_path)
        _set_argv(monkeypatch, ["--config", str(config_path), "--issue-number", VALID_ISSUE])

        with pytest.raises(SystemExit):
            main()

    def test_main_missing_issue_number_arg_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing --issue-number causes SystemExit."""
        config_path = _write_config(tmp_path)
        _set_argv(monkeypatch, ["--config", str(config_path), "--github-sha", VALID_SHA])

        with pytest.raises(SystemExit):
            main()


# ---------------------------------------------------------------------------
# main — invalid SHA
# ---------------------------------------------------------------------------


class TestMainInvalidSha:
    """Main() raises ValidationError when given an invalid SHA."""

    def test_main_invalid_sha_raises(
        # plumb:req-74aa845b
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Short SHA passed via CLI triggers pydantic ValidationError."""
        config_path = _write_config(tmp_path)

        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            [
                "--config",
                str(config_path),
                "--github-sha",
                "abc123",
                "--issue-number",
                VALID_ISSUE,
            ],
        )

        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="github_sha"):
            main()


# ---------------------------------------------------------------------------
# main — newline injection prevention
# ---------------------------------------------------------------------------


class TestNewlineInjection:
    """Main() rejects config values that contain newlines."""

    def test_main_rejects_value_with_newline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A config value containing a newline raises ValueError."""
        yaml_with_newline = """\
dockerfile: docker/ubuntu22_04/Dockerfile
image: "tinaudio/perm\\nevil_key=evil_value"
base_image: "ubuntu@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751"
base_image_tag: ubuntu22_04
build_mode: prebuilt
target_platform: linux/amd64
torch_index_url: "https://download.pytorch.org/whl/cu128"
r2_endpoint: "https://example.r2.cloudflarestorage.com"
r2_bucket: test-bucket
"""
        config_path = tmp_path / "dev-snapshot.yaml"
        config_path.write_text(yaml_with_newline)
        output_file = tmp_path / "github_output.txt"

        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        _set_argv(
            monkeypatch,
            [
                "--config",
                str(config_path),
                "--github-sha",
                VALID_SHA,
                "--issue-number",
                VALID_ISSUE,
            ],
        )

        with pytest.raises(ValueError, match="contains a newline"):
            main()

    def test_main_rejects_value_with_carriage_return(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A config value containing a carriage return raises ValueError."""
        yaml_with_cr = """\
dockerfile: docker/ubuntu22_04/Dockerfile
image: "tinaudio/perm\\revil_key=evil_value"
base_image: "ubuntu@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751"
base_image_tag: ubuntu22_04
build_mode: prebuilt
target_platform: linux/amd64
torch_index_url: "https://download.pytorch.org/whl/cu128"
r2_endpoint: "https://example.r2.cloudflarestorage.com"
r2_bucket: test-bucket
"""
        config_path = tmp_path / "dev-snapshot.yaml"
        config_path.write_text(yaml_with_cr)
        output_file = tmp_path / "github_output.txt"

        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        _set_argv(
            monkeypatch,
            [
                "--config",
                str(config_path),
                "--github-sha",
                VALID_SHA,
                "--issue-number",
                VALID_ISSUE,
            ],
        )

        with pytest.raises(ValueError, match="contains a newline"):
            main()


# ---------------------------------------------------------------------------
# module invocability
# ---------------------------------------------------------------------------


class TestModuleInvocable:
    """Python -m pipeline.ci.load_image_config should work as a CLI."""

    def test_module_help_exits_zero(self) -> None:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pipeline.ci.load_image_config", "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "config" in result.stdout
