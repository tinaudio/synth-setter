"""Tests for scripts/ci/resolve_dataset_run_params.py — run parameter resolver.

Tests are organized around the PUBLIC API:
- main(): parses CLI args, resolves parameters, writes key=value lines to GITHUB_OUTPUT or stdout
- resolve_params(): fills empty inputs with defaults
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.ci.resolve_dataset_run_params import main, resolve_params


def _set_argv(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    """Override sys.argv with the given argument list."""
    monkeypatch.setattr(sys, "argv", ["resolve_dataset_run_params"] + args)


def _parse_output(text: str) -> dict[str, str]:
    """Parse key=value lines into a dict."""
    result = {}
    for line in text.strip().splitlines():
        key, _, value = line.partition("=")
        result[key] = value
    return result


# ---------------------------------------------------------------------------
# resolve_params — default resolution
# ---------------------------------------------------------------------------


class TestResolveParams:
    """resolve_params() fills empty inputs with defaults."""

    def test_all_empty_uses_defaults(self) -> None:
        """Empty inputs resolve to default config, dev-snapshot, upload=true."""
        result = resolve_params(dataset_config="", docker_tag="", upload_to_r2="")

        assert result["dataset_config"] == "configs/dataset/surge-simple-480k-10k.yaml"
        assert result["docker_tag"] == "dev-snapshot"
        assert result["upload_to_r2"] == "true"

    def test_explicit_values_passed_through(self) -> None:
        """Non-empty inputs are passed through unchanged."""
        result = resolve_params(
            dataset_config="configs/dataset/ci-smoke-test.yaml",
            docker_tag="dev-snapshot-abc123",
            upload_to_r2="false",
        )

        assert result["dataset_config"] == "configs/dataset/ci-smoke-test.yaml"
        assert result["docker_tag"] == "dev-snapshot-abc123"
        assert result["upload_to_r2"] == "false"

    def test_upload_to_r2_normalized_to_lowercase(self) -> None:
        """upload_to_r2 is normalized to lowercase."""
        result = resolve_params(dataset_config="", docker_tag="", upload_to_r2="True")
        assert result["upload_to_r2"] == "true"

    def test_invalid_upload_to_r2_raises(self) -> None:
        """Invalid upload_to_r2 value raises ValueError."""
        with pytest.raises(ValueError, match="must be 'true' or 'false'"):
            resolve_params(dataset_config="", docker_tag="", upload_to_r2="yes")

    def test_no_num_samples_in_output(self) -> None:
        """num_samples is not in the output — config is the source of truth."""
        result = resolve_params(dataset_config="", docker_tag="", upload_to_r2="")
        assert "num_samples" not in result


# ---------------------------------------------------------------------------
# GITHUB_OUTPUT file writing
# ---------------------------------------------------------------------------


class TestGithubOutput:
    """Main() writes key=value lines to the GITHUB_OUTPUT file."""

    def test_writes_all_fields_to_github_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All resolved fields are written as key=value lines to GITHUB_OUTPUT file."""
        output_file = tmp_path / "github_output.txt"

        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        _set_argv(monkeypatch, ["--dataset-config", "configs/dataset/ci-smoke-test.yaml"])

        main()

        fields = _parse_output(output_file.read_text())
        assert fields["dataset_config"] == "configs/dataset/ci-smoke-test.yaml"
        assert fields["docker_tag"] == "dev-snapshot"
        assert fields["upload_to_r2"] == "true"

    def test_appends_to_existing_github_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Main() appends to GITHUB_OUTPUT rather than overwriting existing content."""
        output_file = tmp_path / "github_output.txt"
        output_file.write_text("existing=value\n")

        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        _set_argv(monkeypatch, ["--dataset-config", "configs/dataset/ci-smoke-test.yaml"])

        main()

        written = output_file.read_text()
        assert written.startswith("existing=value\n")
        assert "dataset_config=" in written


# ---------------------------------------------------------------------------
# stdout fallback
# ---------------------------------------------------------------------------


class TestStdoutFallback:
    """Main() writes to stdout when GITHUB_OUTPUT is not set."""

    def test_writes_to_stdout_when_no_github_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When GITHUB_OUTPUT is unset, key=value lines go to stdout."""
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(monkeypatch, [])

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["dataset_config"] == "configs/dataset/surge-simple-480k-10k.yaml"
        assert fields["docker_tag"] == "dev-snapshot"
        assert fields["upload_to_r2"] == "true"
