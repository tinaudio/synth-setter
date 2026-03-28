"""Tests for scripts/ci/resolve_dataset_run_params.py — run parameter resolver.

Tests are organized around the PUBLIC API:
- main(): parses CLI args, resolves parameters, writes key=value lines to GITHUB_OUTPUT or stdout
- resolve_params(): resolves parameters from event type and CLI inputs using dataset config
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci.resolve_dataset_run_params import main

_COMPLETE_CONFIG = {
    "param_spec": "surge_simple",
    "plugin_path": "plugins/Surge XT.vst3",
    "output_format": "hdf5",
    "sample_rate": 16000,
    "shard_size": 10000,
    "num_shards": 48,
    "base_seed": 42,
    "splits": {"train": 44, "val": 2, "test": 2},
    "preset_path": "presets/surge-base.vstpreset",
    "channels": 2,
    "velocity": 100,
    "signal_duration_seconds": 4.0,
    "min_loudness": -55.0,
    "sample_batch_size": 32,
}


def _write_config(tmp_path: Path) -> Path:
    """Write a complete dataset config YAML and return its path."""
    config_path = tmp_path / "test-dataset.yaml"
    config_path.write_text(yaml.safe_dump(_COMPLETE_CONFIG, sort_keys=False))
    return config_path


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
# PR mode — pull_request event
# ---------------------------------------------------------------------------


class TestPrMode:
    """Pull request events use sample_batch_size and disable R2 upload."""

    def test_num_samples_equals_sample_batch_size(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """PR mode sets num_samples to sample_batch_size from config (one batch = smoke test)."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            ["--event-name", "pull_request", "--dataset-config", str(config_path)],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["num_samples"] == "32"

    def test_upload_to_r2_is_false(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """PR mode always disables R2 upload."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            ["--event-name", "pull_request", "--dataset-config", str(config_path)],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["upload_to_r2"] == "false"

    def test_docker_tag_uses_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """PR mode uses the default docker tag."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            ["--event-name", "pull_request", "--dataset-config", str(config_path)],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["docker_tag"] == "dev-snapshot"

    def test_dataset_config_passed_through(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """PR mode passes the dataset config path through to output."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            ["--event-name", "pull_request", "--dataset-config", str(config_path)],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["dataset_config"] == str(config_path)


# ---------------------------------------------------------------------------
# Dispatch mode — workflow_dispatch with explicit values
# ---------------------------------------------------------------------------


class TestDispatchMode:
    """Workflow dispatch uses provided CLI values and config-derived num_samples."""

    def test_num_samples_from_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Dispatch mode derives num_samples from shard_size * num_shards."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            [
                "--event-name",
                "workflow_dispatch",
                "--dataset-config",
                str(config_path),
                "--docker-tag",
                "dev-snapshot-abc123",
                "--upload-to-r2",
                "false",
            ],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["num_samples"] == "480000"

    def test_uses_provided_docker_tag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Dispatch mode uses explicitly provided docker tag."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            [
                "--event-name",
                "workflow_dispatch",
                "--dataset-config",
                str(config_path),
                "--docker-tag",
                "dev-snapshot-abc123",
                "--upload-to-r2",
                "true",
            ],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["docker_tag"] == "dev-snapshot-abc123"

    def test_uses_provided_upload_to_r2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Dispatch mode uses explicitly provided upload_to_r2."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            [
                "--event-name",
                "workflow_dispatch",
                "--dataset-config",
                str(config_path),
                "--docker-tag",
                "dev-snapshot-abc123",
                "--upload-to-r2",
                "false",
            ],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["upload_to_r2"] == "false"

    def test_uses_provided_dataset_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Dispatch mode uses explicitly provided dataset config path."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            [
                "--event-name",
                "workflow_dispatch",
                "--dataset-config",
                str(config_path),
                "--docker-tag",
                "dev-snapshot-abc123",
                "--upload-to-r2",
                "true",
            ],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["dataset_config"] == str(config_path)


# ---------------------------------------------------------------------------
# Dispatch mode — fallback defaults
# ---------------------------------------------------------------------------


class TestDispatchDefaults:
    """Workflow dispatch falls back to config-derived values when inputs are empty."""

    def test_num_samples_defaults_to_shard_size_times_num_shards(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Empty num_samples falls back to shard_size * num_shards from config."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            [
                "--event-name",
                "workflow_dispatch",
                "--dataset-config",
                str(config_path),
            ],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        # 10000 * 48 = 480000
        assert fields["num_samples"] == "480000"

    def test_docker_tag_defaults_to_dev_snapshot(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Empty docker_tag falls back to dev-snapshot."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            [
                "--event-name",
                "workflow_dispatch",
                "--dataset-config",
                str(config_path),
            ],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["docker_tag"] == "dev-snapshot"

    def test_upload_to_r2_defaults_to_true(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Empty upload_to_r2 falls back to true for dispatch events."""
        config_path = _write_config(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            [
                "--event-name",
                "workflow_dispatch",
                "--dataset-config",
                str(config_path),
            ],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["upload_to_r2"] == "true"


# ---------------------------------------------------------------------------
# GITHUB_OUTPUT file writing
# ---------------------------------------------------------------------------


class TestGithubOutput:
    """Main() writes key=value lines to the GITHUB_OUTPUT file."""

    def test_writes_all_fields_to_github_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All resolved fields are written as key=value lines to GITHUB_OUTPUT file."""
        config_path = _write_config(tmp_path)
        output_file = tmp_path / "github_output.txt"

        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        _set_argv(
            monkeypatch,
            ["--event-name", "pull_request", "--dataset-config", str(config_path)],
        )

        main()

        fields = _parse_output(output_file.read_text())
        assert fields["dataset_config"] == str(config_path)
        assert fields["num_samples"] == "32"
        assert fields["docker_tag"] == "dev-snapshot"
        assert fields["upload_to_r2"] == "false"

    def test_appends_to_existing_github_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Main() appends to GITHUB_OUTPUT rather than overwriting existing content."""
        config_path = _write_config(tmp_path)
        output_file = tmp_path / "github_output.txt"
        output_file.write_text("existing=value\n")

        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        _set_argv(
            monkeypatch,
            ["--event-name", "pull_request", "--dataset-config", str(config_path)],
        )

        main()

        written = output_file.read_text()
        assert written.startswith("existing=value\n")
        assert "num_samples=32\n" in written


# ---------------------------------------------------------------------------
# stdout fallback
# ---------------------------------------------------------------------------


class TestStdoutFallback:
    """Main() writes to stdout when GITHUB_OUTPUT is not set."""

    def test_writes_to_stdout_when_no_github_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When GITHUB_OUTPUT is unset, key=value lines go to stdout."""
        config_path = _write_config(tmp_path)

        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(
            monkeypatch,
            ["--event-name", "pull_request", "--dataset-config", str(config_path)],
        )

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["dataset_config"] == str(config_path)
        assert fields["num_samples"] == "32"
        assert fields["docker_tag"] == "dev-snapshot"
        assert fields["upload_to_r2"] == "false"
