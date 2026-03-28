"""Tests for scripts/ci/load_dataset_config.py — GITHUB_OUTPUT writer for dataset config.

Tests are organized around the PUBLIC API:
- main(): parses CLI args, loads config, writes key=value lines to GITHUB_OUTPUT or stdout
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci.load_dataset_config import main

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

# Fields that are static (directly from config) and always present.
_STATIC_FIELDS = {
    "dataset_config_id": "test-dataset",
    "param_spec": "surge_simple",
    "plugin_path": "plugins/Surge XT.vst3",
    "preset_path": "presets/surge-base.vstpreset",
    "sample_rate": "16000",
    "channels": "2",
    "velocity": "100",
    "signal_duration_seconds": "4.0",
    "min_loudness": "-55.0",
    "sample_batch_size": "32",
}


def _write_config(tmp_path: Path) -> Path:
    """Write a complete dataset config YAML and return its path."""
    config_path = tmp_path / "test-dataset.yaml"
    config_path.write_text(yaml.safe_dump(_COMPLETE_CONFIG, sort_keys=False))
    return config_path


def _set_argv(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    """Override sys.argv with the given argument list."""
    monkeypatch.setattr(sys, "argv", ["load_dataset_config"] + args)


def _parse_output(text: str) -> dict[str, str]:
    """Parse key=value lines into a dict."""
    result = {}
    for line in text.strip().splitlines():
        key, _, value = line.partition("=")
        result[key] = value
    return result


# ---------------------------------------------------------------------------
# main — GITHUB_OUTPUT file writing
# ---------------------------------------------------------------------------


class TestMainGithubOutput:
    """Main() writes key=value lines to the GITHUB_OUTPUT file."""

    def test_main_writes_all_fields_to_github_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All dataset config fields are written as key=value lines to GITHUB_OUTPUT file."""
        config_path = _write_config(tmp_path)
        output_file = tmp_path / "github_output.txt"

        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        _set_argv(monkeypatch, ["--config", str(config_path)])

        main()

        fields = _parse_output(output_file.read_text())

        for key, expected in _STATIC_FIELDS.items():
            assert fields[key] == expected, f"Field {key}: {fields[key]!r} != {expected!r}"

        # num_samples = shard_size * num_shards = 10000 * 48
        assert fields["num_samples"] == "480000"
        assert "run_id" in fields
        assert "r2_prefix" in fields
        assert fields["r2_prefix"].startswith("data/test-dataset/")

    def test_main_appends_to_existing_github_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Main() appends to GITHUB_OUTPUT rather than overwriting existing content."""
        config_path = _write_config(tmp_path)
        output_file = tmp_path / "github_output.txt"
        output_file.write_text("existing=value\n")

        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        _set_argv(monkeypatch, ["--config", str(config_path)])

        main()

        written = output_file.read_text()
        assert written.startswith("existing=value\n")
        assert "dataset_config_id=test-dataset\n" in written


# ---------------------------------------------------------------------------
# main — stdout fallback
# ---------------------------------------------------------------------------


class TestMainStdoutFallback:
    """Main() writes to stdout when GITHUB_OUTPUT is not set."""

    def test_main_writes_to_stdout_when_no_github_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When GITHUB_OUTPUT is unset, key=value lines go to stdout."""
        config_path = _write_config(tmp_path)

        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(monkeypatch, ["--config", str(config_path)])

        main()

        fields = _parse_output(capsys.readouterr().out)

        for key, expected in _STATIC_FIELDS.items():
            assert fields[key] == expected


# ---------------------------------------------------------------------------
# main — missing required args
# ---------------------------------------------------------------------------


class TestMainMissingArgs:
    """Main() exits when required CLI arguments are missing."""

    def test_main_missing_config_arg_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing --config causes SystemExit."""
        _set_argv(monkeypatch, [])

        with pytest.raises(SystemExit):
            main()


# ---------------------------------------------------------------------------
# main — derived fields
# ---------------------------------------------------------------------------


class TestDerivedFields:
    """num_samples, run ID, and R2 prefix are derived from config."""

    def test_num_samples_is_shard_size_times_num_shards(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """num_samples = shard_size * num_shards from config."""
        config_path = _write_config(tmp_path)

        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(monkeypatch, ["--config", str(config_path)])

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["num_samples"] == "480000"

    def test_run_id_contains_config_id_and_timestamp(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """run_id follows the pattern {config_id}-{YYYYMMDDTHHMMSSZ}."""
        config_path = _write_config(tmp_path)

        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(monkeypatch, ["--config", str(config_path)])

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert re.match(r"test-dataset-\d{8}T\d{6}Z$", fields["run_id"])

    def test_r2_prefix_follows_convention(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """r2_prefix follows data/{config_id}/{run_id}/ pattern with trailing slash."""
        config_path = _write_config(tmp_path)

        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _set_argv(monkeypatch, ["--config", str(config_path)])

        main()

        fields = _parse_output(capsys.readouterr().out)
        assert fields["r2_prefix"].startswith("data/test-dataset/")
        assert fields["r2_prefix"].endswith("/")
