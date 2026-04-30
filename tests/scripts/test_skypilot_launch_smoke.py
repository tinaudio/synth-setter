"""Tests for scripts/skypilot_launch_smoke.py — SkyPilot RunPod smoke launcher.

Mock-based: no real SkyPilot or RunPod calls. The `sky` module is lazily
imported inside `main`, so tests inject a MagicMock via `sys.modules` before
invoking the CLI.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

from pipeline.schemas.spec import DatasetPipelineSpec
from scripts.skypilot_launch_smoke import (
    WORKER_SPEC_PATH,
    load_worker_env,
    main,
    write_spec_to_tempfile,
)

FIXED_NOW = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)

VALID_CONFIG = {
    "param_spec": "surge_simple",
    "plugin_path": "PLACEHOLDER",
    "output_format": "hdf5",
    "sample_rate": 16000,
    "shard_size": 32,
    "num_shards": 1,
    "base_seed": 42,
    "r2_bucket": "intermediate-data",
    "splits": {"train": 1, "val": 0, "test": 0},
    "preset_path": "presets/surge-base.vstpreset",
    "channels": 2,
    "velocity": 100,
    "signal_duration_seconds": 4.0,
    "min_loudness": -55.0,
    "sample_batch_size": 32,
}


@pytest.fixture()
def fake_plugin(tmp_path: Path) -> Path:
    """Build a minimal VST3 bundle with a moduleinfo.json the renderer can read."""
    contents = tmp_path / "FakePlugin.vst3" / "Contents"
    contents.mkdir(parents=True)
    (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')
    return tmp_path / "FakePlugin.vst3"


@pytest.fixture()
def patch_materialize_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out git/timestamp I/O so `materialize_spec` is deterministic."""
    monkeypatch.setattr("pipeline.schemas.spec._get_git_sha", lambda: "abc123def456")
    monkeypatch.setattr("pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr(
        "pipeline.schemas.spec.datetime",
        type(
            "FakeDatetime",
            (),
            {
                "now": staticmethod(lambda tz: FIXED_NOW),
                "fromisoformat": datetime.fromisoformat,
            },
        )(),
    )


@pytest.fixture()
def config_yaml(tmp_path: Path, fake_plugin: Path) -> Path:
    """Write a valid DatasetConfig YAML pointed at the fake plugin."""
    data = dict(VALID_CONFIG, plugin_path=str(fake_plugin))
    path = tmp_path / "ci-smoke-test.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.fixture()
def env_file(tmp_path: Path) -> Path:
    """Write a minimal valid .env.cloud with the rclone-R2 keys the launcher forwards."""
    path = tmp_path / ".env.cloud"
    path.write_text(
        "RCLONE_CONFIG_R2_TYPE=s3\n"
        "RCLONE_CONFIG_R2_PROVIDER=Cloudflare\n"
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID=key\n"
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=secret\n"
        "RCLONE_CONFIG_R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
    )
    return path


@pytest.fixture()
def template_yaml() -> Path:
    """Resolve the in-repo SkyPilot RunPod template path."""
    return (
        Path(__file__).resolve().parent.parent.parent
        / "configs"
        / "compute"
        / "runpod-template.yaml"
    )


@pytest.fixture()
def mock_sky(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject MagicMocks for the lazily-imported `sky` and `sky.jobs` modules."""
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setitem(sys.modules, "sky.jobs", fake.jobs)
    return fake


# ---------------------------------------------------------------------------
# load_worker_env
# ---------------------------------------------------------------------------


class TestLoadWorkerEnv:
    """Behavioral contracts for the worker-env loader (thin wrapper over python-dotenv)."""

    def test_parses_keys_skips_comments_and_strips_quotes(self, tmp_path: Path) -> None:
        """Python-dotenv handles blanks, `#` comments, and quoted values; we only forward dict[str,
        str]."""
        path = tmp_path / ".env.cloud"
        path.write_text(
            "# a comment\n"
            "\n"
            "FOO=bar\n"
            'QUOTED="bar baz"\n'
            "URL=https://acct.r2.cloudflarestorage.com/path?x=1\n"
        )
        assert load_worker_env(path) == {
            "FOO": "bar",
            "QUOTED": "bar baz",
            "URL": "https://acct.r2.cloudflarestorage.com/path?x=1",
        }

    def test_drops_keys_with_no_value(self, tmp_path: Path) -> None:
        """Lines like `BARE` (no `=`) come back as None from dotenv; loader filters them out."""
        path = tmp_path / ".env.cloud"
        path.write_text("FOO=bar\nBARE\n")
        assert load_worker_env(path) == {"FOO": "bar"}


# ---------------------------------------------------------------------------
# write_spec_to_tempfile
# ---------------------------------------------------------------------------


class TestWriteSpecToTempfile:
    """Behavioral contracts for the materialized-spec tempfile helper."""

    def test_writes_content_and_returns_path(self) -> None:
        """Content is written verbatim, the returned Path exists, and it has a `.json` suffix."""
        spec_json = '{"hello": "world"}'
        path = write_spec_to_tempfile(spec_json)
        try:
            assert path.exists()
            assert path.read_text() == spec_json
            assert path.suffix == ".json"
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# main — CLI integration with mocked sky
# ---------------------------------------------------------------------------


class TestMainCli:
    """End-to-end CLI behavior: env validation, spec materialization, sky.* call shape."""

    def test_missing_env_file_fails_before_sky_import(
        self,
        tmp_path: Path,
        config_yaml: Path,
        template_yaml: Path,
        patch_materialize_io: None,
    ) -> None:
        """Missing .env.cloud raises before sky.* is touched (sky absent from sys.modules)."""
        # Ensure sky is not importable: simulate it not being installed.
        original = sys.modules.pop("sky", None)
        try:
            missing = tmp_path / "does-not-exist.env"
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--config",
                    str(config_yaml),
                    "--template",
                    str(template_yaml),
                    "--env-file",
                    str(missing),
                ],
            )
            assert result.exit_code != 0
            assert "Worker env file not found" in result.output
        finally:
            if original is not None:
                sys.modules["sky"] = original

    def test_empty_env_file_fails(
        self,
        tmp_path: Path,
        config_yaml: Path,
        template_yaml: Path,
        patch_materialize_io: None,
    ) -> None:
        """An env file containing only blank/comment lines fails fast with a clear error."""
        empty_env = tmp_path / "empty.env"
        empty_env.write_text("# only comments\n\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--config",
                str(config_yaml),
                "--template",
                str(template_yaml),
                "--env-file",
                str(empty_env),
            ],
        )
        assert result.exit_code != 0
        assert "No env vars parsed" in result.output

    def test_submits_job_with_expected_arguments(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        mock_sky: MagicMock,
    ) -> None:
        """End-to-end: spec is materialized, sky.* is called with expected args."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--config",
                str(config_yaml),
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
                "--job-name",
                "smoke-job-1",
            ],
        )
        assert result.exit_code == 0, result.output

        mock_sky.Task.from_yaml.assert_called_once_with(str(template_yaml))
        task = mock_sky.Task.from_yaml.return_value

        task.update_envs.assert_called_once()
        forwarded_envs = task.update_envs.call_args.args[0]
        assert forwarded_envs["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "key"
        assert forwarded_envs["RCLONE_CONFIG_R2_ENDPOINT"].startswith("https://")

        task.update_file_mounts.assert_called_once()
        mounts: dict[str, str] = task.update_file_mounts.call_args.args[0]
        assert WORKER_SPEC_PATH in mounts
        spec_path = Path(mounts[WORKER_SPEC_PATH])
        try:
            assert spec_path.is_file()
            # Round-trip: the materialized JSON validates as a DatasetPipelineSpec.
            spec = DatasetPipelineSpec.model_validate_json(spec_path.read_text())
            assert spec.code_version == "abc123def456"
            assert spec.is_repo_dirty is False
            assert spec.num_shards == 1
            assert spec.r2_bucket == "intermediate-data"
        finally:
            spec_path.unlink(missing_ok=True)

        mock_sky.jobs.launch.assert_called_once_with(task, name="smoke-job-1")

    def test_default_job_name_uses_config_id_prefix(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        mock_sky: MagicMock,
    ) -> None:
        """When --job-name is omitted, the launcher derives the name from `config_id[:8]`."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--config",
                str(config_yaml),
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
            ],
        )
        assert result.exit_code == 0, result.output

        # Config stem is "ci-smoke-test"; first 8 chars → "ci-smoke".
        kwargs: dict[str, Any] = mock_sky.jobs.launch.call_args.kwargs
        assert kwargs["name"] == "synth-setter-smoke-ci-smoke"

        # Clean up the materialized spec tempfile.
        mounts: dict[str, str] = (
            mock_sky.Task.from_yaml.return_value.update_file_mounts.call_args.args[0]
        )
        Path(mounts[WORKER_SPEC_PATH]).unlink(missing_ok=True)
