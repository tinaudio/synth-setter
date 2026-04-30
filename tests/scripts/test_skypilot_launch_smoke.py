"""Tests for scripts/skypilot_launch_smoke.py — SkyPilot RunPod smoke launcher.

Mock-based: no real SkyPilot or RunPod calls. The `mock_sky` fixture replaces the launcher's
module-level `sky` reference with a MagicMock, and `local_spec_path` redirects the on-disk spec
write to a tmp path so tests don't write into the real repo's data/ dir.
"""

from __future__ import annotations

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
def local_spec_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the launcher's on-disk spec write to a tmp path (don't pollute the real data/)."""
    path = tmp_path / "spec-out" / "skypilot-launch-smoke-spec.json"
    monkeypatch.setattr("scripts.skypilot_launch_smoke.LOCAL_SPEC_PATH", path)
    return path


@pytest.fixture()
def mock_sky(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the launcher's module-level `sky` reference with a MagicMock."""
    fake = MagicMock()
    monkeypatch.setattr("scripts.skypilot_launch_smoke.sky", fake)
    return fake


# ---------------------------------------------------------------------------
# load_worker_env
# ---------------------------------------------------------------------------


class TestLoadWorkerEnv:
    """Behavioral contracts for the worker-env loader (thin wrapper over python-dotenv)."""

    def test_parses_keys_skips_comments_and_strips_quotes(self, tmp_path: Path) -> None:
        """Python-dotenv handles blanks, comments, and quoted values; loader returns dict[str,
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
# main — CLI integration with mocked sky
# ---------------------------------------------------------------------------


class TestMainCli:
    """End-to-end CLI behavior: env validation, spec materialization, sky.* call shape."""

    def test_missing_env_file_fails_with_clear_error(
        self,
        tmp_path: Path,
        config_yaml: Path,
        template_yaml: Path,
        patch_materialize_io: None,
        local_spec_path: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Missing .env.cloud aborts with a clear error and never calls sky.*."""
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
        mock_sky.Task.from_yaml.assert_not_called()
        mock_sky.launch.assert_not_called()

    def test_empty_env_file_fails(
        self,
        tmp_path: Path,
        config_yaml: Path,
        template_yaml: Path,
        patch_materialize_io: None,
        local_spec_path: Path,
        mock_sky: MagicMock,
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
        mock_sky.Task.from_yaml.assert_not_called()
        mock_sky.launch.assert_not_called()

    def test_submits_job_with_expected_arguments(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_path: Path,
        mock_sky: MagicMock,
    ) -> None:
        """End-to-end: spec is materialized to LOCAL_SPEC_PATH and sky.* is called with expected args."""
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
        mounts = task.update_file_mounts.call_args.args[0]
        assert WORKER_SPEC_PATH in mounts
        # Mount source is a sibling copy of LOCAL_SPEC_PATH (so SkyPilot's staging-by-rename
        # doesn't consume the original file the CI artifact upload depends on).
        mount_source = Path(mounts[WORKER_SPEC_PATH])
        assert mount_source != local_spec_path
        assert mount_source.is_file()
        assert mount_source.read_text() == local_spec_path.read_text()

        # Round-trip: the materialized JSON validates as a DatasetPipelineSpec.
        assert local_spec_path.is_file()
        spec = DatasetPipelineSpec.model_validate_json(local_spec_path.read_text())
        assert spec.code_version == "abc123def456"
        assert spec.is_repo_dirty is False
        assert spec.num_shards == 1
        assert spec.r2_bucket == "intermediate-data"

        mock_sky.launch.assert_called_once_with(task, cluster_name="smoke-job-1", down=True)
        mock_sky.stream_and_get.assert_called_once_with(mock_sky.launch.return_value, follow=True)

    def test_default_job_name_uses_config_id_prefix(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_path: Path,
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
        kwargs: dict[str, Any] = mock_sky.launch.call_args.kwargs
        assert kwargs["cluster_name"] == "synth-setter-smoke-ci-smoke"
        assert kwargs["down"] is True
