"""Tests for pipeline/entrypoints/skypilot_launch_smoke.py — SkyPilot RunPod smoke launcher.

Mock-based: no real SkyPilot or RunPod calls. The `mock_sky` fixture replaces the launcher's
module-level `sky` reference with a MagicMock, and `local_spec_dir` redirects the on-disk spec
write under tmp_path so tests don't write into the real /tmp.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

from pipeline.entrypoints.skypilot_launch_smoke import (
    WORKER_SPEC_PATH,
    load_worker_env,
    main,
)
from pipeline.schemas.spec import DatasetPipelineSpec

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
    """Write a minimal valid .env with the rclone-R2 keys the launcher forwards."""
    path = tmp_path / ".env"
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
        Path(__file__).resolve().parent.parent.parent.parent
        / "configs"
        / "compute"
        / "runpod-template.yaml"
    )


@pytest.fixture()
def local_spec_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the launcher's default spec-write directory under tmp_path.

    With LOCAL_SPEC_DIR repointed at a tmp dir, the launcher's per-cluster default path
    (`<LOCAL_SPEC_DIR>/skypilot-launch-smoke-<cluster>.json`) lands inside tmp_path too —
    no pollution of the real /tmp.
    """
    spec_dir = tmp_path / "spec-out"
    spec_dir.mkdir()
    monkeypatch.setattr("pipeline.entrypoints.skypilot_launch_smoke.LOCAL_SPEC_DIR", spec_dir)
    return spec_dir


@pytest.fixture()
def mock_sky(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the launcher's module-level `sky` reference with a MagicMock.

    Happy-path side_effect chain on stream_and_get (one entry per call):
    1. launch RequestId               -> `(job_id=1, handle)`
    2. _wait_for_job job_status poll  -> `{1: SUCCEEDED}` (terminal, breaks poll loop)
    3. pre-teardown job_status diag   -> `{1: SUCCEEDED}` (paper trail before sky.down)
    4. pre-teardown queue diag        -> `[]` (paper trail; ignored on failure)
    5. down RequestId                 -> `None`

    `sky.tail_logs` is the buffered (follow=False) dump after job completion.
    """
    import sky  # noqa: PLC0415 — real enum so JobStatus.is_terminal() works under MagicMock

    fake = MagicMock()
    fake.JobStatus = sky.JobStatus
    fake.stream_and_get.side_effect = [
        (1, MagicMock()),
        {1: sky.JobStatus.SUCCEEDED},
        {1: sky.JobStatus.SUCCEEDED},
        [],
        None,
    ]
    fake.tail_logs.return_value = 0
    monkeypatch.setattr("pipeline.entrypoints.skypilot_launch_smoke.sky", fake)
    return fake


# ---------------------------------------------------------------------------
# load_worker_env
# ---------------------------------------------------------------------------


class TestLoadWorkerEnv:
    """Behavioral contracts for the worker-env loader (thin wrapper over python-dotenv)."""

    def test_parses_keys_skips_comments_and_strips_quotes(self, tmp_path: Path) -> None:
        """Python-dotenv handles blanks, comments, and quoted values; loader returns dict[str,
        str]."""
        path = tmp_path / ".env"
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
        path = tmp_path / ".env"
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
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Missing .env aborts with a clear error and never calls sky.*."""
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
        local_spec_dir: Path,
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
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """End-to-end: spec is materialized under LOCAL_SPEC_DIR and sky.* is called with expected args."""
        # Capture mount-source content at call time. The launcher unlinks the .mount.json
        # sibling in a finally after sky.* returns, so we can't read it post-invoke.
        captured: dict[str, str] = {}

        def _capture(mounts: dict[str, str]) -> None:
            for k, v in mounts.items():
                captured[k] = Path(v).read_text()

        task = mock_sky.Task.from_yaml.return_value
        task.update_file_mounts.side_effect = _capture

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
                "--cluster-name",
                "smoke-job-1",
            ],
        )
        assert result.exit_code == 0, result.output

        mock_sky.Task.from_yaml.assert_called_once_with(str(template_yaml))

        task.update_envs.assert_called_once()
        forwarded_envs = task.update_envs.call_args.args[0]
        assert forwarded_envs["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "key"
        assert forwarded_envs["RCLONE_CONFIG_R2_ENDPOINT"].startswith("https://")

        task.update_file_mounts.assert_called_once()
        mounts = task.update_file_mounts.call_args.args[0]
        assert WORKER_SPEC_PATH in mounts

        # Default spec path embeds the resolved cluster name to avoid parallel-run collisions.
        spec_path = local_spec_dir / "skypilot-launch-smoke-smoke-job-1.json"
        assert spec_path.is_file()

        # Mount source is a sibling copy (so SkyPilot's staging-by-rename doesn't consume
        # the original file the CI artifact upload depends on). Content matches the spec at
        # call time, and the sibling is cleaned up in finally so it doesn't survive the run.
        mount_source = Path(mounts[WORKER_SPEC_PATH])
        assert mount_source != spec_path
        assert captured[WORKER_SPEC_PATH] == spec_path.read_text()
        assert not mount_source.exists()

        # Round-trip: the materialized JSON validates as a DatasetPipelineSpec.
        spec = DatasetPipelineSpec.model_validate_json(spec_path.read_text())
        assert spec.code_version == "abc123def456"
        assert spec.is_repo_dirty is False
        assert spec.num_shards == 1
        assert spec.r2_bucket == "intermediate-data"

        # idle_minutes_to_autostop=0 + down=True — predictable 1-min autodown timer
        # (sky internally bumps idle=0 to 1 minute). Explicit sky.down in finally as
        # belt-and-suspenders.
        mock_sky.launch.assert_called_once_with(
            task, cluster_name="smoke-job-1", idle_minutes_to_autostop=0, down=True
        )
        # stream_and_get is called five times: launch -> (job_id, handle);
        # _wait_for_job job_status poll -> {1: SUCCEEDED}; pre-teardown job_status diag;
        # pre-teardown queue diag; down -> None.
        assert mock_sky.stream_and_get.call_count == 5
        # sky.job_status is the polling target (called by both _wait_for_job and the
        # pre-teardown diagnostic).
        mock_sky.job_status.assert_called_with("smoke-job-1", [1])
        # sky.queue is queried as a pre-teardown diagnostic.
        mock_sky.queue.assert_called_with("smoke-job-1")

        # tail_logs is called with follow=False (buffered dump after the queue poll
        # detected SUCCEEDED), so we never block on the SSH stream that hangs on RunPod.
        mock_sky.tail_logs.assert_called_once_with(
            cluster_name="smoke-job-1", job_id=1, follow=False
        )

        # Explicit teardown — must always run, even on success.
        mock_sky.down.assert_called_once_with("smoke-job-1")

    def test_default_cluster_name_uses_config_id_prefix(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """When --cluster-name is omitted, the launcher derives the name from `config_id[:8]`."""
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
        assert kwargs["idle_minutes_to_autostop"] == 0
        assert kwargs["down"] is True

    def test_spec_out_overrides_default_path(
        self,
        tmp_path: Path,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """--spec-out forces an explicit local path (used by CI to find the spec for upload)."""
        explicit = tmp_path / "explicit-out" / "input_spec.json"
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
                "--spec-out",
                str(explicit),
            ],
        )
        assert result.exit_code == 0, result.output
        assert explicit.is_file()

    def test_worker_failed_status_fails_launcher(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """A worker job in a non-SUCCEEDED terminal status must fail the launcher.

        Without this, a FAILED worker would let the launcher complete and downstream "shard not in
        R2" errors would mask the actual worker traceback.
        """
        import sky  # noqa: PLC0415

        mock_sky.stream_and_get.side_effect = [
            (1, MagicMock()),  # launch RequestId -> (job_id, handle)
            {1: sky.JobStatus.FAILED},  # _wait_for_job poll -> terminal FAILED
            {1: sky.JobStatus.FAILED},  # pre-teardown job_status diagnostic
            [],  # pre-teardown queue diagnostic
            None,  # down RequestId
        ]
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
        assert result.exit_code != 0
        assert "ended with status FAILED" in result.output
        # Teardown still runs even on worker failure.
        mock_sky.down.assert_called_once()
