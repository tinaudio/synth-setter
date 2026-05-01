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
from pipeline.schemas.config import dataset_config_id_from_path
from pipeline.schemas.spec import DatasetPipelineSpec

FIXED_NOW = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)


def _make_config(plugin: Path) -> dict[str, Any]:
    """Return a fresh DatasetConfig dict pointed at the given fake plugin path."""
    return {
        "param_spec": "surge_simple",
        "plugin_path": str(plugin),
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
    path = tmp_path / "ci-smoke-test.yaml"
    path.write_text(yaml.safe_dump(_make_config(fake_plugin), sort_keys=False))
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
    """Redirect the launcher's default spec-write directory under tmp_path."""
    spec_dir = tmp_path / "spec-out"
    spec_dir.mkdir()
    monkeypatch.setattr("pipeline.entrypoints.skypilot_launch_smoke.LOCAL_SPEC_DIR", spec_dir)
    return spec_dir


def _succeeded_run(mock_sky: MagicMock) -> None:
    """Configure `mock_sky` so the polled job reaches SUCCEEDED on first poll."""
    import sky  # noqa: PLC0415 — real enum so JobStatus.is_terminal() works

    mock_sky.JobStatus = sky.JobStatus
    mock_sky.launch.return_value = "launch-req"
    mock_sky.job_status.return_value = "job-status-req"
    mock_sky.queue.return_value = "queue-req"
    mock_sky.down.return_value = "down-req"

    responses = {
        "launch-req": (1, MagicMock()),
        "job-status-req": {1: sky.JobStatus.SUCCEEDED},
        "queue-req": [],
        "down-req": None,
    }
    mock_sky.stream_and_get.side_effect = lambda req: responses[req]
    mock_sky.tail_logs.return_value = 0


def _failed_run(mock_sky: MagicMock, status: Any) -> None:
    """Configure `mock_sky` so the polled job reaches the given terminal `status`."""
    _succeeded_run(mock_sky)
    responses = {
        "launch-req": (1, MagicMock()),
        "job-status-req": {1: status},
        "queue-req": [],
        "down-req": None,
    }
    mock_sky.stream_and_get.side_effect = lambda req: responses[req]


@pytest.fixture()
def mock_sky(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the launcher's module-level `sky` with a MagicMock pre-configured for success.

    Tests that need a different behavior call `_failed_run(...)` (or override individual
    knobs like `mock_sky.tail_logs.side_effect`) on top of this fixture.
    """
    fake = MagicMock()
    monkeypatch.setattr("pipeline.entrypoints.skypilot_launch_smoke.sky", fake)
    _succeeded_run(fake)
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


def _invoke(
    config_yaml: Path,
    template_yaml: Path,
    env_file: Path,
    *extra: str,
) -> Any:
    """Invoke the launcher CLI with the standard required options + any `extra` args."""
    runner = CliRunner()
    return runner.invoke(
        main,
        [
            "--config",
            str(config_yaml),
            "--template",
            str(template_yaml),
            "--env-file",
            str(env_file),
            *extra,
        ],
    )


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
        result = _invoke(config_yaml, template_yaml, missing)
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
        result = _invoke(config_yaml, template_yaml, empty_env)
        assert result.exit_code != 0
        assert "No env vars parsed" in result.output
        mock_sky.Task.from_yaml.assert_not_called()
        mock_sky.launch.assert_not_called()

    # --- Happy-path slices (split from the original monolithic test) ---------

    def test_materialized_spec_round_trips_as_pipeline_spec(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """The on-disk spec validates as DatasetPipelineSpec with the patched git/now values."""
        result = _invoke(config_yaml, template_yaml, env_file, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output

        spec_files = list(local_spec_dir.glob("*.json"))
        assert len(spec_files) == 1
        spec = DatasetPipelineSpec.model_validate_json(spec_files[0].read_text())
        assert spec.code_version == "abc123def456"
        assert spec.is_repo_dirty is False
        assert spec.num_shards == 1
        assert spec.r2_bucket == "intermediate-data"

    def test_worker_env_is_forwarded_to_task(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`task.update_envs` receives the parsed dotenv values."""
        result = _invoke(config_yaml, template_yaml, env_file, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output

        task = mock_sky.Task.from_yaml.return_value
        mock_sky.Task.from_yaml.assert_called_once_with(str(template_yaml))
        task.update_envs.assert_called_once()
        forwarded = task.update_envs.call_args.args[0]
        assert forwarded["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "key"
        assert forwarded["RCLONE_CONFIG_R2_ENDPOINT"].startswith("https://")

    def test_spec_is_mounted_via_sibling_copy_with_matching_content(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """The mount source is a sibling copy whose content matches the materialized spec."""
        captured: dict[str, str] = {}

        def _capture(mounts: dict[str, str]) -> None:
            for k, v in mounts.items():
                captured[k] = Path(v).read_text()

        task = mock_sky.Task.from_yaml.return_value
        task.update_file_mounts.side_effect = _capture

        result = _invoke(config_yaml, template_yaml, env_file, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output

        task.update_file_mounts.assert_called_once()
        mounts = task.update_file_mounts.call_args.args[0]
        assert WORKER_SPEC_PATH in mounts

        spec_files = list(local_spec_dir.glob("*.json"))
        assert len(spec_files) == 1
        spec_path = spec_files[0]
        mount_source = Path(mounts[WORKER_SPEC_PATH])
        assert mount_source != spec_path
        assert captured[WORKER_SPEC_PATH] == spec_path.read_text()

    def test_mount_source_sibling_is_cleaned_up_after_run(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """The `.mount.json` sibling is unlinked in finally so it doesn't survive the run."""
        captured_mount_source: dict[str, Path] = {}

        def _capture(mounts: dict[str, str]) -> None:
            captured_mount_source["path"] = Path(mounts[WORKER_SPEC_PATH])

        task = mock_sky.Task.from_yaml.return_value
        task.update_file_mounts.side_effect = _capture

        result = _invoke(config_yaml, template_yaml, env_file, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output
        assert not captured_mount_source["path"].exists()

    def test_launch_uses_autostop_zero_and_down_true(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`sky.launch` is called with the autostop+down combo the launcher relies on."""
        result = _invoke(config_yaml, template_yaml, env_file, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output

        mock_sky.launch.assert_called_once()
        kwargs = mock_sky.launch.call_args.kwargs
        assert kwargs["cluster_name"] == "smoke-job-1"
        assert kwargs["idle_minutes_to_autostop"] == 0
        assert kwargs["down"] is True

    def test_tail_logs_invoked_with_follow_false(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`sky.tail_logs` is called once with follow=False (buffered post-completion dump)."""
        result = _invoke(config_yaml, template_yaml, env_file, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output
        mock_sky.tail_logs.assert_called_once_with(
            cluster_name="smoke-job-1", job_id=1, follow=False
        )

    def test_teardown_runs_on_success(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`sky.down` is called once on the success path."""
        result = _invoke(config_yaml, template_yaml, env_file, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output
        mock_sky.down.assert_called_once_with("smoke-job-1")

    # --- Cluster-name / spec-out / failure paths -----------------------------

    def test_default_cluster_name_uses_config_id_prefix(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Without --cluster-name the launcher derives the name from `config_id[:8]`."""
        result = _invoke(config_yaml, template_yaml, env_file)
        assert result.exit_code == 0, result.output

        config_id = dataset_config_id_from_path(config_yaml)
        kwargs: dict[str, Any] = mock_sky.launch.call_args.kwargs
        assert kwargs["cluster_name"].startswith("synth-setter-smoke-")
        assert kwargs["cluster_name"].endswith(config_id[:8])
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
        result = _invoke(config_yaml, template_yaml, env_file, "--spec-out", str(explicit))
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
        """A worker job in a non-SUCCEEDED terminal status must fail the launcher."""
        import sky  # noqa: PLC0415

        _failed_run(mock_sky, sky.JobStatus.FAILED)

        result = _invoke(config_yaml, template_yaml, env_file)
        assert result.exit_code != 0
        assert "ended with status FAILED" in result.output
        mock_sky.down.assert_called_once()

    # --- Edge cases ----------------------------------------------------------

    def test_deadline_timeout_raises(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A job that never reaches a terminal status fails the launcher with a deadline error."""
        import sky  # noqa: PLC0415

        # Override only job_status to keep the polled status non-terminal forever.
        responses = {
            "launch-req": (1, MagicMock()),
            "job-status-req": {1: sky.JobStatus.RUNNING},
            "queue-req": [],
            "down-req": None,
        }
        mock_sky.stream_and_get.side_effect = lambda req: responses[req]

        # Zero sleep between polls so the deadline-bounded loop completes synchronously.
        monkeypatch.setattr(
            "pipeline.entrypoints.skypilot_launch_smoke._JOB_POLL_INTERVAL_SECONDS", 0
        )

        result = _invoke(config_yaml, template_yaml, env_file, "--job-deadline-seconds", "0")
        assert result.exit_code != 0
        assert "did not reach a terminal status" in result.output
        mock_sky.down.assert_called_once()

    def test_launch_returning_none_job_id_aborts(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """If sky.launch yields no job_id the launcher aborts before polling/teardown."""
        responses = {
            "launch-req": (None, MagicMock()),
        }
        mock_sky.stream_and_get.side_effect = lambda req: responses[req]

        result = _invoke(config_yaml, template_yaml, env_file)
        assert result.exit_code != 0
        assert "no job_id" in result.output.lower()

    def test_teardown_runs_when_tail_logs_raises(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """An exception out of `sky.tail_logs` must not skip cluster teardown."""
        mock_sky.tail_logs.side_effect = RuntimeError("boom")

        result = _invoke(config_yaml, template_yaml, env_file)
        assert result.exit_code != 0
        mock_sky.down.assert_called_once()

    def test_mount_source_cleaned_up_on_launch_exception(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """If sky.launch raises, the staged `.mount.json` sibling is still cleaned up."""
        mock_sky.launch.side_effect = RuntimeError("boom")

        result = _invoke(config_yaml, template_yaml, env_file)
        assert result.exit_code != 0

        spec_files = list(local_spec_dir.glob("*.json"))
        assert len(spec_files) == 1
        local_spec_path = spec_files[0]
        mount_sibling = local_spec_path.with_suffix(".mount.json")
        assert not mount_sibling.exists()
