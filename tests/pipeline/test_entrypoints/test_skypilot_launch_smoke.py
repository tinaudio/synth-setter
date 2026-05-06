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
    _WORKER_ENV_KEYS,
    _WORKER_SPEC_URI_ENV,
    _override_image_id,
    load_worker_env,
    main,
    resolve_worker_env,
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
    path = tmp_path / "runpod-smoke-shard.yaml"
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


@pytest.fixture(autouse=True)
def clear_worker_env_from_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the rclone-R2 / WANDB keys from the test process env.

    Without this, a developer who has `RCLONE_CONFIG_R2_*` exported in their shell
    (or a CI runner that inherits them globally) would silently satisfy
    `resolve_worker_env`, masking tests that rely on a specific resolution path.
    """
    for key in _WORKER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def mock_rclone_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Globally no-op the rclone subprocess that `upload_spec_to_r2` would invoke.

    Tests that explicitly want to assert on the rclone command shape override this
    by setting their own side_effect on `subprocess.check_call`.
    """
    monkeypatch.setattr(
        "pipeline.entrypoints.skypilot_launch_smoke.subprocess.check_call",
        lambda args: None,
    )


def _succeeded_run(mock_sky: MagicMock) -> None:
    """Configure `mock_sky` so launch + tail_logs + down all succeed.

    `sky.tail_logs` returns an int rc directly (per sky/core.py:1232) — 0 means
    the worker job ended in SUCCEEDED, anything else means it ended in a
    non-SUCCEEDED terminal status.
    """
    mock_sky.launch.return_value = "launch-req"
    mock_sky.down.return_value = "down-req"

    responses = {
        "launch-req": (1, MagicMock()),
        "down-req": None,
    }
    mock_sky.stream_and_get.side_effect = lambda req: responses[req]
    mock_sky.tail_logs.return_value = 0


@pytest.fixture()
def mock_sky(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the launcher's module-level `sky` with a MagicMock pre-configured for success.

    Tests that need a different behavior tweak knobs on the returned mock (e.g. set
    `mock_sky.tail_logs.return_value = 100` for a worker failure, or
    `mock_sky.tail_logs.side_effect = ...` for a transport raise).
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


class TestResolveWorkerEnvGitRefValidation:
    """`WORKER_GIT_REF`, when set, must be a 7-40 char hex git SHA.

    The validation lives at the env-resolution seam (host-side) instead of in the worker template's
    bash because the SHA is rendered into a `git fetch + checkout` invocation; rejecting a
    malformed value at the launcher gives a clear error before the cluster is ever provisioned.
    """

    def test_unset_git_ref_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty/unset WORKER_GIT_REF is the common case (no PR-CI bake-lag bypass)."""
        monkeypatch.delenv("WORKER_GIT_REF", raising=False)
        resolved = resolve_worker_env(None)
        assert "WORKER_GIT_REF" not in resolved

    @pytest.mark.parametrize(
        "good_sha",
        ["abc1234", "abc1234deadbeef", "0" * 40, "f" * 40],
    )
    def test_valid_hex_sha_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, good_sha: str
    ) -> None:
        """7-40 char lowercase hex strings pass — both short and long form."""
        monkeypatch.setenv("WORKER_GIT_REF", good_sha)
        resolved = resolve_worker_env(None)
        assert resolved["WORKER_GIT_REF"] == good_sha

    @pytest.mark.parametrize(
        "bad_sha",
        [
            "main",  # branch name, not SHA
            "ABC1234",  # uppercase rejected
            "abc",  # too short
            "g" * 7,  # non-hex char
            "abc1234; rm -rf /",  # injection attempt
            "abc 1234",  # whitespace
        ],
    )
    def test_invalid_git_ref_raises(self, monkeypatch: pytest.MonkeyPatch, bad_sha: str) -> None:
        """Non-SHA values raise ValueError before the launcher provisions anything."""
        monkeypatch.setenv("WORKER_GIT_REF", bad_sha)
        with pytest.raises(ValueError, match="WORKER_GIT_REF"):
            resolve_worker_env(None)


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

    def test_no_env_anywhere_fails_with_clear_error(
        self,
        tmp_path: Path,
        config_yaml: Path,
        template_yaml: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """With no .env on disk and no rclone-R2 keys in process env, the launcher fails fast and
        never calls sky.*."""
        missing = tmp_path / "does-not-exist.env"
        result = _invoke(config_yaml, template_yaml, missing)
        assert result.exit_code != 0
        assert "No worker env vars resolved" in result.output
        mock_sky.Task.from_yaml.assert_not_called()
        mock_sky.launch.assert_not_called()

    def test_empty_env_file_with_no_process_env_fails(
        self,
        tmp_path: Path,
        config_yaml: Path,
        template_yaml: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """An empty .env on disk and no rclone-R2 keys in process env fails fast — empty .env is
        equivalent to no .env."""
        empty_env = tmp_path / "empty.env"
        empty_env.write_text("# only comments\n\n")
        result = _invoke(config_yaml, template_yaml, empty_env)
        assert result.exit_code != 0
        assert "No worker env vars resolved" in result.output
        mock_sky.Task.from_yaml.assert_not_called()
        mock_sky.launch.assert_not_called()

    def test_process_env_resolves_when_env_file_absent(
        self,
        tmp_path: Path,
        config_yaml: Path,
        template_yaml: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the .env doesn't exist but the rclone-R2 keys are in process env, the launcher
        succeeds and forwards the process-env values via task.update_envs."""
        missing = tmp_path / "does-not-exist.env"
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "s3")
        monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "process-env-key")
        result = _invoke(config_yaml, template_yaml, missing, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output

        task = mock_sky.Task.from_yaml.return_value
        task.update_envs.assert_called_once()
        forwarded = task.update_envs.call_args.args[0]
        assert forwarded["RCLONE_CONFIG_R2_TYPE"] == "s3"
        assert forwarded["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "process-env-key"

    # --- Happy-path slices ---------------------------------------------------

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

    def test_spec_uri_forwarded_to_worker_env_after_r2_upload(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`upload_spec_to_r2` puts the spec at `r2://<bucket>/skypilot-launcher-
        specs/<cluster>.json` and the launcher injects that URI into the worker's env via
        WORKER_SPEC_URI.

        The launcher
        does NOT call `task.update_file_mounts(...)` (#749 workaround).
        """
        rclone_invocations: list[list[str]] = []
        monkeypatch.setattr(
            "pipeline.entrypoints.skypilot_launch_smoke.subprocess.check_call",
            lambda args: rclone_invocations.append(args),
        )

        result = _invoke(config_yaml, template_yaml, env_file, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output

        task = mock_sky.Task.from_yaml.return_value
        task.update_envs.assert_called_once()
        forwarded = task.update_envs.call_args.args[0]
        assert (
            forwarded[_WORKER_SPEC_URI_ENV]
            == "r2://intermediate-data/skypilot-launcher-specs/smoke-job-1.json"
        )

        assert len(rclone_invocations) == 1
        cmd = rclone_invocations[0]
        assert cmd[:2] == ["rclone", "copyto"]
        assert cmd[-1] == "r2:intermediate-data/skypilot-launcher-specs/smoke-job-1.json"

        task.update_file_mounts.assert_not_called()

    def test_launch_uses_autostop_window_and_down_true(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`sky.launch` keeps `down=True` for cleanup and an autostop window as a backstop in case
        the explicit `sky.down` in `finally` is skipped by an unexpected exit path."""
        result = _invoke(config_yaml, template_yaml, env_file, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output

        mock_sky.launch.assert_called_once()
        kwargs = mock_sky.launch.call_args.kwargs
        assert kwargs["cluster_name"] == "smoke-job-1"
        assert kwargs["idle_minutes_to_autostop"] >= 2
        assert kwargs["down"] is True

    def test_tail_logs_invoked_with_follow_true(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`sky.tail_logs` is called with follow=True — the launcher streams logs in real time
        instead of polling job_status and dumping a buffered tail."""
        result = _invoke(config_yaml, template_yaml, env_file, "--cluster-name", "smoke-job-1")
        assert result.exit_code == 0, result.output
        mock_sky.tail_logs.assert_called_once_with(
            cluster_name="smoke-job-1", job_id=1, follow=True
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
        assert kwargs["idle_minutes_to_autostop"] >= 2
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

    def test_worker_failed_rc_fails_launcher(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """A non-zero `tail_logs` rc means the worker job ended in a non-SUCCEEDED terminal status;
        the launcher must surface that as a non-zero exit and still tear down the cluster."""
        mock_sky.tail_logs.return_value = 100

        result = _invoke(config_yaml, template_yaml, env_file)
        assert result.exit_code != 0
        # Aggregate fan-out failure message names every failed rank with its rc.
        assert "rc=100" in result.output
        mock_sky.down.assert_called_once()

    # --- Edge cases ----------------------------------------------------------

    def test_launch_returning_none_job_id_aborts(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """If sky.launch yields no job_id the launcher aborts; teardown is best-effort and
        idempotent on a never-provisioned cluster name (sky.down on a missing cluster is a no-op),
        so it still runs in the finally block to make multi-worker partial-failure cleanup
        uniform."""
        responses = {
            "launch-req": (None, MagicMock()),
        }
        mock_sky.stream_and_get.side_effect = lambda req: responses[req]

        result = _invoke(config_yaml, template_yaml, env_file)
        assert result.exit_code != 0
        assert "no job_id" in result.output.lower()
        mock_sky.tail_logs.assert_not_called()

    def test_teardown_runs_when_tail_logs_raises(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """A `sky.tail_logs` transport error must not skip cluster teardown — the cluster always
        comes down even if the log-stream side raised."""
        mock_sky.tail_logs.side_effect = RuntimeError("boom")

        result = _invoke(config_yaml, template_yaml, env_file)
        assert result.exit_code != 0
        mock_sky.down.assert_called_once()

    def test_local_spec_persists_for_artifact_upload_even_on_launch_exception(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If sky.launch raises after the launcher materialized + R2-uploaded the spec, the local
        spec file under LOCAL_SPEC_DIR is still around for downstream artifact upload."""
        monkeypatch.setattr(
            "pipeline.entrypoints.skypilot_launch_smoke.subprocess.check_call",
            lambda args: None,
        )
        mock_sky.launch.side_effect = RuntimeError("boom")

        result = _invoke(config_yaml, template_yaml, env_file)
        assert result.exit_code != 0

        spec_files = list(local_spec_dir.glob("*.json"))
        assert len(spec_files) == 1
        assert spec_files[0].read_text(), (
            "local spec file should still be on disk for artifact upload"
        )


class TestNumWorkersFanOut:
    """`--num-workers N>1` fans out N independent single-node SkyPilot clusters.

    RunPod's backend doesn't support num_nodes>1, so the launcher synthesizes multi-worker
    partitioning by launching N clusters in parallel and injecting SYNTH_SETTER_WORKER_RANK /
    SYNTH_SETTER_NUM_WORKERS per cluster. Each cluster downloads the same materialized spec;
    pipeline.partitioning.get_my_shards slices each worker's shard ownership.
    """

    @staticmethod
    def _setup_n_workers_mock(
        mock_sky: MagicMock,
        n: int,
        *,
        tail_rcs_by_cluster: dict[str, int] | None = None,
        base_cluster_name: str = "smoke-job-1",
    ) -> dict[str, MagicMock]:
        """Configure `mock_sky` for an N-cluster run with deterministic per-cluster routing.

        Returns a ``cluster_name -> Task`` dict keyed by the cluster name the launcher will
        request, so tests can inspect each rank's ``update_envs`` call without depending on
        ThreadPoolExecutor scheduling order. ``tail_logs`` and ``launch`` route by their
        ``cluster_name`` kwarg (not by call order), so an rc=100 for ``smoke-job-1-r1``
        deterministically attaches to rank 1 regardless of which thread ran first.
        """
        rcs = tail_rcs_by_cluster if tail_rcs_by_cluster is not None else {}
        cluster_names = [f"{base_cluster_name}-r{i}" for i in range(n)]
        tasks = {name: MagicMock(name=f"task-{name}") for name in cluster_names}
        # `cluster_name` isn't a Task.from_yaml arg, so route Tasks by call order. The
        # launcher creates exactly one Task per rank inside _launch_and_tail and immediately
        # update_envs's it with the cluster name in the message — assertions key off the
        # cluster_name in the env, not the Task identity, so call order doesn't matter here.
        mock_sky.Task.from_yaml.side_effect = list(tasks.values())

        # Route launch + down + stream_and_get + tail_logs by cluster_name kwarg, not order.
        launch_reqs = {name: f"launch-{name}" for name in cluster_names}
        down_reqs = {name: f"down-{name}" for name in cluster_names}
        mock_sky.launch.side_effect = lambda task, **kw: launch_reqs[kw["cluster_name"]]
        mock_sky.down.side_effect = lambda name: down_reqs[name]

        stream_responses: dict[str, object] = {
            launch_reqs[name]: (i + 1, MagicMock()) for i, name in enumerate(cluster_names)
        }
        stream_responses.update({req: None for req in down_reqs.values()})
        mock_sky.stream_and_get.side_effect = lambda req: stream_responses[req]

        mock_sky.tail_logs.side_effect = lambda **kw: rcs.get(kw["cluster_name"], 0)
        return tasks

    def test_three_workers_launches_three_clusters(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`--num-workers 3` provisions exactly 3 clusters."""
        self._setup_n_workers_mock(mock_sky, n=3)

        result = _invoke(
            config_yaml,
            template_yaml,
            env_file,
            "--cluster-name",
            "smoke-job-1",
            "--num-workers",
            "3",
        )

        assert result.exit_code == 0, result.output
        assert mock_sky.launch.call_count == 3
        assert mock_sky.tail_logs.call_count == 3
        assert mock_sky.down.call_count == 3

    def test_three_workers_use_rank_suffixed_cluster_names(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """N>1 launches use `<base>-r{i}` cluster names so the per-rank pods are distinguishable in
        SkyPilot's UI / dashboards."""
        self._setup_n_workers_mock(mock_sky, n=3)

        result = _invoke(
            config_yaml,
            template_yaml,
            env_file,
            "--cluster-name",
            "smoke-job-1",
            "--num-workers",
            "3",
        )

        assert result.exit_code == 0, result.output
        launch_cluster_names = sorted(
            call.kwargs["cluster_name"] for call in mock_sky.launch.call_args_list
        )
        assert launch_cluster_names == ["smoke-job-1-r0", "smoke-job-1-r1", "smoke-job-1-r2"]

    def test_one_worker_keeps_unsuffixed_cluster_name(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`--num-workers 1` (default) keeps the unsuffixed cluster name for backward-compat with
        debug workflows / dashboards that key off it."""
        result = _invoke(
            config_yaml,
            template_yaml,
            env_file,
            "--cluster-name",
            "smoke-job-1",
        )

        assert result.exit_code == 0, result.output
        mock_sky.launch.assert_called_once()
        assert mock_sky.launch.call_args.kwargs["cluster_name"] == "smoke-job-1"

    def test_three_workers_inject_distinct_rank_world_per_cluster(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Each rank's task gets ``SYNTH_SETTER_WORKER_RANK=<i>`` and
        ``SYNTH_SETTER_NUM_WORKERS=<N>`` injected.

        Workers read these via ``read_rank_world_from_env`` and partition the
        shared spec via ``get_my_shards``.
        """
        tasks = self._setup_n_workers_mock(mock_sky, n=3)

        result = _invoke(
            config_yaml,
            template_yaml,
            env_file,
            "--cluster-name",
            "smoke-job-1",
            "--num-workers",
            "3",
        )

        assert result.exit_code == 0, result.output
        forwarded = [t.update_envs.call_args.args[0] for t in tasks.values()]
        ranks = sorted(env["SYNTH_SETTER_WORKER_RANK"] for env in forwarded)
        assert ranks == ["0", "1", "2"]
        for env in forwarded:
            assert env["SYNTH_SETTER_NUM_WORKERS"] == "3"
            assert env["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "key"
            assert env["WORKER_SPEC_URI"] == (
                "r2://intermediate-data/skypilot-launcher-specs/smoke-job-1.json"
            )

    def test_three_workers_upload_spec_only_once(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Spec is materialized + uploaded to R2 once and shared across all ranks (single
        ``r2_prefix`` so the partition is one logical dataset, not three)."""
        self._setup_n_workers_mock(mock_sky, n=3)
        rclone_invocations: list[list[str]] = []
        monkeypatch.setattr(
            "pipeline.entrypoints.skypilot_launch_smoke.subprocess.check_call",
            lambda args: rclone_invocations.append(args),
        )

        result = _invoke(
            config_yaml,
            template_yaml,
            env_file,
            "--cluster-name",
            "smoke-job-1",
            "--num-workers",
            "3",
        )

        assert result.exit_code == 0, result.output
        assert len(rclone_invocations) == 1
        assert rclone_invocations[0][-1] == (
            "r2:intermediate-data/skypilot-launcher-specs/smoke-job-1.json"
        )

    def test_one_worker_failure_among_three_fails_launcher_after_full_teardown(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """If any rank's tail_logs returns non-zero, the launcher exits non-zero, but every cluster
        (success or fail) gets torn down — partial-failure cleanup must be uniform."""
        self._setup_n_workers_mock(mock_sky, n=3, tail_rcs_by_cluster={"smoke-job-1-r1": 100})

        result = _invoke(
            config_yaml,
            template_yaml,
            env_file,
            "--cluster-name",
            "smoke-job-1",
            "--num-workers",
            "3",
        )

        assert result.exit_code != 0
        assert "rc=100" in result.output
        assert "smoke-job-1-r1" in result.output
        assert mock_sky.down.call_count == 3

    def test_worker_git_ref_forwarded_to_every_rank(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`WORKER_GIT_REF` from process env reaches every rank's task via update_envs.

        The pod's `run:` block reads this and `git fetch+checkout`s before invoking
        generate_dataset, so the worker runs the dispatcher's source instead of the baked image's
        stale checkout.
        """
        tasks = self._setup_n_workers_mock(mock_sky, n=3)
        monkeypatch.setenv("WORKER_GIT_REF", "abc1234deadbeef")

        result = _invoke(
            config_yaml,
            template_yaml,
            env_file,
            "--cluster-name",
            "smoke-job-1",
            "--num-workers",
            "3",
        )

        assert result.exit_code == 0, result.output
        forwarded = [t.update_envs.call_args.args[0] for t in tasks.values()]
        for env in forwarded:
            assert env["WORKER_GIT_REF"] == "abc1234deadbeef"

    def test_zero_or_negative_num_workers_rejected(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`--num-workers 0` (or negative) is a CLI usage error — never reach sky.*."""
        result = _invoke(
            config_yaml,
            template_yaml,
            env_file,
            "--cluster-name",
            "smoke-job-1",
            "--num-workers",
            "0",
        )
        assert result.exit_code != 0
        assert "must be >= 1" in result.output
        mock_sky.launch.assert_not_called()

    @pytest.mark.parametrize(
        "bad_tag",
        [
            "foo:bar",
            "foo bar",
            "foo/bar",
            ".dotleader",
            "-dashleader",
            "",
        ],
    )
    def test_invalid_worker_image_tag_rejected(
        self,
        config_yaml: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        bad_tag: str,
    ) -> None:
        """`--worker-image-tag` is interpolated into a docker ref; invalid tags must fail before
        sky.* is touched, not produce surprising image refs like `tinaudio/synth-
        setter:foo:bar`."""
        result = _invoke(
            config_yaml,
            template_yaml,
            env_file,
            "--cluster-name",
            "smoke-job-1",
            "--worker-image-tag",
            bad_tag,
        )
        assert result.exit_code != 0
        assert "--worker-image-tag" in result.output
        mock_sky.launch.assert_not_called()


class TestOverrideImageId:
    """Per-backend `image_id` mutation in `_override_image_id`.

    Direct unit tests on the helper, independent of the CLI path. RunPod (and any non-OCI cloud)
    accepts ``image_id: docker:<image>``; OCI's backend rejects it and runs the worker via a
    sub-docker invocation in the YAML's run: block, so OCI Resources must be left untouched.
    """

    @staticmethod
    def _make_resource(cloud: object) -> MagicMock:
        """Fake `sky.Resources` with a `.cloud` attr and a `.copy()` that records image_id."""
        res = MagicMock(spec=["cloud", "copy"])
        res.cloud = cloud

        def _copy(**kwargs: Any) -> MagicMock:
            new = MagicMock(spec=["cloud", "image_id"])
            new.cloud = cloud
            new.image_id = kwargs.get("image_id")
            return new

        res.copy.side_effect = _copy
        return res

    @staticmethod
    def _make_task(resources: list[Any]) -> MagicMock:
        """Fake `sky.Task` carrying `resources` (as a list, so `type(...)` is `list`)."""
        task = MagicMock(spec=["resources", "set_resources"])
        task.resources = list(resources)
        return task

    def test_non_oci_resource_gets_image_id_overridden(self) -> None:
        """A non-OCI Resources entry has its `image_id` rewritten to `docker:<worker_image>`."""
        runpod_cloud = MagicMock(name="RunPodCloud")
        res = self._make_resource(runpod_cloud)
        task = self._make_task([res])

        _override_image_id(task, "tinaudio/synth-setter:test-tag")

        res.copy.assert_called_once_with(image_id="docker:tinaudio/synth-setter:test-tag")
        task.set_resources.assert_called_once()
        new_resources = list(task.set_resources.call_args.args[0])
        assert len(new_resources) == 1
        assert new_resources[0].image_id == "docker:tinaudio/synth-setter:test-tag"

    def test_multiple_non_oci_resources_all_get_image_id_overridden(self) -> None:
        """Every entry in a multi-Resources alt-set (RunPod has 7) is mutated, not just the
        first."""
        runpod_cloud = MagicMock(name="RunPodCloud")
        resources = [self._make_resource(runpod_cloud) for _ in range(3)]
        task = self._make_task(resources)

        _override_image_id(task, "tinaudio/synth-setter:test-tag")

        for res in resources:
            res.copy.assert_called_once_with(image_id="docker:tinaudio/synth-setter:test-tag")
        new_resources = list(task.set_resources.call_args.args[0])
        assert len(new_resources) == 3
        assert all(r.image_id == "docker:tinaudio/synth-setter:test-tag" for r in new_resources)

    def test_oci_resource_left_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An OCI Resources entry is passed through unchanged — no `.copy(image_id=...)`, and
        `set_resources` is not called when nothing mutated."""
        import sky.clouds

        class FakeOCI:
            pass

        monkeypatch.setattr(sky.clouds, "OCI", FakeOCI, raising=False)

        oci_res = self._make_resource(FakeOCI())
        task = self._make_task([oci_res])

        _override_image_id(task, "tinaudio/synth-setter:test-tag")

        oci_res.copy.assert_not_called()
        task.set_resources.assert_not_called()
