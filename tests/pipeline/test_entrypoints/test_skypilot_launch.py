"""Tests for the SkyPilot launcher (RunPod / OCI / kind).

Covers ``src/synth_setter/pipeline/skypilot_launch.py``. Mock-based: no real SkyPilot or RunPod
calls. The `mock_sky` fixture replaces the launcher's module-level `sky` reference with a
MagicMock, and `local_spec_dir` redirects the on-disk spec write under tmp_path so tests don't
write into the real /tmp.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import click
import pytest
import yaml
from click.testing import CliRunner

from synth_setter.pipeline.schemas.compute import ComputeConfig
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.skypilot_launch import (
    _WORKER_ENV_KEYS,
    _WORKER_SPEC_URI_ENV,
    _override_image_id,
    load_worker_env,
    main,
    resolve_worker_env,
)
from synth_setter.pipeline.skypilot_launch import (
    _detect_provider as _real_detect_provider,
)
from synth_setter.pipeline.skypilot_launch import (
    _run_cred_bootstrap as _real_run_cred_bootstrap,
)

FIXED_NOW = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def fake_plugin(tmp_path: Path) -> Path:
    """Build a minimal VST3 bundle with a moduleinfo.json the renderer can read.

    Version pinned to ``1.3.4`` to match ``configs/render/surge_xt.yaml``'s
    ``renderer_version``, which the surge_simple group inherits.
    """
    contents = tmp_path / "FakePlugin.vst3" / "Contents"
    contents.mkdir(parents=True)
    (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')
    return tmp_path / "FakePlugin.vst3"


@pytest.fixture()
def patch_materialize_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out git/timestamp I/O so DatasetSpec construction is deterministic."""
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "abc123def456")
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: FIXED_NOW)


@pytest.fixture()
def experiment(fake_plugin: Path) -> str:
    """Name a Hydra experiment under ``configs/experiment/`` for the launcher's flag.

    Tests pass this verbatim to the launcher's ``--experiment`` flag; ad-hoc Hydra overrides
    (e.g. ``render.plugin_path=...``) flow through ``_invoke``'s positional trailing args.

    Uses ``generate_dataset/smoke-shard`` (12 samples, 3 shards at samples_per_shard=4)
    for fast CI.
    """
    # fake_plugin is a fixture dep so the path is built before tests use it via
    # the plugin_path override threaded through _invoke.
    _ = fake_plugin
    return "generate_dataset/smoke-shard"


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
    monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.LOCAL_SPEC_DIR", spec_dir)
    return spec_dir


@pytest.fixture(autouse=True)
def clear_worker_env_from_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the worker env keys from the test process env.

    Without this, a developer with `RCLONE_CONFIG_R2_*` exported in their shell
    would silently satisfy `resolve_worker_env`, masking tests that rely on a
    specific resolution path.
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
        "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
        lambda args: None,
    )


@pytest.fixture(autouse=True)
def mock_cred_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-op `_run_cred_bootstrap` and `_detect_provider` by default.

    Tests that exercise the bootstrap behavior directly re-patch these to inject a specific
    provider, exception, or call assertion.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.skypilot_launch._run_cred_bootstrap",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.skypilot_launch._detect_provider",
        lambda _task: "runpod",
    )


def _succeeded_run(mock_sky: MagicMock) -> None:
    """Configure `mock_sky` so jobs.launch + jobs.tail_logs + jobs.cancel all succeed.

    `sky.jobs.tail_logs` returns an int rc directly — 0 means the
    managed job ended in SUCCEEDED, anything else means it ended in a non-SUCCEEDED terminal
    status. `sky.jobs.launch` returns a request_id whose `stream_and_get` yields
    `(job_ids: List[int], handle)` — a list of length 1 for single-Task launches.
    """
    mock_sky.jobs.launch.return_value = "launch-req"
    mock_sky.jobs.cancel.return_value = "cancel-req"

    responses = {
        "launch-req": ([1], MagicMock()),
        "cancel-req": None,
    }
    mock_sky.stream_and_get.side_effect = lambda req: responses[req]
    mock_sky.jobs.tail_logs.return_value = 0


@pytest.fixture()
def mock_sky(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the launcher's module-level `sky` with a MagicMock pre-configured for success.

    Tests that need a different behavior tweak knobs on the returned mock (e.g. set
    `mock_sky.jobs.tail_logs.return_value = 100` for a worker failure, or
    `mock_sky.jobs.tail_logs.side_effect = ...` for a transport raise).
    """
    fake = MagicMock()
    monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.sky", fake)
    _succeeded_run(fake)
    return fake


# ---------------------------------------------------------------------------
# load_worker_env
# ---------------------------------------------------------------------------


class TestLoadWorkerEnv:
    """Behavioral contracts for the worker-env loader (thin wrapper over python-dotenv)."""

    def test_parses_keys_skips_comments_and_strips_quotes(self, tmp_path: Path) -> None:
        """Skip blanks/comments and strip quotes when loading the dotenv file."""
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
    malformed value at the launcher gives a clear error before the job is ever submitted.
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
        """Non-SHA values fail with ClickException before the launcher provisions anything."""
        monkeypatch.setenv("WORKER_GIT_REF", bad_sha)
        with pytest.raises(click.ClickException, match="WORKER_GIT_REF"):
            resolve_worker_env(None)


class TestResolveWorkerEnvR2RemoteConstants:
    """Cover rclone-constant defaulting for the R2 type and provider keys.

    Targets ``RCLONE_CONFIG_R2_TYPE`` and ``RCLONE_CONFIG_R2_PROVIDER``. These are constants
    (not secrets) that rclone needs to construct the `r2:` remote. The launcher defaults them
    so workflows and `.env` files don't have to repeat them, while still allowing override for
    non-Cloudflare R2-compatible setups (e.g. self-hosted MinIO test rigs).
    """

    def test_type_and_provider_default_when_unset(self) -> None:
        """Without TYPE/PROVIDER in env or .env, the launcher fills the rclone constants."""
        resolved = resolve_worker_env(None)
        assert resolved["RCLONE_CONFIG_R2_TYPE"] == "s3"
        assert resolved["RCLONE_CONFIG_R2_PROVIDER"] == "Cloudflare"

    def test_type_override_from_env_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit override via process env is preserved (not clobbered by the default)."""
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "s3-other")
        resolved = resolve_worker_env(None)
        assert resolved["RCLONE_CONFIG_R2_TYPE"] == "s3-other"

    def test_provider_override_from_env_file_wins(self, tmp_path: Path) -> None:
        """An explicit override via `.env` is preserved (not clobbered by the default)."""
        env_file = tmp_path / ".env"
        env_file.write_text("RCLONE_CONFIG_R2_PROVIDER=Other\n")
        resolved = resolve_worker_env(env_file)
        assert resolved["RCLONE_CONFIG_R2_PROVIDER"] == "Other"


# ---------------------------------------------------------------------------
# main — CLI integration with mocked sky
# ---------------------------------------------------------------------------


def _invoke(
    experiment: str,
    template_yaml: Path,
    env_file: Path,
    *extra: str,
    fake_plugin: Path | None = None,
) -> Any:
    """Invoke the launcher CLI with the standard required options + any ``extra`` args.

    Threads ``fake_plugin`` (a fixture-built VST3 bundle path) as a trailing Hydra override
    on ``render.plugin_path`` so the composed DatasetSpec points at a path that exists in
    the test environment rather than the real Surge XT install location baked into
    ``configs/render/surge_xt.yaml``.
    """
    runner = CliRunner()
    args = [
        "--experiment",
        experiment,
        "--template",
        str(template_yaml),
        "--env-file",
        str(env_file),
        *extra,
    ]
    if fake_plugin is not None:
        args.append(f"render.plugin_path={fake_plugin}")
    return runner.invoke(main, args)


class TestMainCli:
    """End-to-end CLI behavior: env validation, spec materialization, sky.* call shape."""

    def test_no_env_anywhere_fails_with_clear_error(
        self,
        tmp_path: Path,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify the launcher fails fast and never calls sky.* without rclone-R2 keys."""
        missing = tmp_path / "does-not-exist.env"
        result = _invoke(experiment, template_yaml, missing, fake_plugin=fake_plugin)
        assert result.exit_code != 0
        assert "No worker env vars resolved" in result.output
        mock_sky.Task.from_yaml_config.assert_not_called()
        mock_sky.jobs.launch.assert_not_called()

    def test_unknown_experiment_surfaces_as_click_error(
        self,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify a missing experiment fails as a click error and never launches.

        Hydra raises ``HydraException`` from ``compose`` when the named experiment file isn't
        on the config-search path; the launcher wraps that as a ``click.ClickException`` so the
        user sees a one-line CLI error.
        """
        result = _invoke(
            "this-experiment-does-not-exist",
            template_yaml,
            env_file,
            fake_plugin=fake_plugin,
        )
        assert result.exit_code != 0
        assert "Hydra compose failed for experiment 'this-experiment-does-not-exist'" in (
            result.output
        )
        mock_sky.jobs.launch.assert_not_called()

    def test_unknown_launcher_flag_is_rejected(
        self,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify Click rejects typos of ``--``-style launcher options.

        Hydra overrides are positional ``key=value`` args, not flags, so strict option validation
        catches misspellings of real launcher flags instead of silently forwarding them.
        """
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--experiment",
                "generate_dataset/smoke-shard",
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
                "--not-a-real-flag",
                "oops",
                f"render.plugin_path={fake_plugin}",
            ],
        )
        assert result.exit_code != 0
        assert "no such option" in result.output.lower()
        mock_sky.jobs.launch.assert_not_called()

    def test_empty_env_file_with_no_process_env_fails(
        self,
        tmp_path: Path,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify an empty .env with no rclone-R2 keys in process env fails fast."""
        empty_env = tmp_path / "empty.env"
        empty_env.write_text("# only comments\n\n")
        result = _invoke(experiment, template_yaml, empty_env, fake_plugin=fake_plugin)
        assert result.exit_code != 0
        assert "No worker env vars resolved" in result.output
        mock_sky.Task.from_yaml_config.assert_not_called()
        mock_sky.jobs.launch.assert_not_called()

    def test_process_env_resolves_when_env_file_absent(
        self,
        tmp_path: Path,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the launcher forwards process-env values when .env is absent."""
        missing = tmp_path / "does-not-exist.env"
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "s3")
        monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "process-env-key")
        result = _invoke(
            experiment,
            template_yaml,
            missing,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output

        task = mock_sky.Task.from_yaml_config.return_value
        task.update_envs.assert_called_once()
        forwarded = task.update_envs.call_args.args[0]
        assert forwarded["RCLONE_CONFIG_R2_TYPE"] == "s3"
        assert forwarded["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "process-env-key"

    # --- Happy-path slices ---------------------------------------------------

    def test_materialized_spec_round_trips_as_pipeline_spec(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """The on-disk spec validates as DatasetSpec with the patched git/now values."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output

        spec_files = list(local_spec_dir.glob("*.json"))
        assert len(spec_files) == 1
        spec = DatasetSpec.model_validate_json(spec_files[0].read_text())
        assert spec.git_sha == "abc123def456"
        assert spec.is_repo_dirty is False
        # ``generate_dataset/smoke-shard``: 12 samples / samples_per_shard=4 = 3 shards.
        assert spec.num_shards == 3
        assert spec.r2_bucket == "intermediate-data"

    def test_worker_env_is_forwarded_to_task(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`task.update_envs` receives the parsed dotenv values."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output

        task = mock_sky.Task.from_yaml_config.return_value
        mock_sky.Task.from_yaml_config.assert_called_once()
        # `from_yaml_config` consumes a dict (not a YAML path) — the dict must look like
        # the validated ComputeConfig for the runpod template.
        passed_config = mock_sky.Task.from_yaml_config.call_args.args[0]
        assert passed_config["resources"]["cloud"] == "runpod"
        task.update_envs.assert_called_once()
        forwarded = task.update_envs.call_args.args[0]
        assert forwarded["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "key"
        assert forwarded["RCLONE_CONFIG_R2_ENDPOINT"].startswith("https://")

    def test_spec_uri_forwarded_to_worker_env_after_r2_upload(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the spec lands at the canonical R2 path with WORKER_SPEC_URI injected.

        The spec is uploaded to ``r2://<bucket>/skypilot-launcher-specs/<cluster>.json`` and the
        launcher injects that URI into the worker's env via ``WORKER_SPEC_URI``. The launcher
        does NOT call ``task.update_file_mounts(...)`` (#749 workaround).
        """
        rclone_invocations: list[list[str]] = []
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            lambda args: rclone_invocations.append(args),
        )

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output

        task = mock_sky.Task.from_yaml_config.return_value
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

    def test_launch_submits_by_managed_job_name(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`sky.jobs.launch` is called with `name=<cluster_name>` and no cluster-launch kwargs.

        Managed jobs handle teardown via the controller's terminal-status lifecycle, so the old
        `idle_minutes_to_autostop=5, down=True` knobs aren't applicable.
        """
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output

        mock_sky.jobs.launch.assert_called_once()
        kwargs = mock_sky.jobs.launch.call_args.kwargs
        assert kwargs["name"] == "smoke-job-1"
        assert "cluster_name" not in kwargs
        assert "idle_minutes_to_autostop" not in kwargs
        assert "down" not in kwargs

    def test_tail_logs_invoked_with_follow_true_under_tail(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify ``--tail`` streams logs via ``sky.jobs.tail_logs(job_id=..., follow=True)``.

        The SDK rejects passing both `name=` and `job_id=` ("Cannot specify both name and job_id"),
        so the launcher passes only `job_id=` — the deterministic int the managed-jobs controller
        returned at submit time.
        """
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--tail",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        mock_sky.jobs.tail_logs.assert_called_once_with(job_id=1, follow=True)
        assert "name" not in mock_sky.jobs.tail_logs.call_args.kwargs

    def test_cancel_runs_on_success_under_tail(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Under `--tail`, `sky.jobs.cancel` is called once on the success path."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--tail",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        mock_sky.jobs.cancel.assert_called_once_with(name="smoke-job-1")

    # --- Job-name / spec-out / failure paths ---------------------------------

    def test_default_job_name_uses_task_name_prefix(
        self,
        tmp_path: Path,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify the launcher derives the job name from ``spec.task_name[:8]`` by default."""
        # Pin --spec-out so the test reads back the same spec the launcher composed,
        # rather than assuming spec.task_name equals the experiment id (an experiment
        # YAML may override `task_name`).
        spec_out = tmp_path / "input_spec.json"
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--spec-out",
            str(spec_out),
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        spec = DatasetSpec.model_validate_json(spec_out.read_text())

        kwargs: dict[str, Any] = mock_sky.jobs.launch.call_args.kwargs
        assert kwargs["name"].startswith("synth-setter-smoke-")
        assert kwargs["name"].endswith(spec.task_name[:8])

    def test_spec_out_overrides_default_path(
        self,
        tmp_path: Path,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """--spec-out forces an explicit local path (used by CI to find the spec for upload)."""
        explicit = tmp_path / "explicit-out" / "input_spec.json"
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--spec-out",
            str(explicit),
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        assert explicit.is_file()

    def test_worker_failed_rc_fails_launcher_under_tail(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Surface a non-zero ``jobs.tail_logs`` rc under ``--tail`` as a launcher exit and cancel.

        A non-zero rc from ``jobs.tail_logs`` means the worker job ended in a non-SUCCEEDED terminal
        status; the launcher surfaces that as a non-zero exit and still cancels the managed job.
        """
        mock_sky.jobs.tail_logs.return_value = 100

        result = _invoke(experiment, template_yaml, env_file, "--tail", fake_plugin=fake_plugin)
        assert result.exit_code != 0
        # Aggregate fan-out failure message names every failed rank with its rc.
        assert "rc=100" in result.output
        mock_sky.jobs.cancel.assert_called_once()

    def test_tail_logs_returning_none_with_follow_true_is_treated_as_failure(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`sky.jobs.tail_logs(follow=True)` only returns ``None`` when ``follow=False``.

        A ``None`` back from the SDK while we're passing ``follow=True`` is a contract violation —
        the launcher must surface that as a failure rather than mask it as success, otherwise it
        would exit 0 on a job whose terminal status the launcher never confirmed.
        """
        mock_sky.jobs.tail_logs.return_value = None

        result = _invoke(experiment, template_yaml, env_file, "--tail", fake_plugin=fake_plugin)
        assert result.exit_code != 0
        assert "tail_logs returned None" in result.output
        mock_sky.jobs.cancel.assert_called_once()

    # --- Edge cases ----------------------------------------------------------

    def test_launch_returning_none_job_id_aborts(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify the launcher aborts when sky.jobs.launch yields no job_id.

        Cancel is idempotent on a never-submitted job name (sky.jobs.cancel on a missing job is a
        no-op), so it still runs in the finally block to make multi-worker partial-failure cleanup
        uniform.
        """
        responses = {
            "launch-req": ([], MagicMock()),
        }
        mock_sky.stream_and_get.side_effect = lambda req: responses[req]

        result = _invoke(experiment, template_yaml, env_file, fake_plugin=fake_plugin)
        assert result.exit_code != 0
        assert "no job_id" in result.output.lower()
        mock_sky.jobs.tail_logs.assert_not_called()

    def test_cancel_runs_when_tail_logs_raises_under_tail(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify a ``tail_logs`` transport error under ``--tail`` still triggers job cancel.

        The managed job is always cancelled even if the log-stream side raised.
        """
        mock_sky.jobs.tail_logs.side_effect = RuntimeError("boom")

        result = _invoke(experiment, template_yaml, env_file, "--tail", fake_plugin=fake_plugin)
        assert result.exit_code != 0
        mock_sky.jobs.cancel.assert_called_once()

    def test_local_spec_persists_for_artifact_upload_even_on_launch_exception(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the local spec file persists when sky.jobs.launch raises post-upload.

        If sky.jobs.launch raises after the launcher materialized and R2-uploaded the spec, the
        local spec file under LOCAL_SPEC_DIR is still around for downstream artifact upload.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            lambda args: None,
        )
        mock_sky.jobs.launch.side_effect = RuntimeError("boom")

        result = _invoke(experiment, template_yaml, env_file, fake_plugin=fake_plugin)
        assert result.exit_code != 0

        spec_files = list(local_spec_dir.glob("*.json"))
        assert len(spec_files) == 1
        assert spec_files[0].read_text(), (
            "local spec file should still be on disk for artifact upload"
        )


class TestJobNameAlias:
    """`--job-name` is the primary launcher flag; `--cluster-name` is a deprecated alias.

    Both spellings bind to the same ``job_name`` Python parameter. The legacy ``--cluster-name``
    is preserved so out-of-tree callers (developer scripts, ad-hoc commands) keep working, but
    using it emits a one-line deprecation notice on stderr so callers know to migrate.
    """

    def test_job_name_sets_managed_job_name_kwarg(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`--job-name foo` flows through to ``sky.jobs.launch(name='foo', ...)``."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        assert mock_sky.jobs.launch.call_args.kwargs["name"] == "smoke-job-1"

    def test_cluster_name_alias_still_sets_managed_job_name_kwarg(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify the deprecated ``--cluster-name`` alias binds to the same parameter."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--cluster-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        assert mock_sky.jobs.launch.call_args.kwargs["name"] == "smoke-job-1"

    def test_cluster_name_alias_emits_deprecation_warning_on_stderr(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Using ``--cluster-name`` writes a one-line deprecation notice to stderr.

        The detection scans ``sys.argv`` (Click does not record which alias
        the caller used). ``CliRunner.invoke`` does not replace ``sys.argv``,
        so the test sets it via monkeypatch to mirror what production sees
        when invoked as ``python -m synth_setter.pipeline.skypilot_launch --cluster-name ...``.
        """
        runner = CliRunner(mix_stderr=False)
        args = [
            "--experiment",
            experiment,
            "--template",
            str(template_yaml),
            "--env-file",
            str(env_file),
            "--cluster-name",
            "smoke-job-1",
            f"render.plugin_path={fake_plugin}",
        ]
        monkeypatch.setattr("sys.argv", ["skypilot_launch", *args])
        result = runner.invoke(main, args)
        assert result.exit_code == 0, (result.stdout, result.stderr)
        assert "DEPRECATION" in result.stderr
        assert "--cluster-name" in result.stderr
        assert "--job-name" in result.stderr

    def test_job_name_does_not_emit_deprecation_warning(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The primary ``--job-name`` flag is quiet — no deprecation noise for the new name."""
        runner = CliRunner(mix_stderr=False)
        args = [
            "--experiment",
            experiment,
            "--template",
            str(template_yaml),
            "--env-file",
            str(env_file),
            "--job-name",
            "smoke-job-1",
            f"render.plugin_path={fake_plugin}",
        ]
        monkeypatch.setattr("sys.argv", ["skypilot_launch", *args])
        result = runner.invoke(main, args)
        assert result.exit_code == 0, (result.stdout, result.stderr)
        assert "DEPRECATION" not in result.stderr


class TestJobNameValidation:
    """Validate ``--job-name`` against a strict k8s-style label pattern before any SkyPilot call.

    ``--job-name`` is interpolated into a local filename under ``$TMPDIR`` and into an R2 object
    key before SkyPilot itself ever sees it, so the launcher validates the value up-front and
    rejects anything containing path separators, whitespace, or other shell-meaningful chars.
    """

    @pytest.mark.parametrize(
        "bad_name",
        [
            "../escape",  # parent-directory traversal
            "foo/bar",  # path separator
            "foo bar",  # whitespace
            "-leading-dash",  # must start alnum
            "_leading-underscore",  # must start alnum
            "foo;rm",  # shell metacharacter
            "a" * 64,  # > 63 chars
            "",  # empty string
        ],
    )
    def test_invalid_job_name_is_rejected_before_any_io(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        bad_name: str,
    ) -> None:
        """Verify bad names fail with a ClickException before any spec or sky.* call."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            bad_name,
            fake_plugin=fake_plugin,
        )
        assert result.exit_code != 0
        assert "--job-name" in result.output
        assert list(local_spec_dir.glob("*.json")) == []
        mock_sky.jobs.launch.assert_not_called()

    @pytest.mark.parametrize(
        "good_name",
        [
            "smoke-job-1",
            "abc",
            "A_b-1",
            "a" * 63,  # exactly the max length
            "0starts-with-digit",
        ],
    )
    def test_valid_job_name_is_accepted(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        good_name: str,
    ) -> None:
        """Verify k8s-label-compatible names pass through to sky.jobs.launch unchanged."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            good_name,
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        assert mock_sky.jobs.launch.call_args.kwargs["name"] == good_name

    def test_derived_default_name_with_bad_task_name_is_rejected(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the derived default job name is rejected for disallowed ``spec.task_name``.

        If ``--job-name`` is not passed and ``spec.task_name`` contains characters not allowed in
        a managed-job name (e.g. ``/`` or ``..``), the launcher must reject the derived default
        *before* writing the spec or invoking sky — path-traversal hardening for the local
        tempfile and the R2 object key.
        """
        # Build a real DatasetSpec carrying a task_name with a path separator and have
        # the launcher's hydra-compose helper hand it back to main().
        bad_spec = DatasetSpec.model_validate(
            {
                "task_name": "foo/bar",  # `:8` slice → "foo/bar" — fails _JOB_NAME_RE
                "output_format": "hdf5",
                "train_val_test_sizes": [1, 0, 0],
                "base_seed": 0,
                "r2_bucket": "intermediate-data",
                "render": {
                    "plugin_path": str(fake_plugin),
                    "preset_path": "presets/surge-base.vstpreset",
                    "param_spec_name": "surge_simple",
                    "renderer_version": "1.3.4",
                    "sample_rate": 16000,
                    "channels": 2,
                    "velocity": 100,
                    "signal_duration_seconds": 4.0,
                    "min_loudness": -55.0,
                    "samples_per_render_batch": 1,
                    "samples_per_shard": 1,
                },
            }
        )
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch._compose_dataset_spec",
            lambda *_a, **_k: bad_spec,
        )

        result = _invoke(experiment, template_yaml, env_file, fake_plugin=fake_plugin)

        assert result.exit_code != 0
        assert "derived job name" in result.output
        assert list(local_spec_dir.glob("*.json")) == []
        mock_sky.jobs.launch.assert_not_called()


class TestNoTailMode:
    """Cover default ``--no-tail`` mode: submit each rank, print operator commands, exit 0.

    The launcher prints the per-rank job_id along with the ``sky jobs logs`` / ``sky jobs cancel``
    commands the operator can run, then exits 0 without tailing or cancelling.

    The managed-jobs controller's terminal-status lifecycle is the safety net for jobs left
    running. The launcher only cancels a job in this mode if its own `sky.jobs.launch` raised
    or yielded no job_id (half-submitted).
    """

    def test_no_tail_is_default_and_skips_tail_logs(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Without an explicit flag the launcher detaches: `sky.jobs.tail_logs` is never invoked."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        mock_sky.jobs.tail_logs.assert_not_called()

    def test_no_tail_does_not_cancel_job_on_success(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify a successful detach leaves the managed job running for the controller.

        The controller's terminal-status lifecycle is the safety net, not the launcher's
        ``finally`` cancel.
        """
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--no-tail",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        mock_sky.jobs.cancel.assert_not_called()

    def test_no_tail_prints_sky_jobs_logs_and_cancel_commands_per_rank(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify each submitted job yields copy-pasteable ``sky jobs`` operator commands."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--no-tail",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        assert "sky jobs logs --name smoke-job-1" in result.output
        assert "sky jobs cancel --name smoke-job-1" in result.output

    def test_no_tail_multi_worker_prints_per_rank_block_for_each_rank(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify multi-worker detach prints one operator-commands block per rank."""
        TestNumWorkersFanOut._setup_n_workers_mock(mock_sky, n=3)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "3",
            "--no-tail",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code == 0, result.output
        for i in range(3):
            assert f"sky jobs logs --name smoke-job-1-r{i}" in result.output
            assert f"sky jobs cancel --name smoke-job-1-r{i}" in result.output
        mock_sky.jobs.tail_logs.assert_not_called()
        mock_sky.jobs.cancel.assert_not_called()

    def test_no_tail_partial_launch_failure_only_cancels_failed_job(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify multi-worker detach cancels only the half-submitted rank.

        If rank 1's launch yields no job_id (half-submitted) but the other ranks succeed, only the
        failed job is cancelled. Successful jobs stay running.
        """
        job_names = [f"smoke-job-1-r{i}" for i in range(3)]
        tasks = {name: MagicMock(name=f"task-{name}") for name in job_names}
        mock_sky.Task.from_yaml_config.side_effect = list(tasks.values())

        launch_reqs = {name: f"launch-{name}" for name in job_names}
        cancel_reqs = {name: f"cancel-{name}" for name in job_names}
        mock_sky.jobs.launch.side_effect = lambda task, **kw: launch_reqs[kw["name"]]
        mock_sky.jobs.cancel.side_effect = lambda **kw: cancel_reqs[kw["name"]]

        # rank 1's launch yields an empty job_ids list so the launcher treats it as half-submitted.
        stream_responses: dict[str, object] = {
            launch_reqs["smoke-job-1-r0"]: ([1], MagicMock()),
            launch_reqs["smoke-job-1-r1"]: ([], MagicMock()),
            launch_reqs["smoke-job-1-r2"]: ([3], MagicMock()),
        }
        stream_responses.update({req: None for req in cancel_reqs.values()})
        mock_sky.stream_and_get.side_effect = lambda req: stream_responses[req]

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "3",
            "--no-tail",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code != 0
        cancel_targets = [call.kwargs["name"] for call in mock_sky.jobs.cancel.call_args_list]
        assert cancel_targets == ["smoke-job-1-r1"]
        for surviving_job in ("smoke-job-1-r0", "smoke-job-1-r2"):
            assert f"sky jobs cancel --name {surviving_job}" in result.output

    def test_no_tail_single_worker_launch_failure_cancels_that_job(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify a single-worker detach cancels the half-submitted job when launch raises."""
        mock_sky.jobs.launch.side_effect = RuntimeError("boom")

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--no-tail",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code != 0
        mock_sky.jobs.cancel.assert_called_once_with(name="smoke-job-1")


class TestNumWorkersFanOut:
    """`--num-workers N>1` fans out N independent managed jobs.

    RunPod's backend doesn't support num_nodes>1, so the launcher synthesizes multi-worker
    partitioning by submitting N managed jobs in parallel and injecting SYNTH_SETTER_WORKER_RANK /
    SYNTH_SETTER_NUM_WORKERS per job. Each job downloads the same materialized spec;
    synth_setter.pipeline.partitioning.get_my_shards slices each worker's shard ownership.
    """

    @staticmethod
    def _setup_n_workers_mock(
        mock_sky: MagicMock,
        n: int,
        *,
        tail_rcs_by_name: dict[str, int | None] | None = None,
        base_job_name: str = "smoke-job-1",
    ) -> dict[str, MagicMock]:
        """Configure `mock_sky` for an N-job run with deterministic per-rank routing.

        Returns a ``job_name -> Task`` dict keyed by the managed-job name the launcher will
        request, so tests can inspect each rank's ``update_envs`` call without depending on
        ThreadPoolExecutor scheduling order. ``jobs.launch`` and ``jobs.cancel`` route by
        their ``name`` kwarg; ``jobs.tail_logs`` routes by ``job_id`` (the launcher passes
        only ``job_id`` because the SDK rejects ``name + job_id`` together). Either way
        an rc=100 for the rank-1 job deterministically attaches to rank 1 regardless of
        which thread ran first. Pass ``rc=None`` for a job to simulate the SDK contract
        violation where ``tail_logs(follow=True)`` returns ``None`` (the launcher must
        surface that as a failure rather than mask it as success).
        """
        rcs = tail_rcs_by_name if tail_rcs_by_name is not None else {}
        job_names = [f"{base_job_name}-r{i}" for i in range(n)]
        tasks = {name: MagicMock(name=f"task-{name}") for name in job_names}
        # `name` isn't a Task.from_yaml arg, so route Tasks by call order. The launcher creates
        # exactly one Task per rank inside _launch_and_tail and immediately update_envs's it
        # with the job name in the message — assertions key off the job name in the env, not
        # the Task identity, so call order doesn't matter here.
        mock_sky.Task.from_yaml_config.side_effect = list(tasks.values())

        # Route jobs.launch + jobs.cancel + stream_and_get by `name` kwarg; route
        # jobs.tail_logs by `job_id` since the launcher passes only job_id (the SDK rejects
        # `name + job_id` together with "Cannot specify both name and job_id").
        launch_reqs = {name: f"launch-{name}" for name in job_names}
        cancel_reqs = {name: f"cancel-{name}" for name in job_names}
        mock_sky.jobs.launch.side_effect = lambda task, **kw: launch_reqs[kw["name"]]
        mock_sky.jobs.cancel.side_effect = lambda **kw: cancel_reqs[kw["name"]]

        # rank i → job_id = i + 1 (the controller's id sequence in the stream_and_get response).
        job_id_for_name = {name: i + 1 for i, name in enumerate(job_names)}
        rcs_by_job_id = {job_id_for_name[name]: rc for name, rc in rcs.items()}

        stream_responses: dict[str, object] = {
            launch_reqs[name]: ([job_id_for_name[name]], MagicMock()) for name in job_names
        }
        stream_responses.update({req: None for req in cancel_reqs.values()})
        mock_sky.stream_and_get.side_effect = lambda req: stream_responses[req]

        mock_sky.jobs.tail_logs.side_effect = lambda **kw: rcs_by_job_id.get(kw["job_id"], 0)
        return tasks

    def test_three_workers_launches_three_jobs_under_tail(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify ``--tail --num-workers 3`` submits 3 jobs and tails+cancels each one."""
        self._setup_n_workers_mock(mock_sky, n=3)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "3",
            "--tail",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert mock_sky.jobs.launch.call_count == 3
        assert mock_sky.jobs.tail_logs.call_count == 3
        assert mock_sky.jobs.cancel.call_count == 3

    def test_three_workers_use_rank_suffixed_job_names(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify N>1 launches use ``<base>-r{i}`` job names for distinguishable entries."""
        self._setup_n_workers_mock(mock_sky, n=3)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "3",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        launch_job_names = sorted(
            call.kwargs["name"] for call in mock_sky.jobs.launch.call_args_list
        )
        assert launch_job_names == ["smoke-job-1-r0", "smoke-job-1-r1", "smoke-job-1-r2"]

    def test_one_worker_keeps_unsuffixed_job_name(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`--num-workers 1` (default) keeps the unsuffixed job name for backward-compat.

        Debug workflows / dashboards key off the unsuffixed name.
        """
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        mock_sky.jobs.launch.assert_called_once()
        assert mock_sky.jobs.launch.call_args.kwargs["name"] == "smoke-job-1"

    def test_three_workers_inject_distinct_rank_world_per_job(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify each rank's task gets distinct WORKER_RANK / NUM_WORKERS env vars.

        Workers read ``SYNTH_SETTER_WORKER_RANK`` and ``SYNTH_SETTER_NUM_WORKERS`` via
        ``read_rank_world_from_env`` and partition the shared spec via ``get_my_shards``.
        """
        tasks = self._setup_n_workers_mock(mock_sky, n=3)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "3",
            fake_plugin=fake_plugin,
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
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify spec is uploaded to R2 once and shared across all ranks via one prefix.

        Single ``r2_prefix`` so the partition is one logical dataset, not three.
        """
        self._setup_n_workers_mock(mock_sky, n=3)
        rclone_invocations: list[list[str]] = []
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            lambda args: rclone_invocations.append(args),
        )

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "3",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert len(rclone_invocations) == 1
        assert rclone_invocations[0][-1] == (
            "r2:intermediate-data/skypilot-launcher-specs/smoke-job-1.json"
        )

    def test_one_worker_failure_among_three_fails_launcher_after_full_cancel_under_tail(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify ``--tail`` exits non-zero on any rank's tail rc while cancelling every job.

        Every managed job (success or fail) gets cancelled — partial-failure cleanup must be
        uniform.
        """
        self._setup_n_workers_mock(mock_sky, n=3, tail_rcs_by_name={"smoke-job-1-r1": 100})

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "3",
            "--tail",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code != 0
        assert "rc=100" in result.output
        assert "smoke-job-1-r1" in result.output
        assert mock_sky.jobs.cancel.call_count == 3

    def test_one_worker_tail_logs_returning_none_fails_launcher_after_full_cancel_under_tail(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Fan-out variant: one rank's tail_logs returns None and is treated as failure.

        Companion to ``test_tail_logs_returning_none_with_follow_true_is_treated_as_failure``.
        One rank's `sky.jobs.tail_logs(follow=True)` returns ``None`` (SDK contract violation: the
        SDK only returns None when follow=False). The launcher must surface that as failure rather
        than mask it as success, and every peer job — successes and the None-returner alike —
        must still be cancelled.
        """
        self._setup_n_workers_mock(mock_sky, n=3, tail_rcs_by_name={"smoke-job-1-r1": None})

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "3",
            "--tail",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code != 0
        assert "tail_logs returned None" in result.output
        assert "smoke-job-1-r1" in result.output
        assert mock_sky.jobs.cancel.call_count == 3

    def test_worker_git_ref_forwarded_to_every_rank(
        self,
        experiment: str,
        fake_plugin: Path,
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
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "3",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        forwarded = [t.update_envs.call_args.args[0] for t in tasks.values()]
        for env in forwarded:
            assert env["WORKER_GIT_REF"] == "abc1234deadbeef"

    def test_zero_or_negative_num_workers_rejected(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """`--num-workers 0` (or negative) is a CLI usage error — never reach sky.*."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "0",
            fake_plugin=fake_plugin,
        )
        assert result.exit_code != 0
        assert "must be >= 1" in result.output
        mock_sky.jobs.launch.assert_not_called()

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
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        bad_tag: str,
    ) -> None:
        """Verify invalid ``--worker-image-tag`` values fail before sky.* is touched.

        ``--worker-image-tag`` is interpolated into a docker ref; the launcher must reject bad tags
        rather than produce surprising image refs like ``tinaudio/synth-setter:foo:bar``.
        """
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--worker-image-tag",
            bad_tag,
            fake_plugin=fake_plugin,
        )
        assert result.exit_code != 0
        assert "--worker-image-tag" in result.output
        mock_sky.jobs.launch.assert_not_called()


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
        """Verify every entry in a multi-Resources alt-set is mutated, not just the first."""
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
        """Verify an OCI Resources entry passes through unchanged without ``copy(image_id=...)``.

        The helper always rebuilds `task.resources` from the original entries, so it may
        call `set_resources`; what matters behaviorally is that the OCI entry is never
        copied with a new image_id and is preserved verbatim in the rebuilt list.
        """
        import sky.clouds

        class FakeOCI:
            pass

        monkeypatch.setattr(sky.clouds, "OCI", FakeOCI, raising=False)

        oci_res = self._make_resource(FakeOCI())
        task = self._make_task([oci_res])

        _override_image_id(task, "tinaudio/synth-setter:test-tag")

        oci_res.copy.assert_not_called()
        if task.set_resources.called:
            new_resources = list(task.set_resources.call_args.args[0])
            assert len(new_resources) == 1
            assert new_resources[0] is oci_res


# ---------------------------------------------------------------------------
# _detect_provider — Task YAML → cred-bootstrap --provider flag
# ---------------------------------------------------------------------------


class TestDetectProvider:
    """``_detect_provider`` reads ``ComputeConfig.resources`` and maps cloud → ``--provider``.

    Reads the validated Pydantic config rather than reparsing YAML so detection stays in
    lockstep with what the launcher ships to SkyPilot. Inputs that would fail Pydantic
    validation (non-mapping ``resources``, empty document, etc.) are covered in
    ``tests/pipeline/test_schemas/test_compute_config.py``.
    """

    @staticmethod
    def _make(resources: dict) -> ComputeConfig:  # noqa: DOC101,DOC103,DOC201,DOC203
        """Build a minimal ComputeConfig with the supplied ``resources`` block."""
        return ComputeConfig(
            resources=resources,
            envs={},
            setup="echo setup",
            run="echo run",
        )

    def test_runpod_cloud_detected_as_runpod(self) -> None:
        """Flat `resources.cloud: runpod` template maps to `--provider runpod`."""
        cfg = self._make({"cloud": "runpod"})
        assert _real_detect_provider(cfg) == "runpod"

    def test_oci_any_of_detected_as_oci(self) -> None:
        """OCI templates use `resources.any_of: [{cloud: oci, ...}, ...]` for shape fan-out."""
        cfg = self._make({"any_of": [{"cloud": "oci", "instance_type": "VM.X"}]})

        assert _real_detect_provider(cfg) == "oci"

    def test_kubernetes_cloud_detected_as_local(self) -> None:
        """Verify ``cloud: kubernetes`` maps to the internal ``local`` tag.

        Used by sky-local-up kind clusters. The ``local`` tag gates skipping the cred bootstrap
        (kind needs no compute creds — see PR #876).
        """
        cfg = self._make({"cloud": "kubernetes"})

        assert _real_detect_provider(cfg) == "local"

    def test_unknown_cloud_raises(self) -> None:
        """A cloud the bootstrap doesn't support (e.g. aws) fails loudly with a clear error."""
        cfg = self._make({"cloud": "aws"})
        with pytest.raises(click.ClickException, match="(?i)unsupported cloud"):
            _real_detect_provider(cfg)

    def test_missing_cloud_raises(self) -> None:
        """A template with no detectable `cloud` field raises a launcher misuse error."""
        cfg = self._make({"cpus": "2+"})
        with pytest.raises(click.ClickException, match="(?i)could not detect cloud"):
            _real_detect_provider(cfg)

    def test_non_list_any_of_raises_click_exception(self) -> None:
        """A scalar ``resources.any_of`` raises a clean ClickException.

        Pydantic accepts any value under ``resources`` (it's typed ``dict[str, Any]``), so this
        path is reachable; the launcher rejects it before handing the dict to SkyPilot.
        """
        cfg = self._make({"any_of": "not-a-list"})
        with pytest.raises(click.ClickException, match="(?i)any_of.*list"):
            _real_detect_provider(cfg)

    def test_non_mapping_any_of_first_entry_raises_click_exception(self) -> None:
        """Verify a non-mapping ``resources.any_of[0]`` raises a clean ClickException."""
        cfg = self._make({"any_of": ["not-a-mapping"]})
        with pytest.raises(click.ClickException, match="(?i)any_of\\[0\\].*mapping"):
            _real_detect_provider(cfg)


# ---------------------------------------------------------------------------
# _run_cred_bootstrap — invokes the script; honors SKYPILOT_API_SERVER_ENDPOINT;
# never streams stdout (script is silent by design but defensive capture too).
# ---------------------------------------------------------------------------


class TestRunCredBootstrap:
    """Behavioral contracts for the launcher's wrapping of the cred-bootstrap script."""

    def test_skips_when_api_server_endpoint_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the bootstrap script is NOT invoked when SKYPILOT_API_SERVER_ENDPOINT is set.

        The remote API server holds creds. Returns silently (no exception, no script call).
        """
        monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://api:pw@server/")
        called: list[str] = []
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.run",
            lambda *args, **kwargs: called.append("invoked"),  # type: ignore[misc]
        )
        # bypass the autouse no-op fixture

        _real_run_cred_bootstrap(provider="runpod")
        assert called == [], "bootstrap script should not be invoked in remote-server mode"

    def test_passes_merged_env_to_subprocess(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify ``--env-file`` values are merged into the bootstrap subprocess env.

        ``RCLONE_CONFIG_R2_*`` / ``RUNPOD_API_KEY`` / ``OCI_*`` set only in the dotenv file are
        visible to the bootstrap script (which runs ``resolve_var`` against them).
        """
        env_file = tmp_path / "creds.env"
        env_file.write_text(
            "RUNPOD_API_KEY=from-env-file\nRCLONE_CONFIG_R2_ACCESS_KEY_ID=from-env-file\n"
        )
        monkeypatch.delenv("SKYPILOT_API_SERVER_ENDPOINT", raising=False)
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

        captured: dict[str, Any] = {}

        def fake_run(args: list[str], **kwargs: Any) -> MagicMock:
            captured["env"] = kwargs.get("env", {})
            captured["args"] = args
            result = MagicMock()
            result.stdout = ""
            result.stderr = ""
            result.returncode = 0
            return result

        monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.subprocess.run", fake_run)

        _real_run_cred_bootstrap(provider="runpod", env_file_path=env_file)

        assert captured["env"]["RUNPOD_API_KEY"] == "from-env-file"
        assert captured["env"]["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "from-env-file"

    def test_propagates_script_failure_as_click_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify a non-zero rc from the bootstrap script bubbles up as a ClickException.

        The launcher fails fast rather than continuing with a half-written cred state.
        """
        import subprocess as _subprocess

        monkeypatch.delenv("SKYPILOT_API_SERVER_ENDPOINT", raising=False)

        def fake_run(args: list[str], **kwargs: Any) -> Any:
            raise _subprocess.CalledProcessError(
                returncode=1, cmd=args, stderr="::error::missing var"
            )

        monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.subprocess.run", fake_run)

        with pytest.raises(click.ClickException, match="(?i)cred bootstrap failed"):
            _real_run_cred_bootstrap(provider="runpod")

    def test_capture_output_keeps_stdout_off_caller_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify bootstrap-script stdout is captured via ``capture_output=True``.

        Even if the script ever started emitting stdout (it shouldn't), capture_output=True
        prevents a tee'd workflow caller from leaking it. Pin the kwarg here so a future edit can't
        drop the capture.
        """
        captured: dict[str, Any] = {}

        def fake_run(args: list[str], **kwargs: Any) -> MagicMock:
            captured["kwargs"] = kwargs
            result = MagicMock()
            result.stdout = "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=should-never-leak"
            result.stderr = ""
            result.returncode = 0
            return result

        monkeypatch.delenv("SKYPILOT_API_SERVER_ENDPOINT", raising=False)
        monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.subprocess.run", fake_run)

        # If the launcher were echoing stdout, click.testing.CliRunner would surface it.
        # Here we just assert the call shape: capture_output=True, env supplied.
        _real_run_cred_bootstrap(provider="runpod")
        assert captured["kwargs"].get("capture_output") is True
        assert "env" in captured["kwargs"]


# ---------------------------------------------------------------------------
# main() bridges RCLONE_CONFIG_R2_* into os.environ before upload_spec_to_r2
# ---------------------------------------------------------------------------


class TestNumWorkersConfigPrecedence:
    """``num_workers`` resolution: ``--num-workers`` CLI flag wins; else default 1.

    Worker count is a launcher concern — it's not on ``DatasetSpec``. A legacy
    YAML's ``num_workers`` field is silently ignored by the legacy bridge.
    """

    def test_cli_num_workers_drives_fan_out(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """``--num-workers 2`` drives a 2-cluster fan-out."""
        TestNumWorkersFanOut._setup_n_workers_mock(mock_sky, n=2)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--num-workers",
            "2",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert mock_sky.jobs.launch.call_count == 2

    def test_default_when_cli_omitted(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify the schema default (1) drives a single-cluster fan-out when unset.

        The fixture's experiment has no ``num_workers``, and no ``--num-workers`` is passed.
        """
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert mock_sky.jobs.launch.call_count == 1


class TestDispatchMode:
    """`--api-server` / `--local` make the launcher's remote-vs-local dispatch explicit (#841).

    Today's contract — "if SKYPILOT_API_SERVER_ENDPOINT is set in process env, dispatch remote;
    else local SDK" — is preserved when neither flag is passed (backward-compat), but each call
    site can now declare its intent via argv instead of via env-passthrough.
    """

    def test_local_flag_clears_inherited_api_server_endpoint(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify ``--local`` clears an inherited ``SKYPILOT_API_SERVER_ENDPOINT``."""
        monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://stale.example.com")

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--local",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert "SKYPILOT_API_SERVER_ENDPOINT" not in os.environ

    def test_api_server_flag_sets_endpoint_in_os_environ(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify ``--api-server <url>`` exports the endpoint into ``os.environ`` pre-sky.

        The SDK then dispatches to the remote server.
        """
        monkeypatch.delenv("SKYPILOT_API_SERVER_ENDPOINT", raising=False)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--api-server",
            "https://api.example.com",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert os.environ.get("SKYPILOT_API_SERVER_ENDPOINT") == "https://api.example.com"

    def test_api_server_flag_strips_surrounding_whitespace(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-empty URL with surrounding whitespace is stripped before being exported."""
        monkeypatch.delenv("SKYPILOT_API_SERVER_ENDPOINT", raising=False)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--api-server",
            "  https://api.example.com  ",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert os.environ.get("SKYPILOT_API_SERVER_ENDPOINT") == "https://api.example.com"

    def test_api_server_flag_rejects_blank_value(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify a blank/whitespace-only ``--api-server`` value is rejected with a clear error.

        Rather than silently setting an empty endpoint that makes downstream cred-bootstrap
        behavior confusing.
        """
        monkeypatch.delenv("SKYPILOT_API_SERVER_ENDPOINT", raising=False)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--api-server",
            "   ",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code != 0
        assert "non-empty" in result.output.lower() or "blank" in result.output.lower()
        assert "SKYPILOT_API_SERVER_ENDPOINT" not in os.environ

    def test_api_server_flag_skips_cred_bootstrap(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify ``--api-server <url>`` short-circuits ``_run_cred_bootstrap`` entirely."""
        monkeypatch.delenv("SKYPILOT_API_SERVER_ENDPOINT", raising=False)

        # Re-patch _run_cred_bootstrap to the real impl so the SKYPILOT_API_SERVER_ENDPOINT
        # short-circuit is exercised end-to-end.
        called: list[str] = []

        def fake_run(args: list[str], **kwargs: Any) -> MagicMock:
            called.append("invoked")
            result = MagicMock()
            result.stdout = ""
            result.stderr = ""
            result.returncode = 0
            return result

        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch._run_cred_bootstrap",
            _real_run_cred_bootstrap,
        )
        monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.subprocess.run", fake_run)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--api-server",
            "https://api.example.com",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert called == [], "cred bootstrap should be skipped when --api-server is set"

    def test_local_flag_runs_cred_bootstrap_even_with_inherited_endpoint(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify ``--local`` clears the inherited endpoint AND runs the cred bootstrap.

        The launcher host needs creds on disk for local SDK dispatch.
        """
        monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://stale.example.com")
        called: list[str] = []

        def fake_run(args: list[str], **kwargs: Any) -> MagicMock:
            called.append("invoked")
            result = MagicMock()
            result.stdout = ""
            result.stderr = ""
            result.returncode = 0
            return result

        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch._run_cred_bootstrap",
            _real_run_cred_bootstrap,
        )
        monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.subprocess.run", fake_run)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--local",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert called == ["invoked"], "cred bootstrap must run under --local"

    def test_local_provider_template_skips_cred_bootstrap(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify ``main()`` skips ``_run_cred_bootstrap`` for ``local``-detected templates.

        Kubernetes/kind needs no compute creds and the CI workflow writes the controller-resource
        shrink directly to ``~/.sky/config.yaml``.

        See PR #876.
        """
        bootstrap_calls: list[str] = []

        def tracking_bootstrap(**_kwargs: Any) -> None:
            bootstrap_calls.append("invoked")

        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch._run_cred_bootstrap",
            tracking_bootstrap,
        )
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch._detect_provider",
            lambda _task: "local",
        )

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert bootstrap_calls == [], (
            "cred bootstrap should be skipped for kubernetes/local templates"
        )

    def test_both_flags_passed_is_rejected(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Verify ``--api-server`` and ``--local`` are mutually exclusive (usage error)."""
        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            "--api-server",
            "https://api.example.com",
            "--local",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
        mock_sky.jobs.launch.assert_not_called()

    def test_neither_flag_preserves_inherited_endpoint(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the launcher preserves the inherited endpoint without dispatch flags.

        Without ``--api-server`` or ``--local``, ``SKYPILOT_API_SERVER_ENDPOINT`` is left alone
        for backward-compat with workflows that already rely on env-var passthrough.
        """
        monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://inherited.example.com")

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert os.environ.get("SKYPILOT_API_SERVER_ENDPOINT") == "https://inherited.example.com"

    def test_neither_flag_with_unset_endpoint_leaves_it_unset(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No flag and no inherited env var → still no env var (no surprise default)."""
        monkeypatch.delenv("SKYPILOT_API_SERVER_ENDPOINT", raising=False)

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert "SKYPILOT_API_SERVER_ENDPOINT" not in os.environ


class TestWorkerEnvToOsEnvironBridge:
    """Verify ``main()`` bridges ``RCLONE_CONFIG_R2_*`` from worker_env into ``os.environ``.

    ``upload_spec_to_r2``'s ``rclone copyto`` subprocess inherits them this way.

    Local-dev ``--env-file`` paths populate worker_env without exporting; without this bridge
    rclone would see no creds.
    """

    def test_env_file_prefixed_keys_bridge_to_os_environ(
        self,
        experiment: str,
        fake_plugin: Path,
        template_yaml: Path,
        tmp_path: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify prefixed names from ``--env-file`` are copied into ``os.environ``.

        These get bridged before the rclone subprocess runs, even though they are not exported.
        """
        env_file = tmp_path / "prefixed.env"
        env_file.write_text(
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID=ak-from-env\n"
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=sk-from-env\n"
            "RCLONE_CONFIG_R2_ENDPOINT=https://e.r2\n"
        )
        captured: dict[str, str] = {}

        def fake_rclone(args: list[str]) -> None:
            for key in (
                "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
                "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
                "RCLONE_CONFIG_R2_ENDPOINT",
            ):
                captured[key] = os.environ.get(key, "<unset>")

        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call", fake_rclone
        )

        result = _invoke(
            experiment,
            template_yaml,
            env_file,
            "--job-name",
            "smoke-job-1",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code == 0, result.output
        assert captured["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "ak-from-env"
        assert captured["RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"] == "sk-from-env"  # noqa: S105 — fixture value, not a real secret
        assert captured["RCLONE_CONFIG_R2_ENDPOINT"] == "https://e.r2"


# ---------------------------------------------------------------------------
# _SECRET_WORKER_ENV_KEYS — module-level constant for the unconfigured-creds
# check. Subset of `_WORKER_ENV_KEYS` excluding the non-secret rclone
# defaults that ``_R2_RCLONE_CONSTANTS`` already supplies.
# ---------------------------------------------------------------------------


class TestSecretWorkerEnvKeys:
    """`_SECRET_WORKER_ENV_KEYS` is the residual subset used to detect unconfigured creds."""

    def test_excludes_non_secret_rclone_constants(self) -> None:
        """Verify TYPE / PROVIDER (non-secret rclone config) are excluded from the subset."""
        from synth_setter.pipeline.skypilot_launch import (
            _R2_RCLONE_CONSTANTS,
            _SECRET_WORKER_ENV_KEYS,
        )

        assert "RCLONE_CONFIG_R2_TYPE" not in _SECRET_WORKER_ENV_KEYS
        assert "RCLONE_CONFIG_R2_PROVIDER" not in _SECRET_WORKER_ENV_KEYS
        for key in _R2_RCLONE_CONSTANTS:
            assert key not in _SECRET_WORKER_ENV_KEYS

    def test_includes_runtime_secret_keys(self) -> None:
        """The access-key / secret-key / endpoint triple must remain in the secret subset."""
        from synth_setter.pipeline.skypilot_launch import _SECRET_WORKER_ENV_KEYS

        assert "RCLONE_CONFIG_R2_ACCESS_KEY_ID" in _SECRET_WORKER_ENV_KEYS
        assert "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY" in _SECRET_WORKER_ENV_KEYS
        assert "RCLONE_CONFIG_R2_ENDPOINT" in _SECRET_WORKER_ENV_KEYS

    def test_is_subset_of_worker_env_keys(self) -> None:
        """The secret subset is closed-form derived from `_WORKER_ENV_KEYS`."""
        from synth_setter.pipeline.skypilot_launch import (
            _SECRET_WORKER_ENV_KEYS,
            _WORKER_ENV_KEYS,
        )

        assert set(_SECRET_WORKER_ENV_KEYS).issubset(set(_WORKER_ENV_KEYS))


# ---------------------------------------------------------------------------
# _launch_one_rank — module-level helper for fan-out launch (lifted from
# _run_workers closure for testability).
# ---------------------------------------------------------------------------


class TestLaunchOneRank:
    """`_launch_one_rank` builds the per-rank Task, calls sky.jobs.launch, returns the job_id."""

    def test_returns_job_id_from_stream_and_get(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: stream_and_get returns ``([job_id], handle)`` and the helper returns the job_id."""
        from synth_setter.pipeline.skypilot_launch import _launch_one_rank

        fake_sky = MagicMock()
        fake_sky.Task.from_yaml_config.return_value = MagicMock()
        fake_sky.jobs.launch.return_value = "req-1"
        fake_sky.stream_and_get.return_value = ([42], None)
        monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.sky", fake_sky)

        job_id = _launch_one_rank(
            0,
            job_names=["job-0"],
            worker_env_base={"RCLONE_CONFIG_R2_TYPE": "s3"},
            worker_image="repo:tag",
            compute_config=ComputeConfig(
                resources={"cloud": "runpod"}, envs={}, setup="echo", run="echo"
            ),
        )

        assert job_id == 42

    @pytest.mark.parametrize(
        "stream_and_get_return,expected_match",
        [
            # `launch_result is None` guard — different message than the empty/null job_ids path.
            (None, r"returned None \(no submission handle\)"),
            # The three "no job_id" shapes all hit the empty/null job_ids guard.
            ((None, None), "returned no job_id"),
            (([], None), "returned no job_id"),
            (([None], None), "returned no job_id"),
        ],
        ids=["launch_result_none", "job_ids_none", "job_ids_empty", "job_ids_first_none"],
    )
    def test_raises_when_stream_and_get_yields_no_job_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stream_and_get_return: object,
        expected_match: str,
    ) -> None:
        """Verify all four "no job_id" SDK result shapes raise ``ClickException``.

        ``sky.jobs.launch`` + ``sky.stream_and_get`` together yield ``(Optional[List[int]], handle)``
        (or ``None`` if the whole submission was dropped). All four shapes — entire result is
        ``None``, ``(None, None)``, ``([], None)``, ``([None], None)`` — must raise. The ``None``
        result hits the submission-handle guard; the other three hit the empty/null job_ids guard
        with a distinct message.
        """
        from synth_setter.pipeline.skypilot_launch import _launch_one_rank

        fake_sky = MagicMock()
        fake_sky.Task.from_yaml_config.return_value = MagicMock()
        fake_sky.jobs.launch.return_value = "req-1"
        fake_sky.stream_and_get.return_value = stream_and_get_return
        monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.sky", fake_sky)

        with pytest.raises(click.ClickException, match=expected_match):
            _launch_one_rank(
                0,
                job_names=["job-0"],
                worker_env_base={},
                worker_image="repo:tag",
                compute_config=ComputeConfig(
                    resources={"cloud": "runpod"}, envs={}, setup="echo", run="echo"
                ),
            )

    def test_returned_job_id_comes_from_stream_and_get_not_from_rank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin the contract that the returned job_id is sourced from the SDK, not from rank.

        The returned job_id is ``stream_and_get_result[0][0]``, not derived from ``rank`` (e.g.
        ``rank + 1``) or any other positional convention.

        The ``TestNumWorkersFanOut`` helper happens to map rank ``i`` to job_id ``i + 1`` for
        routing simplicity; this test makes sure that convention can't be baked into the
        production helper by accident.
        """
        from synth_setter.pipeline.skypilot_launch import _launch_one_rank

        fake_sky = MagicMock()
        fake_sky.Task.from_yaml_config.return_value = MagicMock()
        fake_sky.jobs.launch.return_value = "req-1"
        # A job_id deliberately unrelated to any rank-derived value: not rank, not rank+1, not 0.
        fake_sky.stream_and_get.return_value = ([424242], None)
        monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.sky", fake_sky)

        job_id = _launch_one_rank(
            0,
            job_names=["job-rank-0"],
            worker_env_base={},
            worker_image="repo:tag",
            compute_config=ComputeConfig(
                resources={"cloud": "runpod"}, envs={}, setup="echo", run="echo"
            ),
        )

        assert job_id == 424242

    def test_injects_rank_and_num_workers_into_task_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`update_envs` receives the rank, world size, and worker image alongside the base env."""
        from synth_setter.pipeline.partitioning import NUM_WORKERS_ENV_VAR, WORKER_RANK_ENV_VAR
        from synth_setter.pipeline.skypilot_launch import _WORKER_IMAGE_ENV, _launch_one_rank

        fake_task = MagicMock()
        fake_sky = MagicMock()
        fake_sky.Task.from_yaml_config.return_value = fake_task
        fake_sky.jobs.launch.return_value = "req-1"
        fake_sky.stream_and_get.return_value = ([7], None)
        monkeypatch.setattr("synth_setter.pipeline.skypilot_launch.sky", fake_sky)

        _launch_one_rank(
            2,
            job_names=["a", "b", "c", "d"],
            worker_env_base={"BASE_KEY": "base-value"},
            worker_image="repo:tag",
            compute_config=ComputeConfig(
                resources={"cloud": "runpod"}, envs={}, setup="echo", run="echo"
            ),
        )

        envs_passed = fake_task.update_envs.call_args.args[0]
        assert envs_passed["BASE_KEY"] == "base-value"
        assert envs_passed[WORKER_RANK_ENV_VAR] == "2"
        assert envs_passed[NUM_WORKERS_ENV_VAR] == "4"
        assert envs_passed[_WORKER_IMAGE_ENV] == "repo:tag"


# ---------------------------------------------------------------------------
# Hydra `skypilot_launch.yaml` entrypoint — composes a compute template name
# into a ComputeConfig via `compute_config_from_cfg`.
# ---------------------------------------------------------------------------


class TestSkypilotLaunchHydraEntrypoint:
    """Pin the Hydra-composable entrypoint that mirrors `configs/dataset.yaml`.

    The launcher's Click CLI consumes `--template <path>` directly today (CI workflows depend on
    the path-based flag); the Hydra entrypoint at `configs/skypilot_launch.yaml` is the parallel
    composition surface for future use cases that want to pick a compute template by name and drive
    overrides from a CLI like `compute=oci-cpu-template`.
    """

    def test_default_compose_resolves_to_runpod_template(self) -> None:
        """`compose(config_name='skypilot_launch')` defaults compute to runpod-template."""
        from hydra import compose, initialize_config_dir

        from synth_setter.pipeline.schemas.compute import compute_config_from_cfg

        config_dir = str(Path(__file__).resolve().parents[3] / "configs")
        compute_dir = Path(config_dir) / "compute"

        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = compose(config_name="skypilot_launch")

        result = compute_config_from_cfg(cfg, compute_dir=compute_dir)
        assert result.resources["cloud"] == "runpod"

    def test_compute_override_selects_different_template(self) -> None:
        """`compose(..., overrides=['compute_template=local-template'])` picks the k8s template."""
        from hydra import compose, initialize_config_dir

        from synth_setter.pipeline.schemas.compute import compute_config_from_cfg

        config_dir = str(Path(__file__).resolve().parents[3] / "configs")
        compute_dir = Path(config_dir) / "compute"

        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = compose(
                config_name="skypilot_launch",
                overrides=["compute_template=local-template"],
            )

        result = compute_config_from_cfg(cfg, compute_dir=compute_dir)
        assert result.resources["cloud"] == "kubernetes"


# ---------------------------------------------------------------------------
# Malformed-template handling: load failures in `main()` surface as ClickException.
# ---------------------------------------------------------------------------


class TestMainMalformedTemplate:
    """Errors loading or validating the compute YAML surface as a clean ClickException."""

    def test_malformed_yaml_surfaces_as_click_exception(  # noqa: DOC101,DOC103
        self,
        tmp_path: Path,
        experiment: str,
        fake_plugin: Path,
        env_file: Path,
        patch_materialize_io: None,
        local_spec_dir: Path,
        mock_sky: MagicMock,
    ) -> None:
        """A template missing required ComputeConfig fields fails with a wrapped error.

        Pydantic ``ValidationError`` is wrapped in ``click.ClickException`` so the user sees a
        one-line CLI error instead of a Pydantic traceback.
        """
        _ = mock_sky
        bad_template = tmp_path / "bad-template.yaml"
        # No `run:` field; ComputeConfig rejects this.
        bad_template.write_text(yaml.dump({"resources": {"cloud": "runpod"}, "envs": {}}))

        result = _invoke(
            experiment,
            bad_template,
            env_file,
            "--job-name",
            "smoke-job-bad",
            fake_plugin=fake_plugin,
        )

        assert result.exit_code != 0
        assert "failed to load compute template" in result.output
