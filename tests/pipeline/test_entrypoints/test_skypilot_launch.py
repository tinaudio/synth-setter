"""Tests for the SkyPilot launcher (RunPod / OCI / kind).

Covers ``src/synth_setter/pipeline/skypilot_launch.py``. Mock-based: no real SkyPilot or RunPod
calls. The ``mock_sky`` fixture replaces the launcher's module-level ``sky`` reference with a
``MagicMock`` so dispatch-side assertions can read submission shape without provisioning.

The launcher's click CLI no longer composes a DatasetSpec — it shells out to an operator-
supplied inner command (typically ``synth-setter-generate-dataset``) that writes the canonical
``data/<task>/<run>/metadata/input_spec.json``, parses that spec once, forwards
``spec.r2.input_spec_uri()`` as the canonical R2 URI, and dispatches via
``dispatch_via_skypilot``. The ``TestCli`` class pins the new contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import click
import pytest
import yaml
from click.testing import CliRunner

from synth_setter.pipeline.constants import WORKER_SPEC_URI_ENV
from synth_setter.pipeline.partitioning import NUM_WORKERS_ENV_VAR, WORKER_RANK_ENV_VAR
from synth_setter.pipeline.schemas.skypilot_launch import SkypilotLaunchConfig
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.skypilot_launch import (
    _SECRET_WORKER_ENV_KEYS,
    _SKYPILOT_API_SERVER_ENV,
    _SPEC_URI_STDOUT_SENTINEL,
    _WORKER_ENV_KEYS,
    _emit_spec_uri,
    _ensure_ci_sky_config,
    _override_image_id,
    dispatch_via_skypilot,
    load_worker_env,
    main,
    resolve_worker_env,
)
from synth_setter.pipeline.skypilot_launch import (
    _run_cred_bootstrap as _real_run_cred_bootstrap,
)
from synth_setter.resources import configs_dir


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
    return Path(str(configs_dir() / "compute" / "runpod-template.yaml"))


@pytest.fixture(autouse=True)
def clear_worker_env_from_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the worker env keys from the test process env.

    Without this, a developer with ``RCLONE_CONFIG_R2_*`` exported in their shell
    would silently satisfy ``resolve_worker_env``, masking tests that rely on a
    specific resolution path.
    """
    for key in _WORKER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def mock_cred_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-op ``_run_cred_bootstrap`` by default.

    Tests that exercise bootstrap behavior directly re-patch with a tracking or raising stub.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.skypilot_launch._run_cred_bootstrap",
        lambda **_kwargs: None,
    )


def _succeeded_run(mock_sky: MagicMock) -> None:
    """Configure ``mock_sky`` so jobs.launch + jobs.tail_logs + jobs.cancel all succeed.

    ``sky.jobs.tail_logs`` returns an int rc directly — 0 means SUCCEEDED.
    ``sky.jobs.launch`` returns a request_id whose ``stream_and_get`` yields
    ``(job_ids: List[int], handle)``.

    :param mock_sky: Mocked ``sky`` module from fixture.
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
    """Replace the launcher's module-level ``sky`` with a MagicMock pre-configured for success.

    Tests that need a different behavior tweak knobs on the returned mock (e.g. set
    ``mock_sky.jobs.tail_logs.return_value = 100`` for a worker failure, or
    ``mock_sky.jobs.tail_logs.side_effect = ...`` for a transport raise).
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
        """Skip blanks/comments and strip quotes when loading the dotenv file.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
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
        """Lines like ``BARE`` (no ``=``) come back as None from dotenv; loader filters them.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        path = tmp_path / ".env"
        path.write_text("FOO=bar\nBARE\n")
        assert load_worker_env(path) == {"FOO": "bar"}


class TestResolveWorkerEnvGitRefValidation:
    """``WORKER_GIT_REF``, when set, must be a 7-40 char hex git SHA.

    The validation lives at the env-resolution seam (host-side) instead of in the worker template's
    bash because the SHA is rendered into a ``git fetch + checkout`` invocation; rejecting a
    malformed value at the launcher gives a clear error before the job is ever submitted.
    """

    def test_unset_git_ref_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty/unset WORKER_GIT_REF is the common case (no PR-CI bake-lag bypass).

        :param monkeypatch: Pytest fixture for env/attribute mocking.
        """
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
        """7-40 char lowercase hex strings pass — both short and long form.

        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param good_sha: Parametrized 7-40 char lowercase hex git SHA.
        """
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
        """Non-SHA values fail with ClickException before the launcher provisions anything.

        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param bad_sha: Parametrized non-SHA value.
        """
        monkeypatch.setenv("WORKER_GIT_REF", bad_sha)
        with pytest.raises(click.ClickException, match="WORKER_GIT_REF"):
            resolve_worker_env(None)


class TestResolveWorkerEnvR2RemoteConstants:
    """Cover rclone-constant defaulting for the R2 type and provider keys.

    Targets ``RCLONE_CONFIG_R2_TYPE`` and ``RCLONE_CONFIG_R2_PROVIDER``. These are constants
    (not secrets) that rclone needs to construct the ``r2:`` remote. The launcher defaults them
    so workflows and ``.env`` files don't have to repeat them, while still allowing override for
    non-Cloudflare R2-compatible setups (e.g. self-hosted MinIO test rigs).
    """

    def test_type_and_provider_default_when_unset(self) -> None:
        """Without TYPE/PROVIDER in env or .env, the launcher fills the rclone constants."""
        resolved = resolve_worker_env(None)
        assert resolved["RCLONE_CONFIG_R2_TYPE"] == "s3"
        assert resolved["RCLONE_CONFIG_R2_PROVIDER"] == "Cloudflare"

    def test_type_override_from_env_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit override via process env is preserved (not clobbered by the default).

        :param monkeypatch: Pytest fixture for env/attribute mocking.
        """
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "s3-other")
        resolved = resolve_worker_env(None)
        assert resolved["RCLONE_CONFIG_R2_TYPE"] == "s3-other"

    def test_provider_override_from_env_file_wins(self, tmp_path: Path) -> None:
        """An explicit override via ``.env`` is preserved (not clobbered by the default).

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        env_file = tmp_path / ".env"
        env_file.write_text("RCLONE_CONFIG_R2_PROVIDER=Other\n")
        resolved = resolve_worker_env(env_file)
        assert resolved["RCLONE_CONFIG_R2_PROVIDER"] == "Other"


class TestEnsureCiSkyConfig:
    """``_ensure_ci_sky_config`` writes the managed-jobs shrink only when CI mode is on."""

    def test_no_op_when_env_var_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Operator local dev (no env var) leaves ``~/.sky/config.yaml`` untouched.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        """
        monkeypatch.delenv("SYNTH_SETTER_CI_MODE", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        _ensure_ci_sky_config()
        assert not (tmp_path / ".sky" / "config.yaml").exists()

    def test_writes_shrink_yaml_when_env_var_truthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SYNTH_SETTER_CI_MODE=1 → ``~/.sky/config.yaml`` carries the controller shrink.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        """
        monkeypatch.setenv("SYNTH_SETTER_CI_MODE", "1")
        monkeypatch.setenv("HOME", str(tmp_path))
        _ensure_ci_sky_config()
        config_path = tmp_path / ".sky" / "config.yaml"
        assert config_path.is_file()
        body = config_path.read_text(encoding="utf-8")
        # `cpus: 1+` / `memory: 1+` are the kind-allocatable floor that the
        # workflow step used to write; verify both still land in the file.
        assert "cpus: 1+" in body
        assert "memory: 1+" in body
        assert "controller:" in body

    def test_global_config_does_not_carry_pod_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``~/.sky/config.yaml`` must not set ``kubernetes.pod_config`` globally.

        A global ``imagePullPolicy: Never`` blocks the SkyPilot jobs
        controller (its GAR image is not ``kind load``-ed) — see #1255.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        """
        monkeypatch.setenv("SYNTH_SETTER_CI_MODE", "1")
        monkeypatch.setenv("HOME", str(tmp_path))
        _ensure_ci_sky_config()
        doc = yaml.safe_load((tmp_path / ".sky" / "config.yaml").read_text(encoding="utf-8"))
        assert "pod_config" not in doc.get("kubernetes", {})

    def test_idempotent_overwrite(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-invoking the helper is safe; the file is rewritten, not appended.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        """
        monkeypatch.setenv("SYNTH_SETTER_CI_MODE", "1")
        monkeypatch.setenv("HOME", str(tmp_path))
        sky_dir = tmp_path / ".sky"
        sky_dir.mkdir()
        (sky_dir / "config.yaml").write_text("stale: true\n", encoding="utf-8")
        _ensure_ci_sky_config()
        body = (sky_dir / "config.yaml").read_text(encoding="utf-8")
        assert "stale: true" not in body
        assert "cpus: 1+" in body

    @pytest.mark.parametrize("falsy", ["0", "false", "False", "no", "off", "", "  "])
    def test_no_op_for_explicit_falsy_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, falsy: str
    ) -> None:
        """Explicit falsy strings ("0", "false", …) leave ``~/.sky/config.yaml`` untouched.

        Prevents an operator who exports ``SYNTH_SETTER_CI_MODE=0`` from
        accidentally triggering the kind-shrink overwrite — matches the
        docstring promise that only truthy values activate the helper.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param falsy: Parametrized falsy string to set on ``SYNTH_SETTER_CI_MODE``.
        """
        monkeypatch.setenv("SYNTH_SETTER_CI_MODE", falsy)
        monkeypatch.setenv("HOME", str(tmp_path))
        _ensure_ci_sky_config()
        assert not (tmp_path / ".sky" / "config.yaml").exists()

    @pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "On"])
    def test_writes_for_explicit_truthy_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, truthy: str
    ) -> None:
        """Each accepted truthy spelling (case-insensitive) activates the shrink write.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param truthy: Parametrized truthy string to set on ``SYNTH_SETTER_CI_MODE``.
        """
        monkeypatch.setenv("SYNTH_SETTER_CI_MODE", truthy)
        monkeypatch.setenv("HOME", str(tmp_path))
        _ensure_ci_sky_config()
        assert (tmp_path / ".sky" / "config.yaml").is_file()


class TestLocalTemplatePodConfig:
    """``configs/compute/local-template.yaml`` carries the task-scoped pod_config override.

    The override must live on the user worker task — a global write would
    block the SkyPilot jobs controller from pulling its own image (#1255).
    """

    def test_image_pull_policy_never_is_scoped_to_task(self) -> None:
        """Top-level ``config:`` is task-scoped — SkyPilot merges it into the worker pod only."""
        template_path = Path(str(configs_dir() / "compute" / "local-template.yaml"))
        doc = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        containers = doc["config"]["kubernetes"]["pod_config"]["spec"]["containers"]
        assert containers == [{"imagePullPolicy": "Never"}]


class TestEmitSpecUri:
    """``_emit_spec_uri`` prints the canonical URI on a stdout sentinel line.

    The test-dataset-generation workflow greps the tee'd launcher log for this
    sentinel (replacing the previous host-side ``synth-setter-spec-uri`` re-
    invocation in bash, PR #1164).
    """

    def test_marker_format_is_stable(self, capsys: pytest.CaptureFixture[str]) -> None:
        """One-line ``::synth-setter-spec-uri::<uri>`` marker so the workflow grep is unambiguous.

        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        _emit_spec_uri("r2://intermediate-data/run/input_spec.json")
        captured = capsys.readouterr()
        assert (
            captured.out.strip()
            == f"{_SPEC_URI_STDOUT_SENTINEL}r2://intermediate-data/run/input_spec.json"
        )


# ---------------------------------------------------------------------------
# main — new click CLI (subprocess passthrough + spec discovery + dispatch)
# ---------------------------------------------------------------------------


def _build_spec(fake_plugin: Path) -> DatasetSpec:
    """Build a DatasetSpec wired to ``fake_plugin`` for dispatch tests.

    :param fake_plugin: Path passed through as ``render.plugin_path``.
    :return: A ``DatasetSpec`` ready for dispatch-path tests.
    """
    return DatasetSpec(
        task_name="test-dispatch",
        train_val_test_sizes=(10000, 0, 0),
        output_format="hdf5",
        base_seed=42,
        r2={"bucket": "intermediate-data"},  # type: ignore[arg-type]
        render={  # type: ignore[arg-type]
            "plugin_path": str(fake_plugin),
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 32,
            "samples_per_shard": 10000,
        },
    )


def _write_local_spec(cwd: Path, spec: DatasetSpec) -> Path:
    """Materialize ``spec`` at the canonical ``cwd/data/<task>/<run>/metadata/input_spec.json``.

    Mirrors what the inner ``synth-setter-generate-dataset`` command produces
    via ``spec_io.write_spec_locally``.

    :param cwd: Working directory under which ``data/`` is created.
    :param spec: DatasetSpec to materialize on disk.
    :returns: Path the spec was written to.
    """
    target = cwd / "data" / spec.task_name / spec.run_id / "metadata" / "input_spec.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return target


@pytest.fixture()
def cwd_with_spec(
    tmp_path: Path,
    fake_plugin: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, DatasetSpec]:
    """Tmp dir pre-seeded with a canonical input_spec.json and a redirected discovery anchor.

    The launcher reads from ``_LOCAL_DATA_DIR`` (anchored at ``REPO_ROOT/data``).
    Tests monkeypatch that anchor to ``tmp_path/data`` so each test is hermetic.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param fake_plugin: Fixture-provided fake VST3 plugin path.
    :param monkeypatch: Pytest fixture for env/attribute mocking.
    :returns: Tuple of ``(cwd, spec_path, spec)``.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("synth_setter.pipeline.skypilot_launch._LOCAL_DATA_DIR", tmp_path / "data")
    spec = _build_spec(fake_plugin)
    spec_path = _write_local_spec(tmp_path, spec)
    return tmp_path, spec_path, spec


class TestCli:
    """End-to-end CLI behavior on the new ``main(command, ...)`` signature.

    The CLI is a thin orchestrator: run the inner command, find the spec it
    just wrote, parse it once, and dispatch with ``spec.r2.input_spec_uri()``
    as ``WORKER_SPEC_URI``.
    """

    def test_requires_inner_command(
        self, env_file: Path, template_yaml: Path, mock_sky: MagicMock
    ) -> None:
        """No trailing command argument → click usage error before any subprocess call.

        :param env_file: Fixture-provided worker env file path.
        :param template_yaml: Fixture-provided compute template path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--template", str(template_yaml), "--env-file", str(env_file)],
        )
        assert result.exit_code != 0
        mock_sky.jobs.launch.assert_not_called()

    @pytest.mark.parametrize(
        "inner",
        [
            ["synth-setter-generate-dataset", "experiment=foo"],
            ["/venv/main/bin/synth-setter-generate-dataset", "experiment=foo"],
            ["python", "-m", "synth_setter.cli.generate_dataset", "experiment=foo"],
            ["python3", "-m", "synth_setter.cli.generate_dataset", "experiment=foo"],
        ],
        ids=[
            "bare-console-script",
            "absolute-path-to-console-script",
            "python -m module",
            "python3 -m module",
        ],
    )
    def test_rejects_dispatch_owning_inner_command(
        self,
        inner: list[str],
        env_file: Path,
        template_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
    ) -> None:
        """Reject ``synth-setter-generate-dataset`` (and its python -m form) before subprocess.

        The launcher's verbatim worker re-execution would otherwise either
        re-materialize a fresh spec on each worker or attempt to dispatch a
        second time. Catching the misuse at the CLI surface is cheaper than
        debugging it from a failed managed-job run.

        :param inner: Parametrized inner-command argv covering bare console
            script, absolute-path console script, and ``python(3) -m`` forms.
        :param env_file: Fixture-provided worker env file path.
        :param template_yaml: Fixture-provided compute template path.
        :param monkeypatch: Used to pin ``subprocess.check_call`` so an
            accidental fall-through to subprocess execution would be visible
            as a test failure rather than running the real entrypoint.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        check_call_calls: list[list[str]] = []
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            lambda args, **_kwargs: (check_call_calls.append(list(args)), 0)[1],
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
                "--",
                *inner,
            ],
        )

        assert result.exit_code != 0
        assert "synth-setter-generate-dataset" in result.output
        assert "skypilot_launch.compute_template" in result.output
        assert check_call_calls == []
        mock_sky.jobs.launch.assert_not_called()

    def test_allows_non_dispatch_owning_inner_command(
        self,
        cwd_with_spec: tuple[Path, Path, DatasetSpec],
        env_file: Path,
        template_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
    ) -> None:
        """A ``synth-setter-*`` entry point that does NOT own dispatch is accepted.

        ``synth-setter-spec-uri`` is a read-only console script (it emits the
        canonical R2 URI for an input_spec and does not call
        ``dispatch_via_skypilot``), so it must pass the guardrail. This pins
        the guardrail to the actual ``_DISPATCH_OWNING_ENTRYPOINTS`` set
        rather than the broader ``synth-setter-*`` prefix.

        :param cwd_with_spec: Fixture providing a CWD pre-populated with a
            canonical input_spec.json.
        :param env_file: Fixture-provided worker env file path.
        :param template_yaml: Fixture-provided compute template path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        check_call_calls: list[list[str]] = []
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            lambda args, **_kwargs: (check_call_calls.append(list(args)), 0)[1],
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
                "--",
                "synth-setter-spec-uri",
                "some-arg",
            ],
        )

        assert result.exit_code == 0, result.output
        assert check_call_calls == [["synth-setter-spec-uri", "some-arg"]]

    def test_runs_inner_command_via_subprocess(
        self,
        cwd_with_spec: tuple[Path, Path, DatasetSpec],
        env_file: Path,
        template_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
    ) -> None:
        """The CLI delegates spec materialization to a subprocess.check_call of ``command``.

        :param cwd_with_spec: Fixture providing a CWD pre-populated with a canonical input_spec.json.
        :param env_file: Fixture-provided worker env file path.
        :param template_yaml: Fixture-provided compute template path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        # The cwd_with_spec fixture pre-materializes the spec so a no-op check_call
        # leaves the discovery path satisfied; the assertion is purely about which
        # argv the launcher forwarded.
        check_call_calls: list[list[str]] = []
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            lambda args, **_kwargs: (check_call_calls.append(list(args)), 0)[1],
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
                "--",
                "materialize-input-spec",
                "experiment=foo",
            ],
        )
        assert result.exit_code == 0, result.output
        assert check_call_calls == [["materialize-input-spec", "experiment=foo"]]

    def test_spec_uri_is_threaded_into_worker_env(
        self,
        cwd_with_spec: tuple[Path, Path, DatasetSpec],
        env_file: Path,
        template_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
    ) -> None:
        """``spec.r2.input_spec_uri()`` of the discovered spec lands in the worker's env.

        :param cwd_with_spec: Fixture providing a CWD pre-populated with a canonical input_spec.json.
        :param env_file: Fixture-provided worker env file path.
        :param template_yaml: Fixture-provided compute template path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        _, _, spec = cwd_with_spec
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            lambda _args, **_kwargs: 0,
        )
        expected_uri = spec.r2.input_spec_uri()

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
                "--",
                "materialize-input-spec",
            ],
        )
        assert result.exit_code == 0, result.output
        update_envs = mock_sky.Task.from_yaml_config.return_value.update_envs.call_args_list
        assert update_envs, "dispatch_via_skypilot should have updated worker env"
        forwarded = update_envs[0].args[0]
        assert forwarded[WORKER_SPEC_URI_ENV] == expected_uri

    def test_no_spec_under_data_dir_fails_clearly(
        self,
        tmp_path: Path,
        env_file: Path,
        template_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
    ) -> None:
        """No spec under data/ → fail loudly rather than dispatch a broken job.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param template_yaml: Fixture-provided compute template path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch._LOCAL_DATA_DIR", tmp_path / "data"
        )
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            lambda _args, **_kwargs: 0,
        )
        # No check_output stub — we expect to fail before reaching it.

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
                "--",
                "materialize-input-spec",
            ],
        )
        assert result.exit_code != 0
        assert "no input_spec.json" in result.output
        mock_sky.jobs.launch.assert_not_called()

    def test_multiple_specs_under_data_dir_fails_clearly(
        self,
        cwd_with_spec: tuple[Path, Path, DatasetSpec],
        env_file: Path,
        template_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
        fake_plugin: Path,
    ) -> None:
        """Multiple stale runs in ``data/`` is ambiguous → fail loudly.

        :param cwd_with_spec: Fixture providing a CWD pre-populated with a canonical input_spec.json.
        :param env_file: Fixture-provided worker env file path.
        :param template_yaml: Fixture-provided compute template path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        """
        cwd, _, _ = cwd_with_spec
        # Add a second spec under a different task/run.
        second = cwd / "data" / "second-task" / "second-run" / "metadata" / "input_spec.json"
        second.parent.mkdir(parents=True)
        second.write_text(_build_spec(fake_plugin).model_dump_json())

        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            lambda _args, **_kwargs: 0,
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
                "--",
                "materialize-input-spec",
            ],
        )
        assert result.exit_code != 0
        assert "expected exactly one" in result.output
        mock_sky.jobs.launch.assert_not_called()

    def test_inner_command_failure_propagates(
        self,
        env_file: Path,
        template_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
    ) -> None:
        """Inner subprocess non-zero rc → CalledProcessError propagates; no dispatch.

        :param env_file: Fixture-provided worker env file path.
        :param template_yaml: Fixture-provided compute template path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        import subprocess as _subprocess

        def _raising(args: list[str], **_kwargs: Any) -> int:
            raise _subprocess.CalledProcessError(returncode=2, cmd=args)

        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            _raising,
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
                "--",
                "materialize-input-spec",
            ],
        )
        assert result.exit_code != 0
        mock_sky.jobs.launch.assert_not_called()

    @pytest.mark.parametrize(
        "raise_type",
        [ValueError, RuntimeError],
    )
    def test_dispatch_error_surfaces_as_click_exception(
        self,
        cwd_with_spec: tuple[Path, Path, DatasetSpec],
        env_file: Path,
        template_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
        raise_type: type[Exception],
    ) -> None:
        """Surface ``dispatch_via_skypilot`` errors as a clean ``click.ClickException``.

        ``dispatch_via_skypilot`` raises ``ValueError`` for cfg-shape errors and ``RuntimeError`` for
        worker submission failures. Both should reach the operator as a one-line ``click.ClickException``
        rather than as an uncaught traceback.

        :param cwd_with_spec: Fixture providing a CWD pre-populated with a canonical input_spec.json.
        :param env_file: Fixture-provided worker env file path.
        :param template_yaml: Fixture-provided compute template path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
        :param raise_type: Parametrized exception class to simulate from dispatch.
        """

        def _no_op(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.check_call",
            _no_op,
        )

        def _raising_dispatch(*_args: Any, **_kwargs: Any) -> None:
            raise raise_type("simulated dispatch failure")

        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.dispatch_via_skypilot",
            _raising_dispatch,
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--template",
                str(template_yaml),
                "--env-file",
                str(env_file),
                "--",
                "materialize-input-spec",
            ],
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, SystemExit), (
            f"expected SystemExit (ClickException) but got "
            f"{type(result.exception).__name__ if result.exception else 'None'}"
        )
        assert "simulated dispatch failure" in result.output

    def test_drops_old_experiment_and_hydra_override_surface(self) -> None:
        """The breaking-change PR drops ``--experiment``, ``--spec-out``, ``--job-name`` flags.

        Callers must rewrite to the new positional-command form. Pinning the option surface here
        protects against accidental re-introduction.
        """
        # Click stores recognized options on ``main.params``.
        option_flags = {decl for param in main.params for decl in getattr(param, "opts", ())}
        for dropped in ("--experiment", "--spec-out", "--job-name", "--cluster-name"):
            assert dropped not in option_flags, (
                f"{dropped} must not be a recognized launcher option after PR-5"
            )


# ---------------------------------------------------------------------------
# _override_image_id — per-backend image_id mutation
# ---------------------------------------------------------------------------


class TestOverrideImageId:
    """Per-backend ``image_id`` mutation in ``_override_image_id``.

    Direct unit tests on the helper, independent of the CLI path. RunPod (and any non-OCI cloud)
    accepts ``image_id: docker:<image>``; OCI's backend rejects it and runs the worker via a
    sub-docker invocation in the YAML's ``run:`` block, so OCI Resources must be left untouched.
    """

    @staticmethod
    def _make_resource(cloud: object) -> MagicMock:
        """Fake ``sky.Resources`` with a ``.cloud`` attr and a ``.copy()`` that records image_id.

        :param cloud: Pytest fixture.
        """
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
        """Fake ``sky.Task`` carrying ``resources`` (as a list, so ``type(...)`` is ``list``).

        :param resources: Pytest fixture.
        """
        task = MagicMock(spec=["resources", "set_resources"])
        task.resources = list(resources)
        return task

    def test_non_oci_resource_gets_image_id_overridden(self) -> None:
        """Non-OCI Resources entry gets ``image_id`` set to ``docker:<worker_image>``."""
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

        The helper always rebuilds ``task.resources`` from the original entries, so it may
        call ``set_resources``; what matters behaviorally is that the OCI entry is never
        copied with a new image_id and is preserved verbatim in the rebuilt list.

        :param monkeypatch: Pytest fixture for env/attribute mocking.
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
# _run_cred_bootstrap — invokes the script; honors SKYPILOT_API_SERVER_ENDPOINT
# ---------------------------------------------------------------------------


class TestRunCredBootstrap:
    """Behavioral contracts for the launcher's wrapping of the cred-bootstrap script."""

    def test_skips_when_api_server_endpoint_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the bootstrap script is NOT invoked when SKYPILOT_API_SERVER_ENDPOINT is set.

        The remote API server holds creds. Returns silently (no exception, no script call).

        :param monkeypatch: Pytest fixture for env/attribute mocking.
        """
        monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://api:pw@server/")
        called: list[str] = []
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.subprocess.run",
            lambda *args, **kwargs: called.append("invoked"),  # type: ignore[misc]
        )

        _real_run_cred_bootstrap(provider="runpod")
        assert called == [], "bootstrap script should not be invoked in remote-server mode"

    def test_passes_merged_env_to_subprocess(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify ``--env-file`` values are merged into the bootstrap subprocess env.

        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param tmp_path: Pytest fixture providing a fresh test directory.
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

        :param monkeypatch: Pytest fixture for env/attribute mocking.
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
        prevents a tee'd workflow caller from leaking it.

        :param monkeypatch: Pytest fixture for env/attribute mocking.
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

        _real_run_cred_bootstrap(provider="runpod")
        assert captured["kwargs"].get("capture_output") is True
        assert "env" in captured["kwargs"]


# ---------------------------------------------------------------------------
# _SECRET_WORKER_ENV_KEYS — module-level constant for the unconfigured-creds check
# ---------------------------------------------------------------------------


class TestSecretWorkerEnvKeys:
    """``_SECRET_WORKER_ENV_KEYS`` is the residual subset used to detect unconfigured creds."""

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
        assert "RCLONE_CONFIG_R2_ACCESS_KEY_ID" in _SECRET_WORKER_ENV_KEYS
        assert "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY" in _SECRET_WORKER_ENV_KEYS
        assert "RCLONE_CONFIG_R2_ENDPOINT" in _SECRET_WORKER_ENV_KEYS

    def test_is_subset_of_worker_env_keys(self) -> None:
        """The secret subset is closed-form derived from ``_WORKER_ENV_KEYS``."""
        assert set(_SECRET_WORKER_ENV_KEYS).issubset(set(_WORKER_ENV_KEYS))


# ---------------------------------------------------------------------------
# dispatch_via_skypilot — programmatic launcher surface used by the CLI
# ---------------------------------------------------------------------------


def _write_runpod_yaml(
    tmp_path: Path,
    *,
    include_run: bool = False,
    run_body: str | None = None,
) -> Path:
    """Write a minimal RunPod-shaped compute template.

    ``include_run=True`` adds a default ``run:`` block (``echo existing``).
    ``run_body`` overrides the run body — pass a multiline string with
    ``${WORKER_CMD}`` to exercise the sentinel-substitution path.

    :param tmp_path: Directory under which ``compute.yaml`` is written.
    :param include_run: When ``True``, add a default ``run:`` block.
    :param run_body: Override the run body verbatim (multiline allowed).
    :return: Path to the written ``compute.yaml``.
    """
    yaml_text = (
        "resources:\n"
        "  cloud: runpod\n"
        "  accelerators: {RTX3070: 1}\n"
        "envs:\n"
        "  RCLONE_CONFIG_R2_TYPE: ''\n"
    )
    if include_run or run_body is not None:
        body = run_body if run_body is not None else "echo existing"
        indented = "\n".join(f"  {line}" for line in body.splitlines())
        yaml_text += f"run: |\n{indented}\n"
    path = tmp_path / "compute.yaml"
    path.write_text(yaml_text)
    return path


# Sentinel spec URI passed as the dispatch kwarg in tests.
_DISPATCH_SPEC_URI = "r2://intermediate-data/data/run/input_spec.json"


class TestLoadComputeTemplateWithCmd:
    """``_load_compute_template_with_cmd`` injects cmd as run and rejects pre-existing runs."""

    def test_cmd_is_injected_when_yaml_has_no_run(self, tmp_path: Path) -> None:
        """Without a pre-existing run: block, the loaded doc's run: equals cmd.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        from synth_setter.pipeline.skypilot_launch import _load_compute_template_with_cmd

        template = _write_runpod_yaml(tmp_path, include_run=False)
        doc = _load_compute_template_with_cmd(template, "echo hello")
        assert doc["run"] == "echo hello"

    def test_existing_run_block_without_sentinel_raises(self, tmp_path: Path) -> None:
        """A pre-existing run: with no sentinel + non-empty cmd is a conflict, not a silent override.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        from synth_setter.pipeline.skypilot_launch import _load_compute_template_with_cmd

        template = _write_runpod_yaml(tmp_path, include_run=True)
        with pytest.raises(ValueError, match="has a non-empty `run:` block"):
            _load_compute_template_with_cmd(template, "echo hello")

    def test_sentinel_in_run_block_substitutes_cmd(self, tmp_path: Path) -> None:
        """A template with ${WORKER_CMD} in run: substitutes cmd; scaffolding survives.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        from synth_setter.pipeline.skypilot_launch import _load_compute_template_with_cmd

        template = _write_runpod_yaml(
            tmp_path,
            run_body='sudo docker run --rm "$WORKER_IMAGE" bash -c "${WORKER_CMD}"',
        )
        doc = _load_compute_template_with_cmd(template, "echo hello && exec foo")
        assert isinstance(doc["run"], str)
        assert "${WORKER_CMD}" not in doc["run"]
        assert "echo hello && exec foo" in doc["run"]
        assert doc["run"].startswith("sudo docker run --rm")
        assert doc["run"].rstrip().endswith('"echo hello && exec foo"')

    def test_non_string_run_block_raises(self, tmp_path: Path) -> None:
        """A non-string run: (e.g. a list) is a malformed template, raise before substitute.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        from synth_setter.pipeline.skypilot_launch import _load_compute_template_with_cmd

        path = tmp_path / "bad_run.yaml"
        path.write_text("resources:\n  cloud: runpod\nrun:\n  - echo\n  - bad\n")
        with pytest.raises(ValueError, match="`run:` must be a string"):
            _load_compute_template_with_cmd(path, "x")

    def test_missing_template_raises_file_not_found(self, tmp_path: Path) -> None:
        """Mistyped path surfaces a FileNotFoundError, not a confusing parse error downstream.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        from synth_setter.pipeline.skypilot_launch import _load_compute_template_with_cmd

        with pytest.raises(FileNotFoundError):
            _load_compute_template_with_cmd(tmp_path / "missing.yaml", "x")

    def test_non_mapping_top_level_raises(self, tmp_path: Path) -> None:
        """A YAML whose top level is a list, not a mapping, is rejected at load time.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        from synth_setter.pipeline.skypilot_launch import _load_compute_template_with_cmd

        path = tmp_path / "bad.yaml"
        path.write_text("- not\n- a\n- mapping\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            _load_compute_template_with_cmd(path, "x")


class TestDetectProviderFromDoc:
    """``_detect_provider_from_doc`` maps a parsed compute YAML to a cred-bootstrap provider."""

    @pytest.mark.parametrize(
        "doc, expected_provider",
        [
            ({"resources": {"cloud": "runpod"}}, "runpod"),
            ({"resources": {"any_of": [{"cloud": "oci"}]}}, "oci"),
            ({"resources": {"cloud": "kubernetes"}}, "local"),
            ({"resources": {"cloud": "k8s"}}, "local"),
            ({"resources": {"cloud": "RunPod"}}, "runpod"),
        ],
        ids=["flat-runpod", "any-of-oci", "kubernetes-as-local", "k8s-alias", "case-insensitive"],
    )
    def test_supported_clouds_map_to_provider(
        self,
        tmp_path: Path,
        doc: dict[str, object],
        expected_provider: str,
    ) -> None:
        """Each supported ``resources.cloud`` shape maps to the expected cred-bootstrap provider.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param doc: Parametrized parsed-YAML mapping under test.
        :param expected_provider: Parametrized expected provider name.
        """
        from synth_setter.pipeline.skypilot_launch import _detect_provider_from_doc

        assert _detect_provider_from_doc(doc, source=tmp_path / "x.yaml") == expected_provider

    def test_unknown_cloud_raises(self, tmp_path: Path) -> None:
        """An unsupported cloud surfaces as a ValueError naming the offending value.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        from synth_setter.pipeline.skypilot_launch import _detect_provider_from_doc

        doc: dict[str, object] = {"resources": {"cloud": "aws"}}
        with pytest.raises(ValueError, match="Unsupported cloud"):
            _detect_provider_from_doc(doc, source=tmp_path / "x.yaml")


class TestWorkerSpecUriEnvConstant:
    """``WORKER_SPEC_URI_ENV`` is the canonical public env-var name for worker spec URIs."""

    def test_constant_exposed_publicly_from_pipeline_constants(self) -> None:
        """Public constant matches the legacy env-var name used by the worker."""
        assert WORKER_SPEC_URI_ENV == "WORKER_SPEC_URI"


class TestDispatchViaSkypilot:
    """``dispatch_via_skypilot`` rejects degenerate cfgs and threads per-rank fanout through."""

    def test_missing_compute_template_raises(self, fake_plugin: Path) -> None:
        """``compute_template=None`` is the "don't dispatch" sentinel — calling here is a bug.

        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        """
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(compute_template=None, cmd="echo")
        with pytest.raises(ValueError, match="compute_template"):
            dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

    def test_missing_cmd_raises(self, tmp_path: Path, fake_plugin: Path) -> None:
        """No cmd → no run block on the worker → we refuse to launch a no-op task.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        """
        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(compute_template=str(template), cmd=None)
        with pytest.raises(ValueError, match="cmd"):
            dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

    def test_yaml_run_block_conflicts_with_cmd(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        mock_sky: MagicMock,
    ) -> None:
        """End-to-end conflict guard: YAML run + sky_cfg.cmd raises before any SkyPilot side effect.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path, include_run=True)
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(compute_template=str(template), cmd="echo")
        with pytest.raises(ValueError, match="has a non-empty `run:` block"):
            dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)
        mock_sky.jobs.launch.assert_not_called()

    def test_missing_worker_env_raises(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No rclone creds in env → fail loudly rather than launching a task that can't upload.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        """
        for key in _SECRET_WORKER_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="exec synth-setter-generate-dataset-from-hydra experiment=foo",
            env_file=None,
        )
        with pytest.raises(ValueError, match="No worker env vars resolved"):
            dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

    @pytest.mark.parametrize(
        "kwargs_overrides, match",
        [
            ({"compute_template": None}, "compute_template"),
            ({"cmd": None}, "cmd"),
            ({"api_server": "https://api.example", "local": True}, "mutually exclusive"),
            ({"job_name": "has/slash"}, "job_name must match"),
            ({"worker_image_tag": "bad tag"}, "worker_image_tag must match"),
            ({"env_file": None}, "No worker env vars"),
        ],
        ids=[
            "missing-compute-template",
            "missing-cmd",
            "api-server-and-local",
            "bad-job-name",
            "bad-worker-image-tag",
            "missing-creds",
        ],
    )
    def test_phase1_failures_skip_phase2_side_effects(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        env_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
        capsys: pytest.CaptureFixture[str],
        kwargs_overrides: dict[str, object],
        match: str,
    ) -> None:
        """Phase-1 raises leave every Phase-2 side effect untouched.

        Probes all four Phase-2 mutations: ``~/.sky/config.yaml`` write,
        ``_SKYPILOT_API_SERVER_ENV`` set in process env, ``sky.jobs.launch``
        called, and the ``::synth-setter-spec-uri::`` stdout sentinel. A
        regression that promotes any one of them above the cred check or
        any other Phase-1 validator trips this test on every parametrized
        failure mode.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param env_file: Fixture-provided worker env file path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
        :param capsys: Pytest fixture capturing stdout/stderr.
        :param kwargs_overrides: ``SkypilotLaunchConfig`` overrides that trip
            exactly one Phase-1 validator.
        :param match: Regex expected in the ``ValueError`` message.
        """
        # ~/.sky must resolve under tmp_path so the file-write probe is hermetic.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("SYNTH_SETTER_CI_MODE", "1")
        # Only the "missing-creds" case depends on env state; others trip on cfg shape.
        for key in _SECRET_WORKER_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(_SKYPILOT_API_SERVER_ENV, raising=False)

        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        kwargs: dict[str, object] = {
            "compute_template": str(template),
            "cmd": "echo",
            "env_file": str(env_file),
            "job_name": "ok-name",
        }
        kwargs.update(kwargs_overrides)
        sky_cfg = SkypilotLaunchConfig(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match=match):
            dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

        assert not (tmp_path / ".sky").exists()
        assert _SKYPILOT_API_SERVER_ENV not in os.environ
        mock_sky.jobs.launch.assert_not_called()
        assert _SPEC_URI_STDOUT_SENTINEL not in capsys.readouterr().out

    def test_sentinel_does_not_emit_when_cred_bootstrap_raises(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        env_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A ``_run_cred_bootstrap`` raise must skip the sentinel + ``sky.jobs.launch``.

        Pins the Phase-2 ordering invariant ``_emit_spec_uri`` runs *after*
        ``_run_cred_bootstrap`` so a CI workflow that greps the sentinel out of
        the launcher log can treat it as proof the cred bootstrap succeeded —
        not just that Phase-1 cleared.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param env_file: Fixture-provided worker env file path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch._run_cred_bootstrap",
            MagicMock(side_effect=RuntimeError("simulated bootstrap failure")),
        )

        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="bootstrap-raise",
        )

        with pytest.raises(RuntimeError, match="simulated bootstrap failure"):
            dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

        assert _SPEC_URI_STDOUT_SENTINEL not in capsys.readouterr().out
        mock_sky.jobs.launch.assert_not_called()

    def test_end_to_end_dispatch_uses_cmd_as_run_block(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Happy-path dispatch: sky.Task.from_yaml_config receives a doc whose ``run`` is sky_cfg.cmd.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        cmd = "exec synth-setter-generate-dataset-from-hydra experiment=foo"
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd=cmd,
            env_file=str(env_file),
            job_name="dispatch-smoke",
        )

        dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

        mock_sky.Task.from_yaml_config.assert_called()
        passed_doc = mock_sky.Task.from_yaml_config.call_args.args[0]
        assert passed_doc["run"] == cmd

    def test_dispatch_failure_raises_runtime_error(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """A non-success tail rc surfaces as a RuntimeError naming the failed rank.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        mock_sky.stream_and_get.side_effect = RuntimeError("boom")

        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="dispatch-fail",
        )

        with pytest.raises(RuntimeError, match="worker.* failed"):
            dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

    def test_multi_worker_fans_out_one_task_per_rank(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """``num_workers=N`` builds N tasks with -rN job-name suffixes, per the fan-out contract.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="fan-out",
            num_workers=3,
        )

        dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

        assert mock_sky.Task.from_yaml_config.call_count == 3
        submitted_names = sorted(
            call.kwargs["name"] for call in mock_sky.jobs.launch.call_args_list
        )
        assert submitted_names == ["fan-out-r0", "fan-out-r1", "fan-out-r2"]
        ranks_seen = sorted(
            call.args[0][WORKER_RANK_ENV_VAR]
            for call in mock_sky.Task.from_yaml_config.return_value.update_envs.call_args_list
        )
        assert ranks_seen == ["0", "1", "2"]
        for call in mock_sky.Task.from_yaml_config.return_value.update_envs.call_args_list:
            assert call.args[0][NUM_WORKERS_ENV_VAR] == "3"

    def test_single_worker_dispatch_still_injects_rank_world_env(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Single-worker dispatch still injects explicit rank=0 / world=1 partition env.

        The worker-side fallback in ``read_rank_world_from_env`` is a local-mode
        convenience; the launcher must keep injecting the explicit partition env
        on every dispatch — otherwise a future ``num_workers=N`` regression that
        drops the injection would silently fall back to single-worker and
        duplicate every shard across every node (#763).

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="single-worker-env",
        )

        dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

        update_envs_calls = mock_sky.Task.from_yaml_config.return_value.update_envs.call_args_list
        assert len(update_envs_calls) == 1
        injected = update_envs_calls[0].args[0]
        assert injected[WORKER_RANK_ENV_VAR] == "0"
        assert injected[NUM_WORKERS_ENV_VAR] == "1"

    @pytest.mark.parametrize(
        "field, value, match",
        [
            ("job_name", "has/slash", "job_name must match"),
            ("worker_image_tag", "bad tag", "worker_image_tag must match OCI"),
        ],
        ids=["job-name-with-slash", "image-tag-with-space"],
    )
    def test_input_validation_raises_before_disk_or_network(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        env_file: Path,
        mock_sky: MagicMock,
        field: str,
        value: str,
        match: str,
    ) -> None:
        """Malformed launcher params surface as ValueError before any SkyPilot submission.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        :param field: Parametrized launcher-config field under test.
        :param value: Parametrized malformed value for ``field``.
        :param match: Parametrized regex expected in the raised error.
        """
        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        kwargs: dict[str, object] = {
            "compute_template": str(template),
            "cmd": "echo",
            "env_file": str(env_file),
            "job_name": "ok-name",
            field: value,
        }
        sky_cfg = SkypilotLaunchConfig(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match=match):
            dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)
        mock_sky.jobs.launch.assert_not_called()

    def test_job_name_falls_back_to_task_name_prefix_when_unset(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """job_name=None derives the synth-setter-smoke-<task_name[:8]> fallback.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name=None,
        )

        dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

        submitted = mock_sky.jobs.launch.call_args.kwargs["name"]
        assert submitted.startswith("synth-setter-smoke-")
        assert submitted.endswith(spec.task_name[:8])

    def test_api_server_and_local_are_mutually_exclusive(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Setting both api_server and local raises before any launch — opposite dispatch modes.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            api_server="https://api.example.com",
            local=True,
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)
        mock_sky.jobs.launch.assert_not_called()

    def test_spec_uri_kwarg_injected_verbatim_into_worker_env(
        self,
        tmp_path: Path,
        fake_plugin: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """The kwarg-supplied ``spec_uri`` lands in worker env verbatim.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param fake_plugin: Fixture-provided fake VST3 plugin path.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        spec = _build_spec(fake_plugin)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="kwarg-spec-uri",
        )

        dispatch_via_skypilot(spec, sky_cfg, spec_uri=_DISPATCH_SPEC_URI)

        update_envs_calls = mock_sky.Task.from_yaml_config.return_value.update_envs.call_args_list
        assert len(update_envs_calls) == 1
        forwarded = update_envs_calls[0].args[0]
        assert forwarded[WORKER_SPEC_URI_ENV] == _DISPATCH_SPEC_URI
