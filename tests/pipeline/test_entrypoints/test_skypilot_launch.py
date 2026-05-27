"""Tests for the SkyPilot launcher (RunPod / OCI / kind).

Covers ``src/synth_setter/pipeline/skypilot_launch.py``. Mock-based: no real SkyPilot or RunPod
calls. The ``mock_sky`` fixture replaces the launcher's module-level ``sky`` reference with a
``MagicMock`` so dispatch-side assertions can read submission shape without provisioning.

``dispatch_via_skypilot(sky_cfg)`` is the only public surface; the tests exercise the validation
funnel, the per-rank fan-out, and the uuid-stem job-name fallback.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import click
import pytest
import yaml

from synth_setter.pipeline.constants import WORKER_SPEC_URI_ENV
from synth_setter.pipeline.partitioning import NUM_WORKERS_ENV_VAR, WORKER_RANK_ENV_VAR
from synth_setter.pipeline.schemas.skypilot_launch import SkypilotLaunchConfig
from synth_setter.pipeline.skypilot_launch import (
    _SECRET_WORKER_ENV_KEYS,
    _SKYPILOT_API_SERVER_ENV,
    _WORKER_ENV_KEYS,
    _ensure_ci_sky_config,
    _override_image_id,
    dispatch_via_skypilot,
    load_worker_env,
    resolve_worker_env,
)
from synth_setter.pipeline.skypilot_launch import (
    _run_cred_bootstrap as _real_run_cred_bootstrap,
)
from synth_setter.resources import configs_dir


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

    The override must live on the user worker task — a global write would block the SkyPilot jobs
    controller from pulling its own image (#1255).
    """

    def test_image_pull_policy_never_is_scoped_to_task(self) -> None:
        """Top-level ``config:`` is task-scoped — SkyPilot merges it into the worker pod only."""
        template_path = Path(str(configs_dir() / "compute" / "local-template.yaml"))
        doc = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        containers = doc["config"]["kubernetes"]["pod_config"]["spec"]["containers"]
        assert containers == [{"imagePullPolicy": "Never"}]


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

    def test_missing_compute_template_raises(self) -> None:
        """``compute_template=None`` is the "don't dispatch" sentinel — calling here is a bug."""
        sky_cfg = SkypilotLaunchConfig(compute_template=None, cmd="echo")
        with pytest.raises(ValueError, match="compute_template"):
            dispatch_via_skypilot(sky_cfg)

    def test_missing_cmd_raises(self, tmp_path: Path) -> None:
        """No cmd → no run block on the worker → we refuse to launch a no-op task.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(compute_template=str(template), cmd=None)
        with pytest.raises(ValueError, match="cmd"):
            dispatch_via_skypilot(sky_cfg)

    def test_yaml_run_block_conflicts_with_cmd(
        self,
        tmp_path: Path,
        mock_sky: MagicMock,
    ) -> None:
        """End-to-end conflict guard: YAML run + sky_cfg.cmd raises before any SkyPilot side effect.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path, include_run=True)
        sky_cfg = SkypilotLaunchConfig(compute_template=str(template), cmd="echo")
        with pytest.raises(ValueError, match="has a non-empty `run:` block"):
            dispatch_via_skypilot(sky_cfg)
        mock_sky.jobs.launch.assert_not_called()

    def test_missing_worker_env_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No rclone creds in env → fail loudly rather than launching a task that can't upload.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        """
        for key in _SECRET_WORKER_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="exec synth-setter-generate-dataset-from-hydra experiment=foo",
            env_file=None,
        )
        with pytest.raises(ValueError, match="No worker env vars resolved"):
            dispatch_via_skypilot(sky_cfg)

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
        env_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
        kwargs_overrides: dict[str, object],
        match: str,
    ) -> None:
        """Phase-1 raises leave every Phase-2 side effect untouched.

        Probes the three Phase-2 mutations: ``~/.sky/config.yaml`` write,
        ``_SKYPILOT_API_SERVER_ENV`` set in process env, and ``sky.jobs.launch``
        called.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
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
        kwargs: dict[str, object] = {
            "compute_template": str(template),
            "cmd": "echo",
            "env_file": str(env_file),
            "job_name": "ok-name",
        }
        kwargs.update(kwargs_overrides)
        sky_cfg = SkypilotLaunchConfig(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match=match):
            dispatch_via_skypilot(sky_cfg)

        assert not (tmp_path / ".sky").exists()
        assert _SKYPILOT_API_SERVER_ENV not in os.environ
        mock_sky.jobs.launch.assert_not_called()

    def test_cred_bootstrap_raise_skips_launch(
        self,
        tmp_path: Path,
        env_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_sky: MagicMock,
    ) -> None:
        """A ``_run_cred_bootstrap`` raise propagates without reaching ``sky.jobs.launch``.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param monkeypatch: Pytest fixture for env/attribute mocking.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch._run_cred_bootstrap",
            MagicMock(side_effect=RuntimeError("simulated bootstrap failure")),
        )

        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="bootstrap-raise",
        )

        with pytest.raises(RuntimeError, match="simulated bootstrap failure"):
            dispatch_via_skypilot(sky_cfg)

        mock_sky.jobs.launch.assert_not_called()

    def test_end_to_end_dispatch_uses_cmd_as_run_block(
        self,
        tmp_path: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Happy-path dispatch: sky.Task.from_yaml_config receives a doc whose ``run`` is sky_cfg.cmd.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        cmd = "exec synth-setter-generate-dataset-from-hydra experiment=foo"
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd=cmd,
            env_file=str(env_file),
            job_name="dispatch-smoke",
        )

        dispatch_via_skypilot(sky_cfg)

        mock_sky.Task.from_yaml_config.assert_called()
        passed_doc = mock_sky.Task.from_yaml_config.call_args.args[0]
        assert passed_doc["run"] == cmd

    def test_dispatch_failure_raises_runtime_error(
        self,
        tmp_path: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """A non-success tail rc surfaces as a RuntimeError naming the failed rank.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        mock_sky.stream_and_get.side_effect = RuntimeError("boom")

        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="dispatch-fail",
        )

        with pytest.raises(RuntimeError, match="worker.* failed"):
            dispatch_via_skypilot(sky_cfg)

    def test_multi_worker_fans_out_one_task_per_rank(
        self,
        tmp_path: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """``num_workers=N`` builds N tasks with -rN job-name suffixes, per the fan-out contract.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="fan-out",
            num_workers=3,
        )

        dispatch_via_skypilot(sky_cfg)

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

    def test_extra_envs_forwarded_to_each_rank(
        self,
        tmp_path: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Caller-supplied ``sky_cfg.extra_envs`` lands in every rank's worker env.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="extra-envs",
            num_workers=2,
            extra_envs={"FOO": "bar"},
        )

        dispatch_via_skypilot(sky_cfg)

        update_envs_calls = mock_sky.Task.from_yaml_config.return_value.update_envs.call_args_list
        assert len(update_envs_calls) == 2
        ranks_seen = sorted(call.args[0][WORKER_RANK_ENV_VAR] for call in update_envs_calls)
        assert ranks_seen == ["0", "1"]
        for call in update_envs_calls:
            injected = call.args[0]
            assert injected["FOO"] == "bar"
            assert injected[NUM_WORKERS_ENV_VAR] == "2"

    def test_rank_world_envs_override_caller_extra_envs(
        self,
        tmp_path: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Launcher-owned rank/world keys win over collisions in ``sky_cfg.extra_envs``.

        Pins the schema's documented precedence rule: a caller smuggling a
        bogus ``SYNTH_SETTER_WORKER_RANK`` cannot corrupt partitioning.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="rank-precedence",
            num_workers=2,
            extra_envs={WORKER_RANK_ENV_VAR: "999", NUM_WORKERS_ENV_VAR: "999"},
        )

        dispatch_via_skypilot(sky_cfg)

        update_envs_calls = mock_sky.Task.from_yaml_config.return_value.update_envs.call_args_list
        ranks_seen = sorted(call.args[0][WORKER_RANK_ENV_VAR] for call in update_envs_calls)
        assert ranks_seen == ["0", "1"]
        for call in update_envs_calls:
            assert call.args[0][NUM_WORKERS_ENV_VAR] == "2"

    def test_extra_envs_collision_with_resolved_env_keys_raises(
        self,
        tmp_path: Path,
        env_file: Path,
    ) -> None:
        """Reject ``extra_envs`` keys that overlap ``_WORKER_ENV_KEYS``.

        Prevents a caller from silently bypassing ``resolve_worker_env``'s
        ``.env``-then-process-env resolution for secrets.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        """
        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="cred-overlap",
            extra_envs={"RCLONE_CONFIG_R2_ACCESS_KEY_ID": "bypass"},
        )

        with pytest.raises(ValueError, match="extra_envs keys collide"):
            dispatch_via_skypilot(sky_cfg)

    def test_launcher_does_not_emit_worker_spec_uri_without_extra_envs(
        self,
        tmp_path: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Empty ``extra_envs`` → no rank receives ``WORKER_SPEC_URI`` from the launcher.

        Spec-URI emission moved to the caller (``generate_dataset.main``);
        the launcher is now spec-agnostic.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="no-spec-uri",
            num_workers=2,
            extra_envs={},
        )

        dispatch_via_skypilot(sky_cfg)

        update_envs_calls = mock_sky.Task.from_yaml_config.return_value.update_envs.call_args_list
        assert len(update_envs_calls) == 2
        for call in update_envs_calls:
            assert WORKER_SPEC_URI_ENV not in call.args[0]

    def test_single_worker_dispatch_still_injects_rank_world_env(
        self,
        tmp_path: Path,
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
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            job_name="single-worker-env",
        )

        dispatch_via_skypilot(sky_cfg)

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
        env_file: Path,
        mock_sky: MagicMock,
        field: str,
        value: str,
        match: str,
    ) -> None:
        """Malformed launcher params surface as ValueError before any SkyPilot submission.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        :param field: Parametrized launcher-config field under test.
        :param value: Parametrized malformed value for ``field``.
        :param match: Parametrized regex expected in the raised error.
        """
        template = _write_runpod_yaml(tmp_path)
        kwargs: dict[str, object] = {
            "compute_template": str(template),
            "cmd": "echo",
            "env_file": str(env_file),
            "job_name": "ok-name",
            field: value,
        }
        sky_cfg = SkypilotLaunchConfig(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match=match):
            dispatch_via_skypilot(sky_cfg)
        mock_sky.jobs.launch.assert_not_called()

    def test_api_server_and_local_are_mutually_exclusive(
        self,
        tmp_path: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """Setting both api_server and local raises before any launch — opposite dispatch modes.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo",
            env_file=str(env_file),
            api_server="https://api.example.com",
            local=True,
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            dispatch_via_skypilot(sky_cfg)
        mock_sky.jobs.launch.assert_not_called()

    def test_job_name_falls_back_to_uuid8_stem_when_unset(
        self,
        tmp_path: Path,
        env_file: Path,
        mock_sky: MagicMock,
    ) -> None:
        """``job_name=None`` falls back to ``synth-setter-<uuid8>``.

        Pins the domain-neutral fallback: a caller that doesn't pin a stem gets
        an 8-hex-char uuid suffix, not a dataset-flavored name.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param env_file: Fixture-provided worker env file path.
        :param mock_sky: Mocked ``sky`` module from fixture.
        """
        template = _write_runpod_yaml(tmp_path)
        sky_cfg = SkypilotLaunchConfig(
            compute_template=str(template),
            cmd="echo hi",
            env_file=str(env_file),
            job_name=None,
            extra_envs={},
        )

        dispatch_via_skypilot(sky_cfg)

        mock_sky.jobs.launch.assert_called_once()
        submitted = mock_sky.jobs.launch.call_args.kwargs["name"]
        assert re.fullmatch(r"synth-setter-[0-9a-f]{8}", submitted), submitted
