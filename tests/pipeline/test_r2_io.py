"""Tests for synth_setter.pipeline.r2_io — rclone-backed R2 I/O helpers.

State-based tests use the ``fake_r2_remote`` fixture (see ``tests/pipeline/
conftest.py``): rclone runs against a local-typed remote rooted at a tmp dir,
so each test asserts on the filesystem state after the helper runs instead of
on the rclone argv list. Two narrow argv-shape tests survive (the reliability-
flag set on ``upload_to_uri`` and the ``lsf --format=s`` shape on
``object_size``) to pin invariants state-based tests cannot observe.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest
import yaml

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.object_storage import (
    STORAGE_REQUIRED_ENV_KEYS,
)


@pytest.fixture
def synthetic_unreachable_rclone_env(free_tcp_port: int) -> dict[str, str]:
    """Configure synthetic R2 credentials against an unused loopback port.

    :param free_tcp_port: Unbound loopback port allocated by pytest.
    :returns: Environment that makes real rclone transfers fail locally and quickly.
    """
    if shutil.which("rclone") is None:
        pytest.skip("requires the real rclone binary")
    return {
        "RCLONE_CONFIG": os.devnull,
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "synthetic-access-id-2190",
        "RCLONE_CONFIG_R2_ENDPOINT": f"http://127.0.0.1:{free_tcp_port}",
        "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "synthetic-secret-key-2190",
        "RCLONE_CONFIG_R2_TYPE": "s3",
        "RCLONE_LOW_LEVEL_RETRIES": "1",
        "RCLONE_RETRIES_SLEEP": "0s",
    }


def _assert_redacted_rclone_failure(
    logs: str, rclone_env: dict[str, str], expected_context: str
) -> None:
    """Require credential-free INFO/error logs that identify the failed operation.

    :param logs: Combined process stdout and stderr.
    :param rclone_env: Synthetic credentials whose values must be absent.
    :param expected_context: Operation-specific path that diagnostics must retain.
    """
    assert rclone_env["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] not in logs
    assert rclone_env["RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"] not in logs
    assert " DEBUG " not in logs
    assert "Failed to copy" in logs
    assert expected_context in logs


def _run_debug_template(
    template_name: str, sentinel_name: str, rclone_env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Execute one repository-owned rclone canary run block.

    :param template_name: Repository task template to execute.
    :param sentinel_name: Task-created file removed after execution.
    :param rclone_env: Isolated credentials and endpoint.
    :returns: Captured task process result.
    """
    template_path = (
        Path(__file__).parents[2] / "src" / "synth_setter" / "configs" / "compute" / template_name
    )
    document = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    run_script = document["run"]
    assert isinstance(run_script, str)
    env = {
        **os.environ,
        **rclone_env,
        "R2_BUCKET": "safe-test-bucket",
        "R2_DEBUG_PREFIX": "issue-2190",
    }
    try:
        return subprocess.run(  # noqa: S603 — run block is repository-owned.
            ["/bin/bash", "-c", run_script],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env=env,
        )
    finally:
        (Path(tempfile.gettempdir()) / sentinel_name).unlink(missing_ok=True)


class TestIsR2Uri:
    """Tests for is_r2_uri scheme detection."""

    def test_r2_uri_returns_true(self) -> None:
        """An r2://bucket/key URI is recognized."""
        assert r2_io.is_r2_uri("r2://bucket/key.json")

    def test_local_path_returns_false(self) -> None:
        """An absolute local path is not an r2 URI."""
        assert not r2_io.is_r2_uri("local-spec.json")

    def test_relative_path_returns_false(self) -> None:
        """A relative local path is not an r2 URI."""
        assert not r2_io.is_r2_uri("spec.json")

    def test_other_scheme_returns_false(self) -> None:
        """A non-r2 URI scheme (s3://) is not recognized."""
        assert not r2_io.is_r2_uri("s3://bucket/key.json")

    def test_empty_string_returns_false(self) -> None:
        """Empty string is not an r2 URI."""
        assert not r2_io.is_r2_uri("")


class TestToRclonePath:
    """Tests for _to_rclone_path URI translation."""

    def test_strips_r2_scheme(self) -> None:
        """`r2://bucket/key` becomes rclone's `r2:bucket/key`."""
        assert r2_io._to_rclone_path("r2://bucket/key.json") == "r2:bucket/key.json"

    def test_handles_nested_keys(self) -> None:
        """Nested key paths under the bucket are preserved."""
        assert r2_io._to_rclone_path("r2://bucket/a/b/c.json") == "r2:bucket/a/b/c.json"

    def test_rejects_local_path(self) -> None:
        """Non-r2 URIs raise ValueError so callers branch on is_r2_uri first."""
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io._to_rclone_path("local-spec.json")


class TestRcloneArgv:
    """Tests for _rclone_argv — the centralized reliability-flag builder."""

    def test_default_timeout_emits_full_reliability_flag_set(self) -> None:
        """The default builder pins the verb, reliability flags, then operands in order."""
        assert r2_io._rclone_argv("copyto", "r2:bucket/key", "dest/file") == [
            "rclone",
            "copyto",
            "-v",
            "--checksum",
            "--contimeout=30s",
            "--timeout=300s",
            "--retries=3",
            "r2:bucket/key",
            "dest/file",
        ]

    def test_custom_timeout_widens_only_the_io_timeout(self) -> None:
        """A non-default timeout (the dir-upload 3h) lands on ``--timeout`` alone."""
        assert r2_io._rclone_argv("copy", "src/dir", "r2:bucket/p", timeout="3h") == [
            "rclone",
            "copy",
            "-v",
            "--checksum",
            "--contimeout=30s",
            "--timeout=3h",
            "--retries=3",
            "src/dir",
            "r2:bucket/p",
        ]


class TestRcloneDebugTemplates:
    """Tests for SkyPilot rclone canary task logging."""

    @pytest.mark.parametrize(
        ("template_name", "sentinel_name"),
        [
            ("local-debug-rclone-template.yaml", "skypilot-local-debug-sentinel.txt"),
            ("runpod-debug-rclone-template.yaml", "skypilot-debug-rclone-sentinel.txt"),
        ],
    )
    def test_run_failure_redacts_credentials_and_keeps_error_context(
        self,
        template_name: str,
        sentinel_name: str,
        synthetic_unreachable_rclone_env: dict[str, str],
    ) -> None:
        """A real task run omits credentials while retaining its failure cause.

        :param template_name: Repository task template to execute.
        :param sentinel_name: Task-created file removed after execution.
        :param synthetic_unreachable_rclone_env: Isolated credentials and endpoint.
        """
        result = _run_debug_template(
            template_name, sentinel_name, synthetic_unreachable_rclone_env
        )
        logs = f"{result.stdout}\n{result.stderr}"
        assert result.returncode != 0
        _assert_redacted_rclone_failure(logs, synthetic_unreachable_rclone_env, "safe-test-bucket")


class TestToS3Uri:
    """Tests for to_s3_uri — r2:// → s3:// scheme rewrite for W&B references."""

    def test_rewrites_r2_scheme_to_s3(self) -> None:
        """`r2://bucket/key` becomes `s3://bucket/key`, preserving the path verbatim."""
        assert r2_io.to_s3_uri("r2://bucket/key.ckpt") == "s3://bucket/key.ckpt"

    def test_preserves_nested_prefix(self) -> None:
        """Nested key paths under the bucket are preserved unchanged."""
        assert r2_io.to_s3_uri("r2://bucket/a/b/last.ckpt") == "s3://bucket/a/b/last.ckpt"

    def test_rejects_non_r2_scheme(self) -> None:
        """A non-r2:// URI raises ValueError rather than silently passing through."""
        with pytest.raises(ValueError, match="r2://"):
            r2_io.to_s3_uri("s3://bucket/already-s3.ckpt")


class TestFromS3Uri:
    """Tests for from_s3_uri — s3:// → r2:// rewrite that inverts to_s3_uri."""

    def test_rewrites_s3_scheme_to_r2(self) -> None:
        """`s3://bucket/key` becomes `r2://bucket/key`, preserving the path verbatim."""
        assert r2_io.from_s3_uri("s3://bucket/key.ckpt") == "r2://bucket/key.ckpt"

    def test_round_trips_with_to_s3_uri(self) -> None:
        """from_s3_uri inverts to_s3_uri so a reference URI survives the round trip."""
        r2_uri = "r2://intermediate-data/checkpoints/flow-simple/model.ckpt"
        assert r2_io.from_s3_uri(r2_io.to_s3_uri(r2_uri)) == r2_uri

    def test_rejects_non_s3_scheme(self) -> None:
        """A non-s3:// URI raises ValueError rather than silently passing through."""
        with pytest.raises(ValueError, match="s3://"):
            r2_io.from_s3_uri("r2://bucket/not-s3.ckpt")


class TestR2StorageOptions:
    """Tests for r2_storage_options — Lance object-store config from R2 env vars."""

    @pytest.fixture(autouse=True)
    def _clear_r2_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drop stray storage env and the workspace dotenv so tests do not read a developer shell.

        :param tmp_path: Pytest tmp dir used for an intentionally missing default dotenv.
        :param monkeypatch: Pytest fixture used to remove env vars.
        """
        import os

        monkeypatch.setattr(r2_io, "_DEFAULT_ENV_FILE", tmp_path / "missing.env")
        for key in list(os.environ):
            if key.startswith(("SYNTH_SETTER_STORAGE_", "RCLONE_CONFIG_R2_")):
                monkeypatch.delenv(key, raising=False)

    def test_builds_object_store_dict_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The canonical storage settings map to the documented S3 keys plus region.

        :param monkeypatch: Pytest fixture used to set the R2 secret env vars.
        """
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv(
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com"
        )
        assert r2_io.r2_storage_options() == {
            "access_key_id": "ak",
            "secret_access_key": "sk",
            "endpoint": "https://acct.r2.cloudflarestorage.com",
            "aws_endpoint": "https://acct.r2.cloudflarestorage.com",
            "region": "auto",
        }

    def test_strips_secret_env_values_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Surrounding env whitespace is ignored before returning storage options.

        :param monkeypatch: Pytest fixture used to set the R2 secret env vars.
        """
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", " ak ")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "\tsk\n")
        monkeypatch.setenv(
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL", " https://acct.r2.cloudflarestorage.com "
        )
        assert r2_io.r2_storage_options() == {
            "access_key_id": "ak",
            "secret_access_key": "sk",
            "endpoint": "https://acct.r2.cloudflarestorage.com",
            "aws_endpoint": "https://acct.r2.cloudflarestorage.com",
            "region": "auto",
        }

    def test_raises_when_secret_env_keys_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing secret raises RuntimeError rather than emitting a partial dict.

        :param monkeypatch: Pytest fixture used to clear the R2 secret env vars.
        """
        for key in STORAGE_REQUIRED_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        with pytest.raises(RuntimeError, match="Object storage settings unresolved"):
            r2_io.r2_storage_options()

    def test_blank_secret_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A present-but-blank secret is treated as missing, not built into a partial dict.

        :param monkeypatch: Pytest fixture used to set the R2 secret env vars.
        """
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "   ")
        monkeypatch.setenv(
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com"
        )
        with pytest.raises(RuntimeError, match="SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY"):
            r2_io.r2_storage_options()

    def test_legacy_rclone_env_builds_storage_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy rclone names remain compatible with storage-option loading.

        :param monkeypatch: Pytest fixture used to set the R2 secret env vars.
        """
        monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
        assert r2_io.r2_storage_options() == {
            "access_key_id": "ak",
            "secret_access_key": "sk",
            "endpoint": "https://acct.r2.cloudflarestorage.com",
            "aws_endpoint": "https://acct.r2.cloudflarestorage.com",
            "region": "auto",
        }


class TestR2DirectoryExists:
    """Tests for r2_directory_exists — prefix existence probe via non-recursive ``rclone lsf``.

    Present and missing-prefix cases are state-based against the fake-local
    remote; both backends normalize a missing prefix to ``False`` via the
    shared listing probe.
    """

    def test_missing_prefix_returns_false(self, fake_r2_remote: Path) -> None:
        """A never-created prefix reads as absent on the local backend too.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        assert r2_io.r2_directory_exists("r2://bucket/never/created") is False

    def test_present_prefix_returns_true(self, fake_r2_remote: Path) -> None:
        """A prefix containing at least one object returns ``True``.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        obj = fake_r2_remote / "bucket" / "shard-000000.lance" / "data" / "part.lance"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b"x")

        assert r2_io.r2_directory_exists("r2://bucket/shard-000000.lance") is True

    def test_empty_listing_returns_false(self) -> None:
        """Empty ``rclone lsf`` stdout (R2's missing-prefix shape) returns ``False``."""
        completed = MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = ""
        completed.returncode = 0
        with patch("synth_setter.pipeline.r2_io.subprocess.run", return_value=completed):
            assert r2_io.r2_directory_exists("r2://bucket/missing.lance") is False

    def test_nonzero_rclone_exit_propagates(self) -> None:
        """A non-zero rclone exit (auth/network) fails fast rather than reading as absent."""
        err = subprocess.CalledProcessError(returncode=1, cmd=["rclone", "lsf"])
        with (
            patch("synth_setter.pipeline.r2_io.subprocess.run", side_effect=err),
            pytest.raises(subprocess.CalledProcessError),
        ):
            r2_io.r2_directory_exists("r2://bucket/shard.lance")


class TestDownloadToPath:
    """Tests for download_to_path — file→file copy."""

    def test_lands_remote_bytes_at_local_path(self, fake_r2_remote: Path, tmp_path: Path) -> None:
        """Downloading writes the remote object's bytes verbatim to ``dest_path``.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir used for the download destination.
        """
        remote_obj = fake_r2_remote / "bucket" / "key.json"
        remote_obj.parent.mkdir(parents=True)
        remote_obj.write_text('{"ok": true}')
        dest = tmp_path / "out.json"

        r2_io.download_to_path("r2://bucket/key.json", dest)

        assert dest.read_text() == '{"ok": true}'

    def test_preserves_destination_filename_not_source_basename(
        self, fake_r2_remote: Path, tmp_path: Path
    ) -> None:
        """``copyto`` (not ``copy``) keeps the destination filename as-is.

        ``rclone copy`` would treat ``dest`` as a directory and write the source
        basename inside it; ``copyto`` preserves the destination filename verbatim.
        The dest's filename differs from the source's on purpose so the wrong-verb
        regression would surface as a missing file.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir used for the download destination.
        """
        remote_obj = fake_r2_remote / "bucket" / "src-name.json"
        remote_obj.parent.mkdir(parents=True)
        remote_obj.write_text("{}")
        dest = tmp_path / "different-name.json"

        r2_io.download_to_path("r2://bucket/src-name.json", dest)

        assert dest.is_file()
        assert dest.read_text() == "{}"

    def test_rejects_non_r2_uri(self, tmp_path: Path) -> None:
        """A local-path source is rejected — caller must branch on is_r2_uri.

        :param tmp_path: Pytest tmp dir used to build a local destination path.
        """
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.download_to_path("local-spec.json", tmp_path / "out.json")

    def test_command_carries_rclone_reliability_flags(self, tmp_path: Path) -> None:
        """Pin the rclone reliability-flag set on the per-shard copy-source fetch.

        This is the ``r2://`` copy-root download on the renderer hot path; dropping
        ``--retries`` would surface a transient network blip as a hard mid-render
        failure rather than a retried fetch. One mock-based argv assertion guards
        the invariant the state-based tests cannot observe.

        :param tmp_path: Pytest tmp dir used for the download destination.
        """
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            r2_io.download_to_path("r2://bucket/key.json", tmp_path / "out.json")
        args = mock_call.call_args[0][0]
        assert args[:2] == ["rclone", "copyto"]
        assert "-v" in args
        assert "--checksum" in args
        assert "--contimeout=30s" in args
        assert "--timeout=300s" in args
        assert "--retries=3" in args


class TestDownloadDirNoOverwrite:
    """Tests for download_dir_no_overwrite — prefix→directory copy."""

    def test_lands_remote_tree_under_dest_dir(self, fake_r2_remote: Path, tmp_path: Path) -> None:
        """Every object under the prefix is copied into ``dest_path``.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir used for the download destination.
        """
        prefix = fake_r2_remote / "bucket" / "dataset"
        prefix.mkdir(parents=True)
        (prefix / "train.lance").write_text("train")
        (prefix / "stats.npz").write_text("stats")
        dest = tmp_path / "root"

        r2_io.download_dir_no_overwrite("r2://bucket/dataset", dest)

        assert (dest / "train.lance").read_text() == "train"
        assert (dest / "stats.npz").read_text() == "stats"

    def test_lands_file_uri_tree_under_dest_dir(self, tmp_path: Path) -> None:
        """Every file under a local URI is copied into ``dest_path``.

        :param tmp_path: Pytest tmp dir holding the source and destination trees.
        """
        source = tmp_path / "network volume"
        source.mkdir()
        (source / "train.lance").write_text("train")
        dest = tmp_path / "root"

        r2_io.download_dir_no_overwrite(source.as_uri(), dest)

        assert (dest / "train.lance").read_text() == "train"

    def test_rejects_unsupported_source_uri(self, tmp_path: Path) -> None:
        """A source outside the R2 and local-file contracts is rejected.

        :param tmp_path: Pytest tmp dir used to build a local destination path.
        """
        with pytest.raises(ValueError, match="r2:// or file://"):
            r2_io.download_dir_no_overwrite("https://example.com/dataset", tmp_path / "root")

    def test_command_carries_immutable_and_reliability_flags(self, tmp_path: Path) -> None:
        """Pin the rclone verb + ``--immutable`` + reliability-flag set.

        ``--immutable`` is the entire point of this helper (no-clobber) and is
        unobservable from filesystem state alone; the reliability flags must
        match the upload helpers so a transient blip retries instead of failing
        the eval. One mock-based argv assertion guards both invariants.

        :param tmp_path: Pytest tmp dir used for the download destination.
        """
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            r2_io.download_dir_no_overwrite("r2://bucket/dataset", tmp_path / "root")
        args = mock_call.call_args[0][0]
        assert args[:2] == ["rclone", "copy"]
        assert "--immutable" in args
        assert "--checksum" in args
        assert "-v" in args
        assert "--contimeout=30s" in args
        assert "--timeout=300s" in args
        assert "--retries=3" in args


class TestUploadDir:
    """Tests for upload_dir — directory→prefix upload (mirror of download_dir_no_overwrite)."""

    def test_lands_local_tree_under_remote_prefix(
        self, fake_r2_remote: Path, tmp_path: Path
    ) -> None:
        """Every file under ``local_dir`` is copied beneath the destination prefix.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir holding the local source tree.
        """
        local_dir = tmp_path / "run"
        (local_dir / "metrics").mkdir(parents=True)
        (local_dir / "metrics" / "metrics.json").write_text('{"ok": true}')
        (local_dir / "config.log").write_text("cfg")

        r2_io.upload_dir(local_dir, "r2://bucket/evals/run-1")

        dest = fake_r2_remote / "bucket" / "evals" / "run-1"
        assert (dest / "metrics" / "metrics.json").read_text() == '{"ok": true}'
        assert (dest / "config.log").read_text() == "cfg"

    def test_rejects_non_r2_uri(self, tmp_path: Path) -> None:
        """A local-path destination is rejected — caller must pass an ``r2://`` URI.

        :param tmp_path: Pytest tmp dir used to build a local source path.
        """
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.upload_dir(tmp_path / "run", "local-dest")

    def test_reupload_overwrites_changed_file(self, fake_r2_remote: Path, tmp_path: Path) -> None:
        """Re-uploading a changed source overwrites the remote copy — no ``--immutable``.

        The caller pushes its own freshly-produced run dir, so a second upload
        must replace stale objects rather than hard-fail the way the
        ``--immutable`` download guard would.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir holding the local source tree.
        """
        local_dir = tmp_path / "run"
        local_dir.mkdir()
        (local_dir / "metrics.json").write_text('{"param_mse": 1.0}')
        r2_io.upload_dir(local_dir, "r2://bucket/evals/run-1")

        (local_dir / "metrics.json").write_text('{"param_mse": 0.0}')
        r2_io.upload_dir(local_dir, "r2://bucket/evals/run-1")

        dest = fake_r2_remote / "bucket" / "evals" / "run-1"
        assert (dest / "metrics.json").read_text() == '{"param_mse": 0.0}'

    def test_command_widens_io_timeout_and_omits_immutable(self, tmp_path: Path) -> None:
        """Pin the rclone verb, the 3h IO timeout, and the absence of ``--immutable``.

        The widened ``--timeout`` lets a whole run dir stream past the single-file
        default, and the missing ``--immutable`` is what lets a re-upload overwrite
        — both unobservable from filesystem state. One argv assertion guards them.

        :param tmp_path: Pytest tmp dir used to build the local source path.
        """
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            r2_io.upload_dir(tmp_path / "run", "r2://bucket/evals/run-1")
        args = mock_call.call_args[0][0]
        assert args[:2] == ["rclone", "copy"]
        assert "--immutable" not in args
        assert "--checksum" in args
        assert "--contimeout=30s" in args
        assert "--timeout=3h" in args
        assert "--retries=3" in args


class TestUploadToUri:
    """Tests for upload_to_uri — file→file upload with reliability flags."""

    def test_lands_local_bytes_at_remote_uri(self, fake_r2_remote: Path, tmp_path: Path) -> None:
        """Upload writes the local file's bytes to the URI's path under the bucket.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir used for the upload source file.
        """
        src = tmp_path / "in.json"
        src.write_text('{"payload": 42}')

        r2_io.upload_to_uri(src, "r2://bucket/nested/key.json")

        assert (fake_r2_remote / "bucket" / "nested" / "key.json").read_text() == '{"payload": 42}'

    def test_preserves_destination_filename_not_source_basename(
        self, fake_r2_remote: Path, tmp_path: Path
    ) -> None:
        """``copyto`` (not ``copy``) keeps the destination object name as-is.

        ``rclone copy`` would treat the URI as a directory and write the source
        basename inside it. The dest's last segment differs from the source's
        so the wrong-verb regression would surface as a missing object.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir used for the upload source file.
        """
        src = tmp_path / "src-name.json"
        src.write_text("{}")

        r2_io.upload_to_uri(src, "r2://bucket/different-name.json")

        assert (fake_r2_remote / "bucket" / "different-name.json").is_file()
        assert not (fake_r2_remote / "bucket" / "different-name.json" / "src-name.json").exists()

    def test_rejects_non_r2_uri(self, tmp_path: Path) -> None:
        """A local-path destination is rejected.

        :param tmp_path: Pytest tmp dir used to build a local source path.
        """
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.upload_to_uri(tmp_path / "in.json", "local-dest.json")

    def test_failure_logs_redact_credentials_and_keep_error_context(
        self,
        tmp_path: Path,
        synthetic_unreachable_rclone_env: dict[str, str],
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A real failed rclone upload omits credentials but retains its cause.

        :param tmp_path: Local upload source directory.
        :param synthetic_unreachable_rclone_env: Isolated credentials and endpoint.
        :param capfd: Captures the child rclone process's inherited descriptors.
        """
        src = tmp_path / "in.json"
        src.write_text("{}")

        with (
            patch.dict(os.environ, synthetic_unreachable_rclone_env),
            pytest.raises(subprocess.CalledProcessError),
        ):
            r2_io.upload_to_uri(src, "r2://safe-test-bucket/issue-2190/object")

        captured = capfd.readouterr()
        logs = f"{captured.out}\n{captured.err}"
        _assert_redacted_rclone_failure(
            logs, synthetic_unreachable_rclone_env, "issue-2190/object"
        )

    def test_command_carries_rclone_reliability_flags(self, tmp_path: Path) -> None:
        """Pin the rclone reliability-flag set on upload.

        State-based tests cover the file-landing contract but cannot observe the
        ``-v / --checksum / --contimeout / --timeout / --retries`` flags. Losing
        any of them is a silent correctness regression (e.g. dropping
        ``--checksum`` would let half-uploaded objects pass; dropping
        ``--retries`` would surface transient network blips as hard failures).
        One mock-based argv assertion guards the invariant.

        :param tmp_path: Pytest tmp dir used for the upload source file.
        """
        src = tmp_path / "in.json"
        src.write_text("{}")
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            r2_io.upload_to_uri(src, "r2://bucket/key.json")
        args = mock_call.call_args[0][0]
        assert args[:2] == ["rclone", "copyto"]
        assert "-v" in args
        assert "--checksum" in args
        assert "--contimeout=30s" in args
        assert "--timeout=300s" in args
        assert "--retries=3" in args


class TestIsR2Reachable:
    """Tests for ``is_r2_reachable`` — boolean auth-probe used as a test-skip gate."""

    @pytest.fixture(autouse=True)
    def _clear_r2_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drop stray storage env and the workspace dotenv for a clean unreachable baseline.

        :param tmp_path: Pytest tmp dir used for an intentionally missing default dotenv.
        :param monkeypatch: Pytest fixture used to remove env vars.
        """
        import os

        monkeypatch.setattr(r2_io, "_DEFAULT_ENV_FILE", tmp_path / "missing.env")
        for key in list(os.environ):
            if key.startswith(("SYNTH_SETTER_STORAGE_", "RCLONE_CONFIG_R2_")):
                monkeypatch.delenv(key, raising=False)

    def test_returns_true_when_rclone_lsd_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: rclone on PATH, env keys present, probe exits 0.

        :param monkeypatch: Pytest fixture used to stub PATH + env + ``subprocess.run``.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        for key in STORAGE_REQUIRED_ENV_KEYS:
            monkeypatch.setenv(key, "stub")

        class _OK:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr("synth_setter.pipeline.r2_io.subprocess.run", lambda *a, **kw: _OK())
        assert r2_io.is_r2_reachable() is True

    def test_returns_true_when_default_dotenv_has_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default workspace dotenv credentials satisfy the integration-test skip gate.

        :param tmp_path: Pytest tmp dir for the default dotenv file.
        :param monkeypatch: Pytest fixture used to isolate env and subprocess behavior.
        """
        default_env_file = tmp_path / ".env"
        default_env_file.write_text(
            "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID=id-from-default\n"
            "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY=secret-from-default\n"
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL=endpoint-from-default\n"
        )
        monkeypatch.setattr(r2_io, "_DEFAULT_ENV_FILE", default_env_file)
        monkeypatch.setattr(r2_io.shutil, "which", lambda name: f"/usr/bin/{name}")

        with patch.object(r2_io.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            assert r2_io.is_r2_reachable() is True

    def test_probe_subprocess_receives_projected_rclone_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The auth ping runs with the resolved settings, not just ambient rclone config.

        Without the projection a developer's local rclone remote could make the
        probe pass while the canonical settings are wrong, so the gate would
        disagree with :func:`ensure_r2_env_loaded`.

        :param tmp_path: Pytest tmp dir for the default dotenv file.
        :param monkeypatch: Pytest fixture used to isolate env and subprocess behavior.
        """
        default_env_file = tmp_path / ".env"
        default_env_file.write_text(
            "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID=id-from-default\n"
            "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY=secret-from-default\n"
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL=endpoint-from-default\n"
        )
        monkeypatch.setattr(r2_io, "_DEFAULT_ENV_FILE", default_env_file)
        monkeypatch.setattr(r2_io.shutil, "which", lambda name: f"/usr/bin/{name}")
        captured: dict[str, object] = {}

        def _capture(*_a: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured.update(kwargs)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            assert r2_io.is_r2_reachable() is True

        env = captured["env"]
        assert isinstance(env, dict)
        assert env["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "id-from-default"
        assert captured["timeout"] == r2_io._AUTH_PING_TIMEOUT_SECONDS  # noqa: SLF001

    def test_returns_false_when_auth_ping_times_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hung auth ping does not let the integration-test gate pass.

        :param monkeypatch: Pytest fixture used to stub PATH + env + the timed-out probe.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        for key in STORAGE_REQUIRED_ENV_KEYS:
            monkeypatch.setenv(key, "stub")

        def _timeout(*_args: object, **_kwargs: object) -> NoReturn:
            raise subprocess.TimeoutExpired(cmd=["rclone", "lsd", "r2:"], timeout=45)

        monkeypatch.setattr("synth_setter.pipeline.r2_io.subprocess.run", _timeout)
        assert r2_io.is_r2_reachable() is False

    def test_returns_false_when_secret_env_keys_are_blank(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blank storage secret values do not satisfy the integration-test skip gate.

        :param tmp_path: Pytest tmp dir used for an intentionally missing default dotenv.
        :param monkeypatch: Pytest fixture used to isolate env and subprocess behavior.
        """
        monkeypatch.setattr(r2_io, "_DEFAULT_ENV_FILE", tmp_path / "missing.env")
        monkeypatch.setattr(r2_io.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "   ")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", "endpoint")
        monkeypatch.setattr(
            r2_io.subprocess,
            "run",
            lambda *a, **kw: pytest.fail("subprocess.run should not be reached"),
        )

        assert r2_io.is_r2_reachable() is False

    def test_returns_false_when_rclone_lsd_exits_non_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auth failure: rclone + env keys present but the probe exits non-zero.

        :param monkeypatch: Pytest fixture used to stub PATH + env + ``subprocess.run``.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        for key in STORAGE_REQUIRED_ENV_KEYS:
            monkeypatch.setenv(key, "stub")

        def fake_run(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise subprocess.CalledProcessError(returncode=1, cmd=["rclone", "lsd", "r2:"])

        monkeypatch.setattr("synth_setter.pipeline.r2_io.subprocess.run", fake_run)
        assert r2_io.is_r2_reachable() is False

    def test_returns_false_when_rclone_not_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bare clone / missing binary: rclone is absent from PATH.

        :param monkeypatch: Pytest fixture used to stub ``shutil.which``.
        """
        monkeypatch.setattr("synth_setter.pipeline.r2_io.shutil.which", lambda _name: None)
        # subprocess.run must never be called — short-circuit on PATH miss.
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.subprocess.run",
            lambda *a, **kw: pytest.fail("subprocess.run should not be reached"),
        )
        assert r2_io.is_r2_reachable() is False

    def test_returns_false_when_secret_env_keys_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rclone-on-PATH + working local config but no env keys → skip, not hard-fail later.

        Mirrors the contract of ``ensure_r2_env_loaded`` so a test that
        gates on ``is_r2_reachable`` doesn't pass the gate and then crash
        on ``RuntimeError`` from the env-key check downstream.

        :param tmp_path: Pytest tmp dir used for an intentionally missing default dotenv.
        :param monkeypatch: Pytest fixture used to clear env + stub the probe.
        """
        monkeypatch.setattr(r2_io, "_DEFAULT_ENV_FILE", tmp_path / "missing.env")
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        for key in STORAGE_REQUIRED_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        # subprocess.run must never be called — short-circuit on missing env.
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.subprocess.run",
            lambda *a, **kw: pytest.fail("subprocess.run should not be reached"),
        )
        assert r2_io.is_r2_reachable() is False


class TestUpload:
    """Tests for ``upload`` — source-type-tolerant wrapper over ``rclone copyto``."""

    def test_local_path_source_lands_at_destination(
        self, fake_r2_remote: Path, tmp_path: Path
    ) -> None:
        """A local ``Path`` source uploads via the same path ``upload_to_uri`` exercises.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir used for the upload source file.
        """
        src = tmp_path / "in.json"
        src.write_text('{"payload": 7}')

        r2_io.upload(src, "r2://bucket/key.json")

        assert (fake_r2_remote / "bucket" / "key.json").read_text() == '{"payload": 7}'

    def test_local_str_path_source_lands_at_destination(
        self, fake_r2_remote: Path, tmp_path: Path
    ) -> None:
        """A local path passed as ``str`` is coerced to ``Path`` and uploaded.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir used for the upload source file.
        """
        src = tmp_path / "in.json"
        src.write_text("{}")

        r2_io.upload(str(src), "r2://bucket/key.json")

        assert (fake_r2_remote / "bucket" / "key.json").is_file()

    def test_r2_uri_source_copies_remote_to_remote(
        self, fake_r2_remote: Path, tmp_path: Path
    ) -> None:
        """An ``r2://`` source triggers an rclone R2→R2 copy (not a local upload).

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir (unused but threaded for fixture symmetry).
        """
        seed = fake_r2_remote / "bucket" / "src" / "key.json"
        seed.parent.mkdir(parents=True)
        seed.write_text('{"seed": true}')

        r2_io.upload("r2://bucket/src/key.json", "r2://bucket/dst/key.json")

        assert (fake_r2_remote / "bucket" / "dst" / "key.json").read_text() == '{"seed": true}'

    def test_r2_uri_source_uses_rclone_copyto_with_reliability_flags(self, tmp_path: Path) -> None:
        """R2→R2 path carries the same reliability flag set as the local-upload path.

        :param tmp_path: Pytest tmp dir (unused; threaded so fixture isolation is consistent).
        """
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            r2_io.upload("r2://bucket/src/key.json", "r2://bucket/dst/key.json")
        args = mock_call.call_args[0][0]
        assert args[:2] == ["rclone", "copyto"]
        assert "--checksum" in args
        assert "--contimeout=30s" in args
        assert "--timeout=300s" in args
        assert "--retries=3" in args
        assert args[-2:] == ["r2:bucket/src/key.json", "r2:bucket/dst/key.json"]

    def test_rejects_path_whose_text_looks_like_r2_uri(self) -> None:
        """A ``Path("r2://...")`` is rejected so dispatch is unambiguous between local and R2."""
        with pytest.raises(TypeError, match=r"upload\(\) received Path.*r2://"):
            r2_io.upload(Path("r2://bucket/src/key.json"), "r2://bucket/dst/key.json")


class TestDownloadedToTempfile:
    """Tests for the downloaded_to_tempfile context manager."""

    def test_yields_local_path_named_after_uri_basename(self, fake_r2_remote: Path) -> None:
        """Yielded path name matches URI's last segment; tempdir is cleaned on exit.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        remote_obj = fake_r2_remote / "bucket" / "path" / "spec.json"
        remote_obj.parent.mkdir(parents=True)
        remote_obj.write_text('{"ok": true}')
        seen: list[Path] = []

        with r2_io.downloaded_to_tempfile("r2://bucket/path/spec.json") as local:
            seen.append(local)
            assert local.name == "spec.json"
            assert local.read_text() == '{"ok": true}'

        assert not seen[0].exists()

    def test_cleanup_runs_even_on_exception(self, fake_r2_remote: Path) -> None:
        """If the with-block raises, the tempdir is still cleaned up.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :raises RuntimeError: Deliberately raised inside the ``with`` block to
            exercise the cleanup-on-exception path.
        """
        remote_obj = fake_r2_remote / "bucket" / "spec.json"
        remote_obj.parent.mkdir(parents=True)
        remote_obj.write_text("{}")
        seen: list[Path] = []

        with pytest.raises(RuntimeError, match="boom"):
            with r2_io.downloaded_to_tempfile("r2://bucket/spec.json") as local:
                seen.append(local)
                raise RuntimeError("boom")

        assert not seen[0].exists()


class TestShardUri:
    """Tests for shard_uri — canonical R2 URI builder for shards."""

    def test_constructs_full_uri_from_bucket_prefix_filename(self) -> None:
        """The URI follows the r2://{bucket}/{prefix}{filename} convention exactly."""
        assert (
            r2_io.shard_uri("intermediate-data", "data/run-x/", "shard-000007.lance")
            == "r2://intermediate-data/data/run-x/shard-000007.lance"
        )

    def test_preserves_nested_prefix(self) -> None:
        """Multi-segment prefixes are joined verbatim (caller controls trailing slash)."""
        assert (
            r2_io.shard_uri("bucket", "a/b/c/", "shard-000000.lance")
            == "r2://bucket/a/b/c/shard-000000.lance"
        )


class TestObjectSize:
    """Exercise object-size and absence probes across local and R2 semantics.

    R2 returns empty stdout for a missing key under an existing bucket; both
    backends normalize absence to ``None``.
    """

    @staticmethod
    def _mock_run(stdout: str) -> MagicMock:
        """Build a CompletedProcess-shaped mock with the given stdout."""
        completed = MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = stdout
        completed.returncode = 0
        return completed

    def test_present_returns_int_size(self, fake_r2_remote: Path) -> None:
        """A present object returns its size in bytes.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        obj = fake_r2_remote / "bucket" / "key.lance"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b"x" * 12345)

        assert r2_io.object_size("r2://bucket/key.lance") == 12345

    def test_zero_size_returns_zero(self, fake_r2_remote: Path) -> None:
        """A zero-byte object exists; return 0 (callers decide whether to treat as present).

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        obj = fake_r2_remote / "bucket" / "key.lance"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b"")

        assert r2_io.object_size("r2://bucket/key.lance") == 0

    def test_absent_returns_none(self) -> None:
        """Empty stdout means the object is missing; return None.

        R2-specific behavior — see class docstring for why this stays mock-based.
        """
        with patch.object(r2_io.subprocess, "run", return_value=self._mock_run("")):
            assert r2_io.object_size("r2://bucket/key.lance") is None

    def test_absent_parent_directory_returns_none(self, fake_r2_remote: Path) -> None:
        """A missing parent directory means the object is absent; return None.

        The local backend errors "directory not found" where R2 lists empty —
        both normalize to ``None`` so the local compute mode probes identically.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        assert r2_io.object_size("r2://bucket/never/created/key.lance") is None

    def test_probe_failure_propagates(self) -> None:
        """Non-zero rclone exit (not a missing dir) raises — fail-fast on env issues."""
        completed = MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = ""
        completed.stderr = "Failed to lsf: AccessDenied"
        completed.returncode = 1
        with patch.object(r2_io.subprocess, "run", return_value=completed):
            with pytest.raises(subprocess.CalledProcessError):
                r2_io.object_size("r2://bucket/key.lance")

    def test_non_integer_stdout_raises_with_original_cause(self) -> None:
        """Unparsable rclone stdout raises a contextual error chained to the ValueError.

        Guards the generate failure path: ``int(out)`` on garbage stdout would
        otherwise surface a bare ``invalid literal for int()`` with no probed
        URI, losing the rclone-listing context. The re-raise must keep the
        original ``ValueError`` as ``__cause__``.
        """
        with patch.object(r2_io.subprocess, "run", return_value=self._mock_run("not-a-number")):
            with pytest.raises(RuntimeError, match="r2://bucket/key.lance") as excinfo:
                r2_io.object_size("r2://bucket/key.lance")
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_invokes_rclone_lsf_format_s(self) -> None:
        """Probe argv shape: rclone lsf --format=s <translated path>.

        Pins the ``lsf --format=s`` probe shape; the state-based ``present``/
        ``zero_size`` tests above exercise the happy paths but cannot observe
        the exact argv, and a wrong flag here would silently break the absent
        detection (which relies on empty stdout).
        """
        with patch.object(r2_io.subprocess, "run", return_value=self._mock_run("42")) as mock_run:
            r2_io.object_size("r2://bucket/path/key.lance")
        args = mock_run.call_args[0][0]
        assert args == [
            "rclone",
            "lsf",
            "--format=s",
            "--retries=3",
            "--contimeout=30s",
            "r2:bucket/path/key.lance",
        ]
        kwargs = mock_run.call_args[1]
        assert kwargs.get("check") is False
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True

    def test_rejects_non_r2_uri(self) -> None:
        """Local paths are rejected via _to_rclone_path."""
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.object_size("local/key.lance")


class TestListEntries:
    """Tests for list_entries — mtime-carrying directory listing via `rclone lsjson`."""

    def test_missing_directory_lists_empty(self, fake_r2_remote: Path) -> None:
        """An absent staging directory is a normal reconciliation answer, not an error.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        assert r2_io.list_entries("r2://bucket/never/created/") == []

    def test_recursive_listing_returns_sorted_relative_paths_with_mtimes(
        self, fake_r2_remote: Path
    ) -> None:
        """Nested files list with slash-joined relative paths, sorted, mtimes parsed.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        root = fake_r2_remote / "bucket" / "staging"
        (root / "shard-000001").mkdir(parents=True)
        (root / "shard-000000").mkdir(parents=True)
        (root / "shard-000001" / "b.valid").write_bytes(b"")
        (root / "shard-000000" / "a.valid").write_bytes(b"xy")

        entries = r2_io.list_entries("r2://bucket/staging/", recursive=True)

        assert [entry.path for entry in entries] == [
            "shard-000000/a.valid",
            "shard-000001/b.valid",
        ]
        assert entries[0].size == 2
        assert entries[0].mtime.tzinfo is not None

    def test_non_recursive_listing_excludes_nested_files(self, fake_r2_remote: Path) -> None:
        """Without ``recursive`` only the top level lists.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        root = fake_r2_remote / "bucket" / "staging"
        (root / "nested").mkdir(parents=True)
        (root / "top.valid").write_bytes(b"")
        (root / "nested" / "deep.valid").write_bytes(b"")

        entries = r2_io.list_entries("r2://bucket/staging/")

        assert [entry.path for entry in entries] == ["top.valid"]

    def test_listing_record_schema_drift_fails_at_strict_json_boundary(self) -> None:
        """Malformed rclone records fail with the offending field named contextually."""
        payload = '[{"Path": "a.valid", "ModTime": "2026-01-01T00:00:00Z", "Size": "2"}]'

        with patch.object(r2_io, "_run_listing_probe", return_value=payload):
            with pytest.raises(ValueError, match="Size"):
                r2_io.list_entries("r2://bucket/staging/")

    def test_listing_failure_other_than_missing_directory_propagates(self) -> None:
        """A genuine rclone failure raises rather than reading as "no attempts staged"."""
        completed = MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = ""
        completed.stderr = "Failed to lsjson: AccessDenied"
        completed.returncode = 1
        with patch.object(r2_io.subprocess, "run", return_value=completed):
            with pytest.raises(subprocess.CalledProcessError):
                r2_io.list_entries("r2://bucket/staging/", recursive=True)

    def test_local_missing_directory_exit_lists_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The local backend's exit-code-3 missing directory reads as absent.

        :param monkeypatch: Selects the local rclone backend for the probe.
        """
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
        completed = MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = ""
        completed.stderr = "Failed to lsjson: directory not found"
        completed.returncode = 3

        with patch.object(r2_io.subprocess, "run", return_value=completed):
            assert r2_io.list_entries("r2://bucket/staging/") == []

    def test_local_non_missing_exit_with_directory_text_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only rclone's missing-directory exit code is normalized for local probes.

        :param monkeypatch: Selects the local rclone backend for the probe.
        """
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
        completed = MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = ""
        completed.stderr = "Failed to lsjson: directory not found"
        completed.returncode = 1

        with patch.object(r2_io.subprocess, "run", return_value=completed):
            with pytest.raises(subprocess.CalledProcessError):
                r2_io.list_entries("r2://bucket/staging/")

    @pytest.mark.parametrize("remote_type", [None, "s3"])
    def test_s3_missing_bucket_exit_propagates(
        self, monkeypatch: pytest.MonkeyPatch, remote_type: str | None
    ) -> None:
        """S3's exit-code-3 missing bucket is infrastructure failure, not absence.

        :param monkeypatch: Selects the explicit or default S3 backend.
        :param remote_type: Explicit ``s3`` or unset, whose default is also S3.
        """
        if remote_type is None:
            monkeypatch.delenv("RCLONE_CONFIG_R2_TYPE", raising=False)
        else:
            monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", remote_type)
        completed = MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = ""
        completed.stderr = "Failed to lsjson: directory not found"
        completed.returncode = 3

        with patch.object(r2_io.subprocess, "run", return_value=completed):
            with pytest.raises(subprocess.CalledProcessError):
                r2_io.list_entries("r2://missing-bucket/staging/")

    def test_invokes_rclone_lsjson_with_reliability_flags(self) -> None:
        """Argv pin: the listing probe carries the shared retry/contimeout flags."""
        completed = MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = "[]"
        completed.returncode = 0
        with patch.object(r2_io.subprocess, "run", return_value=completed) as mock_run:
            r2_io.list_entries("r2://bucket/staging/", recursive=True)
        assert mock_run.call_args[0][0] == [
            "rclone",
            "lsjson",
            "--files-only",
            "--use-server-modtime",
            "--retries=3",
            "--contimeout=30s",
            "-R",
            "r2:bucket/staging/",
        ]


class TestLanceTarget:
    """Tests for lance_target — r2:// URI to Lance (uri, storage_options) resolution."""

    def test_local_remote_resolves_to_cwd_relative_path_without_options(
        self, fake_r2_remote: Path
    ) -> None:
        """The local compute mode reads the same bytes rclone writes.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir
            (sets ``RCLONE_CONFIG_R2_TYPE=local`` and chdirs there).
        """
        target, options = r2_io.lance_target("r2://bucket/run/train.lance")

        assert target == str(fake_r2_remote / "bucket" / "run" / "train.lance")
        assert options is None

    def test_non_r2_uri_rejected_in_local_mode_too(self, fake_r2_remote: Path) -> None:
        """A bare path fails fast instead of resolving to a nonsense cwd-relative target.

        :param fake_r2_remote: Local-typed rclone remote (local-backend mode).
        """
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.lance_target("bucket/run/train.lance")

    @pytest.mark.parametrize("remote_type", [None, "s3"])
    def test_s3_remote_resolves_to_s3_uri_with_storage_options(
        self, monkeypatch: pytest.MonkeyPatch, remote_type: str | None
    ) -> None:
        """Unset and explicit-``s3`` remote types both pair the s3:// URI with credentials.

        :param monkeypatch: Pins the R2 credential env vars the options read.
        :param remote_type: ``RCLONE_CONFIG_R2_TYPE`` value; ``None`` leaves it unset.
        """
        if remote_type is None:
            monkeypatch.delenv("RCLONE_CONFIG_R2_TYPE", raising=False)
        else:
            monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", remote_type)
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", "https://r2.example")
        monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", "https://r2.example")

        target, options = r2_io.lance_target("r2://bucket/run/train.lance")

        assert target == "s3://bucket/run/train.lance"
        assert options is not None and options["endpoint"] == "https://r2.example"


class TestPurgePrefix:
    """Tests for purge_prefix — best-effort recursive delete via `rclone purge`."""

    def test_removes_all_objects_under_prefix(self, fake_r2_remote: Path) -> None:
        """Every key under the prefix is gone after the call.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        prefix_root = fake_r2_remote / "bucket" / "runs" / "abc"
        prefix_root.mkdir(parents=True)
        (prefix_root / "shard-000000.lance").write_bytes(b"x")
        (prefix_root / "shard-000001.lance").write_bytes(b"y")

        r2_io.purge_prefix("bucket", "runs/abc/")

        assert not prefix_root.exists()

    def test_leaves_sibling_prefixes_untouched(self, fake_r2_remote: Path) -> None:
        """A purge bounded by ``prefix`` does not touch keys outside it.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        target = fake_r2_remote / "bucket" / "runs" / "abc"
        sibling = fake_r2_remote / "bucket" / "runs" / "xyz"
        target.mkdir(parents=True)
        sibling.mkdir(parents=True)
        (target / "shard.lance").write_bytes(b"x")
        keeper = sibling / "shard.lance"
        keeper.write_bytes(b"y")

        r2_io.purge_prefix("bucket", "runs/abc/")

        assert keeper.exists()

    def test_missing_prefix_does_not_raise(self, fake_r2_remote: Path) -> None:
        """A non-existent prefix is a no-op — `check=False` swallows rclone's exit code.

        Critical for use in ``finally`` blocks, where a missing prefix means the
        test failed before any upload occurred and purge should not mask that.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        r2_io.purge_prefix("bucket", "never-created/")

    def test_invokes_rclone_purge_with_translated_path(self) -> None:
        """Argv shape: ``rclone purge r2:{bucket}/{prefix}`` with rclone + subprocess timeouts."""
        completed = MagicMock(spec=subprocess.CompletedProcess)
        completed.returncode = 0
        with patch.object(r2_io.subprocess, "run", return_value=completed) as mock_run:
            r2_io.purge_prefix("bucket", "runs/abc/")
        args = mock_run.call_args[0][0]
        assert args == [
            "rclone",
            "purge",
            "r2:bucket/runs/abc/",
            "--contimeout=10s",
            "--timeout=60s",
        ]
        kwargs = mock_run.call_args[1]
        assert kwargs.get("check") is False
        assert kwargs.get("capture_output") is True
        assert kwargs.get("timeout") == 120

    @pytest.mark.parametrize("bad_prefix", ["", "/", " ", "runs/abc"])
    def test_rejects_unsafe_prefix(self, bad_prefix: str) -> None:
        """Empty, ``/``, or non-trailing-slash prefixes raise before shelling out.

        Guards against ``rclone purge r2:{bucket}/`` wiping an entire bucket on a
        formatting mistake in caller code.

        :param bad_prefix: Indirect-parametrized prefix value the guard must reject.
        """
        with (
            patch.object(r2_io.subprocess, "run") as mock_run,
            pytest.raises(ValueError, match="purge_prefix refuses"),
        ):
            r2_io.purge_prefix("bucket", bad_prefix)
        mock_run.assert_not_called()


def _set_all_r2_secrets(monkeypatch: pytest.MonkeyPatch, suffix: str = "from-env") -> None:
    """Populate canonical storage settings so `ensure_r2_env_loaded` finds them.

    :param monkeypatch: Pytest fixture used to set env vars.
    :param suffix: Value suffix (lets one targeted test distinguish dotenv vs os.environ origin).
    """
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", f"id-{suffix}")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", f"secret-{suffix}")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", f"endpoint-{suffix}")


class TestEnsureR2EnvLoaded:
    """`ensure_r2_env_loaded` dotenv-loads + validates + auth-pings rclone.

    Tests focus on contract behaviors (no-op without file, missing keys raises, bad auth raises).
    One isolated test confirms the load mechanism actually delivers env values to the rclone
    subprocess; remaining tests don't repeat that process-state introspection.
    """

    @pytest.fixture(autouse=True)
    def _clear_r2_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """Drop storage/rclone env and the workspace dotenv so each test starts from a known state.

        :param tmp_path: Pytest tmp dir used for an intentionally missing default dotenv.
        :param monkeypatch: Pytest fixture used to remove env vars.
        """
        monkeypatch.setattr(r2_io, "_DEFAULT_ENV_FILE", tmp_path / "missing.env")
        for key in list(os.environ):
            if key.startswith(("SYNTH_SETTER_STORAGE_", "RCLONE_CONFIG_R2_")):
                monkeypatch.delenv(key, raising=False)
        yield
        for key in list(os.environ):
            if key.startswith(("SYNTH_SETTER_STORAGE_", "RCLONE_CONFIG_R2_")):
                os.environ.pop(key)

    def test_legacy_dotenv_values_reach_the_rclone_subprocess(self, tmp_path: Path) -> None:
        """Legacy dotenv credentials are projected into the rclone auth-ping env.

        Captures ``os.environ`` at the moment ``subprocess.run`` is invoked — that's
        the contract boundary, and the only place we need to verify it.

        :param tmp_path: Pytest tmp dir for the env_file.
        """
        env_file = tmp_path / ".env"
        env_file.write_text(
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID=id-from-file\n"
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=secret-from-file\n"
            "RCLONE_CONFIG_R2_ENDPOINT=endpoint-from-file\n"
        )
        captured: dict[str, str] = {}

        def _capture(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.update(os.environ)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            r2_io.ensure_r2_env_loaded(env_file)

        assert captured["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "id-from-file"
        assert captured["RCLONE_CONFIG_R2_ENDPOINT"] == "endpoint-from-file"
        assert captured["SYNTH_SETTER_STORAGE_ACCESS_KEY_ID"] == "id-from-file"

    def test_dotenv_settings_persist_for_later_canonical_readers(self, tmp_path: Path) -> None:
        """Settings loaded from an explicit env_file survive into later env-only reads.

        A common local-dev sequence is ``ensure_r2_env_loaded(<.env>)`` followed by
        ``r2_storage_options()`` with no arguments; the preflight must leave the
        canonical keys in ``os.environ`` for that second resolution to succeed.

        :param tmp_path: Pytest tmp dir for the env_file.
        """
        env_file = tmp_path / "custom.env"
        env_file.write_text(
            "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID=id-from-file\n"
            "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY=secret-from-file\n"
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL=endpoint-from-file\n"
        )

        with patch.object(r2_io.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            r2_io.ensure_r2_env_loaded(env_file)

        assert os.environ["SYNTH_SETTER_STORAGE_ACCESS_KEY_ID"] == "id-from-file"
        assert r2_io.r2_storage_options()["access_key_id"] == "id-from-file"

    def test_no_env_file_loads_workspace_dotenv_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``env_file=None`` reads the operator workspace ``.env`` automatically.

        :param tmp_path: Pytest tmp dir for the default dotenv file.
        :param monkeypatch: Pytest fixture used to point the workspace default at the test dotenv
            file.
        """
        default_env_file = tmp_path / ".env"
        default_env_file.write_text(
            "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID=id-from-default\n"
            "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY=secret-from-default\n"
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL=endpoint-from-default\n"
        )
        monkeypatch.setattr(r2_io, "_DEFAULT_ENV_FILE", default_env_file)
        captured: dict[str, str] = {}

        def _capture(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.update(os.environ)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            r2_io.ensure_r2_env_loaded(env_file=None)

        assert captured["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "id-from-default"

    def test_default_env_file_ignores_blank_workspace_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whitespace ``SYNTH_SETTER_WORKSPACE`` falls back to checkout discovery.

        :param tmp_path: Pytest tmp dir used as the cwd that must not be selected.
        :param monkeypatch: Pytest fixture used to isolate env and cwd.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", "   ")

        checkout_env = next(
            parent / ".env"
            for parent in Path(r2_io.__file__).resolve().parents
            if (parent / ".project-root").is_file()
        )

        assert r2_io._default_env_file() == checkout_env  # noqa: SLF001

    def test_missing_secret_keys_raises_actionable_error(self) -> None:
        """Missing storage settings raise an actionable RuntimeError.

        No readable dotenv and no ``SYNTH_SETTER_STORAGE_*`` in os.environ → the
        function names all three missing keys in its error message.
        """
        with pytest.raises(RuntimeError, match="Object storage settings unresolved") as excinfo:
            r2_io.ensure_r2_env_loaded(env_file=None)
        msg = str(excinfo.value)
        for key in STORAGE_REQUIRED_ENV_KEYS:
            assert key in msg

    def test_blank_secret_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A present-but-blank setting is rejected.

        :param monkeypatch: Pytest fixture used to set env vars.
        """
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "   ")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setenv(
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL", "https://stub.r2.cloudflarestorage.com"
        )
        with pytest.raises(RuntimeError, match="SYNTH_SETTER_STORAGE_ACCESS_KEY_ID"):
            r2_io.ensure_r2_env_loaded(env_file=None)

    def test_blank_type_is_normalized_to_default_for_the_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rclone type is projected from storage config's default.

        :param monkeypatch: Pytest fixture used to set env vars.
        """
        import os

        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "id")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setenv(
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL", "https://stub.r2.cloudflarestorage.com"
        )
        captured: dict[str, str] = {}

        def _capture(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.update(os.environ)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            r2_io.ensure_r2_env_loaded(env_file=None)

        assert captured["RCLONE_CONFIG_R2_TYPE"] == "s3"

    def test_padded_env_file_value_is_stripped_for_the_subprocess(self, tmp_path: Path) -> None:
        """A quoted/padded ``.env`` value lands stripped in the auth-ping env, not raw.

        :param tmp_path: Pytest tmp dir for the env_file.
        """
        import os

        env_file = tmp_path / ".env"
        env_file.write_text(
            'SYNTH_SETTER_STORAGE_ACCESS_KEY_ID="  ak  "\n'
            "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY=secret\n"
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL=https://stub.r2.cloudflarestorage.com\n"
        )
        captured: dict[str, str] = {}

        def _capture(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.update(os.environ)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            r2_io.ensure_r2_env_loaded(env_file)

        assert captured["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "ak"

    def test_blank_env_file_value_does_not_clobber_real_process_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blank ``.env`` entry is skipped, so a real process-env credential survives.

        :param tmp_path: Pytest tmp dir for the env_file.
        :param monkeypatch: Pytest fixture used to set the process-env value.
        """
        import os

        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "from-process-env")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setenv(
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL", "https://stub.r2.cloudflarestorage.com"
        )
        env_file = tmp_path / ".env"
        env_file.write_text("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID=\n")
        captured: dict[str, str] = {}

        def _capture(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.update(os.environ)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            r2_io.ensure_r2_env_loaded(env_file)

        assert captured["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "from-process-env"

    def test_auth_ping_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rclone non-zero exit on the auth ping → RuntimeError with stderr excerpt.

        :param monkeypatch: Pytest fixture used to populate secrets.
        """
        _set_all_r2_secrets(monkeypatch)

        with patch.object(r2_io.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="Forbidden: invalid credentials"
            )
            with pytest.raises(RuntimeError, match="rclone failed to authenticate"):
                r2_io.ensure_r2_env_loaded(env_file=None)

    def test_auth_ping_invoked_with_lsd_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The auth ping is `rclone lsd r2:` (bucket-list against the configured remote).

        :param monkeypatch: Pytest fixture used to populate secrets.
        """
        _set_all_r2_secrets(monkeypatch)

        with patch.object(r2_io.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            r2_io.ensure_r2_env_loaded(env_file=None)

        args = mock_run.call_args[0][0]
        assert args[:3] == ["rclone", "lsd", "r2:"]

    def test_auth_ping_has_python_wall_clock_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The auth ping caps wall-clock time at the Python subprocess boundary.

        :param monkeypatch: Pytest fixture used to populate secrets.
        """
        _set_all_r2_secrets(monkeypatch)
        captured: dict[str, object] = {}

        def _capture(*_a: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured.update(kwargs)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            r2_io.ensure_r2_env_loaded(env_file=None)

        assert captured["timeout"] == r2_io._AUTH_PING_TIMEOUT_SECONDS  # noqa: SLF001

    def test_auth_ping_timeout_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hung auth ping raises a timeout-specific RuntimeError.

        :param monkeypatch: Pytest fixture used to populate secrets.
        """
        _set_all_r2_secrets(monkeypatch)

        with patch.object(
            r2_io.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd=["rclone", "lsd", "r2:"], timeout=45),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                r2_io.ensure_r2_env_loaded(env_file=None)

    def test_no_env_file_uses_process_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``env_file=None`` skips dotenv; succeeds when os.environ already has the keys.

        This is the GHA / ``docker run -e ...`` case — vars are injected by the
        runtime and no ``.env`` file is on disk.

        :param monkeypatch: Pytest fixture used to populate secrets.
        """
        _set_all_r2_secrets(monkeypatch)

        with patch.object(r2_io.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            r2_io.ensure_r2_env_loaded(env_file=None)

        mock_run.assert_called_once()

    def test_missing_env_file_uses_process_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-existent ``env_file`` path is treated as 'no file' — falls back to os.environ.

        :param tmp_path: Pytest tmp dir for the (non-existent) env_file path.
        :param monkeypatch: Pytest fixture used to populate secrets in os.environ.
        """
        _set_all_r2_secrets(monkeypatch)
        nonexistent = tmp_path / "missing.env"

        with patch.object(r2_io.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            r2_io.ensure_r2_env_loaded(nonexistent)

        mock_run.assert_called_once()

    def test_defaults_type_and_provider_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`TYPE=s3` / `PROVIDER=Cloudflare` are defaulted into env when callers don't set them.

        rclone's env-override convention (`RCLONE_CONFIG_<remote>_<key>`) needs a complete
        remote definition — without ``TYPE`` and ``PROVIDER`` it reports
        ``didn't find section in config file``. Callers that set only the three secrets
        (the failing matrix step in ``generate-dataset-shards.yaml``) hit this. The function
        defaults the structural keys in-process so the auth ping sees a usable remote.

        :param monkeypatch: Pytest fixture used to populate secrets.
        """
        _set_all_r2_secrets(monkeypatch)
        captured: dict[str, str] = {}

        def _capture(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.update(os.environ)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            r2_io.ensure_r2_env_loaded(env_file=None)

        assert captured["RCLONE_CONFIG_R2_TYPE"] == "s3"
        assert captured["RCLONE_CONFIG_R2_PROVIDER"] == "Cloudflare"

    def test_rclone_projection_overwrites_legacy_type_and_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy rclone structural env is replaced by the storage config projection.

        :param monkeypatch: Pytest fixture used to populate env vars.
        """
        _set_all_r2_secrets(monkeypatch)
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "caller-type")
        monkeypatch.setenv("RCLONE_CONFIG_R2_PROVIDER", "caller-provider")
        captured: dict[str, str] = {}

        def _capture(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.update(os.environ)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            r2_io.ensure_r2_env_loaded(env_file=None)

        assert captured["RCLONE_CONFIG_R2_TYPE"] == "s3"
        assert captured["RCLONE_CONFIG_R2_PROVIDER"] == "Cloudflare"
