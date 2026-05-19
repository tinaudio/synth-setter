"""Tests for synth_setter.pipeline.r2_io — rclone-backed R2 I/O helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synth_setter.pipeline import r2_io


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


class TestDownloadToPath:
    """Tests for download_to_path — file→file copy."""

    def test_invokes_rclone_copyto_with_checksum(self, tmp_path: Path) -> None:
        """Verifies the rclone command shape (copyto + --checksum + URI translation)."""
        dest = tmp_path / "out.json"
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            r2_io.download_to_path("r2://bucket/key.json", dest)
        args = mock_call.call_args[0][0]
        assert args[:3] == ["rclone", "copyto", "--checksum"]
        assert args[3] == "r2:bucket/key.json"
        assert args[4] == str(dest)

    def test_rejects_non_r2_uri(self, tmp_path: Path) -> None:
        """A local-path source is rejected — caller must branch on is_r2_uri."""
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.download_to_path("local-spec.json", tmp_path / "out.json")


class TestUploadToUri:
    """Tests for upload_to_uri — file→file upload with reliability flags."""

    def test_invokes_rclone_copyto_with_reliability_flags(self, tmp_path: Path) -> None:
        """Upload command includes -vv, --contimeout, --timeout, --retries, --checksum."""
        src = tmp_path / "in.json"
        src.write_text("{}")
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            r2_io.upload_to_uri(src, "r2://bucket/key.json")
        args = mock_call.call_args[0][0]
        assert args[:2] == ["rclone", "copyto"]
        assert "-vv" in args
        assert "--checksum" in args
        assert "--contimeout=30s" in args
        assert "--timeout=300s" in args
        assert "--retries=3" in args
        assert args[-2] == str(src)
        assert args[-1] == "r2:bucket/key.json"

    def test_rejects_non_r2_uri(self, tmp_path: Path) -> None:
        """A local-path destination is rejected."""
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.upload_to_uri(tmp_path / "in.json", "local-dest.json")


class TestDownloadedToTempfile:
    """Tests for the downloaded_to_tempfile context manager."""

    def test_yields_local_path_named_after_uri_basename(self) -> None:
        """Yielded path name matches URI's last segment; tempdir is cleaned on exit."""
        seen: list[Path] = []

        def fake_check_call(args: list[str]) -> None:
            Path(args[-1]).write_text('{"ok": true}')

        with patch.object(r2_io.subprocess, "check_call", side_effect=fake_check_call):
            with r2_io.downloaded_to_tempfile("r2://bucket/path/spec.json") as local:
                seen.append(local)
                assert local.name == "spec.json"
                assert local.read_text() == '{"ok": true}'

        assert not seen[0].exists()

    def test_cleanup_runs_even_on_exception(self) -> None:
        """If the with-block raises, the tempdir is still cleaned up."""
        seen: list[Path] = []

        def fake_check_call(args: list[str]) -> None:
            Path(args[-1]).write_text("{}")

        with patch.object(r2_io.subprocess, "check_call", side_effect=fake_check_call):
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
            r2_io.shard_uri("intermediate-data", "data/run-x/", "shard-000007.h5")
            == "r2://intermediate-data/data/run-x/shard-000007.h5"
        )

    def test_preserves_nested_prefix(self) -> None:
        """Multi-segment prefixes are joined verbatim (caller controls trailing slash)."""
        assert (
            r2_io.shard_uri("bucket", "a/b/c/", "shard-000000.h5")
            == "r2://bucket/a/b/c/shard-000000.h5"
        )


class TestObjectSize:
    """Tests for object_size — existence + size probe via `rclone lsf --format=s`."""

    @staticmethod
    def _mock_run(stdout: str) -> MagicMock:
        """Build a CompletedProcess-shaped mock with the given stdout."""
        completed = MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = stdout
        completed.returncode = 0
        return completed

    def test_present_returns_int_size(self) -> None:
        """A non-empty integer stdout means the object exists; return its size in bytes."""
        with patch.object(r2_io.subprocess, "run", return_value=self._mock_run("12345\n")):
            assert r2_io.object_size("r2://bucket/key.h5") == 12345

    def test_absent_returns_none(self) -> None:
        """Empty stdout means the object is missing; return None."""
        with patch.object(r2_io.subprocess, "run", return_value=self._mock_run("")):
            assert r2_io.object_size("r2://bucket/key.h5") is None

    def test_zero_size_returns_zero(self) -> None:
        """A zero-byte object exists; return 0 (callers decide whether to treat as present)."""
        with patch.object(r2_io.subprocess, "run", return_value=self._mock_run("0\n")):
            assert r2_io.object_size("r2://bucket/key.h5") == 0

    def test_probe_failure_propagates(self) -> None:
        """Non-zero rclone exit raises CalledProcessError — fail-fast on env issues."""
        err = subprocess.CalledProcessError(returncode=1, cmd=["rclone"])
        with patch.object(r2_io.subprocess, "run", side_effect=err):
            with pytest.raises(subprocess.CalledProcessError):
                r2_io.object_size("r2://bucket/key.h5")

    def test_invokes_rclone_lsf_format_s(self) -> None:
        """Probe argv shape: rclone lsf --format=s <translated path>."""
        with patch.object(r2_io.subprocess, "run", return_value=self._mock_run("42")) as mock_run:
            r2_io.object_size("r2://bucket/path/key.h5")
        args = mock_run.call_args[0][0]
        assert args == ["rclone", "lsf", "--format=s", "r2:bucket/path/key.h5"]
        kwargs = mock_run.call_args[1]
        assert kwargs.get("check") is True
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True

    def test_rejects_non_r2_uri(self) -> None:
        """Local paths are rejected via _to_rclone_path."""
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.object_size("local/key.h5")


def _set_all_r2_secrets(monkeypatch: pytest.MonkeyPatch, suffix: str = "from-env") -> None:
    """Populate `RCLONE_CONFIG_R2_*` secrets so `ensure_r2_env_loaded` finds them.

    :param monkeypatch: Pytest fixture used to set env vars.
    :param suffix: Value suffix (lets one targeted test distinguish dotenv vs os.environ origin).
    """
    monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", f"id-{suffix}")
    monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", f"secret-{suffix}")
    monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", f"endpoint-{suffix}")


class TestEnsureR2EnvLoaded:
    """`ensure_r2_env_loaded` dotenv-loads + validates + auth-pings rclone.

    Tests focus on contract behaviors (no-op without file, missing keys raises, bad auth raises).
    One isolated test confirms the load mechanism actually delivers env values to the rclone
    subprocess; remaining tests don't repeat that process-state introspection.
    """

    @pytest.fixture(autouse=True)
    def _clear_r2_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drop any `RCLONE_CONFIG_R2_*` keys so each test starts from a known state.

        :param monkeypatch: Pytest fixture used to remove env vars.
        """
        import os

        for key in list(os.environ):
            if key.startswith("RCLONE_CONFIG_R2_"):
                monkeypatch.delenv(key, raising=False)

    def test_dotenv_values_reach_the_rclone_subprocess(self, tmp_path: Path) -> None:
        """An env_file's secrets are visible in the rclone auth-ping's environment.

        Captures ``os.environ`` at the moment ``subprocess.run`` is invoked — that's
        the contract boundary, and the only place we need to verify it.

        :param tmp_path: Pytest tmp dir for the env_file.
        """
        import os

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

    def test_missing_secret_keys_raises_actionable_error(self) -> None:
        """Missing R2 secret keys raise an actionable RuntimeError.

        No env_file and no ``RCLONE_CONFIG_R2_*`` in os.environ → the function
        names all three missing keys in its error message.
        """
        with pytest.raises(RuntimeError, match="R2 credentials missing") as excinfo:
            r2_io.ensure_r2_env_loaded(env_file=None)
        msg = str(excinfo.value)
        for key in (
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
            "RCLONE_CONFIG_R2_ENDPOINT",
        ):
            assert key in msg

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
