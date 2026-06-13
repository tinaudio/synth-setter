"""Tests for synth_setter.pipeline.r2_io — rclone-backed R2 I/O helpers.

State-based tests use the ``fake_r2_remote`` fixture (see ``tests/pipeline/
conftest.py``): rclone runs against a local-typed remote rooted at a tmp dir,
so each test asserts on the filesystem state after the helper runs instead of
on the rclone argv list. Two narrow argv-shape tests survive (the reliability-
flag set on ``upload_to_uri`` and the ``lsf --format=s`` shape on
``object_size``) to pin invariants state-based tests cannot observe.
"""

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


class TestRcloneArgv:
    """Tests for _rclone_argv — the centralized reliability-flag builder."""

    def test_default_timeout_emits_full_reliability_flag_set(self) -> None:
        """The default builder pins the verb, reliability flags, then operands in order."""
        assert r2_io._rclone_argv("copyto", "r2:bucket/key", "dest/file") == [
            "rclone",
            "copyto",
            "-vv",
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
            "-vv",
            "--checksum",
            "--contimeout=30s",
            "--timeout=3h",
            "--retries=3",
            "src/dir",
            "r2:bucket/p",
        ]


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


class TestR2StorageOptions:
    """Tests for r2_storage_options — object_store kwargs for lance's native S3 backend."""

    _SECRETS = {
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "ak-123",
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "sk-456",
        "RCLONE_CONFIG_R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com",
    }

    def test_maps_env_secrets_to_object_store_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The three R2 secrets land on the aws_* keys object_store reads, with region 'auto'.

        :param monkeypatch: Pytest fixture used to set the R2 secret env vars.
        """
        for key, value in self._SECRETS.items():
            monkeypatch.setenv(key, value)

        assert r2_io.r2_storage_options() == {
            "aws_access_key_id": "ak-123",
            "aws_secret_access_key": "sk-456",
            "aws_endpoint": "https://acct.r2.cloudflarestorage.com",
            "aws_region": "auto",
        }

    @pytest.mark.parametrize("absent_key", list(_SECRETS))
    def test_missing_secret_raises_listing_the_absent_key(
        self, monkeypatch: pytest.MonkeyPatch, absent_key: str
    ) -> None:
        """Any absent secret raises RuntimeError naming that key, not a bare KeyError.

        :param monkeypatch: Pytest fixture used to set all secrets then drop one.
        :param absent_key: The single secret dropped before the call.
        """
        for key, value in self._SECRETS.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delenv(absent_key)

        with pytest.raises(RuntimeError, match=absent_key):
            r2_io.r2_storage_options()


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
        assert "-vv" in args
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
        (prefix / "train.h5").write_text("train")
        (prefix / "stats.npz").write_text("stats")
        dest = tmp_path / "root"

        r2_io.download_dir_no_overwrite("r2://bucket/dataset", dest)

        assert (dest / "train.h5").read_text() == "train"
        assert (dest / "stats.npz").read_text() == "stats"

    def test_rejects_non_r2_uri(self, tmp_path: Path) -> None:
        """A local-path source is rejected — caller must branch on is_r2_uri.

        :param tmp_path: Pytest tmp dir used to build a local destination path.
        """
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.download_dir_no_overwrite("local-dir", tmp_path / "root")

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
        assert "-vv" in args
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

    def test_command_carries_rclone_reliability_flags(self, tmp_path: Path) -> None:
        """Pin the rclone reliability-flag set on upload.

        State-based tests cover the file-landing contract but cannot observe the
        ``-vv / --checksum / --contimeout / --timeout / --retries`` flags. Losing
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
        assert "-vv" in args
        assert "--checksum" in args
        assert "--contimeout=30s" in args
        assert "--timeout=300s" in args
        assert "--retries=3" in args


class TestIsR2Reachable:
    """Tests for ``is_r2_reachable`` — boolean auth-probe used as a test-skip gate."""

    def test_returns_true_when_rclone_lsd_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: rclone on PATH, env keys present, probe exits 0.

        :param monkeypatch: Pytest fixture used to stub PATH + env + ``subprocess.run``.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        for key in r2_io._SECRET_R2_ENV_KEYS:  # noqa: SLF001 — test asserts contract
            monkeypatch.setenv(key, "stub")

        class _OK:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr("synth_setter.pipeline.r2_io.subprocess.run", lambda *a, **kw: _OK())
        assert r2_io.is_r2_reachable() is True

    def test_returns_false_when_rclone_lsd_exits_non_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auth failure: rclone + env keys present but the probe exits non-zero.

        :param monkeypatch: Pytest fixture used to stub PATH + env + ``subprocess.run``.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        for key in r2_io._SECRET_R2_ENV_KEYS:  # noqa: SLF001 — test asserts contract
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
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rclone-on-PATH + working local config but no env keys → skip, not hard-fail later.

        Mirrors the contract of ``ensure_r2_env_loaded`` so a test that
        gates on ``is_r2_reachable`` doesn't pass the gate and then crash
        on ``RuntimeError`` from the env-key check downstream.

        :param monkeypatch: Pytest fixture used to clear env + stub the probe.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        for key in r2_io._SECRET_R2_ENV_KEYS:  # noqa: SLF001 — test asserts contract
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
    """Tests for object_size — existence + size probe via `rclone lsf --format=s`.

    Present and zero-size cases are state-based against the fake-local remote.
    Absent and probe-failure cases stay mock-based: on R2 ``lsf`` returns empty
    stdout for a missing key (the bucket exists, the key does not); on the
    local backend the parent directory may not exist, so ``lsf`` exits non-zero
    instead — a behavior divergence the fixture deliberately leaves outside
    its coverage (see issue #1124's "Out of scope" note).
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
        obj = fake_r2_remote / "bucket" / "key.h5"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b"x" * 12345)

        assert r2_io.object_size("r2://bucket/key.h5") == 12345

    def test_zero_size_returns_zero(self, fake_r2_remote: Path) -> None:
        """A zero-byte object exists; return 0 (callers decide whether to treat as present).

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        obj = fake_r2_remote / "bucket" / "key.h5"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b"")

        assert r2_io.object_size("r2://bucket/key.h5") == 0

    def test_absent_returns_none(self) -> None:
        """Empty stdout means the object is missing; return None.

        R2-specific behavior — see class docstring for why this stays mock-based.
        """
        with patch.object(r2_io.subprocess, "run", return_value=self._mock_run("")):
            assert r2_io.object_size("r2://bucket/key.h5") is None

    def test_probe_failure_propagates(self) -> None:
        """Non-zero rclone exit raises CalledProcessError — fail-fast on env issues."""
        err = subprocess.CalledProcessError(returncode=1, cmd=["rclone"])
        with patch.object(r2_io.subprocess, "run", side_effect=err):
            with pytest.raises(subprocess.CalledProcessError):
                r2_io.object_size("r2://bucket/key.h5")

    def test_non_integer_stdout_raises_with_original_cause(self) -> None:
        """Unparsable rclone stdout raises a contextual error chained to the ValueError.

        Guards the generate failure path: ``int(out)`` on garbage stdout would
        otherwise surface a bare ``invalid literal for int()`` with no probed
        URI, losing the rclone-listing context. The re-raise must keep the
        original ``ValueError`` as ``__cause__``.
        """
        with patch.object(r2_io.subprocess, "run", return_value=self._mock_run("not-a-number")):
            with pytest.raises(RuntimeError, match="r2://bucket/key.h5") as excinfo:
                r2_io.object_size("r2://bucket/key.h5")
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_invokes_rclone_lsf_format_s(self) -> None:
        """Probe argv shape: rclone lsf --format=s <translated path>.

        Pins the ``lsf --format=s`` probe shape; the state-based ``present``/
        ``zero_size`` tests above exercise the happy paths but cannot observe
        the exact argv, and a wrong flag here would silently break the absent
        detection (which relies on empty stdout).
        """
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


class TestPurgePrefix:
    """Tests for purge_prefix — best-effort recursive delete via `rclone purge`."""

    def test_removes_all_objects_under_prefix(self, fake_r2_remote: Path) -> None:
        """Every key under the prefix is gone after the call.

        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        prefix_root = fake_r2_remote / "bucket" / "runs" / "abc"
        prefix_root.mkdir(parents=True)
        (prefix_root / "shard-000000.h5").write_bytes(b"x")
        (prefix_root / "shard-000001.h5").write_bytes(b"y")

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
        (target / "shard.h5").write_bytes(b"x")
        keeper = sibling / "shard.h5"
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

    def test_defaults_type_and_provider_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`TYPE=s3` / `PROVIDER=Cloudflare` are defaulted into env when callers don't set them.

        rclone's env-override convention (`RCLONE_CONFIG_<remote>_<key>`) needs a complete
        remote definition — without ``TYPE`` and ``PROVIDER`` it reports
        ``didn't find section in config file``. Callers that set only the three secrets
        (the failing matrix step in ``generate-dataset-shards.yaml``) hit this. The function
        defaults the structural keys in-process so the auth ping sees a usable remote.

        :param monkeypatch: Pytest fixture used to populate secrets.
        """
        import os

        _set_all_r2_secrets(monkeypatch)
        captured: dict[str, str] = {}

        def _capture(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.update(os.environ)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            r2_io.ensure_r2_env_loaded(env_file=None)

        assert captured["RCLONE_CONFIG_R2_TYPE"] == "s3"
        assert captured["RCLONE_CONFIG_R2_PROVIDER"] == "Cloudflare"

    def test_does_not_overwrite_caller_provided_type_and_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller's ``TYPE`` / ``PROVIDER`` win — the defaults only fill the unset case.

        Preserves the rclone env-override design: callers (e.g. a future non-Cloudflare
        S3-compatible backend) can override the structural keys without having to opt out
        of ``ensure_r2_env_loaded``.

        :param monkeypatch: Pytest fixture used to populate env vars.
        """
        import os

        _set_all_r2_secrets(monkeypatch)
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "caller-type")
        monkeypatch.setenv("RCLONE_CONFIG_R2_PROVIDER", "caller-provider")
        captured: dict[str, str] = {}

        def _capture(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.update(os.environ)
            return subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(r2_io.subprocess, "run", side_effect=_capture):
            r2_io.ensure_r2_env_loaded(env_file=None)

        assert captured["RCLONE_CONFIG_R2_TYPE"] == "caller-type"
        assert captured["RCLONE_CONFIG_R2_PROVIDER"] == "caller-provider"
