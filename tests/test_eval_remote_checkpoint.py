"""Behavior tests for eval checkpoint localization from R2-backed URIs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from synth_setter.cli import eval as eval_module


@pytest.fixture()
def storage_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the local-backed rclone remote through application settings.

    :param monkeypatch: Environment override fixture.
    """
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", "http://localhost:0")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_RCLONE_TYPE", "local")


def test_eval_checkpoint_local_path_is_returned_unchanged(tmp_path: Path) -> None:
    """A local checkpoint remains the path supplied by the run config.

    :param tmp_path: Temporary checkpoint directory.
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"local checkpoint")

    assert eval_module._localize_eval_checkpoint(str(checkpoint)) == str(checkpoint)


def test_eval_checkpoint_r2_uri_refreshes_deterministic_cache_when_remote_changes(
    fake_r2_remote: Path,
    storage_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An R2 checkpoint refreshes the same cache path when remote bytes change.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param storage_credentials: Dummy application credentials for the local backend.
    :param monkeypatch: Routes the shared cache into the temporary directory.
    """
    source = fake_r2_remote / "bucket" / "runs" / "model.ckpt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"remote checkpoint")
    cache_home = fake_r2_remote / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

    first = eval_module._localize_eval_checkpoint("r2://bucket/runs/model.ckpt")
    source.write_bytes(b"updated checkpoint")
    second = eval_module._localize_eval_checkpoint("r2://bucket/runs/model.ckpt")

    assert first is not None
    assert first == second
    assert Path(first).read_bytes() == b"updated checkpoint"
    assert Path(first).is_relative_to(cache_home / "synth-setter")


def test_eval_checkpoint_s3_uri_downloads_through_r2_remote(
    fake_r2_remote: Path,
    storage_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An S3-spelled R2 object is localized through the repository's R2 convention.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param storage_credentials: Dummy application credentials for the local backend.
    :param monkeypatch: Routes the shared cache into the temporary directory.
    """
    source = fake_r2_remote / "bucket" / "runs" / "model.ckpt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"s3 checkpoint")
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))

    localized = eval_module._localize_eval_checkpoint("s3://bucket/runs/model.ckpt")

    assert localized is not None
    assert Path(localized).read_bytes() == b"s3 checkpoint"


def test_eval_checkpoint_missing_remote_object_raises_clear_error(
    fake_r2_remote: Path,
    storage_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing R2 object is identified before Lightning starts.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param storage_credentials: Dummy application credentials for the local backend.
    :param monkeypatch: Routes the shared cache into the temporary directory.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))

    with pytest.raises(FileNotFoundError, match="remote eval checkpoint does not exist"):
        eval_module._localize_eval_checkpoint("r2://bucket/runs/missing.ckpt")


def test_eval_checkpoint_empty_remote_object_raises_clear_error(
    fake_r2_remote: Path,
    storage_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-byte R2 object is rejected before it enters the shared cache.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param storage_credentials: Dummy application credentials for the local backend.
    :param monkeypatch: Routes the shared cache into the temporary directory.
    """
    source = fake_r2_remote / "bucket" / "runs" / "empty.ckpt"
    source.parent.mkdir(parents=True)
    source.touch()
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))

    with pytest.raises(RuntimeError, match="remote eval checkpoint is empty"):
        eval_module._localize_eval_checkpoint("r2://bucket/runs/empty.ckpt")


def test_eval_checkpoint_unavailable_credentials_raise_clear_error(
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unavailable rclone remote reports credential guidance without secrets.

    :param fake_r2_remote: Activates the local-backed fixture before its config is removed.
    :param monkeypatch: Removes storage settings and isolates rclone configuration.
    :param tmp_path: Holds an empty dotenv and rclone config.
    """
    for key in tuple(os.environ):
        if key.startswith(("RCLONE_CONFIG_R2_", "SYNTH_SETTER_STORAGE_")):
            monkeypatch.delenv(key, raising=False)
    empty_env = tmp_path / "empty.env"
    empty_env.touch()
    empty_rclone_config = tmp_path / "rclone.conf"
    empty_rclone_config.touch()
    monkeypatch.setattr(eval_module.r2_io, "_DEFAULT_ENV_FILE", empty_env)
    monkeypatch.setenv("RCLONE_CONFIG", str(empty_rclone_config))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with pytest.raises(RuntimeError, match="rclone R2 credentials are unavailable"):
        eval_module._localize_eval_checkpoint("r2://bucket/runs/model.ckpt")
