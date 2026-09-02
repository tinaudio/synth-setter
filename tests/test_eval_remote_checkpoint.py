"""Behavior tests for eval checkpoint localization from R2-backed URIs."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.cli import eval as eval_module
from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train


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


def test_eval_checkpoint_local_digest_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest-pinned local checkpoint is consumed from an immutable verified copy.

    :param tmp_path: Temporary checkpoint and cache directory.
    :param monkeypatch: Routes the shared cache into the temporary directory.
    """
    checkpoint = tmp_path / "model.ckpt"
    content = b"local checkpoint"
    checkpoint.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    localized = eval_module._localize_eval_checkpoint(str(checkpoint), digest)
    checkpoint.write_bytes(b"replacement checkpoint")

    assert localized != str(checkpoint)
    assert localized is not None
    assert Path(localized).read_bytes() == content
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        eval_module._localize_eval_checkpoint(str(checkpoint), "0" * 64)


def test_eval_checkpoint_r2_uri_refreshes_cache_when_remote_checksum_changes(
    fake_r2_remote: Path,
    storage_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutable recovery URI refreshes when same-size remote bytes change.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param storage_credentials: Dummy application credentials for the local backend.
    :param monkeypatch: Routes the shared cache into the temporary directory.
    """
    source = fake_r2_remote / "bucket" / "runs" / "last.ckpt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"step 99000")
    cache_home = fake_r2_remote / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

    first = eval_module._localize_eval_checkpoint(
        "r2://bucket/runs/last.ckpt",
        hashlib.sha256(b"step 99000").hexdigest(),
    )
    source.write_bytes(b"step final")
    second = eval_module._localize_eval_checkpoint(
        "r2://bucket/runs/last.ckpt",
        hashlib.sha256(b"step final").hexdigest(),
    )

    assert first is not None
    assert second is not None
    assert first != second
    assert Path(first).read_bytes() == b"step 99000"
    assert Path(second).read_bytes() == b"step final"
    assert Path(second).is_relative_to(cache_home / "synth-setter")


def test_eval_checkpoint_cached_digest_is_reused_when_remote_is_unavailable(
    fake_r2_remote: Path,
    storage_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified content-addressed cache entry survives an R2 outage.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param storage_credentials: Dummy application credentials for the local backend.
    :param monkeypatch: Routes the shared cache into the temporary directory.
    """
    source = fake_r2_remote / "bucket" / "runs" / "model.ckpt"
    source.parent.mkdir(parents=True)
    content = b"cached checkpoint"
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))

    first = eval_module._localize_eval_checkpoint("r2://bucket/runs/model.ckpt", digest)
    source.unlink()
    second = eval_module._localize_eval_checkpoint("r2://bucket/runs/model.ckpt", digest)

    assert first == second
    assert first is not None
    assert Path(first).read_bytes() == content


def test_eval_checkpoint_digest_without_checkpoint_raises() -> None:
    """A digest cannot silently accompany an in-memory model evaluation."""
    with pytest.raises(ValueError, match="ckpt_sha256 requires ckpt_path"):
        eval_module._localize_eval_checkpoint(None, "0" * 64)


def test_eval_checkpoint_remote_uri_without_digest_raises() -> None:
    """Remote checkpoints require immutable content provenance."""
    with pytest.raises(ValueError, match="requires ckpt_sha256"):
        eval_module._localize_eval_checkpoint("r2://bucket/runs/model.ckpt")


def test_eval_checkpoint_non_string_path_raises() -> None:
    """Malformed checkpoint path types fail as configuration errors."""
    with pytest.raises(ValueError, match="ckpt_path must be a string"):
        eval_module._localize_eval_checkpoint(cast("str", 123))


def test_eval_checkpoint_non_string_digest_raises() -> None:
    """Malformed digest types fail with the documented configuration error."""
    with pytest.raises(ValueError, match="64 hexadecimal characters"):
        eval_module._localize_eval_checkpoint(
            "r2://bucket/runs/model.ckpt",
            expected_sha256=cast("str", 123),
        )


def test_eval_checkpoint_malformed_string_digest_raises() -> None:
    """Malformed digest text fails before any remote access."""
    with pytest.raises(ValueError, match="64 hexadecimal characters"):
        eval_module._localize_eval_checkpoint("r2://bucket/model.ckpt", "not-a-digest")


def test_eval_checkpoint_none_without_digest_uses_in_memory_model() -> None:
    """An explicitly uncheckpointed evaluation keeps Lightning's in-memory model."""
    assert eval_module._localize_eval_checkpoint(None) is None


def test_eval_checkpoint_incomplete_download_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated remote transfer never enters the content-addressed cache.

    :param tmp_path: Isolated checkpoint cache.
    :param monkeypatch: Stubs the remote transport at the failure boundary.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(eval_module.r2_io, "ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(eval_module.r2_io, "object_size", lambda _uri: 20)
    monkeypatch.setattr(
        eval_module.r2_io,
        "download_to_path",
        lambda _uri, path: path.write_bytes(b"short"),
    )

    with pytest.raises(RuntimeError, match="downloaded eval checkpoint is incomplete"):
        eval_module._localize_eval_checkpoint("r2://bucket/model.ckpt", "0" * 64)


def test_eval_checkpoint_transport_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed remote transfer is translated into checkpoint-specific guidance.

    :param tmp_path: Isolated checkpoint cache.
    :param monkeypatch: Stubs the remote transport at the failure boundary.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(eval_module.r2_io, "ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(eval_module.r2_io, "object_size", lambda _uri: 20)

    def _fail_download(_uri: str, _path: Path) -> None:
        raise subprocess.CalledProcessError(1, "rclone")

    monkeypatch.setattr(eval_module.r2_io, "download_to_path", _fail_download)

    with pytest.raises(RuntimeError, match="rclone cannot download eval checkpoint"):
        eval_module._localize_eval_checkpoint("r2://bucket/model.ckpt", "0" * 64)


def test_eval_checkpoint_remote_digest_mismatch_raises(
    fake_r2_remote: Path,
    storage_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote checkpoint must match its configured content digest.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param storage_credentials: Dummy application credentials for the local backend.
    :param monkeypatch: Routes the shared cache into the temporary directory.
    """
    source = fake_r2_remote / "bucket" / "runs" / "model.ckpt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"unexpected checkpoint")
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))

    expected = "0" * 64
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="SHA-256 mismatch") as exc_info:
        eval_module._localize_eval_checkpoint(
            "r2://bucket/runs/model.ckpt",
            expected_sha256=expected,
        )

    assert expected in str(exc_info.value)
    assert actual in str(exc_info.value)
    cached = (
        fake_r2_remote
        / "cache"
        / "synth-setter"
        / "checkpoints"
        / "evaluation"
        / expected
        / "model.ckpt"
    )
    assert not cached.exists()


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

    localized = eval_module._localize_eval_checkpoint(
        "s3://bucket/runs/model.ckpt",
        hashlib.sha256(b"s3 checkpoint").hexdigest(),
    )

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
        eval_module._localize_eval_checkpoint("r2://bucket/runs/missing.ckpt", "0" * 64)


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
        eval_module._localize_eval_checkpoint(
            "r2://bucket/runs/empty.ckpt", hashlib.sha256(b"").hexdigest()
        )


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
        eval_module._localize_eval_checkpoint("r2://bucket/runs/model.ckpt", "0" * 64)


def test_evaluate_consumes_real_checkpoint_downloaded_from_r2(
    cfg_train: DictConfig,
    cfg_eval: DictConfig,
    fake_r2_remote: Path,
    storage_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lightning evaluates a real train-produced checkpoint fetched through rclone.

    :param cfg_train: Tiny TorchSynth CPU training configuration.
    :param cfg_eval: Matching TorchSynth CPU evaluation configuration.
    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param storage_credentials: Dummy application credentials for the local backend.
    :param monkeypatch: Routes the shared cache into the temporary directory.
    """
    for cfg in (cfg_train, cfg_eval):
        with open_dict(cfg):
            cfg.datamodule.signal_length = 512
            cfg.model.net.channels = 2
            cfg.model.net.encoder_blocks = 1
            cfg.model.net.hidden_dim = 8
            cfg.model.net.norm = "ln"
            cfg.model.net.trunk_blocks = 1
    with open_dict(cfg_train):
        cfg_train.test = False
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
    with open_dict(cfg_eval):
        cfg_eval.trainer.limit_test_batches = 1

    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    local_checkpoint = Path(cfg_train.paths.output_dir) / "checkpoints" / "last.ckpt"
    remote_checkpoint = fake_r2_remote / "bucket" / "runs" / "last.ckpt"
    remote_checkpoint.parent.mkdir(parents=True)
    shutil.copyfile(local_checkpoint, remote_checkpoint)
    original_uri = "r2://bucket/runs/last.ckpt"
    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = original_uri
        cfg_eval.ckpt_sha256 = hashlib.sha256(local_checkpoint.read_bytes()).hexdigest()
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))

    HydraConfig().set_config(cfg_eval)
    metrics, objects = evaluate(cfg_eval)

    assert math.isfinite(metrics["test/param_mse"].item())
    assert cfg_eval.ckpt_path == original_uri
    assert Path(objects["trainer"].ckpt_path).is_file()
    assert objects["trainer"].ckpt_path != original_uri
