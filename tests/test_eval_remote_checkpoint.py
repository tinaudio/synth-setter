"""Behavior tests for eval checkpoint localization from R2-backed URIs."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

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

    first = eval_module._localize_eval_checkpoint("r2://bucket/runs/last.ckpt")
    source.write_bytes(b"step final")
    second = eval_module._localize_eval_checkpoint("r2://bucket/runs/last.ckpt")

    assert first is not None
    assert second is not None
    assert first == second
    assert Path(second).read_bytes() == b"step final"
    assert Path(second).is_relative_to(cache_home / "synth-setter")


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
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))

    HydraConfig().set_config(cfg_eval)
    metrics, objects = evaluate(cfg_eval)

    assert math.isfinite(metrics["test/param_mse"].item())
    assert cfg_eval.ckpt_path == original_uri
    assert Path(objects["trainer"].ckpt_path).is_file()
    assert objects["trainer"].ckpt_path != original_uri
