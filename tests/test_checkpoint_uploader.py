"""Test checkpoint mirroring without invoking rclone or the network."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli.train import (
    _checkpoint_prefix_uri,
    _configure_checkpoint_durability,
    _make_launch_namespace,
)
from synth_setter.pipeline import r2_io
from synth_setter.utils import callbacks as callbacks_module
from synth_setter.utils.callbacks import CheckpointUploader


@pytest.fixture(autouse=True)
def _stub_durability_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests offline unless they explicitly replace the R2 preflight.

    :param monkeypatch: Replaces the R2 environment/authentication probe.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda: None)


def _raise_programming_error(_local_path: Path, _uri: str) -> None:
    """Raise an unexpected error from a patched upload transport.

    :param _local_path: Unused local checkpoint path.
    :param _uri: Unused remote checkpoint URI.
    :raises RuntimeError: Always, to model a programming error.
    """
    raise RuntimeError("programming bug")


class _FakeCheckpointCallback:
    """Stand-in for Lightning's ``ModelCheckpoint`` exposing ``last_model_path``."""

    def __init__(self, last_model_path: str) -> None:
        """Bind the checkpoint path.

        :param last_model_path: Path ``ModelCheckpoint`` last wrote (``""`` if none).
        """
        self.last_model_path = last_model_path


class _FakeTrainer:
    """Minimal trainer double carrying the rank flag and the checkpoint callback."""

    def __init__(
        self, last_model_path: str = "", *, is_global_zero: bool = True, world_size: int = 1
    ) -> None:
        """Bind the rank flag, world size, and the checkpoint callback double.

        :param last_model_path: Path exposed via ``checkpoint_callback.last_model_path``.
        :param is_global_zero: Whether this stands in for the rank-0 process.
        :param world_size: Number of processes; ``> 1`` exercises the DDP warning.
        """
        self.is_global_zero = is_global_zero
        self.world_size = world_size
        self.checkpoint_callback = _FakeCheckpointCallback(last_model_path)
        self.callbacks: list[Callback] = []


def _trainer(
    last_model_path: str = "", *, is_global_zero: bool = True, world_size: int = 1
) -> Trainer:
    """Return a ``_FakeTrainer`` typed as ``Trainer`` for the callback hook.

    :param last_model_path: Path exposed via ``checkpoint_callback.last_model_path``.
    :param is_global_zero: Whether this stands in for the rank-0 process.
    :param world_size: Number of processes; ``> 1`` exercises the DDP warning.
    :returns: The fake trainer cast to ``Trainer`` so the typed hook accepts it.
    """
    return cast(
        Trainer,
        _FakeTrainer(last_model_path, is_global_zero=is_global_zero, world_size=world_size),
    )


def _cfg(
    *,
    during_training: bool | None = None,
    upload_checkpoints_uri: str | None = None,
) -> DictConfig:
    """Build a minimal train cfg for the prefix/append helpers.

    :param during_training: Value for ``training.upload_checkpoints_during_training``;
        ``None`` omits the key entirely (mirrors a config that never set it).
    :param upload_checkpoints_uri: Optional verbatim ``r2://`` upload-target override.
    :returns: A DictConfig carrying ``task_name``, ``r2.bucket``, and ``training``.
    """
    training: dict[str, object] = {"upload_checkpoints_uri": upload_checkpoints_uri}
    if during_training is not None:
        training["upload_checkpoints_during_training"] = during_training
    return cast(
        DictConfig,
        OmegaConf.create(
            {
                "task_name": "flow-simple",
                "r2": {"bucket": "intermediate-data"},
                "training": training,
            }
        ),
    )


def _stub_r2(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str]]:
    """Patch R2 env-load to a no-op and record every ``upload_to_uri`` call.

    :param monkeypatch: Pytest fixture used to swap the R2 transport.
    :returns: The list each ``(local_path, uri)`` upload is appended to.
    """
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        r2_io, "upload_to_uri", lambda local_path, uri: calls.append((local_path, uri))
    )
    return calls


def test_uploader_uploads_new_checkpoint_to_prefixed_last_ckpt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A freshly-written ``last_model_path`` uploads to ``{prefix}/last.ckpt``.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    :param caplog: Captures the recovery URI emitted after a successful upload.
    """
    calls = _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://intermediate-data/checkpoints/flow-simple")
    with caplog.at_level(logging.INFO):
        uploader.on_train_batch_end(_trainer(str(ckpt)), None, None, None, 0)
    assert calls == [(ckpt, "r2://intermediate-data/checkpoints/flow-simple/last.ckpt")]
    assert "uploaded to r2://intermediate-data/checkpoints/flow-simple/last.ckpt" in caplog.text


def test_uploader_skips_reupload_when_path_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same ``last_model_path`` across batches uploads exactly once.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    """
    calls = _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    trainer = _trainer(str(ckpt))
    uploader.on_train_batch_end(trainer, None, None, None, 0)
    uploader.on_train_batch_end(trainer, None, None, None, 1)
    assert len(calls) == 1


def test_uploader_reuploads_when_last_ckpt_rewritten_in_place(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Use file metadata, not only its stable path, to detect in-place rewrites.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds the ``last.ckpt`` rewritten in place between batches.
    """
    calls = _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"v1")
    os.utime(ckpt, (1000, 1000))
    uploader = CheckpointUploader("r2://b/c")
    trainer = _trainer(str(ckpt))
    uploader.on_train_batch_end(trainer, None, None, None, 0)
    ckpt.write_bytes(b"v2")
    os.utime(ckpt, (2000, 2000))  # newer mtime = a fresh ModelCheckpoint save
    uploader.on_train_batch_end(trainer, None, None, None, 1)
    assert [uri for _, uri in calls] == ["r2://b/c/last.ckpt", "r2://b/c/last.ckpt"]


def test_uploader_skips_non_global_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-rank-0 process never uploads (DDP ranks must not race duplicates).

    :param monkeypatch: Swaps the R2 transport for a recorder.
    """
    calls = _stub_r2(monkeypatch)
    uploader = CheckpointUploader("r2://b/c")
    uploader.on_train_batch_end(
        _trainer("/x/last.ckpt", is_global_zero=False), None, None, None, 0
    )
    assert calls == []


def test_uploader_noop_before_first_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty ``last_model_path`` (no checkpoint yet) uploads nothing.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    """
    calls = _stub_r2(monkeypatch)
    uploader = CheckpointUploader("r2://b/c")
    uploader.on_train_batch_end(_trainer(""), None, None, None, 0)
    assert calls == []


def test_uploader_swallows_upload_error_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed upload is logged and swallowed, never propagating out of the hook.

    :param monkeypatch: Swaps the R2 transport for a raising stub.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    :param caplog: Captures the warning the swallowed failure must still emit.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *_args, **_kwargs: None)

    def _boom(_local_path: Path, _uri: str) -> None:
        raise subprocess.CalledProcessError(1, ["rclone", "copyto"])

    monkeypatch.setattr(r2_io, "upload_to_uri", _boom)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    with caplog.at_level(logging.WARNING):
        uploader.on_train_batch_end(_trainer(str(ckpt)), None, None, None, 0)  # must not raise
    assert "upload to r2://b/c/last.ckpt failed" in caplog.text


def test_uploader_propagates_unexpected_upload_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Programming errors from the upload boundary are not hidden as R2 failures.

    :param monkeypatch: Makes the upload transport raise an unexpected error.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(r2_io, "upload_to_uri", _raise_programming_error)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="programming bug"):
        CheckpointUploader("r2://b/c").on_train_batch_end(_trainer(str(ckpt)), None, None, None, 0)


def test_uploader_retries_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unchanged checkpoint that failed once is re-uploaded on the next hook.

    :param monkeypatch: Swaps the R2 transport (first raising, then recording).
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *_args, **_kwargs: None)

    def _boom(_local_path: Path, _uri: str) -> None:
        raise subprocess.CalledProcessError(1, ["rclone", "copyto"])

    monkeypatch.setattr(r2_io, "upload_to_uri", _boom)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    trainer = _trainer(str(ckpt))
    uploader.on_train_batch_end(trainer, None, None, None, 0)

    calls = _stub_r2(monkeypatch)
    uploader.on_train_batch_end(trainer, None, None, None, 1)
    assert calls == [(ckpt, "r2://b/c/last.ckpt")]


def test_uploader_stops_retrying_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stop probing R2 after one checkpoint revision exhausts its retry cap.

    :param monkeypatch: Swaps the R2 transport for an attempt-counting raising stub.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    """
    attempts = 0

    def _boom(_local_path: Path, _uri: str) -> None:
        nonlocal attempts
        attempts += 1
        raise subprocess.CalledProcessError(1, ["rclone", "copyto"])

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(r2_io, "upload_to_uri", _boom)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    trainer = _trainer(str(ckpt))
    for batch_idx in range(10):
        uploader.on_train_batch_end(trainer, None, None, None, batch_idx)
    assert attempts == 3


def test_uploader_resumes_after_new_checkpoint_follows_exhausted_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep an exhausted revision's retry budget from suppressing newer checkpoints.

    :param monkeypatch: Swaps the R2 transport (failing, then recovering).
    :param tmp_path: Holds the ``last.ckpt`` rewritten between the two versions.
    """
    failing = True
    recorded: list[str] = []

    def _upload(_local_path: Path, uri: str) -> None:
        if failing:
            raise subprocess.CalledProcessError(1, ["rclone", "copyto"])
        recorded.append(uri)

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(r2_io, "upload_to_uri", _upload)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"v1")
    os.utime(ckpt, (1000, 1000))
    uploader = CheckpointUploader("r2://b/c")
    trainer = _trainer(str(ckpt))
    for batch_idx in range(4):
        uploader.on_train_batch_end(trainer, None, None, None, batch_idx)
    assert recorded == []

    failing = False  # R2 recovers as a newer checkpoint version arrives
    ckpt.write_bytes(b"v2-longer")
    os.utime(ckpt, (2000, 2000))
    uploader.on_train_batch_end(trainer, None, None, None, 99)
    assert recorded == ["r2://b/c/last.ckpt"]


def test_checkpoint_prefix_uri_rejects_override_without_key() -> None:
    """A bucket-root override (no key segment) raises rather than yielding a bad prefix."""
    with pytest.raises(ValueError, match="r2://bucket/key"):
        _checkpoint_prefix_uri(
            _cfg(upload_checkpoints_uri="r2://mybucket"), "flow-simple-20260715T000000000Z"
        )


def test_uploader_reuploads_when_size_changes_at_same_mtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Use file size to distinguish checkpoint rewrites within one mtime tick.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds the ``last.ckpt`` rewritten in place between hooks.
    """
    calls = _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"v1")
    os.utime(ckpt, (1000, 1000))
    uploader = CheckpointUploader("r2://b/c")
    trainer = _trainer(str(ckpt))
    uploader.on_train_batch_end(trainer, None, None, None, 0)
    ckpt.write_bytes(b"much-longer-weights")
    os.utime(ckpt, (1000, 1000))
    uploader.on_train_batch_end(trainer, None, None, None, 1)
    assert len(calls) == 2


def test_uploader_reuploads_equal_size_rewrite_at_same_mtime_after_new_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A new ModelCheckpoint save distinguishes an otherwise identical file key.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds equal-size checkpoint rewrites with a fixed mtime.
    """
    calls = _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"v1")
    os.utime(ckpt, (1000, 1000))
    uploader = CheckpointUploader("r2://b/c")
    trainer = _trainer(str(ckpt))
    save_token = 1
    monkeypatch.setattr(callbacks_module, "_checkpoint_save_token", lambda _checkpoint: save_token)
    uploader.on_train_batch_end(trainer, None, None, None, 0)

    ckpt.write_bytes(b"v2")
    os.utime(ckpt, (1000, 1000))
    save_token = 2
    uploader.on_train_batch_end(trainer, None, None, None, 1)

    assert len(calls) == 2


def test_checkpoint_save_token_reads_model_checkpoint_compatibility_field() -> None:
    """The Lightning compatibility adapter reads the current completed-save token."""
    assert callbacks_module._checkpoint_save_token(ModelCheckpoint()) == 0


def test_uploader_swallows_missing_checkpoint_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``last_model_path`` at a vanished/rotated file is swallowed, no upload, no raise.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    """
    calls = _stub_r2(monkeypatch)
    uploader = CheckpointUploader("r2://b/c")
    uploader.on_train_batch_end(_trainer("/does/not/exist.ckpt"), None, None, None, 0)
    assert calls == []


def test_uploader_swallows_unreachable_r2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """R2 being unreachable (env-load raises) is swallowed, no upload attempted.

    :param monkeypatch: Makes ``ensure_r2_env_loaded`` raise, simulating absent R2 creds.
    :param tmp_path: Holds the fake checkpoint file so the env-load path is reached.
    """

    def _unavailable(*_a: object, **_k: object) -> None:
        raise RuntimeError("no creds")

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _unavailable)
    uploaded: list[str] = []
    monkeypatch.setattr(r2_io, "upload_to_uri", lambda local_path, uri: uploaded.append(uri))
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    uploader.on_train_batch_end(_trainer(str(ckpt)), None, None, None, 0)
    assert uploaded == []


def test_uploader_flushes_final_checkpoint_on_train_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``on_train_end`` mirrors the final checkpoint when no later batch hook fires.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    """
    calls = _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    uploader.on_train_end(_trainer(str(ckpt)), cast(LightningModule, object()))
    assert calls == [(ckpt, "r2://b/c/last.ckpt")]


@pytest.mark.parametrize(
    ("hook_name", "hook_args"),
    [
        ("on_validation_end", (None,)),
        ("on_train_epoch_end", (None,)),
        ("on_exception", (None, RuntimeError("cuda oom"))),
    ],
)
def test_uploader_mirrors_checkpoint_from_lifecycle_hook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hook_name: str,
    hook_args: tuple[object, ...],
) -> None:
    """Lifecycle hooks mirror the completed ``last.ckpt`` revision.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds the crash-time checkpoint file the uploader reads.
    :param hook_name: Callback hook under test.
    :param hook_args: Additional Lightning hook payload.
    """
    calls = _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    getattr(uploader, hook_name)(_trainer(str(ckpt)), *hook_args)
    assert calls == [(ckpt, "r2://b/c/last.ckpt")]


def test_uploader_warns_once_under_ddp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Under DDP (``world_size > 1``) a one-time synchronous-stall warning fires.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    :param caplog: Captures the one-time DDP warning.
    """
    _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    trainer = _trainer(str(ckpt), world_size=2)
    with caplog.at_level(logging.WARNING):
        uploader.on_train_batch_end(trainer, None, None, None, 0)
        ckpt.write_bytes(b"weights-2")
        os.utime(ckpt, (2000, 2000))
        uploader.on_train_batch_end(trainer, None, None, None, 1)
    assert caplog.text.count("stalls other DDP ranks") == 1


def test_trainer_preserves_uploader_after_model_checkpoint() -> None:
    """Keep the uploader after ModelCheckpoint so it mirrors the current save."""
    model_checkpoint = ModelCheckpoint()
    uploader = CheckpointUploader("r2://b/c", model_checkpoint)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        callbacks=[model_checkpoint, uploader],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    callbacks = cast(_FakeTrainer, trainer).callbacks
    assert callbacks.index(model_checkpoint) < callbacks.index(uploader)


def test_uploader_setup_rejects_replaced_model_checkpoint() -> None:
    """Final Lightning callback topology cannot replace the configured writer."""
    configured = ModelCheckpoint()
    replacement = ModelCheckpoint()
    uploader = CheckpointUploader("r2://b/c", configured)
    trainer = _trainer()
    cast(_FakeTrainer, trainer).callbacks = [uploader, replacement]
    with pytest.raises(ValueError, match="replaced or reordered"):
        uploader.setup(trainer, cast(LightningModule, object()), "fit")


def test_uploader_warns_on_train_end_when_no_checkpoint_seen(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Upload enabled but no checkpoint ever written warns at train-end (save_last likely off).

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param caplog: Captures the train-end diagnostic warning.
    """
    _stub_r2(monkeypatch)
    uploader = CheckpointUploader("r2://b/c")
    with caplog.at_level(logging.WARNING):
        uploader.on_train_end(_trainer(""), cast(LightningModule, object()))
    assert "save_last" in caplog.text


def test_checkpoint_prefix_uri_strips_the_basename() -> None:
    """The prefix is the derived checkpoint URI with its ``model.ckpt`` basename removed."""
    prefix = _checkpoint_prefix_uri(_cfg(), "flow-simple-20260715T000000000Z")
    assert prefix == (
        "r2://intermediate-data/checkpoints/flow-simple/flow-simple-20260715T000000000Z"
    )


def test_make_launch_namespace_distinguishes_same_run_id_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each launch gets an exact isolated namespace even when its run ID collides.

    :param monkeypatch: Supplies deterministic UUIDs at the production use site.
    """
    launch_uuids = iter((UUID(int=1), UUID(int=2)))
    monkeypatch.setattr("synth_setter.cli.train.uuid4", lambda: next(launch_uuids))
    run_id = "flow-simple-20260715T000000000Z"
    assert _make_launch_namespace(run_id) == f"{run_id}-{'0' * 31}1"
    assert _make_launch_namespace(run_id) == f"{run_id}-{'0' * 31}2"


def test_checkpoint_prefix_uri_honors_override() -> None:
    """A verbatim ``upload_checkpoints_uri`` override still yields its parent prefix."""
    cfg = _cfg(upload_checkpoints_uri="r2://models/run/model.ckpt")
    assert _checkpoint_prefix_uri(cfg, "run-20260715T000000000Z") == (
        "r2://models/run/run-20260715T000000000Z"
    )


def test_checkpoint_prefix_uri_rejects_override_with_trailing_slash() -> None:
    """A bucket-root override with a trailing slash is not a checkpoint object URI."""
    with pytest.raises(ValueError, match="r2://bucket/key"):
        _checkpoint_prefix_uri(
            _cfg(upload_checkpoints_uri="r2://mybucket/"),
            "flow-simple-20260715T000000000Z",
        )


def test_configure_durability_attaches_callback_when_enabled() -> None:
    """The flag enabled (with a ModelCheckpoint present) appends one ``CheckpointUploader``."""
    callbacks: list[Callback] = [ModelCheckpoint()]
    _configure_checkpoint_durability(_cfg(during_training=True), callbacks, "run-id")
    assert sum(isinstance(cb, CheckpointUploader) for cb in callbacks) == 1


def test_configure_durability_preflights_r2_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Durability validates R2 before training can start.

    :param monkeypatch: Records the R2 environment/authentication probe.
    """
    preflights = 0

    def _record_preflight() -> None:
        nonlocal preflights
        preflights += 1

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _record_preflight)
    _configure_checkpoint_durability(_cfg(during_training=True), [ModelCheckpoint()], "run-id")
    assert preflights == 1


def test_configure_durability_propagates_failed_r2_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable durability destination aborts before callback mutation.

    :param monkeypatch: Makes the R2 environment/authentication probe fail.
    """
    model_checkpoint = ModelCheckpoint(save_last=False, save_on_exception=False)
    callbacks: list[Callback] = [model_checkpoint]

    def _raise_unavailable() -> None:
        raise RuntimeError("R2 authentication failed")

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _raise_unavailable)
    with pytest.raises(RuntimeError, match="R2 authentication failed"):
        _configure_checkpoint_durability(_cfg(during_training=True), callbacks, "run-id")
    assert callbacks == [model_checkpoint]
    assert model_checkpoint.save_last is False
    assert model_checkpoint.save_on_exception is False


def test_configure_durability_noop_when_disabled() -> None:
    """The flag disabled leaves the callback list untouched."""
    callbacks: list[Callback] = []
    _configure_checkpoint_durability(_cfg(during_training=False), callbacks, "run-id")
    assert callbacks == []


def test_configure_durability_noop_when_flag_absent() -> None:
    """A config that never set the flag attaches nothing."""
    callbacks: list[Callback] = []
    _configure_checkpoint_durability(_cfg(), callbacks, "run-id")
    assert callbacks == []


def test_configure_durability_appends_after_existing_model_checkpoint() -> None:
    """The uploader is appended last, preserving a pre-existing ModelCheckpoint's position."""
    existing = ModelCheckpoint()
    callbacks: list[Callback] = [existing]
    _configure_checkpoint_durability(_cfg(during_training=True), callbacks, "run-id")
    assert callbacks[0] is existing
    assert isinstance(callbacks[-1], CheckpointUploader)


def test_configure_durability_rejects_missing_model_checkpoint() -> None:
    """Durability fails before training when no checkpoint writer is configured."""
    callbacks: list[Callback] = []
    with pytest.raises(ValueError, match="exactly one ModelCheckpoint; found 0"):
        _configure_checkpoint_durability(_cfg(during_training=True), callbacks, "run-id")
    assert callbacks == []


def test_configure_durability_enables_save_on_exception() -> None:
    """Appending the uploader flips ``save_on_exception`` on the ModelCheckpoint (default off)."""
    model_checkpoint = ModelCheckpoint()
    assert model_checkpoint.save_on_exception is False
    _configure_checkpoint_durability(_cfg(during_training=True), [model_checkpoint], "run-id")
    assert model_checkpoint.save_on_exception is True


def test_configure_durability_enables_save_last() -> None:
    """Appending the uploader ensures ModelCheckpoint exposes ``last_model_path``."""
    model_checkpoint = ModelCheckpoint(save_last=False)
    _configure_checkpoint_durability(_cfg(during_training=True), [model_checkpoint], "run-id")
    assert model_checkpoint.save_last is True


def test_configure_durability_rejects_multiple_model_checkpoints() -> None:
    """Durability fails before training when checkpoint ownership is ambiguous."""
    callbacks: list[Callback] = [ModelCheckpoint(), ModelCheckpoint()]
    with pytest.raises(ValueError, match="exactly one ModelCheckpoint; found 2"):
        _configure_checkpoint_durability(_cfg(during_training=True), callbacks, "run-id")
    assert len(callbacks) == 2
    assert not any(isinstance(callback, CheckpointUploader) for callback in callbacks)
