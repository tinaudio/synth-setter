"""Tests for CheckpointUploader and its CLI wiring.

The R2 transport is monkeypatched throughout — no ``rclone`` binary, no
network — so only the callback's rank-gating, change-detection, best-effort
error handling, and URI derivation run for real.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import cast

import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli.train import _append_checkpoint_uploader, _checkpoint_prefix_uri
from synth_setter.pipeline import r2_io
from synth_setter.utils.callbacks import _MAX_UPLOAD_ATTEMPTS, CheckpointUploader


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
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *a, **k: None)
    monkeypatch.setattr(
        r2_io, "upload_to_uri", lambda local_path, uri: calls.append((local_path, uri))
    )
    return calls


def test_uploader_uploads_new_checkpoint_to_prefixed_last_ckpt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A freshly-written ``last_model_path`` uploads to ``{prefix}/last.ckpt``.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    """
    calls = _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://intermediate-data/checkpoints/flow-simple")
    uploader.on_train_batch_end(_trainer(str(ckpt)), None, None, None, 0)
    assert calls == [(ckpt, "r2://intermediate-data/checkpoints/flow-simple/last.ckpt")]


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
    """``save_last`` overwrites ``last.ckpt`` in place; a newer mtime re-uploads it.

    Guards against keying change-detection on the (stable) path alone, which
    would upload the first save and never its later overwrites.

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
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *a, **k: None)

    def _boom(_p: Path, _uri: str) -> None:
        raise RuntimeError("r2 down")

    monkeypatch.setattr(r2_io, "upload_to_uri", _boom)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    with caplog.at_level(logging.WARNING):
        uploader.on_train_batch_end(_trainer(str(ckpt)), None, None, None, 0)  # must not raise
    assert "upload to r2://b/c/last.ckpt failed" in caplog.text


def test_uploader_retries_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unchanged checkpoint that failed once is re-uploaded on the next hook.

    :param monkeypatch: Swaps the R2 transport (first raising, then recording).
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *a, **k: None)

    def _boom(_p: Path, _uri: str) -> None:
        raise RuntimeError("r2 down")

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
    """A persistently failing upload of one unchanged file backs off after the retry cap.

    Without a bound, every subsequent batch re-hits R2 (a synchronous auth-ping
    + copy) for the rest of the run once R2 goes unreachable.

    :param monkeypatch: Swaps the R2 transport for an attempt-counting raising stub.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    """
    attempts = 0

    def _boom(_p: Path, _uri: str) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("r2 down")

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *a, **k: None)
    monkeypatch.setattr(r2_io, "upload_to_uri", _boom)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    trainer = _trainer(str(ckpt))
    for batch_idx in range(10):
        uploader.on_train_batch_end(trainer, None, None, None, batch_idx)
    assert attempts == _MAX_UPLOAD_ATTEMPTS


def test_uploader_resumes_after_new_checkpoint_follows_exhausted_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A newer checkpoint uploads even after retries were exhausted on the prior one.

    Guards the pending-key reset: without it, one bad R2 stretch would permanently
    stop mirroring every later checkpoint — the exact durability regression this
    feature prevents.

    :param monkeypatch: Swaps the R2 transport (failing, then recovering).
    :param tmp_path: Holds the ``last.ckpt`` rewritten between the two versions.
    """
    failing = True
    recorded: list[str] = []

    def _upload(_local_path: Path, uri: str) -> None:
        if failing:
            raise RuntimeError("r2 down")
        recorded.append(uri)

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *a, **k: None)
    monkeypatch.setattr(r2_io, "upload_to_uri", _upload)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"v1")
    os.utime(ckpt, (1000, 1000))
    uploader = CheckpointUploader("r2://b/c")
    trainer = _trainer(str(ckpt))
    for batch_idx in range(_MAX_UPLOAD_ATTEMPTS + 2):  # exhaust retries on v1
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
        _checkpoint_prefix_uri(_cfg(upload_checkpoints_uri="r2://mybucket"))


def test_uploader_reuploads_when_size_changes_at_same_mtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A same-mtime overwrite of a different size still re-uploads.

    ``mtime`` has coarse (second-level) resolution on some filesystems, so two
    saves within one tick would collide on ``mtime`` alone — the size term of
    the change key breaks the tie.

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
    ckpt.write_bytes(b"much-longer-weights")  # different size
    os.utime(ckpt, (1000, 1000))  # same coarse mtime tick
    uploader.on_train_batch_end(trainer, None, None, None, 1)
    assert len(calls) == 2


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
    uploader.on_train_end(_trainer(str(ckpt)))
    assert calls == [(ckpt, "r2://b/c/last.ckpt")]


def test_uploader_uploads_on_validation_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A checkpoint written on the validation cadence is mirrored from ``on_validation_end``.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    """
    calls = _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    uploader.on_validation_end(_trainer(str(ckpt)))
    assert calls == [(ckpt, "r2://b/c/last.ckpt")]


def test_uploader_uploads_on_train_epoch_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A checkpoint written on the per-epoch cadence is mirrored from ``on_train_epoch_end``.

    :param monkeypatch: Swaps the R2 transport for a recorder.
    :param tmp_path: Holds the fake checkpoint file the uploader reads.
    """
    calls = _stub_r2(monkeypatch)
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    uploader = CheckpointUploader("r2://b/c")
    uploader.on_train_epoch_end(_trainer(str(ckpt)))
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


def test_uploader_is_reordered_after_modelcheckpoint() -> None:
    """As a ``Checkpoint`` subclass, Lightning dispatches the uploader after ``ModelCheckpoint``.

    Pins the fix for the one-write-stale bug: a plain ``Callback`` would be grouped
    before the checkpoint callbacks and mirror the previous save.
    """
    from lightning.pytorch.callbacks import Checkpoint, ModelCheckpoint
    from lightning.pytorch.trainer.connectors.callback_connector import _CallbackConnector

    uploader = CheckpointUploader("r2://b/c")
    assert isinstance(uploader, Checkpoint)
    ordered = _CallbackConnector._reorder_callbacks([ModelCheckpoint(), uploader])
    model_checkpoint = next(c for c in ordered if isinstance(c, ModelCheckpoint))
    assert ordered.index(uploader) > ordered.index(model_checkpoint)


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
        uploader.on_train_end(_trainer(""))
    assert "save_last" in caplog.text


def test_checkpoint_prefix_uri_strips_the_basename() -> None:
    """The prefix is the derived checkpoint URI with its ``model.ckpt`` basename removed."""
    assert _checkpoint_prefix_uri(_cfg()) == "r2://intermediate-data/checkpoints/flow-simple"


def test_checkpoint_prefix_uri_honors_override() -> None:
    """A verbatim ``upload_checkpoints_uri`` override still yields its parent prefix."""
    cfg = _cfg(upload_checkpoints_uri="r2://models/run/model.ckpt")
    assert _checkpoint_prefix_uri(cfg) == "r2://models/run"


def test_append_uploader_attaches_callback_when_enabled() -> None:
    """The flag enabled appends exactly one ``CheckpointUploader``."""
    callbacks: list[Callback] = []
    _append_checkpoint_uploader(_cfg(during_training=True), callbacks)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], CheckpointUploader)


def test_append_uploader_noop_when_disabled() -> None:
    """The flag disabled leaves the callback list untouched."""
    callbacks: list[Callback] = []
    _append_checkpoint_uploader(_cfg(during_training=False), callbacks)
    assert callbacks == []


def test_append_uploader_noop_when_flag_absent() -> None:
    """A config that never set the flag attaches nothing."""
    callbacks: list[Callback] = []
    _append_checkpoint_uploader(_cfg(), callbacks)
    assert callbacks == []


def test_append_uploader_appends_after_existing_model_checkpoint() -> None:
    """The uploader is appended last, preserving a pre-existing ModelCheckpoint's position."""
    existing = ModelCheckpoint()
    callbacks: list[Callback] = [existing]
    _append_checkpoint_uploader(_cfg(during_training=True), callbacks)
    assert callbacks[0] is existing
    assert isinstance(callbacks[-1], CheckpointUploader)
