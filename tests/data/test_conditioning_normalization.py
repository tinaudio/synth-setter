"""Tests for per-channel z-scoring of the generic conditioning column."""

from pathlib import Path

import numpy as np
import pytest
import torch

from synth_setter.conditioning import EmbeddingConditioningSpec
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.data.vst_datamodule import RawBatch, prepare_batch
from synth_setter.param_spec_name import ParamSpecName
from tests.data.test_embedding_conditioning import _write_embedding_shard

_CHANNELS = 4
_FRAMES = 6


def _skewed_conditioning(rows: int, seed: int = 0) -> np.ndarray:
    """Draw per-channel offset/scaled conditioning values.

    :param rows: Number of rows.
    :param seed: RNG seed.
    :returns: Array of shape ``(rows, _CHANNELS, _FRAMES)`` with distinct channel stats.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((rows, _CHANNELS, _FRAMES)).astype(np.float32)
    offsets = np.arange(_CHANNELS, dtype=np.float32).reshape(1, _CHANNELS, 1)
    scales = (1.0 + np.arange(_CHANNELS, dtype=np.float32)).reshape(1, _CHANNELS, 1)
    return base * scales + offsets


def _channel_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std over rows and frames.

    :param values: Array of shape ``(rows, channels, frames)``.
    :returns: ``(mean, std)`` arrays of shape ``(channels, 1)``.
    """
    return (
        values.mean(axis=(0, 2), keepdims=True)[0],
        values.std(axis=(0, 2), keepdims=True)[0],
    )


def _prepare(
    values: np.ndarray,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> dict[str, torch.Tensor | None]:
    """Run ``prepare_batch`` over a conditioning-only raw batch.

    :param values: Conditioning values of shape ``(rows, channels, frames)``.
    :param mean: Per-channel conditioning mean, or ``None``.
    :param std: Per-channel conditioning std, or ``None``.
    :returns: Prepared model batch.
    """
    raw: RawBatch = {
        "param_array": np.zeros((values.shape[0], 2), dtype=np.float32),
        "conditioning": values,
    }
    return prepare_batch(
        raw,
        mean=None,
        std=None,
        conditioning_mean=mean,
        conditioning_std=std,
        rescale_params=True,
        ot=False,
        generator=torch.Generator().manual_seed(0),
    )


def test_prepare_batch_conditioning_stats_normalize_to_zero_mean_unit_std() -> None:
    """Supplied per-channel stats z-score the conditioning column."""
    values = _skewed_conditioning(rows=64)
    mean, std = _channel_stats(values)

    normalized = _prepare(values, mean, std)["conditioning"]

    assert normalized is not None
    per_channel_mean = normalized.mean(dim=(0, 2))
    per_channel_std = normalized.std(dim=(0, 2))
    torch.testing.assert_close(
        per_channel_mean, torch.zeros(_CHANNELS), atol=1e-4, rtol=0
    )
    torch.testing.assert_close(
        per_channel_std, torch.ones(_CHANNELS), atol=1e-2, rtol=0
    )


def test_prepare_batch_conditioning_without_stats_passes_raw_values() -> None:
    """The no-stats path forwards the stored conditioning unchanged."""
    values = _skewed_conditioning(rows=4)

    conditioning = _prepare(values, None, None)["conditioning"]

    assert conditioning is not None
    torch.testing.assert_close(conditioning, torch.from_numpy(values))


def test_prepare_batch_conditioning_half_supplied_stats_raise() -> None:
    """Supplying exactly one of mean/std fails loudly instead of skipping."""
    values = _skewed_conditioning(rows=4)
    mean, std = _channel_stats(values)

    with pytest.raises(ValueError, match="together"):
        _prepare(values, mean, None)
    with pytest.raises(ValueError, match="together"):
        _prepare(values, None, std)


def test_prepare_batch_conditioning_nonpositive_std_raises() -> None:
    """A zero std channel fails loudly instead of dividing to inf."""
    values = _skewed_conditioning(rows=4)
    mean, std = _channel_stats(values)
    std[0] = 0.0

    with pytest.raises(ValueError, match="conditioning std"):
        _prepare(values, mean, std)


def test_prepare_batch_conditioning_nonfinite_mean_raises() -> None:
    """A non-finite mean fails loudly before corrupting the batch."""
    values = _skewed_conditioning(rows=4)
    mean, std = _channel_stats(values)
    mean[0] = np.nan

    with pytest.raises(ValueError, match="conditioning mean"):
        _prepare(values, mean, std)


def _stats_module(root: Path, column: str) -> LanceVSTDataModule:
    """Build a datamodule with conditioning normalization enabled.

    :param root: Dataset root containing train/val shards.
    :param column: Embedding column under test.
    :returns: Unset-up datamodule.
    """
    return LanceVSTDataModule(
        dataset_root=root,
        batch_size=2,
        conditioning=EmbeddingConditioningSpec(
            column=column, input_shape=(_CHANNELS, _FRAMES)
        ),
        use_saved_mean_and_variance=True,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("surge_xt"),
    )


def test_lance_datamodule_embedding_conditioning_normalizes_with_train_stats(
    tmp_path: Path,
) -> None:
    """Setup computes per-channel train-split stats and the loader applies them.

    :param tmp_path: Per-test dataset root.
    """
    train_values = _skewed_conditioning(rows=8, seed=1)
    val_values = _skewed_conditioning(rows=4, seed=2)
    _write_embedding_shard(tmp_path / "train.lance", column="emb", values=train_values)
    _write_embedding_shard(tmp_path / "val.lance", column="emb", values=val_values)
    module = _stats_module(tmp_path, "emb")

    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    mean, std = _channel_stats(train_values)
    expected = torch.from_numpy((val_values[:2] - mean) / std).to(torch.float32)
    torch.testing.assert_close(batch["conditioning"], expected, atol=1e-5, rtol=1e-5)


def test_lance_datamodule_computed_conditioning_stats_persist_across_runs(
    tmp_path: Path,
) -> None:
    """First setup writes a stats sidecar that later runs reload verbatim.

    :param tmp_path: Per-test dataset root.
    """
    train_values = _skewed_conditioning(rows=8, seed=1)
    val_values = _skewed_conditioning(rows=4, seed=2)
    _write_embedding_shard(tmp_path / "train.lance", column="emb", values=train_values)
    _write_embedding_shard(tmp_path / "val.lance", column="emb", values=val_values)

    _stats_module(tmp_path, "emb").setup("validate")

    sidecar = tmp_path / "conditioning_stats_emb.npz"
    assert sidecar.exists()
    # Replace the train shard with differently distributed rows: the sidecar,
    # not the new shard, must drive normalization from now on.
    shifted = _skewed_conditioning(rows=8, seed=3) + 100.0
    _write_embedding_shard(tmp_path / "train.lance", column="emb", values=shifted)
    module = _stats_module(tmp_path, "emb")
    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    mean, std = _channel_stats(train_values)
    expected = torch.from_numpy((val_values[:2] - mean) / std).to(torch.float32)
    torch.testing.assert_close(batch["conditioning"], expected, atol=1e-5, rtol=1e-5)


def test_lance_datamodule_constant_conditioning_channel_normalizes_without_error(
    tmp_path: Path,
) -> None:
    """A channel constant across the sample floors its std instead of raising.

    :param tmp_path: Per-test dataset root.
    """
    train_values = _skewed_conditioning(rows=8, seed=1)
    train_values[:, 0, :] = 7.5
    val_values = _skewed_conditioning(rows=4, seed=2)
    val_values[:, 0, :] = 7.5
    _write_embedding_shard(tmp_path / "train.lance", column="emb", values=train_values)
    _write_embedding_shard(tmp_path / "val.lance", column="emb", values=val_values)
    module = _stats_module(tmp_path, "emb")

    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    conditioning = batch["conditioning"]
    assert torch.isfinite(conditioning).all()
    # Constant channel: x - mean == 0, so the floored divisor yields exact zeros.
    torch.testing.assert_close(conditioning[:, 0], torch.zeros_like(conditioning[:, 0]))


def test_lance_datamodule_saved_conditioning_stats_wrong_shape_raises(
    tmp_path: Path,
) -> None:
    """A sidecar whose arrays cannot broadcast per-channel fails at setup.

    :param tmp_path: Per-test dataset root.
    """
    train_values = _skewed_conditioning(rows=8, seed=1)
    _write_embedding_shard(tmp_path / "train.lance", column="emb", values=train_values)
    _write_embedding_shard(
        tmp_path / "val.lance", column="emb", values=_skewed_conditioning(rows=4, seed=2)
    )
    np.savez(
        tmp_path / "conditioning_stats_emb.npz",
        mean=np.zeros(_CHANNELS, dtype=np.float32),
        std=np.ones(_CHANNELS, dtype=np.float32),
    )
    module = _stats_module(tmp_path, "emb")

    with pytest.raises(ValueError, match="shape"):
        module.setup("validate")


def test_lance_datamodule_predict_only_setup_skips_train_shard(
    tmp_path: Path,
) -> None:
    """``setup("predict")`` must not require the train shard.

    :param tmp_path: Per-test dataset root holding only the predict shard.
    """
    _write_embedding_shard(
        tmp_path / "test.lance", column="emb", values=_skewed_conditioning(rows=4, seed=2)
    )
    module = _stats_module(tmp_path, "emb")

    module.setup("predict")
    try:
        batch = next(iter(module.predict_dataloader()))
    finally:
        module.teardown()

    assert batch["conditioning"] is not None


def test_lance_datamodule_saved_conditioning_stats_take_precedence(
    tmp_path: Path,
) -> None:
    """A saved stats file beside the train split overrides on-the-fly computation.

    :param tmp_path: Per-test dataset root.
    """
    train_values = _skewed_conditioning(rows=8, seed=1)
    val_values = _skewed_conditioning(rows=4, seed=2)
    _write_embedding_shard(tmp_path / "train.lance", column="emb", values=train_values)
    _write_embedding_shard(tmp_path / "val.lance", column="emb", values=val_values)
    saved_mean = np.full((_CHANNELS, 1), 1.0, dtype=np.float32)
    saved_std = np.full((_CHANNELS, 1), 2.0, dtype=np.float32)
    np.savez(
        tmp_path / "conditioning_stats_emb.npz", mean=saved_mean, std=saved_std
    )
    module = _stats_module(tmp_path, "emb")

    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    expected = torch.from_numpy((val_values[:2] - saved_mean) / saved_std).to(
        torch.float32
    )
    torch.testing.assert_close(batch["conditioning"], expected, atol=1e-5, rtol=1e-5)
