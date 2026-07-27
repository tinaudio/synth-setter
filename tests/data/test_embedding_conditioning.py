"""End-to-end data-path tests for fixed-shape embedding conditioning."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

from synth_setter.conditioning import ConditioningMode, EmbeddingConditioningSpec
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.data.vst_datamodule import RawBatch, prepare_batch
from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data.lance_shard import tensor_array, write_lance_dataset
from tests.helpers.lance_fixtures import (
    M2L_DIM_1,
    M2L_DIM_2,
    make_shard_columns,
    shard_record_batch,
)


def _fixed_size_list(values: np.ndarray) -> pa.FixedSizeListArray:
    """Encode one rank-two matrix as an Arrow fixed-size-list column.

    :param values: Matrix shaped ``(rows, width)``.
    :returns: Fixed-size-list array retaining the row width.
    """
    flat = pa.array(values.reshape(-1), type=pa.from_numpy_dtype(values.dtype))
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _write_embedding_shard(
    path: Path,
    *,
    column: str,
    values: np.ndarray | list[list[float]],
    variable_length: bool = False,
) -> None:
    """Write a real VST fixture shard with one extra embedding column.

    :param path: Destination Lance dataset.
    :param column: Added embedding column name.
    :param values: Per-row embedding values.
    :param variable_length: Whether to encode Arrow ``List`` rather than fixed-size values.
    """
    rows = len(values)
    batch = shard_record_batch(make_shard_columns(rows, seed=9))
    if variable_length:
        array = pa.array(values, type=pa.list_(pa.float32()))
        field = pa.field(column, array.type, nullable=False)
    else:
        fixed_values = np.asarray(values)
        if fixed_values.ndim == 2:
            array = _fixed_size_list(fixed_values)
            field = pa.field(column, array.type, nullable=False)
        else:
            shape = fixed_values.shape[1:]
            array = tensor_array(fixed_values, fixed_values.dtype, shape)
            field = pa.field(
                column,
                pa.fixed_shape_tensor(pa.from_numpy_dtype(fixed_values.dtype), shape),
                nullable=False,
            )
    extended = batch.append_column(field, array)
    write_lance_dataset(path, extended.schema, [extended])


def _embedding_module(
    root: Path,
    conditioning: EmbeddingConditioningSpec | ConditioningMode,
    *,
    fake: bool = False,
) -> LanceVSTDataModule:
    """Build a cheap validation-only embedding datamodule.

    :param root: Dataset root or unused fake-mode path.
    :param conditioning: Conditioning configuration under test.
    :param fake: Whether to synthesize data instead of opening Lance.
    :returns: Unset-up datamodule.
    """
    return LanceVSTDataModule(
        dataset_root=root,
        batch_size=2,
        conditioning=conditioning,
        fake=fake,
        use_saved_mean_and_variance=False,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("surge_xt"),
    )


def test_conditioning_spec_clap_column_routes_to_batch(tmp_path: Path) -> None:
    """A fixed-size CLAP column reaches the canonical model-batch key.

    :param tmp_path: Per-test dataset root.
    """
    values = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    _write_embedding_shard(tmp_path / "val.lance", column="clap", values=values)
    module = _embedding_module(
        tmp_path, EmbeddingConditioningSpec(column="clap", input_shape=(5,))
    )

    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    assert batch["mel_spec"] is None
    assert batch["m2l"] is None
    torch.testing.assert_close(batch["conditioning"], torch.from_numpy(values[:2]))


def test_conditioning_spec_m2l_nondefault_seq_len_pools(tmp_path: Path) -> None:
    """A fixed m2l sequence uses its configured non-42 sequence length.

    :param tmp_path: Per-test dataset root.
    """
    source = make_shard_columns(4, seed=9)["music2latent"]
    _write_embedding_shard(
        tmp_path / "val.lance", column="alternate_m2l", values=source
    )
    spec = EmbeddingConditioningSpec(
        column="alternate_m2l", input_shape=(M2L_DIM_1, M2L_DIM_2)
    )
    module = _embedding_module(tmp_path, spec)

    module.setup("validate")
    try:
        embedding = next(iter(module.val_dataloader()))["conditioning"]
    finally:
        module.teardown()

    assert embedding is not None
    sequence = embedding
    pool = EmbeddingPool(
        embed_dim=M2L_DIM_1,
        d_model=12,
        num_heads=3,
        max_seq_len=M2L_DIM_2,
    )
    assert pool(sequence).shape == (2, 12)


def test_conditioning_spec_missing_column_raises(tmp_path: Path) -> None:
    """A missing configured column fails during setup.

    :param tmp_path: Per-test dataset root.
    """
    values = np.ones((4, 5), dtype=np.float32)
    _write_embedding_shard(tmp_path / "val.lance", column="clap", values=values)
    module = _embedding_module(
        tmp_path, EmbeddingConditioningSpec(column="absent", input_shape=(5,))
    )

    with pytest.raises(KeyError, match=r"absent.*val\.lance"):
        module.setup("validate")


def test_conditioning_spec_wrong_input_shape_raises(tmp_path: Path) -> None:
    """A configured width mismatch fails during setup.

    :param tmp_path: Per-test dataset root.
    """
    values = np.ones((4, 5), dtype=np.float32)
    _write_embedding_shard(tmp_path / "val.lance", column="clap", values=values)
    module = _embedding_module(
        tmp_path, EmbeddingConditioningSpec(column="clap", input_shape=(6,))
    )

    with pytest.raises(ValueError, match=r"clap.*shape.*\(5,\).*expected.*\(6,\)"):
        module.setup("validate")


def test_conditioning_spec_variable_length_column_raises(tmp_path: Path) -> None:
    """Arrow List embeddings are rejected before a dataloader is built.

    :param tmp_path: Per-test dataset root.
    """
    values = [[1.0], [2.0, 3.0], [4.0], [5.0, 6.0]]
    _write_embedding_shard(
        tmp_path / "val.lance",
        column="ragged",
        values=values,
        variable_length=True,
    )
    module = _embedding_module(
        tmp_path, EmbeddingConditioningSpec(column="ragged", input_shape=(2,))
    )

    with pytest.raises(TypeError, match=r"ragged.*variable-length.*list"):
        module.setup("validate")


def test_conditioning_spec_nonfloating_column_raises(tmp_path: Path) -> None:
    """Integer embeddings fail the floating-point storage contract at setup.

    :param tmp_path: Per-test dataset root.
    """
    values = np.ones((4, 5), dtype=np.int32)
    _write_embedding_shard(tmp_path / "val.lance", column="tokens", values=values)
    module = _embedding_module(
        tmp_path, EmbeddingConditioningSpec(column="tokens", input_shape=(5,))
    )

    with pytest.raises(TypeError, match=r"tokens.*floating-point"):
        module.setup("validate")


def test_conditioning_spec_nonfinite_sample_raises(tmp_path: Path) -> None:
    """A poisoned sample fails before training starts.

    :param tmp_path: Per-test dataset root.
    """
    values = np.ones((4, 5), dtype=np.float32)
    values[0, 0] = np.nan
    _write_embedding_shard(tmp_path / "val.lance", column="clap", values=values)
    module = _embedding_module(
        tmp_path, EmbeddingConditioningSpec(column="clap", input_shape=(5,))
    )

    with pytest.raises(ValueError, match=r"clap.*non-finite"):
        module.setup("validate")


def test_conditioning_spec_fake_mode_derives_shape(tmp_path: Path) -> None:
    """Fake embedding batches use the configured fixed per-row shape.

    :param tmp_path: Empty path proving fake mode avoids storage.
    """
    module = _embedding_module(
        tmp_path,
        EmbeddingConditioningSpec(column="clap", input_shape=(17,)),
        fake=True,
    )

    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    assert batch["conditioning"] is not None
    assert batch["conditioning"].shape == (2, 17)


def test_legacy_m2l_string_still_routes(tmp_path: Path) -> None:
    """The m2l literal keeps its historical key while exposing generic conditioning.

    :param tmp_path: Empty path proving the compatibility path works in fake mode.
    """
    module = _embedding_module(tmp_path, "m2l", fake=True)

    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    assert batch["m2l"] is not None
    assert batch["conditioning"] is not None
    assert batch["m2l"].shape == (2, 128, 42)
    assert torch.equal(batch["conditioning"], batch["m2l"])


def test_prepare_batch_ot_keeps_generic_conditioning_aligned() -> None:
    """Hungarian matching applies the parameter permutation to generic embeddings."""
    row_ids = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    raw: RawBatch = {
        "param_array": np.repeat(row_ids[:, None], 5, axis=1),
        "conditioning": np.repeat(row_ids[:, None], 3, axis=1),
    }

    batch = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=False,
        ot=True,
        generator=torch.Generator().manual_seed(17),
    )

    conditioning = batch["conditioning"]
    params = batch["params"]
    assert conditioning is not None
    assert params is not None
    assert torch.equal(conditioning[:, 0], params[:, 0])
    assert not torch.equal(params[:, 0], torch.from_numpy(row_ids))
