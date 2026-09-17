"""Behavior tests for additive cached-conditioning statistics."""

import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pytest
import torch

from synth_setter.conditioning import EmbeddingConditioningSpec
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.data.vst_datamodule import (
    RawBatch,
    load_conditioning_statistics,
    prepare_batch,
)
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import conditioning_stats_filename
from synth_setter.pipeline.data.lance_shard import tensor_array, write_lance_dataset
from synth_setter.pipeline.data.stats import (
    _conditioning_array_to_numpy,
    get_conditioning_stats_lance,
    main,
    stream_conditioning_stats_lance,
)


def _write_conditioning_dataset(
    path: Path,
    column: str,
    values: np.ndarray,
    *,
    fixed_size_list: bool = False,
) -> None:
    """Write one real fixed-shape Lance conditioning column.

    :param path: Destination Lance dataset.
    :param column: Conditioning column name.
    :param values: Values shaped ``(rows, *input_shape)``.
    :param fixed_size_list: Store vector rows without the tensor extension.
    :raises ValueError: Fixed-size-list storage is requested for sequence rows.
    """
    shape = values.shape[1:]
    if fixed_size_list:
        if len(shape) != 1:
            raise ValueError("fixed-size-list fixture requires vector rows")
        primitive = pa.array(values.reshape(-1), type=pa.from_numpy_dtype(values.dtype))
        array = pa.FixedSizeListArray.from_arrays(primitive, shape[0])
        field = pa.field(
            column, pa.list_(pa.from_numpy_dtype(values.dtype), shape[0]), nullable=False
        )
    else:
        array = tensor_array(values, values.dtype, shape)
        field = pa.field(
            column,
            pa.fixed_shape_tensor(pa.from_numpy_dtype(values.dtype), shape),
            nullable=False,
        )
    params = np.linspace(0.0, 1.0, len(values) * 2, dtype=np.float32).reshape(len(values), 2)
    param_array = tensor_array(params, params.dtype, (2,))
    schema = pa.schema(
        [
            field,
            pa.field(
                "param_array",
                pa.fixed_shape_tensor(pa.float32(), (2,)),
                nullable=False,
            ),
        ]
    )
    batch = pa.RecordBatch.from_arrays([array, param_array], schema=schema)
    write_lance_dataset(path, batch.schema, [batch])


def test_stream_conditioning_stats_vector_per_channel_matches_population_moments(
    tmp_path: Path,
) -> None:
    """Vector statistics retain one affine per embedding channel.

    :param tmp_path: Directory receiving the real Lance dataset.
    """
    dataset = tmp_path / "train.lance"
    values = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0], [7.0, 22.0]], dtype=np.float32)
    _write_conditioning_dataset(dataset, "embedding", values)

    mean, std = stream_conditioning_stats_lance(
        dataset,
        column="embedding",
        input_shape=(2,),
        normalization="per_channel",
    )

    np.testing.assert_allclose(mean, np.array([4.0, 16.0], dtype=np.float32))
    np.testing.assert_allclose(std, np.array([np.sqrt(5.0), np.sqrt(20.0)], dtype=np.float32))


def test_stream_conditioning_stats_fixed_size_list_vector_matches_population_moments(
    tmp_path: Path,
) -> None:
    """Production fixed-size-list vectors use the same per-channel affine.

    :param tmp_path: Directory receiving the real Lance dataset.
    """
    dataset = tmp_path / "train.lance"
    values = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]], dtype=np.float32)
    _write_conditioning_dataset(dataset, "embedding", values, fixed_size_list=True)

    mean, std = stream_conditioning_stats_lance(
        dataset,
        column="embedding",
        input_shape=(2,),
        normalization="per_channel",
    )

    np.testing.assert_allclose(mean, np.array([3.0, 14.0], dtype=np.float32))
    np.testing.assert_allclose(
        std,
        np.array([np.sqrt(8.0 / 3.0), np.sqrt(32.0 / 3.0)], dtype=np.float32),
    )


def test_conditioning_array_to_numpy_fixed_size_list_slice_respects_offset() -> None:
    """Fixed-size-list decoding reads only the logical Arrow slice."""
    values = np.arange(10, dtype=np.float32).reshape(5, 2)
    primitive = pa.array(values.reshape(-1), type=pa.float32())
    array = pa.FixedSizeListArray.from_arrays(primitive, 2).slice(1, 2)

    decoded = _conditioning_array_to_numpy(array, (2,))

    np.testing.assert_array_equal(decoded, values[1:3])


def test_stream_conditioning_stats_single_row_masks_every_channel(tmp_path: Path) -> None:
    """One observation has per-channel zero variance rather than scalar variance.

    :param tmp_path: Directory receiving the real Lance dataset.
    """
    dataset = tmp_path / "train.lance"
    values = np.array([[1.0, 10.0]], dtype=np.float32)
    _write_conditioning_dataset(dataset, "embedding", values)

    mean, std = stream_conditioning_stats_lance(
        dataset,
        column="embedding",
        input_shape=(2,),
        normalization="per_channel",
    )

    np.testing.assert_array_equal(mean, values[0])
    np.testing.assert_array_equal(std, np.ones(2, dtype=np.float32))


def test_stream_conditioning_stats_sequence_aggregates_frames_per_channel(
    tmp_path: Path,
) -> None:
    """Sequence statistics aggregate each channel across rows and time.

    :param tmp_path: Directory receiving the real Lance dataset.
    """
    dataset = tmp_path / "train.lance"
    values = np.array(
        [
            [[1.0, 3.0], [10.0, 14.0]],
            [[5.0, 7.0], [18.0, 22.0]],
        ],
        dtype=np.float32,
    )
    _write_conditioning_dataset(dataset, "embedding", values)

    mean, std = stream_conditioning_stats_lance(
        dataset,
        column="embedding",
        input_shape=(2, 2),
        normalization="per_channel",
    )

    np.testing.assert_allclose(mean, np.array([4.0, 16.0], dtype=np.float32))
    np.testing.assert_allclose(std, np.array([np.sqrt(5.0), np.sqrt(20.0)], dtype=np.float32))


def test_stream_conditioning_stats_multiple_batches_matches_population_moments(
    tmp_path: Path,
) -> None:
    """Uneven scanner batches merge into exact dataset-level moments.

    :param tmp_path: Directory receiving the real Lance dataset.
    """
    dataset = tmp_path / "train.lance"
    values = np.array(
        [[1.0, 10.0], [3.0, 14.0], [5.0, 18.0], [7.0, 22.0], [9.0, 26.0]],
        dtype=np.float32,
    )
    _write_conditioning_dataset(dataset, "embedding", values)

    mean, std = stream_conditioning_stats_lance(
        dataset,
        column="embedding",
        input_shape=(2,),
        normalization="per_channel",
        batch_size=2,
    )

    np.testing.assert_allclose(mean, np.array([5.0, 18.0], dtype=np.float32))
    np.testing.assert_allclose(std, np.array([np.sqrt(8.0), np.sqrt(32.0)], dtype=np.float32))


def test_stream_conditioning_stats_global_uses_one_shared_affine(tmp_path: Path) -> None:
    """Global statistics reduce every row, channel, and frame together.

    :param tmp_path: Directory receiving the real Lance dataset.
    """
    dataset = tmp_path / "train.lance"
    values = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    _write_conditioning_dataset(dataset, "embedding", values)

    mean, std = stream_conditioning_stats_lance(
        dataset,
        column="embedding",
        input_shape=(2, 2),
        normalization="global",
    )

    np.testing.assert_allclose(mean, np.array([4.5], dtype=np.float32))
    np.testing.assert_allclose(std, np.array([np.sqrt(5.25)], dtype=np.float32))


def test_conditioning_stats_validation_uses_training_affine(tmp_path: Path) -> None:
    """Validation values are transformed only with training-split statistics.

    :param tmp_path: Dataset root shared by the producer and datamodule consumer.
    """
    values = np.array([[1.0, 9.0], [3.0, 9.0], [5.0, 9.0], [7.0, 9.0]], dtype=np.float32)
    validation_values = np.array(
        [[11.0, 19.0], [13.0, 19.0], [15.0, 19.0], [17.0, 19.0]],
        dtype=np.float32,
    )
    _write_conditioning_dataset(tmp_path / "train.lance", "embedding", values)
    _write_conditioning_dataset(tmp_path / "val.lance", "embedding", validation_values)
    get_conditioning_stats_lance(
        tmp_path / "train.lance",
        column="embedding",
        input_shape=(2,),
        normalization="per_channel",
    )
    module = LanceVSTDataModule(
        dataset_root=tmp_path,
        batch_size=2,
        conditioning=EmbeddingConditioningSpec(
            column="embedding", input_shape=(2,), normalization="per_channel"
        ),
        use_saved_mean_and_variance=False,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("surge_xt"),
    )

    module.setup("validate")
    try:
        conditioning = next(iter(module.val_dataloader()))["conditioning"]
    finally:
        module.teardown()

    assert conditioning is not None
    torch.testing.assert_close(
        conditioning,
        torch.tensor(
            [[3.130495, 10.0], [4.024922, 10.0]],
            dtype=torch.float32,
        ),
    )


def test_conditioning_stats_near_dead_channel_uses_unit_standard_deviation(
    tmp_path: Path,
) -> None:
    """Sub-threshold channel variation is not amplified during standardization.

    :param tmp_path: Directory receiving the real Lance dataset.
    """
    dataset = tmp_path / "train.lance"
    base = np.float32(9.0)
    perturbed = np.nextafter(base, np.float32(np.inf))
    values = np.array([[1.0, base], [3.0, perturbed], [5.0, base]], dtype=np.float32)
    raw_std = values[:, 1].astype(np.float64).std()
    assert perturbed > base
    assert 0 < raw_std < 1e-6
    _write_conditioning_dataset(dataset, "embedding", values)

    _, std = stream_conditioning_stats_lance(
        dataset,
        column="embedding",
        input_shape=(2,),
        normalization="per_channel",
    )

    np.testing.assert_array_equal(std, np.array([np.sqrt(8.0 / 3.0), 1.0], dtype=np.float32))


def test_prepare_batch_standardizes_sequence_per_channel() -> None:
    """Per-channel arrays broadcast over sequence time in ``prepare_batch``."""
    raw: RawBatch = {
        "param_array": np.zeros((2, 2), dtype=np.float32),
        "conditioning": np.array(
            [
                [[1.0, 2.0], [10.0, 20.0]],
                [[3.0, 4.0], [30.0, 40.0]],
            ],
            dtype=np.float32,
        ),
    }

    batch = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=False,
        ot=False,
        generator=torch.Generator().manual_seed(0),
        conditioning_mean=np.array([2.0, 20.0], dtype=np.float32),
        conditioning_std=np.array([1.0, 10.0], dtype=np.float32),
    )

    torch.testing.assert_close(
        batch["conditioning"],
        torch.tensor(
            [
                [[-1.0, 0.0], [-1.0, 0.0]],
                [[1.0, 2.0], [1.0, 2.0]],
            ]
        ),
    )


def test_prepare_batch_global_standardization_uses_shared_affine() -> None:
    """Global statistics apply one scalar affine to every channel and frame."""
    raw: RawBatch = {
        "param_array": np.zeros((1, 2), dtype=np.float32),
        "conditioning": np.array([[[0.0, 2.0], [4.0, 6.0]]], dtype=np.float32),
    }

    batch = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=False,
        ot=False,
        generator=torch.Generator().manual_seed(0),
        conditioning_mean=np.array([2.0], dtype=np.float32),
        conditioning_std=np.array([2.0], dtype=np.float32),
    )

    torch.testing.assert_close(batch["conditioning"], torch.tensor([[[-1.0, 0.0], [1.0, 2.0]]]))


def test_prepare_batch_incomplete_conditioning_affine_raises() -> None:
    """A missing affine operand cannot silently disable requested standardization."""
    raw: RawBatch = {
        "param_array": np.zeros((1, 2), dtype=np.float32),
        "conditioning": np.ones((1, 2), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="conditioning_mean and conditioning_std"):
        prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=False,
            ot=False,
            generator=torch.Generator().manual_seed(0),
            conditioning_mean=np.zeros(2, dtype=np.float32),
            conditioning_std=None,
        )


def test_prepare_batch_conditioning_normalization_overflow_raises() -> None:
    """Finite affine operands may not produce non-finite conditioning output."""
    raw: RawBatch = {
        "param_array": np.zeros((2, 2), dtype=np.float32),
        "conditioning": np.full((2, 2), np.finfo(np.float32).max, dtype=np.float32),
    }

    with np.errstate(over="ignore"):
        mean = np.full(2, -np.finfo(np.float32).max, dtype=np.float32)
    with pytest.raises(ValueError, match="conditioning normalization produced non-finite"):
        prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=False,
            ot=False,
            generator=torch.Generator().manual_seed(0),
            conditioning_mean=mean,
            conditioning_std=np.ones(2, dtype=np.float32),
        )


@pytest.mark.parametrize("normalization", ["per_channel", "global"])
def test_load_conditioning_statistics_normalized_profile_missing_artifact_raises(
    tmp_path: Path, normalization: str
) -> None:
    """A configured affine requires its column-specific statistics artifact.

    :param tmp_path: Dataset root without conditioning statistics.
    :param normalization: Normalization policy requiring an artifact.
    """
    spec = EmbeddingConditioningSpec.model_validate(
        {
            "column": "embedding",
            "input_shape": (2,),
            "normalization": normalization,
        }
    )

    with pytest.raises(FileNotFoundError, match="conditioning_stats.embedding.npz"):
        load_conditioning_statistics(tmp_path / "train.lance", spec)


def test_load_conditioning_statistics_none_missing_artifact_returns_none(
    tmp_path: Path,
) -> None:
    """A profile that disables normalization needs no statistics artifact.

    :param tmp_path: Dataset root without conditioning statistics.
    """
    spec = EmbeddingConditioningSpec(column="embedding", input_shape=(2,))

    assert load_conditioning_statistics(tmp_path / "train.lance", spec) is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"mean": np.zeros(2)}, "must contain mean and std"),
        ({"std": np.ones(2)}, "must contain mean and std"),
        (
            {"mean": np.zeros(2), "std": np.ones(2), "count": np.array([2])},
            "must contain mean and std",
        ),
        ({"mean": np.zeros(1), "std": np.ones(2)}, "have shapes"),
        ({"mean": np.zeros(2), "std": np.ones(1)}, "have shapes"),
        ({"mean": np.array([np.nan, 0.0]), "std": np.ones(2)}, "finite values"),
        ({"mean": np.zeros(2), "std": np.array([np.inf, 1.0])}, "finite values"),
        ({"mean": np.zeros(2), "std": np.array([0.0, 1.0])}, "must be positive"),
        ({"mean": np.zeros(2), "std": np.array([-1.0, 1.0])}, "must be positive"),
    ],
)
def test_load_conditioning_statistics_malformed_artifact_raises(
    tmp_path: Path, payload: dict[str, np.ndarray], message: str
) -> None:
    """Malformed affine artifacts fail at the loading boundary.

    :param tmp_path: Dataset root receiving the malformed artifact.
    :param payload: Arrays written to the archive.
    :param message: Expected contract violation.
    """
    np.savez(
        tmp_path / conditioning_stats_filename("embedding"), **cast(Any, payload)
    )
    spec = EmbeddingConditioningSpec(
        column="embedding", input_shape=(2,), normalization="per_channel"
    )

    with pytest.raises(ValueError, match=message):
        load_conditioning_statistics(tmp_path / "train.lance", spec)


def test_stream_conditioning_stats_float32_overflow_raises(tmp_path: Path) -> None:
    """Finite source values must produce finite float32 artifact arrays.

    :param tmp_path: Dataset root receiving the float64 Lance column.
    """
    dataset = tmp_path / "train.lance"
    _write_conditioning_dataset(
        dataset,
        "embedding",
        np.array([[-1e40], [1e40]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="finite float32"):
        get_conditioning_stats_lance(
            dataset,
            column="embedding",
            input_shape=(1,),
            normalization="per_channel",
        )

    assert not (tmp_path / conditioning_stats_filename("embedding")).exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--conditioning-column", "embedding"],
        ["--conditioning-shape", "2"],
        ["--conditioning-normalization", "per_channel"],
        ["--conditioning-column", "embedding", "--conditioning-shape", "2"],
        ["--conditioning-column", "embedding", "--conditioning-normalization", "per_channel"],
        ["--conditioning-shape", "2", "--conditioning-normalization", "per_channel"],
        ["--conditioning-batch-size", "4"],
    ],
)
def test_conditioning_stats_cli_incomplete_flags_raise(tmp_path: Path, flags: list[str]) -> None:
    """Any partial conditioning invocation fails before legacy dispatch.

    :param tmp_path: Input path that legacy dispatch must not consume.
    :param flags: Incomplete conditioning option group.
    """
    with pytest.raises(ValueError, match="must be provided together"):
        main([str(tmp_path), *flags])


def test_conditioning_stats_cli_writes_one_immutable_artifact_per_column(tmp_path: Path) -> None:
    """The real CLI gives independently materializable columns separate artifacts.

    :param tmp_path: Dataset root receiving the additive archive.
    """
    first = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)
    second = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    _write_conditioning_dataset(tmp_path / "first.lance", "first", first)
    _write_conditioning_dataset(tmp_path / "second.lance", "second", second)
    mel_stats = tmp_path / "stats.npz"
    np.savez(mel_stats, mean=np.array([0.0]), std=np.array([1.0]))
    original_mel_stats = mel_stats.read_bytes()

    main(
        [
            str(tmp_path / "first.lance"),
            "--conditioning-column",
            "first",
            "--conditioning-shape",
            "2",
            "--conditioning-normalization",
            "per_channel",
        ]
    )
    main(
        [
            str(tmp_path / "second.lance"),
            "--conditioning-column",
            "second",
            "--conditioning-shape",
            "2",
            "2",
            "--conditioning-normalization",
            "global",
        ]
    )

    with np.load(tmp_path / conditioning_stats_filename("first")) as stats:
        assert set(stats.files) == {"mean", "std"}
        np.testing.assert_allclose(stats["mean"], [2.0, 4.0])
    with np.load(tmp_path / conditioning_stats_filename("second")) as stats:
        assert set(stats.files) == {"mean", "std"}
        np.testing.assert_allclose(stats["mean"], [4.5])
    assert mel_stats.read_bytes() == original_mel_stats


def test_conditioning_stats_conflicting_local_artifact_raises(tmp_path: Path) -> None:
    """A column artifact cannot be replaced by statistics from another snapshot.

    :param tmp_path: Dataset root receiving the immutable artifact.
    """
    first = tmp_path / "first.lance"
    second = tmp_path / "second.lance"
    _write_conditioning_dataset(first, "embedding", np.array([[1.0], [3.0]], dtype=np.float32))
    _write_conditioning_dataset(second, "embedding", np.array([[10.0], [14.0]], dtype=np.float32))
    get_conditioning_stats_lance(
        first,
        column="embedding",
        input_shape=(1,),
        normalization="per_channel",
    )

    with pytest.raises(FileExistsError, match="conflict with existing"):
        get_conditioning_stats_lance(
            second,
            column="embedding",
            input_shape=(1,),
            normalization="per_channel",
        )


def test_conditioning_stats_r2_publish_race_accepts_identical_winner(
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent identical immutable upload is an idempotent success.

    :param fake_r2_remote: Local filesystem backing the real rclone R2 boundary.
    :param monkeypatch: Fixture simulating another publisher winning the race.
    """
    dataset = fake_r2_remote / "bucket" / "run" / "train.lance"
    _write_conditioning_dataset(
        dataset,
        "embedding",
        np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32),
    )
    original_upload = r2_io.upload_to_uri_immutable

    def upload_after_winner(local_path: Path, destination: str) -> None:
        original_upload(local_path, destination)
        raise subprocess.CalledProcessError(1, ["rclone", "copyto", "--immutable"])

    monkeypatch.setattr(r2_io, "upload_to_uri_immutable", upload_after_winner)

    output = get_conditioning_stats_lance(
        "r2://bucket/run/train.lance",
        column="embedding",
        input_shape=(2,),
        normalization="per_channel",
    )

    assert output == "r2://bucket/run/conditioning_stats.embedding.npz"


def test_conditioning_stats_r2_publish_race_rejects_different_winner(
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent conflicting immutable upload remains an error.

    :param fake_r2_remote: Local filesystem backing the real rclone R2 boundary.
    :param monkeypatch: Fixture simulating another publisher winning the race.
    """
    dataset = fake_r2_remote / "bucket" / "run" / "train.lance"
    _write_conditioning_dataset(
        dataset,
        "embedding",
        np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32),
    )

    def upload_after_winner(_: Path, destination: str) -> None:
        winner = fake_r2_remote / destination.removeprefix("r2://")
        np.savez(winner, mean=np.array([99.0]), std=np.array([1.0]))
        raise subprocess.CalledProcessError(1, ["rclone", "copyto", "--immutable"])

    monkeypatch.setattr(r2_io, "upload_to_uri_immutable", upload_after_winner)

    with pytest.raises(FileExistsError, match="conflict with existing"):
        get_conditioning_stats_lance(
            "r2://bucket/run/train.lance",
            column="embedding",
            input_shape=(2,),
            normalization="per_channel",
        )


def test_conditioning_stats_r2_uri_reads_and_uploads_artifact(
    fake_r2_remote: Path,
) -> None:
    """Canonical R2 input publishes selected-column statistics beside the dataset.

    :param fake_r2_remote: Local filesystem backing the real rclone R2 boundary.
    """
    dataset = fake_r2_remote / "bucket" / "run" / "train.lance"
    _write_conditioning_dataset(
        dataset,
        "embedding",
        np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32),
    )

    output = get_conditioning_stats_lance(
        "r2://bucket/run/train.lance",
        column="embedding",
        input_shape=(2,),
        normalization="per_channel",
    )

    assert output == "r2://bucket/run/conditioning_stats.embedding.npz"
    with np.load(
        fake_r2_remote / "bucket" / "run" / conditioning_stats_filename("embedding")
    ) as stats:
        np.testing.assert_allclose(stats["mean"], [2.0, 4.0])
