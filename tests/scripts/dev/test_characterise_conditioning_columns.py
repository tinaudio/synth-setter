"""Behavior tests for conditioning-column statistics."""

from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path
from typing import cast

import lance
import numpy as np
import pyarrow as pa
import pytest

from scripts.dev.characterise_conditioning_columns import (
    StreamingStatistics,
    analyse_dataset,
    discover_conditioning_columns,
    matpac_band_views,
)


def test_discover_conditioning_columns_cached_profiles_returns_shapes(tmp_path: Path) -> None:
    """Cached profiles contribute columns while online profiles do not.

    :param tmp_path: Directory holding test conditioning profiles.
    """
    (tmp_path / "cached.yaml").write_text(
        "model:\n"
        "  conditioning:\n"
        "    column: sequence\n"
        "    input_shape: [2, 3]\n"
        "datamodule:\n"
        "  conditioning:\n"
        "    column: sequence\n"
        "    input_shape: [2, 3]\n"
    )
    (tmp_path / "online.yaml").write_text("model:\n  encoder_output_dim: 64\n")

    columns = discover_conditioning_columns(tmp_path)

    assert columns == {"sequence": (2, 3)}


def test_streaming_statistics_vector_reports_channel_and_row_metrics() -> None:
    """Vector rows produce channel, global, and whole-row norm metrics."""
    statistics = StreamingStatistics(shape=(2,))
    statistics.update(np.array([[3.0, 4.0]], dtype=np.float32))
    statistics.update(np.array([[0.0, 0.0]], dtype=np.float32))

    result = statistics.result()

    assert result.rows == 2
    assert result.channel_mean == pytest.approx((1.5, 1.75, 2.0))
    assert result.channel_std == pytest.approx((1.5, 1.75, 2.0))
    assert result.global_mean == pytest.approx(1.75)
    assert result.global_std == pytest.approx(1.7853571071)
    assert result.global_min == pytest.approx(0.0)
    assert result.global_max == pytest.approx(4.0)
    assert result.dead_channels == 0
    assert result.row_l2_mean == pytest.approx(2.5)
    assert result.row_l2_std == pytest.approx(2.5)
    assert result.frame_l2_cv_mean is None


def test_streaming_statistics_sequence_reports_within_row_frame_norm_cv() -> None:
    """Sequence rows report frame-magnitude variation within each row."""
    statistics = StreamingStatistics(shape=(2, 2))
    statistics.update(np.array([[[3.0, 0.0], [4.0, 0.0]]], dtype=np.float32))

    result = statistics.result()

    assert result.channel_mean == pytest.approx((1.5, 1.75, 2.0))
    assert result.channel_std == pytest.approx((1.5, 1.75, 2.0))
    assert result.row_l2_mean == pytest.approx(5.0)
    assert result.frame_l2_cv_mean == pytest.approx(1.0)


def test_streaming_statistics_dead_channel_counts_constant_channel() -> None:
    """A channel constant across rows and frames is classified as dead."""
    statistics = StreamingStatistics(shape=(2, 2))
    statistics.update(np.array([[[1.0, 1.0], [0.0, 2.0]]], dtype=np.float32))

    assert statistics.result().dead_channels == 1


def test_streaming_statistics_zero_magnitude_sequence_reports_undefined_frame_cv() -> None:
    """An all-zero sequence has no defined frame-norm coefficient of variation."""
    statistics = StreamingStatistics(shape=(2, 2))
    statistics.update(np.zeros((1, 2, 2), dtype=np.float32))

    assert math.isnan(cast(float, statistics.result().frame_l2_cv_mean))


def test_matpac_band_views_splits_channel_axis_without_changing_frames() -> None:
    """MATPAC band views split only the channel dimension."""
    values = np.arange(24, dtype=np.float32).reshape(1, 8, 3)

    bands = matpac_band_views(values, band_width=2)

    assert len(bands) == 4
    np.testing.assert_array_equal(bands[2], values[:, 4:6, :])


def test_matpac_band_views_with_zero_band_width_raises_value_error() -> None:
    """A zero-width band is rejected as an invalid shape contract."""
    values = np.zeros((1, 8, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="positive"):
        matpac_band_views(values, band_width=0)


def test_analyse_dataset_tensor_child_null_raises(tmp_path: Path) -> None:
    """A tensor child null cannot silently enter statistics as zero.

    :param tmp_path: Directory holding the child-null Lance fixture.
    """
    primitive = pa.array([1.0, None, 3.0, 4.0], type=pa.float32())
    storage = pa.FixedSizeListArray.from_arrays(primitive, 2)
    tensor_type = pa.fixed_shape_tensor(pa.float32(), [2])
    values = pa.ExtensionArray.from_storage(tensor_type, storage)
    dataset_path = tmp_path / "child-null.lance"
    lance.write_dataset(pa.table({"vector": values}), dataset_path)

    with pytest.raises(RuntimeError, match="failed to scan conditioning column"):
        analyse_dataset(str(dataset_path), {"vector": (2,)}, row_limit=2, batch_size=1)


def test_analyse_dataset_mismatched_tensor_shape_raises(tmp_path: Path) -> None:
    """A stored tensor cannot be regrouped into a different configured shape.

    :param tmp_path: Directory holding the mismatched Lance fixture.
    """
    values = pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    )
    dataset_path = tmp_path / "mismatched.lance"
    lance.write_dataset(pa.table({"sequence": values}), dataset_path)

    with pytest.raises(RuntimeError, match="failed to scan conditioning column"):
        analyse_dataset(str(dataset_path), {"sequence": (2, 3)}, row_limit=1, batch_size=1)


def test_analyse_dataset_invalid_matpac_width_raises(tmp_path: Path) -> None:
    """MATPAC++ must retain its five 768-channel frequency bands.

    :param tmp_path: Directory holding the invalid MATPAC++ Lance fixture.
    """
    values = pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((1, 4, 3), dtype=np.float32))
    dataset_path = tmp_path / "matpac.lance"
    lance.write_dataset(pa.table({"matpac_plus": values}), dataset_path)

    with pytest.raises(RuntimeError, match="failed to scan conditioning column"):
        analyse_dataset(str(dataset_path), {"matpac_plus": (4, 3)}, row_limit=1, batch_size=1)


def test_analyse_dataset_nested_fixed_size_lists_reports_sequence_statistics(
    tmp_path: Path,
) -> None:
    """Nested Arrow lists decode to the registered channel-frame shape.

    :param tmp_path: Directory holding the real nested-list Lance fixture.
    """
    primitive = pa.array(np.arange(12, dtype=np.float32))
    frames = pa.FixedSizeListArray.from_arrays(primitive, 3)
    rows = pa.FixedSizeListArray.from_arrays(frames, 2)
    dataset_path = tmp_path / "nested.lance"
    lance.write_dataset(pa.table({"sequence": rows}), dataset_path)

    results = analyse_dataset(str(dataset_path), {"sequence": (2, 3)}, row_limit=2, batch_size=1)

    assert results["sequence"].rows == 2
    assert results["sequence"].global_mean == pytest.approx(5.5)


def test_analyse_dataset_percent_encoded_file_uri_opens_dataset(tmp_path: Path) -> None:
    """A percent-encoded file URI resolves to its local Lance dataset.

    :param tmp_path: Directory holding a Lance fixture whose name contains a space.
    """
    dataset_path = tmp_path / "sample data.lance"
    values = pa.FixedShapeTensorArray.from_numpy_ndarray(np.array([[3.0, 4.0]], dtype=np.float32))
    lance.write_dataset(pa.table({"vector": values}), dataset_path)

    results = analyse_dataset(dataset_path.as_uri(), {"vector": (2,)}, row_limit=1, batch_size=1)

    assert results["vector"].row_l2_mean == pytest.approx(5.0)


def test_cli_real_lance_dataset_writes_markdown_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real script entrypoint reads Lance and materializes its report.

    :param tmp_path: Directory holding the real Lance fixture and report.
    :param monkeypatch: Process argument isolation for the entrypoint.
    """
    config_dir = tmp_path / "conditioning"
    config_dir.mkdir()
    (config_dir / "vector.yaml").write_text(
        "model:\n  conditioning:\n    column: vector\n    input_shape: [2]\n"
    )
    values = pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    )
    dataset_path = tmp_path / "sample.lance"
    lance.write_dataset(pa.table({"vector": values}), dataset_path)
    output_path = tmp_path / "report.md"
    script = Path(__file__).parents[3] / "scripts/dev/characterise_conditioning_columns.py"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            str(dataset_path),
            "--config-dir",
            str(config_dir),
            "--rows",
            "2",
            "--output",
            str(output_path),
        ],
    )

    runpy.run_path(str(script), run_name="__main__")

    report = output_path.read_text()
    assert "| vector | 2 |" in report
    assert "2.500000" in report
