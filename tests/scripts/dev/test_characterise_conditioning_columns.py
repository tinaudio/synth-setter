"""Behavior tests for conditioning-column statistics."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from scripts.dev.characterise_conditioning_columns import (
    StreamingStatistics,
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


def test_matpac_band_views_splits_channel_axis_without_changing_frames() -> None:
    """MATPAC band views split only the channel dimension."""
    values = np.arange(24, dtype=np.float32).reshape(1, 8, 3)

    bands = matpac_band_views(values, band_width=2)

    assert len(bands) == 4
    np.testing.assert_array_equal(bands[2], values[:, 4:6, :])


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
