"""Behavioral tests for the eval predictions Lance table."""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    ParamSpec,
)
from synth_setter.evaluation.prediction_table import (
    prediction_rows,
    write_prediction_table,
)


def _spec() -> ParamSpec:
    """Build a three-wide spec whose rendered values differ from raw model output.

    :returns: Spec with one clipped, one snapped, and one discrete parameter.
    """
    return ParamSpec(
        [
            ContinuousParameter("gain"),
            CategoricalParameter("mode", ["a", "b"], raw_values=[0.25, 0.75], encoding="scalar"),
        ],
        [DiscreteLiteralParameter("pitch", min=0, max=4)],
    )


def _column(table: pa.Table, name: str) -> np.ndarray:
    """Read one tensor column back as a ``(rows, width)`` array.

    :param table: Arrow table holding the column.
    :param name: Column to read.
    :returns: Stacked row values.
    """
    return table.column(name).combine_chunks().to_numpy_ndarray()


def test_prediction_rows_keeps_raw_model_output_unchanged() -> None:
    """The raw column records what the model predicted, including out-of-range values."""
    predictions = np.array([[1.5, 0.2, 0.1]], dtype=np.float32)

    table = prediction_rows(predictions, None, _spec())

    np.testing.assert_array_equal(
        _column(table, "pred_params_raw"), np.array([[1.5, 0.2, 0.1]], dtype=np.float32)
    )


def test_prediction_rows_effective_column_holds_rendered_values() -> None:
    """The effective column is the clipped and snapped row a renderer actually plays."""
    predictions = np.array([[1.5, 0.2, 0.1]], dtype=np.float32)

    table = prediction_rows(predictions, None, _spec())

    np.testing.assert_array_equal(
        _column(table, "pred_params_effective"), np.array([[1.0, 0.5, 0.0]], dtype=np.float32)
    )


def test_prediction_rows_stores_target_params_when_given() -> None:
    """Ground-truth params ride along so one row is a complete render request."""
    predictions = np.zeros((1, 3), dtype=np.float32)
    targets = np.array([[-1.0, 0.5, 1.0]], dtype=np.float32)

    table = prediction_rows(predictions, targets, _spec())

    np.testing.assert_array_equal(
        _column(table, "target_params"), np.array([[-1.0, 0.5, 1.0]], dtype=np.float32)
    )


def test_prediction_rows_without_targets_omits_target_column() -> None:
    """Corpora with no ground-truth patch produce a table without a target column."""
    predictions = np.zeros((1, 3), dtype=np.float32)

    table = prediction_rows(predictions, None, _spec())

    assert "target_params" not in table.column_names


def test_prediction_rows_nonfinite_prediction_raises_value_error() -> None:
    """A NaN prediction is rejected at write time, before any worker renders it."""
    predictions = np.array([[np.nan, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        prediction_rows(predictions, None, _spec())


def test_prediction_rows_wrong_width_raises_value_error() -> None:
    """A prediction narrower than the spec cannot be silently mis-decoded."""
    predictions = np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        prediction_rows(predictions, None, _spec())


def test_prediction_rows_target_row_count_mismatch_raises_value_error() -> None:
    """Predictions and targets must pair up row for row."""
    predictions = np.zeros((2, 3), dtype=np.float32)
    targets = np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="target"):
        prediction_rows(predictions, targets, _spec())


def test_write_prediction_table_numbers_rows_across_batches(tmp_path: Path) -> None:
    """Batches land as one table whose row ids count through them in order.

    :param tmp_path: Hosts the Lance table.
    """
    uri = tmp_path / "predictions.lance"
    first = prediction_rows(
        np.array([[1.5, 0.2, 0.1], [0.0, 0.0, 0.0]], dtype=np.float32), None, _spec()
    )
    second = prediction_rows(np.array([[-1.5, -0.9, 0.9]], dtype=np.float32), None, _spec())

    write_prediction_table(uri, [first, second])

    stored = lance.dataset(str(uri)).to_table()
    assert stored.column("row_id").to_pylist() == [0, 1, 2]
    np.testing.assert_array_equal(
        _column(stored, "pred_params_raw")[2], np.array([-1.5, -0.9, 0.9], dtype=np.float32)
    )


def test_write_prediction_table_replaces_stale_table(tmp_path: Path) -> None:
    """A new predict pass into the same directory replaces the previous pass's rows.

    :param tmp_path: Hosts the Lance table.
    """
    uri = tmp_path / "predictions.lance"
    stale = prediction_rows(np.full((2, 3), 0.5, dtype=np.float32), None, _spec())
    fresh = prediction_rows(np.zeros((1, 3), dtype=np.float32), None, _spec())
    write_prediction_table(uri, [stale])

    write_prediction_table(uri, [fresh])

    assert lance.dataset(str(uri)).to_table().column("row_id").to_pylist() == [0]


def test_write_prediction_table_batches_disagreeing_on_targets_raises(tmp_path: Path) -> None:
    """A pass cannot carry ground truth for only some of its batches.

    :param tmp_path: Hosts the Lance table.
    """
    predictions = np.zeros((1, 3), dtype=np.float32)
    without_targets = prediction_rows(predictions, None, _spec())
    with_targets = prediction_rows(predictions, predictions, _spec())

    with pytest.raises(ValueError, match="[Ss]chema"):
        write_prediction_table(tmp_path / "predictions.lance", [without_targets, with_targets])
