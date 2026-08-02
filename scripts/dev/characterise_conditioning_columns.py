#!/usr/bin/env python
"""Stream conditioning columns from Lance and report their numeric scales.

Typical usage::

    uv run python scripts/dev/characterise_conditioning_columns.py DATASET.lance \
        --rows 1000 --output conditioning-statistics.md
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import lance
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import yaml
from pydantic import BaseModel, ConfigDict

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import _retry_lance_read

DEFAULT_ROWS = 1_000
DEFAULT_BATCH_SIZE = 16
DEAD_CHANNEL_STD = 1e-6
MATPAC_BAND_WIDTH = 768
MATPAC_FREQUENCY_BANDS = 5

type FloatArray = npt.NDArray[np.float16 | np.float32 | np.float64]
type Float64Array = npt.NDArray[np.float64]
type MomentValue = float | Float64Array


class _ConditioningConfig(BaseModel):
    """Validate the fields consumed from a conditioning profile.

    .. attribute :: model_config

        Strict Pydantic validation policy.

    .. attribute :: column

        Lance column name.

    .. attribute :: input_shape

        Channel-first stored shape.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    column: str
    input_shape: list[int]


@dataclass(frozen=True)
class ColumnStatistics:
    """Summary statistics for one vector or sequence column.

    .. attribute :: rows

        Number of sampled rows.

    .. attribute :: channel_mean

        Minimum, median, and maximum channel means.

    .. attribute :: channel_std

        Minimum, median, and maximum channel standard deviations.

    .. attribute :: global_mean

        Mean across all values.

    .. attribute :: global_std

        Standard deviation across all values.

    .. attribute :: global_min

        Minimum value.

    .. attribute :: global_max

        Maximum value.

    .. attribute :: dead_channels

        Channels below the dead-channel standard-deviation threshold.

    .. attribute :: row_l2_mean

        Mean whole-row L2 norm.

    .. attribute :: row_l2_std

        Standard deviation of whole-row L2 norms.

    .. attribute :: frame_l2_cv_mean

        Mean within-row frame-L2 coefficient of variation for sequences.
    """

    rows: int
    channel_mean: tuple[float, float, float]
    channel_std: tuple[float, float, float]
    global_mean: float
    global_std: float
    global_min: float
    global_max: float
    dead_channels: int
    row_l2_mean: float
    row_l2_std: float
    frame_l2_cv_mean: float | None


class StreamingStatistics:
    """Accumulate population statistics without retaining sampled rows."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize empty moments for a channel-first shape.

        :param shape: Vector width or sequence channel/frame dimensions.
        :raises ValueError: The shape is not a positive rank-one or rank-two shape.
        """
        if len(shape) not in (1, 2) or any(size <= 0 for size in shape):
            raise ValueError(f"expected a positive vector or channel-frame shape, got {shape}")
        self.shape = shape
        self.rows = 0
        self._channel_count = 0
        self._channel_mean = np.zeros(shape[0], dtype=np.float64)
        self._channel_m2 = np.zeros(shape[0], dtype=np.float64)
        self._global_count = 0
        self._global_mean = 0.0
        self._global_m2 = 0.0
        self._global_min = math.inf
        self._global_max = -math.inf
        self._row_count = 0
        self._row_mean = 0.0
        self._row_m2 = 0.0
        self._frame_cv_sum = 0.0

    def update(self, values: FloatArray) -> None:
        """Add a row batch shaped ``(rows, *shape)``.

        :param values: Finite vector or channel-frame rows.
        :raises ValueError: Values have the wrong shape or contain non-finite data.
        """
        expected_tail = self.shape
        if values.ndim != len(expected_tail) + 1 or values.shape[1:] != expected_tail:
            raise ValueError(f"expected rows shaped {expected_tail}, got {values.shape}")
        if len(values) == 0:
            return
        values64 = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values64).all():
            raise ValueError("conditioning values must be finite")

        self._update_value_moments(values64)
        self._update_row_norms(values64)
        self._update_frame_cv(values64)
        self.rows += len(values64)

    def _update_value_moments(self, values: Float64Array) -> None:
        """Accumulate channel and global value moments.

        :param values: Validated float64 rows.
        """
        channel_observations = np.moveaxis(values, 1, -1).reshape(-1, self.shape[0])
        self._channel_count, self._channel_mean, self._channel_m2 = _merge_moments(
            self._channel_count,
            self._channel_mean,
            self._channel_m2,
            channel_observations,
            axis=0,
        )
        flattened = values.reshape(-1)
        self._global_count, self._global_mean, self._global_m2 = _merge_moments(
            self._global_count,
            self._global_mean,
            self._global_m2,
            flattened,
            axis=None,
        )
        self._global_min = min(self._global_min, float(flattened.min()))
        self._global_max = max(self._global_max, float(flattened.max()))

    def _update_row_norms(self, values: Float64Array) -> None:
        """Accumulate whole-row L2-norm moments.

        :param values: Validated float64 rows.
        """
        row_norms = np.linalg.norm(values.reshape(len(values), -1), axis=1)
        self._row_count, self._row_mean, self._row_m2 = _merge_moments(
            self._row_count,
            self._row_mean,
            self._row_m2,
            row_norms,
            axis=None,
        )

    def _update_frame_cv(self, values: Float64Array) -> None:
        """Accumulate within-row frame-norm variation for sequences.

        :param values: Validated float64 rows.
        """
        if len(self.shape) != 2:
            return
        frame_norms = np.linalg.norm(values, axis=1)
        frame_means = frame_norms.mean(axis=1)
        frame_stds = frame_norms.std(axis=1)
        frame_cv = np.divide(
            frame_stds,
            frame_means,
            out=np.zeros_like(frame_stds),
            where=frame_means > 0,
        )
        self._frame_cv_sum += float(frame_cv.sum())

    def result(self) -> ColumnStatistics:
        """Return the accumulated population statistics.

        :returns: Per-channel summaries and global, row-norm, and frame-norm metrics.
        :raises ValueError: No rows have been accumulated.
        """
        if self.rows == 0:
            raise ValueError("cannot report statistics for zero rows")
        channel_std = np.sqrt(self._channel_m2 / self._channel_count)
        return ColumnStatistics(
            rows=self.rows,
            channel_mean=_min_median_max(np.asarray(self._channel_mean)),
            channel_std=_min_median_max(channel_std),
            global_mean=float(self._global_mean),
            global_std=math.sqrt(float(self._global_m2) / self._global_count),
            global_min=self._global_min,
            global_max=self._global_max,
            dead_channels=int(np.count_nonzero(channel_std < DEAD_CHANNEL_STD)),
            row_l2_mean=float(self._row_mean),
            row_l2_std=math.sqrt(float(self._row_m2) / self._row_count),
            frame_l2_cv_mean=(self._frame_cv_sum / self.rows if len(self.shape) == 2 else None),
        )


def _merge_moments(
    count: int,
    mean: MomentValue,
    m2: MomentValue,
    values: Float64Array,
    *,
    axis: int | None,
) -> tuple[
    int,
    MomentValue,
    MomentValue,
]:
    """Merge one numeric batch into scalar or per-channel moments.

    :param count: Existing observation count.
    :param mean: Existing scalar or channel means.
    :param m2: Existing sum of squared deviations.
    :param values: New observations.
    :param axis: Observation axis, or ``None`` for scalar moments.
    :returns: Merged count, mean, and sum of squared deviations.
    """
    batch_count = values.shape[axis] if axis is not None else values.size
    batch_mean = values.mean(axis=axis)
    batch_m2 = ((values - batch_mean) ** 2).sum(axis=axis)
    if count == 0:
        return batch_count, batch_mean, batch_m2
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * batch_count / total
    merged_m2 = m2 + batch_m2 + delta**2 * count * batch_count / total
    return total, merged_mean, merged_m2


def _min_median_max(values: Float64Array) -> tuple[float, float, float]:
    """Summarize a numeric vector.

    :param values: Values to summarize.
    :returns: Minimum, median, and maximum.
    """
    return float(values.min()), float(np.median(values)), float(values.max())


def discover_conditioning_columns(config_dir: Path) -> dict[str, tuple[int, ...]]:
    """Discover materialized conditioning columns and shapes from Hydra profiles.

    :param config_dir: Directory containing conditioning YAML profiles.
    :returns: Column names mapped to their channel-first shapes.
    :raises ValueError: Duplicate profiles disagree about a column shape.
    """
    columns: dict[str, tuple[int, ...]] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            continue
        model = raw.get("model")
        conditioning = model.get("conditioning") if isinstance(model, dict) else None
        if not isinstance(conditioning, dict) or "column" not in conditioning:
            continue
        parsed = _ConditioningConfig.model_validate(conditioning)
        shape = tuple(parsed.input_shape)
        previous = columns.get(parsed.column)
        if previous is not None and previous != shape:
            raise ValueError(
                f"conditioning profiles disagree about {parsed.column}: {previous} versus {shape}"
            )
        columns[parsed.column] = shape
    return columns


def matpac_band_views(values: FloatArray, band_width: int = MATPAC_BAND_WIDTH) -> list[FloatArray]:
    """Split MATPAC++ values into contiguous frequency-band channel views.

    :param values: Channel-first sequence rows.
    :param band_width: ViT width represented by each frequency patch.
    :returns: Non-copying views, one per frequency patch.
    :raises ValueError: The width is non-positive or does not divide the channel count.
    """
    if band_width <= 0:
        raise ValueError(f"band width must be positive, got {band_width}")
    if values.ndim != 3 or values.shape[1] % band_width:
        raise ValueError(f"cannot split shape {values.shape} into {band_width}-channel bands")
    return list(np.split(values, values.shape[1] // band_width, axis=1))


def _array_to_numpy(array: pa.Array, shape: tuple[int, ...]) -> FloatArray:
    """Decode a Lance Arrow array to its registered dense shape.

    :param array: Arrow extension or fixed-size-list row batch.
    :param shape: Registered shape excluding the row dimension.
    :returns: Dense NumPy rows.
    :raises ValueError: The column contains null rows.
    """
    if array.null_count:
        raise ValueError("conditioning columns must not contain null rows")
    to_tensor = getattr(array, "to_numpy_ndarray", None)
    if callable(to_tensor):
        values = to_tensor()
    elif pa.types.is_fixed_size_list(array.type):
        values = array.values.to_numpy(zero_copy_only=False).reshape(len(array), *shape)
    else:
        values = np.asarray(array.to_pylist())
    return cast(FloatArray, np.asarray(values).reshape(len(array), *shape))


def _open_dataset(dataset_uri: str) -> lance.LanceDataset:
    """Open a local or R2 Lance dataset with bounded transient retries.

    :param dataset_uri: Local, ``file://``, or R2 Lance dataset URI.
    :returns: Open Lance dataset.
    :raises RuntimeError: The dataset cannot be opened.
    """
    try:
        if r2_io.is_r2_uri(dataset_uri):
            target, storage_options = r2_io.lance_target(dataset_uri)
        else:
            target, storage_options = dataset_uri.removeprefix("file://"), None
        return _retry_lance_read(
            "conditioning_statistics_open",
            lambda: lance.dataset(target, storage_options=storage_options),
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError("failed to open conditioning statistics Lance dataset") from error


def _band_accumulators(column: str, shape: tuple[int, ...]) -> list[StreamingStatistics]:
    """Build MATPAC++ band accumulators when required.

    :param column: Conditioning column name.
    :param shape: Registered channel-first shape.
    :returns: Five band accumulators for MATPAC++, otherwise an empty list.
    :raises ValueError: MATPAC++ is not registered as a sequence.
    """
    if column != "matpac_plus":
        return []
    if len(shape) != 2:
        raise ValueError(f"matpac_plus must be a sequence, got shape {shape}")
    return [
        StreamingStatistics((MATPAC_BAND_WIDTH, shape[1])) for _ in range(MATPAC_FREQUENCY_BANDS)
    ]


def _stream_column(
    dataset: lance.LanceDataset,
    column: str,
    shape: tuple[int, ...],
    *,
    row_limit: int,
    batch_size: int,
) -> dict[str, ColumnStatistics]:
    """Stream one column and return its report rows.

    :param dataset: Open Lance dataset.
    :param column: Conditioning column name.
    :param shape: Registered channel-first shape.
    :param row_limit: Maximum sampled rows.
    :param batch_size: Lance scanner batch size.
    :returns: Main statistics plus MATPAC++ band statistics when applicable.
    """
    accumulator = StreamingStatistics(shape)
    band_accumulators = _band_accumulators(column, shape)
    for batch in dataset.scanner(columns=[column], batch_size=batch_size).to_batches():
        remaining = row_limit - accumulator.rows
        if remaining <= 0:
            break
        values = _array_to_numpy(batch.column(0), shape)[:remaining]
        accumulator.update(values)
        if band_accumulators:
            for band_accumulator, band in zip(
                band_accumulators, matpac_band_views(values), strict=True
            ):
                band_accumulator.update(band)
    results = {column: accumulator.result()}
    results.update(
        {
            f"{column} band {index}": band.result()
            for index, band in enumerate(band_accumulators, start=1)
        }
    )
    return results


def _analyse_column(
    dataset: lance.LanceDataset,
    column: str,
    shape: tuple[int, ...],
    *,
    row_limit: int,
    batch_size: int,
) -> dict[str, ColumnStatistics]:
    """Analyze one column under the bounded Lance retry policy.

    :param dataset: Open Lance dataset.
    :param column: Conditioning column name.
    :param shape: Registered channel-first shape.
    :param row_limit: Maximum sampled rows.
    :param batch_size: Lance scanner batch size.
    :returns: Main and optional MATPAC++ band report rows.
    :raises RuntimeError: The column cannot be scanned.
    """
    try:
        return _retry_lance_read(
            "conditioning_statistics_scan",
            lambda: _stream_column(
                dataset, column, shape, row_limit=row_limit, batch_size=batch_size
            ),
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"failed to scan conditioning column {column!r}") from error


def analyse_dataset(
    dataset_uri: str,
    columns: dict[str, tuple[int, ...]],
    *,
    row_limit: int,
    batch_size: int,
) -> dict[str, ColumnStatistics]:
    """Stream selected Lance columns and calculate their statistics.

    :param dataset_uri: Local, ``file://``, or R2 Lance dataset URI.
    :param columns: Selected column names and channel-first shapes.
    :param row_limit: Maximum rows sampled from each column.
    :param batch_size: Lance scanner batch size.
    :returns: Statistics keyed by column, plus MATPAC++ band rows when selected.
    """
    dataset = _open_dataset(dataset_uri)
    results: dict[str, ColumnStatistics] = {}
    for column, shape in columns.items():
        results.update(
            _analyse_column(dataset, column, shape, row_limit=row_limit, batch_size=batch_size)
        )
    return results


def render_markdown(results: dict[str, ColumnStatistics], dataset_uri: str) -> str:
    """Render statistics as a markdown table.

    :param results: Statistics keyed by report row name.
    :param dataset_uri: Analysed Lance dataset URI.
    :returns: Markdown report text.
    """
    header = (
        "| Column | Rows | Channel mean min / median / max | Channel std min / median / max "
        "| Global mean | Global std | Global min | Global max | Dead channels | "
        "Row L2 mean | Row L2 std | Mean within-row frame-L2 CV |\n"
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = [header]
    for name, stats in results.items():
        frame_cv = "—" if stats.frame_l2_cv_mean is None else _format(stats.frame_l2_cv_mean)
        rows.append(
            f"| {name} | {stats.rows} | {_triple(stats.channel_mean)} | "
            f"{_triple(stats.channel_std)} | {_format(stats.global_mean)} | "
            f"{_format(stats.global_std)} | {_format(stats.global_min)} | "
            f"{_format(stats.global_max)} | {stats.dead_channels} | "
            f"{_format(stats.row_l2_mean)} | {_format(stats.row_l2_std)} | {frame_cv} |"
        )
    return (
        f"# Conditioning-column statistics\n\nDataset: `{dataset_uri}`\n\n"
        + "\n".join(rows)
        + "\n"
    )


def _format(value: float) -> str:
    """Format one report value to six decimal places.

    :param value: Numeric report value.
    :returns: Fixed-precision decimal text.
    """
    return f"{value:.6f}"


def _triple(values: tuple[float, float, float]) -> str:
    """Format a minimum/median/maximum triple.

    :param values: Three report values.
    :returns: Slash-separated fixed-precision text.
    """
    return " / ".join(_format(value) for value in values)


def _parse_args() -> argparse.Namespace:
    """Parse command-line report settings.

    :returns: Parsed dataset, selection, streaming, and output settings.
    """
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_uri")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=project_root / "src/synth_setter/configs/conditioning",
    )
    parser.add_argument("--columns", nargs="*", help="Optional subset of discovered columns")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the command-line report generator.

    :raises ValueError: Row or batch limits are non-positive, or a requested column has no cached
        conditioning profile.
    """
    args = _parse_args()
    if args.rows <= 0 or args.batch_size <= 0:
        raise ValueError("--rows and --batch-size must be positive")
    discovered = discover_conditioning_columns(args.config_dir)
    selected_names = args.columns or list(discovered)
    missing = sorted(set(selected_names) - set(discovered))
    if missing:
        raise ValueError(f"columns are not registered in conditioning configs: {missing}")
    selected = {name: discovered[name] for name in selected_names}
    results = analyse_dataset(
        args.dataset_uri,
        selected,
        row_limit=args.rows,
        batch_size=args.batch_size,
    )
    report = render_markdown(results, args.dataset_uri)
    if args.output is None:
        sys.stdout.write(report)
    else:
        args.output.write_text(report)


if __name__ == "__main__":
    main()
