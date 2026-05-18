"""Tests for `synth_setter.pipeline.data.reshard` shard-size parameterization (#1091)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from synth_setter.pipeline.data.reshard import main

AUDIO_TRAILING_SHAPE = (1, 8)
MEL_TRAILING_SHAPE = (1, 4, 4)
PARAM_TRAILING_SHAPE = (5,)


def _write_shard(path: Path, rows: int, fill: float) -> None:
    """Write a shard whose every element is ``fill`` — lets assertions trace rows to source.

    :param path: Destination ``shard-*.h5`` location.
    :param rows: Leading-axis length of the three datasets.
    :param fill: Constant value used for every element, so a virtual-dataset row's source
        shard is recoverable downstream.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "audio",
            data=np.full((rows, *AUDIO_TRAILING_SHAPE), fill, dtype=np.float32),
        )
        f.create_dataset(
            "mel_spec",
            data=np.full((rows, *MEL_TRAILING_SHAPE), fill, dtype=np.float32),
        )
        f.create_dataset(
            "param_array",
            data=np.full((rows, *PARAM_TRAILING_SHAPE), fill, dtype=np.float32),
        )


@pytest.fixture
def shard_dir(tmp_path: Path) -> Path:
    """Directory of glob-ordered shards where shard ``i`` is uniformly filled with ``i + 1``.

    :param tmp_path: Per-test temp directory (pytest fixture).
    :returns: The directory containing the shard files.
    :rtype: Path
    """
    for idx in range(4):
        _write_shard(tmp_path / f"shard-{idx:06d}.h5", rows=3, fill=float(idx + 1))
    return tmp_path


def test_help_advertises_shard_size_flag_with_range_constraint() -> None:
    """``reshard --help`` documents ``--shard-size``/``-s`` and the ``x>=1`` constraint."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--shard-size" in result.output
    assert "-s" in result.output
    assert "x>=1" in result.output


def test_reshard_writes_split_files_with_lengths_proportional_to_shard_size(
    shard_dir: Path,
) -> None:
    """Each split file's leading axis equals ``files_in_split * shard_size``.

    :param shard_dir: Directory of source shards (fixture).
    """
    runner = CliRunner()
    result = runner.invoke(
        main,
        [str(shard_dir), "-t", "2", "-v", "1", "-e", "1", "-s", "3"],
    )
    assert result.exit_code == 0, result.output

    with h5py.File(shard_dir / "train.h5", "r") as f:
        assert cast(h5py.Dataset, f["audio"]).shape == (6, *AUDIO_TRAILING_SHAPE)
        assert cast(h5py.Dataset, f["mel_spec"]).shape == (6, *MEL_TRAILING_SHAPE)
        assert cast(h5py.Dataset, f["param_array"]).shape == (6, *PARAM_TRAILING_SHAPE)
    with h5py.File(shard_dir / "val.h5", "r") as f:
        assert cast(h5py.Dataset, f["audio"]).shape == (3, *AUDIO_TRAILING_SHAPE)
    with h5py.File(shard_dir / "test.h5", "r") as f:
        assert cast(h5py.Dataset, f["audio"]).shape == (3, *AUDIO_TRAILING_SHAPE)


def test_reshard_split_file_reads_back_shard_data_in_glob_order(shard_dir: Path) -> None:
    """Virtual dataset surfaces each source shard's rows contiguously and in glob order.

    :param shard_dir: Directory of source shards (fixture).
    """
    runner = CliRunner()
    result = runner.invoke(
        main,
        [str(shard_dir), "-t", "2", "-v", "1", "-e", "1", "-s", "3"],
    )
    assert result.exit_code == 0, result.output

    with h5py.File(shard_dir / "train.h5", "r") as f:
        audio = cast(h5py.Dataset, f["audio"])[:]
        params = cast(h5py.Dataset, f["param_array"])[:]

    expected_audio = np.concatenate(
        [
            np.full((3, *AUDIO_TRAILING_SHAPE), 1.0, dtype=np.float32),
            np.full((3, *AUDIO_TRAILING_SHAPE), 2.0, dtype=np.float32),
        ]
    )
    expected_params = np.concatenate(
        [
            np.full((3, *PARAM_TRAILING_SHAPE), 1.0, dtype=np.float32),
            np.full((3, *PARAM_TRAILING_SHAPE), 2.0, dtype=np.float32),
        ]
    )
    np.testing.assert_array_equal(audio, expected_audio)
    np.testing.assert_array_equal(params, expected_params)


@pytest.mark.parametrize("bad_value", ["0", "-1"])
def test_reshard_rejects_non_positive_shard_size(shard_dir: Path, bad_value: str) -> None:
    """``--shard-size`` below 1 fails CLI validation before any split file is written.

    :param shard_dir: Directory of source shards (fixture).
    :param bad_value: Out-of-range string passed to ``-s``.
    """
    runner = CliRunner()
    result = runner.invoke(
        main,
        [str(shard_dir), "-t", "2", "-v", "1", "-e", "1", "-s", bad_value],
    )
    assert result.exit_code != 0
    assert not (shard_dir / "train.h5").exists()


def test_reshard_raises_when_file_count_does_not_match_split_sum(shard_dir: Path) -> None:
    """An ``AssertionError`` fires when ``train + val + test`` disagrees with the file count.

    :param shard_dir: Directory of source shards (fixture).
    """
    runner = CliRunner()
    result = runner.invoke(
        main,
        [str(shard_dir), "-t", "3", "-v", "1", "-e", "1", "-s", "3"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, AssertionError)
