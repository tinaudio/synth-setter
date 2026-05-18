"""Tests for ``synth_setter.pipeline.data.reshard``.

The reshard CLI consumes a ``DatasetSpec`` to determine train/val/test split
sizes instead of taking sample-count CLI flags. These tests pin three contracts:

1. Per-split shard counts are derived from ``spec.train_val_test_sizes`` and
   ``spec.render.samples_per_shard`` and the globbed shard list is sliced
   accordingly.
2. When the on-disk shard count drifts from the spec's expected total the CLI
   fails loud with a message naming both totals.
3. The CLI accepts ``r2://`` spec URIs via ``load_spec_from_uri``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from synth_setter.pipeline.data import reshard
from synth_setter.pipeline.schemas.spec import DatasetSpec


def _write_fake_shard(path: Path, rows: int) -> None:  # noqa: DOC101, DOC103
    """Write a minimal HDF5 shard with the three datasets reshard expects."""
    with h5py.File(path, "w") as f:
        f.create_dataset("audio", data=np.zeros((rows, 2, 4), dtype=np.float32))
        f.create_dataset("mel_spec", data=np.zeros((rows, 2, 4, 4), dtype=np.float32))
        f.create_dataset("param_array", data=np.zeros((rows, 3), dtype=np.float32))


def _make_spec(  # noqa: DOC101, DOC103, DOC201, DOC203
    kwargs: dict[str, Any], *, sizes: tuple[int, int, int]
) -> DatasetSpec:
    """Build a DatasetSpec from fixture kwargs with the given split sizes."""
    kwargs = dict(kwargs)
    kwargs["train_val_test_sizes"] = list(sizes)
    return DatasetSpec(**kwargs)


def _make_shards(  # noqa: DOC101, DOC103, DOC201, DOC203
    root: Path, count: int, rows: int
) -> list[Path]:
    """Write ``count`` fake shards under ``root`` and return their paths."""
    paths: list[Path] = []
    for i in range(count):
        path = root / f"shard-{i:06d}.h5"
        _write_fake_shard(path, rows)
        paths.append(path)
    return paths


def test_split_shard_counts_matches_spec(  # noqa: DOC101, DOC103
    valid_dataset_spec_kwargs: dict[str, Any],
) -> None:
    """``split_shard_counts`` divides each split size by ``samples_per_shard``."""
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(30000, 20000, 10000))

    assert reshard.split_shard_counts(spec) == (3, 2, 1)


def test_assign_shards_to_splits_slices_in_spec_order(  # noqa: DOC101, DOC103
    valid_dataset_spec_kwargs: dict[str, Any],
) -> None:
    """Sorted shards are sliced train|val|test in order, no overlap."""
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(30000, 20000, 10000))
    shard_paths = [Path(f"shard-{i:06d}.h5") for i in range(6)]

    splits = reshard.assign_shards_to_splits(shard_paths, spec)

    assert splits["train"] == shard_paths[:3]
    assert splits["val"] == shard_paths[3:5]
    assert splits["test"] == shard_paths[5:]


def test_assign_shards_to_splits_fails_loud_on_count_mismatch(  # noqa: DOC101, DOC103
    valid_dataset_spec_kwargs: dict[str, Any],
) -> None:
    """An on-disk shard count != spec's expected total raises with both numbers."""
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(30000, 20000, 10000))
    # Spec expects 6 shards; supply 5 to provoke drift.
    too_few = [Path(f"shard-{i:06d}.h5") for i in range(5)]

    with pytest.raises(ValueError, match=r"expected.*6.*observed.*5"):
        reshard.assign_shards_to_splits(too_few, spec)


def test_cli_writes_split_files_using_spec(  # noqa: DOC101, DOC103
    tmp_path: Path, valid_dataset_spec_kwargs: dict[str, Any]
) -> None:
    """End-to-end: CLI consumes a local spec, slices shards, and writes split files."""
    rows = valid_dataset_spec_kwargs["render"]["samples_per_shard"]
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(rows * 3, rows * 2, rows * 1))

    _make_shards(tmp_path, count=6, rows=rows)
    spec_path = tmp_path / "input_spec.json"
    spec_path.write_text(spec.model_dump_json())

    result = CliRunner().invoke(
        reshard.main,
        [str(tmp_path), "--spec", str(spec_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    for split, expected_rows in (("train", rows * 3), ("val", rows * 2), ("test", rows * 1)):
        split_file = tmp_path / f"{split}.h5"
        assert split_file.exists()
        with h5py.File(split_file, "r") as f:
            for name in ("audio", "mel_spec", "param_array"):
                dataset = f[name]
                # ``h5py.File.__getitem__`` returns ``Group | Dataset | Datatype``;
                # narrow to ``Dataset`` so ``.shape`` is type-checked.
                assert isinstance(dataset, h5py.Dataset)
                assert dataset.shape[0] == expected_rows


def test_cli_fails_loud_when_shards_disagree_with_spec(  # noqa: DOC101, DOC103
    tmp_path: Path, valid_dataset_spec_kwargs: dict[str, Any]
) -> None:
    """Globbed-vs-spec mismatch is a fail-loud ValueError, not a silent slice."""
    rows = valid_dataset_spec_kwargs["render"]["samples_per_shard"]
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(rows * 3, rows * 2, rows * 1))

    # Spec expects 6 shards; write only 5.
    _make_shards(tmp_path, count=5, rows=rows)
    spec_path = tmp_path / "input_spec.json"
    spec_path.write_text(spec.model_dump_json())

    result = CliRunner().invoke(
        reshard.main,
        [str(tmp_path), "--spec", str(spec_path)],
    )

    assert result.exit_code != 0
    message = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "expected" in message
    assert "6" in message and "5" in message


def test_cli_accepts_r2_spec_uri(  # noqa: DOC101, DOC103
    tmp_path: Path, valid_dataset_spec_kwargs: dict[str, Any]
) -> None:
    """``--spec r2://...`` is loaded via ``load_spec_from_uri`` (mocked, no real R2 I/O)."""
    rows = valid_dataset_spec_kwargs["render"]["samples_per_shard"]
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(rows * 3, rows * 2, rows * 1))

    _make_shards(tmp_path, count=6, rows=rows)
    spec_uri = "r2://intermediate-data/data/foo/input_spec.json"

    with mock.patch.object(reshard, "load_spec_from_uri", return_value=spec) as loader:
        result = CliRunner().invoke(
            reshard.main,
            [str(tmp_path), "--spec", spec_uri],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    loader.assert_called_once_with(spec_uri)
    assert (tmp_path / "train.h5").exists()
