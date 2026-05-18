"""Tests for ``synth_setter.pipeline.data.reshard``.

The reshard CLI consumes a ``DatasetSpec`` to determine split sizes and the
exact shard filenames to read. These tests pin three contracts:

1. Per-split shard counts are derived from ``spec.train_val_test_sizes`` and
   ``spec.render.samples_per_shard``.
2. The CLI builds its shard list from ``spec.shards`` (not a filesystem glob),
   so it reads the canonical filenames defined by the spec and fails loud when
   one is missing.
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
    kwargs: dict[str, Any],
    *,
    sizes: tuple[int, int, int],
    samples_per_shard: int | None = None,
) -> DatasetSpec:
    """Build a DatasetSpec from fixture kwargs, optionally overriding samples_per_shard."""
    kwargs = dict(kwargs)
    kwargs["train_val_test_sizes"] = list(sizes)
    if samples_per_shard is not None:
        render = dict(kwargs["render"])
        render["samples_per_shard"] = samples_per_shard
        kwargs["render"] = render
    return DatasetSpec(**kwargs)


def _materialize_shards_for(spec: DatasetSpec, root: Path) -> None:  # noqa: DOC101, DOC103
    """Write the exact shard files ``spec.shards`` names under ``root``."""
    rows = spec.render.samples_per_shard
    for shard in spec.shards:
        _write_fake_shard(root / shard.filename, rows)


def test_split_shard_counts_matches_spec(  # noqa: DOC101, DOC103
    valid_dataset_spec_kwargs: dict[str, Any],
) -> None:
    """``split_shard_counts`` divides each split size by ``samples_per_shard``."""
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(30000, 20000, 10000))

    assert reshard.split_shard_counts(spec) == (3, 2, 1)


def test_cli_writes_split_files_using_spec(  # noqa: DOC101, DOC103
    tmp_path: Path, valid_dataset_spec_kwargs: dict[str, Any]
) -> None:
    """End-to-end: CLI consumes a local spec, reads spec.shards, writes split files."""
    rows = valid_dataset_spec_kwargs["render"]["samples_per_shard"]
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(rows * 3, rows * 2, rows * 1))

    _materialize_shards_for(spec, tmp_path)
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


def test_cli_skips_empty_split(  # noqa: DOC101, DOC103
    tmp_path: Path, valid_dataset_spec_kwargs: dict[str, Any]
) -> None:
    """A zero-sized split (e.g. ``test=0``) writes no file and exits zero."""
    rows = valid_dataset_spec_kwargs["render"]["samples_per_shard"]
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(rows * 3, rows * 1, 0))

    _materialize_shards_for(spec, tmp_path)
    spec_path = tmp_path / "input_spec.json"
    spec_path.write_text(spec.model_dump_json())

    result = CliRunner().invoke(
        reshard.main,
        [str(tmp_path), "--spec", str(spec_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "train.h5").exists()
    assert (tmp_path / "val.h5").exists()
    assert not (tmp_path / "test.h5").exists()


def test_cli_uses_spec_samples_per_shard_not_a_hardcoded_value(  # noqa: DOC101, DOC103
    tmp_path: Path, valid_dataset_spec_kwargs: dict[str, Any]
) -> None:
    """Resharded VDS rows-per-split track ``spec.render.samples_per_shard``, not 10k."""
    non_default = 7  # deliberately tiny and != fixture default to catch hardcoded sizes
    spec = _make_spec(
        valid_dataset_spec_kwargs,
        sizes=(non_default * 3, non_default * 2, non_default * 1),
        samples_per_shard=non_default,
    )

    _materialize_shards_for(spec, tmp_path)
    spec_path = tmp_path / "input_spec.json"
    spec_path.write_text(spec.model_dump_json())

    result = CliRunner().invoke(
        reshard.main,
        [str(tmp_path), "--spec", str(spec_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    for split, expected_rows in (
        ("train", non_default * 3),
        ("val", non_default * 2),
        ("test", non_default * 1),
    ):
        with h5py.File(tmp_path / f"{split}.h5", "r") as f:
            dataset = f["audio"]
            assert isinstance(dataset, h5py.Dataset)
            assert dataset.shape[0] == expected_rows


def test_cli_fails_loud_when_spec_filename_is_missing(  # noqa: DOC101, DOC103
    tmp_path: Path, valid_dataset_spec_kwargs: dict[str, Any]
) -> None:
    """If a filename in ``spec.shards`` is absent on disk, the CLI fails (not silent)."""
    rows = valid_dataset_spec_kwargs["render"]["samples_per_shard"]
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(rows * 3, rows * 2, rows * 1))

    _materialize_shards_for(spec, tmp_path)
    # Delete one canonical shard; a stale extra file is intentionally left.
    (tmp_path / spec.shards[0].filename).unlink()
    (tmp_path / "shard-999999.h5").touch()
    spec_path = tmp_path / "input_spec.json"
    spec_path.write_text(spec.model_dump_json())

    result = CliRunner().invoke(
        reshard.main,
        [str(tmp_path), "--spec", str(spec_path)],
    )

    assert result.exit_code != 0


def test_cli_accepts_r2_spec_uri(  # noqa: DOC101, DOC103
    tmp_path: Path, valid_dataset_spec_kwargs: dict[str, Any]
) -> None:
    """``--spec r2://...`` is loaded via ``load_spec_from_uri`` (mocked, no real R2 I/O)."""
    rows = valid_dataset_spec_kwargs["render"]["samples_per_shard"]
    spec = _make_spec(valid_dataset_spec_kwargs, sizes=(rows * 3, rows * 2, rows * 1))

    _materialize_shards_for(spec, tmp_path)
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
