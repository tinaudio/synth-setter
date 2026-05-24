"""Pin the WDS / HDF5 readers in ``test_finalize_artifact_oracle`` against synthetic shards.

The integration test that consumes these helpers requires a live R2
generate + finalize round-trip; the fixtures below mirror the writer's
on-disk shape from :func:`synth_setter.data.vst.writers.save_wds_samples`
so any layout drift between writer and reader trips at unit speed.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from tests.integration.test_finalize_artifact_oracle import (
    _load_param_array_from_hdf5,
    _load_param_array_from_wds_tar,
)


def _write_wds_tar(
    tar_path: Path,
    records: list[tuple[str, np.ndarray]],
) -> None:
    """Write a synthetic WDS shard matching ``save_wds_samples``'s on-disk shape.

    The real writer keys each tar record by ``f"{start_idx:08d}"`` and emits
    ``<key>.param_array.npy`` whose payload is ``np.stack([s.param_array for s
    in batch])`` — i.e. a 2-D ``(B, P)`` array, never a 1-D row. The fixture
    mirrors that exactly so the reader is exercised against the writer's real
    contract, not a simplified one.

    :param tar_path: Destination shard path.
    :param records: ``(key, payload)`` pairs; ``payload`` must be ``(B, P)`` so
        the assertion below catches a regression to 3-D stacking.
    """
    with tarfile.open(tar_path, mode="w") as tar:
        for key, payload in records:
            assert payload.ndim == 2, "writer always emits (B, P) per record"
            buf = io.BytesIO()
            np.save(buf, payload.astype(np.float32), allow_pickle=False)
            data = buf.getvalue()
            info = tarfile.TarInfo(name=f"{key}.param_array.npy")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_load_param_array_from_wds_tar_concatenates_batched_records(tmp_path: Path) -> None:
    """Multi-record shard with ``(B, P)`` payloads loads as a flat ``(sum(B), P)`` array.

    Two ``(4, 3)`` records must collapse to ``(8, 3)`` row-wise so the
    integration test's ``num_samples, num_params = param_array.shape``
    unpacking stays valid.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    tar_path = tmp_path / "shard-000000.tar"
    first_record = np.arange(12, dtype=np.float32).reshape(4, 3)
    second_record = np.arange(12, 24, dtype=np.float32).reshape(4, 3)
    _write_wds_tar(
        tar_path,
        [("00000000", first_record), ("00000004", second_record)],
    )

    param_array = _load_param_array_from_wds_tar(tar_path)

    assert param_array.shape == (8, 3)
    assert param_array.dtype == np.float32
    np.testing.assert_array_equal(param_array[:4], first_record)
    np.testing.assert_array_equal(param_array[4:], second_record)


def test_load_param_array_from_wds_tar_single_record_returns_2d(tmp_path: Path) -> None:
    """A shard with one ``(B, P)`` record loads as ``(B, P)`` — not ``(1, B, P)``.

    Pins the actual shape produced by the smoke matrix's wds row, which writes
    ``samples_per_render_batch`` rows per record and the test expects the
    helper to drop the per-record axis.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    tar_path = tmp_path / "shard-000000.tar"
    record = np.full((4, 92), 0.5, dtype=np.float32)
    _write_wds_tar(tar_path, [("00000000", record)])

    param_array = _load_param_array_from_wds_tar(tar_path)

    assert param_array.shape == (4, 92)
    assert param_array.dtype == np.float32
    np.testing.assert_array_equal(param_array, record)


def test_load_param_array_from_wds_tar_preserves_record_order(tmp_path: Path) -> None:
    """Records are concatenated in tar-member name order, not insertion order.

    The writer keys records by ``start_idx``, but tar member order is only
    guaranteed deterministic when the reader sorts. Insert records out of
    order and confirm the reader still returns rows in key order.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    tar_path = tmp_path / "shard-000000.tar"
    later_record = np.full((2, 3), 9.0, dtype=np.float32)
    earlier_record = np.full((2, 3), 1.0, dtype=np.float32)
    _write_wds_tar(
        tar_path,
        [("00000002", later_record), ("00000000", earlier_record)],
    )

    param_array = _load_param_array_from_wds_tar(tar_path)

    np.testing.assert_array_equal(param_array[:2], earlier_record)
    np.testing.assert_array_equal(param_array[2:], later_record)


def test_load_param_array_from_wds_tar_empty_returns_2d_zero(tmp_path: Path) -> None:
    """A shard with no ``.param_array.npy`` members returns ``(0, 0)`` float32.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    tar_path = tmp_path / "shard-empty.tar"
    with tarfile.open(tar_path, mode="w") as tar:
        info = tarfile.TarInfo(name="00000000.audio.npy")
        info.size = 0
        tar.addfile(info, io.BytesIO(b""))

    param_array = _load_param_array_from_wds_tar(tar_path)

    assert param_array.shape == (0, 0)
    assert param_array.dtype == np.float32


def test_load_param_array_from_hdf5_returns_2d_float32(tmp_path: Path) -> None:
    """``param_array`` from an hdf5 split file loads as ``(N, P)`` float32.

    The reader must cast to ``float32`` regardless of the on-disk dtype so the
    integration test's torch tensors stay on the oracle's expected precision.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    h5 = pytest.importorskip("h5py")
    h5_path = tmp_path / "train.h5"
    expected = np.arange(24, dtype=np.float64).reshape(8, 3)
    with h5.File(h5_path, "w") as f:
        f.create_dataset("param_array", data=expected)

    param_array = _load_param_array_from_hdf5(h5_path)

    assert param_array.shape == (8, 3)
    assert param_array.dtype == np.float32
    np.testing.assert_array_equal(param_array, expected.astype(np.float32))
