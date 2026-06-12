"""Local decode-contract tests for the Lance param-array loader in the oracle helper.

Pins values, row order, and the float32 cast so a loader regression fails in
``test-fast`` without R2 creds; the end-to-end path lives in
``test_finalize_artifact_oracle.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synth_setter.data.vst.shapes import PARAM_ARRAY_FIELD
from tests.helpers.lance_fixtures import write_lance_shard
from tests.integration.test_finalize_artifact_oracle import _load_param_array_from_lance


def test_load_param_array_from_lance_preserves_values_and_row_order(tmp_path: Path) -> None:
    """The loader returns ``param_array`` values unchanged and in shard row order.

    :param tmp_path: Dir hosting the written-then-read ``.lance`` shard fixture.
    """
    shard = tmp_path / "train.lance"
    # Distinct per-row values so a row-order or reshape bug cannot pass.
    param_array = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    write_lance_shard(shard, {PARAM_ARRAY_FIELD: param_array})

    loaded = _load_param_array_from_lance(shard)

    np.testing.assert_array_equal(loaded, param_array)
    assert loaded.dtype == np.float32


def test_load_param_array_from_lance_casts_non_float32_column_to_float32(tmp_path: Path) -> None:
    """A wider on-disk dtype is downcast to float32 with values preserved.

    :param tmp_path: Dir hosting the written-then-read ``.lance`` shard fixture.
    """
    shard = tmp_path / "train.lance"
    param_array = np.arange(2 * 3, dtype=np.float64).reshape(2, 3)
    write_lance_shard(shard, {PARAM_ARRAY_FIELD: param_array})

    loaded = _load_param_array_from_lance(shard)

    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, param_array.astype(np.float32))
