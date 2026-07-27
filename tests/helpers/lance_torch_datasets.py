"""Deterministic random Lance datasets for the ``lance_torch`` dataloader tests.

Shared by the local e2e suite (``tests/data/test_lance_torch.py``) and the
real-R2 streaming suite (``tests/integration/test_lance_torch_r2.py``) so both
lanes exercise identical schema and values through the real pipeline writer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synth_setter.data.vst.shapes import DATASET_FIELD_DTYPES
from tests.helpers.lance_fixtures import write_lance_shard

ROWS = 32
NUM_PARAMS = 5
FIELD_SHAPES = {
    "audio": (ROWS, 2, 10),
    "mel_spec": (ROWS, 2, 128, 3),
    "param_array": (ROWS, NUM_PARAMS),
}


def write_random_lance_dataset(dest: Path) -> dict[str, np.ndarray]:
    """Write a real Lance dataset of deterministic random rows via the pipeline writer.

    :param dest: Destination ``.lance`` dataset directory.
    :returns: The exact per-field source arrays, for value round-trip asserts.
    """
    rng = np.random.default_rng(seed=7)
    arrays = {
        field: rng.standard_normal(shape).astype(DATASET_FIELD_DTYPES[field])
        for field, shape in FIELD_SHAPES.items()
    }
    write_lance_shard(dest, arrays)
    return arrays
