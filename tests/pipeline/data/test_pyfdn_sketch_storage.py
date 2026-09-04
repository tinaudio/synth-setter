"""Behavioral tests for pyFDN temporal-sketch Lance storage."""

from __future__ import annotations

from typing import cast

import numpy as np
import pyarrow as pa
import pytest

from synth_setter.data.vst.shapes import AUDIO_FIELD
from synth_setter.pipeline.data.add_embeddings import (
    EMBEDDING_REGISTRY,
    _encode_pyfdn_sketch_column,
)
from synth_setter.pipeline.data.lance_shard import pyfdn_sketch_struct_array

PYFDN_SKETCH_FRAMES = 32
PYFDN_EDC_BANDS = 8
PYFDN_SKETCH_CONTROLS = 10


def _controls(rows: int = 2) -> np.ndarray:
    values = np.arange(
        rows * PYFDN_SKETCH_CONTROLS * PYFDN_SKETCH_FRAMES, dtype=np.float32
    )
    return values.reshape(rows, PYFDN_SKETCH_CONTROLS, PYFDN_SKETCH_FRAMES) / values.size


def test_pyfdn_sketch_struct_array_uses_exact_storage_schema() -> None:
    """The persisted struct has only the three fixed-shape float32 children."""
    struct = pyfdn_sketch_struct_array(_controls())

    assert struct.type == pa.struct(
        [
            pa.field(
                "edc",
                pa.fixed_shape_tensor(pa.float32(), [PYFDN_EDC_BANDS, PYFDN_SKETCH_FRAMES]),
            ),
            pa.field("echo_density", pa.list_(pa.float32(), PYFDN_SKETCH_FRAMES)),
            pa.field("spectral_flatness", pa.list_(pa.float32(), PYFDN_SKETCH_FRAMES)),
        ]
    )


def test_pyfdn_sketch_registry_policy_is_checkpoint_free_and_unindexed() -> None:
    """The post-finalize policy reads audio without loading weights or building IVF."""
    spec = EMBEDDING_REGISTRY["pyfdn_sketch"]

    assert spec.column == "pyfdn_sketch"
    assert spec.default_checkpoint == ""
    assert spec.input_fields == (AUDIO_FIELD,)
    assert spec.index is None


def test_pyfdn_sketch_encode_column_builds_exact_struct() -> None:
    """A conformant extractor result is split into the persisted child layout."""
    audio = np.ones((2, 1, 64), dtype=np.float32)
    controls = _controls()

    struct = _encode_pyfdn_sketch_column(
        {AUDIO_FIELD: audio}, 48000, lambda batch, sample_rate: controls
    )

    assert struct.type == pyfdn_sketch_struct_array(controls).type
    edc = cast("pa.FixedShapeTensorArray", struct.field("edc")).to_numpy_ndarray()
    np.testing.assert_array_equal(edc, controls[:, :8])


def test_pyfdn_sketch_encode_column_with_wrong_shape_raises() -> None:
    """The encoder must return one fixed control matrix per source waveform."""
    audio = np.ones((2, 1, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="produced shape"):
        _encode_pyfdn_sketch_column(
            {AUDIO_FIELD: audio},
            48000,
            lambda batch, sample_rate: np.zeros((1, 10, 32), dtype=np.float32),
        )


def test_pyfdn_sketch_encode_column_with_non_finite_child_raises() -> None:
    """A non-finite value in any child fails before the permanent commit."""
    audio = np.ones((2, 1, 64), dtype=np.float32)
    controls = _controls()
    controls[0, 9, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        _encode_pyfdn_sketch_column(
            {AUDIO_FIELD: audio}, 48000, lambda batch, sample_rate: controls
        )


@pytest.mark.parametrize("control_row", [0, 8, 9])
def test_pyfdn_sketch_encode_column_with_out_of_range_child_raises(
    control_row: int,
) -> None:
    """EDC, echo-density, and spectral-flatness rows each enforce unit bounds.

    :param control_row: Representative child row poisoned for this scenario.
    """
    audio = np.ones((2, 1, 64), dtype=np.float32)
    controls = _controls()
    controls[0, control_row, 0] = 1.01

    with pytest.raises(ValueError, match="controls out of bounds"):
        _encode_pyfdn_sketch_column(
            {AUDIO_FIELD: audio}, 48000, lambda batch, sample_rate: controls
        )


def test_pyfdn_sketch_struct_array_preserves_control_values() -> None:
    """Splitting the control stack into children is lossless."""
    controls = _controls()
    struct = pyfdn_sketch_struct_array(controls)

    edc = cast("pa.FixedShapeTensorArray", struct.field("edc")).to_numpy_ndarray()
    echo_density = np.asarray(struct.field("echo_density").flatten()).reshape(2, 32)
    spectral_flatness = np.asarray(struct.field("spectral_flatness").flatten()).reshape(2, 32)

    np.testing.assert_array_equal(edc, controls[:, :8])
    np.testing.assert_array_equal(echo_density, controls[:, 8])
    np.testing.assert_array_equal(spectral_flatness, controls[:, 9])


@pytest.mark.parametrize("shape", [(2, 10, 31), (2, 9, 32)])
def test_pyfdn_sketch_struct_array_with_wrong_shape_raises(shape: tuple[int, ...]) -> None:
    """Malformed control stacks fail before Arrow storage.

    :param shape: Invalid control-stack shape.
    """
    with pytest.raises(ValueError, match="pyFDN sketch controls"):
        pyfdn_sketch_struct_array(np.zeros(shape, dtype=np.float32))
