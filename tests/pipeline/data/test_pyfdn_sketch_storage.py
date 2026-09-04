"""Behavioral tests for pyFDN temporal-sketch Lance storage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.data.vst.shapes import AUDIO_FIELD
from synth_setter.features.pyfdn_controls import extract_reverb_sketch
from synth_setter.pipeline.data.add_embeddings import (
    EMBEDDING_REGISTRY,
    SketchEncodeFn,
    _encode_pyfdn_sketch_column,
    _missing_embedding_specs,
    _write_columns,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig
from synth_setter.pipeline.data.lance_shard import pyfdn_sketch_struct_array

PYFDN_SKETCH_FRAMES = 32
PYFDN_EDC_BANDS = 8
PYFDN_SKETCH_CONTROLS = 10
_REAL_ADD_COLUMNS = lance.LanceDataset.add_columns


def _add_columns_in_process(
    dataset: lance.LanceDataset,
    udf: Any,
    *,
    read_columns: list[str],
    batch_size: int,
) -> None:
    outputs = []
    for batch in dataset.to_batches(columns=read_columns, batch_size=batch_size):
        output = udf(batch)
        outputs.append(pa.record_batch(output.columns, schema=udf.output_schema))
    reader = pa.RecordBatchReader.from_batches(udf.output_schema, outputs)
    _REAL_ADD_COLUMNS(dataset, reader, batch_size=batch_size)


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


@pytest.mark.parametrize("control_row", [0, 8, 9])
def test_pyfdn_sketch_encode_column_with_non_finite_child_raises(
    control_row: int,
) -> None:
    """A non-finite value in any child fails before the permanent commit.

    :param control_row: Representative child row poisoned for this scenario.
    """
    audio = np.ones((2, 1, 64), dtype=np.float32)
    controls = _controls()
    controls[0, control_row, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        _encode_pyfdn_sketch_column(
            {AUDIO_FIELD: audio}, 48000, lambda batch, sample_rate: controls
        )


@pytest.mark.parametrize(
    ("control_row", "value"),
    [(0, -1.01), (0, 1.01), (8, -1.01), (8, 1.01), (9, -1.01), (9, 1.01)],
)
def test_pyfdn_sketch_encode_column_with_out_of_range_child_raises(
    control_row: int, value: float
) -> None:
    """EDC, echo-density, and spectral-flatness rows each enforce unit bounds.

    :param control_row: Representative child row poisoned for this scenario.
    :param value: Out-of-range value written to that child.
    """
    audio = np.ones((2, 1, 64), dtype=np.float32)
    controls = _controls()
    controls[0, control_row, 0] = value

    with pytest.raises(ValueError, match="controls out of bounds"):
        _encode_pyfdn_sketch_column(
            {AUDIO_FIELD: audio}, 48000, lambda batch, sample_rate: controls
        )


def test_pyfdn_sketch_registry_encoder_uses_canonical_real_extractor() -> None:
    """The registry adapter computes each row through the public pyFDN transform."""
    sample_rate = 44_100
    rng = np.random.default_rng(2021)
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    response = rng.standard_normal(sample_rate) * np.exp(-8.0 * time)
    audio = response.astype(np.float32)[None, None]
    spec = EMBEDDING_REGISTRY["pyfdn_sketch"]
    encoder = spec.load_encoder(
        "", AddEmbeddingsConfig(lance_uri="fixture.lance", embeddings=("pyfdn_sketch",))
    )

    actual = cast("SketchEncodeFn", encoder)(audio, sample_rate)

    np.testing.assert_allclose(actual[0], extract_reverb_sketch(audio[0, 0], sample_rate))


def test_pyfdn_sketch_artifact_identity_covers_output_policy() -> None:
    """Persisted identity names DSP, normalization, bins, and package versions."""
    identity = EMBEDDING_REGISTRY["pyfdn_sketch"].resolve_artifact_identity("")

    assert "dsp:octave-edc-abel-huang-density-rfft-flatness-v1" in identity
    assert "normalization:signed-unit-edc-floor-60db-density-rational-flatness-linear" in identity
    assert "temporal:equal-duration-32" in identity
    assert "packages:numpy:" in identity
    assert ",pyfdn:" in identity
    assert ",scipy:" in identity


def test_pyfdn_sketch_config_with_checkpoint_override_raises() -> None:
    """The strict config rejects learned weights for the checkpoint-free transform."""
    with pytest.raises(ValueError, match="checkpoint-free"):
        AddEmbeddingsConfig(
            lance_uri="fixture.lance",
            embeddings=("pyfdn_sketch",),
            checkpoints={"pyfdn_sketch": "weights.ckpt"},
        )


def test_pyfdn_sketch_artifact_identity_with_checkpoint_raises() -> None:
    """A learned-weight override cannot masquerade as the checkpoint-free policy."""
    with pytest.raises(ValueError, match="checkpoint-free"):
        EMBEDDING_REGISTRY["pyfdn_sketch"].resolve_artifact_identity("weights.ckpt")


def test_pyfdn_sketch_write_columns_preserves_rows_and_persists_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry writes one struct per source row through Lance schema evolution.

    :param tmp_path: Parent of the real local Lance dataset.
    :param monkeypatch: Fixture running the Lance UDF synchronously for observability.
    """
    uri = tmp_path / "rows.lance"
    audio = np.arange(3 * 1 * 64, dtype=np.float32).reshape(3, 1, 64) / 192
    lance.write_dataset(
        pa.table(
            {
                AUDIO_FIELD: pa.FixedShapeTensorArray.from_numpy_ndarray(audio),
                "row": pa.array([7, 11, 13], type=pa.int32()),
            }
        ),
        str(uri),
    )
    spec = replace(
        EMBEDDING_REGISTRY["pyfdn_sketch"],
        load_encoder=lambda checkpoint, config: lambda batch, sample_rate: _controls(len(batch)),
        resolve_artifact_identity=lambda checkpoint: "pyfdn-sketch:test-policy-v1",
    )

    monkeypatch.setattr(lance.LanceDataset, "add_columns", _add_columns_in_process)
    dataset = lance.dataset(str(uri))
    config = AddEmbeddingsConfig(
        lance_uri=str(uri),
        embeddings=("pyfdn_sketch",),
        batch_size=2,
        build_index=False,
    )

    _write_columns(dataset, [spec], 48_000, config)

    reread = lance.dataset(str(uri))
    assert reread.count_rows() == 3
    assert reread.to_table(columns=["row"])["row"].to_pylist() == [7, 11, 13]
    field = reread.schema.field("pyfdn_sketch")
    assert field.type == pyfdn_sketch_struct_array(_controls(1)).type
    assert field.metadata[b"synth_setter.embedding.artifact"] == b"pyfdn-sketch:test-policy-v1"


def test_pyfdn_sketch_existing_identity_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted sketch cannot be resumed under a different extraction policy.

    :param tmp_path: Parent of the real local Lance dataset.
    :param monkeypatch: Fixture running the Lance UDF synchronously.
    """
    uri = tmp_path / "identity.lance"
    audio = np.ones((1, 1, 64), dtype=np.float32)
    lance.write_dataset(
        pa.table({AUDIO_FIELD: pa.FixedShapeTensorArray.from_numpy_ndarray(audio)}), str(uri)
    )
    base_spec = EMBEDDING_REGISTRY["pyfdn_sketch"]
    spec = replace(
        base_spec,
        load_encoder=lambda checkpoint, config: lambda batch, sample_rate: _controls(len(batch)),
        resolve_artifact_identity=lambda checkpoint: "pyfdn-sketch:test-policy-v1",
    )

    monkeypatch.setattr(lance.LanceDataset, "add_columns", _add_columns_in_process)
    config = AddEmbeddingsConfig(
        lance_uri=str(uri), embeddings=("pyfdn_sketch",), build_index=False
    )
    dataset = lance.dataset(str(uri))
    _write_columns(dataset, [spec], 48_000, config)
    changed = replace(
        spec,
        resolve_artifact_identity=lambda checkpoint: "pyfdn-sketch:test-policy-v2",
    )

    with pytest.raises(ValueError, match="checkpoint identity"):
        _missing_embedding_specs(dataset, [changed], config)


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
