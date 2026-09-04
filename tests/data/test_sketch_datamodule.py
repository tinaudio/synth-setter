"""Tests for sketch-control batch preparation and datamodule plumbing."""

from pathlib import Path

import numpy as np
import pytest
import torch

import pyarrow as pa

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    NUM_SKETCH_TRACK_ROWS,
    SKETCH_CENTROID_CHILD,
    SKETCH_LOUDNESS_CHILD,
    SKETCH_PITCH_SLICE,
    SKETCH_STORAGE_FRAMES,
    SKETCH_STRUCT_FIELD,
    SketchControlSpec,
)
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.data.vst_datamodule import RawBatch, prepare_batch
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data.lance_shard import (
    pyfdn_sketch_struct_array,
    sketch_struct_array,
    write_lance_dataset,
)
from tests.data.test_embedding_conditioning import _write_embedding_shard
from tests.helpers.lance_fixtures import (
    make_shard_columns,
    shard_record_batch,
    write_lance_shard_with_sketch,
)

_NUM_FRAMES = 7
_THRESHOLD = 0.1


def _write_sketch_split(path: Path, values: np.ndarray) -> None:
    """Write one split carrying the nested sketch struct column.

    :param path: Destination Lance dataset.
    :param values: ``(rows, NUM_SKETCH_CONTROLS, _NUM_FRAMES)`` stacked controls.
    """
    write_lance_shard_with_sketch(path, make_shard_columns(len(values), seed=9), values)


def _sketch_rows(rows: int, seed: int = 0) -> np.ndarray:
    """Draw stored-layout sketch rows.

    :param rows: Batch size.
    :param seed: RNG seed.
    :returns: ``(rows, NUM_SKETCH_CONTROLS, _NUM_FRAMES)`` float32 controls with
        signed-unit loudness/centroid and unit-interval pitch activations.
    """
    rng = np.random.default_rng(seed)
    values = rng.random((rows, NUM_SKETCH_CONTROLS, _NUM_FRAMES)).astype(np.float32)
    values[:, :NUM_SKETCH_TRACK_ROWS] = values[:, :NUM_SKETCH_TRACK_ROWS] * 2 - 1
    return values


def _prepare_sketch(
    values: np.ndarray, *, threshold: float | None, ot: bool = False
) -> dict[str, torch.Tensor | None]:
    """Run ``prepare_batch`` over a sketch-only raw batch.

    :param values: Stored sketch rows.
    :param threshold: Pitch zero-bin threshold, or ``None`` to skip binning.
    :param ot: Whether to Hungarian-match noise to parameters.
    :returns: Prepared model batch.
    """
    raw: RawBatch = {
        "param_array": np.zeros((values.shape[0], 2), dtype=np.float32),
        "sketch_ctrl": values,
    }
    return prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=True,
        ot=ot,
        generator=torch.Generator().manual_seed(0),
        sketch_pitch_zero_threshold=threshold,
    )


def test_prepare_batch_pitch_cells_below_threshold_zeroed() -> None:
    """Pitch activations under the threshold zero-bin; the rest survive."""
    values = _sketch_rows(rows=4)

    sketch = _prepare_sketch(values, threshold=_THRESHOLD)["sketch_ctrl"]

    assert sketch is not None
    pitch = sketch[:, SKETCH_PITCH_SLICE]
    original_pitch = torch.from_numpy(values[:, SKETCH_PITCH_SLICE])
    below = original_pitch < _THRESHOLD
    assert below.any(), "fixture must exercise the zero-bin branch"
    assert (pitch[below] == 0.0).all()
    torch.testing.assert_close(pitch[~below], original_pitch[~below])


def test_prepare_batch_loudness_centroid_rows_bit_identical_under_binning() -> None:
    """Zero-binning never touches the loudness and centroid rows."""
    values = _sketch_rows(rows=4)

    sketch = _prepare_sketch(values, threshold=_THRESHOLD)["sketch_ctrl"]

    assert sketch is not None
    stored_tracks = torch.from_numpy(values[:, :NUM_SKETCH_TRACK_ROWS])
    assert torch.equal(sketch[:, :NUM_SKETCH_TRACK_ROWS], stored_tracks)


def test_prepare_batch_without_threshold_passes_sketch_unchanged() -> None:
    """A ``None`` threshold forwards the stored controls bit-identically."""
    values = _sketch_rows(rows=4)

    sketch = _prepare_sketch(values, threshold=None)["sketch_ctrl"]

    assert sketch is not None
    assert torch.equal(sketch, torch.from_numpy(values))


def test_prepare_batch_ot_keeps_sketch_aligned_with_params() -> None:
    """OT row permutation moves sketch rows together with their parameters."""
    # Row markers stay inside the stored sketch contract so the batch-boundary
    # range check does not reject them before the permutation runs.
    row_ids = np.arange(6, dtype=np.float32) / 8
    raw: RawBatch = {
        "param_array": np.stack([row_ids, row_ids], axis=1),
        "sketch_ctrl": np.broadcast_to(
            row_ids[:, None, None], (6, NUM_SKETCH_CONTROLS, 4)
        ).copy(),
    }

    batch = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=False,
        ot=True,
        generator=torch.Generator().manual_seed(3),
    )

    sketch = batch["sketch_ctrl"]
    params = batch["params"]
    assert sketch is not None
    assert params is not None
    torch.testing.assert_close(sketch[:, 0, 0], params[:, 0])


def _sketch_module(root: Path, *, fake: bool) -> LanceVSTDataModule:
    """Build a sketch-configured datamodule over mel conditioning.

    :param root: Dataset root, or an unused path in fake mode.
    :param fake: Whether to synthesize batches.
    :returns: Unset-up datamodule.
    """
    return LanceVSTDataModule(
        dataset_root=root,
        batch_size=2,
        sketch=SketchControlSpec(num_frames=_NUM_FRAMES),
        fake=fake,
        use_saved_mean_and_variance=False,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("surge_xt"),
    )


def test_sketch_spec_adds_struct_column_to_loader_projection(tmp_path: Path) -> None:
    """A configured sketch spec projects its stored struct column for every split.

    :param tmp_path: Per-test dataset root.
    """
    module = _sketch_module(tmp_path, fake=True)

    assert SKETCH_STRUCT_FIELD in module._loader_columns(read_audio=False)  # noqa: SLF001


def test_fake_mode_with_sketch_spec_yields_sketch_batch_key(tmp_path: Path) -> None:
    """Fake batches expose sketch controls with the stored layout.

    :param tmp_path: Per-test dataset root.
    """
    module = _sketch_module(tmp_path, fake=True)

    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    sketch = batch["sketch_ctrl"]
    assert sketch is not None
    assert sketch.shape == (2, NUM_SKETCH_CONTROLS, _NUM_FRAMES)
    assert sketch.dtype == torch.float32


def test_real_lance_split_with_sketch_struct_yields_float32_batch(
    tmp_path: Path,
) -> None:
    """A stored sketch struct reaches the model batch as float32 ``sketch_ctrl``.

    :param tmp_path: Per-test dataset root.
    """
    values = _sketch_rows(rows=4, seed=1)
    _write_sketch_split(tmp_path / "val.lance", values)
    module = _sketch_module(tmp_path, fake=False)

    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    sketch = batch["sketch_ctrl"]
    assert sketch is not None
    assert sketch.shape == (2, NUM_SKETCH_CONTROLS, _NUM_FRAMES)
    assert sketch.dtype == torch.float32
    pitch = sketch[:, SKETCH_PITCH_SLICE]
    assert (pitch[torch.from_numpy(values[:2, SKETCH_PITCH_SLICE]) < _THRESHOLD] == 0.0).all()


def test_real_lance_split_reassembles_struct_bit_identical_to_stack(
    tmp_path: Path,
) -> None:
    """Struct-child reassembly restores the stacked control tensor bit-for-bit.

    Pitch cells sit above the zero-bin threshold so binning is a no-op and the batch must equal the
    pre-split stack exactly.

    :param tmp_path: Per-test dataset root.
    """
    values = _sketch_rows(rows=4, seed=2)
    # Keep pitch above the zero-bin threshold so binning is a no-op.
    values[:, SKETCH_PITCH_SLICE] = values[:, SKETCH_PITCH_SLICE] / 2 + 0.5
    _write_sketch_split(tmp_path / "val.lance", values)
    module = _sketch_module(tmp_path, fake=False)

    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    sketch = batch["sketch_ctrl"]
    assert sketch is not None
    assert torch.equal(sketch, torch.from_numpy(values[:2]))


def test_real_lance_split_missing_sketch_column_raises(tmp_path: Path) -> None:
    """A configured sketch spec fails loudly when the stored column is absent.

    :param tmp_path: Per-test dataset root.
    """
    _write_embedding_shard(
        tmp_path / "val.lance",
        column="unrelated",
        values=np.zeros((4, 3), dtype=np.float32),
    )
    module = _sketch_module(tmp_path, fake=False)

    with pytest.raises(KeyError, match=SKETCH_STRUCT_FIELD):
        module.setup("validate")


def _write_raw_struct_split(path: Path, struct: pa.StructArray) -> None:
    """Write one split carrying an arbitrary struct as the sketch column.

    :param path: Destination Lance dataset.
    :param struct: Struct column values, valid or deliberately malformed.
    """
    batch = shard_record_batch(make_shard_columns(len(struct), seed=9))
    extended = batch.append_column(
        pa.field(SKETCH_STRUCT_FIELD, struct.type, nullable=False), struct
    )
    write_lance_dataset(path, extended.schema, [extended])


def test_real_lance_split_missing_struct_child_raises(tmp_path: Path) -> None:
    """A struct lacking a required child fails validation by child name.

    :param tmp_path: Per-test dataset root.
    """
    full = sketch_struct_array(_sketch_rows(rows=4, seed=4))
    kept = [field.name for field in full.type if field.name != SKETCH_CENTROID_CHILD]
    struct = pa.StructArray.from_arrays([full.field(name) for name in kept], names=kept)
    _write_raw_struct_split(tmp_path / "val.lance", struct)
    module = _sketch_module(tmp_path, fake=False)

    with pytest.raises(ValueError, match=f"missing struct child '{SKETCH_CENTROID_CHILD}'"):
        module.setup("validate")


def test_real_lance_split_mis_shaped_struct_child_raises(tmp_path: Path) -> None:
    """A child off the configured frame grid fails validation with both shapes.

    :param tmp_path: Per-test dataset root.
    """
    full = sketch_struct_array(_sketch_rows(rows=4, seed=5))
    wrong_frames = _NUM_FRAMES + 2
    bad_loudness = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(4 * wrong_frames, dtype=np.float32)), wrong_frames
    )
    names = [field.name for field in full.type]
    struct = pa.StructArray.from_arrays(
        [
            bad_loudness if name == SKETCH_LOUDNESS_CHILD else full.field(name)
            for name in names
        ],
        names=names,
    )
    _write_raw_struct_split(tmp_path / "val.lance", struct)
    module = _sketch_module(tmp_path, fake=False)

    with pytest.raises(ValueError, match=f"'{SKETCH_LOUDNESS_CHILD}'.*has shape"):
        module.setup("validate")


def test_real_lance_split_nonfinite_struct_child_raises(tmp_path: Path) -> None:
    """A non-finite stored control value fails validation by child name.

    :param tmp_path: Per-test dataset root.
    """
    values = _sketch_rows(rows=4, seed=6)
    values[0, 0, 0] = np.nan
    _write_raw_struct_split(tmp_path / "val.lance", sketch_struct_array(values))
    module = _sketch_module(tmp_path, fake=False)

    with pytest.raises(ValueError, match=f"'{SKETCH_LOUDNESS_CHILD}'.*non-finite"):
        module.setup("validate")


def _write_pyfdn_sketch_split(path: Path, values: np.ndarray) -> None:
    """Write one split carrying the pyFDN reverb-sketch struct.

    :param path: Destination Lance dataset.
    :param values: Signed-unit ``(rows, 10, 32)`` controls.
    """
    struct = pyfdn_sketch_struct_array(values)
    batch = shard_record_batch(make_shard_columns(len(values), seed=10))
    extended = batch.append_column(
        pa.field("pyfdn_sketch", struct.type, nullable=False), struct
    )
    write_lance_dataset(path, extended.schema, [extended])


def _pyfdn_sketch_module(root: Path, *, fake: bool) -> LanceVSTDataModule:
    """Build a pyFDN sketch-profile datamodule.

    :param root: Dataset root, or an unused path in fake mode.
    :param fake: Whether to synthesize batches.
    :returns: Unset-up datamodule.
    """
    return LanceVSTDataModule(
        dataset_root=root,
        batch_size=2,
        sketch=SketchControlSpec(
            column="pyfdn_sketch",
            profile="pyfdn_reverb",
            num_frames=SKETCH_STORAGE_FRAMES,
        ),
        fake=fake,
        use_saved_mean_and_variance=False,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("pyfdn_n8_mono_householder"),
    )


def test_real_pyfdn_sketch_profile_reassembles_float32_controls(
    tmp_path: Path,
) -> None:
    """The three stored children reassemble as model-ready ``float32[B,10,32]``.

    :param tmp_path: Per-test dataset root.
    """
    values = np.linspace(-0.9, 0.9, 4 * 10 * 32, dtype=np.float32).reshape(4, 10, 32)
    _write_pyfdn_sketch_split(tmp_path / "val.lance", values)
    module = _pyfdn_sketch_module(tmp_path, fake=False)

    module.setup("validate")
    try:
        sketch = next(iter(module.val_dataloader()))["sketch_ctrl"]
    finally:
        module.teardown()

    assert sketch is not None
    assert sketch.dtype == torch.float32
    assert sketch.shape == (2, 10, SKETCH_STORAGE_FRAMES)
    assert torch.equal(sketch, torch.from_numpy(values[:2]))


def test_real_pyfdn_sketch_profile_does_not_apply_music_pitch_threshold(
    tmp_path: Path,
) -> None:
    """Low signed reverb coordinates survive the music-only pitch zero-bin.

    :param tmp_path: Per-test dataset root.
    """
    values = np.full((4, 10, SKETCH_STORAGE_FRAMES), 0.05, dtype=np.float32)
    values[:, 2:] *= -1
    _write_pyfdn_sketch_split(tmp_path / "val.lance", values)
    module = _pyfdn_sketch_module(tmp_path, fake=False)

    module.setup("validate")
    try:
        sketch = next(iter(module.val_dataloader()))["sketch_ctrl"]
    finally:
        module.teardown()

    assert sketch is not None
    assert torch.equal(sketch, torch.from_numpy(values[:2]))


def test_fake_pyfdn_sketch_profile_uses_reverb_layout(tmp_path: Path) -> None:
    """Synthetic batches honor the selected profile's channel count and range.

    :param tmp_path: Per-test dataset root.
    """
    module = _pyfdn_sketch_module(tmp_path, fake=True)

    module.setup("validate")
    try:
        sketch = next(iter(module.val_dataloader()))["sketch_ctrl"]
    finally:
        module.teardown()

    assert sketch is not None
    assert sketch.shape == (2, 10, SKETCH_STORAGE_FRAMES)
    assert torch.all((sketch >= -1) & (sketch <= 1))


def test_real_pyfdn_sketch_profile_with_mis_shaped_child_raises(
    tmp_path: Path,
) -> None:
    """Every reverb child must match its profile-specific fixed shape.

    :param tmp_path: Per-test dataset root.
    """
    values = np.zeros((4, 10, SKETCH_STORAGE_FRAMES), dtype=np.float32)
    full = pyfdn_sketch_struct_array(values)
    wrong = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(4 * 31, dtype=np.float32)), 31
    )
    struct = pa.StructArray.from_arrays(
        [full.field("edc"), wrong, full.field("spectral_flatness")],
        names=["edc", "echo_density", "spectral_flatness"],
    )
    batch = shard_record_batch(make_shard_columns(4, seed=10))
    extended = batch.append_column(
        pa.field("pyfdn_sketch", struct.type, nullable=False), struct
    )
    write_lance_dataset(tmp_path / "val.lance", extended.schema, [extended])
    module = _pyfdn_sketch_module(tmp_path, fake=False)

    with pytest.raises(ValueError, match="echo_density.*has shape"):
        module.setup("validate")


def test_real_lance_split_with_legacy_flat_sketch_column_raises(tmp_path: Path) -> None:
    """A flat tensor in the configured column fails with a rewrite instruction.

    :param tmp_path: Per-test dataset root.
    """
    _write_embedding_shard(
        tmp_path / "val.lance",
        column=SKETCH_STRUCT_FIELD,
        values=_sketch_rows(rows=4, seed=3),
    )
    module = _sketch_module(tmp_path, fake=False)

    with pytest.raises(ValueError, match="non-struct type.*flat layout"):
        module.setup("validate")
