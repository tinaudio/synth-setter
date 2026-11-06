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
from synth_setter.features import sketch_controls
from synth_setter.features.sketch_controls import (
    DEFAULT_PESTO_CHECKPOINT,
    load_pesto_model,
)
from synth_setter.data.vst_datamodule import RawBatch, prepare_batch
from synth_setter.models.components.sketch_tokens import SketchControlTokens
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
    :param values: ``(rows, NUM_SKETCH_CONTROLS, frames)`` stacked controls.
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
        sketch=SketchControlSpec(
            num_frames=_NUM_FRAMES,
            num_control_tokens=_NUM_FRAMES,
        ),
        fake=fake,
        use_saved_mean_and_variance=False,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("surge_xt"),
    )


def test_stored_sketch_projection_does_not_request_training_audio(tmp_path: Path) -> None:
    """Stored controls do not hydrate waveform audio solely for sketch extraction.

    :param tmp_path: Per-test dataset root.
    """
    module = _sketch_module(tmp_path, fake=True)

    assert module.projection["train"] == ["param_array", "mel_spec", "sketch"]
    assert module.projection["val"] == ["param_array", "mel_spec", "sketch"]


def test_sketch_spec_adds_struct_column_to_loader_projection(tmp_path: Path) -> None:
    """A configured sketch spec projects its stored struct column for every split.

    :param tmp_path: Per-test dataset root.
    """
    module = _sketch_module(tmp_path, fake=True)

    assert SKETCH_STRUCT_FIELD in module._loader_columns(read_audio=False)  # noqa: SLF001


def test_stored_sketch_setup_does_not_load_pesto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stored controls remain independent of the online extractor lifecycle.

    :param tmp_path: Per-test dataset root.
    :param monkeypatch: Fails if setup attempts to load PESTO.
    """
    monkeypatch.setattr(
        "synth_setter.data.lance_datamodule.load_pesto_model",
        lambda *args, **kwargs: pytest.fail(f"unexpected PESTO load {args} {kwargs}"),
    )
    module = _sketch_module(tmp_path, fake=True)

    module.setup("validate")


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


def test_dense_lance_controls_reach_model_token_grid(tmp_path: Path) -> None:
    """Dense stored controls pass Lance validation and model tokenization.

    :param tmp_path: Per-test dataset root.
    """
    values = np.zeros((4, NUM_SKETCH_CONTROLS, 96), dtype=np.float32)
    values[:, 0] = 0.25
    values[:, SKETCH_PITCH_SLICE] = 0.05
    values[:, SKETCH_PITCH_SLICE.start] = 0.2
    source_values = values.copy()
    _write_sketch_split(tmp_path / "val.lance", values)
    spec = SketchControlSpec(num_frames=96, num_control_tokens=64)
    module = LanceVSTDataModule(
        dataset_root=tmp_path,
        batch_size=2,
        sketch=spec,
        fake=False,
        use_saved_mean_and_variance=False,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("surge_xt"),
    )

    module.setup("validate")
    try:
        controls = next(iter(module.val_dataloader()))["sketch_ctrl"]
    finally:
        module.teardown()
    assert controls is not None
    loaded_controls = controls.clone()
    tokenizer = SketchControlTokens(d_model=1, num_control_tokens=64)
    loudness_projection = tokenizer.projections["loudness"]
    pitch_projection = tokenizer.projections["pitch"]
    assert isinstance(loudness_projection, torch.nn.Linear)
    assert isinstance(pitch_projection, torch.nn.Linear)
    with torch.no_grad():
        loudness_projection.weight.fill_(1.0)
        pitch_projection.weight.fill_(1.0)
    keep = torch.ones(controls.shape[0], 3, dtype=torch.bool)

    contribution = tokenizer(controls, keep) - tokenizer.unconditional(controls.shape[0])

    torch.testing.assert_close(
        contribution[:, :, 0],
        torch.full((2, 64), 0.45),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(controls, loaded_controls, rtol=0, atol=0)
    np.testing.assert_array_equal(values, source_values)


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


def _online_music_module(root: Path, *, sample_rate: int = 44_100) -> LanceVSTDataModule:
    """Build an online music-sketch datamodule.

    :param root: Dataset root, unused in fake mode.
    :param sample_rate: Module waveform rate.
    :returns: Online music-sketch datamodule.
    """
    return LanceVSTDataModule(
        dataset_root=root,
        batch_size=2,
        sketch=SketchControlSpec(
            profile="music",
            source="online",
            sample_rate=44_100,
            num_frames=32,
            num_control_tokens=32,
        ),
        fake=True,
        use_saved_mean_and_variance=False,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("surge_xt"),
        sample_rate=sample_rate,
    )


def test_online_music_sample_rate_mismatch_rejected(tmp_path: Path) -> None:
    """Online extraction and dataset waveform rates must agree exactly.

    :param tmp_path: Per-test dataset root.
    """
    with pytest.raises(ValueError, match="sketch sample_rate.*datamodule sample_rate"):
        _online_music_module(tmp_path, sample_rate=48_000)


def test_online_music_projects_audio_without_stored_sketch(tmp_path: Path) -> None:
    """Online music controls hydrate audio instead of a sketch column.

    :param tmp_path: Per-test dataset root.
    """
    module = _online_music_module(tmp_path)

    assert module.projection["train"] == ["param_array", "mel_spec", "audio"]
    assert SKETCH_STRUCT_FIELD not in module.projection["train"]


def test_online_music_setup_selects_canonical_pesto_after_custom_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup resets a process cache previously loaded with noncanonical weights.

    :param tmp_path: Per-test dataset root.
    :param monkeypatch: Records checkpoint selections without loading external weights.
    """
    import pesto

    calls: list[str] = []

    def record(checkpoint: str, *, step_size: float) -> torch.nn.Module:
        del step_size
        calls.append(checkpoint)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(sketch_controls, "_pesto_model", None)
    monkeypatch.setattr(sketch_controls, "_pesto_checkpoint", None)
    monkeypatch.setattr(pesto, "load_model", record)
    load_pesto_model("custom-checkpoint", device="cpu")
    module = _online_music_module(tmp_path)

    module.setup("validate")

    assert calls == ["custom-checkpoint", DEFAULT_PESTO_CHECKPOINT]


class _ResidentModel(torch.nn.Linear):
    """Fake cached extractor that records every device move."""

    def __init__(self) -> None:
        """Build a one-weight stand-in for the PESTO network."""
        super().__init__(1, 1)
        self.moves: list[str] = []

    def to(self, *args: object, **kwargs: object) -> "_ResidentModel":  # type: ignore[override]
        """Record the requested destination instead of moving anything.

        :param *args: Positional ``nn.Module.to`` arguments.
        :param **kwargs: Keyword ``nn.Module.to`` arguments.
        :returns: This module, unchanged.
        """
        self.moves.append(f"{args}{kwargs}")
        return self


def test_online_music_setup_keeps_resident_canonical_pesto_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup warms the canonical cache without relocating an already-resident model.

    :param tmp_path: Per-test dataset root.
    :param monkeypatch: Installs a canonical cached stand-in that records moves.
    """
    resident = _ResidentModel()
    monkeypatch.setattr(sketch_controls, "_pesto_model", resident)
    monkeypatch.setattr(sketch_controls, "_pesto_checkpoint", DEFAULT_PESTO_CHECKPOINT)
    module = _online_music_module(tmp_path)

    module.setup("validate")
    module.setup("fit")

    assert load_pesto_model() is resident
    assert resident.moves == []


def _sine_batch(*frequencies_hz: float, samples: int = 44_100) -> torch.Tensor:
    """Build one-second mono sinusoids at 44.1 kHz, one row per frequency.

    :param *frequencies_hz: Row frequencies.
    :param samples: Samples per row; one second keeps every 32-frame grid satisfiable.
    :returns: ``(rows, 1, samples)`` float32 waveform batch.
    """
    time = torch.arange(samples, dtype=torch.float32) / 44_100
    return torch.stack(
        [torch.sin(2.0 * torch.pi * frequency * time) for frequency in frequencies_hz]
    )[:, None]


def test_online_music_after_transfer_emits_checkpoint_grid_controls(tmp_path: Path) -> None:
    """The production hook yields thresholded controls that track the audio content.

    :param tmp_path: Per-test dataset root.
    """
    module = _online_music_module(tmp_path)
    audio = _sine_batch(440.0, 880.0)

    controls = module.on_after_batch_transfer({"audio": audio}, 0)["sketch_ctrl"]

    assert controls is not None
    assert controls.shape == (2, NUM_SKETCH_CONTROLS, 32)
    assert controls.dtype == torch.float32
    assert controls.device == audio.device
    tracks = controls[:, : SKETCH_PITCH_SLICE.start]
    assert ((tracks >= -1.0) & (tracks <= 1.0)).all()
    pitch = controls[:, SKETCH_PITCH_SLICE]
    assert ((pitch == 0.0) | ((pitch >= _THRESHOLD) & (pitch <= 1.0))).all()
    # One octave is 12 semitones at three PESTO bins per semitone.
    peak_bins = pitch[:, :, 4:-4].argmax(dim=1).float().mean(dim=1)
    assert peak_bins[1] - peak_bins[0] == pytest.approx(36.0, abs=1.5)
    assert controls[1, 1].mean() > controls[0, 1].mean()


def test_online_music_controls_support_trainable_projection_backward(tmp_path: Path) -> None:
    """No-grad extraction still permits weight gradients in downstream projections.

    :param tmp_path: Per-test dataset root.
    """
    module = _online_music_module(tmp_path)
    audio = _sine_batch(440.0)
    controls = module.on_after_batch_transfer({"audio": audio}, 0)["sketch_ctrl"]
    assert controls is not None
    projection = torch.nn.Linear(NUM_SKETCH_CONTROLS, 1)

    loss = projection(controls.transpose(1, 2)).square().mean()
    loss.backward()

    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_online_music_after_transfer_keeps_controls_on_cuda(tmp_path: Path) -> None:
    """Online extraction follows transferred audio onto CUDA end to end.

    :param tmp_path: Per-test dataset root.
    """
    module = _online_music_module(tmp_path)
    audio = _sine_batch(440.0).to("cuda")

    controls = module.on_after_batch_transfer({"audio": audio}, 0)["sketch_ctrl"]

    assert controls is not None
    assert controls.device.type == "cuda"
    assert controls.dtype == torch.float32
    assert torch.isfinite(controls).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_online_music_validation_then_training_reuses_canonical_cache_on_cuda(
    tmp_path: Path,
) -> None:
    """Validation and training setup preserve deterministic CUDA extraction.

    :param tmp_path: Per-test dataset root.
    """
    module = _online_music_module(tmp_path)
    audio = _sine_batch(440.0).to("cuda")
    module.setup("validate")
    validation = module.on_after_batch_transfer({"audio": audio}, 0)["sketch_ctrl"]

    module.setup("fit")
    training = module.on_after_batch_transfer({"audio": audio}, 0)["sketch_ctrl"]

    assert validation is not None
    assert training is not None
    assert validation.device.type == "cuda"
    assert next(load_pesto_model().parameters()).device.type == "cuda"
    assert torch.equal(training, validation)


def test_tiv_online_projects_audio_without_stored_sketch(tmp_path: Path) -> None:
    """Online TIV reads waveform audio and never requests a sketch column.

    :param tmp_path: Per-test dataset root.
    """
    module = LanceVSTDataModule(
        dataset_root=tmp_path,
        batch_size=2,
        sketch=SketchControlSpec(
            profile="tiv", source="online", sample_rate=44_100, num_frames=32
        ),
        fake=True,
        use_saved_mean_and_variance=False,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("surge_xt"),
    )

    assert module.projection["train"] == ["param_array", "mel_spec", "audio"]
    assert module.projection["val"] == ["param_array", "mel_spec", "audio"]
    assert module.projection["test"] == ["param_array", "mel_spec", "audio"]


def test_tiv_online_after_transfer_extracts_finite_controls(tmp_path: Path) -> None:
    """Transferred waveform batches gain model-ready online TIV controls.

    :param tmp_path: Per-test dataset root.
    """
    module = LanceVSTDataModule(
        dataset_root=tmp_path,
        batch_size=2,
        sketch=SketchControlSpec(
            profile="tiv", source="online", sample_rate=44_100, num_frames=32
        ),
        fake=True,
        use_saved_mean_and_variance=False,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("surge_xt"),
    )
    batch = {
        "audio": torch.zeros(2, 2, 4096),
        "params": torch.zeros(2, 2),
        "noise": torch.zeros(2, 2),
    }

    prepared = module.on_after_batch_transfer(batch, 0)

    controls = prepared["sketch_ctrl"]
    assert controls is not None
    assert controls.shape == (2, 12, 32)
    assert torch.isfinite(controls).all()
    assert torch.count_nonzero(controls) == 0


def test_tiv_cpu_after_transfer_preserves_reference_a4_coordinates(tmp_path: Path) -> None:
    """The CPU datamodule route uses peak-based HPCP rather than STFT-bin chroma.

    :param tmp_path: Per-test dataset root.
    """
    pytest.importorskip("essentia")
    module = LanceVSTDataModule(
        dataset_root=tmp_path,
        batch_size=1,
        sketch=SketchControlSpec(
            profile="tiv", source="online", sample_rate=44_100,
            tiv_backend="essentia", num_frames=32,
        ),
        fake=True,
        use_saved_mean_and_variance=False,
        num_workers=0,
        param_spec_name=ParamSpecName("surge_xt"),
    )
    samples = torch.arange(4096, dtype=torch.float32)
    audio = torch.sin(2.0 * torch.pi * 440.0 * samples / 44_100)[None, None]

    prepared = module.on_after_batch_transfer({"audio": audio}, 0)

    controls = prepared["sketch_ctrl"]
    assert controls is not None
    expected = torch.tensor(
        [0.0, 3.0, -8.0, 0.0, 0.0, -11.5, 15.0, 0.0, 0.0, 14.5, -7.5, 0.0]
    )
    torch.testing.assert_close(controls[0, :, 16], expected)


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
