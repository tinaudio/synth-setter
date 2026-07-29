"""Focused contracts for online TorchSynth sampling and rendering."""

import dataclasses
import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from synth_setter.data.torchsynth_datamodule import (
    _PARAM_CLAMP_EPS,
    NUM_PARAMS,
    PARAM_SPEC,
    TorchSynthBatch,
    TorchSynthDataModule,
    TorchSynthDataset,
    _make_renderer,
    _verify_voice_matches_spec,
    render_torchsynth,
)
from synth_setter.data.sample_seed import derive_sample_seed
from synth_setter.data.vst.param_spec import (
    DiscreteLiteralParameter,
    NoteDurationParameter,
    NoteParams,
)
from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_FULL_PARAM_SPEC
from tests.helpers.run_if import RunIf

_RENDER_KWARGS = {"sample_rate": 44_100, "signal_length": 4_410}
_ENCODED_WIDTH = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
_SYNTH_COLUMNS = TORCHSYNTH_FULL_PARAM_SPEC.synth_columns
_PITCH_PARAM = next(
    param
    for param in TORCHSYNTH_FULL_PARAM_SPEC.note_params
    if isinstance(param, DiscreteLiteralParameter)
)
_NOTE_WINDOW_PARAM = next(
    param
    for param in TORCHSYNTH_FULL_PARAM_SPEC.note_params
    if isinstance(param, NoteDurationParameter)
)
_BUFFER_SECONDS = _RENDER_KWARGS["signal_length"] / _RENDER_KWARGS["sample_rate"]


# The live-voice drift test lives with the pinned spec in
# tests/data/vst/test_torchsynth_param_spec.py::test_pinned_spec_matches_live_voice.
def test_verify_voice_against_perturbed_spec_raises_naming_param() -> None:
    """Verification against a spec with one drifted range fails and names the parameter."""
    voice = _make_renderer(_RENDER_KWARGS["sample_rate"], _RENDER_KWARGS["signal_length"]).voice
    perturbed = (dataclasses.replace(PARAM_SPEC[0], maximum=99.0), *PARAM_SPEC[1:])
    with pytest.raises(ValueError, match="adsr_1"):
        _verify_voice_matches_spec(voice, spec=perturbed)


def test_dataset_same_index_deterministic_different_index_distinct() -> None:
    """Repeated reads are stable while adjacent rows remain distinct."""
    dataset = TorchSynthDataset(2, 123, **_RENDER_KWARGS)
    audio_a, params_a, _ = dataset[0]
    audio_b, params_b, _ = dataset[0]
    _, params_c, _ = dataset[1]

    assert torch.equal(params_a, params_b)
    assert torch.equal(audio_a, audio_b)
    assert not torch.equal(params_a, params_c)


def test_dataset_affine_collision_pair_produces_distinct_params() -> None:
    """Keep crafted base/index collision pairs in distinct parameter streams."""
    first_params = TorchSynthDataset(1, 0, **_RENDER_KWARGS)[0][1]
    second_params = TorchSynthDataset(
        2, 7_682_673_210_995_763_517, **_RENDER_KWARGS
    )[1][1]

    assert not torch.equal(first_params, second_params)


def test_dataset_item_has_normalized_float32_params_and_audio() -> None:
    """Online rows expose renderable labels and finite normalized audio."""
    dataset = TorchSynthDataset(1, 123, **_RENDER_KWARGS)
    audio, params, render_fn = dataset[0]

    assert params.dtype == audio.dtype == torch.float32
    assert torch.all((0 <= params) & (params <= 1))
    assert torch.all((-1 <= audio) & (audio <= 1))
    assert torch.equal(render_fn(params), audio)


def test_dataset_row_width_matches_spec_encoded_width() -> None:
    """Online rows carry the registry spec's full encoded width, not a synth-only subset."""
    _, params, _ = TorchSynthDataset(1, 123, **_RENDER_KWARGS)[0]

    assert params.shape == (1, TORCHSYNTH_FULL_PARAM_SPEC.encoded_width)


def _decoded_note_params(dataset: TorchSynthDataset, index: int) -> NoteParams:
    """Decode one dataset row's note columns.

    :param dataset: Online dataset under test.
    :param index: Logical row index to read.
    :returns: The row's decoded pitch and note window.
    """
    _, params, _ = dataset[index]
    return TORCHSYNTH_FULL_PARAM_SPEC.decode(params[0].numpy())[1]


def test_dataset_note_columns_vary_across_rows_within_the_spec_ranges() -> None:
    """Pitch and the note window are drawn per row and stay inside the spec's declared ranges."""
    dataset = TorchSynthDataset(8, 123, **_RENDER_KWARGS)
    decoded = [_decoded_note_params(dataset, index) for index in range(len(dataset))]

    assert len({note["pitch"] for note in decoded}) > 1
    assert len({note["note_start_and_end"] for note in decoded}) > 1
    for note in decoded:
        assert _PITCH_PARAM.min <= note["pitch"] <= _PITCH_PARAM.max
        start, end = note["note_start_and_end"]
        assert 0.0 <= start <= end <= _NOTE_WINDOW_PARAM.max_note_duration_seconds


def test_dataset_row_matches_spec_sample_for_the_row_seed() -> None:
    """Synth and note columns come from the spec's own sampler seeded by the row's derived seed."""
    _, params, _ = TorchSynthDataset(4, 123, **_RENDER_KWARGS)[2]

    synth_values, note_params = TORCHSYNTH_FULL_PARAM_SPEC.decode(params[0].numpy())
    expected_synth, expected_note = TORCHSYNTH_FULL_PARAM_SPEC.sample(
        np.random.default_rng(derive_sample_seed(123, 2))
    )

    assert synth_values == pytest.approx(expected_synth)
    assert note_params["pitch"] == expected_note["pitch"]
    assert note_params["note_start_and_end"] == pytest.approx(
        expected_note["note_start_and_end"], abs=1e-5
    )


def test_datamodule_split_seeds_produce_distinct_parameters_across_indices() -> None:
    """Keep train, validation, and test parameter streams disjoint across several indices."""
    datamodule = TorchSynthDataModule(
        sample_rate=44_100,
        signal_length=4_410,
        train_val_test_sizes=(2, 2, 2),
        num_workers=0,
    )
    datamodule.setup(None)
    rows = []
    for split in (datamodule.train, datamodule.val, datamodule.test):
        for index in range(2):
            rows.append(tuple(split[index][1].flatten().tolist()))
    assert len(set(rows)) == len(rows)


def test_datamodule_setup_num_params_mismatch_raises() -> None:
    """A configured ``num_params`` disagreeing with the row spec fails fast in ``setup``."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        num_params=1,
        train_val_test_sizes=(1, 1, 1),
        num_workers=0,
    )
    width = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
    with pytest.raises(
        ValueError, match=rf"Configured num_params=1, torchsynth_full encodes {width}"
    ):
        datamodule.setup(None)


def test_datamodule_default_num_params_matches_spec_encoded_width() -> None:
    """The datamodule's default width is the spec's, so model configs need no literal."""
    assert (
        TorchSynthDataModule().num_params == TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
    )


def test_datamodule_test_dataloader_yields_finite_batch() -> None:
    """``setup('test')`` builds the test split and ``test_dataloader`` yields a finite batch."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(1, 1, 2),
        batch_size=2,
        num_workers=0,
    )
    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))
    audio, params = batch["audio"], batch["params"]
    assert audio.shape[0] == params.shape[0] == 2
    assert params.shape[1] == datamodule.num_params
    assert torch.isfinite(audio).all()


def test_datamodule_validate_stage_builds_only_validation_split() -> None:
    """``setup('validate')`` creates the validation dataset without other splits."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(1, 1, 1),
        num_workers=0,
    )
    datamodule.setup("validate")

    assert hasattr(datamodule, "val")
    assert not hasattr(datamodule, "train")
    assert not hasattr(datamodule, "test")
    audio = next(iter(datamodule.val_dataloader()))["audio"]
    assert torch.isfinite(audio).all()


def test_datamodule_loaders_shuffle_only_training_rows() -> None:
    """Training shuffles logical indices; validation and test retain a fixed order."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(1, 1, 1),
        num_workers=0,
    )
    datamodule.setup(None)

    assert isinstance(datamodule.train_dataloader().sampler, RandomSampler)
    assert isinstance(datamodule.val_dataloader().sampler, SequentialSampler)
    assert isinstance(datamodule.test_dataloader().sampler, SequentialSampler)


def test_datamodule_default_retains_train_remainder() -> None:
    """The default loader keeps a trailing partial training batch."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(3, 1, 1),
        batch_size=2,
        num_workers=0,
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    assert loader.drop_last is False
    assert [len(batch["audio"]) for batch in loader] == [2, 1]


def test_datamodule_drop_last_discards_train_remainder() -> None:
    """An opted-in loader discards a trailing partial training batch."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(3, 1, 1),
        batch_size=2,
        num_workers=0,
        drop_last=True,
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    assert loader.drop_last is True
    assert [len(batch["audio"]) for batch in loader] == [2]


def test_datamodule_drop_last_tiny_split_retains_partial_batch() -> None:
    """An undersized opted-in split remains nonempty so runtime validation can reject it."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(1, 1, 1),
        batch_size=2,
        num_workers=0,
        drop_last=True,
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    assert loader.drop_last is False
    assert len(next(iter(loader))["audio"]) == 1


def _epoch_param_rows(loader: DataLoader[TorchSynthBatch]) -> list[tuple[float, ...]]:
    """Collect one epoch of parameter rows as hashable tuples.

    :param loader: Batched loader over one online split.
    :returns: One flattened parameter tuple per batch, in iteration order.
    """
    return [tuple(batch["params"].flatten().tolist()) for batch in loader]


def test_datamodule_resample_train_per_epoch_yields_fresh_rows_each_epoch() -> None:
    """With resampling on, every epoch draws parameter rows never seen in prior epochs."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(2, 1, 1),
        batch_size=1,
        num_workers=0,
        resample_train_per_epoch=True,
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    first_epoch = _epoch_param_rows(loader)
    second_epoch = _epoch_param_rows(loader)

    assert len(first_epoch) == len(second_epoch) == 2
    assert set(first_epoch).isdisjoint(second_epoch)


def _two_epoch_resampled_rows() -> list[tuple[float, ...]]:
    """Draw two consecutive resampled train epochs from a freshly built datamodule.

    :returns: Concatenated parameter rows of both epochs, in iteration order.
    """
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(2, 1, 1),
        batch_size=1,
        num_workers=0,
        resample_train_per_epoch=True,
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    return _epoch_param_rows(loader) + _epoch_param_rows(loader)


def test_datamodule_resample_train_per_epoch_sequence_reproducible_across_runs() -> None:
    """Two identically seeded runs draw the same fresh-row sequence over two epochs."""
    assert _two_epoch_resampled_rows() == _two_epoch_resampled_rows()


def test_datamodule_resample_train_per_epoch_default_repeats_rows_each_epoch() -> None:
    """Without the option, every epoch revisits the same fixed train rows."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(2, 1, 1),
        batch_size=1,
        num_workers=0,
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    assert set(_epoch_param_rows(loader)) == set(_epoch_param_rows(loader))


def test_datamodule_resample_train_per_epoch_keeps_val_rows_fixed() -> None:
    """Resampling applies to the train split only; validation stays deterministic."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(1, 2, 1),
        batch_size=1,
        num_workers=0,
        resample_train_per_epoch=True,
    )
    datamodule.setup("fit")
    loader = datamodule.val_dataloader()

    assert _epoch_param_rows(loader) == _epoch_param_rows(loader)


@pytest.mark.dataloader_multiprocess
@pytest.mark.xdist_group(name="dataloader-multiprocess")
@pytest.mark.slow
def test_datamodule_multiprocessing_workers_render_finite_batches() -> None:
    """Iterating a split with ``num_workers>0`` renders finite batches through forked workers.

    Exercises the per-worker ``@cache`` / PL-shim re-import path (CPU rendering in forked workers)
    that the single-process tests never reach.
    """
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(4, 1, 1),
        batch_size=2,
        num_workers=2,
    )
    datamodule.setup("fit")
    batches = list(datamodule.train_dataloader())

    assert len(batches) == 2
    for batch in batches:
        audio, params = batch["audio"], batch["params"]
        assert audio.shape[0] == params.shape[0] == 2
        assert params.shape[1] == datamodule.num_params
        assert torch.isfinite(audio).all()


def _encoded_row(seed: int, pitch: int, window: tuple[float, float]) -> torch.Tensor:
    """Encode one sampled synth patch with explicit note conditioning.

    :param seed: Seed for the spec's synth-parameter draw.
    :param pitch: MIDI pitch written into the pitch column.
    :param window: Note start/end seconds written into the note columns.
    :returns: Float32 encoded row shaped ``(1, encoded_width)``.
    """
    synth_values, _ = TORCHSYNTH_FULL_PARAM_SPEC.sample(np.random.default_rng(seed))
    row = TORCHSYNTH_FULL_PARAM_SPEC.encode(
        synth_values, {"pitch": pitch, "note_start_and_end": window}
    )
    return torch.from_numpy(row).unsqueeze(0)


def test_render_torchsynth_renders_each_row_at_its_own_pitch() -> None:
    """Pitch is per row: a batch's rows match their single-row renders and differ from each other."""
    window = (0.0, _BUFFER_SECONDS)
    low = _encoded_row(0, _PITCH_PARAM.min, window)
    high = _encoded_row(0, _PITCH_PARAM.max, window)

    batch = render_torchsynth(torch.cat((low, high)), **_RENDER_KWARGS, render_batch_size=2)

    assert torch.equal(batch[0], render_torchsynth(low, **_RENDER_KWARGS)[0])
    assert torch.equal(batch[1], render_torchsynth(high, **_RENDER_KWARGS)[0])
    assert not torch.equal(batch[0], batch[1])


def test_render_torchsynth_renders_each_row_at_its_own_note_duration() -> None:
    """The note-on length is per row, so a batch mixes early and late releases."""
    short = _encoded_row(0, 60, (0.0, 0.02))
    held = _encoded_row(0, 60, (0.0, _BUFFER_SECONDS))

    batch = render_torchsynth(torch.cat((short, held)), **_RENDER_KWARGS, render_batch_size=2)

    assert torch.equal(batch[0], render_torchsynth(short, **_RENDER_KWARGS)[0])
    assert torch.equal(batch[1], render_torchsynth(held, **_RENDER_KWARGS)[0])
    assert not torch.equal(batch[0], batch[1])


def test_render_torchsynth_note_start_delays_each_row_independently() -> None:
    """A row's note start zero-fills its own head, matching the offline torchsynth backend."""
    offset_seconds = _BUFFER_SECONDS / 4
    offset_samples = round(offset_seconds * _RENDER_KWARGS["sample_rate"])
    at_zero = _encoded_row(0, 60, (0.0, _BUFFER_SECONDS))
    delayed = _encoded_row(0, 60, (offset_seconds, offset_seconds + _BUFFER_SECONDS))

    batch = render_torchsynth(torch.cat((at_zero, delayed)), **_RENDER_KWARGS, render_batch_size=2)

    assert torch.equal(batch[1, :offset_samples], torch.zeros(offset_samples))
    assert torch.equal(
        batch[1, offset_samples:], batch[0, : _RENDER_KWARGS["signal_length"] - offset_samples]
    )


def test_render_torchsynth_out_of_range_note_columns_clamp_instead_of_raising() -> None:
    """Raw model rows can leave ``[0, 1]``; note columns clamp rather than trip the voice."""
    wild = _encoded_row(0, 60, (0.0, _BUFFER_SECONDS))
    wild[:, TORCHSYNTH_FULL_PARAM_SPEC.synth_param_length :] = torch.tensor([-3.0, 0.0, -1.0])

    audio = render_torchsynth(wild, **_RENDER_KWARGS)

    assert torch.isfinite(audio).all()
    assert audio.shape == (1, _RENDER_KWARGS["signal_length"])


def test_render_torchsynth_note_window_beyond_the_spec_maximum_clamps() -> None:
    """A window longer than the note param's range renders as the longest renderable note."""
    longest = _NOTE_WINDOW_PARAM.max_note_duration_seconds

    assert torch.equal(
        render_torchsynth(_encoded_row(0, 60, (0.0, 2 * longest)), **_RENDER_KWARGS),
        render_torchsynth(_encoded_row(0, 60, (0.0, longest)), **_RENDER_KWARGS),
    )


def test_render_torchsynth_synth_only_width_raises() -> None:
    """A row missing the note columns is a contract violation, not a silent truncation."""
    with pytest.raises(ValueError, match="encoded parameter columns"):
        render_torchsynth(torch.full((1, NUM_PARAMS), 0.4), **_RENDER_KWARGS)


def test_render_torchsynth_multirow_preserves_shape_and_bounds() -> None:
    """A multi-row renderer call preserves batch shape and numeric contracts."""
    params = torch.rand((3, _ENCODED_WIDTH), generator=torch.Generator().manual_seed(0))
    audio = render_torchsynth(params, **_RENDER_KWARGS, render_batch_size=3)

    assert audio.shape == (3, _RENDER_KWARGS["signal_length"])
    assert type(audio) is torch.Tensor
    assert audio.dtype == torch.float32
    assert torch.isfinite(audio).all()
    assert torch.all((-1 <= audio) & (audio <= 1))


@pytest.mark.slow
def test_render_torchsynth_deterministic_across_processes() -> None:
    """Rendering identical params in a fresh interpreter yields byte-identical audio.

    ``reproducible=False`` disables torchsynth's own reproducibility guarantees, so the
    fixed val/test audio's cross-process stability rests on torchsynth seeding its
    ``Noise`` buffer deterministically at construction. A fresh subprocess has independent
    default RNG, so a matching hash *is* the determinism proof — pinned here so a
    torchsynth upgrade that breaks it fails loudly instead of silently shifting the audio.
    """
    params = torch.full((2, _ENCODED_WIDTH), 0.3)
    reference_hash = hashlib.sha256(
        render_torchsynth(params, **_RENDER_KWARGS, render_batch_size=2).numpy().tobytes()
    ).hexdigest()
    script = (
        "import hashlib, torch;"
        "from synth_setter.data.torchsynth_datamodule import render_torchsynth;"
        f"audio = render_torchsynth(torch.full((2, {_ENCODED_WIDTH}), 0.3),"
        f" sample_rate={_RENDER_KWARGS['sample_rate']},"
        f" signal_length={_RENDER_KWARGS['signal_length']}, render_batch_size=2);"
        "print(hashlib.sha256(audio.numpy().tobytes()).hexdigest())"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip().splitlines()[-1] == reference_hash


def test_render_torchsynth_concurrent_calls_match_serial_results() -> None:
    """Serialize shared cached voice mutation without cross-contaminating renders."""
    parameter_rows = [torch.full((1, _ENCODED_WIDTH), value) for value in (0.25, 0.75)]
    expected = [render_torchsynth(row, **_RENDER_KWARGS) for row in parameter_rows]
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(
            executor.map(lambda row: render_torchsynth(row, **_RENDER_KWARGS), parameter_rows)
        )
    for concurrent, serial in zip(actual, expected, strict=True):
        assert torch.equal(concurrent, serial)


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(float("-inf"), id="negative-inf"),
    ],
)
def test_render_torchsynth_non_finite_params_raise(bad_value: float) -> None:
    """NaN or Inf parameter values are contract violations, not silently coerced.

    :param bad_value: Non-finite value injected into one parameter.
    """
    params = torch.full((1, _ENCODED_WIDTH), 0.5)
    params[0, 3] = bad_value
    with pytest.raises(ValueError, match="params must be finite"):
        render_torchsynth(params, **_RENDER_KWARGS)


def test_render_torchsynth_out_of_range_synth_params_clamp_to_valid_domain() -> None:
    """Finite out-of-range synth params (raw model predictions) render as their clamped form."""
    wild = _encoded_row(0, 60, (0.0, _BUFFER_SECONDS))
    wild[:, _SYNTH_COLUMNS] = 1.5
    wild[0, 0 : _SYNTH_COLUMNS.stop : 2] = -0.5
    clamped = wild.clone()
    clamped[:, _SYNTH_COLUMNS] = wild[:, _SYNTH_COLUMNS].clamp(
        _PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS
    )

    assert torch.equal(
        render_torchsynth(wild, **_RENDER_KWARGS), render_torchsynth(clamped, **_RENDER_KWARGS)
    )


@pytest.mark.gpu
@RunIf(min_gpus=1)
def test_render_torchsynth_preserves_gpu_device() -> None:
    """Render on the device used by the default GPU experiment."""
    params = torch.rand((2, _ENCODED_WIDTH), device="cuda")
    audio = render_torchsynth(params, **_RENDER_KWARGS, render_batch_size=2)
    assert audio.device == params.device
    assert torch.isfinite(audio).all()
