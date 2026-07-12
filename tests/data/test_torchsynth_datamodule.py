"""Focused contracts for online TorchSynth sampling and rendering."""

import dataclasses
import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from synth_setter.data.torchsynth_datamodule import (
    _PARAM_CLAMP_EPS,
    NUM_PARAMS,
    PARAM_SPEC,
    TorchSynthDataModule,
    TorchSynthDataset,
    _make_renderer,
    _spec_from_voice,
    _verify_voice_matches_spec,
    render_torchsynth,
)
from tests.helpers.run_if import RunIf

_RENDER_KWARGS = {"sample_rate": 44_100, "signal_length": 4_410, "midi_pitch": 60}


def test_param_spec_matches_live_voice() -> None:
    """The checked-in ``PARAM_SPEC`` snapshot equals the spec extracted from a live voice.

    This is the drift test: a torchsynth upgrade that adds, renames, reorders, or
    re-ranges any voice parameter fails here (and in ``setup()``) instead of silently
    mislabeling the model's positional targets.
    """
    assert NUM_PARAMS == 76
    voice = _make_renderer(_RENDER_KWARGS["sample_rate"], _RENDER_KWARGS["signal_length"]).voice
    assert _spec_from_voice(voice) == PARAM_SPEC


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


def test_dataset_item_has_normalized_float32_params_and_audio() -> None:
    """Online rows expose renderable labels and finite normalized audio."""
    dataset = TorchSynthDataset(1, 123, **_RENDER_KWARGS)
    audio, params, render_fn = dataset[0]

    assert params.dtype == audio.dtype == torch.float32
    assert torch.all((_PARAM_CLAMP_EPS <= params) & (params <= 1 - _PARAM_CLAMP_EPS))
    assert torch.all((-1 <= audio) & (audio <= 1))
    assert torch.equal(render_fn(params), audio)


def test_datamodule_split_seeds_produce_distinct_parameters_across_indices() -> None:
    """Keep train, validation, and test parameter streams disjoint across several indices."""
    datamodule = TorchSynthDataModule(
        sample_rate=44_100,
        signal_length=4_410,
        midi_pitch=60,
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
    """A configured ``num_params`` disagreeing with the live voice fails fast in ``setup``."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        num_params=1,
        train_val_test_sizes=(1, 1, 1),
        num_workers=0,
    )
    with pytest.raises(
        ValueError, match=rf"Configured num_params=1, TorchSynth exposes {NUM_PARAMS}"
    ):
        datamodule.setup(None)


def test_datamodule_test_dataloader_yields_finite_batch() -> None:
    """``setup('test')`` builds the test split and ``test_dataloader`` yields a finite batch."""
    datamodule = TorchSynthDataModule(
        signal_length=4_410,
        train_val_test_sizes=(1, 1, 2),
        batch_size=2,
        num_workers=0,
    )
    datamodule.setup("test")
    audio, params, *_ = next(iter(datamodule.test_dataloader()))
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
    audio, *_ = next(iter(datamodule.val_dataloader()))
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


@pytest.mark.slow
def test_datamodule_multiprocessing_workers_render_finite_batches() -> None:
    """Iterating a split with ``num_workers>0`` renders finite batches through forked workers.

    The production config defaults to ``num_workers=4``; this exercises the per-worker
    ``@cache`` / PL-shim re-import path (CPU rendering in forked workers, the real
    train-on-GPU geometry) that the ``num_workers=0`` tests never reach.
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
    for audio, params, *_ in batches:
        assert audio.shape[0] == params.shape[0] == 2
        assert params.shape[1] == datamodule.num_params
        assert torch.isfinite(audio).all()


def test_render_torchsynth_multirow_preserves_shape_and_bounds() -> None:
    """A multi-row renderer call preserves batch shape and numeric contracts."""
    params = torch.rand((3, NUM_PARAMS), generator=torch.Generator().manual_seed(0))
    audio = render_torchsynth(params, **_RENDER_KWARGS)

    assert audio.shape == (3, _RENDER_KWARGS["signal_length"])
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
    params = torch.full((2, NUM_PARAMS), 0.3)
    reference_hash = hashlib.sha256(
        render_torchsynth(params, **_RENDER_KWARGS).numpy().tobytes()
    ).hexdigest()
    script = (
        "import hashlib, torch;"
        "from synth_setter.data.torchsynth_datamodule import render_torchsynth;"
        f"audio = render_torchsynth(torch.full((2, {NUM_PARAMS}), 0.3),"
        f" sample_rate={_RENDER_KWARGS['sample_rate']},"
        f" signal_length={_RENDER_KWARGS['signal_length']},"
        f" midi_pitch={_RENDER_KWARGS['midi_pitch']});"
        "print(hashlib.sha256(audio.numpy().tobytes()).hexdigest())"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip().splitlines()[-1] == reference_hash


def test_render_torchsynth_concurrent_calls_match_serial_results() -> None:
    """Serialize shared cached voice mutation without cross-contaminating renders."""
    parameter_rows = [torch.full((1, NUM_PARAMS), value) for value in (0.25, 0.75)]
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
    params = torch.full((1, NUM_PARAMS), 0.5)
    params[0, 3] = bad_value
    with pytest.raises(ValueError, match="params must be finite"):
        render_torchsynth(params, **_RENDER_KWARGS)


def test_render_torchsynth_out_of_range_params_clamp_to_valid_domain() -> None:
    """Finite out-of-range params (raw model predictions) render as their clamped equivalents."""
    wild = torch.full((1, NUM_PARAMS), 1.5)
    wild[0, ::2] = -0.5
    clamped = wild.clamp(_PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS)
    assert torch.equal(
        render_torchsynth(wild, **_RENDER_KWARGS), render_torchsynth(clamped, **_RENDER_KWARGS)
    )


def test_render_torchsynth_wrong_parameter_width_raises() -> None:
    """Reject parameter rows that do not match the native TorchSynth voice."""
    with pytest.raises(ValueError, match=rf"Expected {NUM_PARAMS} TorchSynth parameters"):
        render_torchsynth(torch.rand((1, NUM_PARAMS - 1)), **_RENDER_KWARGS)


@pytest.mark.gpu
@RunIf(min_gpus=1)
def test_render_torchsynth_preserves_gpu_device() -> None:
    """Render on the device used by the default GPU experiment."""
    params = torch.rand((2, NUM_PARAMS), device="cuda")
    audio = render_torchsynth(params, **_RENDER_KWARGS)
    assert audio.device == params.device
    assert torch.isfinite(audio).all()
