"""Focused contracts for online TorchSynth sampling and rendering."""

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import cast

import pytest
import torch

from synth_setter.data.torchsynth_datamodule import (
    TorchSynthDataModule,
    TorchSynthDataset,
    render_torchsynth,
)
from tests.helpers.run_if import RunIf as _RunIf

_RENDER_KWARGS = {"sample_rate": 44_100, "signal_length": 4_410, "midi_pitch": 60}
RunIf = cast(Callable[..., pytest.MarkDecorator], _RunIf)


def test_dataset_same_index_deterministic_different_index_distinct() -> None:
    """Repeated reads are stable while adjacent rows remain distinct."""
    dataset = TorchSynthDataset(2, 123, **_RENDER_KWARGS)
    audio_a, params_a, _ = dataset[0]
    audio_b, params_b, _ = dataset[0]
    _, params_c, _ = dataset[1]

    assert torch.equal(params_a, params_b)
    assert torch.equal(audio_a, audio_b)
    assert not torch.equal(params_a, params_c)
    assert params_a.dtype == audio_a.dtype == torch.float32
    assert torch.all((0 <= params_a) & (params_a <= 1))
    assert torch.all((-1 <= audio_a) & (audio_a <= 1))


def test_datamodule_split_seeds_produce_distinct_first_parameters() -> None:
    """Keep train, validation, and test seed streams disjoint at index zero."""
    datamodule = TorchSynthDataModule(
        sample_rate=44_100,
        signal_length=4_410,
        midi_pitch=60,
        train_val_test_sizes=(1, 1, 1),
        num_workers=0,
    )
    datamodule.setup(None)
    split_params = [datamodule.train[0][1], datamodule.val[0][1], datamodule.test[0][1]]
    assert not torch.equal(split_params[0], split_params[1])
    assert not torch.equal(split_params[0], split_params[2])
    assert not torch.equal(split_params[1], split_params[2])


def test_render_torchsynth_multirow_preserves_shape_and_bounds() -> None:
    """A multi-row renderer call preserves batch shape and numeric contracts."""
    params = torch.rand((3, 76))
    audio = render_torchsynth(params, **_RENDER_KWARGS)

    assert audio.shape == (3, _RENDER_KWARGS["signal_length"])
    assert audio.dtype == torch.float32
    assert torch.isfinite(audio).all()
    assert torch.all((-1 <= audio) & (audio <= 1))


def test_render_torchsynth_concurrent_calls_match_serial_results() -> None:
    """Serialize shared cached voice mutation without cross-contaminating renders."""
    parameter_rows = [torch.full((1, 76), value) for value in (0.25, 0.75)]
    expected = [render_torchsynth(row, **_RENDER_KWARGS) for row in parameter_rows]
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(executor.map(lambda row: render_torchsynth(row, **_RENDER_KWARGS), parameter_rows))
    for concurrent, serial in zip(actual, expected, strict=True):
        assert torch.equal(concurrent, serial)


def test_render_torchsynth_wrong_parameter_width_raises() -> None:
    """Reject parameter rows that do not match the native TorchSynth voice."""
    with pytest.raises(ValueError, match="Expected 76 TorchSynth parameters"):
        render_torchsynth(torch.rand((1, 75)), **_RENDER_KWARGS)


@pytest.mark.gpu
@RunIf(min_gpus=1)
def test_render_torchsynth_preserves_gpu_device() -> None:
    """Render on the device used by the default GPU experiment."""
    params = torch.rand((2, 76), device="cuda")
    audio = render_torchsynth(params, **_RENDER_KWARGS)
    assert audio.device == params.device
    assert torch.isfinite(audio).all()
