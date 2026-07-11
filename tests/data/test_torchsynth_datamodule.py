"""Focused contracts for online TorchSynth sampling and rendering."""

import pytest
import torch

from synth_setter.data.torchsynth_datamodule import TorchSynthDataset, render_torchsynth
from tests.helpers.run_if import RunIf

_RENDER_KWARGS = {"sample_rate": 44_100, "signal_length": 4_410, "midi_pitch": 60}


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


def test_render_torchsynth_multirow_preserves_shape_and_bounds() -> None:
    """A multi-row renderer call preserves batch shape and numeric contracts."""
    params = torch.rand((3, 76))
    audio = render_torchsynth(params, **_RENDER_KWARGS)

    assert audio.shape == (3, _RENDER_KWARGS["signal_length"])
    assert audio.dtype == torch.float32
    assert torch.isfinite(audio).all()
    assert torch.all((-1 <= audio) & (audio <= 1))


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
