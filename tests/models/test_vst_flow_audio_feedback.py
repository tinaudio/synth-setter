"""The audio-feedback loss wired into the production flow module.

Exercises the integrated path: an online torchsynth dict batch through
``VSTFlowMatchingModule`` with an attached :class:`AudioFeedbackLoss`, plus the
runtime guards that refuse configurations the differentiable renderer cannot
serve.
"""

from __future__ import annotations

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from synth_setter.data.torchsynth_datamodule import TorchSynthDataModule
from synth_setter.models.components.audio_feedback import AudioDistance, AudioFeedbackLoss
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_SAMPLE_RATE = 44_100
_SIGNAL_LENGTH = 4_410
_MIDI_PITCH = 60
_BATCH = 4
_NUM_PARAMS = 76
_CONDITIONING_DIM = 32


class _WaveformEncoder(torch.nn.Module):
    """Minimal raw-audio conditioning encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_SIGNAL_LENGTH, _CONDITIONING_DIM)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Map a waveform batch to a flat conditioning vector.

        :param audio: Audio shaped ``(batch, _SIGNAL_LENGTH)``.
        :returns: Conditioning shaped ``(batch, _CONDITIONING_DIM)``.
        """
        return self.linear(audio)


def _audio_loss() -> AudioFeedbackLoss:
    """Build an audio-feedback loss with test-sized render settings.

    :returns: Configured loss module.
    """
    return AudioFeedbackLoss(
        lambda_audio=1.0,
        t_min=0.0,
        distance=AudioDistance.MSLM,
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        midi_pitch=_MIDI_PITCH,
    )


def _module(
    audio_loss: AudioFeedbackLoss | None = None, compile: bool = False
) -> VSTFlowMatchingModule:
    """Build a tiny flow module conditioned on raw audio.

    :param audio_loss: Optional audio-feedback term to attach.
    :param compile: Value forwarded to the module's ``compile`` flag.
    :returns: Configured module.
    """
    return VSTFlowMatchingModule(
        encoder=_WaveformEncoder(),
        vector_field=VectorField(
            field_dim=_NUM_PARAMS,
            hidden_dim=32,
            conditioning_dim=_CONDITIONING_DIM,
            num_blocks=2,
        ),
        optimizer=torch.optim.Adam,  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_NUM_PARAMS,
        conditioning="audio",
        audio_loss=audio_loss,
        compile=compile,
    )


def _datamodule() -> TorchSynthDataModule:
    """Build a tiny online datamodule already set up for fitting.

    :returns: The datamodule.
    """
    datamodule = TorchSynthDataModule(
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        midi_pitch=_MIDI_PITCH,
        train_val_test_sizes=(8, 4, 4),
        train_val_test_seeds=(1, 2, 3),
        batch_size=_BATCH,
        num_workers=0,
    )
    datamodule.setup("fit")
    return datamodule


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Build a render-free batch matching the online dict contract.

    :returns: Batch with random params, noise, and audio.
    """
    generator = torch.Generator().manual_seed(0)
    return {
        "params": torch.rand(_BATCH, _NUM_PARAMS, generator=generator) * 2 - 1,
        "noise": torch.randn(_BATCH, _NUM_PARAMS, generator=generator),
        "audio": torch.randn(_BATCH, _SIGNAL_LENGTH, generator=generator),
    }


@pytest.mark.slow
def test_train_step_with_audio_loss_backprops_a_finite_nonzero_audio_term():
    """A real online batch produces a finite audio term with gradients in the flow."""
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss())
    batch = next(iter(_datamodule().train_dataloader()))

    loss, audio_term, _ = module._train_step(batch)
    assert audio_term is not None
    total = loss + audio_term
    total.backward()

    assert torch.isfinite(total)
    assert audio_term.item() > 0.0
    gradients = torch.cat(
        [p.grad.flatten() for p in module.vector_field.parameters() if p.grad is not None]
    )
    assert torch.isfinite(gradients).all()
    assert (gradients != 0).any()


def test_train_step_without_audio_loss_returns_no_audio_term():
    """The default module pays for no render and reports no audio term."""
    torch.manual_seed(0)
    module = _module()

    loss, audio_term, _ = module._train_step(_synthetic_batch())

    assert audio_term is None
    assert torch.isfinite(loss)


def test_module_with_audio_loss_and_torch_compile_raises_at_construction():
    """Compiling over the functional_call render miscompiles, so refuse up front."""
    with pytest.raises(ValueError, match="compile"):
        _module(audio_loss=_audio_loss(), compile=True)


def test_fit_with_audio_loss_and_non_drop_last_loader_raises():
    """A trailing partial batch would miss the renderer cache; fit must refuse it."""
    module = _module(audio_loss=_audio_loss())
    batch = _synthetic_batch()
    rows = [{key: value[i] for key, value in batch.items()} for i in range(_BATCH)]
    # A plain list is a valid map-style dataset; pyright's stub only admits Dataset.
    loader = DataLoader(rows, batch_size=_BATCH, drop_last=False)  # pyright: ignore[reportArgumentType]
    trainer = Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False
    )

    with pytest.raises(ValueError, match="drop_last"):
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)


@pytest.mark.slow
def test_fit_with_audio_loss_on_the_online_datamodule_completes_one_step():
    """The supported configuration trains end to end through the real datamodule."""
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss())
    trainer = Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False
    )

    trainer.fit(module, datamodule=_datamodule())

    assert trainer.state.finished
