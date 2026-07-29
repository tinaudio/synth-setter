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

from synth_setter.data.torchsynth_datamodule import TorchSynthDataModule, _make_renderer
from synth_setter.models.components.audio_feedback import AudioFeedbackLoss
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_SAMPLE_RATE = 44_100
_SIGNAL_LENGTH = 4_410
_MIDI_PITCH = 60
_BATCH = 4
_NUM_PARAMS = 76
_CONDITIONING_DIM = 32
_OVERFIT_STEPS = 300
_OVERFIT_TOTAL_THRESHOLD = 0.1


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

    :returns: Configured loss module at the shipped weight.
    """
    return AudioFeedbackLoss(
        lambda_audio=0.03,
        t_min=0.0,
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        midi_pitch=_MIDI_PITCH,
    )


def _module(
    audio_loss: AudioFeedbackLoss | None = None,
    compile: bool = False,
    cfg_dropout_rate: float = 0.1,
) -> VSTFlowMatchingModule:
    """Build a tiny flow module conditioned on raw audio.

    :param audio_loss: Optional audio-feedback term to attach.
    :param compile: Value forwarded to the module's ``compile`` flag.
    :param cfg_dropout_rate: Probability of replacing conditioning with the CFG token.
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
        cfg_dropout_rate=cfg_dropout_rate,
        compile=compile,
    )


def _datamodule(batch_size: int = _BATCH) -> TorchSynthDataModule:
    """Build a tiny online datamodule already set up for fitting.

    :param batch_size: Number of online examples per batch.
    :returns: The datamodule.
    """
    datamodule = TorchSynthDataModule(
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        midi_pitch=_MIDI_PITCH,
        train_val_test_sizes=(8, 4, 4),
        train_val_test_seeds=(1, 2, 3),
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
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


def test_train_step_with_audio_loss_backprops_a_finite_nonzero_audio_term() -> None:
    """A real online batch produces a finite audio term with gradients in the flow."""
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss())
    batch = next(iter(_datamodule().train_dataloader()))

    outputs = module._train_step(batch)
    assert outputs.audio_term is not None
    encoder_gradients = torch.autograd.grad(
        outputs.loss, tuple(module.encoder.parameters()), retain_graph=True, allow_unused=True
    )
    total = outputs.loss + outputs.audio_term
    total.backward()

    assert torch.isfinite(total)
    assert outputs.audio_term.item() > 0.0
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient).item() > 0
        for gradient in encoder_gradients
    )
    # Every trainable field parameter must join the graph; only the CFG dropout token
    # may legitimately sit out of a step where no row was dropped.
    for name, parameter in module.vector_field.named_parameters():
        if "dropout" in name:
            continue
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient is non-finite"
        assert (parameter.grad != 0).any(), f"{name} gradient is identically zero"


def _overfit_one_fixed_example() -> tuple[float, float]:
    """Overfit one deterministic online example.

    :returns: Initial and final combined objectives.
    """
    _make_renderer.cache_clear()
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss(), cfg_dropout_rate=0.0)
    module._sample_time = lambda n, device: torch.full((n, 1), 0.9, device=device)
    batch = next(iter(_datamodule(batch_size=1).train_dataloader()))
    optimizer = torch.optim.Adam(module.parameters(), lr=3e-4)
    initial_total: float | None = None

    for _ in range(_OVERFIT_STEPS):
        outputs = module._train_step(batch)
        assert outputs.audio_term is not None
        total = outputs.loss + outputs.audio_term
        if initial_total is None:
            initial_total = total.item()
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    final = module._train_step(batch)
    assert initial_total is not None
    assert final.audio_term is not None
    return initial_total, (final.loss + final.audio_term).item()


@pytest.mark.slow
def test_combined_audio_objective_overfits_one_fixed_online_example() -> None:
    """The integrated objective can fit one fixed example to near zero."""
    initial_total, final_total = _overfit_one_fixed_example()

    assert final_total < initial_total * 0.1
    assert final_total < _OVERFIT_TOTAL_THRESHOLD


def test_train_step_without_audio_loss_returns_no_audio_term() -> None:
    """The default module pays for no render and reports no audio term."""
    torch.manual_seed(0)
    module = _module()

    outputs = module._train_step(_synthetic_batch())

    assert outputs.audio_term is None
    assert torch.isfinite(outputs.loss)


def test_module_with_audio_loss_and_nonzero_sigma_min_raises() -> None:
    """A sigma-bearing path makes the one-step estimate inexact, so refuse it."""
    with pytest.raises(ValueError, match="rectified_sigma_min"):
        VSTFlowMatchingModule(
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
            audio_loss=_audio_loss(),
            rectified_sigma_min=0.01,
            compile=False,
        )


def test_audio_loss_keep_mask_zeroes_cfg_dropped_rows() -> None:
    """Rows dropped by CFG must contribute nothing to the audio term."""
    torch.manual_seed(0)
    loss = _audio_loss()
    batch = next(iter(_datamodule().train_dataloader()))
    theta_hat = torch.rand(_BATCH, _NUM_PARAMS) * 2 - 1
    t = torch.full((_BATCH, 1), 0.9)

    encoder = _WaveformEncoder()
    all_dropped = loss(theta_hat, t, batch["audio"], encoder, keep=torch.zeros(_BATCH))
    all_kept = loss(theta_hat, t, batch["audio"], encoder, keep=torch.ones(_BATCH))

    assert all_dropped.item() == 0.0
    assert all_kept.item() > 0.0


def test_grad_render_matches_the_row_at_a_time_production_render() -> None:
    """Batch grad renders align noise with the per-row target renderer.

    Without the chunk-0 noise alignment, every row past the first compares against a target
    rendered with a different noise realization.
    """
    from synth_setter.data.torchsynth_datamodule import render_torchsynth
    from synth_setter.data.torchsynth_grad_render import render_torchsynth_grad

    params01 = torch.rand(_BATCH, _NUM_PARAMS, generator=torch.Generator().manual_seed(7))
    row_targets = torch.cat(
        [
            render_torchsynth(
                row.unsqueeze(0),
                sample_rate=_SAMPLE_RATE,
                signal_length=_SIGNAL_LENGTH,
                midi_pitch=_MIDI_PITCH,
            )
            for row in params01
        ]
    )
    with torch.no_grad():
        batched = render_torchsynth_grad(
            params01,
            sample_rate=_SAMPLE_RATE,
            signal_length=_SIGNAL_LENGTH,
            midi_pitch=_MIDI_PITCH,
        ).clamp(-1, 1)

    assert torch.allclose(batched, row_targets, atol=1e-5)


def test_module_with_audio_loss_and_torch_compile_raises_at_construction() -> None:
    """Compiling over the functional_call render miscompiles, so refuse up front."""
    with pytest.raises(ValueError, match="compile"):
        _module(audio_loss=_audio_loss(), compile=True)


def test_fit_with_audio_loss_and_non_drop_last_loader_raises() -> None:
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


def test_fit_with_audio_loss_on_the_online_datamodule_completes_one_step() -> None:
    """The supported configuration trains end to end through the real datamodule."""
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss())
    trainer = Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False
    )

    trainer.fit(module, datamodule=_datamodule())

    assert trainer.state.finished
