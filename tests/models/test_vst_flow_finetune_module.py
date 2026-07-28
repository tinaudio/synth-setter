"""Smoke tests for the simulator-feedback finetune module (spike #2554).

Deliberately minimal for the spike: construction, one training step against a
mocked renderer, and the trainable-parameter split.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import numpy as np
import torch

from synth_setter.data.vst.renderers import AudioRenderer
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.vst_flow_finetune_module import VSTFlowMatchingFinetuneModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

# surge_simple encoded width; the module decodes rows against that spec.
_NUM_PARAMS = 92
_MEL_CHANNELS = 2
_MEL_N_MELS = 128
_MEL_N_FRAMES = 11
_COND_DIM = 8
_BATCH = 3


class _TinyEncoder(torch.nn.Module):
    """Flatten-and-project conditioning encoder for tiny test batches."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_MEL_CHANNELS * _MEL_N_MELS * _MEL_N_FRAMES, _COND_DIM)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Project a mel batch to one conditioning vector per sample.

        :param mel_spec: Batch of mel spectrograms.
        :returns: Conditioning of shape ``(B, _COND_DIM)``.
        """
        return self.linear(mel_spec.flatten(start_dim=1))


class _FakeRenderer:
    """Renderer stand-in returning silent audio and counting calls."""

    def __init__(self) -> None:
        self.calls = 0

    def render(
        self,
        params: dict[str, float],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
    ) -> np.ndarray:
        """Return silent stereo audio for any request.

        :param params: Ignored native parameter dict.
        :param midi_note: Ignored MIDI pitch.
        :param velocity: Ignored MIDI velocity.
        :param note_start_and_end: Note window; must be a valid ordered pair.
        :returns: Silent audio shaped ``(2, 4410)``.
        """
        start, end = note_start_and_end
        assert 0.0 <= start < end
        self.calls += 1
        return np.zeros((2, 4410), dtype=np.float32)


def _tiny_base() -> VSTFlowMatchingModule:
    return VSTFlowMatchingModule(
        encoder=_TinyEncoder(),
        vector_field=VectorField(
            field_dim=_NUM_PARAMS, hidden_dim=16, conditioning_dim=_COND_DIM, num_blocks=1
        ),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_NUM_PARAMS,
    )


def _make_module(*, feedback_enabled: bool) -> tuple[VSTFlowMatchingFinetuneModule, _FakeRenderer]:
    module = VSTFlowMatchingFinetuneModule(
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        base_module=_tiny_base(),
        feedback_enabled=feedback_enabled,
        control_dim=16,
        control_hidden_dim=32,
        control_num_blocks=1,
        signal_duration_seconds=0.1,
    )
    fake = _FakeRenderer()
    module._renderer = cast(AudioRenderer, fake)
    module._runtime_ready = True
    return module, fake


def _fake_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    params = torch.rand(_BATCH, _NUM_PARAMS, generator=generator) * 2 - 1
    return {
        "params": params,
        "noise": torch.randn(_BATCH, _NUM_PARAMS, generator=generator),
        "mel_spec": torch.rand(
            _BATCH, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES, generator=generator
        ),
    }


def test_training_step_feedback_enabled_renders_and_trains_control_only() -> None:
    """One feedback step renders per row and produces grads only on control nets."""
    module, fake = _make_module(feedback_enabled=True)
    loss = module.training_step(_fake_batch(), 0)

    assert torch.isfinite(loss)
    assert fake.calls == _BATCH

    loss.backward()
    control_params = list(module.control_encoder.parameters()) + list(
        module.control_field.parameters()
    )
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in control_params)
    assert all(not p.requires_grad for p in module.base.parameters())


def test_training_step_feedback_disabled_skips_renderer() -> None:
    """The ablation path never touches the renderer."""
    module, fake = _make_module(feedback_enabled=False)
    loss = module.training_step(_fake_batch(), 0)

    assert torch.isfinite(loss)
    assert fake.calls == 0


def test_zero_init_control_field_matches_base_loss_at_start() -> None:
    """Zero-init control output makes the initial loss equal the frozen-base loss."""
    module, _ = _make_module(feedback_enabled=True)
    loss, metrics = module._feedback_step(_fake_batch())

    assert torch.allclose(loss, cast(torch.Tensor, metrics["base_loss"]), atol=1e-6)


def test_sample_with_feedback_renders_only_in_control_window() -> None:
    """Euler sampling renders once per row per control-window step and keeps shape."""
    module, fake = _make_module(feedback_enabled=True)
    batch = _fake_batch()

    sample = module.sample_with_feedback(batch["mel_spec"], batch["noise"], steps=10)

    assert sample.shape == batch["noise"].shape
    # t_min=0.8 and 10 steps leave 2 control-window steps of _BATCH renders each.
    assert fake.calls == 2 * _BATCH


def test_render_reps_batched_torch_mel_matches_dataset_librosa_mel() -> None:
    """The GPU log-mel path reproduces the dataset's librosa mel within tolerance."""
    from synth_setter.data.vst.generate_vst_dataset import make_spectrogram

    sample_rate = 44100
    duration = 0.1
    n = int(sample_rate * duration)
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((2, n)).astype(np.float32) * 0.1

    class _ToneRenderer(_FakeRenderer):
        def render(
            self,
            params: dict[str, float],
            midi_note: int,
            velocity: int,
            note_start_and_end: tuple[float, float],
        ) -> np.ndarray:
            """Return the fixed reference audio for any request.

            :param params: Ignored native parameter dict.
            :param midi_note: Ignored MIDI pitch.
            :param velocity: Ignored MIDI velocity.
            :param note_start_and_end: Note window forwarded to the base fake.
            :returns: The module-level reference audio, shaped ``(2, n)``.
            """
            super().render(params, midi_note, velocity, note_start_and_end)
            return audio

    module, _ = _make_module(feedback_enabled=True)
    module._renderer = cast(AudioRenderer, _ToneRenderer())
    theta_hat = _fake_batch()["params"]

    reps, _, _ = module._render_reps(theta_hat)

    expected = make_spectrogram(audio, sample_rate)
    assert reps.shape == (_BATCH, *expected.shape)
    np.testing.assert_allclose(reps[0].cpu().numpy(), expected, atol=0.02)


def test_render_reps_applies_dataset_mel_statistics() -> None:
    """Loaded mel mean/std normalize the rendered rep exactly like the spike path."""
    module, _ = _make_module(feedback_enabled=True)
    mean = np.full((1, 1, 1), 2.0, dtype=np.float32)
    std = np.full((1, 1, 1), 4.0, dtype=np.float32)

    raw, _, _ = module._render_reps(_fake_batch()["params"])
    module._mel_stats = (mean, std)
    module._mel_stats_tensors = None
    normalized, _, _ = module._render_reps(_fake_batch()["params"])

    torch.testing.assert_close(normalized, (raw - 2.0) / 4.0)


def test_validation_step_first_batch_returns_probe_preds() -> None:
    """Batch 0 returns a dict carrying sampled preds so ValAudioProbe can stage them."""
    module, _ = _make_module(feedback_enabled=True)
    batch = _fake_batch()

    outputs = module.validation_step(batch, 0)

    assert isinstance(outputs, dict)
    assert outputs["preds"].shape == batch["params"].shape
    assert torch.isfinite(outputs["loss"])


def test_validation_step_later_batches_skip_sampling() -> None:
    """Batches past 0 return the plain scalar loss without probe sampling renders."""
    module, fake = _make_module(feedback_enabled=True)
    batch = _fake_batch()

    outputs = module.validation_step(batch, 1)

    assert isinstance(outputs, torch.Tensor)
    # One CFM feedback pass renders exactly once per row; probe sampling would add more.
    assert fake.calls == _BATCH


def test_configure_optimizers_trains_only_control_parameters() -> None:
    """The optimizer covers exactly the control encoder + field parameters."""
    module, _ = _make_module(feedback_enabled=True)
    optimizer = module.configure_optimizers()["optimizer"]

    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    control = {
        id(p) for p in (*module.control_encoder.parameters(), *module.control_field.parameters())
    }
    assert optimized == control
