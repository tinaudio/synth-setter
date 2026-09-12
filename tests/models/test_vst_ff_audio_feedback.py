"""Audio-feedback behavior of the feed-forward parameter backbone."""

from functools import partial

import pytest
import torch

from synth_setter.models.components.audio_feedback import AudioFeedbackLoss
from synth_setter.models.vst_ff_module import VSTFeedForwardModule


class _ParameterRenderer(torch.nn.Module):
    """Expand one predicted parameter into a two-sample waveform."""

    def validate(self, params: torch.Tensor) -> None:
        """Reject non-finite parameter rows.

        :param params: Predicted model-space parameters.
        :raises ValueError: A parameter is non-finite.
        """
        if not torch.isfinite(params).all():
            raise ValueError("params must be finite")

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Render each scalar parameter as a constant waveform.

        :param params: Predicted model-space parameters shaped ``(batch, 1)``.
        :returns: Waveforms shaped ``(batch, 2)``.
        """
        return params.expand(-1, 2)


class _SquaredAudioDistance(torch.nn.Module):
    """Return per-row waveform mean-square error."""

    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Measure corresponding waveform samples.

        :param rendered: Rendered waveforms.
        :param target: Target waveforms.
        :returns: Per-row mean-square error.
        """
        return (rendered - target).square().mean(dim=-1)


def _audio_loss() -> AudioFeedbackLoss:
    """Build a deterministic differentiable audio term.

    :returns: Audio-feedback loss with weight ``0.5``.
    """
    return AudioFeedbackLoss(
        lambda_audio=0.5,
        t_min=0.8,
        sample_rate=2,
        signal_length=2,
        render_batch_size=1,
        distance=_SquaredAudioDistance(),
        renderer=_ParameterRenderer(),  # pyright: ignore[reportArgumentType]
    )


def _module(
    *, audio_loss: AudioFeedbackLoss | None, compile: bool = False
) -> VSTFeedForwardModule:
    """Build a one-parameter feed-forward module.

    :param audio_loss: Optional rendered-audio training term.
    :param compile: Whether fit setup compiles the network.
    :returns: Configured module.
    """
    net = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        net.weight.fill_(1.0)
    return VSTFeedForwardModule(
        net=net,
        optimizer=partial(torch.optim.SGD, lr=0.1),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        conditioning="mel",
        audio_loss=audio_loss,
        compile=compile,
    )


def test_training_step_with_audio_feedback_adds_rendered_distance() -> None:
    """The FFN objective combines parameter MSE and full-weight rendered-audio error."""
    module = _module(audio_loss=_audio_loss())
    batch = {
        "audio": torch.tensor([[0.0, 0.0]]),
        "mel": torch.tensor([[1.0]]),
        "params": torch.tensor([[0.0]]),
    }

    loss = module.training_step(batch, batch_idx=0)

    assert loss.item() == pytest.approx(1.5)


def test_training_step_audio_feedback_backpropagates_to_ffn() -> None:
    """Rendered-audio error contributes a nonzero gradient to FFN weights."""
    module = _module(audio_loss=_audio_loss())
    batch = {
        "audio": torch.tensor([[0.0, 0.0]]),
        "mel": torch.tensor([[1.0]]),
        "params": torch.tensor([[1.0]]),
    }

    loss = module.training_step(batch, batch_idx=0)
    loss.backward()

    assert isinstance(module.net, torch.nn.Linear)
    assert module.net.weight.grad is not None
    assert module.net.weight.grad.item() == pytest.approx(1.0)


def test_constructor_with_audio_feedback_and_compile_raises() -> None:
    """FFN audio feedback rejects compilation before fit setup."""
    with pytest.raises(ValueError, match="audio feedback is incompatible with torch.compile"):
        _module(audio_loss=_audio_loss(), compile=True)
