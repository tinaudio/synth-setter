"""Behaviour tests for the rendered-audio reward that scores sampled parameters."""

import torch

from synth_setter.data.torchsynth_datamodule import render_torchsynth
from synth_setter.data.torchsynth_grad_render import differentiable_decode
from synth_setter.models.components.audio_distance import MultiScaleSpectralDistance
from synth_setter.models.components.rendered_reward import RenderedAudioReward
from tests.models.test_vst_flow_finetune_module import _audible_model_rows

_SAMPLE_RATE = 16_000
_SIGNAL_LENGTH = 8_192


def _reward(render_batch_size: int) -> RenderedAudioReward:
    return RenderedAudioReward(
        distance=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        render_batch_size=render_batch_size,
    )


def _target_audio(params: torch.Tensor) -> torch.Tensor:
    return render_torchsynth(
        differentiable_decode(params),
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        render_batch_size=params.shape[0],
    )


def test_rendered_audio_reward_preserves_stored_target_waveforms() -> None:
    """Stored torchsynth audio needs no preparation before candidate grouping."""
    target = _target_audio(_audible_model_rows(2, seed=2))

    prepared = _reward(render_batch_size=2).prepare_target(target)

    assert prepared is target


def test_rendered_audio_reward_prefers_the_row_that_produced_the_target() -> None:
    """The reward is non-positive and highest for the parameters the target was rendered from."""
    rows = _audible_model_rows(2, seed=3)
    target = _target_audio(rows[:1]).expand(2, -1)

    rewards = _reward(render_batch_size=2)(rows, target)

    assert rewards.shape == (2,)
    assert rewards[0] > rewards[1]
    assert (rewards <= 0).all()


def test_rendered_audio_reward_chunks_batches_wider_than_the_render_voice() -> None:
    """Rows beyond the voice width are rendered in further chunks, not dropped or rejected."""
    rows = _audible_model_rows(4, seed=4)
    target = _target_audio(rows[:1]).expand(4, -1)

    chunked = _reward(render_batch_size=2)(rows, target)
    whole = _reward(render_batch_size=4)(rows, target)

    torch.testing.assert_close(chunked, whole)


def test_rendered_audio_reward_carries_no_gradient() -> None:
    """Rewards are scored, never differentiated: a leaf estimate leaves them detached."""
    rows = _audible_model_rows(2, seed=5).requires_grad_(True)
    target = _target_audio(rows.detach()[:1]).expand(2, -1)

    rewards = _reward(render_batch_size=2)(rows, target)

    assert not rewards.requires_grad


def test_rendered_audio_reward_scores_each_row_against_its_own_target() -> None:
    """Swapping one row's target changes only that row's reward."""
    rows = _audible_model_rows(3, seed=6)
    targets = _target_audio(rows)
    reward = _reward(render_batch_size=3)

    aligned = reward(rows, targets)
    swapped = reward(rows, torch.stack([targets[0], targets[2], targets[2]]))

    torch.testing.assert_close(aligned[[0, 2]], swapped[[0, 2]])
    assert swapped[1] < aligned[1]
