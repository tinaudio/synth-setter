"""Behaviour tests for the audio-loss flow spike prototypes.

The spike code lives outside ``src`` under ``prototypes/``; the helper below
puts the repo root on ``sys.path`` before importing it (declared spike).
"""

import sys
from pathlib import Path

import pytest
import torch

from synth_setter.models.components.vector_field import VectorField

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BATCH = 4


def _import_spike():
    """Import the spike modules with the repo root on ``sys.path``.

    :returns: The ``audio_loss`` and ``flow`` prototype modules.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from prototypes.torchsynth_feedback import audio_loss, flow

    return audio_loss, flow


def _flow_grads(vector_field: VectorField) -> torch.Tensor:
    """Flatten every gradient the flow actually received.

    ``cfg_dropout_token`` is skipped: stage B trains without CFG dropout, so it
    is never part of the graph.

    :param vector_field: Flow whose ``.grad`` tensors are collected.
    :returns: Concatenated gradients.
    """
    return torch.cat([p.grad.flatten() for p in vector_field.parameters() if p.grad is not None])


def _render_target(batch: int = _BATCH) -> torch.Tensor:
    """Render a fixed random patch batch as the audio-matching target.

    :param batch: Rows to render.
    :returns: Audio shaped ``(batch, SIGNAL_LENGTH)``.
    """
    _, flow = _import_spike()
    from synth_setter.data.torchsynth_datamodule import render_torchsynth

    generator = torch.Generator().manual_seed(4321)
    params01 = torch.rand((batch, 76), generator=generator)
    with torch.no_grad():
        return render_torchsynth(
            params01,
            sample_rate=flow.SAMPLE_RATE,
            signal_length=flow.SIGNAL_LENGTH,
            midi_pitch=flow.MIDI_PITCH,
        )


def test_differentiable_decode_in_range_input_matches_affine_map():
    """An in-range model-space value maps to ``(theta + 1) / 2`` exactly."""
    audio_loss, _ = _import_spike()
    theta = torch.tensor([[-0.5, 0.0, 0.5]])
    decoded = audio_loss.differentiable_decode(theta)
    assert torch.allclose(decoded, torch.tensor([[0.25, 0.5, 0.75]]))


def test_differentiable_decode_saturated_input_clamps_forward_value():
    """A model-space value outside ``[-1, 1]`` lands strictly inside ``(0, 1)``."""
    audio_loss, _ = _import_spike()
    decoded = audio_loss.differentiable_decode(torch.tensor([[-4.0, 4.0]]))
    assert (decoded > 0.0).all()
    assert (decoded < 1.0).all()


def test_differentiable_decode_saturated_input_keeps_gradient():
    """Saturated entries still receive gradient, so the loss can pull them back."""
    audio_loss, _ = _import_spike()
    theta = torch.tensor([[-4.0, 4.0]], requires_grad=True)
    (gradient,) = torch.autograd.grad(audio_loss.differentiable_decode(theta).sum(), theta)
    assert torch.allclose(gradient, torch.tensor([[0.5, 0.5]]))


def test_audio_weight_below_t_min_is_zero():
    """No audio weight before the feedback window opens."""
    audio_loss, _ = _import_spike()
    weight = audio_loss.audio_weight(torch.tensor([[0.0], [0.5], [0.79]]), 1.0, 0.8)
    assert torch.all(weight == 0.0)


def test_audio_weight_at_final_time_equals_lambda():
    """The ramp reaches the configured lambda at t=1."""
    audio_loss, _ = _import_spike()
    weight = audio_loss.audio_weight(torch.tensor([[1.0]]), 0.25, 0.8)
    assert torch.allclose(weight, torch.tensor([[0.25]]))


def test_audio_weight_midway_through_window_is_half_lambda():
    """The ramp is linear across the feedback window."""
    audio_loss, _ = _import_spike()
    weight = audio_loss.audio_weight(torch.tensor([[0.9]]), 1.0, 0.8)
    assert torch.allclose(weight, torch.tensor([[0.5]]))


@pytest.mark.slow
def test_combined_loss_backprops_render_gradient_into_vector_field():
    """A real torchsynth render sends finite, non-zero gradient into the flow weights."""
    audio_loss, flow = _import_spike()
    torch.manual_seed(0)
    encoder, vector_field = flow.build_base_flow("cpu")
    target_audio = _render_target()
    params = torch.rand((_BATCH, 76)) * 2 - 1
    x0 = torch.randn_like(params)
    t = torch.full((_BATCH, 1), 0.9)

    loss, metrics = audio_loss.combined_loss(
        encoder,
        vector_field,
        audio_loss.FlowBatch(params, target_audio, x0, t),
        audio_loss.AudioLossConfig(lambda_audio=1.0, t_min=0.8),
    )
    loss.backward()

    grads = _flow_grads(vector_field)
    assert torch.isfinite(grads).all()
    assert (grads != 0).any()
    assert metrics["audio_loss"] > 0.0


@pytest.mark.slow
def test_combined_loss_zero_lambda_matches_cfm_only_gradient():
    """With lambda 0 the audio branch contributes nothing to the gradient."""
    audio_loss, flow = _import_spike()
    torch.manual_seed(0)
    encoder, vector_field = flow.build_base_flow("cpu")
    target_audio = _render_target()
    params = torch.rand((_BATCH, 76)) * 2 - 1
    x0 = torch.randn_like(params)
    t = torch.full((_BATCH, 1), 0.9)

    loss, metrics = audio_loss.combined_loss(
        encoder,
        vector_field,
        audio_loss.FlowBatch(params, target_audio, x0, t),
        audio_loss.AudioLossConfig(lambda_audio=0.0, t_min=0.8),
    )
    loss.backward()
    with_audio_term = _flow_grads(vector_field)

    vector_field.zero_grad()
    x_t = x0 * (1 - t) + params * t
    prediction = vector_field(x_t, t, encoder(target_audio))
    (prediction - (params - x0)).square().mean().backward()
    cfm_only = _flow_grads(vector_field)

    assert torch.allclose(with_audio_term, cfm_only)
    assert metrics["audio_loss"] == 0.0


@pytest.mark.slow
def test_finetune_audio_loss_runs_and_updates_the_vector_field():
    """The stage-B entrypoint drives real renders and moves the flow weights."""
    audio_loss, flow = _import_spike()
    torch.manual_seed(0)
    encoder, vector_field = flow.build_base_flow("cpu")
    before = torch.cat([p.detach().flatten().clone() for p in vector_field.parameters()])

    history = audio_loss.finetune_audio_loss(
        encoder,
        vector_field,
        "cpu",
        audio_loss.FinetuneConfig(steps=2, batch_size=_BATCH, lambda_audio=1.0),
    )

    after = torch.cat([p.detach().flatten() for p in vector_field.parameters()])
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(record["loss"])) for record in history)
    assert not torch.allclose(before, after)


@pytest.mark.slow
def test_finetune_audio_loss_frozen_encoder_keeps_encoder_weights():
    """Stage B trains the flow only; the conditioning encoder stays put."""
    audio_loss, flow = _import_spike()
    torch.manual_seed(0)
    encoder, vector_field = flow.build_base_flow("cpu")
    before = torch.cat([p.detach().flatten().clone() for p in encoder.parameters()])

    audio_loss.finetune_audio_loss(
        encoder,
        vector_field,
        "cpu",
        audio_loss.FinetuneConfig(steps=1, batch_size=_BATCH, lambda_audio=1.0),
    )

    after = torch.cat([p.detach().flatten() for p in encoder.parameters()])
    assert torch.allclose(before, after)


@pytest.mark.slow
def test_run_stage_b_writes_a_flow_checkpoint_a_fresh_model_can_load(tmp_path: Path):
    """The stage-B entrypoint produces a checkpoint that loads into a real flow.

    :param tmp_path: Artifacts directory for the run.
    """
    audio_loss, flow = _import_spike()
    from prototypes.torchsynth_feedback import step_d_audio_loss

    torch.manual_seed(0)
    encoder, vector_field = flow.build_base_flow("cpu")
    torch.save(
        {"encoder": encoder.state_dict(), "vector_field": vector_field.state_dict()},
        tmp_path / step_d_audio_loss.BASE_FLOW_NAME,
    )

    metrics = step_d_audio_loss.run_stage_b(
        audio_loss.FinetuneConfig(steps=2, batch_size=_BATCH, lambda_audio=1.0),
        tmp_path,
        eval_batch_size=_BATCH,
        eval_batches=1,
        eval_sample_steps=2,
    )

    fresh_encoder, fresh_field = flow.build_base_flow("cpu")
    saved = torch.load(tmp_path / step_d_audio_loss.FINETUNED_NAME, weights_only=True)
    fresh_encoder.load_state_dict(saved["encoder"])
    fresh_field.load_state_dict(saved["vector_field"])
    predictions = flow.sample_ode(
        fresh_encoder,
        fresh_field,
        _render_target(),
        torch.randn(_BATCH, 76),
        config=flow.SampleConfig(steps=2),
    )

    assert predictions.shape == (_BATCH, 76)
    assert torch.isfinite(predictions).all()
    assert set(metrics) == {"before", "after"}
    assert (tmp_path / step_d_audio_loss.METRICS_NAME).is_file()


@pytest.mark.slow
def test_run_stage_b_without_a_base_flow_raises():
    """Stage B refuses to start when stage A has not been run."""
    audio_loss, _ = _import_spike()
    from prototypes.torchsynth_feedback import step_d_audio_loss

    with pytest.raises(FileNotFoundError, match="step_b_pretrain"):
        step_d_audio_loss.run_stage_b(audio_loss.FinetuneConfig(steps=1), Path("/nonexistent"))


@pytest.mark.slow
def test_per_param_grad_norms_returns_one_entry_per_synth_parameter():
    """The diagnostic reports the audio-gradient magnitude of every synth parameter."""
    audio_loss, flow = _import_spike()
    torch.manual_seed(0)
    target_audio = _render_target()
    theta = (torch.rand((_BATCH, 76)) * 2 - 1).requires_grad_(True)

    norms = audio_loss.per_param_grad_norms(theta, target_audio)

    assert norms.shape == (76,)
    assert torch.isfinite(norms).all()
    assert (norms > 0).any()
