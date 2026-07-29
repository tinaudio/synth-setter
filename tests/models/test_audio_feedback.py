"""Behaviour tests for the torchsynth audio-feedback loss and its runtime guards."""

import pytest
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset

from synth_setter.data.torchsynth_datamodule import render_torchsynth
from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
)
from synth_setter.data.vst.torchsynth_param_spec import INFERABLE_SPEC
from synth_setter.models.components.audio_feedback import (
    AudioFeedbackLoss,
    gradient_balance,
    validate_audio_feedback_runtime,
)

_SAMPLE_RATE = 44_100
_SIGNAL_LENGTH = 4_410
_MIDI_PITCH = 60
_BATCH = 4
_NUM_PARAMS = 76


def _render(params01: torch.Tensor) -> torch.Tensor:
    """Render a parameter batch without gradients.

    :param params01: Parameters in ``[0, 1]``.
    :returns: Rendered audio.
    """
    with torch.no_grad():
        return render_torchsynth_grad(
            params01,
            sample_rate=_SAMPLE_RATE,
            signal_length=_SIGNAL_LENGTH,
            midi_pitch=_MIDI_PITCH,
        )


def _loss(**kwargs: object) -> AudioFeedbackLoss:
    """Build an audio-feedback loss with test-sized render settings.

    :param **kwargs: Overrides forwarded to :class:`AudioFeedbackLoss`.
    :returns: Configured loss module.
    """
    settings = {
        "lambda_audio": 1.0,
        "t_min": 0.8,
        "sample_rate": _SAMPLE_RATE,
        "signal_length": _SIGNAL_LENGTH,
        "midi_pitch": _MIDI_PITCH,
    }
    return AudioFeedbackLoss(**(settings | kwargs))


def _drop_last_loader(drop_last: bool) -> DataLoader:
    """Build a plain loader exposing the requested partial-batch policy.

    :param drop_last: Loader policy under test.
    :returns: One-row tensor loader.
    """
    return DataLoader(TensorDataset(torch.tensor([0.0])), batch_size=1, drop_last=drop_last)


def _linear_encoder(scale: float = 1.0) -> torch.nn.Module:
    """Build a deterministic encoder whose output magnitude is set by ``scale``.

    :param scale: Multiplier applied to every weight and bias.
    :returns: Encoder mapping ``(batch, signal_length)`` to ``(batch, 8)``.
    """
    torch.manual_seed(0)
    encoder = torch.nn.Linear(_SIGNAL_LENGTH, 8)
    with torch.no_grad():
        encoder.weight.mul_(scale)
        encoder.bias.mul_(scale)
    return encoder


def test_differentiable_decode_matches_the_linear_map_across_the_working_range() -> None:
    """Inside the clamp bounds the decode is exactly the ``(theta + 1) / 2`` param map.

    Agreement is what keeps the audio path's parameters identical to the ones
    ``val/param_mse`` and the dataset targets are expressed in.
    """
    # 4e-4 = 2 * _PARAM_CLAMP_EPS in theta units, clear of float32 rounding onto the bound.
    theta = torch.linspace(-1.0 + 4e-4, 1.0 - 4e-4, 401).unsqueeze(0)
    assert torch.equal(differentiable_decode(theta), (theta + 1) / 2)


def test_differentiable_decode_at_the_working_range_edges_hits_the_clamp_bounds() -> None:
    """``theta = +-1`` lands on ``[eps, 1 - eps]``, the interval the renderer accepts."""
    decoded = differentiable_decode(torch.tensor([[-1.0, 1.0]]))
    assert decoded[0, 0].item() == pytest.approx(1e-4, abs=1e-6)
    assert decoded[0, 1].item() == pytest.approx(1 - 1e-4, abs=1e-6)


def test_differentiable_decode_of_far_out_of_range_theta_stays_strictly_inside_zero_one() -> None:
    """Even a diverged estimate renders inside the open interval the renderer accepts."""
    decoded = differentiable_decode(torch.tensor([[-1e3, -5.0, 0.0, 5.0, 1e3]]))
    assert (decoded > 0.0).all()
    assert (decoded < 1.0).all()


def test_differentiable_decode_gradient_is_the_linear_slope_even_when_saturated() -> None:
    """Straight-through backward: saturated entries keep the pull-back gradient a clamp zeros."""
    theta = torch.tensor([[-5.0, 0.0, 5.0]], requires_grad=True)
    (gradient,) = torch.autograd.grad(differentiable_decode(theta).sum(), theta)
    assert torch.equal(gradient, torch.full_like(gradient, 0.5))


def test_audio_weight_below_t_min_is_zero() -> None:
    """The audio term is inactive before the feedback window opens."""
    weight = _loss().audio_weight(torch.tensor([[0.0], [0.5], [0.79]]))
    assert torch.all(weight == 0.0)


def test_audio_weight_at_final_time_equals_lambda() -> None:
    """The ramp reaches the configured weight at t=1."""
    weight = _loss(lambda_audio=0.25).audio_weight(torch.tensor([[1.0]]))
    assert torch.allclose(weight, torch.tensor([[0.25]]))


def test_grad_render_output_matches_the_documented_audio_contract() -> None:
    """The differentiable render emits finite float32 ``(batch, signal_length)`` audio in range."""
    audio = _render(torch.rand(_BATCH, _NUM_PARAMS, generator=torch.Generator().manual_seed(0)))

    assert audio.shape == (_BATCH, _SIGNAL_LENGTH)
    assert audio.dtype == torch.float32
    assert torch.isfinite(audio).all()
    assert audio.abs().max() <= 1.0


def test_grad_render_of_a_batch_matches_the_per_row_production_render_bitwise() -> None:
    """The training render reproduces the stored targets sample for sample.

    :class:`TorchSynthDataset` renders one row at a time, so every target sees
    torchsynth ``Noise`` chunk 0; the batched render broadcasts that same chunk.
    Drop the broadcast and rows past the first carry a different noise realization,
    leaving the audio loss chasing a target it can never reach.
    """
    params = torch.rand(_BATCH, _NUM_PARAMS, generator=torch.Generator().manual_seed(0))
    for index, spec in enumerate(INFERABLE_SPEC):
        if "noise" in spec.name:
            # Turn the noise path up so a mismatched realization cannot hide under the oscillators.
            params[:, index] = 0.9

    batched = _render(params)
    per_row = torch.cat(
        [
            render_torchsynth(
                params[row : row + 1],
                sample_rate=_SAMPLE_RATE,
                signal_length=_SIGNAL_LENGTH,
                midi_pitch=_MIDI_PITCH,
            )
            for row in range(_BATCH)
        ],
        dim=0,
    )

    assert torch.equal(batched, per_row)


def test_grad_render_each_output_row_depends_only_on_matching_parameter_row() -> None:
    """A row-local render cannot leak gradients across batch examples."""
    params = torch.rand(
        2, _NUM_PARAMS, generator=torch.Generator().manual_seed(11)
    ).requires_grad_()
    audio = render_torchsynth_grad(
        params,
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        midi_pitch=_MIDI_PITCH,
    )

    for output_row in range(2):
        (gradient,) = torch.autograd.grad(
            audio[output_row].square().mean(), params, retain_graph=True
        )
        assert torch.isfinite(gradient[output_row]).all()
        assert torch.count_nonzero(gradient[output_row]).item() > 0
        assert torch.equal(gradient[1 - output_row], torch.zeros_like(gradient[1 - output_row]))


def test_grad_render_of_saturated_parameters_still_backprops_nonzero_gradient() -> None:
    """Straight-through clamp: rows arriving at/over ``[0, 1]`` keep a pull-back gradient.

    A hard clamp zeroes gradient for every saturated entry, so a diverged estimate can
    never be pulled back into range by the audio loss.
    """
    params = torch.rand(1, _NUM_PARAMS, generator=torch.Generator().manual_seed(11))
    saturated = torch.arange(0, _NUM_PARAMS, 2)
    params[:, saturated] = (torch.arange(len(saturated)) % 2).float()
    params.requires_grad_()

    audio = render_torchsynth_grad(
        params,
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        midi_pitch=_MIDI_PITCH,
    )
    (gradient,) = torch.autograd.grad(audio.square().mean(), params)

    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[:, saturated]).item() > 0


def test_grad_render_leaves_the_torchsynth_module_class_unmutated_mid_render() -> None:
    """A concurrent caller must never observe a swapped process-global ``SynthModule.p``.

    Sampled from inside the render, where a monkeypatch would still be installed.
    """
    from torchsynth.module import SynthModule

    stock_p = SynthModule.p
    seen: list[object] = []

    def record(module: torch.nn.Module, *_: object) -> None:
        seen.append(SynthModule.p)

    handle = torch.nn.modules.module.register_module_forward_hook(record)
    try:
        _render(torch.rand(1, _NUM_PARAMS, generator=torch.Generator().manual_seed(3)))
    finally:
        handle.remove()

    assert seen
    assert all(observed is stock_p for observed in seen)
    assert SynthModule.p is stock_p


def test_grad_render_parameter_gradients_match_the_pinned_baseline() -> None:
    """The gradient the audio loss sees is pinned against a pre-refactor measurement."""
    params = torch.rand(
        2, _NUM_PARAMS, generator=torch.Generator().manual_seed(11)
    ).requires_grad_()
    audio = render_torchsynth_grad(
        params,
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        midi_pitch=_MIDI_PITCH,
    )

    energy = audio.square().mean()
    (gradient,) = torch.autograd.grad(energy, params)

    assert energy.item() == pytest.approx(0.07681834697723389, rel=1e-5)
    assert gradient.sum().item() == pytest.approx(1.3048763275146484, rel=1e-5)
    assert gradient.abs().sum().item() == pytest.approx(2.6795685291290283, rel=1e-5)


def test_latent_loss_with_all_zero_weights_skips_render_and_preserves_scalar_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inactive batch returns a graph-connected scalar without rendering.

    :param monkeypatch: Pytest patcher used to make any render fail.
    """

    def fail_render(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("renderer must not run")

    monkeypatch.setattr(
        "synth_setter.models.components.audio_feedback.render_torchsynth_grad", fail_render
    )
    theta = torch.zeros(_BATCH, _NUM_PARAMS, dtype=torch.float64, requires_grad=True)

    value = _loss()(
        theta, torch.zeros(_BATCH, 1), torch.empty(_BATCH, _SIGNAL_LENGTH), _linear_encoder()
    )

    assert value.shape == torch.Size([])
    assert value.device == theta.device
    assert value.dtype == theta.dtype
    assert value.requires_grad


def test_latent_loss_with_partially_zero_weights_still_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One active row keeps the batch render enabled.

    :param monkeypatch: Pytest patcher used to observe the render boundary.
    """
    render_calls = 0

    def fake_render(params: torch.Tensor, **kwargs: object) -> torch.Tensor:
        nonlocal render_calls
        render_calls += 1
        return params[:, :1].expand(-1, _SIGNAL_LENGTH)

    monkeypatch.setattr(
        "synth_setter.models.components.audio_feedback.render_torchsynth_grad", fake_render
    )
    theta = torch.zeros(_BATCH, _NUM_PARAMS, requires_grad=True)
    keep = torch.tensor([True, False, False, False])

    value = _loss()(
        theta,
        torch.full((_BATCH, 1), 0.9),
        torch.zeros(_BATCH, _SIGNAL_LENGTH),
        _linear_encoder(),
        keep=keep,
    )

    assert value.ndim == 0
    assert render_calls == 1


def test_latent_loss_with_zero_weights_still_rejects_wrong_parameter_width() -> None:
    """The render skip retains the renderer's parameter-shape boundary."""
    theta = torch.zeros(_BATCH, _NUM_PARAMS - 1, requires_grad=True)

    with pytest.raises(ValueError, match="Expected"):
        _loss()(
            theta, torch.zeros(_BATCH, 1), torch.empty(_BATCH, _SIGNAL_LENGTH), _linear_encoder()
        )


def test_latent_loss_with_zero_weights_still_rejects_non_finite_parameters() -> None:
    """The render skip cannot hide a diverged parameter estimate."""
    theta = torch.zeros(_BATCH, _NUM_PARAMS, requires_grad=True)
    with torch.no_grad():
        theta[0, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        _loss()(
            theta, torch.zeros(_BATCH, 1), torch.empty(_BATCH, _SIGNAL_LENGTH), _linear_encoder()
        )


def test_latent_loss_with_all_zero_weights_backprops_zero_theta_gradients() -> None:
    """The skipped render keeps a zero autograd path to every estimate row."""
    theta = torch.zeros(_BATCH, _NUM_PARAMS, requires_grad=True)

    value = _loss()(
        theta, torch.zeros(_BATCH, 1), torch.empty(_BATCH, _SIGNAL_LENGTH), _linear_encoder()
    )
    (gradient,) = torch.autograd.grad(value, theta)

    assert torch.equal(gradient, torch.zeros_like(theta))


def test_gradient_balance_with_all_zero_audio_weights_is_finite_zero() -> None:
    """An inactive audio term remains measurable by the training diagnostic."""
    prediction = torch.randn(_BATCH, _NUM_PARAMS, requires_grad=True)
    flow_loss = prediction.square().mean()
    audio_term = _loss()(
        prediction,
        torch.zeros(_BATCH, 1),
        torch.empty(_BATCH, _SIGNAL_LENGTH),
        _linear_encoder(),
    )

    ratio, cosine = gradient_balance(flow_loss=flow_loss, audio_term=audio_term, shared=prediction)

    assert torch.isfinite(ratio)
    assert torch.isfinite(cosine)
    assert ratio.item() == 0.0
    assert cosine.item() == 0.0


def test_latent_loss_backprops_gradient_through_the_encoder() -> None:
    """The latent distance differentiates through both the render and the encoder."""
    torch.manual_seed(0)
    target_audio = _render(torch.rand(_BATCH, _NUM_PARAMS))
    theta = (torch.rand(_BATCH, _NUM_PARAMS) * 2 - 1).requires_grad_(True)

    value = _loss().forward(
        theta, torch.full((_BATCH, 1), 0.9), target_audio, encoder=_linear_encoder()
    )
    (gradient,) = torch.autograd.grad(value, theta)

    assert value.item() > 0.0
    assert torch.isfinite(gradient).all()
    assert (gradient != 0).any()


def test_latent_loss_is_invariant_to_encoder_output_scale() -> None:
    """Cosine geometry: scaling the encoder's outputs must not move the distance."""
    torch.manual_seed(0)
    params = torch.rand(_BATCH, _NUM_PARAMS)
    target_audio = _render(params)
    theta = torch.rand(_BATCH, _NUM_PARAMS) * 2 - 1
    t = torch.full((_BATCH, 1), 0.9)

    unscaled = _loss().forward(theta, t, target_audio, encoder=_linear_encoder(scale=1.0))
    scaled = _loss().forward(theta, t, target_audio, encoder=_linear_encoder(scale=10.0))

    assert torch.allclose(unscaled, scaled, atol=1e-5)


def test_latent_loss_of_a_perfect_estimate_is_zero() -> None:
    """An estimate that decodes to the target's own parameters renders an exact match."""
    torch.manual_seed(0)
    params = torch.rand(_BATCH, _NUM_PARAMS).clamp(0.01, 0.99)
    target_audio = _render(params)

    value = _loss().forward(
        params * 2 - 1, torch.full((_BATCH, 1), 0.9), target_audio, encoder=_linear_encoder()
    )

    assert value.item() == pytest.approx(0.0, abs=1e-6)


def test_latent_loss_with_a_sequence_encoder_reduces_to_a_scalar() -> None:
    """Token-emitting encoders must still yield one distance per sample, not per token."""

    class _TokenEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(_SIGNAL_LENGTH, 24)

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            return self.linear(audio).reshape(audio.shape[0], 3, 8)

    torch.manual_seed(0)
    target_audio = _render(torch.rand(_BATCH, _NUM_PARAMS))
    theta = torch.rand(_BATCH, _NUM_PARAMS) * 2 - 1

    value = _loss().forward(
        theta, torch.full((_BATCH, 1), 0.9), target_audio, encoder=_TokenEncoder()
    )

    assert value.ndim == 0
    assert torch.isfinite(value)


def test_latent_loss_leaves_encoder_weights_and_stats_untouched() -> None:
    """The latent space is frozen: no weight gradients, no BatchNorm stat drift."""
    torch.manual_seed(0)
    encoder = torch.nn.Sequential(
        torch.nn.Linear(_SIGNAL_LENGTH, 8), torch.nn.BatchNorm1d(8), torch.nn.GELU()
    )
    encoder.train()
    batch_norm = encoder[1]
    assert isinstance(batch_norm, torch.nn.BatchNorm1d)
    assert batch_norm.running_mean is not None
    stats_before = batch_norm.running_mean.clone()
    target_audio = _render(torch.rand(_BATCH, _NUM_PARAMS))
    theta = (torch.rand(_BATCH, _NUM_PARAMS) * 2 - 1).requires_grad_(True)

    value = _loss().forward(theta, torch.full((_BATCH, 1), 0.9), target_audio, encoder=encoder)
    value.backward()

    assert theta.grad is not None and (theta.grad != 0).any()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert torch.equal(batch_norm.running_mean, stats_before)
    assert encoder.training


@pytest.mark.parametrize("lambda_audio", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_audio_loss_with_non_finite_or_non_positive_lambda_raises(lambda_audio: float) -> None:
    """Only finite positive audio weights define an enabled loss.

    :param lambda_audio: Invalid weight under test.
    """
    with pytest.raises(ValueError, match="lambda_audio"):
        _loss(lambda_audio=lambda_audio)


@pytest.mark.parametrize("lambda_audio", [torch.finfo(torch.float32).tiny, 0.03, 1.0])
def test_audio_loss_with_finite_positive_lambda_is_accepted(lambda_audio: float) -> None:
    """Every finite positive weight is valid.

    :param lambda_audio: Valid weight under test.
    """
    assert _loss(lambda_audio=lambda_audio).lambda_audio == lambda_audio


def test_gradient_balance_of_a_term_against_itself_is_unit_ratio_and_alignment() -> None:
    """Identical terms contribute identically: ratio 1, cosine 1."""
    shared = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    term = shared.square().sum()

    ratio, cosine = gradient_balance(flow_loss=term, audio_term=term, shared=shared)

    assert ratio.item() == pytest.approx(1.0)
    assert cosine.item() == pytest.approx(1.0)


def test_gradient_balance_scaled_audio_term_scales_the_ratio_only() -> None:
    """The ratio tracks relative gradient magnitude; the cosine ignores it."""
    shared = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    flow = shared.square().sum()

    ratio, cosine = gradient_balance(flow_loss=flow, audio_term=3.0 * flow, shared=shared)

    assert ratio.item() == pytest.approx(3.0)
    assert cosine.item() == pytest.approx(1.0)


def test_gradient_balance_opposed_terms_reports_negative_cosine() -> None:
    """A term pulling against the flow loss is the conflict signal we want to see."""
    shared = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    flow = shared.square().sum()

    ratio, cosine = gradient_balance(flow_loss=flow, audio_term=-flow, shared=shared)

    assert ratio.item() == pytest.approx(1.0)
    assert cosine.item() == pytest.approx(-1.0)


def test_gradient_balance_with_a_detached_flow_loss_is_finite() -> None:
    """A zero flow gradient must not divide by zero."""
    shared = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)

    ratio, cosine = gradient_balance(
        flow_loss=(shared.detach() * shared.detach()).sum() + 0.0 * shared.sum(),
        audio_term=shared.square().sum(),
        shared=shared,
    )

    assert torch.isfinite(ratio)
    assert torch.isfinite(cosine)


def test_validate_runtime_with_plain_drop_last_false_loader_raises() -> None:
    """An explicit partial-batch leaf is unsupported."""
    loader = _drop_last_loader(False)

    with pytest.raises(ValueError, match="drop_last=False"):
        validate_audio_feedback_runtime(train_dataloader=loader, compiled=False, world_size=1)


def test_validate_runtime_with_plain_drop_last_true_loader_is_accepted() -> None:
    """A plain fixed-batch loader exposes sufficient metadata."""
    loader = _drop_last_loader(True)

    validate_audio_feedback_runtime(train_dataloader=loader, compiled=False, world_size=1)


def test_validate_runtime_with_mixed_nested_loaders_raises() -> None:
    """One explicit partial-batch leaf invalidates the nested loader tree."""
    loaders = {
        "conditioned": _drop_last_loader(True),
        "reference": [_drop_last_loader(False)],
    }

    with pytest.raises(ValueError, match="drop_last=False"):
        validate_audio_feedback_runtime(train_dataloader=loaders, compiled=False, world_size=1)


def test_validate_runtime_with_all_true_nested_loaders_is_accepted() -> None:
    """Every true leaf in a list/dict loader tree satisfies the fixed-batch contract."""
    loaders = {
        "conditioned": _drop_last_loader(True),
        "reference": [_drop_last_loader(True)],
    }

    validate_audio_feedback_runtime(train_dataloader=loaders, compiled=False, world_size=1)


def test_validate_runtime_with_combined_loader_inspects_nested_leaves() -> None:
    """Lightning's CombinedLoader wrapper does not hide leaf metadata."""
    loaders = CombinedLoader(
        {
            "conditioned": _drop_last_loader(True),
            "reference": _drop_last_loader(False),
        }
    )

    with pytest.raises(ValueError, match="drop_last=False"):
        validate_audio_feedback_runtime(train_dataloader=loaders, compiled=False, world_size=1)


def test_validate_runtime_with_attribute_less_input_reports_indeterminate_metadata() -> None:
    """Missing metadata is distinct from an explicit partial-batch loader."""
    with pytest.raises(ValueError, match="could not determine"):
        validate_audio_feedback_runtime(train_dataloader=object(), compiled=False, world_size=1)


def test_validate_runtime_with_torch_compile_raises() -> None:
    """Compiling over the functional_call render graph-breaks or miscompiles."""
    with pytest.raises(ValueError, match="compile"):
        validate_audio_feedback_runtime(compiled=True, world_size=1)


def test_validate_runtime_with_multiple_devices_raises() -> None:
    """The renderer is process-local and single-device only."""
    with pytest.raises(ValueError, match="world_size"):
        validate_audio_feedback_runtime(compiled=False, world_size=2)


def test_validate_runtime_accepts_a_supported_configuration() -> None:
    """A pre-trainer compile check can omit loader metadata."""
    validate_audio_feedback_runtime(compiled=False, world_size=1)
