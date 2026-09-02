"""Tests for sketch-control conditioning in the flow-matching module."""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Literal, cast

import pytest
import torch

from synth_setter.conditioning import SketchControlSpec
from synth_setter.data.vst.shapes import NUM_SKETCH_CONTROLS
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.vst_flow_matching_module import (
    ConditioningKeepMasks,
    ControlTokenBranches,
    VSTFlowMatchingModule,
    build_guided_velocity,
    joint_cfg_velocity,
    multi_cfg_velocity,
    rk4_step,
)

_BATCH = 2
_D_MODEL = 16
_NUM_PARAMS = 6
_NUM_FRAMES = 9
_MEL_SHAPE = (2, 8, 5)

if TYPE_CHECKING:
    from synth_setter.models.components.audio_feedback import AudioFeedbackLoss


class _FlattenEncoder(torch.nn.Module):
    """Minimal mel encoder projecting the flattened spectrogram to one vector."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(_MEL_SHAPE[0] * _MEL_SHAPE[1] * _MEL_SHAPE[2], 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


def _module(
    sketch: SketchControlSpec | None,
    *,
    sketch_dropout_rate: float = 0.1,
    all_conditioning_dropout_rate: float = 0.1,
    cfg_dropout_rate: float = 0.1,
    audio_loss: torch.nn.Module | None = None,
) -> VSTFlowMatchingModule:
    """Build a tiny CPU flow module.

    :param sketch: Sketch-control spec, or ``None`` for the baseline model.
    :param sketch_dropout_rate: Independent per-control drop probability.
    :param all_conditioning_dropout_rate: Probability of dropping every conditioning stream.
    :param cfg_dropout_rate: Audio-conditioning CFG drop probability.
    :param audio_loss: Focused audio-loss implementation, or ``None``.
    :returns: Module ready for ``_train_step``.
    """
    torch.manual_seed(0)
    field = ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=_D_MODEL,
            d_token=_D_MODEL,
            num_params=_NUM_PARAMS,
            num_tokens=5,
        ),
        num_layers=2,
        d_model=_D_MODEL,
        conditioning_dim=8,
        num_heads=2,
        d_ff=_D_MODEL,
        num_tokens=5,
        # "none" is a valid runtime value model/vst_flow.yaml ships; the
        # constructor's Literal annotation omits it.
        pe_type=cast(Literal["initial", "layerwise"], "none"),
        time_encoding="scalar",
        learn_projection=True,
    )
    return VSTFlowMatchingModule(
        encoder=_FlattenEncoder(),
        vector_field=field,
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_NUM_PARAMS,
        sketch_controls=sketch,
        sketch_dropout_rate=sketch_dropout_rate,
        all_conditioning_dropout_rate=all_conditioning_dropout_rate,
        cfg_dropout_rate=cfg_dropout_rate,
        audio_loss=cast("AudioFeedbackLoss | None", audio_loss),
        validation_sample_steps=2,
    )


class _KeepCountAudioLoss(torch.nn.Module):
    """Focused fake returning the number of identity-conditioned rows."""

    def forward(
        self,
        theta_hat: torch.Tensor,
        t: torch.Tensor,
        target_audio: torch.Tensor,
        keep: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the production keep input to an observable scalar.

        :param theta_hat: Estimated parameters.
        :param t: Flow time.
        :param target_audio: Target audio.
        :param keep: Positive identity keep state.
        :returns: Number of retained rows in ``theta_hat`` dtype.
        """
        del t, target_audio
        return keep.to(theta_hat.dtype).sum()


def _constant_field(
    value: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a time-independent field for exact CFG algebra checks.

    :param value: Constant velocity value.
    :returns: Field matching the sampler's state-and-time signature.
    """

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return torch.full_like(x, value)

    return field


def _batch(*, with_sketch: bool) -> dict[str, torch.Tensor]:
    """Draw a deterministic synthetic mel batch.

    :param with_sketch: Whether the batch carries a ``sketch_ctrl`` tensor.
    :returns: Model batch.
    """
    generator = torch.Generator().manual_seed(7)
    batch = {
        "mel": torch.randn((_BATCH, *_MEL_SHAPE), generator=generator),
        "params": torch.rand((_BATCH, _NUM_PARAMS), generator=generator) * 2 - 1,
        "noise": torch.randn((_BATCH, _NUM_PARAMS), generator=generator),
    }
    if with_sketch:
        batch["sketch_ctrl"] = torch.rand(
            (_BATCH, NUM_SKETCH_CONTROLS, _NUM_FRAMES), generator=generator
        )
    return batch


class _BranchField(torch.nn.Module):
    """Field exposing content and control-token branch selection in its value."""

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor | None,
        *,
        control_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the selected branch into a constant velocity.

        :param x: Parameter state whose shape the velocity follows.
        :param t: Flow time.
        :param conditioning: Content conditioning, or ``None`` on the unconditional branch.
        :param control_tokens: Branch-specific control tokens.
        :returns: Constant velocity identifying both branch inputs.
        """
        del t
        value = control_tokens.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        if conditioning is not None:
            value = value + 10.0
        return value.expand_as(x)


def test_build_guided_velocity_applies_separate_sketch_and_content_strengths() -> None:
    """Sketch CFG isolates full controls before adding content conditioning."""
    x = torch.zeros(_BATCH, _NUM_PARAMS)
    guided_velocity = build_guided_velocity(
        _BranchField(),
        torch.ones(_BATCH, 8),
        cfg_strength=4.0,
        sketch_cfg_strength=3.0,
        control_tokens=ControlTokenBranches(
            conditional=torch.full((_BATCH, 2, _D_MODEL), 2.0),
            unconditional=torch.full((_BATCH, 2, _D_MODEL), 3.0),
        ),
    )

    guided = guided_velocity(x, torch.zeros(_BATCH, 1))

    torch.testing.assert_close(guided, torch.full_like(x, 40.0))


def test_build_guided_velocity_defaults_sketch_strength_to_content_strength() -> None:
    """Omitted sketch scale applies the content scale to both guidance deltas."""
    x = torch.zeros(_BATCH, _NUM_PARAMS)
    guided_velocity = build_guided_velocity(
        _BranchField(),
        torch.ones(_BATCH, 8),
        cfg_strength=4.0,
        control_tokens=ControlTokenBranches(
            conditional=torch.full((_BATCH, 2, _D_MODEL), 2.0),
            unconditional=torch.full((_BATCH, 2, _D_MODEL), 3.0),
        ),
    )

    guided = guided_velocity(x, torch.zeros(_BATCH, 1))

    torch.testing.assert_close(guided, torch.full_like(x, 39.0))


def test_build_guided_velocity_accepts_zero_sketch_strength() -> None:
    """Zero sketch guidance leaves the content delta independently active."""
    x = torch.zeros(_BATCH, _NUM_PARAMS)
    guided_velocity = build_guided_velocity(
        _BranchField(),
        torch.ones(_BATCH, 8),
        cfg_strength=4.0,
        sketch_cfg_strength=0.0,
        control_tokens=ControlTokenBranches(
            conditional=torch.full((_BATCH, 2, _D_MODEL), 2.0),
            unconditional=torch.full((_BATCH, 2, _D_MODEL), 3.0),
        ),
    )

    guided = guided_velocity(x, torch.zeros(_BATCH, 1))

    torch.testing.assert_close(guided, torch.full_like(x, 43.0))


def test_sample_routes_separate_sketch_strength_to_guided_velocity() -> None:
    """Sampling keeps sketch guidance independent from content guidance."""
    module = _module(None)
    module.vector_field = _BranchField()
    control_tokens = ControlTokenBranches(
        conditional=torch.full((_BATCH, 2, _D_MODEL), 2.0),
        unconditional=torch.full((_BATCH, 2, _D_MODEL), 3.0),
    )

    sample = module._sample(  # noqa: SLF001
        torch.zeros(_BATCH, *_MEL_SHAPE),
        torch.zeros(_BATCH, _NUM_PARAMS),
        steps=1,
        cfg_strength=4.0,
        sketch_cfg_strength=3.0,
        control_tokens=control_tokens,
    )

    torch.testing.assert_close(sample, torch.full_like(sample, 40.0))


def test_sample_without_content_uses_explicit_sketch_strength() -> None:
    """Sketch-only sampling ignores the separate content strength."""
    module = _module(None)
    module.vector_field = _BranchField()
    control_tokens = ControlTokenBranches(
        conditional=torch.full((_BATCH, 2, _D_MODEL), 2.0),
        unconditional=torch.full((_BATCH, 2, _D_MODEL), 3.0),
    )

    sample = module._sample(  # noqa: SLF001
        None,
        torch.zeros(_BATCH, _NUM_PARAMS),
        steps=1,
        cfg_strength=4.0,
        sketch_cfg_strength=3.0,
        control_tokens=control_tokens,
    )

    torch.testing.assert_close(sample, torch.zeros_like(sample))


def test_multi_cfg_velocity_applies_three_branch_algebra() -> None:
    """Multi-CFG scales sketch and content deltas independently."""
    guided_velocity = multi_cfg_velocity(
        _constant_field(2.0),
        _constant_field(5.0),
        _constant_field(11.0),
        sketch_cfg_strength=3.0,
        content_cfg_strength=4.0,
    )

    guided = guided_velocity(torch.zeros(2, 3), torch.zeros(2, 1))

    torch.testing.assert_close(guided, torch.full((2, 3), 35.0))


def test_multi_cfg_equal_strength_matches_joint_cfg() -> None:
    """Equal sketch and content scales preserve joint-CFG output."""
    unconditional = _constant_field(2.0)
    content_sketch = _constant_field(11.0)
    x = torch.zeros(2, 3)
    t = torch.zeros(2, 1)

    multi_guided = multi_cfg_velocity(
        unconditional,
        _constant_field(5.0),
        content_sketch,
        sketch_cfg_strength=4.0,
        content_cfg_strength=4.0,
    )
    joint_guided = joint_cfg_velocity(content_sketch, unconditional, cfg_strength=4.0)

    torch.testing.assert_close(multi_guided(x, t), joint_guided(x, t))


def test_joint_cfg_velocity_applies_two_branch_algebra() -> None:
    """Joint CFG combines one conditional and one unconditional velocity."""

    def conditional(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return torch.full_like(x, 5.0)

    def unconditional(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return torch.full_like(x, 2.0)

    guided_velocity = joint_cfg_velocity(conditional, unconditional, cfg_strength=4.0)

    guided = guided_velocity(torch.zeros(2, 3), torch.zeros(2, 1))

    torch.testing.assert_close(guided, torch.full((2, 3), 14.0))


def test_rk4_step_integrates_two_argument_time_field() -> None:
    """Pure RK4 advances a two-argument field without conditioning knowledge."""
    x = torch.ones(1, 1)

    result = rk4_step(lambda state, time: state, x, torch.zeros(1, 1), 0.1)

    torch.testing.assert_close(result, torch.tensor([[1.1051708]]))


def test_sample_batch_same_explicit_noise_returns_identical_predictions() -> None:
    """Explicit noise makes repeated sketch sampling deterministic."""
    model = _module(SketchControlSpec(num_frames=_NUM_FRAMES))
    batch = _batch(with_sketch=True)
    noise = batch["noise"].clone()

    first = model.sample_batch(
        batch,
        noise=noise,
        content_cfg_strength=2.0,
        sketch_cfg_strength=1.0,
        sample_steps=2,
    )
    second = model.sample_batch(
        batch,
        noise=noise,
        content_cfg_strength=2.0,
        sketch_cfg_strength=1.0,
        sample_steps=2,
    )

    assert first.shape == (_BATCH, _NUM_PARAMS)
    assert first.dtype is torch.float32
    assert torch.equal(first, second)
    assert not first.requires_grad


def test_sample_batch_guidance_and_sketch_controls_change_fixed_noise_output() -> None:
    """The public sampler consumes both CFG axes and sketch-control values."""
    model = _module(SketchControlSpec(num_frames=_NUM_FRAMES))
    model.vector_field = _BranchField()
    assert model.sketch_tokens is not None
    with torch.no_grad():
        for projection in model.sketch_tokens.projections.children():
            assert isinstance(projection, torch.nn.Linear)
            projection.weight.fill_(1.0)
    batch = _batch(with_sketch=True)
    noise = torch.zeros_like(batch["noise"])
    baseline = model.sample_batch(
        batch,
        noise=noise,
        content_cfg_strength=0.0,
        sketch_cfg_strength=0.0,
        sample_steps=1,
    )
    content_guided = model.sample_batch(
        batch,
        noise=noise,
        content_cfg_strength=1.0,
        sketch_cfg_strength=0.0,
        sample_steps=1,
    )
    sketch_guided = model.sample_batch(
        batch,
        noise=noise,
        content_cfg_strength=0.0,
        sketch_cfg_strength=1.0,
        sample_steps=1,
    )
    changed_batch = dict(batch)
    changed_batch["sketch_ctrl"] = torch.zeros_like(batch["sketch_ctrl"])
    changed_controls = model.sample_batch(
        changed_batch,
        noise=noise,
        content_cfg_strength=0.0,
        sketch_cfg_strength=1.0,
        sample_steps=1,
    )

    assert not torch.equal(baseline, content_guided)
    assert not torch.equal(baseline, sketch_guided)
    assert not torch.equal(sketch_guided, changed_controls)


def test_sample_batch_wrong_noise_shape_raises() -> None:
    """Explicit noise must provide one parameter vector per input row."""
    model = _module(SketchControlSpec(num_frames=_NUM_FRAMES))
    batch = _batch(with_sketch=True)

    with pytest.raises(ValueError, match="noise shape"):
        model.sample_batch(
            batch,
            noise=torch.zeros((_BATCH, _NUM_PARAMS + 1)),
            content_cfg_strength=2.0,
            sketch_cfg_strength=1.0,
            sample_steps=2,
        )


def test_sample_batch_nonfinite_noise_raises() -> None:
    """Explicit noise must contain finite flow states."""
    model = _module(SketchControlSpec(num_frames=_NUM_FRAMES))
    batch = _batch(with_sketch=True)
    noise = batch["noise"].clone()
    noise[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        model.sample_batch(
            batch,
            noise=noise,
            content_cfg_strength=2.0,
            sketch_cfg_strength=1.0,
        )


def test_sample_batch_negative_guidance_raises() -> None:
    """Guidance cannot reverse a conditioning direction."""
    model = _module(SketchControlSpec(num_frames=_NUM_FRAMES))
    batch = _batch(with_sketch=True)

    with pytest.raises(ValueError, match="sketch_cfg_strength"):
        model.sample_batch(
            batch,
            noise=batch["noise"],
            content_cfg_strength=2.0,
            sketch_cfg_strength=-1.0,
        )


def test_sample_batch_nonpositive_steps_raises() -> None:
    """Sampling requires at least one integration step."""
    model = _module(SketchControlSpec(num_frames=_NUM_FRAMES))
    batch = _batch(with_sketch=True)

    with pytest.raises(ValueError, match="positive integer"):
        model.sample_batch(
            batch,
            noise=batch["noise"],
            content_cfg_strength=2.0,
            sketch_cfg_strength=1.0,
            sample_steps=0,
        )


def test_train_step_with_sketch_batch_produces_finite_loss() -> None:
    """Sketch-configured training consumes ``sketch_ctrl`` and stays finite."""
    module = _module(SketchControlSpec(num_frames=_NUM_FRAMES))

    loss = module._train_step(_batch(with_sketch=True)).loss  # noqa: SLF001

    assert torch.isfinite(loss)


def test_train_step_none_spec_ignores_sketch_free_batch() -> None:
    """The default configuration trains on batches without ``sketch_ctrl``."""
    module = _module(None)

    loss = module._train_step(_batch(with_sketch=False)).loss  # noqa: SLF001

    assert module.sketch_tokens is None
    assert torch.isfinite(loss)


def test_train_step_none_spec_matches_loss_before_sketch_support() -> None:
    """Verify sketch-disabled training preserves the baseline RNG stream and field inputs.

    Equality against the reference recipe below on a fixed seed pins that the
    ``None`` path draws no extra randomness and passes no extra field inputs.
    """
    module = _module(None)
    batch = _batch(with_sketch=False)

    torch.manual_seed(11)
    loss = module._train_step(batch).loss  # noqa: SLF001

    torch.manual_seed(11)
    field = cast(ApproxEquivTransformer, module.vector_field)
    z, _keep = field.apply_dropout(
        module.encoder(batch["mel"]), module.hparams["cfg_dropout_rate"]
    )
    t = torch.rand(_BATCH, 1)
    x_t = batch["noise"] * (1 - t) + batch["params"] * t
    target = batch["params"] - batch["noise"]
    prediction = field(x_t, t, z)
    expected = (prediction - target).square().mean(dim=-1).mean()

    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize(
    (
        "cfg_dropout_rate",
        "sketch_dropout_rate",
        "all_conditioning_dropout_rate",
        "expected_content",
        "expected_sketch_groups",
    ),
    [
        (0.0, 0.0, 0.0, True, True),
        (0.0, 1.0, 0.0, True, False),
        (1.0, 0.0, 0.0, False, True),
        (0.0, 0.0, 1.0, False, False),
    ],
)
def test_conditioning_keep_masks_unit_rates_match_training_truth_table(
    cfg_dropout_rate: float,
    sketch_dropout_rate: float,
    all_conditioning_dropout_rate: float,
    expected_content: bool,
    expected_sketch_groups: bool,
) -> None:
    """Unit-rate policies produce full, content-only, sketch-only, and unconditional rows.

    :param cfg_dropout_rate: Content drop probability.
    :param sketch_dropout_rate: Per-sketch-group drop probability.
    :param all_conditioning_dropout_rate: Global all-conditioning drop probability.
    :param expected_content: Expected content keep state.
    :param expected_sketch_groups: Expected keep state for every sketch group.
    """
    module = _module(
        SketchControlSpec(num_frames=_NUM_FRAMES),
        sketch_dropout_rate=sketch_dropout_rate,
        all_conditioning_dropout_rate=all_conditioning_dropout_rate,
        cfg_dropout_rate=cfg_dropout_rate,
    )

    keep = module._sample_conditioning_keep_masks(_BATCH, torch.device("cpu"))  # noqa: SLF001

    assert torch.equal(keep.content, torch.full((_BATCH,), expected_content))
    assert torch.equal(
        keep.sketch_groups,
        torch.full((_BATCH, 3), expected_sketch_groups),
    )


def test_train_step_content_drop_preserves_sketch_only_state() -> None:
    """Content CFG dropout does not erase independently retained sketch groups."""
    module = _module(
        SketchControlSpec(num_frames=_NUM_FRAMES),
        sketch_dropout_rate=0.0,
        all_conditioning_dropout_rate=0.0,
        cfg_dropout_rate=1.0,
    )

    outputs = module._train_step(_batch(with_sketch=True))  # noqa: SLF001

    assert not outputs.conditioning_keep.content.any()
    assert outputs.conditioning_keep.sketch_groups.all()


def test_conditioning_keep_masks_identity_keeps_any_present_stream() -> None:
    """Identity survives whenever content or at least one sketch group is retained."""
    keep = ConditioningKeepMasks(
        content=torch.tensor([True, True, False, False]),
        sketch_groups=torch.tensor(
            [
                [True, True, True],
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ]
        ),
    )

    assert torch.equal(keep.identity_keep, torch.tensor([True, True, True, False]))


def test_audio_feedback_sketch_only_rows_remain_in_identity_loss() -> None:
    """Audio feedback retains rows with dropped content and any retained sketch group."""
    module = _module(
        SketchControlSpec(num_frames=_NUM_FRAMES),
        sketch_dropout_rate=0.0,
        all_conditioning_dropout_rate=0.0,
        cfg_dropout_rate=1.0,
        audio_loss=_KeepCountAudioLoss(),
    )
    batch = _batch(with_sketch=True)
    batch["audio"] = torch.zeros(_BATCH, 1)

    audio_term = module._train_step(batch).audio_term  # noqa: SLF001

    assert audio_term is not None
    torch.testing.assert_close(audio_term, torch.tensor(float(_BATCH)))


def test_validation_step_with_sketch_runs_cfg_sampling() -> None:
    """CFG sampling consumes undropped sketch tokens on the conditional branch."""
    module = _module(SketchControlSpec(num_frames=_NUM_FRAMES))

    out = module.validation_step(_batch(with_sketch=True), 0)

    assert torch.isfinite(out["param_mse"])
    assert out["preds"].shape == (_BATCH, _NUM_PARAMS)


@pytest.mark.slow
def test_sketch_conditioned_training_fixed_batch_lowers_loss_and_updates_projections() -> None:
    """A fixed sketch batch trains the zero-init control projections."""
    module = _module(
        SketchControlSpec(num_frames=_NUM_FRAMES, num_control_tokens=2),
        sketch_dropout_rate=0.0,
        all_conditioning_dropout_rate=0.0,
        cfg_dropout_rate=0.0,
    )
    batch = _batch(with_sketch=True)
    assert module.sketch_tokens is not None
    projections = module.sketch_tokens.projections.values()
    assert all(
        torch.count_nonzero(cast(torch.nn.Linear, projection).weight) == 0
        for projection in projections
    )

    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
    torch.manual_seed(11)
    initial_loss = module._train_step(batch).loss.item()  # noqa: SLF001

    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = module._train_step(batch).loss  # noqa: SLF001
        loss.backward()
        optimizer.step()

    torch.manual_seed(11)
    final_loss = module._train_step(batch).loss.item()  # noqa: SLF001

    assert final_loss < 0.01
    assert final_loss < initial_loss / 100
    assert all(
        torch.count_nonzero(cast(torch.nn.Linear, projection).weight) > 0
        for projection in module.sketch_tokens.projections.values()
    )
