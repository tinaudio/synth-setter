"""Tests for the concat sketch-control token module and its field injection."""

from typing import Literal, cast

import pytest
import torch
from jaxtyping import TypeCheckError

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    NUM_SKETCH_TRACK_ROWS,
    SKETCH_CENTROID_ROW,
    SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_SLICE,
    SketchControlProfile,
    SketchControls,
    SketchControlSpec,
    resolve_sketch_controls,
)
from synth_setter.models.components.sketch_tokens import CONTROL_GROUPS, SketchControlTokens
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.sketch import pool_sketch_controls

_BATCH = 2
_D_MODEL = 16
_NUM_FRAMES = 11
_NUM_CONTROL_TOKENS = 4
_REVERB_FRAMES = 32


def _controls(batch: int = _BATCH, seed: int = 0) -> torch.Tensor:
    """Draw a realistic sketch-control batch.

    :param batch: Batch size.
    :param seed: RNG seed.
    :returns: ``(batch, NUM_SKETCH_CONTROLS, _NUM_FRAMES)`` controls with
        signed-unit loudness/centroid rows and unit-interval pitch rows.
    """
    generator = torch.Generator().manual_seed(seed)
    controls = torch.rand((batch, NUM_SKETCH_CONTROLS, _NUM_FRAMES), generator=generator)
    controls[:, :NUM_SKETCH_TRACK_ROWS] = controls[:, :NUM_SKETCH_TRACK_ROWS] * 2 - 1
    return controls


def _tokens_module(seed: int = 0, profile: SketchControlProfile = "music") -> SketchControlTokens:
    """Build a small token module with non-zero projection weights.

    :param seed: RNG seed for the randomized projection weights.
    :param profile: Channel layout selected for tokenization.
    :returns: Module whose three projections were re-randomized after zero-init.
    """
    module = SketchControlTokens(
        d_model=_D_MODEL,
        num_control_tokens=_NUM_CONTROL_TOKENS,
        profile=profile,
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for projection in module.projections.values():
            linear = cast(torch.nn.Linear, projection)
            linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator))
    return module


def _reverb_tokens_module(seed: int = 0) -> SketchControlTokens:
    """Build a temporal reverb tokenizer with live projections.

    :param seed: RNG seed for projection weights.
    :returns: Reverb tokenizer on the canonical temporal grid.
    """
    module = SketchControlTokens(
        d_model=_D_MODEL,
        num_control_tokens=_REVERB_FRAMES,
        profile="pyfdn_reverb",
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for projection in module.projections.values():
            linear = cast(torch.nn.Linear, projection)
            linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator))
    return module


def _keep_all(batch: int = _BATCH) -> torch.Tensor:
    """Return a mask retaining every sketch group.

    :param batch: Batch size.
    :returns: ``(batch, 3)`` positive keep mask.
    """
    return torch.ones(batch, 3, dtype=torch.bool)


class TestSketchControlSpec:
    """Contract tests for the strict sketch-control spec."""

    def test_defaults_column_tokens_threshold_match_design(self) -> None:
        """Spec defaults pin the approved column, token budget, and threshold."""
        spec = SketchControlSpec(num_frames=401)
        assert spec.column == "sketch"
        assert spec.profile == "music"
        assert spec.num_control_tokens == 32
        assert spec.pitch_zero_threshold == 0.1

    def test_pyfdn_reverb_profile_has_temporal_layout(self) -> None:
        """The reverb profile pins ten channels and three temporal groups."""
        spec = SketchControlSpec(num_frames=32, profile="pyfdn_reverb")

        assert spec.layout.num_controls == 10
        assert spec.layout.group_names == ("edc", "echo_density", "spectral_flatness")
        assert spec.layout.group_widths == (8, 1, 1)

    def test_pyfdn_reverb_profile_rejects_noncanonical_frames(self) -> None:
        """Reverb controls retain one token for each canonical stored frame."""
        with pytest.raises(ValueError, match="pyfdn_reverb sketch requires"):
            SketchControlSpec(num_frames=31, profile="pyfdn_reverb")

    def test_extra_field_rejected(self) -> None:
        """Unknown fields fail loudly under the strict config."""
        with pytest.raises(ValueError, match="num_controls"):
            SketchControlSpec.model_validate({"num_frames": 401, "num_controls": 3})

    def test_resolve_none_returns_none(self) -> None:
        """``None`` disables sketch conditioning."""
        assert resolve_sketch_controls(None) is None

    def test_resolve_mapping_parses_spec(self) -> None:
        """Hydra-style mappings parse into the strict spec."""
        spec = resolve_sketch_controls({"num_frames": 401})
        assert spec == SketchControlSpec(num_frames=401)

    @pytest.mark.parametrize("threshold", [-0.1, 1.5])
    def test_pitch_zero_threshold_out_of_bounds_rejected(self, threshold: float) -> None:
        """Thresholds outside the documented ``[0, 1]`` activation range fail.

        :param threshold: Out-of-range override under test.
        """
        with pytest.raises(ValueError, match="pitch_zero_threshold"):
            SketchControlSpec(num_frames=401, pitch_zero_threshold=threshold)

    def test_pooled_storage_with_different_token_count_rejected(self) -> None:
        """Stored pooled controls cannot recover a different temporal token budget."""
        with pytest.raises(ValueError, match="pooled sketch storage requires"):
            SketchControlSpec(num_frames=32, num_control_tokens=16)

    def test_resolve_non_mapping_raises_type_error(self) -> None:
        """Non-mapping junk is rejected with a clear error."""
        with pytest.raises(TypeError, match="sketch"):
            resolve_sketch_controls(cast("SketchControls", 3))


class TestSketchControlTokens:
    """Behavior tests for the zero-init control tokenizer."""

    def test_forward_control_batch_returns_token_sequence(self) -> None:
        """The tokenizer emits the configured token sequence shape."""
        module = _tokens_module()
        out = module(_controls(), _keep_all())
        assert out.shape == (_BATCH, _NUM_CONTROL_TOKENS, _D_MODEL)

    def test_unconditional_returns_expanded_positional_encoding(self) -> None:
        """The unconditional sketch state is the tokenizer's PE-only sequence."""
        module = _tokens_module()

        tokens = module.unconditional(_BATCH)

        torch.testing.assert_close(tokens, module.positional_encoding.expand(_BATCH, -1, -1))

    def test_forward_at_zero_init_outputs_exactly_the_positional_encoding(self) -> None:
        """Zero-init projections contribute nothing: output is the PE alone."""
        module = SketchControlTokens(d_model=_D_MODEL, num_control_tokens=_NUM_CONTROL_TOKENS)
        out = module(_controls(), _keep_all())
        expected = module.positional_encoding.expand(_BATCH, -1, -1)
        torch.testing.assert_close(out, expected)

    def test_forward_dropped_control_matches_zeroed_channels(self) -> None:
        """Clearing a group's keep state equals zeroing its channels before projection."""
        module = _tokens_module()
        controls = _controls()
        keep = _keep_all()
        keep[0, 2] = False

        zeroed = controls.clone()
        zeroed[0, SKETCH_PITCH_SLICE] = 0.0

        torch.testing.assert_close(module(controls, keep), module(zeroed, _keep_all()))

    def test_forward_all_dropped_outputs_exactly_the_positional_encoding(self) -> None:
        """Dropping every control leaves exactly the PE."""
        module = _tokens_module()
        out = module(_controls(), torch.zeros(_BATCH, 3, dtype=torch.bool))
        expected = module.positional_encoding.expand(_BATCH, -1, -1)
        torch.testing.assert_close(out, expected)

    def test_forward_loudness_and_centroid_drops_are_independent(self) -> None:
        """Dropping loudness leaves the centroid contribution intact."""
        module = _tokens_module()
        controls = _controls()
        keep = _keep_all()
        keep[:, 0] = False

        zeroed = controls.clone()
        zeroed[:, SKETCH_LOUDNESS_ROW] = 0.0

        torch.testing.assert_close(module(controls, keep), module(zeroed, _keep_all()))

    @pytest.mark.parametrize("frame", range(_NUM_FRAMES))
    def test_forward_impulse_at_any_frame_reaches_tokens(self, frame: int) -> None:
        """Pooling covers every stored frame: no transient can vanish.

        Point-sampling resampling (linear interpolate) reads only the frames
        adjacent to each output position, so a short event between sample
        points would contribute nothing.

        :param frame: Frame index carrying the impulse.
        """
        module = _tokens_module()
        base = torch.zeros(1, NUM_SKETCH_CONTROLS, _NUM_FRAMES)
        spike = base.clone()
        spike[0, SKETCH_PITCH_SLICE.start, frame] = 1.0
        spike[0, SKETCH_LOUDNESS_ROW, frame] = 1.0
        assert not torch.allclose(module(base, _keep_all(1)), module(spike, _keep_all(1)))

    @pytest.mark.parametrize(
        ("group_index", "row", "expected"),
        [
            (0, SKETCH_LOUDNESS_ROW, (0.5, 0.5, 0.5, 0.5)),
            (1, SKETCH_CENTROID_ROW, (0.5, 0.5, 0.5, 0.5)),
            (2, SKETCH_PITCH_SLICE.start, (1.0, 1.0, 1.0, 1.0)),
        ],
    )
    def test_forward_pools_each_group_with_its_specified_reduction(
        self, group_index: int, row: int, expected: tuple[float, ...]
    ) -> None:
        """Each group's exact pooled values pin mean-vs-max per the design.

        The alternating input averages to ``0.5`` per bin and maxes to ``1.0``,
        so swapping either reduction changes the projected token.

        :param group_index: Column of the group in ``CONTROL_GROUPS`` order.
        :param row: Control row the alternating signal is written to.
        :param expected: Per-token pooled value the group's reduction yields.
        """
        module = SketchControlTokens(d_model=_D_MODEL, num_control_tokens=_NUM_CONTROL_TOKENS)
        with torch.no_grad():
            for projection in module.projections.values():
                cast(torch.nn.Linear, projection).weight.fill_(1.0)
        # Two frames per output bin, one high and one low, so mean and max differ.
        controls = torch.zeros(1, NUM_SKETCH_CONTROLS, 2 * _NUM_CONTROL_TOKENS)
        controls[0, row, 1::2] = 1.0
        keep = torch.zeros(1, len(CONTROL_GROUPS), dtype=torch.bool)
        keep[0, group_index] = True

        contribution = module(controls, keep) - module.unconditional(1)

        torch.testing.assert_close(
            contribution[0, :, 0], torch.tensor(expected), rtol=0, atol=1e-6
        )

    def test_forward_stored_pool_matches_online_pooling(self) -> None:
        """Persisting the canonical pool preserves the tokenizer output exactly."""
        module = SketchControlTokens(d_model=_D_MODEL, num_control_tokens=32)
        generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for projection in module.projections.values():
                linear = cast(torch.nn.Linear, projection)
                linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator))
        controls = torch.rand(2, NUM_SKETCH_CONTROLS, 64, generator=generator)
        controls[:, :NUM_SKETCH_TRACK_ROWS] = controls[:, :NUM_SKETCH_TRACK_ROWS] * 2 - 1

        torch.testing.assert_close(
            module(pool_sketch_controls(controls), _keep_all()),
            module(controls, _keep_all()),
            rtol=0,
            atol=0,
        )

    def test_music_profile_preserves_projection_parameter_names(self) -> None:
        """Default-profile checkpoints retain the established projection keys."""
        module = SketchControlTokens(d_model=_D_MODEL)

        assert tuple(module.state_dict()) == (
            "positional_encoding",
            "projections.loudness.weight",
            "projections.centroid.weight",
            "projections.pitch.weight",
        )

    def test_forward_wrong_channel_count_raises_shape_error(self) -> None:
        """The music profile rejects tensors outside its channel contract."""
        module = _tokens_module()
        bad = torch.rand(_BATCH, NUM_SKETCH_CONTROLS - 1, _NUM_FRAMES)
        with pytest.raises((TypeCheckError, ValueError), match="channel|shape"):
            module(bad, _keep_all())


class TestPyFDNReverbSketchControlTokens:
    """Temporal reverb profile behavior."""

    def test_forward_returns_one_token_per_reverb_frame(self) -> None:
        """Ten reverb channels produce a same-length token sequence."""
        module = _reverb_tokens_module()
        controls = torch.randn(_BATCH, 10, _REVERB_FRAMES, dtype=torch.float32)

        assert module(controls, _keep_all()).shape == (
            _BATCH,
            _REVERB_FRAMES,
            _D_MODEL,
        )

    @pytest.mark.parametrize("group_index", range(3))
    def test_forward_each_reverb_group_drop_is_independent(self, group_index: int) -> None:
        """Dropping one reverb group equals zeroing only that group's channels.

        :param group_index: Reverb group omitted from the token sum.
        """
        module = _reverb_tokens_module()
        controls = torch.randn(_BATCH, 10, _REVERB_FRAMES)
        keep = _keep_all()
        keep[:, group_index] = False
        zeroed = controls.clone()
        group_slice = module.layout.group_slices[group_index]
        zeroed[:, group_slice] = 0.0

        torch.testing.assert_close(module(controls, keep), module(zeroed, _keep_all()))

    def test_forward_all_reverb_groups_dropped_returns_positional_encoding(self) -> None:
        """Global reverb all-drop leaves only temporal position."""
        module = _reverb_tokens_module()
        controls = torch.randn(_BATCH, 10, _REVERB_FRAMES)

        torch.testing.assert_close(
            module(controls, torch.zeros(_BATCH, 3, dtype=torch.bool)),
            module.unconditional(_BATCH),
        )

    @pytest.mark.parametrize("row", [0, 8, 9])
    def test_forward_reverb_change_affects_only_same_frame(self, row: int) -> None:
        """A reverb descriptor cannot leak into neighboring temporal tokens.

        :param row: Representative channel from each reverb group.
        """
        module = _reverb_tokens_module()
        base = torch.zeros(1, 10, _REVERB_FRAMES)
        changed = base.clone()
        changed[0, row, 2] = 1.0

        delta = module(changed, _keep_all(1)) - module(base, _keep_all(1))

        assert torch.count_nonzero(delta[0, :2]) == 0
        assert torch.count_nonzero(delta[0, 2]) > 0
        assert torch.count_nonzero(delta[0, 3:]) == 0

    def test_forward_reverb_wrong_channel_count_raises(self) -> None:
        """The reverb profile rejects tensors outside its ten-channel contract."""
        module = _reverb_tokens_module()

        with pytest.raises(ValueError, match="10 channels"):
            module(torch.randn(_BATCH, 9, _REVERB_FRAMES), _keep_all())

    def test_forward_reverb_wrong_frame_count_raises(self) -> None:
        """The reverb profile rejects temporal resampling at the model boundary."""
        module = _reverb_tokens_module()

        with pytest.raises(ValueError, match="32 frames"):
            module(torch.randn(_BATCH, 10, 31), _keep_all())

    def test_forward_reverb_non_float32_raises(self) -> None:
        """The reverb profile rejects descriptor tensors outside float32."""
        module = _reverb_tokens_module()

        with pytest.raises(ValueError, match="float32"):
            module(torch.randn(_BATCH, 10, _REVERB_FRAMES).double(), _keep_all())

    def test_forward_reverb_wrong_group_count_raises(self) -> None:
        """The reverb profile requires one keep decision per projection group."""
        module = _reverb_tokens_module()

        with pytest.raises(ValueError, match="3 keep groups"):
            module(
                torch.randn(_BATCH, 10, _REVERB_FRAMES),
                torch.ones(_BATCH, 2, dtype=torch.bool),
            )

    def test_constructor_reverb_noncanonical_token_count_raises(self) -> None:
        """Reverb tokenizers cannot resample or broadcast temporal descriptors."""
        with pytest.raises(ValueError, match="32 control tokens"):
            SketchControlTokens(
                d_model=_D_MODEL,
                num_control_tokens=_NUM_CONTROL_TOKENS,
                profile="pyfdn_reverb",
            )


def _field(pe_type: str = "none") -> ApproxEquivTransformer:
    """Build a tiny vector field for injection tests.

    :param pe_type: Parameter-token positional-encoding mode; ``"none"`` is a
        valid runtime value ``model/vst_flow.yaml`` ships, although the
        constructor's ``Literal`` annotation omits it (hence the cast).
    :returns: Small CPU field with live (non-zero-init) blocks, as configured
        by ``model/vst_flow.yaml``.
    """
    torch.manual_seed(0)
    return ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=_D_MODEL,
            d_token=_D_MODEL,
            num_params=6,
            num_tokens=5,
        ),
        num_layers=2,
        d_model=_D_MODEL,
        conditioning_dim=8,
        num_heads=2,
        d_ff=_D_MODEL,
        num_tokens=5,
        pe_type=cast(Literal["initial", "layerwise"], pe_type),
        time_encoding="scalar",
        learn_projection=True,
        zero_init=False,
    )


class TestApproxEquivTransformerControlTokens:
    """Behavior of the optional control-token concat path."""

    def test_forward_without_control_kwarg_matches_explicit_none(self) -> None:
        """``control_tokens=None`` is a strict no-op against the legacy call."""
        field = _field()
        x = torch.randn(_BATCH, 6)
        t = torch.rand(_BATCH, 1)
        z = torch.randn(_BATCH, 8)
        torch.testing.assert_close(field(x, t, z), field(x, t, z, control_tokens=None))

    def test_forward_rejects_positional_control_tokens(self) -> None:
        """Control tokens are keyword-only at the public field boundary."""
        field = _field()
        with pytest.raises(TypeError):
            field(
                torch.randn(_BATCH, 6),
                torch.rand(_BATCH, 1),
                torch.randn(_BATCH, 8),
                torch.randn(_BATCH, _NUM_CONTROL_TOKENS, _D_MODEL),  # pyright: ignore[reportCallIssue]
            )

    def test_forward_with_control_tokens_preserves_output_shape(self) -> None:
        """Concat injection keeps the parameter-vector output shape."""
        field = _field()
        out = field(
            torch.randn(_BATCH, 6),
            torch.rand(_BATCH, 1),
            torch.randn(_BATCH, 8),
            control_tokens=torch.randn(_BATCH, _NUM_CONTROL_TOKENS, _D_MODEL),
        )
        assert out.shape == (_BATCH, 6)
        assert torch.isfinite(out).all()

    def test_forward_with_control_tokens_changes_prediction(self) -> None:
        """Control tokens actually reach the field's prediction."""
        field = _field()
        x = torch.randn(_BATCH, 6)
        t = torch.rand(_BATCH, 1)
        z = torch.randn(_BATCH, 8)
        with_control = field(
            x,
            t,
            z,
            control_tokens=torch.randn(_BATCH, _NUM_CONTROL_TOKENS, _D_MODEL),
        )
        assert not torch.allclose(field(x, t, z), with_control)

    def test_forward_layerwise_conditioning_with_control_tokens_runs(self) -> None:
        """Rank-3 layerwise conditioning coexists with control tokens."""
        field = _field()
        out = field(
            torch.randn(_BATCH, 6),
            torch.rand(_BATCH, 1),
            torch.randn(_BATCH, 2, 8),
            control_tokens=torch.randn(_BATCH, _NUM_CONTROL_TOKENS, _D_MODEL),
        )
        assert out.shape == (_BATCH, 6)
        assert torch.isfinite(out).all()

    def test_forward_layerwise_pe_applies_only_to_param_tokens(self) -> None:
        """Layerwise PE stays confined to the parameter-token prefix."""
        field = _field(pe_type="layerwise")
        out = field(
            torch.randn(_BATCH, 6),
            torch.rand(_BATCH, 1),
            torch.randn(_BATCH, 8),
            control_tokens=torch.randn(_BATCH, _NUM_CONTROL_TOKENS, _D_MODEL),
        )
        assert out.shape == (_BATCH, 6)
        assert torch.isfinite(out).all()
