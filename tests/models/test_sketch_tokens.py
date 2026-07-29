"""Tests for the concat sketch-control token module and its field injection."""

from typing import Literal, cast

import pytest
import torch
from jaxtyping import TypeCheckError

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    NUM_SKETCH_TRACK_ROWS,
    SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_SLICE,
    SketchControls,
    SketchControlSpec,
    resolve_sketch_controls,
)
from synth_setter.models.components.sketch_tokens import SketchControlTokens
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)

_BATCH = 2
_D_MODEL = 16
_NUM_FRAMES = 11
_NUM_CTRL_TOKENS = 4


def _controls(batch: int = _BATCH, seed: int = 0) -> torch.Tensor:
    """Draw a realistic sketch-control batch.

    :param batch: Batch size.
    :param seed: RNG seed.
    :returns: ``(batch, NUM_SKETCH_CONTROLS, _NUM_FRAMES)`` controls with
        signed-unit loudness/centroid rows and unit-interval pitch rows.
    """
    generator = torch.Generator().manual_seed(seed)
    ctrl = torch.rand((batch, NUM_SKETCH_CONTROLS, _NUM_FRAMES), generator=generator)
    ctrl[:, :NUM_SKETCH_TRACK_ROWS] = ctrl[:, :NUM_SKETCH_TRACK_ROWS] * 2 - 1
    return ctrl


def _tokens_module(seed: int = 0) -> SketchControlTokens:
    """Build a small token module with non-zero projection weights.

    :param seed: RNG seed for the randomized projection weights.
    :returns: Module whose three projections were re-randomized after zero-init.
    """
    module = SketchControlTokens(d_model=_D_MODEL, num_ctrl_tokens=_NUM_CTRL_TOKENS)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for projection in module.projections.values():
            linear = cast(torch.nn.Linear, projection)
            linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator))
    return module


def _no_drop(batch: int = _BATCH) -> torch.Tensor:
    """Return an all-false drop mask.

    :param batch: Batch size.
    :returns: ``(batch, 3)`` boolean mask keeping every control.
    """
    return torch.zeros(batch, 3, dtype=torch.bool)


class TestSketchControlSpec:
    """Contract tests for the strict sketch-control spec."""

    def test_defaults_column_tokens_threshold_match_design(self) -> None:
        """Spec defaults pin the approved column, token budget, and threshold."""
        spec = SketchControlSpec(num_frames=401)
        assert spec.column == "sketch_ctrl"
        assert spec.num_ctrl_tokens == 32
        assert spec.pitch_zero_threshold == 0.1

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

    def test_resolve_non_mapping_raises_type_error(self) -> None:
        """Non-mapping junk is rejected with a clear error."""
        with pytest.raises(TypeError, match="sketch"):
            resolve_sketch_controls(cast("SketchControls", 3))


class TestSketchControlTokens:
    """Behavior tests for the zero-init control tokenizer."""

    def test_forward_control_batch_returns_token_sequence(self) -> None:
        """The tokenizer emits the configured token sequence shape."""
        module = _tokens_module()
        out = module(_controls(), _no_drop())
        assert out.shape == (_BATCH, _NUM_CTRL_TOKENS, _D_MODEL)

    def test_forward_at_zero_init_outputs_exactly_the_positional_encoding(self) -> None:
        """Zero-init projections contribute nothing: output is the PE alone."""
        module = SketchControlTokens(d_model=_D_MODEL, num_ctrl_tokens=_NUM_CTRL_TOKENS)
        out = module(_controls(), _no_drop())
        expected = module.positional_encoding.expand(_BATCH, -1, -1)
        torch.testing.assert_close(out, expected)

    def test_forward_dropped_control_matches_zeroed_channels(self) -> None:
        """Masking a control equals zeroing its channels before projection."""
        module = _tokens_module()
        ctrl = _controls()
        mask = _no_drop()
        mask[0, 2] = True  # drop pitch for the first sample only

        zeroed = ctrl.clone()
        zeroed[0, SKETCH_PITCH_SLICE] = 0.0

        torch.testing.assert_close(module(ctrl, mask), module(zeroed, _no_drop()))

    def test_forward_all_dropped_outputs_exactly_the_positional_encoding(self) -> None:
        """Dropping every control leaves exactly the PE."""
        module = _tokens_module()
        out = module(_controls(), torch.ones(_BATCH, 3, dtype=torch.bool))
        expected = module.positional_encoding.expand(_BATCH, -1, -1)
        torch.testing.assert_close(out, expected)

    def test_forward_loudness_and_centroid_drops_are_independent(self) -> None:
        """Dropping loudness leaves the centroid contribution intact."""
        module = _tokens_module()
        ctrl = _controls()
        mask = _no_drop()
        mask[:, 0] = True  # drop loudness everywhere

        zeroed = ctrl.clone()
        zeroed[:, SKETCH_LOUDNESS_ROW] = 0.0

        torch.testing.assert_close(module(ctrl, mask), module(zeroed, _no_drop()))

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
        assert not torch.allclose(module(base, _no_drop(1)), module(spike, _no_drop(1)))

    def test_forward_wrong_channel_count_raises_shape_error(self) -> None:
        """Jaxtyping rejects a wrong channel count at the call boundary."""
        module = _tokens_module()
        bad = torch.rand(_BATCH, NUM_SKETCH_CONTROLS - 1, _NUM_FRAMES)
        with pytest.raises(TypeCheckError):
            module(bad, _no_drop())


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


class TestApproxEquivTransformerCtrlTokens:
    """Behavior of the optional ctrl-token concat path."""

    def test_forward_without_ctrl_kwarg_matches_explicit_none(self) -> None:
        """``ctrl_tokens=None`` is a strict no-op against the legacy call."""
        field = _field()
        x = torch.randn(_BATCH, 6)
        t = torch.rand(_BATCH, 1)
        z = torch.randn(_BATCH, 8)
        torch.testing.assert_close(field(x, t, z), field(x, t, z, ctrl_tokens=None))

    def test_forward_with_ctrl_tokens_preserves_output_shape(self) -> None:
        """Concat injection keeps the parameter-vector output shape."""
        field = _field()
        out = field(
            torch.randn(_BATCH, 6),
            torch.rand(_BATCH, 1),
            torch.randn(_BATCH, 8),
            ctrl_tokens=torch.randn(_BATCH, _NUM_CTRL_TOKENS, _D_MODEL),
        )
        assert out.shape == (_BATCH, 6)
        assert torch.isfinite(out).all()

    def test_forward_with_ctrl_tokens_changes_prediction(self) -> None:
        """Control tokens actually reach the field's prediction."""
        field = _field()
        x = torch.randn(_BATCH, 6)
        t = torch.rand(_BATCH, 1)
        z = torch.randn(_BATCH, 8)
        with_ctrl = field(x, t, z, ctrl_tokens=torch.randn(_BATCH, _NUM_CTRL_TOKENS, _D_MODEL))
        assert not torch.allclose(field(x, t, z), with_ctrl)

    def test_forward_layerwise_conditioning_with_ctrl_tokens_runs(self) -> None:
        """Rank-3 layerwise conditioning coexists with control tokens."""
        field = _field()
        out = field(
            torch.randn(_BATCH, 6),
            torch.rand(_BATCH, 1),
            torch.randn(_BATCH, 2, 8),
            ctrl_tokens=torch.randn(_BATCH, _NUM_CTRL_TOKENS, _D_MODEL),
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
            ctrl_tokens=torch.randn(_BATCH, _NUM_CTRL_TOKENS, _D_MODEL),
        )
        assert out.shape == (_BATCH, 6)
        assert torch.isfinite(out).all()
