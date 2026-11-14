"""Tests for the concat sketch-control token module and its field injection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from unittest import mock

import pytest
import torch
from jaxtyping import TypeCheckError

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    NUM_SKETCH_TRACK_ROWS,
    SKETCH_CENTROID_ROW,
    SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_SLICE,
    TIV_SKETCH_CONTROLS,
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


def _tokens_module(
    seed: int = 0,
    profile: SketchControlProfile = "music",
    median_filter_max_kernel_size: int = 1,
    median_filter_max_window_fraction: float | None = None,
    temporal_resampling: Literal["average", "linear"] = "average",
) -> SketchControlTokens:
    """Build a small token module with non-zero projection weights.

    :param seed: RNG seed for the randomized projection weights.
    :param profile: Channel layout selected for tokenization.
    :param median_filter_max_kernel_size: Largest literal training-time median window.
    :param median_filter_max_window_fraction: Token-grid fraction bounding the median window.
    :param temporal_resampling: Temporal reduction used before projection.
    :returns: Module whose three projections were re-randomized after zero-init.
    """
    module = SketchControlTokens(
        d_model=_D_MODEL,
        num_control_tokens=_NUM_CONTROL_TOKENS,
        profile=profile,
        median_filter_max_kernel_size=median_filter_max_kernel_size,
        median_filter_max_window_fraction=median_filter_max_window_fraction,
        temporal_resampling=temporal_resampling,
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
        assert spec.median_filter_max_kernel_size == 1
        assert spec.median_filter_max_window_fraction is None
        assert spec.temporal_resampling == "average"

    def test_pyfdn_reverb_profile_has_temporal_layout(self) -> None:
        """The reverb profile pins its channel and temporal-group layout."""
        spec = SketchControlSpec(num_frames=32, profile="pyfdn_reverb")

        assert spec.layout.num_controls == 10
        assert spec.layout.group_names == ("edc", "echo_density", "spectral_flatness")
        assert spec.layout.group_widths == (8, 1, 1)

    def test_tiv_profile_is_online_audio_only_with_one_control_group(self) -> None:
        """TIV opts into audio extraction and exposes all coordinates as one group."""
        spec = SketchControlSpec(
            profile="tiv",
            source="online",
            sample_rate=44_100,
            num_frames=32,
        )

        assert spec.layout.num_controls == 12
        assert spec.layout.group_names == ("tiv",)
        assert spec.layout.group_widths == (12,)

    def test_tiv_profile_without_audio_source_rejected(self) -> None:
        """TIV cannot silently request a nonexistent stored sketch column."""
        with pytest.raises(ValueError, match="tiv sketch requires source='online'"):
            SketchControlSpec(profile="tiv", num_frames=32)

    def test_online_audio_source_without_sample_rate_rejected(self) -> None:
        """Online extraction requires the waveform sample rate explicitly."""
        with pytest.raises(ValueError, match="sample_rate"):
            SketchControlSpec(profile="tiv", source="online", num_frames=32)

    def test_nonmusic_profile_rejects_median_filter_augmentation(self) -> None:
        """Median filtering is confined to the paper's music-control profile."""
        with pytest.raises(ValueError, match="median filtering is supported only"):
            SketchControlSpec(
                num_frames=32,
                profile="pyfdn_reverb",
                median_filter_max_kernel_size=3,
            )

    def test_music_online_source_without_sample_rate_rejected(self) -> None:
        """Online music extraction requires the waveform rate explicitly."""
        with pytest.raises(ValueError, match="online sketch extraction requires sample_rate"):
            SketchControlSpec(profile="music", source="online", num_frames=96)

    def test_music_online_source_with_sample_rate_is_accepted(self) -> None:
        """Music controls may be extracted directly from waveform batches."""
        spec = SketchControlSpec(
            profile="music", source="online", sample_rate=44_100, num_frames=96
        )

        assert spec.source == "online"
        assert spec.sample_rate == 44_100

    def test_pyfdn_online_source_rejected(self) -> None:
        """The reverb profile remains a stored-control-only contract."""
        with pytest.raises(ValueError, match="pyfdn_reverb.*source='online'"):
            SketchControlSpec(
                profile="pyfdn_reverb",
                source="online",
                sample_rate=44_100,
                num_frames=32,
            )

    def test_median_filter_kernel_cannot_exceed_token_grid(self) -> None:
        """The configured training window must fit the filtered token grid."""
        with pytest.raises(ValueError, match="cannot exceed num_control_tokens"):
            SketchControlSpec(
                num_frames=4,
                num_control_tokens=4,
                median_filter_max_kernel_size=5,
            )

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

    @pytest.mark.parametrize("num_control_tokens", [32, 64, 96])
    def test_music_dense_storage_accepts_equal_or_smaller_token_grid(
        self, num_control_tokens: int
    ) -> None:
        """Dense music controls may retain or reduce their temporal resolution.

        :param num_control_tokens: Accepted model token count.
        """
        spec = SketchControlSpec(num_frames=96, num_control_tokens=num_control_tokens)
        assert spec.num_control_tokens == num_control_tokens

    def test_music_storage_rejects_larger_token_grid(self) -> None:
        """Tokenization cannot invent a denser grid than stored music controls."""
        with pytest.raises(ValueError, match="cannot exceed num_frames"):
            SketchControlSpec(num_frames=32, num_control_tokens=96)

    def test_nonmusic_profile_rejects_nondefault_temporal_resampling(self) -> None:
        """Linear temporal resampling is confined to music controls."""
        with pytest.raises(ValueError, match="temporal resampling.*music"):
            SketchControlSpec(
                num_frames=32,
                profile="pyfdn_reverb",
                temporal_resampling="linear",
            )

    @pytest.mark.parametrize(
        ("num_control_tokens", "fraction", "expected_maximum"),
        [(32, None, 1), (32, 0.15625, 5), (96, 0.15625, 15), (32, 0.001, 1)],
    )
    def test_effective_median_window_resolves_fraction_on_token_grid(
        self, num_control_tokens: int, fraction: float | None, expected_maximum: int
    ) -> None:
        """The spec reports the training-time window maximum its tokenizer will use.

        :param num_control_tokens: Token grid under test.
        :param fraction: Optional token-grid fraction.
        :param expected_maximum: Effective positive window maximum.
        """
        spec = SketchControlSpec(
            num_frames=num_control_tokens,
            num_control_tokens=num_control_tokens,
            median_filter_max_window_fraction=fraction,
        )
        assert spec.effective_median_filter_max_kernel_size == expected_maximum

    def test_fractional_median_window_rejects_ambiguous_literal_policy(self) -> None:
        """Fractional and non-disabled literal maxima cannot both select a policy."""
        with pytest.raises(ValueError, match="median_filter_max_kernel_size=1"):
            SketchControlSpec(
                num_frames=32,
                median_filter_max_kernel_size=5,
                median_filter_max_window_fraction=0.15625,
            )

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("inf"), float("nan")])
    def test_fractional_median_window_out_of_range_rejected(self, fraction: float) -> None:
        """The optional median-window fraction must be finite and in ``(0, 1]``.

        :param fraction: Invalid fraction under test.
        """
        with pytest.raises(ValueError, match="median_filter_max_window_fraction"):
            SketchControlSpec(
                num_frames=32,
                median_filter_max_window_fraction=fraction,
            )

    def test_resolve_non_mapping_raises_type_error(self) -> None:
        """Non-mapping junk is rejected with a clear error."""
        with pytest.raises(TypeError, match="sketch"):
            resolve_sketch_controls(cast("SketchControls", 3))


class TestSketchControlTokens:
    """Behavior tests for the zero-init control tokenizer."""

    @pytest.mark.parametrize(
        ("num_control_tokens", "expected_maximum"),
        [(32, 5), (64, 10), (96, 15)],
    )
    def test_fractional_median_window_scales_with_token_grid(
        self, num_control_tokens: int, expected_maximum: int
    ) -> None:
        """Fractional policy rounds to the nearest positive token-window maximum.

        :param num_control_tokens: Token grid under test.
        :param expected_maximum: Effective maximum median window.
        """
        module = SketchControlTokens(
            d_model=_D_MODEL,
            num_control_tokens=num_control_tokens,
            median_filter_max_window_fraction=0.15625,
        )
        assert module.effective_median_filter_max_kernel_size == expected_maximum

    @pytest.mark.parametrize(
        ("fraction", "expected_maximum"),
        [(0.046875, 2), (0.001, 1)],
    )
    def test_fractional_median_window_half_up_and_minimum_bounds(
        self, fraction: float, expected_maximum: int
    ) -> None:
        """Fractional maxima round half up and never disable the one-frame baseline.

        :param fraction: Token-grid fraction under test.
        :param expected_maximum: Rounded positive window maximum.
        """
        module = SketchControlTokens(
            d_model=1,
            num_control_tokens=32,
            median_filter_max_window_fraction=fraction,
        )
        assert module.effective_median_filter_max_kernel_size == expected_maximum

    def test_fractional_and_literal_median_policies_match_training_output(self) -> None:
        """Equivalent fractional and literal maxima sample identical training filters."""
        fractional = SketchControlTokens(
            d_model=1,
            num_control_tokens=32,
            median_filter_max_window_fraction=0.15625,
        )
        literal = SketchControlTokens(
            d_model=1,
            num_control_tokens=32,
            median_filter_max_kernel_size=5,
        )
        for module in (fractional, literal):
            with torch.no_grad():
                cast(torch.nn.Linear, module.projections["loudness"]).weight.fill_(1.0)
        controls = torch.zeros(2, NUM_SKETCH_CONTROLS, 32)
        controls[:, SKETCH_LOUDNESS_ROW, 8:24] = 1.0
        keep = torch.tensor(((True, False, False), (True, False, False)))

        torch.manual_seed(41)
        fractional_tokens = fractional(controls, keep)
        torch.manual_seed(41)
        literal_tokens = literal(controls, keep)

        torch.testing.assert_close(fractional_tokens, literal_tokens, rtol=0, atol=0)

    def test_forward_disabled_group_with_nonfinite_controls_matches_clean_tokens(self) -> None:
        """Masking removes a disabled group's values entirely, so NaN there cannot leak."""
        module = SketchControlTokens(d_model=1, num_control_tokens=4).eval()
        for projection in module.projections.values():
            with torch.no_grad():
                cast(torch.nn.Linear, projection).weight.fill_(1.0)
        clean = torch.rand(1, NUM_SKETCH_CONTROLS, 4)
        poisoned = clean.clone()
        poisoned[:, SKETCH_CENTROID_ROW] = float("nan")
        poisoned[:, SKETCH_PITCH_SLICE.start] = float("inf")
        keep = torch.tensor(((True, False, False),))

        tokens = module(poisoned, keep)

        assert torch.isfinite(tokens).all()
        torch.testing.assert_close(tokens, module(clean, keep), rtol=0, atol=0)

    def test_forward_eval_ignores_fractional_median_policy(self) -> None:
        """Evaluation with a fractional policy equals the disabled compatibility path."""
        fractional = SketchControlTokens(
            d_model=1,
            num_control_tokens=32,
            median_filter_max_window_fraction=0.15625,
        ).eval()
        disabled = SketchControlTokens(d_model=1, num_control_tokens=32).eval()
        for module in (fractional, disabled):
            with torch.no_grad():
                cast(torch.nn.Linear, module.projections["loudness"]).weight.fill_(1.0)
        controls = torch.zeros(1, NUM_SKETCH_CONTROLS, 32)
        controls[:, SKETCH_LOUDNESS_ROW, 15] = 1.0
        keep = torch.tensor(((True, False, False),))

        torch.manual_seed(3)
        fractional_tokens = fractional(controls, keep)
        torch.manual_seed(29)
        disabled_tokens = disabled(controls, keep)

        torch.testing.assert_close(fractional_tokens, disabled_tokens, rtol=0, atol=0)

    def test_literal_median_window_remains_unchanged_on_larger_grid(self) -> None:
        """Legacy literal policy remains five frames rather than scaling with tokens."""
        module = SketchControlTokens(
            d_model=_D_MODEL,
            num_control_tokens=96,
            median_filter_max_kernel_size=5,
        )
        assert module.effective_median_filter_max_kernel_size == 5

    def test_forward_resampling_modes_produce_distinct_projected_tokens(self) -> None:
        """Checkpoint-selected temporal modes reach nonzero projection outputs."""
        average = SketchControlTokens(d_model=1, num_control_tokens=4)
        linear = SketchControlTokens(
            d_model=1,
            num_control_tokens=4,
            temporal_resampling="linear",
        )
        for module in (average, linear):
            with torch.no_grad():
                cast(torch.nn.Linear, module.projections["loudness"]).weight.fill_(1.0)
        controls = torch.zeros(1, NUM_SKETCH_CONTROLS, 2)
        controls[0, SKETCH_LOUDNESS_ROW, 1] = 1.0
        keep = torch.tensor(((True, False, False),))

        average_contribution = average(controls, keep) - average.unconditional(1)
        linear_contribution = linear(controls, keep) - linear.unconditional(1)

        torch.testing.assert_close(
            average_contribution[0, :, 0], torch.tensor((0.0, 0.0, 1.0, 1.0))
        )
        torch.testing.assert_close(
            linear_contribution[0, :, 0], torch.tensor((0.0, 0.25, 0.75, 1.0))
        )

    def test_forward_training_applies_configured_random_median_filter(self) -> None:
        """Training smooths music controls before their group projections."""
        module = SketchControlTokens(
            d_model=1,
            num_control_tokens=4,
            median_filter_max_kernel_size=2,
        )
        with torch.no_grad():
            cast(torch.nn.Linear, module.projections["loudness"]).weight.fill_(1.0)
        controls = torch.zeros(1, NUM_SKETCH_CONTROLS, 4)
        controls[0, SKETCH_LOUDNESS_ROW, 1] = 1.0
        keep = torch.tensor(((True, False, False),))

        module.eval()
        unfiltered = module(controls, keep) - module.unconditional(1)
        module.train()
        # Pin the per-row draw to the two-frame window; a draw of one would leave the row unfiltered.
        with mock.patch.object(torch, "randint", return_value=torch.tensor([2])):
            filtered = module(controls, keep) - module.unconditional(1)

        torch.testing.assert_close(unfiltered[0, :, 0], controls[0, SKETCH_LOUDNESS_ROW])
        torch.testing.assert_close(filtered, torch.zeros_like(filtered), rtol=0, atol=0)

    def test_forward_inference_mode_skips_augmentation_even_in_training_mode(self) -> None:
        """Sampling under inference_mode is deterministic although the module is still training."""
        module = SketchControlTokens(
            d_model=1,
            num_control_tokens=4,
            median_filter_max_kernel_size=4,
        )
        with torch.no_grad():
            cast(torch.nn.Linear, module.projections["loudness"]).weight.fill_(1.0)
        controls = torch.zeros(1, NUM_SKETCH_CONTROLS, 4)
        controls[0, SKETCH_LOUDNESS_ROW, 1] = 1.0
        keep = torch.tensor(((True, False, False),))
        module.train()

        with (
            torch.inference_mode(),
            mock.patch.object(torch, "randint", return_value=torch.tensor([4])),
        ):
            tokens = module(controls, keep) - module.unconditional(1)

        torch.testing.assert_close(tokens[0, :, 0], controls[0, SKETCH_LOUDNESS_ROW])

    def test_forward_training_same_seed_repeats_augmentation(self) -> None:
        """Resetting the Torch seed reproduces training-time control tokens."""
        module = _tokens_module(median_filter_max_kernel_size=4)
        controls = _controls()

        torch.manual_seed(23)
        first = module(controls, _keep_all())
        torch.manual_seed(23)
        second = module(controls, _keep_all())

        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_forward_eval_ignores_median_filter_policy(self) -> None:
        """Evaluation stays deterministic and consumes controls without smoothing."""
        module = _tokens_module(median_filter_max_kernel_size=4)
        module.eval()
        controls = _controls()

        torch.manual_seed(3)
        first = module(controls, _keep_all())
        torch.manual_seed(29)
        second = module(controls, _keep_all())

        torch.testing.assert_close(first, second, rtol=0, atol=0)

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
            (2, SKETCH_PITCH_SLICE.start, (0.5, 0.5, 0.5, 0.5)),
        ],
    )
    def test_forward_pools_all_groups_by_mean(
        self, group_index: int, row: int, expected: tuple[float, ...]
    ) -> None:
        """Every group's pooled values pin the shared mean reduction.

        The alternating input averages to ``0.5`` per bin and maxes to ``1.0``,
        so restoring maximum pooling for any group changes the projected token.

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


class TestTIVSketchControlTokens:
    """Validate temporal TIV pooling and channel contracts."""

    @pytest.mark.gpu
    def test_forward_tiv_cpu_and_cuda_outputs_are_numerically_equivalent(self) -> None:
        """TIV pooling and projection preserve outputs across CPU and CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        module = _tokens_module(seed=19, profile="tiv")
        controls = torch.randn(2, 12, 47, generator=torch.Generator().manual_seed(23))
        keep = torch.ones(2, 1, dtype=torch.bool)

        with torch.no_grad():
            expected = module(controls, keep)
            actual = module.cuda()(controls.cuda(), keep.cuda()).cpu()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_forward_pools_tiv_coordinates_by_mean(self) -> None:
        """TIV coordinates use average pooling."""
        module = SketchControlTokens(d_model=1, num_control_tokens=2, profile="tiv")
        with torch.no_grad():
            cast(torch.nn.Linear, module.projections["tiv"]).weight.fill_(1.0)
        controls = torch.zeros(1, 12, 4)
        controls[0, 0] = torch.tensor([0.0, 2.0, 4.0, 6.0])
        keep = torch.ones(1, 1, dtype=torch.bool)

        contribution = module(controls, keep) - module.unconditional(1)

        torch.testing.assert_close(contribution[0, :, 0], torch.tensor([1.0, 5.0]))

    def test_forward_tiv_wrong_channel_count_raises(self) -> None:
        """TIV rejects controls not carrying six real-imaginary pairs."""
        module = SketchControlTokens(d_model=_D_MODEL, profile="tiv")

        with pytest.raises(ValueError, match="12 channels"):
            module(torch.randn(_BATCH, 11, _NUM_FRAMES), torch.ones(_BATCH, 1, dtype=torch.bool))


class TestPyFDNReverbSketchControlTokens:
    """Validate pyFDN temporal alignment and CFG dropout invariants."""

    def test_forward_returns_one_token_per_reverb_frame(self) -> None:
        """Token count equals the canonical reverb frame count."""
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
        """The reverb profile rejects tensors outside its channel contract."""
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


def _field(
    pe_type: Literal["initial", "layerwise", "none"] = "none",
) -> ApproxEquivTransformer:
    """Build a tiny vector field for injection tests.

    :param pe_type: Parameter-token positional-encoding mode.
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
        pe_type=pe_type,
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


@pytest.mark.parametrize("mode", ["average", "linear"])
def test_music_tiv_tokens_mask_tiv_independently(mode: Literal["average", "linear"]) -> None:
    """Dropping TIV retains precisely the original three music projections.

    :param mode: Shared temporal reduction before group projection.
    """
    module = _tokens_module(profile="music_tiv", temporal_resampling=mode)
    music = _tokens_module(temporal_resampling=mode)
    music.load_state_dict(
        {
            key: value
            for key, value in module.state_dict().items()
            if key != "projections.tiv.weight"
        }
    )
    controls = torch.cat(
        (
            _controls(),
            torch.nn.functional.one_hot(
                torch.zeros(_BATCH, _NUM_FRAMES, dtype=torch.long), TIV_SKETCH_CONTROLS
            )
            .transpose(1, 2)
            .float(),
        ),
        dim=1,
    )
    keep_music = torch.tensor([[True, True, True, False]]).expand(_BATCH, -1)

    masked = module(controls, keep_music)

    torch.testing.assert_close(masked, music(controls[:, :NUM_SKETCH_CONTROLS], _keep_all()))
    assert not torch.equal(masked, module(controls, torch.ones(_BATCH, 4, dtype=torch.bool)))


def test_music_tiv_spec_stored_source_rejected() -> None:
    """Combined controls cannot be read from a stored music-only column."""
    with pytest.raises(ValueError, match="music_tiv sketch requires source='online'"):
        SketchControlSpec(profile="music_tiv", num_frames=96)


@pytest.mark.parametrize("group_index", [0, 1, 2])
def test_music_tiv_tokens_training_filters_each_music_group(group_index: int) -> None:
    """The shared training median window removes isolated music-control impulses.

    :param group_index: The only retained music group.
    """
    module = SketchControlTokens(
        d_model=1,
        num_control_tokens=4,
        profile="music_tiv",
        median_filter_max_kernel_size=3,
    )
    with torch.no_grad():
        for projection in module.projections.values():
            cast(torch.nn.Linear, projection).weight.fill_(1.0)
    controls = torch.zeros(1, NUM_SKETCH_CONTROLS + TIV_SKETCH_CONTROLS, 4)
    controls[:, :, 1] = 1.0
    keep = torch.zeros(1, 4, dtype=torch.bool)
    keep[:, group_index] = True
    module.eval()
    unfiltered = module(controls, keep) - module.unconditional(1)
    module.train()

    with mock.patch.object(torch, "randint", return_value=torch.tensor([3])):
        filtered = module(controls, keep) - module.unconditional(1)

    assert torch.count_nonzero(unfiltered) > 0
    torch.testing.assert_close(filtered, torch.zeros_like(filtered), rtol=0, atol=0)


_PRE_MEDIAN_FILTER_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sketch"


@dataclass(frozen=True)
class _RecordedTokens:
    """One recorded tokenizer pass from before median filtering existed.

    .. attribute :: state_dict

        Projection weights and positional-encoding buffer of the recorded module.

    .. attribute :: controls

        ``(batch, channels, frames)`` stored-sketch-like input batch.

    .. attribute :: keep

        ``(batch, groups)`` keep mask recorded with the inputs.

    .. attribute :: tokens

        Train-mode output tokens the recorded module produced.
    """

    state_dict: dict[str, torch.Tensor]
    controls: torch.Tensor
    keep: torch.Tensor
    tokens: torch.Tensor

    def module(self) -> SketchControlTokens:
        """Rebuild the recorded music tokenizer with the filter left at its disabled default.

        :returns: Tokenizer carrying the recorded projection weights.
        """
        module = SketchControlTokens(d_model=16, num_control_tokens=32, profile="music")
        module.load_state_dict(self.state_dict)
        return module


def _pre_median_filter_reference(commit: str, frames: int) -> _RecordedTokens:
    """Load one recorded tokenizer pass (see the fixture README for the commits).

    :param commit: Abbreviated commit the fixture was recorded at.
    :param frames: Stored frame grid of the recorded pass.
    :returns: Recorded weights, inputs, and train-mode tokens.
    """
    fixture = torch.load(_PRE_MEDIAN_FILTER_FIXTURES / f"sketch_tokens_{commit}.pt")
    return _RecordedTokens(
        state_dict=fixture["state_dict"],
        controls=fixture[f"controls_{frames}"],
        keep=fixture["keep"],
        tokens=fixture[f"tokens_{frames}"],
    )


@pytest.mark.parametrize(
    ("commit", "frames"),
    [
        pytest.param("1416c2da", 32, id="benchmark-commit-identity-grid"),
        pytest.param("d37e80c8e", 32, id="pre-3730-parent-identity-grid"),
        pytest.param("d37e80c8e", 64, id="pre-3730-parent-pooled-grid"),
    ],
)
def test_forward_training_without_median_filter_matches_pre_augmentation_tokens(
    commit: str, frames: int
) -> None:
    """With the filter off, train-mode tokens are bit-identical to the pre-#3730 tokenizer.

    :param commit: Fixture commit the reference tokens were recorded at.
    :param frames: Stored frame grid replayed through the 32-token module.
    """
    reference = _pre_median_filter_reference(commit, frames)
    module = reference.module().train()

    tokens = module(reference.controls, reference.keep)

    assert torch.equal(tokens, reference.tokens)


def test_forward_training_without_median_filter_matches_eval_forward() -> None:
    """With the filter off no stochastic path remains between train and eval mode."""
    reference = _pre_median_filter_reference("d37e80c8e", frames=64)
    module = reference.module()

    train_tokens = module.train()(reference.controls, reference.keep)
    eval_tokens = module.eval()(reference.controls, reference.keep)

    assert torch.equal(train_tokens, eval_tokens)
