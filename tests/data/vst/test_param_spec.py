"""Direct tests for :func:`decode_model_output`'s inverse-scale contract."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousArrayParameter,
    ContinuousParameter,
    DiscreteArrayParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
    decode_model_output,
)
from synth_setter.models.components.transformer import LearntProjection

LEVEL_MIN_DB = -70.0
LEVEL_MID_DB = -15.0
LEVEL_MAX_DB = 40.0


def test_continuous_array_roundtrip_preserves_matrix_shape() -> None:
    parameter = ContinuousArrayParameter(
        name="feedback_matrix",
        shape=(8, 8),
        min=-1.0,
        max=1.0,
    )
    raw = np.linspace(-1.0, 1.0, 64, dtype=np.float64).reshape(8, 8)

    decoded = parameter.decode(parameter.encode(raw))

    np.testing.assert_allclose(decoded, raw, atol=1e-7)
    assert decoded.shape == (8, 8)


def test_discrete_array_decode_rounds_to_int64() -> None:
    parameter = DiscreteArrayParameter(name="delays", shape=(8,), min=400, max=1200)

    decoded = parameter.decode(
        np.array([0.0, 0.1, 0.25, 0.499, 0.501, 0.75, 0.9, 1.0])
    )

    np.testing.assert_array_equal(
        decoded,
        np.array([400, 480, 600, 799, 801, 1000, 1120, 1200], dtype=np.int64),
    )
    assert decoded.dtype == np.int64


@pytest.mark.parametrize("shape", [(8, 1), (1, 8), (1, 1)])
def test_continuous_array_roundtrip_preserves_rectangular_shape(
    shape: tuple[int, ...],
) -> None:
    parameter = ContinuousArrayParameter(
        name="matrix",
        shape=shape,
        min=-1.0,
        max=1.0,
    )
    raw = np.zeros(shape, dtype=np.float64)

    decoded = parameter.decode(parameter.encode(raw))

    assert decoded.shape == shape


def test_continuous_array_encode_returns_flat_float32() -> None:
    parameter = ContinuousArrayParameter(
        name="matrix",
        shape=(2, 2),
        min=-1.0,
        max=1.0,
    )

    encoded = parameter.encode(np.array([[-1.0, 0.0], [0.5, 1.0]]))

    np.testing.assert_array_equal(
        encoded,
        np.array([0.0, 0.5, 0.75, 1.0], dtype=np.float32),
    )
    assert encoded.dtype == np.float32


def test_continuous_array_decode_returns_float64() -> None:
    parameter = ContinuousArrayParameter(
        name="matrix",
        shape=(1, 1),
        min=-1.0,
        max=1.0,
    )

    decoded = parameter.decode(np.array([0.5], dtype=np.float32))

    assert decoded.dtype == np.float64


def test_continuous_array_encode_rejects_wrong_shape() -> None:
    parameter = ContinuousArrayParameter(
        name="matrix",
        shape=(2, 2),
        min=-1.0,
        max=1.0,
    )

    with pytest.raises(ValueError, match=r"matrix must have shape \(2, 2\)"):
        parameter.encode(np.zeros((4,), dtype=np.float64))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_continuous_array_encode_rejects_nonfinite_value(value: float) -> None:
    parameter = ContinuousArrayParameter(
        name="matrix",
        shape=(1, 1),
        min=-1.0,
        max=1.0,
    )

    with pytest.raises(ValueError, match="must contain only finite values"):
        parameter.encode(np.array([[value]]))


@pytest.mark.parametrize("value", [-1.1, 1.1])
def test_continuous_array_encode_rejects_out_of_bounds_value(value: float) -> None:
    parameter = ContinuousArrayParameter(
        name="matrix",
        shape=(1, 1),
        min=-1.0,
        max=1.0,
    )

    with pytest.raises(ValueError, match=r"must be within \[-1.0, 1.0\]"):
        parameter.encode(np.array([[value]]))


def test_discrete_array_encode_rejects_fractional_native_value() -> None:
    parameter = DiscreteArrayParameter(name="delays", shape=(2,), min=400, max=1200)

    with pytest.raises(ValueError, match="must contain only integer values"):
        parameter.encode(np.array([400.0, 400.5]))


def test_array_length_is_product_of_native_shape() -> None:
    parameter = ContinuousArrayParameter(
        name="matrix",
        shape=(2, 3, 4),
        min=-1.0,
        max=1.0,
    )

    assert len(parameter) == 24


def test_discrete_array_roundtrip_preserves_values() -> None:
    parameter = DiscreteArrayParameter(name="delays", shape=(8,), min=400, max=1200)
    raw = np.array([400, 401, 500, 600, 700, 800, 1000, 1200], dtype=np.int64)

    decoded = parameter.decode(parameter.encode(raw))

    np.testing.assert_array_equal(decoded, raw)


def test_array_encoded_names_follow_c_order_coordinates() -> None:
    parameter = ContinuousArrayParameter(
        name="feedback_matrix",
        shape=(2, 3),
        min=-1.0,
        max=1.0,
    )

    assert parameter.encoded_names() == (
        "feedback_matrix.0.0",
        "feedback_matrix.0.1",
        "feedback_matrix.0.2",
        "feedback_matrix.1.0",
        "feedback_matrix.1.1",
        "feedback_matrix.1.2",
    )


def _tiny_spec() -> ParamSpec:
    return ParamSpec(
        [
            ContinuousParameter(name="cutoff"),
            CategoricalParameter(
                name="mode",
                values=["Digital", "Analog"],
                raw_values=[0.25, 0.75],
                encoding="onehot",
            ),
        ],
        [
            DiscreteLiteralParameter(name="pitch", min=21, max=108),
            NoteDurationParameter(name="note_start_and_end", max_note_duration_seconds=4.0),
        ],
    )


# Widths: cutoff 1, mode onehot 2, pitch 1, note duration 2 -> 6.
_ROW = [0.0, -1.0, 1.0, 0.0, 0.2, 0.2]


def test_continuous_parameter_encodes_native_range_to_unit_interval() -> None:
    """Continuous controls encode arbitrary native ranges onto the model domain."""
    parameter = ContinuousParameter(
        name="level_db", min=LEVEL_MIN_DB, max=LEVEL_MAX_DB
    )

    assert LEVEL_MIN_DB <= parameter.sample(np.random.default_rng(0)) <= LEVEL_MAX_DB
    assert parameter.encode(LEVEL_MIN_DB) == pytest.approx(np.array([0.0]))
    assert parameter.encode(LEVEL_MID_DB) == pytest.approx(np.array([0.5]))
    assert parameter.encode(LEVEL_MAX_DB) == pytest.approx(np.array([1.0]))
    assert parameter.decode(np.array([0.0])) == pytest.approx(LEVEL_MIN_DB)
    assert parameter.decode(np.array([0.5])) == pytest.approx(LEVEL_MID_DB)
    assert parameter.decode(np.array([1.0])) == pytest.approx(LEVEL_MAX_DB)


def test_continuous_parameter_samples_uniform_native_domain() -> None:
    """Sampling maps the generator's uniform draws to the complete native domain."""
    actual_rng = np.random.default_rng(19)
    expected_rng = np.random.default_rng(19)
    parameter = ContinuousParameter(
        name="level_db", min=LEVEL_MIN_DB, max=LEVEL_MAX_DB
    )

    actual = np.array([parameter.sample(actual_rng) for _ in range(32)])
    expected = expected_rng.uniform(LEVEL_MIN_DB, LEVEL_MAX_DB, size=32)

    np.testing.assert_allclose(actual, expected)


def test_continuous_parameter_constant_branch_uses_native_domain() -> None:
    """An enabled constant samples and encodes within its native range."""
    parameter = ContinuousParameter(
        name="level_db",
        min=LEVEL_MIN_DB,
        max=LEVEL_MAX_DB,
        constant_val_p=1.0,
        constant_val=LEVEL_MID_DB,
    )

    sampled = parameter.sample(np.random.default_rng(0))

    assert sampled == LEVEL_MID_DB
    assert parameter.encode(sampled) == pytest.approx(np.array([0.5]))


def test_continuous_parameter_rejects_nonfinite_native_span() -> None:
    """Native bounds must retain a finite normalization denominator."""
    with pytest.raises(ValueError, match="span must be finite"):
        ContinuousParameter(name="overflow", min=-1e308, max=1e308)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min": -np.inf}, "bounds must be finite"),
        ({"max": np.inf}, "bounds must be finite"),
        ({"min": 1.0, "max": 1.0}, "max must be greater than min"),
        ({"min": 1.0, "max": 0.0}, "max must be greater than min"),
        ({"constant_val_p": -0.1}, "constant_val_p must be in"),
        ({"constant_val_p": 1.1}, "constant_val_p must be in"),
        ({"constant_val_p": 1.0, "constant_val": -1.0}, "constant_val must be within"),
        ({"constant_val_p": 1.0, "constant_val": 2.0}, "constant_val must be within"),
    ],
)
def test_continuous_parameter_rejects_invalid_native_configuration(
    overrides: dict[str, float],
    message: str,
) -> None:
    """Invalid ranges and enabled constants fail outside optimized assertions.

    :param overrides: Invalid constructor values under test.
    :param message: Expected validation-error fragment.
    """
    values = {"name": "native", "min": 0.0, "max": 1.0, **overrides}

    with pytest.raises(ValueError, match=message):
        ContinuousParameter(**values)  # type: ignore[arg-type]


def test_continuous_parameter_allows_disabled_constant_outside_native_domain() -> None:
    """A disabled constant is not part of the sampled native domain."""
    parameter = ContinuousParameter(
        name="level_db",
        min=LEVEL_MIN_DB,
        max=LEVEL_MAX_DB,
        constant_val_p=0.0,
        constant_val=-120.0,
    )

    assert LEVEL_MIN_DB <= parameter.sample(np.random.default_rng(0)) <= LEVEL_MAX_DB


def test_param_spec_sampling_roundtrip_preserves_native_domain() -> None:
    """The dataset sampling path round-trips renderer-native continuous values."""
    spec = ParamSpec(
        [
            ContinuousParameter(
                name="level_db", min=LEVEL_MIN_DB, max=LEVEL_MAX_DB
            )
        ],
        [
            DiscreteLiteralParameter(name="pitch", min=21, max=108),
            NoteDurationParameter(
                name="note_start_and_end", max_note_duration_seconds=4.0
            ),
        ],
    )

    synth_params, note_params = spec.sample(np.random.default_rng(5))
    decoded_synth_params, decoded_note_params = spec.decode(
        spec.encode(synth_params, note_params)
    )

    sampled_level = synth_params["level_db"]
    decoded_level = decoded_synth_params["level_db"]
    assert isinstance(sampled_level, float)
    assert isinstance(decoded_level, float)
    assert LEVEL_MIN_DB <= sampled_level <= LEVEL_MAX_DB
    assert decoded_level == pytest.approx(sampled_level, abs=1e-5)
    assert decoded_note_params["pitch"] == note_params["pitch"]
    assert decoded_note_params["note_start_and_end"] == pytest.approx(
        note_params["note_start_and_end"]
    )


@pytest.mark.parametrize(
    ("model_output", "expected"),
    [
        (-1.0, LEVEL_MIN_DB),
        (0.0, LEVEL_MID_DB),
        (1.0, LEVEL_MAX_DB),
    ],
)
def test_decode_model_output_maps_to_native_continuous_domain(
    model_output: float, expected: float
) -> None:
    """Model-domain values decode to renderer-native values.

    :param model_output: Scalar prediction in the model domain.
    :param expected: Expected renderer-native value.
    """
    spec = ParamSpec(
        [
            ContinuousParameter(
                name="level_db", min=LEVEL_MIN_DB, max=LEVEL_MAX_DB
            )
        ],
        [],
    )

    synth_params, _ = decode_model_output(np.array([model_output]), spec)

    assert synth_params["level_db"] == pytest.approx(expected)


def test_param_spec_encoded_names_match_encoded_width() -> None:
    spec = ParamSpec(
        [
            ContinuousParameter(name="gain"),
            CategoricalParameter(
                name="mode",
                values=["a", "b"],
                raw_values=[0.0, 1.0],
                encoding="onehot",
            ),
            ContinuousArrayParameter(
                name="matrix",
                shape=(1, 2),
                min=-1.0,
                max=1.0,
            ),
        ],
        [NoteDurationParameter(name="window", max_note_duration_seconds=4.0)],
    )

    assert spec.encoded_names == [
        "gain",
        "mode.0",
        "mode.1",
        "matrix.0.0",
        "matrix.0.1",
        "window.0",
        "window.1",
    ]
    assert len(spec.encoded_names) == spec.encoded_width


def test_param_spec_names_remain_logical_field_names() -> None:
    spec = ParamSpec(
        [
            ContinuousArrayParameter(
                name="matrix",
                shape=(2, 2),
                min=-1.0,
                max=1.0,
            )
        ],
        [],
    )

    assert spec.names == ["matrix"]


def test_param_spec_sample_supports_empty_synth_group() -> None:
    spec = ParamSpec([], [DiscreteLiteralParameter(name="pitch", min=48, max=72)])

    synth_values, _ = spec.sample(np.random.default_rng(0))

    assert synth_values == {}


def test_param_spec_sample_supports_empty_note_group() -> None:
    spec = ParamSpec([ContinuousParameter(name="gain")], [])

    _, note_values = spec.sample(np.random.default_rng(0))

    assert note_values == {}


def test_param_spec_decode_supports_both_groups_empty() -> None:
    spec = ParamSpec([], [])

    synth_values, note_values = spec.decode(np.empty((0,), dtype=np.float32))

    assert synth_values == {}
    assert note_values == {}


def test_param_spec_encode_supports_empty_synth_group() -> None:
    spec = ParamSpec([], [DiscreteLiteralParameter(name="pitch", min=48, max=72)])

    encoded = spec.encode({}, {"pitch": 60})

    np.testing.assert_array_equal(encoded, np.array([0.5], dtype=np.float32))


def test_param_spec_encode_supports_empty_note_group() -> None:
    spec = ParamSpec([ContinuousParameter(name="gain")], [])

    encoded = spec.encode({"gain": 0.25}, {})

    np.testing.assert_array_equal(encoded, np.array([0.25], dtype=np.float32))


def test_param_spec_encode_supports_both_groups_empty() -> None:
    spec = ParamSpec([], [])

    encoded = spec.encode({}, {})

    assert encoded.shape == (0,)
    assert encoded.dtype == np.float32


def test_scalar_parameter_golden_encoding_remains_unchanged() -> None:
    parameter = ContinuousParameter(name="gain", min=-1.0, max=1.0)

    encoded = parameter.encode(0.0)

    np.testing.assert_array_equal(encoded, np.array([0.5]))
    assert parameter.decode(encoded) == 0.0


def test_discrete_parameter_golden_encoding_remains_unchanged() -> None:
    parameter = DiscreteLiteralParameter(name="pitch", min=48, max=72)

    encoded = parameter.encode(60)

    np.testing.assert_array_equal(encoded, np.array([0.5]))
    assert parameter.decode(encoded) == 60


def test_categorical_onehot_golden_encoding_remains_unchanged() -> None:
    parameter = CategoricalParameter(
        name="mode",
        values=["a", "b", "c"],
        raw_values=[0.0, 0.5, 1.0],
        encoding="onehot",
    )

    encoded = parameter.encode(0.5)

    np.testing.assert_array_equal(encoded, np.array([0.0, 1.0, 0.0]))
    assert parameter.decode(encoded) == 0.5


def test_note_duration_golden_encoding_remains_unchanged() -> None:
    parameter = NoteDurationParameter(
        name="note_start_and_end",
        max_note_duration_seconds=4.0,
    )

    encoded = parameter.encode((1.0, 3.0))

    np.testing.assert_array_equal(encoded, np.array([0.25, 0.75]))
    assert parameter.decode(encoded) == (1.0, 3.0)


def test_param2tok_projection_accepts_flat_array_parameter_width() -> None:
    spec = ParamSpec(
        [
            ContinuousArrayParameter(
                name="matrix",
                shape=(2, 3),
                min=-1.0,
                max=1.0,
            )
        ],
        [],
    )
    projection = LearntProjection(
        d_model=4,
        d_token=4,
        num_params=spec.encoded_width,
        num_tokens=2,
        initial_ffn=False,
        final_ffn=False,
    )
    params = torch.zeros(3, spec.encoded_width)

    reconstructed = projection.token_to_param(projection.param_to_token(params))

    assert reconstructed.shape == (3, 6)


def test_encoded_width_counts_onehot_and_note_columns() -> None:
    """Encoded width reflects expanded parameter columns, not source parameter count."""
    spec = _tiny_spec()

    assert spec.encoded_width == 6
    assert len(spec) == 6


class TestEncodedSlices:
    """The name-to-column-span contract callers need to index an encoded row."""

    def test_spans_cover_every_column_in_encoding_order(self) -> None:
        """Slices are contiguous from 0 and ordered synth-then-note, matching ``encode``."""
        spans = [(param.name, sl.start, sl.stop) for param, sl in _tiny_spec().encoded_slices()]

        assert spans == [
            ("cutoff", 0, 1),
            ("mode", 1, 3),
            ("pitch", 3, 4),
            ("note_start_and_end", 4, 6),
        ]

    def test_final_stop_equals_encoded_width(self) -> None:
        """The walk consumes exactly the encoded row — no trailing columns are unclaimed."""
        spec = _tiny_spec()

        *_, (_, last) = spec.encoded_slices()

        assert last.stop == spec.encoded_width

    def test_slice_indexes_the_column_the_parameter_encoded(self) -> None:
        """Indexing an encoded row by a parameter's slice returns that parameter's columns."""
        spec = _tiny_spec()
        row = spec.encode({"cutoff": 0.5, "mode": 0.75}, {"pitch": 60, "note_start_and_end": (0, 1)})
        spans = dict((param.name, sl) for param, sl in spec.encoded_slices())

        assert row[spans["mode"]].tolist() == [0.0, 1.0]


class TestSynthColumns:
    """The synth/note split renderers use to keep note columns away from the voice."""

    def test_span_covers_the_synth_params_and_stops_before_the_note_params(self) -> None:
        """The span ends where the first note parameter's own span begins."""
        spec = _tiny_spec()
        spans = dict((param.name, sl) for param, sl in spec.encoded_slices())

        assert spec.synth_columns == slice(spans["cutoff"].start, spans["mode"].stop)
        assert spec.synth_columns.stop == spans["pitch"].start

    def test_indexing_a_row_by_the_span_drops_every_note_column(self) -> None:
        """A row sliced by the span keeps exactly the synth parameters' encoded width."""
        spec = _tiny_spec()
        row = spec.encode({"cutoff": 0.5, "mode": 0.75}, {"pitch": 60, "note_start_and_end": (0, 1)})

        assert len(row[spec.synth_columns]) == spec.synth_param_length

    def test_note_only_spec_has_an_empty_span(self) -> None:
        """A spec with no synth params yields an empty span rather than an index error."""
        note_only = ParamSpec([], [DiscreteLiteralParameter(name="pitch", min=48, max=72)])

        assert note_only.synth_columns == slice(0, 0)


class TestDecodeModelOutput:
    """The rescale-then-clip contract, pinned independently of any caller."""

    def test_midpoint_prediction_decodes_to_encoded_half(self) -> None:
        """A 0.0 model output rescales to the encoded midpoint 0.5."""
        result = decode_model_output(np.array(_ROW, dtype=np.float32), _tiny_spec())

        assert isinstance(result, tuple) and len(result) == 2
        synth_params, _ = result
        assert synth_params["cutoff"] == pytest.approx(0.5)

    def test_extreme_predictions_decode_to_unit_bounds(self) -> None:
        """Model outputs -1 and 1 rescale to the encoded bounds 0 and 1."""
        low_row = np.array([-1.0, *_ROW[1:]], dtype=np.float32)
        high_row = np.array([1.0, *_ROW[1:]], dtype=np.float32)

        low, _ = decode_model_output(low_row, _tiny_spec())
        high, _ = decode_model_output(high_row, _tiny_spec())

        assert low["cutoff"] == pytest.approx(0.0)
        assert high["cutoff"] == pytest.approx(1.0)

    def test_out_of_range_predictions_clip_to_unit_bounds(self) -> None:
        """Values outside [-1, 1] clip to the encoded bounds instead of overshooting."""
        low_row = np.array([-7.5, *_ROW[1:]], dtype=np.float32)
        high_row = np.array([7.5, *_ROW[1:]], dtype=np.float32)

        low, _ = decode_model_output(low_row, _tiny_spec())
        high, _ = decode_model_output(high_row, _tiny_spec())

        assert low["cutoff"] == pytest.approx(0.0)
        assert high["cutoff"] == pytest.approx(1.0)

    def test_categorical_logits_decode_to_nearest_raw_value(self) -> None:
        """Onehot positions survive the rescale: the larger logit picks the raw_value."""
        synth_params, _ = decode_model_output(np.array(_ROW, dtype=np.float32), _tiny_spec())

        assert synth_params["mode"] == pytest.approx(0.75)

    def test_note_params_decode_to_native_domain(self) -> None:
        """Note params come back in their native domains, not the encoded [0, 1]."""
        row = np.array([*_ROW[:3], 1.0, 0.2, 0.2], dtype=np.float32)

        _, note_params = decode_model_output(row, _tiny_spec())

        assert note_params["pitch"] == 108
        # 0.2 in [-1, 1] rescales to 0.6, then lerps onto the 4 s duration grid.
        assert note_params["note_start_and_end"] == pytest.approx((2.4, 2.4))

    def test_input_row_is_not_mutated(self) -> None:
        """Decoding never mutates the caller's row (callers reuse prediction tensors)."""
        row = np.array([7.5, *_ROW[1:]], dtype=np.float32)
        before = row.copy()

        decode_model_output(row, _tiny_spec())

        assert np.array_equal(row, before)

    def test_nan_predictions_pass_through_undetected(self) -> None:
        """Current contract: NaN survives np.clip and decodes through unchanged.

        Pinned so adding a NaN guard is a deliberate contract change, not a regression.
        """
        row = np.array([math.nan, *_ROW[1:]], dtype=np.float32)

        synth_params, _ = decode_model_output(row, _tiny_spec())
        cutoff = synth_params["cutoff"]

        assert isinstance(cutoff, float)
        assert math.isnan(cutoff)

    def test_over_long_rows_are_silently_truncated(self) -> None:
        """Current contract: extra trailing values are ignored by ParamSpec.decode.

        Pinned so adding a width check is a deliberate, visible contract change.
        """
        row = np.array([*_ROW, 9.9, 9.9], dtype=np.float32)

        synth_params, _ = decode_model_output(row, _tiny_spec())

        assert synth_params["cutoff"] == pytest.approx(0.5)

    def test_rows_truncated_to_starve_a_scalar_param_fail_loudly(self) -> None:
        """Current contract: truncation that empties a later scalar's slice raises ValueError.

        The truncated-through categorical itself decodes silently (argmax of the
        short slice); the loud failure is pitch's empty slice hitting .item().
        """
        row = np.array(_ROW[:2], dtype=np.float32)

        with pytest.raises(ValueError):
            decode_model_output(row, _tiny_spec())

    def test_tail_truncated_rows_corrupt_note_duration_silently(self) -> None:
        """Current contract: a row missing only tail values decodes without raising.

        The note-duration value comes back malformed (a 1-tuple) — pinned so a
        future width guard is a deliberate contract change.
        """
        row = np.array(_ROW[:5], dtype=np.float32)

        _, note_params = decode_model_output(row, _tiny_spec())
        note_window = note_params["note_start_and_end"]

        assert isinstance(note_window, tuple)
        assert len(note_window) == 1


class TestModelSpaceConversion:
    """The single-owner width splice and ``[-1, 1]`` <-> ``[0, 1]`` affine."""

    def test_model_to_encoded_inverts_encoded_to_model(self) -> None:
        """Round-tripping an in-range encoded row returns it unchanged."""
        spec = _tiny_spec()
        encoded = np.linspace(0.0, 1.0, spec.encoded_width)

        assert spec.model_to_encoded(spec.encoded_to_model(encoded)) == pytest.approx(encoded)

    def test_model_to_encoded_clips_predictions_outside_the_model_range(self) -> None:
        """Out-of-range predictions saturate at the encoded domain's bounds."""
        spec = _tiny_spec()

        encoded = spec.model_to_encoded(np.array([-3.0, 3.0]))

        assert encoded.tolist() == [0.0, 1.0]
