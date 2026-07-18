"""Direct tests for :func:`decode_model_output`'s inverse-scale contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
    decode_model_output,
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


def test_encoded_width_counts_onehot_and_note_columns() -> None:
    """Width is the post-expansion column count — cutoff 1 + mode onehot 2 + pitch 1 +
    note start/end 2 = 6 — not the 4-entry parameter count, and it matches ``len``."""
    spec = _tiny_spec()

    assert spec.encoded_width == 6
    assert len(spec) == 6


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

        assert math.isnan(synth_params["cutoff"])

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

        assert len(note_params["note_start_and_end"]) == 1
