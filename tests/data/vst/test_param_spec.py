"""Direct tests for :func:`decode_model_output`'s inverse-scale contract."""

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
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
        [DiscreteLiteralParameter(name="pitch", min=21, max=108)],
    )


class TestDecodeModelOutput:
    """The rescale-then-clip contract, pinned independently of any caller."""

    def test_midpoint_prediction_decodes_to_encoded_half(self):
        """A 0.0 model output rescales to the encoded midpoint 0.5."""
        row = np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float32)

        synth_params, _ = decode_model_output(row, _tiny_spec())

        assert synth_params["cutoff"] == pytest.approx(0.5)

    def test_extreme_predictions_decode_to_unit_bounds(self):
        """Model outputs -1 and 1 rescale to the encoded bounds 0 and 1."""
        low, _ = decode_model_output(np.array([-1.0, -1.0, 1.0, 0.0], np.float32), _tiny_spec())
        high, _ = decode_model_output(np.array([1.0, -1.0, 1.0, 0.0], np.float32), _tiny_spec())

        assert low["cutoff"] == pytest.approx(0.0)
        assert high["cutoff"] == pytest.approx(1.0)

    def test_out_of_range_predictions_clip_to_unit_bounds(self):
        """Values outside [-1, 1] clip to the encoded bounds instead of overshooting."""
        low, _ = decode_model_output(np.array([-7.5, -1.0, 1.0, 0.0], np.float32), _tiny_spec())
        high, _ = decode_model_output(np.array([7.5, -1.0, 1.0, 0.0], np.float32), _tiny_spec())

        assert low["cutoff"] == pytest.approx(0.0)
        assert high["cutoff"] == pytest.approx(1.0)

    def test_categorical_logits_decode_to_nearest_raw_value(self):
        """Onehot positions survive the rescale: the larger logit picks the raw_value."""
        pick_analog = np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float32)

        synth_params, _ = decode_model_output(pick_analog, _tiny_spec())

        assert synth_params["mode"] == pytest.approx(0.75)

    def test_note_params_decode_to_native_domain(self):
        """Note params come back in their native domain, not the encoded [0, 1]."""
        row = np.array([0.0, -1.0, 1.0, 1.0], dtype=np.float32)

        _, note_params = decode_model_output(row, _tiny_spec())

        assert note_params["pitch"] == 108
