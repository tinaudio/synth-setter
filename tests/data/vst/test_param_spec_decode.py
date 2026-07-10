"""Direct tests for ``param_spec.decode_model_output`` (shared inverse-scale contract)."""

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import (
    ContinuousParameter,
    DiscreteLiteralParameter,
    ParamSpec,
    decode_model_output,
)

_SPEC = ParamSpec(
    [ContinuousParameter(name="gain")],
    [DiscreteLiteralParameter(name="pitch", min=0, max=100)],
)


def test_decode_model_output_zero_maps_to_encoded_midpoint() -> None:
    """A model output of 0.0 rescales to the encoded midpoint 0.5."""
    synth_params, note_params = decode_model_output(np.array([0.0, 0.0]), _SPEC)

    assert synth_params == {"gain": 0.5}
    assert note_params == {"pitch": 50}


def test_decode_model_output_out_of_range_values_clip_to_unit_interval() -> None:
    """Values beyond [-1, 1] clip to the encoded bounds instead of overshooting."""
    synth_params, note_params = decode_model_output(np.array([37.0, -37.0]), _SPEC)

    assert synth_params == {"gain": 1.0}
    assert note_params == {"pitch": 0}


def test_decode_model_output_interior_value_rescales_linearly() -> None:
    """A non-trivial interior value maps through (x + 1) / 2 exactly."""
    synth_params, _ = decode_model_output(np.array([0.4, 0.0]), _SPEC)

    assert synth_params == {"gain": pytest.approx(0.7)}
