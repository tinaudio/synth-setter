"""Native Dexed parameter encoding contracts."""

import numpy as np
from dexed import Preset

from synth_setter.data.vst.dexed_param_spec import DEXED_PARAM_SPEC


def test_native_parameters_match_upstream_preset_layout() -> None:
    """Named coordinates cover the exact native Preset array ordering."""
    preset = Preset(algorithm=31, lfo_wave=5)
    values = []
    for parameter in DEXED_PARAM_SPEC.synth_params:
        field, *coordinates = parameter.name.split(".")
        value = getattr(preset, field)
        if coordinates:
            value = value[tuple(map(int, coordinates))]
        values.append(value)
    np.testing.assert_array_equal(np.asarray(values, dtype=np.float32), preset.to_array())
    assert DEXED_PARAM_SPEC.synth_param_length == 145


def test_algorithm_round_trip_uses_native_integer_and_normalized_encoding() -> None:
    """Model encoding normalizes categories without changing native integer values."""
    algorithm = next(p for p in DEXED_PARAM_SPEC.synth_params if p.name == "algorithm")
    np.testing.assert_array_equal(algorithm.encode(31), [1.0])
    assert algorithm.decode(np.array([1.0])) == 31
