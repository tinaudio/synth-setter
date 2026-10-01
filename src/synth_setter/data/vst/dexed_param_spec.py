"""Dexed-py 0.3.0 Preset layout without importing the native extension."""

from itertools import product

from synth_setter.data.vst.param_spec import (
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
)

DEXED_VERSION = "0.3.0"

# Preset.to_array() orders normalized fields first, then native discrete fields.
_CONTINUOUS_FIELDS = (
    ("feedback", ()),
    ("transpose", ()),
    ("pitch_mod_sensitivity", ()),
    ("lfo_speed", ()),
    ("lfo_delay", ()),
    ("lfo_pitch_mod_depth", ()),
    ("lfo_amp_mod_depth", ()),
    ("pitch_env_rates", (4,)),
    ("pitch_env_levels", (4,)),
    ("op_env_rates", (6, 4)),
    ("op_env_levels", (6, 4)),
    ("op_output_level", (6,)),
    ("op_frequency_coarse", (6,)),
    ("op_frequency_fine", (6,)),
    ("op_detune", (6,)),
    ("op_velocity_sensitivity", (6,)),
    ("op_amp_mod_sensitivity", (6,)),
    ("op_rate_scaling", (6,)),
    ("op_breakpoint", (6,)),
    ("op_left_depth", (6,)),
    ("op_right_depth", (6,)),
)
_DISCRETE_FIELDS = (
    ("osc_key_sync", (), 1),
    ("lfo_sync", (), 1),
    ("algorithm", (), 31),
    ("lfo_wave", (), 5),
    ("op_frequency_mode", (6,), 1),
    ("op_left_curve", (6,), 3),
    ("op_right_curve", (6,), 3),
)


def _coordinate_names(name: str, shape: tuple[int, ...]) -> list[str]:
    return [
        ".".join((name, *(str(index) for index in coordinate)))
        for coordinate in product(*(range(size) for size in shape))
    ]


def _synth_parameters() -> list[ContinuousParameter | DiscreteLiteralParameter]:
    continuous = [
        ContinuousParameter(key)
        for name, shape in _CONTINUOUS_FIELDS
        for key in _coordinate_names(name, shape)
    ]
    discrete = [
        DiscreteLiteralParameter(key, min=0, max=maximum)
        for name, shape, maximum in _DISCRETE_FIELDS
        for key in _coordinate_names(name, shape)
    ]
    return [*continuous, *discrete]


DEXED_PARAMETERS = tuple(_synth_parameters())
DEXED_PARAM_SPEC = ParamSpec(
    synth_params=list(DEXED_PARAMETERS),
    note_params=[
        DiscreteLiteralParameter("pitch", min=0, max=127),
        NoteDurationParameter("note_start_and_end", max_note_duration_seconds=4.0),
    ],
)
