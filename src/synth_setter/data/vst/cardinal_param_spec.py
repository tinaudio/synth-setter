"""Curated host controls for the Cardinal Synth 26.02 baseline patch."""

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
)

CARDINAL_HOST_PARAMETER_TARGETS = {
    "parameter_1_v": "VCO frequency",
    "parameter_2_v": "VCO pulse width",
    "parameter_3_v": "amplitude envelope attack",
    "parameter_4_v": "amplitude envelope decay",
    "parameter_5_v": "amplitude envelope sustain",
    "parameter_6_v": "amplitude envelope release",
    "parameter_7_v": "VCA level",
    "parameter_8_v": "host output level",
    "parameter_9_v": "VCA response mode",
}

CARDINAL_PARAM_SPEC = ParamSpec(
    [
        ContinuousParameter(name="parameter_1_v", min=0.42, max=0.58),
        ContinuousParameter(name="parameter_2_v", min=0.1, max=0.9),
        ContinuousParameter(name="parameter_3_v", min=0.0, max=0.55),
        ContinuousParameter(name="parameter_4_v", min=0.1, max=0.65),
        ContinuousParameter(name="parameter_5_v", min=0.35, max=1.0),
        ContinuousParameter(name="parameter_6_v", min=0.1, max=0.65),
        ContinuousParameter(name="parameter_7_v", min=0.6, max=1.0),
        ContinuousParameter(name="parameter_8_v", min=0.7, max=0.85),
        CategoricalParameter(
            name="parameter_9_v",
            values=["exponential", "linear"],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
    ],
    [
        DiscreteLiteralParameter(name="pitch", min=48, max=72),
        NoteDurationParameter(
            name="note_start_and_end",
            max_note_duration_seconds=4.0,
        ),
    ],
)
