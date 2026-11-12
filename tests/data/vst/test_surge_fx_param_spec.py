"""Contracts pinning the Surge FX spec to the live SurgePy engine it was introspected from."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.data.vst.surge_fx_param_spec import SURGE_FX_PARAM_SPEC

_SAMPLE_RATE = 44_100
_AUDIO_IN_TYPE = 29
_FX_TYPE_SPAN = 29
# Global slots G1 (impulse injector) and G2 (effect under test) in SurgePy's patch FX list.
_INJECTOR_SLOT = 6
_EFFECT_SLOT = 7
_INJECTOR_LEVEL_PARAM = 2
_INJECTOR_MIX_PARAM = 10
# Unused effect slots report a full-width int range; real discrete params stay far narrower.
_MAX_DISCRETE_SPAN = 64
# Surge's legacy integer automation scale, mirrored from ``SurgePyRenderer``.
_SURGE_INT_OFFSET = 0.005
_SURGE_INT_SCALE = 0.99
_TYPE_PARAM = "fx_g2_fx_type"
_RENDER_WINDOW_SECONDS = 4.0
_TAIL_FLOOR = 1e-3  # -60 dBFS
_CAPPED = [
    ("Bonsai", "fx_g2_bonsai_sensitivity"),
    ("Combulator", "fx_g2_combulator_feedback"),
    ("Delay", "fx_g2_delay_crossfeed"),
    ("Delay", "fx_g2_delay_feedback"),
    ("Delay", "fx_g2_delay_left"),
    ("Delay", "fx_g2_delay_right"),
    ("Distortion", "fx_g2_distortion_feedback"),
    ("Freq Shift", "fx_g2_freq_shift_feedback"),
    ("Freq Shift", "fx_g2_freq_shift_time"),
    ("Reverb 1", "fx_g2_reverb_1_decay_time"),
    ("Reverb 2", "fx_g2_reverb_2_decay_time"),
]


def _param(name: str) -> Any:
    """Look up one synth param of the spec by name.

    :param name: Repository parameter name.
    :returns: The matching spec parameter.
    """
    return next(p for p in SURGE_FX_PARAM_SPEC.synth_params if p.name == name)


def _surge_int(raw: float, minimum: float, maximum: float) -> int:
    """Map a normalized value to the native integer ``SurgePyRenderer`` would write.

    :param raw: Normalized value in ``[0, 1]``.
    :param minimum: Native minimum of the integer parameter.
    :param maximum: Native maximum of the integer parameter.
    :returns: Native integer value.
    """
    span = maximum - minimum
    return int((raw - _SURGE_INT_OFFSET) * span / _SURGE_INT_SCALE + 0.5) + int(minimum)


def _type_raw_values() -> dict[str, float]:
    """Map each selectable FX type name to its normalized raw value.

    :returns: Raw value keyed by SurgePy type display name.
    """
    fx_type = _param(_TYPE_PARAM)
    return dict(zip(fx_type.values, fx_type.raw_values, strict=True))


def _slug(text: str) -> str:
    """Normalize a SurgePy label into a parameter-name fragment.

    :param text: Verbatim SurgePy label.
    :returns: Lowercase underscore-joined fragment.
    """
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _engine(native_type: int) -> tuple[Any, Any]:
    """Build a fresh engine with Audio In feeding the effect at unity level and full mix.

    :param native_type: Native Surge FX type loaded into slot G2.
    :returns: The engine and its G2 slot patch node.
    """
    import surgepy

    synth = surgepy.createSurge(_SAMPLE_RATE)
    fx = synth.getPatch()["fx"]
    synth.setParamVal(fx[_INJECTOR_SLOT]["type"], _AUDIO_IN_TYPE)
    synth.setParamVal(fx[_EFFECT_SLOT]["type"], native_type)
    synth.process()
    injector = fx[_INJECTOR_SLOT]["p"]
    synth.setParamVal(injector[_INJECTOR_LEVEL_PARAM], 0.0)  # dB
    synth.setParamVal(injector[_INJECTOR_MIX_PARAM], 1.0)
    synth.process()
    return synth, fx[_EFFECT_SLOT]


def _impulse_response(synth: Any, seconds: float) -> np.ndarray:
    """Render a unit stereo impulse through the engine's audio input.

    :param synth: Engine prepared by ``_engine``.
    :param seconds: Minimum render length, rounded up to whole blocks.
    :returns: Stereo channel-leading output.
    """
    blocks = int(np.ceil(seconds * _SAMPLE_RATE / synth.getBlockSize()))
    excitation = np.zeros((2, blocks * synth.getBlockSize()), dtype=np.float32)
    excitation[:, 0] = 1.0
    output = synth.createMultiBlock(blocks)
    synth.processMultiBlockWithInput(excitation, output)
    return output


def _live_params(native_type: int) -> list[tuple[int, str, str, float, float, list[str]]]:
    """List the effect's populated slots as ``(slot, label, type, min, max, displays)``.

    :param native_type: Native Surge FX type to introspect.
    :returns: One row per populated slot, in slot order; displays are empty for floats.
    """
    synth, slot = _engine(native_type)
    rows = []
    for index, parameter in enumerate(slot["p"]):
        value_type = synth.getParamValType(parameter)
        minimum, maximum = synth.getParamMin(parameter), synth.getParamMax(parameter)
        if value_type == "int" and maximum - minimum > _MAX_DISCRETE_SPAN:
            continue
        displays = []
        if value_type in ("int", "bool"):
            for native in range(int(minimum), int(maximum) + 1):
                synth.setParamVal(parameter, native)
                displays.append(synth.getParamDisplay(parameter))
        label = synth.getParameterName(parameter.getId()).removeprefix("FX G2 ")
        rows.append((index, label, value_type, minimum, maximum, displays))
    return rows


def test_surge_fx_spec_impulse_excitation_has_no_note_params() -> None:
    """Impulse-excited effects carry no MIDI note, like the pyFDN impulse specs."""
    assert SURGE_FX_PARAM_SPEC.note_params == []


def test_surge_fx_spec_type_selector_is_onehot_and_leads_the_spec() -> None:
    """The type selector comes first so every per-type block reads against it."""
    fx_type = SURGE_FX_PARAM_SPEC.synth_params[0]
    assert isinstance(fx_type, CategoricalParameter)
    assert (fx_type.name, fx_type.encoding) == (_TYPE_PARAM, "onehot")


@pytest.mark.requires_surgepy
def test_surge_fx_spec_type_raw_values_select_named_type_on_live_engine() -> None:
    """Each type's raw value lands on the live engine type with that display name."""
    selected = {}
    for name, raw in _type_raw_values().items():
        synth, slot = _engine(_surge_int(raw, 0, _FX_TYPE_SPAN))
        selected[name] = synth.getParamDisplay(slot["type"])
    assert selected == {name: name for name in selected}


@pytest.mark.requires_surgepy
def test_surge_fx_spec_per_type_params_match_live_slot_labels() -> None:
    """Each type's params mirror its live populated slots, in slot order."""
    spec_names = [p.name for p in SURGE_FX_PARAM_SPEC.synth_params[1:]]
    expected = []
    for name, raw in _type_raw_values().items():
        rows = _live_params(_surge_int(raw, 0, _FX_TYPE_SPAN))
        labels = [label for _, label, *_ in rows]
        for index, label, *_ in rows:
            suffix = f"_p{index + 1}" if labels.count(label) > 1 else ""
            expected.append(f"fx_g2_{_slug(name)}_{_slug(label)}{suffix}")
    assert spec_names == expected


def _paired_live_params() -> list[tuple[Any, str, float, float, list[str]]]:
    """Pair each per-type spec param with its live slot's ``(type, min, max, displays)``.

    :returns: Spec parameter, then the live slot's type, range, and displays, in spec order.
    """
    live_rows = [
        row[2:]
        for raw_type in _type_raw_values().values()
        for row in _live_params(_surge_int(raw_type, 0, _FX_TYPE_SPAN))
    ]
    spec_params = SURGE_FX_PARAM_SPEC.synth_params[1:]
    return [(parameter, *row) for parameter, row in zip(spec_params, live_rows, strict=True)]


@pytest.mark.requires_surgepy
def test_surge_fx_spec_float_slots_are_continuous_params() -> None:
    """Continuous live slots stay continuous in the spec."""
    mismatched = [
        parameter.name
        for parameter, value_type, *_ in _paired_live_params()
        if value_type == "float" and not isinstance(parameter, ContinuousParameter)
    ]
    assert mismatched == []


@pytest.mark.requires_surgepy
def test_surge_fx_spec_discrete_raw_values_select_live_displays() -> None:
    """Discrete params decode to the live display each one-hot key names."""
    mismatches = []
    for parameter, value_type, minimum, maximum, displays in _paired_live_params():
        if value_type == "float":
            continue
        for value, raw in zip(parameter.values, parameter.raw_values, strict=True):
            is_bool = value_type == "bool"
            native = int(raw > 0.5) if is_bool else _surge_int(raw, minimum, maximum)
            if displays[native - int(minimum)] != value:
                mismatches.append(f"{parameter.name}={value!r}")
    assert mismatches == []


@pytest.mark.requires_surgepy
def test_surge_fx_spec_every_type_has_audible_deterministic_impulse_response() -> None:
    """Every selectable type answers an impulse audibly and identically on fresh engines."""
    failures = []
    for name, raw in _type_raw_values().items():
        native = _surge_int(raw, 0, _FX_TYPE_SPAN)
        first = _impulse_response(_engine(native)[0], seconds=0.5)
        second = _impulse_response(_engine(native)[0], seconds=0.5)
        if not np.any(first) or not np.array_equal(first, second):
            failures.append(name)
    assert failures == []


@pytest.mark.requires_surgepy
@pytest.mark.parametrize(("type_name", "param_name"), _CAPPED)
def test_surge_fx_spec_capped_param_tail_decays_within_render_window(
    type_name: str, param_name: str
) -> None:
    """A tail-driving param at its cap decays below -60 dBFS inside the 4 s window.

    :param type_name: SurgePy display name of the effect type.
    :param param_name: Capped spec parameter of that type.
    """
    native = _surge_int(_type_raw_values()[type_name], 0, _FX_TYPE_SPAN)
    names = [p.name for p in SURGE_FX_PARAM_SPEC.synth_params]
    prefix = f"fx_g2_{_slug(type_name)}_"
    type_names = [n for n in names if n.startswith(prefix)]
    slot_index = _live_params(native)[type_names.index(param_name)][0]
    synth, slot = _engine(native)
    live = slot["p"][slot_index]
    cap = _param(param_name).max
    minimum, maximum = synth.getParamMin(live), synth.getParamMax(live)
    synth.setParamVal(live, minimum + cap * (maximum - minimum))
    synth.process()
    response = _impulse_response(synth, seconds=_RENDER_WINDOW_SECONDS + 2.0)
    audible = np.flatnonzero(np.abs(response).max(axis=0) > _TAIL_FLOOR)
    assert audible[-1] / _SAMPLE_RATE < _RENDER_WINDOW_SECONDS
