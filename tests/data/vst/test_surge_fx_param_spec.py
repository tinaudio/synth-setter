"""Contracts pinning the Surge FX spec to the live SurgePy engine it was introspected from."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
)
from synth_setter.data.vst.surge_fx_param_spec import (
    DRIVE_MODELS,
    FX_SLOT_COUNT,
    LFO_WAVEFORMS,
    SURGE_FX_PARAM_SPEC,
    SURGE_FX_TYPE_LAYOUTS,
    shared_native_index,
)

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
_SHARED_PARAMS = {"fx_g2_lfo_waveform": LFO_WAVEFORMS, "fx_g2_drive_model": DRIVE_MODELS}
# Phaser's native display for the canonical "Sawtooth" LFO shape.
_NATIVE_LABEL_ALIASES = {("Phaser", "Sawtooth"): "Saw"}


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


def _live_params(type_name: str) -> list[tuple[int, str, str, float, float, list[str]]]:
    """List the effect's populated slots as ``(slot, label, type, min, max, displays)``.

    :param type_name: SurgePy display name of the effect type.
    :returns: One row per populated slot, in slot order; displays are empty for floats.
    """
    synth, slot = _engine(_surge_int(_type_raw_values()[type_name], 0, _FX_TYPE_SPAN))
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


def _live_discrete(type_name: str) -> dict[int, tuple[str, float, float, list[str]]]:
    """Map each live discrete slot of a type to its native ``(type, min, max, displays)``.

    :param type_name: SurgePy display name of the effect type.
    :returns: Value type, native range and displays keyed by native slot index.
    """
    return {
        index: (value_type, minimum, maximum, displays)
        for index, _, value_type, minimum, maximum, displays in _live_params(type_name)
        if value_type != "float"
    }


def test_surge_fx_spec_impulse_excitation_has_no_note_params() -> None:
    """Impulse-excited effects carry no MIDI note, like the pyFDN impulse specs."""
    assert SURGE_FX_PARAM_SPEC.note_params == []


def test_surge_fx_spec_type_selector_is_onehot_and_leads_the_spec() -> None:
    """The type selector comes first so every per-type block reads against it."""
    fx_type = SURGE_FX_PARAM_SPEC.synth_params[0]
    assert isinstance(fx_type, CategoricalParameter)
    assert (fx_type.name, fx_type.encoding) == (_TYPE_PARAM, "onehot")


def test_surge_fx_spec_encoded_width_is_eighty_three() -> None:
    """Generic slots, scalar counts and shared categoricals encode the spec in 83 columns."""
    assert SURGE_FX_PARAM_SPEC.encoded_width == 83


def test_surge_fx_spec_has_twenty_six_logical_fields() -> None:
    """The spec has 26 logical fields: type, 12 slots and 13 discrete selectors."""
    assert len(SURGE_FX_PARAM_SPEC.synth_params) == 26


def test_surge_fx_spec_onehot_selectors_are_ten_spanning_forty_four_columns() -> None:
    """Besides the type selector, ten one-hot categoricals occupy 44 encoded columns."""
    onehots = [
        p
        for p in SURGE_FX_PARAM_SPEC.synth_params[1:]
        if isinstance(p, CategoricalParameter) and p.encoding == "onehot"
    ]
    assert (len(onehots), sum(len(p) for p in onehots)) == (10, 44)


def test_surge_fx_spec_continuous_slots_are_generic_and_uncapped() -> None:
    """The continuous block is FX_SLOT_COUNT generic slots spanning each native range."""
    slots = [p for p in SURGE_FX_PARAM_SPEC.synth_params if isinstance(p, ContinuousParameter)]
    assert [(p.name, p.min, p.max) for p in slots] == [
        (f"fx_g2_slot_{i}", 0.0, 1.0) for i in range(1, FX_SLOT_COUNT + 1)
    ]


def test_surge_fx_layouts_cover_exactly_the_selectable_types() -> None:
    """Every selectable type has a layout and no layout names an unselectable type."""
    assert list(SURGE_FX_TYPE_LAYOUTS) == _param(_TYPE_PARAM).values


def test_surge_fx_layouts_fit_the_generic_slot_count() -> None:
    """No type needs more continuous slots than the spec provides."""
    widest = max(len(layout.continuous) for layout in SURGE_FX_TYPE_LAYOUTS.values())
    assert widest == FX_SLOT_COUNT


def test_surge_fx_layouts_route_every_discrete_selector_in_the_spec() -> None:
    """Layout discrete entries are exactly the spec's non-type discrete params."""
    routed = {s.name for layout in SURGE_FX_TYPE_LAYOUTS.values() for s in layout.discrete}
    spec_discrete = {
        p.name
        for p in SURGE_FX_PARAM_SPEC.synth_params[1:]
        if not isinstance(p, ContinuousParameter)
    }
    assert routed == spec_discrete


def test_surge_fx_shared_selectors_use_canonical_positions_not_native_raw_values() -> None:
    """Shared selectors encode canonical label order; natives come from ``shared_native_index``."""
    for name, labels in _SHARED_PARAMS.items():
        parameter = _param(name)
        assert parameter.values == list(labels)
        assert parameter.raw_values == pytest.approx(np.linspace(0.0, 1.0, len(labels)))


def test_shared_native_index_rejects_type_without_that_selector() -> None:
    """Asking for a shared selector on a type that lacks it fails loudly."""
    with pytest.raises(KeyError):
        shared_native_index("Delay", "fx_g2_drive_model", "Soft")


def test_shared_native_index_rejects_label_outside_canonical_set() -> None:
    """A label absent from the canonical set fails instead of selecting a neighbour."""
    with pytest.raises(KeyError):
        shared_native_index("Waveshaper", "fx_g2_drive_model", "Additive 1+2")


@pytest.mark.requires_surgepy
def test_surge_fx_spec_type_raw_values_select_named_type_on_live_engine() -> None:
    """Each type's raw value lands on the live engine type with that display name."""
    selected = {}
    for name, raw in _type_raw_values().items():
        synth, slot = _engine(_surge_int(raw, 0, _FX_TYPE_SPAN))
        selected[name] = synth.getParamDisplay(slot["type"])
    assert selected == {name: name for name in selected}


@pytest.mark.requires_surgepy
def test_surge_fx_layout_continuous_slots_match_live_float_slots() -> None:
    """Each type's generic slots map, in order, onto its live float slots and labels."""
    mismatched = {}
    for name, layout in SURGE_FX_TYPE_LAYOUTS.items():
        rows = _live_params(name)
        labels = [label for _, label, *_ in rows]
        expected = [
            (index, _slug(label) + (f"_p{index + 1}" if labels.count(label) > 1 else ""))
            for index, label, value_type, *_ in rows
            if value_type == "float"
        ]
        actual = [(s.native_slot, s.name) for s in layout.continuous]
        if actual != expected:
            mismatched[name] = actual
    assert mismatched == {}


@pytest.mark.requires_surgepy
def test_surge_fx_layout_discrete_slots_match_live_discrete_slots() -> None:
    """Each type routes a spec selector to every live discrete slot and no other."""
    mismatched = {
        name: [s.native_slot for s in layout.discrete]
        for name, layout in SURGE_FX_TYPE_LAYOUTS.items()
        if sorted(s.native_slot for s in layout.discrete) != sorted(_live_discrete(name))
    }
    assert mismatched == {}


@pytest.mark.requires_surgepy
def test_surge_fx_per_type_onehot_raw_values_select_live_displays() -> None:
    """Per-type one-hot selectors decode to the live display each key names."""
    mismatches = []
    for name, layout in SURGE_FX_TYPE_LAYOUTS.items():
        live = _live_discrete(name)
        for routed in layout.discrete:
            parameter = _param(routed.name)
            if routed.name in _SHARED_PARAMS or not isinstance(parameter, CategoricalParameter):
                continue
            value_type, minimum, maximum, displays = live[routed.native_slot]
            for value, raw in zip(parameter.values, parameter.raw_values, strict=True):
                is_bool = value_type == "bool"
                native = int(raw > 0.5) if is_bool else _surge_int(raw, minimum, maximum)
                if displays[native - int(minimum)] != value:
                    mismatches.append(f"{routed.name}={value!r}")
    assert mismatches == []


@pytest.mark.requires_surgepy
def test_surge_fx_scalar_counts_span_the_live_native_range() -> None:
    """Scalar discrete selectors cover exactly their live native integer range."""
    mismatched = []
    for name, layout in SURGE_FX_TYPE_LAYOUTS.items():
        live = _live_discrete(name)
        for routed in layout.discrete:
            parameter = _param(routed.name)
            if not isinstance(parameter, DiscreteLiteralParameter):
                continue
            _, minimum, maximum, _ = live[routed.native_slot]
            if (parameter.min, parameter.max) != (int(minimum), int(maximum)):
                mismatched.append(routed.name)
    assert mismatched == []


@pytest.mark.requires_surgepy
def test_shared_native_index_selects_the_canonical_label_on_every_type() -> None:
    """Shared selectors land on the same-named live shape per type, never by position."""
    mismatches = []
    for name, layout in SURGE_FX_TYPE_LAYOUTS.items():
        live = _live_discrete(name)
        for routed in layout.discrete:
            if routed.name not in _SHARED_PARAMS:
                continue
            _, minimum, _, displays = live[routed.native_slot]
            for label in _SHARED_PARAMS[routed.name]:
                native = shared_native_index(name, routed.name, label)
                expected = _NATIVE_LABEL_ALIASES.get((name, label), label)
                if displays[native - int(minimum)] != expected:
                    mismatches.append(f"{name}:{label}")
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
