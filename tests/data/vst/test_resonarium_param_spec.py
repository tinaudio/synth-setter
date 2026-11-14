"""Resonarium host-normalized encoding contracts, checked against the real engine where present."""

import re

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.data.vst.resonarium_param_spec import (
    RESONARIUM_CONTROLS,
    RESONARIUM_PARAM_SPEC,
    ResonariumControl,
)

# Controls that cannot change one fresh, unmodulated, untempo'd note, editor-only, or level trims.
_OMITTED = re.compile(
    r"^(lfo|rnd|mseg)\d|^Env \d|^macro\d|^ui|^extIn|^[Ss]ample|Sync|sync$|Beat|beat$"
    r"|^(numVoices|soloParameter|polyEffectChain|masterGain)$|^(input|output)Gain wb\d$"
    r"|^useSemitones wb\d$|^(stereoResonators|distPrePostFilter)$"
)


def _control(uid: str) -> ResonariumControl:
    return next(control for control in RESONARIUM_CONTROLS if control.uid == uid)


@pytest.mark.parametrize(
    ("uid", "normalized", "expected"),
    [
        ("resonatorFrequency wb0r0", 0.0, 10.0),
        ("resonatorFrequency wb0r0", 1.0, 20000.0),
        ("resonatorFrequency wb0r0", 0.5, 1993.2592),
        ("decayTime wb2r7", 0.5, 1.875),
        ("pitch wb0r0", 0.5, 3.78),
        ("svfType wb3r1", 0.5, 1.0),
        ("couplingMode wb1", 0.74, 1.0),
        ("couplingMode wb1", 0.76, 2.0),
        ("cascadeAmount wb0", 0.333, -0.33),
    ],
)
def test_native_value_applies_juce_skew_and_snapping(
    uid: str, normalized: float, expected: float
) -> None:
    """A host-normalized position maps to the skewed, interval-snapped native value.

    :param uid: Native control identifier.
    :param normalized: Host automation position.
    :param expected: Native value the engine stores for that position.
    """
    assert _control(uid).native_value(normalized) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("normalized", [-0.01, 1.01, float("nan")])
def test_native_value_outside_unit_interval_is_rejected(normalized: float) -> None:
    """Positions outside [0, 1] never reach the engine.

    :param normalized: Invalid host position.
    """
    with pytest.raises(ValueError, match="decayTime wb0r0"):
        _control("decayTime wb0r0").native_value(normalized)


def test_enumerations_sample_only_legal_positions() -> None:
    """Stepped controls are categorical over their legal normalized positions."""
    filter_type = next(
        p for p in RESONARIUM_PARAM_SPEC.synth_params if p.name == "impExciter0filterType"
    )
    assert isinstance(filter_type, CategoricalParameter)
    assert filter_type.raw_values == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def test_fine_interval_controls_are_continuous() -> None:
    """A 0.01-step control is sampled continuously and snapped by the engine mapping."""
    pick = next(
        p for p in RESONARIUM_PARAM_SPEC.synth_params if p.name == "impExciter0 pickPosition"
    )
    assert isinstance(pick, ContinuousParameter)


def test_spec_covers_every_control_once_in_table_order() -> None:
    """Spec coordinates are the control table, unique and in order."""
    names = [parameter.name for parameter in RESONARIUM_PARAM_SPEC.synth_params]
    assert names == [control.uid for control in RESONARIUM_CONTROLS]
    assert len(set(names)) == len(names) == 692


@pytest.mark.requires_resonarium
def test_controls_match_engine_ranges_and_omissions() -> None:
    """Every spec control exists in the engine with its range; every other control is omitted."""
    import resonarium

    engine = {info.id: info for info in resonarium.Resonarium().get_all_params()}
    for control in RESONARIUM_CONTROLS:
        info = engine[control.uid]
        assert (info.min, info.max) == pytest.approx((control.minimum, control.maximum))
    unexplained = sorted(
        uid
        for uid in engine.keys() - {c.uid for c in RESONARIUM_CONTROLS}
        if not _OMITTED.search(uid)
    )
    assert unexplained == []


@pytest.mark.requires_resonarium
@pytest.mark.parametrize("normalized", [0.0, 0.137, 0.5, 0.861, 1.0])
def test_engine_reads_back_mapped_value_for_every_control(normalized: float) -> None:
    """The engine stores each mapped value unchanged, so the mapping's snapping is the engine's.

    :param normalized: Host position written to every control.
    """
    import resonarium

    synth = resonarium.Resonarium()
    expected = [control.native_value(normalized) for control in RESONARIUM_CONTROLS]
    for control, value in zip(RESONARIUM_CONTROLS, expected, strict=True):
        synth.set_param(control.uid, value)
    stored = [synth.get_param(control.uid).value for control in RESONARIUM_CONTROLS]
    np.testing.assert_allclose(stored, expected, rtol=1e-6, atol=1e-6)
