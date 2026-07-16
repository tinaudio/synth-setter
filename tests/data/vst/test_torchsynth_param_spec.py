"""Contracts for the pure-Python torchsynth param-spec module and its registry entries."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from synth_setter.data.vst.param_spec_registry import param_specs, plugin_state_paths
from synth_setter.data.vst.torchsynth_param_spec import (
    DEFAULT_PATCH,
    INFERABLE_SPEC,
    NUM_PARAMS,
    PARAM_SPEC,
    TORCHSYNTH_ADSR_PARAM_SPEC,
    TORCHSYNTH_FULL_PARAM_SPEC,
    TORCHSYNTH_SIMPLE_PARAM_SPEC,
    default_normalized_row,
)

_ALL_SPECS = {
    "torchsynth_adsr": TORCHSYNTH_ADSR_PARAM_SPEC,
    "torchsynth_simple": TORCHSYNTH_SIMPLE_PARAM_SPEC,
    "torchsynth_full": TORCHSYNTH_FULL_PARAM_SPEC,
}


def test_module_imports_without_torch_or_pedalboard() -> None:
    """The spec module stays importable in interpreter-only contexts (no torch, no pedalboard)."""
    probe = (
        "import sys; import synth_setter.data.vst.torchsynth_param_spec; "
        "leaked = [m for m in ('torch', 'torchsynth', 'pedalboard') if m in sys.modules]; "
        "assert not leaked, f'heavy imports leaked: {leaked}'"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_datamodule_reexports_pinned_spec_objects() -> None:
    """The datamodule keeps exposing the moved spec names for existing training code."""
    from synth_setter.data import torchsynth_datamodule

    assert torchsynth_datamodule.PARAM_SPEC is PARAM_SPEC
    assert torchsynth_datamodule.INFERABLE_SPEC is INFERABLE_SPEC
    assert torchsynth_datamodule.NUM_PARAMS == NUM_PARAMS


def test_pinned_spec_matches_live_voice() -> None:
    """The checked-in snapshot equals the spec extracted from a live torchsynth voice."""
    from synth_setter.data.torchsynth_datamodule import _make_renderer, _spec_from_voice

    voice = _make_renderer(44_100, 4_410).voice
    assert _spec_from_voice(voice) == PARAM_SPEC


def test_normalization_round_trip_matches_live_voice_curves() -> None:
    """Pure-Python ``to_0to1``/``from_0to1`` mirror torchsynth's curve math for every param.

    Covers linear, curved, and symmetric ranges at interior and boundary points, so a torchsynth
    curve-math change (or a mirroring bug) fails loudly here.
    """
    import torch

    from synth_setter.data.torchsynth_datamodule import _make_renderer

    voice = _make_renderer(44_100, 4_410).voice
    live_ranges = {
        (module, name): parameter.parameter_range
        for (module, name), parameter in voice.get_parameters().items()
    }
    for param in PARAM_SPEC:
        live = live_ranges[(param.module, param.name)]
        for normalized in (0.0, 0.121, 0.5, 0.878, 1.0):
            expected_human = float(live.from_0to1(torch.tensor(normalized, dtype=torch.float64)))
            human = param.from_0to1(normalized)
            assert human == pytest.approx(expected_human, abs=1e-6), (
                f"{param.module}.{param.name} from_0to1({normalized})"
            )
            assert param.to_0to1(human) == pytest.approx(normalized, abs=1e-6), (
                f"{param.module}.{param.name} to_0to1 round trip at {normalized}"
            )


def test_full_spec_names_follow_native_voice_order() -> None:
    """``torchsynth_full`` exposes every inferable param as ``module.name`` in native order."""
    assert TORCHSYNTH_FULL_PARAM_SPEC.synth_param_names == [
        f"{param.module}.{param.name}" for param in INFERABLE_SPEC
    ]
    assert len(TORCHSYNTH_FULL_PARAM_SPEC.synth_params) == NUM_PARAMS


@pytest.mark.parametrize("name", sorted(_ALL_SPECS))
def test_torchsynth_specs_registered_with_state_path_entries(name: str) -> None:
    """Each torchsynth spec resolves from the registry and has a state-path entry.

    :param name: Registry key under test.
    """
    assert param_specs[name] is _ALL_SPECS[name]
    assert name in plugin_state_paths


@pytest.mark.parametrize("name", sorted(_ALL_SPECS))
def test_torchsynth_spec_sample_encode_decode_round_trip(name: str) -> None:
    """A sampled patch survives encode → decode in keys, values, and width.

    :param name: Registry key under test.
    """
    spec = _ALL_SPECS[name]

    synth, note = spec.sample()
    encoded = spec.encode(synth, note)
    decoded_synth, decoded_note = spec.decode(encoded)

    assert encoded.shape == (len(spec),)
    assert np.all((encoded >= 0.0) & (encoded <= 1.0))
    assert decoded_synth == pytest.approx(synth, abs=1e-6)
    assert decoded_note["pitch"] == note["pitch"]


def test_spec_widths_match_known_values() -> None:
    """Hardcoded width tripwires guard the shipped specs against silent drift."""
    assert len(TORCHSYNTH_ADSR_PARAM_SPEC) == 8
    assert len(TORCHSYNTH_SIMPLE_PARAM_SPEC) == 19
    assert len(TORCHSYNTH_FULL_PARAM_SPEC) == 79


@pytest.mark.parametrize("name", ["torchsynth_adsr", "torchsynth_simple"])
def test_partial_spec_params_are_a_subset_of_the_full_spec(name: str) -> None:
    """Reduced specs only expose knobs that exist on the voice.

    :param name: Registry key under test.
    """
    full_names = set(TORCHSYNTH_FULL_PARAM_SPEC.synth_param_names)
    assert set(_ALL_SPECS[name].synth_param_names) <= full_names


def test_default_patch_covers_exactly_the_inferable_params_within_range() -> None:
    """The baseline patch pins every inferable param to an in-range human value."""
    assert set(DEFAULT_PATCH) == {f"{p.module}.{p.name}" for p in INFERABLE_SPEC}
    for param in INFERABLE_SPEC:
        value = DEFAULT_PATCH[f"{param.module}.{param.name}"]
        assert param.minimum <= value <= param.maximum, f"{param.module}.{param.name}"


def test_default_normalized_row_round_trips_the_default_patch() -> None:
    """The precomputed normalized row denormalizes back to the human baseline patch."""
    row = default_normalized_row()
    assert len(row) == NUM_PARAMS
    for normalized, param in zip(row, INFERABLE_SPEC, strict=True):
        assert 0.0 <= normalized <= 1.0
        expected = DEFAULT_PATCH[f"{param.module}.{param.name}"]
        assert param.from_0to1(normalized) == pytest.approx(expected, abs=1e-6)


def test_default_patch_disables_modulation_except_amp_envelope_routing() -> None:
    """The baseline routes adsr_1 as the amp envelope and zeroes every other mod path.

    Reduced specs rely on this: an un-sampled mod-matrix slot must not inject
    modulation, and the amp route must be open or every render would be silent.
    """
    amp_routes = {
        "mod_matrix.adsr_1->vco_1_amp",
        "mod_matrix.adsr_1->vco_2_amp",
        "mod_matrix.adsr_1->noise_amp",
    }
    pitch_routes = {
        "mod_matrix.adsr_2->vco_1_pitch",
        "mod_matrix.adsr_2->vco_2_pitch",
        "mod_matrix.lfo_1->vco_1_pitch",
        "mod_matrix.lfo_1->vco_2_pitch",
    }
    for name, value in DEFAULT_PATCH.items():
        if not name.startswith("mod_matrix."):
            continue
        if name in amp_routes or name in pitch_routes:
            assert value == 1.0, name
        else:
            assert value == 0.0, name
    # Pitch routes stay silent at baseline because both VCO pitch-mod depths are zero.
    assert DEFAULT_PATCH["vco_1.mod_depth"] == 0.0
    assert DEFAULT_PATCH["vco_2.mod_depth"] == 0.0
    assert DEFAULT_PATCH["lfo_1.mod_depth"] == 0.0
    assert DEFAULT_PATCH["lfo_2.mod_depth"] == 0.0
