"""Tests for the pedalboard-free param-spec registry helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.param_spec_registry import (
    default_plugin_path,
    param_specs,
    preset_paths,
)

_ENV_VAR = "SYNTH_SETTER_PLUGIN_PATH"
_BUNDLED_PATH = "plugins/Surge XT.vst3"

# Absent from the spec: inert or harmful under the harness's single-note,
# monophonic, no-pitch-bend playback (``core.make_midi_events``).
_OBXF_PRUNED_PARAMS = (
    "bypass",
    "pitch_bend_up_semitones",
    "pitch_bend_down_semitones",
    "pitch_bend_osc_2_only",
    "polyphony_voices",
    "glide",
    "glide_slop",
    "note_priority",
    "envelope_legato_mode",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_default_plugin_path_falls_back_to_bundle_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the override unset, the helper returns the in-repo bundle path.

    :param monkeypatch: removes ``SYNTH_SETTER_PLUGIN_PATH``.
    """
    monkeypatch.delenv(_ENV_VAR, raising=False)
    assert default_plugin_path() == _BUNDLED_PATH


def test_default_plugin_path_uses_env_override_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty ``SYNTH_SETTER_PLUGIN_PATH`` overrides the bundle path.

    :param monkeypatch: sets a custom override path.
    """
    monkeypatch.setenv(_ENV_VAR, "/custom/Surge XT.vst3")
    assert default_plugin_path() == "/custom/Surge XT.vst3"


def test_default_plugin_path_falls_back_to_bundle_when_env_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty override falls back to the bundle (the ``or`` semantic).

    :param monkeypatch: sets an empty ``SYNTH_SETTER_PLUGIN_PATH``.
    """
    monkeypatch.setenv(_ENV_VAR, "")
    assert default_plugin_path() == _BUNDLED_PATH


def test_param_spec_widths_match_known_values() -> None:
    """Hardcoded width tripwires guard the shipped specs against silent drift."""
    assert len(param_specs["surge_xt"]) == 300
    assert len(param_specs["surge_simple"]) == 92
    assert len(param_specs["surge_4"]) == 7
    assert len(param_specs["obxf"]) == 187


def test_every_param_spec_has_a_preset_path() -> None:
    """``param_specs`` and ``preset_paths`` cover the same keys — no spec lacks a preset."""
    assert set(param_specs) == set(preset_paths)


def test_obxf_is_registered_with_an_existing_preset() -> None:
    """``obxf`` is keyed in both registry dicts and its preset file is in the tree."""
    assert "obxf" in param_specs
    assert "obxf" in preset_paths
    assert (_REPO_ROOT / preset_paths["obxf"]).is_file()


def test_obxf_spec_encode_decode_round_trip_preserves_values_and_shape() -> None:
    """A sampled OB-Xf param set survives encode → decode in keys, values, and width."""
    spec = param_specs["obxf"]

    synth, note = spec.sample()
    encoded = spec.encode(synth, note)
    decoded_synth, decoded_note = spec.decode(encoded)

    assert encoded.shape == (187,)
    assert encoded.dtype == np.float32
    assert np.all((encoded >= 0.0) & (encoded <= 1.0))
    assert not np.any(np.isnan(encoded))
    assert not np.any(np.isinf(encoded))
    assert set(decoded_synth) == set(spec.synth_param_names)
    assert set(decoded_note) == set(spec.note_param_names)
    # encode() stores float32, so the round trip drifts ~1e-7 from the float64
    # samples; pin an explicit float32 tolerance and keep ``pitch`` an exact int.
    assert decoded_synth == pytest.approx(synth, abs=1e-6)
    assert decoded_note["pitch"] == note["pitch"]
    assert decoded_note["note_start_and_end"] == pytest.approx(
        note["note_start_and_end"], abs=1e-6
    )


def test_obxf_spec_has_94_synth_params_after_prune() -> None:
    """Pin the synth param count and encoded width; catching a count-preserving tensor reshape requires both."""
    spec = param_specs["obxf"]

    assert len(spec.synth_params) == 94
    assert spec.synth_param_length == 184
    assert spec.note_param_length == 3


@pytest.mark.parametrize("pruned", _OBXF_PRUNED_PARAMS)
def test_obxf_spec_omits_inert_param_under_single_note_harness(pruned: str) -> None:
    """Each param inert/harmful under the single-note harness is absent from the spec.

    ``bypass`` is the load-bearing case — sampling it would silence renders.

    :param pruned: Name of a pruned param expected to be absent from the spec.
    """
    assert pruned not in param_specs["obxf"].synth_param_names
