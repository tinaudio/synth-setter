"""Contracts for the full Ultramaster KR-106 parameter surface."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.vst.param_map import load_param_map
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    decode_model_output,
)
from synth_setter.data.vst.param_spec_registry import param_specs, plugin_state_paths
from synth_setter.data.vst.renderers import DawDreamerRenderer
from synth_setter.resources import as_file, param_map
from synth_setter.synth_spec import SYNTHS, SynthName, validate_synth_identity

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLUGIN_PATH = _REPO_ROOT / "plugins" / "Ultramaster KR-106.vst3"
_EXPECTED_SYNTH_PARAMS = (
    "program",
    "adsr_mode",
    "bender_dco",
    "bender_vcf",
    "bender_lfo",
    "arp_rate",
    "lfo_rate",
    "lfo_delay",
    "dco_lfo",
    "dco_pwm",
    "dco_sub",
    "dco_noise",
    "hpf",
    "vcf_freq",
    "vcf_res",
    "vcf_env",
    "vcf_lfo",
    "vcf_kbd",
    "volume",
    "attack",
    "decay",
    "sustain",
    "release",
    "transpose",
    "hold",
    "arpeggio",
    "pulse",
    "saw",
    "sub_sw",
    "chorus_off",
    "chorus_i",
    "chorus_ii",
    "octave",
    "arp_mode",
    "arp_range",
    "lfo_mode",
    "pwm_mode",
    "vcf_env_inv",
    "vca_mode",
    "bender",
    "tuning",
    "power",
    "porta_mode",
    "porta_rate",
    "transpose_offset",
    "master_volume",
    "voices",
    "vcf_oversample",
    "ignore_velocity",
    "arp_limit_kbd",
    "arp_sync_host",
    "lfo_sync_host",
    "mono_retrigger",
    "send_midi_sysex",
    "arp_quantize",
    "lfo_quantize",
    "oscillator_mode",
    "bypass",
)
_SINGLE_NOTE_SYNTH_PARAMS = (
    "adsr_mode",
    "bender_dco",
    "bender_vcf",
    "bender_lfo",
    "lfo_rate",
    "lfo_delay",
    "dco_lfo",
    "dco_pwm",
    "dco_sub",
    "dco_noise",
    "hpf",
    "vcf_freq",
    "vcf_res",
    "vcf_env",
    "vcf_lfo",
    "vcf_kbd",
    "volume",
    "attack",
    "decay",
    "sustain",
    "release",
    "hold",
    "pulse",
    "saw",
    "sub_sw",
    "chorus_i",
    "chorus_ii",
    "octave",
    "lfo_mode",
    "pwm_mode",
    "vcf_env_inv",
    "vca_mode",
    "bender",
    "tuning",
    "porta_mode",
    "porta_rate",
    "transpose_offset",
    "master_volume",
    "vcf_oversample",
    "ignore_velocity",
    "lfo_sync_host",
    "lfo_quantize",
    "oscillator_mode",
)


def test_ultramaster_kr106_spec_covers_every_host_parameter() -> None:
    """The registered spec exposes the complete v2.5.13 automatable surface."""
    assert tuple(param_specs["ultramaster_kr106"].synth_param_names) == _EXPECTED_SYNTH_PARAMS


def test_ultramaster_kr106_spec_has_expected_full_width() -> None:
    """All controls plus note conditioning occupy the expected vector width."""
    spec = param_specs["ultramaster_kr106"]

    assert len(spec.synth_params) == 58
    assert spec.synth_param_length == 243
    assert spec.note_param_length == 3
    assert spec.encoded_width == 246


def test_ultramaster_kr106_onehot_spec_changes_only_voices_width() -> None:
    """The opt-in identity expands voices without changing another coordinate."""
    scalar = param_specs["ultramaster_kr106"]
    onehot = param_specs["ultramaster_kr106_onehot"]

    assert scalar.encoded_width == 246
    assert onehot.encoded_width == 250
    assert onehot.encoded_names[:200] == scalar.encoded_names[:200]
    assert onehot.encoded_names[200:205] == [
        "voices.0",
        "voices.1",
        "voices.2",
        "voices.3",
        "voices.4",
    ]
    assert onehot.encoded_names[205:] == scalar.encoded_names[201:]


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (0.0, [1.0, 0.0, 0.0, 0.0, 0.0]),
        (0.25, [0.0, 1.0, 0.0, 0.0, 0.0]),
        (0.5, [0.0, 0.0, 1.0, 0.0, 0.0]),
        (0.75, [0.0, 0.0, 0.0, 1.0, 0.0]),
        (1.0, [0.0, 0.0, 0.0, 0.0, 1.0]),
    ],
)
def test_ultramaster_kr106_onehot_voices_round_trip_preserves_raw_map(
    raw_value: float, expected: list[float]
) -> None:
    """Each renderer-native voices value round-trips through its onehot coordinate.

    :param raw_value: Renderer-native value for one voices setting.
    :param expected: Onehot encoding for that setting.
    """
    voices = next(
        param
        for param in param_specs["ultramaster_kr106_onehot"].synth_params
        if param.name == "voices"
    )

    assert isinstance(voices, CategoricalParameter)
    assert voices.values == [6, 7, 8, 9, 10]
    assert voices.raw_values == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert voices.encode(raw_value).tolist() == expected
    assert voices.decode(np.asarray(expected)) == raw_value


def test_ultramaster_kr106_onehot_transpose_offset_stays_scalar() -> None:
    """The high-cardinality transpose offset retains its numerical encoding."""
    transpose_offset = next(
        param
        for param in param_specs["ultramaster_kr106_onehot"].synth_params
        if param.name == "transpose_offset"
    )

    assert isinstance(transpose_offset, CategoricalParameter)
    assert transpose_offset.encoding == "scalar"
    assert len(transpose_offset) == 1


def test_ultramaster_kr106_onehot_pitch_stays_scalar_with_midi_rounding() -> None:
    """The MIDI pitch remains a numerical scalar rounded to the nearest note."""
    pitch = param_specs["ultramaster_kr106_onehot"].note_params[0]

    assert isinstance(pitch, DiscreteLiteralParameter)
    assert pitch.encoding == "scalar"
    assert len(pitch) == 1
    assert pitch.decode(np.asarray([0.521])) == 61


def test_ultramaster_kr106_onehot_identity_reuses_renderer_artifacts() -> None:
    """The opt-in identity changes only its registry and ParamSpec names."""
    scalar = SYNTHS[SynthName("ultramaster_kr106")]
    onehot = SYNTHS[SynthName("ultramaster_kr106_onehot")]

    assert scalar.name == "ultramaster_kr106"
    assert scalar.param_spec_name == "ultramaster_kr106"
    assert onehot.name == "ultramaster_kr106_onehot"
    assert onehot.param_spec_name == "ultramaster_kr106_onehot"
    assert onehot.plugin_path == scalar.plugin_path
    assert onehot.plugin_state_path == scalar.plugin_state_path
    assert onehot.synth_version == scalar.synth_version


def test_ultramaster_kr106_single_note_spec_contains_only_audible_controls() -> None:
    """The fresh single-note identity excludes controls without audible variation."""
    spec = param_specs["ultramaster_kr106_single_note"]

    assert tuple(spec.synth_param_names) == _SINGLE_NOTE_SYNTH_PARAMS
    assert len(spec.synth_params) == 43
    assert spec.synth_param_length == 78
    assert spec.note_param_length == 3
    assert spec.encoded_width == 81


def test_ultramaster_kr106_single_note_spec_uses_canonical_host_values() -> None:
    """Aliased host settings collapse to one renderer-native value."""
    categorical_params = {
        param.name: param
        for param in param_specs["ultramaster_kr106_single_note"].synth_params
        if isinstance(param, CategoricalParameter)
    }

    assert categorical_params["porta_mode"].values == ["Mono", "Poly I"]
    assert categorical_params["porta_mode"].raw_values == [0.0, 0.5]
    assert categorical_params["vcf_oversample"].values == ["Off", "2x", "4x"]
    assert categorical_params["vcf_oversample"].raw_values == [0.0, 1.0 / 3.0, 1.0]


def test_ultramaster_kr106_single_note_spec_round_trip_preserves_values() -> None:
    """A single-note sample survives the public ParamSpec codec."""
    spec = param_specs["ultramaster_kr106_single_note"]
    synth, note = spec.sample(np.random.default_rng(3309))

    decoded_synth, decoded_note = spec.decode(spec.encode(synth, note))

    assert decoded_synth == pytest.approx(synth, abs=1e-6)
    assert decoded_note["pitch"] == note["pitch"]
    assert decoded_note["note_start_and_end"] == pytest.approx(
        note["note_start_and_end"], abs=1e-6
    )


def test_ultramaster_kr106_spec_round_trip_preserves_values() -> None:
    """A deterministic sample survives encoding and decoding."""
    spec = param_specs["ultramaster_kr106"]
    synth, note = spec.sample(np.random.default_rng(106))

    encoded = spec.encode(synth, note)
    decoded_synth, decoded_note = spec.decode(encoded)

    assert encoded.dtype == np.float32
    assert encoded.shape == (spec.encoded_width,)
    assert np.isfinite(encoded).all()
    assert ((0.0 <= encoded) & (encoded <= 1.0)).all()
    assert decoded_synth == pytest.approx(synth, abs=1e-6)
    assert decoded_note["pitch"] == note["pitch"]
    assert decoded_note["note_start_and_end"] == pytest.approx(
        note["note_start_and_end"], abs=1e-6
    )


def test_ultramaster_kr106_master_volume_samples_calibrated_range() -> None:
    """Post-chorus gain remains variable within its calibrated ceiling."""
    master_volume = next(
        param
        for param in param_specs["ultramaster_kr106"].synth_params
        if param.name == "master_volume"
    )

    assert isinstance(master_volume, ContinuousParameter)
    assert master_volume.min == 0.0
    assert master_volume.max == 0.25
    assert master_volume.constant_val_p == 0.0


def test_ultramaster_kr106_command_and_silence_states_are_not_sampled() -> None:
    """Dataset sampling excludes modes that cannot produce valid single-note audio."""
    categorical_params = {
        param.name: param
        for param in param_specs["ultramaster_kr106"].synth_params
        if isinstance(param, CategoricalParameter)
    }

    assert categorical_params["transpose"].weights == [1.0, 0.0]
    assert categorical_params["power"].weights == [0.0, 1.0]
    assert categorical_params["bypass"].weights == [1.0, 0.0]


def test_ultramaster_kr106_preset_is_committed() -> None:
    """The registered baseline resolves to a captured plugin state."""
    preset = _REPO_ROOT / plugin_state_paths["ultramaster_kr106"]
    assert preset.is_file()


def test_ultramaster_kr106_onehot_param_map_uses_new_identity() -> None:
    """The new identity packages the shared KR-106 host projection under its own name."""
    with as_file(param_map("ultramaster_kr106_onehot")) as path:
        joint_map = load_param_map(path)

    assert joint_map.param_spec_name == "ultramaster_kr106_onehot"
    assert set(joint_map.params) == set(_EXPECTED_SYNTH_PARAMS)


@pytest.mark.slow
@pytest.mark.requires_vst
def test_ultramaster_kr106_onehot_selector_renders_decoded_model_output() -> None:
    """The opt-in identity decodes and renders through its real host projection."""
    if platform.machine() != "x86_64":
        pytest.skip("Ultramaster KR-106 is source-built only on x86_64")
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        selected = compose(config_name="synth/ultramaster_kr106_onehot")
    identity = validate_synth_identity(selected)
    assert identity is not None
    plugin_path = _REPO_ROOT / identity.plugin_path
    assert plugin_path.is_dir(), f"Ultramaster KR-106 is not installed at {plugin_path}"

    spec = param_specs[identity.param_spec_name]
    sampled_synth, sampled_note = spec.sample(np.random.default_rng(3270))
    sampled_note["note_start_and_end"] = (0.05, 0.3)
    model_output = spec.encoded_to_model(spec.encode(sampled_synth, sampled_note))
    decoded_synth, decoded_note = decode_model_output(model_output, spec)
    with as_file(param_map(identity.param_spec_name)) as path:
        joint_map = load_param_map(path)
    renderer = DawDreamerRenderer(
        plugin_path=str(plugin_path),
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=0.5,
        plugin_state_path=str(_REPO_ROOT / identity.plugin_state_path),
        parameter_map=joint_map,
        reload_plugin_each_render=True,
    )

    audio = renderer.render(
        decoded_synth,
        cast(int, decoded_note["pitch"]),
        100,
        cast(tuple[float, float], decoded_note["note_start_and_end"]),
    )

    assert model_output.shape == (250,)
    assert decoded_synth["voices"] == sampled_synth["voices"]
    assert decoded_note["pitch"] == sampled_note["pitch"]
    assert audio.shape == (2, 22_050)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 1e-4


def test_ultramaster_kr106_single_note_preset_is_committed() -> None:
    """The single-note identity owns the baseline that fixes omitted safe states."""
    preset = _REPO_ROOT / plugin_state_paths["ultramaster_kr106_single_note"]
    assert preset.is_file()


def test_ultramaster_kr106_param_map_covers_full_spec_without_clap() -> None:
    """The committed cross-host map covers every KR-106 control without CLAP provenance."""
    with as_file(param_map("ultramaster_kr106")) as path:
        joint_map = load_param_map(path)

    assert set(joint_map.params) == set(_EXPECTED_SYNTH_PARAMS)
    assert joint_map.clap is None
    assert len(set(joint_map.dawdreamer_indices().values())) == len(_EXPECTED_SYNTH_PARAMS)
    assert all(
        identity.dawdreamer.name.casefold().replace(" ", "_") == name
        for name, identity in joint_map.params.items()
    )
    assert all(
        identity.dawdreamer.index < joint_map.dawdreamer.parameter_count
        for identity in joint_map.params.values()
    )


def test_ultramaster_kr106_single_note_param_map_covers_curated_spec() -> None:
    """The single-note map dispatches exactly its curated audible controls."""
    with as_file(param_map("ultramaster_kr106_single_note")) as path:
        joint_map = load_param_map(path)

    assert joint_map.param_spec_name == "ultramaster_kr106_single_note"
    assert tuple(joint_map.params) == _SINGLE_NOTE_SYNTH_PARAMS
    assert joint_map.preset_resource == "presets/ultramaster_kr106_single_note-base.vstpreset"
    assert joint_map.clap is None
    assert len(set(joint_map.dawdreamer_indices().values())) == len(_SINGLE_NOTE_SYNTH_PARAMS)
    assert all(
        identity.dawdreamer.name.casefold().replace(" ", "_") == name
        for name, identity in joint_map.params.items()
    )
    assert all(
        0 <= identity.dawdreamer.index < joint_map.dawdreamer.parameter_count
        for identity in joint_map.params.values()
    )


@pytest.mark.slow
@pytest.mark.requires_vst
def test_ultramaster_kr106_single_note_preset_fixes_omitted_safe_states() -> None:
    """The real single-note baseline disables controls omitted for safety."""
    if platform.machine() != "x86_64":
        pytest.skip("Ultramaster KR-106 is source-built only on x86_64")
    assert _PLUGIN_PATH.is_dir(), f"Ultramaster KR-106 is not installed at {_PLUGIN_PATH}"

    from synth_setter.data.vst.core import load_plugin, load_preset

    plugin = load_plugin(str(_PLUGIN_PATH))
    load_preset(
        plugin,
        str(_REPO_ROOT / plugin_state_paths["ultramaster_kr106_single_note"]),
    )

    assert plugin.parameters["arpeggio"].raw_value == 0.0  # type: ignore[attr-defined]
    assert plugin.parameters["transpose"].raw_value == 0.0  # type: ignore[attr-defined]
    assert plugin.parameters["power"].raw_value == 1.0  # type: ignore[attr-defined]
    assert plugin.parameters["bypass"].raw_value == 0.0  # type: ignore[attr-defined]
    assert plugin.parameters["chorus_off"].raw_value == 0.0  # type: ignore[attr-defined]


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.parametrize(
    ("name", "retained_raw", "alias_raw"),
    [
        pytest.param("porta_mode", 0.5, 1.0, id="poly_i-poly_ii"),
        pytest.param("voices", 0.0, 0.25, id="six-seven-voices"),
        pytest.param("voices", 0.0, 0.5, id="six-eight-voices"),
        pytest.param("voices", 0.0, 0.75, id="six-nine-voices"),
        pytest.param("voices", 0.0, 1.0, id="six-ten-voices"),
        pytest.param("vcf_oversample", 1.0 / 3.0, 2.0 / 3.0, id="two-three-times"),
    ],
)
def test_ultramaster_kr106_single_note_excluded_settings_render_identically(
    name: str, retained_raw: float, excluded_raw: float
) -> None:
    """Each excluded host setting produces a retained setting's exact audio.

    :param name: Host parameter under comparison.
    :param retained_raw: Normalized reference value.
    :param excluded_raw: Excluded normalized value with equivalent audio.
    """
    if platform.machine() != "x86_64":
        pytest.skip("Ultramaster KR-106 is source-built only on x86_64")
    assert _PLUGIN_PATH.is_dir(), f"Ultramaster KR-106 is not installed at {_PLUGIN_PATH}"

    synth_params = {
        "attack": 0.0,
        "dco_noise": 0.0,
        "master_volume": 0.5,
        "pulse": 1.0,
        "saw": 1.0,
        "sustain": 0.8,
        "volume": 0.5,
    }
    with as_file(param_map("ultramaster_kr106")) as path:
        joint_map = load_param_map(path)
    renderer = DawDreamerRenderer(
        plugin_path=str(_PLUGIN_PATH),
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=1.0,
        plugin_state_path=str(_REPO_ROOT / plugin_state_paths["ultramaster_kr106"]),
        parameter_map=joint_map,
        reload_plugin_each_render=True,
    )

    retained = renderer.render(
        {**synth_params, name: retained_raw}, 60, 100, (0.05, 0.75)
    )
    excluded = renderer.render(
        {**synth_params, name: excluded_raw}, 60, 100, (0.05, 0.75)
    )

    assert np.max(np.abs(retained)) > 1e-4
    assert np.array_equal(excluded, retained)


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.parametrize(
    ("name", "first_raw", "second_raw"),
    [
        pytest.param("porta_mode", 0.0, 0.5, id="mono-poly_i"),
        pytest.param("vcf_oversample", 0.0, 1.0 / 3.0, id="off-two-times"),
        pytest.param("vcf_oversample", 1.0 / 3.0, 1.0, id="two-four-times"),
    ],
)
def test_ultramaster_kr106_single_note_retained_categories_change_audio(
    name: str, first_raw: float, second_raw: float
) -> None:
    """Each retained host category produces distinct single-note audio.

    :param name: Host parameter under comparison.
    :param first_raw: First retained normalized value.
    :param second_raw: Second retained normalized value.
    """
    if platform.machine() != "x86_64":
        pytest.skip("Ultramaster KR-106 is source-built only on x86_64")
    assert _PLUGIN_PATH.is_dir(), f"Ultramaster KR-106 is not installed at {_PLUGIN_PATH}"

    synth_params = {
        "attack": 0.0,
        "chorus_i": 0.0,
        "chorus_ii": 0.0,
        "dco_noise": 0.0,
        "decay": 0.2,
        "hpf": 1.0 / 3.0,
        "master_volume": 0.25,
        "porta_rate": 1.0,
        "pulse": 1.0,
        "release": 0.1,
        "saw": 1.0,
        "sustain": 0.8,
        "vcf_env": 0.4,
        "vcf_freq": 0.45,
        "vcf_res": 0.7,
        "volume": 0.5,
    }
    with as_file(param_map("ultramaster_kr106_single_note")) as path:
        joint_map = load_param_map(path)
    renderer = DawDreamerRenderer(
        plugin_path=str(_PLUGIN_PATH),
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=1.0,
        plugin_state_path=str(
            _REPO_ROOT / plugin_state_paths["ultramaster_kr106_single_note"]
        ),
        parameter_map=joint_map,
        reload_plugin_each_render=True,
    )

    first = renderer.render({**synth_params, name: first_raw}, 60, 100, (0.05, 0.75))
    second = renderer.render({**synth_params, name: second_raw}, 60, 100, (0.05, 0.75))
    relative_difference = np.linalg.norm(first - second) / max(
        np.linalg.norm(first), np.linalg.norm(second)
    )

    assert np.max(np.abs(first)) > 1e-4
    assert relative_difference > 1e-3


@pytest.mark.slow
@pytest.mark.requires_vst
def test_ultramaster_kr106_single_note_dawdreamer_renders_audio() -> None:
    """The curated parameters drive a real fresh KR-106 render."""
    if platform.machine() != "x86_64":
        pytest.skip("Ultramaster KR-106 is source-built only on x86_64")
    assert _PLUGIN_PATH.is_dir(), f"Ultramaster KR-106 is not installed at {_PLUGIN_PATH}"

    spec = param_specs["ultramaster_kr106_single_note"]
    synth_params, _ = spec.sample(np.random.default_rng(3309))
    synth_params.update(
        {
            "attack": 0.0,
            "dco_noise": 0.0,
            "master_volume": 0.5,
            "porta_mode": 0.5,
            "pulse": 1.0,
            "saw": 1.0,
            "sustain": 0.8,
            "volume": 0.5,
        }
    )
    with as_file(param_map("ultramaster_kr106_single_note")) as path:
        joint_map = load_param_map(path)
    renderer = DawDreamerRenderer(
        plugin_path=str(_PLUGIN_PATH),
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=1.0,
        plugin_state_path=str(
            _REPO_ROOT / plugin_state_paths["ultramaster_kr106_single_note"]
        ),
        parameter_map=joint_map,
        reload_plugin_each_render=True,
    )

    audio = renderer.render(synth_params, 60, 100, (0.05, 0.75))

    assert audio.shape == (2, 44_100)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 1e-4


@pytest.mark.slow
@pytest.mark.requires_vst
def test_ultramaster_kr106_dawdreamer_master_volume_controls_audio() -> None:
    """The real mapped master-volume control changes valid DawDreamer audio."""
    if platform.machine() != "x86_64":
        pytest.skip("Ultramaster KR-106 is source-built only on x86_64")
    assert _PLUGIN_PATH.is_dir(), f"Ultramaster KR-106 is not installed at {_PLUGIN_PATH}"

    spec = param_specs["ultramaster_kr106"]
    sampled_params, _ = spec.sample(np.random.default_rng(106))
    synth_params = {**sampled_params, "master_volume": 1.0}
    with as_file(param_map("ultramaster_kr106")) as path:
        joint_map = load_param_map(path)
    renderer = DawDreamerRenderer(
        plugin_path=str(_PLUGIN_PATH),
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=2.0,
        plugin_state_path=str(_REPO_ROOT / plugin_state_paths["ultramaster_kr106"]),
        parameter_map=joint_map,
        reload_plugin_each_render=True,
    )

    audio = renderer.render(synth_params, 60, 100, (0.1, 1.5))
    muted = renderer.render({**synth_params, "master_volume": 0.0}, 60, 100, (0.1, 1.5))

    assert audio.shape == (2, 88_200)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 1e-4
    assert np.max(np.abs(muted)) == 0.0


@pytest.mark.slow
@pytest.mark.requires_vst
def test_ultramaster_kr106_live_plugin_matches_registered_surface() -> None:
    """The real bundled VST and committed specification expose identical names."""
    if platform.machine() != "x86_64":
        pytest.skip("Ultramaster KR-106 is source-built only on x86_64")
    assert _PLUGIN_PATH.is_dir(), f"Ultramaster KR-106 is not installed at {_PLUGIN_PATH}"

    from synth_setter.data.vst.core import load_plugin, load_preset

    plugin = load_plugin(str(_PLUGIN_PATH))
    load_preset(plugin, str(_REPO_ROOT / plugin_state_paths["ultramaster_kr106"]))

    assert len(plugin.parameters) == len(_EXPECTED_SYNTH_PARAMS)  # type: ignore[attr-defined]
    assert set(plugin.parameters) == set(_EXPECTED_SYNTH_PARAMS)  # type: ignore[attr-defined]


@pytest.mark.slow
@pytest.mark.requires_vst
def test_ultramaster_kr106_categorical_params_cover_every_host_setting() -> None:
    """Every discrete host value has an exact renderer-native value in the spec."""
    if platform.machine() != "x86_64":
        pytest.skip("Ultramaster KR-106 is source-built only on x86_64")
    assert _PLUGIN_PATH.is_dir(), f"Ultramaster KR-106 is not installed at {_PLUGIN_PATH}"

    from synth_setter.data.vst.core import load_plugin

    plugin = load_plugin(str(_PLUGIN_PATH))
    spec = param_specs["ultramaster_kr106"]
    categorical_params = {
        param.name: param for param in spec.synth_params if isinstance(param, CategoricalParameter)
    }

    for name, param in categorical_params.items():
        host_param = plugin.parameters[name]  # type: ignore[attr-defined]
        expected_labels = set()
        for value in host_param.valid_values:
            host_param.raw_value = host_param.get_raw_value_for(value)
            expected_labels.add(host_param.string_value)

        reached_labels = set()
        for raw_value in param.raw_values:
            host_param.raw_value = raw_value
            reached_labels.add(host_param.string_value)

        assert reached_labels == expected_labels, name
