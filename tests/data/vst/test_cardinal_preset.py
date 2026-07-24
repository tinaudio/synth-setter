"""Real-plugin contract for Cardinal's mapped Rack patch."""

from __future__ import annotations

import base64
import io
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from synth_setter.data.vst.cardinal_param_spec import CARDINAL_HOST_PARAMETER_TARGETS
from synth_setter.data.vst.core import load_plugin, load_preset
from synth_setter.data.vst.param_spec_registry import param_specs, plugin_state_paths
from synth_setter.data.vst.renderers import PedalboardRenderer

pytestmark = [pytest.mark.requires_vst, pytest.mark.slow]

_BASE_PARAMS = {
    "parameter_1_v": 0.5,
    "parameter_2_v": 0.5,
    "parameter_3_v": 0.1,
    "parameter_4_v": 0.25,
    "parameter_5_v": 0.75,
    "parameter_6_v": 0.3,
    "parameter_7_v": 0.8,
    "parameter_8_v": 0.78,
    "parameter_9_v": 1.0,
}
_NOTE_WINDOW = (0.1, 1.5)
_SAMPLE_RATE = 44_100
_PLUGIN_PATH = "plugins/CardinalSynth.vst3"
_EXPECTED_HOST_PARAMETER_MAP = {
    "parameter_1_v": ("VCO frequency", "Fundamental", "VCO", 2),
    "parameter_2_v": ("VCO pulse width", "Fundamental", "VCO", 5),
    "parameter_3_v": ("amplitude envelope attack", "Fundamental", "ADSR", 0),
    "parameter_4_v": ("amplitude envelope decay", "Fundamental", "ADSR", 1),
    "parameter_5_v": ("amplitude envelope sustain", "Fundamental", "ADSR", 2),
    "parameter_6_v": ("amplitude envelope release", "Fundamental", "ADSR", 3),
    "parameter_7_v": ("VCA level", "Fundamental", "VCA-1", 0),
    "parameter_8_v": ("host output level", "Cardinal", "HostAudio2", 0),
    "parameter_9_v": ("VCA response mode", "Fundamental", "VCA-1", 1),
}


class _HostParameterMapping(BaseModel):
    """Validated Cardinal host-slot target.

    .. attribute :: model_config

       Strict parsing configuration.

    .. attribute :: host_param_id

       Zero-based Cardinal host slot.

    .. attribute :: module_id

       Rack module receiving the host value.

    .. attribute :: param_id

       Module-local parameter index.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    host_param_id: int = Field(alias="hostParamId")
    module_id: int = Field(alias="moduleId")
    param_id: int = Field(alias="paramId")


class _HostParameterMapData(BaseModel):
    """Validated mappings stored by Cardinal's HostParametersMap module.

    .. attribute :: model_config

       Strict parsing configuration.

    .. attribute :: maps

       Cardinal host-slot targets.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    maps: list[_HostParameterMapping]


class _RackModule(BaseModel):
    """Rack module fields needed to resolve host mappings.

    .. attribute :: model_config

       Strict parsing configuration.

    .. attribute :: id

       Patch-local module identifier.

    .. attribute :: plugin

       Rack plugin slug.

    .. attribute :: model

       Rack module slug.

    .. attribute :: data

       Module-specific state for boundary validation.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    id: int
    plugin: str
    model: str
    data: object = None


class _RackPatch(BaseModel):
    """Validated module list from the committed Cardinal patch.

    .. attribute :: model_config

       Strict parsing configuration.

    .. attribute :: modules

       Modules embedded in the patch.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    modules: list[_RackModule]


def _read_rack_patch() -> _RackPatch:
    preset = Path(plugin_state_paths["cardinal"]).read_bytes()
    encoded_patch = preset.split(b"patch\0", 1)[1].split(b"\0", 1)[0]
    archive = subprocess.run(  # noqa: S603 — fixed zstd decoder and in-memory input
        ["zstd", "--decompress", "--quiet", "--stdout"],
        input=base64.b64decode(encoded_patch),
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as patch_archive:
        patch_file = patch_archive.extractfile("./patch.json")
        assert patch_file is not None
        return _RackPatch.model_validate_json(patch_file.read())


@pytest.fixture(scope="module")
def cardinal_renderer() -> PedalboardRenderer:
    """Load the committed Cardinal state once for mapped-control assertions.

    :returns: Renderer with the Cardinal patch loaded.
    """
    if not Path(_PLUGIN_PATH).exists():
        pytest.skip(f"Cardinal bundle not found at {_PLUGIN_PATH}")
    plugin = load_plugin(_PLUGIN_PATH)
    load_preset(plugin, plugin_state_paths["cardinal"])
    return PedalboardRenderer(
        plugin_path=_PLUGIN_PATH,
        plugin_state_path=plugin_state_paths["cardinal"],
        sample_rate=_SAMPLE_RATE,
        channels=2,
        signal_duration_seconds=2.0,
        plugin=plugin,
        reset_plugin_before_process=False,
    )


def _render(cardinal_renderer: PedalboardRenderer, **overrides: float) -> np.ndarray:
    params = {**_BASE_PARAMS, **overrides}
    return cardinal_renderer.render(params, 60, 100, _NOTE_WINDOW)


def test_cardinal_preset_renders_finite_non_silent_stereo(
    cardinal_renderer: PedalboardRenderer,
) -> None:
    """The production preset turns one MIDI note into bounded stereo audio.

    :param cardinal_renderer: Renderer with the Cardinal patch loaded.
    """
    audio = _render(cardinal_renderer)

    assert audio.shape == (2, 2 * _SAMPLE_RATE)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) <= 1.0
    assert np.sqrt(np.mean(np.square(audio))) > 0.01


def test_cardinal_preset_exposes_every_curated_host_slot(
    cardinal_renderer: PedalboardRenderer,
) -> None:
    """Every ParamSpec control exists on the loaded Cardinal bundle.

    :param cardinal_renderer: Renderer with the Cardinal patch loaded.
    """
    plugin = cardinal_renderer.plugin
    assert plugin is not None
    assert set(CARDINAL_HOST_PARAMETER_TARGETS) == set(param_specs["cardinal"].synth_param_names)
    assert set(CARDINAL_HOST_PARAMETER_TARGETS).issubset(getattr(plugin, "parameters"))


def test_cardinal_preset_maps_every_curated_slot_to_expected_rack_control() -> None:
    """The opaque preset maps all curated slots to the documented module controls."""
    patch = _read_rack_patch()
    modules = {module.id: (module.plugin, module.model) for module in patch.modules}
    host_map_module = next(
        module for module in patch.modules if module.model == "HostParametersMap"
    )
    host_map = _HostParameterMapData.model_validate(host_map_module.data)
    actual = {
        f"parameter_{mapping.host_param_id + 1}_v": (
            CARDINAL_HOST_PARAMETER_TARGETS[f"parameter_{mapping.host_param_id + 1}_v"],
            *modules[mapping.module_id],
            mapping.param_id,
        )
        for mapping in host_map.maps
        if mapping.host_param_id < len(CARDINAL_HOST_PARAMETER_TARGETS)
    }

    assert actual == _EXPECTED_HOST_PARAMETER_MAP


def test_cardinal_frequency_mapping_moves_spectral_peak_upward(
    cardinal_renderer: PedalboardRenderer,
) -> None:
    """Host slot 1 controls VCO frequency in the committed Rack patch.

    :param cardinal_renderer: Renderer with the Cardinal patch loaded.
    """
    low = _render(cardinal_renderer, parameter_1_v=0.44)
    high = _render(cardinal_renderer, parameter_1_v=0.56)
    frequencies = np.fft.rfftfreq(low.shape[1], d=1.0 / _SAMPLE_RATE)
    low_peak = frequencies[np.argmax(np.abs(np.fft.rfft(low[0])))]
    high_peak = frequencies[np.argmax(np.abs(np.fft.rfft(high[0])))]

    assert high_peak > low_peak * 1.5


def test_cardinal_attack_mapping_reduces_early_energy(
    cardinal_renderer: PedalboardRenderer,
) -> None:
    """Host slot 3 controls amplitude-envelope attack.

    :param cardinal_renderer: Renderer with the Cardinal patch loaded.
    """
    fast = _render(cardinal_renderer, parameter_3_v=0.0)
    slow = _render(cardinal_renderer, parameter_3_v=0.55)
    early = slice(int(0.11 * _SAMPLE_RATE), int(0.16 * _SAMPLE_RATE))

    fast_rms = np.sqrt(np.mean(np.square(fast[:, early])))
    slow_rms = np.sqrt(np.mean(np.square(slow[:, early])))

    assert fast_rms > slow_rms * 2.0


def test_cardinal_output_mapping_changes_rms_monotonically(
    cardinal_renderer: PedalboardRenderer,
) -> None:
    """Host slot 8 controls output level without silencing either render.

    :param cardinal_renderer: Renderer with the Cardinal patch loaded.
    """
    quiet = _render(cardinal_renderer, parameter_8_v=0.7)
    loud = _render(cardinal_renderer, parameter_8_v=0.85)

    quiet_rms = np.sqrt(np.mean(np.square(quiet)))
    loud_rms = np.sqrt(np.mean(np.square(loud)))

    assert quiet_rms > 0.0
    assert loud_rms > quiet_rms * 1.25
