"""Real Faust super-shimmer FDN source and parameter contracts."""

from __future__ import annotations

import hashlib
from importlib import import_module

import numpy as np
import pytest

from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.faust_sources import resolve_faust_dsp
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.synth_spec import SYNTHS, SynthName

_IDENTITY = ParamSpecName("faust_super_shimmer_fdn")
_SAMPLE_RATE = 44_100
_BLOCK_SIZE = 128
_RENDER_SECONDS = 2.0
_DRY_WET = "/superShimmerFDN/Output/dry/wet"
_LEVEL = "/superShimmerFDN/Output/level"
_LFO_RATE = "/superShimmerFDN/Modulation/rate"
_LFO_DEPTH = "/superShimmerFDN/Modulation/depth"
_GRAIN_DURATION = "/superShimmerFDN/Granular/duration"
_GRAIN_POSITION = "/superShimmerFDN/Granular/position"
_GRAIN_JITTER = "/superShimmerFDN/Granular/jitter"
_EXPECTED_DOMAINS = [
    ("/superShimmerFDN/FDN/T60_low", 0.1, 20.0, False),
    ("/superShimmerFDN/FDN/T60_high", 0.05, 20.0, False),
    ("/superShimmerFDN/FDN/crossover", 200.0, 16_000.0, False),
    ("/superShimmerFDN/Modulation/rate", 0.01, 0.5, False),
    ("/superShimmerFDN/Modulation/depth", 0.0, 256.0, False),
    ("/superShimmerFDN/Shimmer/transpose", -2_400.0, 2_400.0, False),
    ("/superShimmerFDN/Shimmer/window", 64.0, 8_192.0, False),
    ("/superShimmerFDN/Shimmer/shifted_lines/line__0", 0.0, 1.0, True),
    ("/superShimmerFDN/Shimmer/shifted_lines/line__1", 0.0, 1.0, True),
    ("/superShimmerFDN/Shimmer/shifted_lines/line__2", 0.0, 1.0, True),
    ("/superShimmerFDN/Shimmer/shifted_lines/line__3", 0.0, 1.0, True),
    ("/superShimmerFDN/Shimmer/shifted_lines/line__4", 0.0, 1.0, True),
    ("/superShimmerFDN/Shimmer/shifted_lines/line__5", 0.0, 1.0, True),
    ("/superShimmerFDN/Shimmer/shifted_lines/line__6", 0.0, 1.0, True),
    ("/superShimmerFDN/Shimmer/shifted_lines/line__7", 0.0, 1.0, True),
    ("/superShimmerFDN/Shimmer/DC_comp_max", 0.0, 12.0, False),
    ("/superShimmerFDN/Granular/duration", 0.04, 0.12, False),
    ("/superShimmerFDN/Granular/position", 0.08, 0.25, False),
    ("/superShimmerFDN/Granular/jitter", 0.0, 0.02, False),
    ("/superShimmerFDN/Output/dry/wet", 0.0, 1.0, False),
    ("/superShimmerFDN/Output/level", -40.0, 12.0, False),
    ("/superShimmerFDN/Safety/loop_ceiling", -40.0, 0.0, False),
    ("/superShimmerFDN/Safety/energy_guard_bypass", 0.0, 1.0, True),
]


def _compile_super_shimmer():
    dd = import_module("dawdreamer")
    dsp = resolve_faust_dsp(_IDENTITY)
    engine = dd.RenderEngine(_SAMPLE_RATE, _BLOCK_SIZE)
    processor = engine.make_faust_processor("super_shimmer")
    processor.num_voices = dsp.num_voices
    assert processor.set_dsp_string(dsp.source)
    assert processor.compile()
    engine.load_graph([(processor, [])])
    return engine, processor


def _default_patch() -> dict[str, float]:
    patch: dict[str, float] = {}
    for parameter in resolve_faust_param_spec(_IDENTITY).synth_params:
        if isinstance(parameter, ContinuousParameter):
            patch[parameter.name] = (parameter.min + parameter.max) / 2.0
        elif isinstance(parameter, CategoricalParameter):
            patch[parameter.name] = float(parameter.raw_values[0])
        else:
            raise TypeError(type(parameter).__name__)
    patch[_DRY_WET] = 1.0
    patch[_LEVEL] = 0.0
    return patch


def _render(patch: dict[str, float]) -> np.ndarray:
    engine, processor = _compile_super_shimmer()
    for address, value in patch.items():
        assert processor.set_parameter(address, value)
    assert engine.render(_RENDER_SECONDS)
    return np.asarray(engine.get_audio())


def test_super_shimmer_fdn_compiled_ui_matches_exact_spec() -> None:
    """Native Faust metadata matches every modeled address and domain."""
    _, processor = _compile_super_shimmer()
    descriptions = processor.get_parameters_description()

    assert len(descriptions) == len(_EXPECTED_DOMAINS)
    for item, (address, minimum, maximum, is_discrete) in zip(
        descriptions, _EXPECTED_DOMAINS, strict=True
    ):
        assert item["name"] == address
        assert item["min"] == pytest.approx(minimum)
        assert item["max"] == pytest.approx(maximum)
        assert item["isDiscrete"] is is_discrete


def test_super_shimmer_fdn_registry_pins_source_geometry_and_digest() -> None:
    """Source, parameter, and synth registries share one immutable identity."""
    dsp = resolve_faust_dsp(_IDENTITY)
    synth = SYNTHS[SynthName(_IDENTITY)]

    assert (dsp.num_voices, dsp.outputs) == (0, 2)
    assert resolve_param_spec(_IDENTITY).encoded_width == 32
    assert synth.plugin_path == "registry://faust/faust_super_shimmer_fdn"
    assert synth.source_sha256 == hashlib.sha256(dsp.source.encode()).hexdigest()


def test_super_shimmer_fdn_granular_controls_change_wet_tail() -> None:
    """The fixed-ratio granular lines respond to grain timing and jitter controls."""
    compact_patch = _default_patch()
    cloud_patch = dict(compact_patch)
    compact_patch.update({_GRAIN_DURATION: 0.04, _GRAIN_POSITION: 0.08, _GRAIN_JITTER: 0.0})
    cloud_patch.update({_GRAIN_DURATION: 0.12, _GRAIN_POSITION: 0.25, _GRAIN_JITTER: 0.02})

    compact = _render(compact_patch)
    cloud = _render(cloud_patch)

    assert not np.allclose(compact[:, 6_000:], cloud[:, 6_000:], atol=1e-7, rtol=1e-5)


def test_super_shimmer_fdn_lfo_controls_change_wet_tail() -> None:
    """Phase-offset delay modulation responds to its exposed rate and depth."""
    static_patch = _default_patch()
    modulated_patch = dict(static_patch)
    static_patch[_LFO_DEPTH] = 0.0
    modulated_patch.update({_LFO_RATE: 0.5, _LFO_DEPTH: 256.0})

    static = _render(static_patch)
    modulated = _render(modulated_patch)

    assert not np.allclose(static[:, 6_000:], modulated[:, 6_000:], atol=1e-7, rtol=1e-5)


def test_super_shimmer_fdn_wet_output_has_finite_bounded_tail() -> None:
    """The impulse traverses all guarded feedback paths without unstable output."""
    audio = _render(_default_patch())

    assert audio.shape == (2, 2 * _SAMPLE_RATE)
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio[:, 6_000:]))) > 1e-5
    assert float(np.max(np.abs(audio))) <= 1.0
