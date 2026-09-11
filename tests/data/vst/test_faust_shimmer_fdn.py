"""Real Faust shimmer FDN source and parameter contracts."""

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

_IDENTITY = ParamSpecName("faust_shimmer_fdn")
_SAMPLE_RATE = 44_100
_BLOCK_SIZE = 128
_RENDER_SECONDS = 1.0
_DRY_WET = "/shimmerFDN/Output/dry/wet"
_LEVEL = "/shimmerFDN/Output/level"
_T60_LOW = "/shimmerFDN/FDN/T60_low"
_EXPECTED_DOMAINS = [
    ("/shimmerFDN/FDN/T60_low", 0.1, 20.0, False),
    ("/shimmerFDN/FDN/T60_high", 0.05, 20.0, False),
    ("/shimmerFDN/FDN/crossover", 200.0, 16_000.0, False),
    ("/shimmerFDN/Shimmer/transpose", -2_400.0, 2_400.0, False),
    ("/shimmerFDN/Shimmer/window", 64.0, 8_192.0, False),
    ("/shimmerFDN/Shimmer/shifted_lines/line__0", 0.0, 1.0, True),
    ("/shimmerFDN/Shimmer/shifted_lines/line__1", 0.0, 1.0, True),
    ("/shimmerFDN/Shimmer/shifted_lines/line__2", 0.0, 1.0, True),
    ("/shimmerFDN/Shimmer/shifted_lines/line__3", 0.0, 1.0, True),
    ("/shimmerFDN/Shimmer/shifted_lines/line__4", 0.0, 1.0, True),
    ("/shimmerFDN/Shimmer/shifted_lines/line__5", 0.0, 1.0, True),
    ("/shimmerFDN/Shimmer/shifted_lines/line__6", 0.0, 1.0, True),
    ("/shimmerFDN/Shimmer/shifted_lines/line__7", 0.0, 1.0, True),
    ("/shimmerFDN/Shimmer/DC_comp_max", 0.0, 12.0, False),
    ("/shimmerFDN/Output/dry/wet", 0.0, 1.0, False),
    ("/shimmerFDN/Output/level", -40.0, 12.0, False),
    ("/shimmerFDN/Safety/loop_ceiling", -40.0, 0.0, False),
    ("/shimmerFDN/Safety/energy_guard_bypass", 0.0, 1.0, True),
]


def _compile_shimmer():
    dd = import_module("dawdreamer")
    dsp = resolve_faust_dsp(_IDENTITY)
    engine = dd.RenderEngine(_SAMPLE_RATE, _BLOCK_SIZE)
    processor = engine.make_faust_processor("shimmer")
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
    engine, processor = _compile_shimmer()
    for address, value in patch.items():
        assert processor.set_parameter(address, value)
    assert engine.render(_RENDER_SECONDS)
    return np.asarray(engine.get_audio())


def test_shimmer_fdn_compiled_ui_matches_exact_spec() -> None:
    """Native Faust metadata matches every modeled address and domain."""
    _, processor = _compile_shimmer()
    descriptions = processor.get_parameters_description()

    assert len(descriptions) == len(_EXPECTED_DOMAINS)
    for item, (address, minimum, maximum, is_discrete) in zip(
        descriptions, _EXPECTED_DOMAINS, strict=True
    ):
        assert item["name"] == address
        assert item["min"] == pytest.approx(minimum)
        assert item["max"] == pytest.approx(maximum)
        assert item["isDiscrete"] is is_discrete
    assert resolve_faust_param_spec(_IDENTITY).synth_param_names == [
        address for address, _, _, _ in _EXPECTED_DOMAINS
    ]


def test_shimmer_fdn_registry_pins_source_geometry_and_digest() -> None:
    """The source, parameter, and synth registries share one immutable identity."""
    dsp = resolve_faust_dsp(_IDENTITY)
    synth = SYNTHS[SynthName(_IDENTITY)]

    assert (dsp.num_voices, dsp.outputs) == (0, 2)
    assert resolve_param_spec(_IDENTITY).encoded_width == 27
    assert synth.plugin_path == "registry://faust/faust_shimmer_fdn"
    assert synth.source_sha256 == hashlib.sha256(dsp.source.encode()).hexdigest()


def test_shimmer_fdn_sample_decode_roundtrip_has_fixed_note_mapping() -> None:
    """Sampling stores only DSP controls while returning a valid fixed render window."""
    spec = resolve_faust_param_spec(_IDENTITY)
    sampled_synth, sampled_note = spec.sample(np.random.default_rng(42))
    encoded = spec.encode(sampled_synth, sampled_note)
    decoded_synth, decoded_note = spec.decode(encoded)

    assert spec.note_params == []
    assert spec.note_param_names == []
    assert encoded.shape == (27,)
    assert decoded_synth == pytest.approx(sampled_synth)
    assert sampled_note == {"pitch": 60, "note_start_and_end": (0.0, 4.0)}
    assert decoded_note == sampled_note


def test_shimmer_fdn_dry_only_is_unit_impulse_at_sample_zero() -> None:
    """The in-memory source excites the effect once at the first sample."""
    patch = _default_patch()
    patch[_DRY_WET] = 0.0
    audio = _render(patch)

    assert audio.shape == (2, _SAMPLE_RATE)
    assert audio[:, 0] == pytest.approx(np.ones(2), abs=0.0)
    assert np.count_nonzero(audio[:, 1:]) == 0


def test_shimmer_fdn_wet_output_has_finite_audible_delayed_tail() -> None:
    """The impulse traverses the FDN into a bounded tail after its first delay."""
    audio = _render(_default_patch())

    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio[:, 1_000:]))) > 1e-5
    assert float(np.max(np.abs(audio))) <= 1.0


def test_shimmer_fdn_same_patch_recompiles_repeatably() -> None:
    """Fresh native processors render the same patch sample-for-sample."""
    patch = _default_patch()

    assert _render(patch) == pytest.approx(_render(patch), abs=1e-7)


def test_shimmer_fdn_t60_control_changes_wet_tail() -> None:
    """A modeled decay control causally changes the rendered tail."""
    short_patch = _default_patch()
    long_patch = dict(short_patch)
    short_patch[_T60_LOW] = 0.1
    long_patch[_T60_LOW] = 20.0

    short = _render(short_patch)
    long = _render(long_patch)

    assert not np.allclose(short[:, 1_000:], long[:, 1_000:], atol=1e-7, rtol=1e-5)
