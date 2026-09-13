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
    """Compile the registered DSP into a fresh graph.

    :returns: Engine and processor with reset delay and filter state.
    """
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
    """Build a wet-only midpoint patch with pitch paths bypassed.

    :returns: Native scalar controls for isolated effect comparisons.
    :raises TypeError: A registered control has an unsupported parameter kind.
    """
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
    """Render a fixed impulse through a fresh processor.

    :param patch: Complete native control mapping.
    :returns: Stereo audio in channel-first layout.
    """
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


@pytest.mark.parametrize(
    ("address", "low", "high"),
    [
        (_GRAIN_DURATION, 0.04, 0.12),
        (_GRAIN_POSITION, 0.08, 0.25),
        (_GRAIN_JITTER, 0.0, 0.02),
        (_LFO_RATE, 0.01, 0.5),
        (_LFO_DEPTH, 0.0, 256.0),
    ],
)
def test_super_shimmer_fdn_single_texture_control_changes_wet_tail(
    address: str, low: float, high: float
) -> None:
    """Each texture control independently changes the rendered feedback tail.

    :param address: Native granular or modulation control address.
    :param low: First endpoint of the exposed control domain.
    :param high: Second endpoint, changed without altering any other control.
    """
    first_patch = _default_patch()
    first_patch[address] = low
    second_patch = dict(first_patch)
    second_patch[address] = high

    first = _render(first_patch)
    second = _render(second_patch)

    assert not np.allclose(first[:, 6_000:], second[:, 6_000:], atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("line", range(8))
def test_super_shimmer_fdn_shift_line_enabled_changes_wet_tail(line: int) -> None:
    """Each of the eight retained pitch paths contributes to the rendered tail.

    :param line: Zero-based pitch-shifting delay-line index.
    """
    bypassed_patch = _default_patch()
    bypassed_patch["/superShimmerFDN/Shimmer/transpose"] = 1200.0
    enabled_patch = dict(bypassed_patch)
    enabled_patch[f"/superShimmerFDN/Shimmer/shifted_lines/line__{line}"] = 1.0

    bypassed = _render(bypassed_patch)
    enabled = _render(enabled_patch)

    assert not np.allclose(bypassed[:, 6_000:], enabled[:, 6_000:], atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize(
    ("address", "low", "high"),
    [
        ("/superShimmerFDN/Shimmer/transpose", -1200.0, 1200.0),
        ("/superShimmerFDN/Shimmer/window", 256.0, 4096.0),
    ],
)
def test_super_shimmer_fdn_single_pitch_control_changes_wet_tail(
    address: str, low: float, high: float
) -> None:
    """Transpose and window independently affect an enabled pitch path.

    :param address: Native pitch-shifter control address.
    :param low: First control value.
    :param high: Second control value with the remaining patch fixed.
    """
    first_patch = _default_patch()
    first_patch["/superShimmerFDN/Shimmer/shifted_lines/line__0"] = 1.0
    first_patch["/superShimmerFDN/Shimmer/transpose"] = 1200.0
    first_patch[address] = low
    second_patch = dict(first_patch)
    second_patch[address] = high

    first = _render(first_patch)
    second = _render(second_patch)

    assert not np.allclose(first[:, 6_000:], second[:, 6_000:], atol=1e-7, rtol=1e-5)


def test_super_shimmer_fdn_wet_output_has_finite_bounded_tail() -> None:
    """The impulse traverses all guarded feedback paths without unstable output."""
    audio = _render(_default_patch())

    assert audio.shape == (2, 2 * _SAMPLE_RATE)
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio[:, 6_000:]))) > 1e-5
    assert float(np.max(np.abs(audio))) <= 1.0
