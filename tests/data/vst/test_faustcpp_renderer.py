"""Native C++ Faust renderer contracts."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.core import extract_backend_version
from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.data.vst.renderers import AudioRenderer
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SYNTHS, SynthName

_HAS_TOOLCHAIN = shutil.which("faust") is not None and shutil.which("g++") is not None


def _config(identity: str = "faust_bright_organ", channels: int = 2) -> RenderConfig:
    return RenderConfig(
        synth=SYNTHS[SynthName(identity)],
        renderer_backend="faustcpp",
        backend_version=extract_backend_version("faustcpp") if _HAS_TOOLCHAIN else "2.88.0",
        block_size=64,
        render_contract_version=2,
        sample_rate=44_100,
        channels=channels,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-100.0,
        samples_per_render_batch=1,
        samples_per_shard=1,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )


def _midpoint_patch(identity: str) -> dict[str, float]:
    patch: dict[str, float] = {}
    for parameter in resolve_faust_param_spec(ParamSpecName(identity)).synth_params:
        if isinstance(parameter, ContinuousParameter):
            patch[parameter.name] = (parameter.min + parameter.max) / 2.0
        elif isinstance(parameter, CategoricalParameter):
            patch[parameter.name] = float(parameter.raw_values[-1])
        else:
            raise TypeError(type(parameter).__name__)
    return patch


@pytest.mark.skipif(not _HAS_TOOLCHAIN, reason="install the Faust CLI and g++")
@pytest.mark.parametrize(
    ("identity", "channels"),
    [
        ("faust_bright_organ", 2),
        ("faust_bubble", 2),
        ("faust_church_organ", 2),
        ("faust_filter_osc", 1),
    ],
)
def test_faustcpp_factory_renders_real_checked_in_source(
    identity: str, channels: int
) -> None:
    """The public factory compiles every checked-in source to audible audio.

    :param identity: Checked-in Faust source identity.
    :param channels: Expected output channel count.
    """
    from synth_setter.data.vst.faustcpp_renderer import FaustCppRenderer

    renderer = make_audio_renderer(_config(identity, channels))
    patch = _midpoint_patch(identity)
    if identity == "faust_bubble":
        patch["/bubble/drop"] = 1.0
    if identity == "faust_church_organ":
        patch["/churchOrgan/gate"] = 1.0
        patch["/churchOrgan/gain"] = 0.2
    audio = renderer.render(patch, 60, 100, (0.05, 0.3))

    assert isinstance(renderer, FaustCppRenderer)
    assert audio.shape == (channels, 176_400)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 1e-4


@pytest.fixture(scope="module")
def bubble_renderer() -> AudioRenderer:
    """Compile one renderer shared by malformed-patch cases.

    :returns: Real native bubble renderer.
    """
    if not _HAS_TOOLCHAIN:
        pytest.skip("install the Faust CLI and g++")
    return make_audio_renderer(_config("faust_bubble", channels=2))


@pytest.mark.parametrize(
    ("case", "expected_exception", "match"),
    [
        pytest.param("missing", ValueError, "missing Faust parameter", id="missing-address"),
        pytest.param("unknown", KeyError, "unknown Faust parameter", id="unknown-address"),
        pytest.param("nonfinite", ValueError, "outside native domain", id="nonfinite-value"),
        pytest.param("continuous", ValueError, "outside native domain", id="continuous-domain"),
        pytest.param(
            "categorical",
            ValueError,
            "outside discrete native domain",
            id="categorical-domain",
        ),
    ],
)
def test_faustcpp_render_rejects_malformed_patch(
    bubble_renderer: AudioRenderer,
    case: str,
    expected_exception: type[Exception],
    match: str,
) -> None:
    """Malformed patches fail before native process execution.

    :param bubble_renderer: Module-scoped real native renderer.
    :param case: Invalid patch transformation under test.
    :param expected_exception: Required validation exception type.
    :param match: Required diagnostic fragment.
    """
    patch = _midpoint_patch("faust_bubble")
    continuous_address = "/bubble/bubble/freq"
    categorical_address = "/bubble/drop"
    if case == "missing":
        del patch[continuous_address]
    elif case == "unknown":
        patch["/unknown"] = 0.5
    elif case == "nonfinite":
        patch[continuous_address] = float("nan")
    elif case == "continuous":
        patch[continuous_address] = 2_001.0
    elif case == "categorical":
        patch[categorical_address] = 0.5

    with pytest.raises(expected_exception, match=match):
        bubble_renderer.render(patch, 60, 100, (0.05, 0.3))


@pytest.mark.skipif(not _HAS_TOOLCHAIN, reason="install the Faust CLI and g++")
def test_faustcpp_render_isolates_state_across_calls() -> None:
    """A repeated A patch is bit-identical after rendering a different patch."""
    renderer = make_audio_renderer(_config("faust_filter_osc", channels=1))
    patch_a = {
        "/SINE_WAVE_OSCILLATOR_oscrs/Amplitude": -20.0,
        "/SINE_WAVE_OSCILLATOR_oscrs/Frequency": 48.0,
        "/SINE_WAVE_OSCILLATOR_oscrs/Portamento": 0.001,
    }
    patch_b = patch_a | {"/SINE_WAVE_OSCILLATOR_oscrs/Frequency": 72.0}

    first = renderer.render(patch_a, 60, 100, (0.0, 0.3))
    renderer.render(patch_b, 60, 100, (0.0, 0.3))
    repeated = renderer.render(patch_a, 60, 100, (0.0, 0.3))

    np.testing.assert_array_equal(first, repeated)


def test_faustcpp_block_size_above_cpp_int_range_raises() -> None:
    """Native request integers reject block sizes they cannot represent."""
    config = _config().model_copy(update={"block_size": 2_147_483_648})

    with pytest.raises(ValueError, match=r"block_size must be an integer in \[1, 2147483647\]"):
        make_audio_renderer(config)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        pytest.param(
            {"sample_rate": 2_147_483_648.0},
            "sample_rate exceeds native int range",
            id="sample-rate",
        ),
        pytest.param(
            {"signal_duration_seconds": 50_000.0},
            "sample_rate \\* signal_duration_seconds exceeds native int range",
            id="frame-count",
        ),
    ],
)
def test_faustcpp_render_geometry_above_cpp_int_range_raises(
    updates: dict[str, float], match: str
) -> None:
    """Native request dimensions reject values C++ cannot represent.

    :param updates: Render geometry override beyond the C++ integer range.
    :param match: Required validation diagnostic.
    """
    config = _config().model_copy(update=updates)

    with pytest.raises(ValueError, match=match):
        make_audio_renderer(config)


def test_faustcpp_missing_toolchain_reports_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing compiler provenance fails with an actionable dependency message.

    :param monkeypatch: Removes the native toolchain from ``PATH``.
    """
    monkeypatch.setenv("PATH", str(Path("/nonexistent")))

    with pytest.raises(RuntimeError, match="install the Faust CLI"):
        extract_backend_version("faustcpp")
