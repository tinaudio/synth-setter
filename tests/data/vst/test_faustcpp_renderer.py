"""Native C++ Faust renderer contracts."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.core import extract_backend_version
from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
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


def test_faustcpp_missing_toolchain_reports_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing compiler provenance fails with an actionable dependency message.

    :param monkeypatch: Removes the native toolchain from ``PATH``.
    """
    monkeypatch.setenv("PATH", str(Path("/nonexistent")))

    with pytest.raises(RuntimeError, match="install the Faust CLI"):
        extract_backend_version("faustcpp")
