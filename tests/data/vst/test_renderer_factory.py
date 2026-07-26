"""Tests for the shared offline audio-renderer factory."""

import pytest

from synth_setter.data.vst.renderers import (
    DawDreamerRenderer,
    PedalboardRenderer,
    TorchSynthRenderer,
)
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer


def _render_config(**overrides: object) -> RenderConfig:
    """Return a valid render configuration with selected overrides.

    :param \\**overrides: Render fields replacing test defaults.
    :returns: Validated renderer configuration.
    """
    values: dict[str, object] = {
        "plugin_path": "plugins/Surge XT.vst3",
        "plugin_state_path": "presets/surge-base.vstpreset",
        "param_spec_name": "surge_simple",
        "renderer_version": "1.3.4",
        "sample_rate": 44100,
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "samples_per_render_batch": 2,
        "samples_per_shard": 4,
        "gui_toggle_cadence": "never",
    }
    values.update(overrides)
    return RenderConfig(**values)  # type: ignore[arg-type]


def test_make_audio_renderer_torchsynth_returns_configured_renderer() -> None:
    """The public factory constructs TorchSynth from the shared render config."""
    config = _render_config(
        renderer_backend="torchsynth",
        plugin_path="torchsynth",
        plugin_state_path="",
        param_spec_name="torchsynth_adsr",
        renderer_version="1.0.2",
        sample_rate=22050,
        signal_duration_seconds=0.5,
    )

    renderer = make_audio_renderer(config)

    assert isinstance(renderer, TorchSynthRenderer)
    assert (renderer.sample_rate, renderer.channels) == (22050, 2)
    assert renderer.signal_duration_seconds == 0.5


@pytest.mark.requires_vst
@pytest.mark.slow
def test_make_audio_renderer_pedalboard_once_loads_real_plugin_session() -> None:
    """The factory prepares the real cached Pedalboard session selected by the config."""
    renderer = make_audio_renderer(_render_config(plugin_reload_cadence="once"))

    assert isinstance(renderer, PedalboardRenderer)
    assert renderer.plugin is not None


@pytest.mark.parametrize(
    ("cadence", "reload_each_render"),
    [("once", False), ("render", True)],
)
@pytest.mark.requires_vst
@pytest.mark.slow
def test_make_audio_renderer_dawdreamer_maps_reload_cadence(
    cadence: str,
    reload_each_render: bool,
) -> None:
    """DawDreamer receives the configured native-plugin lifecycle policy.

    :param cadence: Public reload cadence under test.
    :param reload_each_render: Expected renderer lifecycle flag.
    """
    renderer = make_audio_renderer(
        _render_config(
            renderer_backend="dawdreamer",
            plugin_reload_cadence=cadence,
        )
    )

    assert isinstance(renderer, DawDreamerRenderer)
    assert renderer.reload_plugin_each_render is reload_each_render
