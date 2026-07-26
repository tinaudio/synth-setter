"""Tests for the shared offline audio-renderer factory."""

from pathlib import Path
from unittest.mock import MagicMock

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
        "synth": {
            "name": "surge_simple",
            "param_spec_name": "surge_simple",
            "plugin_path": "plugins/Surge XT.vst3",
            "plugin_state_path": "presets/surge-simple.vstpreset",
            "synth_version": "1.3.4",
        },
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
        synth={
            "name": "torchsynth_adsr",
            "param_spec_name": "torchsynth_adsr",
            "plugin_path": "torchsynth",
            "plugin_state_path": "",
            "synth_version": "1.0.2",
        },
        sample_rate=22050,
        signal_duration_seconds=0.5,
    )

    renderer = make_audio_renderer(config)

    assert isinstance(renderer, TorchSynthRenderer)
    assert (renderer.sample_rate, renderer.channels) == (22050, 2)
    assert renderer.signal_duration_seconds == 0.5


def test_make_audio_renderer_pedalboard_render_cadence_stays_lazy() -> None:
    """Render cadence defers plugin loading to each renderer call."""
    renderer = make_audio_renderer(_render_config(plugin_reload_cadence="render"))

    assert isinstance(renderer, PedalboardRenderer)
    assert renderer.plugin is None


def test_make_audio_renderer_pedalboard_preserves_process_reset_mode() -> None:
    """The factory forwards the host reset policy to Pedalboard renders."""
    renderer = make_audio_renderer(
        _render_config(plugin_reload_cadence="render", process_reset_mode="preserve")
    )

    assert isinstance(renderer, PedalboardRenderer)
    assert renderer.process_reset_mode == "preserve"


@pytest.mark.requires_vst
@pytest.mark.slow
def test_make_audio_renderer_pedalboard_once_loads_real_plugin_session() -> None:
    """The factory prepares the real cached Pedalboard session selected by the config."""
    renderer = make_audio_renderer(_render_config(plugin_reload_cadence="once"))

    assert isinstance(renderer, PedalboardRenderer)
    assert renderer.plugin is not None


def test_make_audio_renderer_pedalboard_once_loads_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached Pedalboard construction loads the configured plugin and preset.

    :param monkeypatch: Replaces native plugin construction for the CPU unit lane.
    """
    plugin = MagicMock(name="plugin")
    loaded_presets: list[tuple[object, str]] = []
    monkeypatch.setattr("synth_setter.data.vst.core.load_plugin", lambda _path: plugin)
    monkeypatch.setattr(
        "synth_setter.data.vst.core.load_preset",
        lambda loaded, path: loaded_presets.append((loaded, path)),
    )
    config = _render_config(plugin_reload_cadence="once")

    renderer = make_audio_renderer(config)

    assert isinstance(renderer, PedalboardRenderer)
    assert renderer.plugin is plugin
    assert loaded_presets == [(plugin, config.plugin_state_path)]


@pytest.mark.parametrize(
    ("cadence", "reload_each_render"),
    [("once", False), ("render", True)],
)
def test_make_audio_renderer_dawdreamer_unit_maps_reload_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cadence: str,
    reload_each_render: bool,
) -> None:
    """DawDreamer receives the configured native-plugin lifecycle policy.

    :param tmp_path: Provides a concrete packaged-map stand-in path.
    :param monkeypatch: Replaces native host construction for the CPU unit lane.
    :param cadence: Public reload cadence under test.
    :param reload_each_render: Expected renderer lifecycle flag.
    """
    map_path = tmp_path / "map.json"
    map_path.write_text("{}")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "synth_setter.data.vst.dawdreamer_runtime.ensure_dawdreamer_runtime",
        lambda _backend: None,
    )
    monkeypatch.setattr("synth_setter.resources.param_map", lambda _name: map_path)
    monkeypatch.setattr(
        "synth_setter.data.vst.param_map.load_param_map", lambda _path: MagicMock(name="map")
    )
    monkeypatch.setattr(
        "synth_setter.renderer_factory.DawDreamerRenderer",
        lambda **kwargs: captured.update(kwargs) or MagicMock(),
    )
    config = _render_config(
        renderer_backend="dawdreamer",
        plugin_reload_cadence=cadence,
    )

    make_audio_renderer(config)

    assert captured["reload_plugin_each_render"] is reload_each_render


@pytest.mark.parametrize(
    ("cadence", "reload_each_render"),
    [("once", False), ("render", True)],
)
@pytest.mark.requires_vst
@pytest.mark.slow
def test_make_audio_renderer_dawdreamer_real_maps_reload_cadence(
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
