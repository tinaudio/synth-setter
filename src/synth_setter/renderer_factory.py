"""Construct offline audio renderers from the shared render configuration.

Typical usage passes the same validated ``RenderConfig`` to generation and
evaluation before calling ``renderer.render(...)``.
"""

from __future__ import annotations

from typing import assert_never

from synth_setter.data.vst.renderers import (
    AudioRenderer,
    DawDreamerFaustRenderer,
    DawDreamerRenderer,
    PedalboardRenderer,
    TorchSynthRenderer,
)
from synth_setter.pipeline.schemas.spec import RenderConfig


def make_audio_renderer(render_config: RenderConfig) -> AudioRenderer:
    """Construct one renderer session for the configured backend.

    :param render_config: Backend identity and host lifecycle shared across pipeline stages.
    :returns: Renderer whose native-host lifetime follows the configured reload cadence.
    """
    backend = render_config.renderer_backend
    if backend == "dawdreamer_faust":
        from synth_setter.data.vst.dawdreamer_runtime import ensure_dawdreamer_runtime

        ensure_dawdreamer_runtime(backend, renderer_version=render_config.synth.synth_version)
        return DawDreamerFaustRenderer(
            plugin_path=render_config.plugin_path,
            sample_rate=render_config.sample_rate,
            channels=render_config.channels,
            signal_duration_seconds=render_config.signal_duration_seconds,
            plugin_state_path=render_config.plugin_state_path,
            param_spec_name=render_config.param_spec_name,
            reload_processor_each_render=render_config.plugin_reload_cadence == "render",
        )
    if backend == "dawdreamer":
        from synth_setter.data.vst.dawdreamer_runtime import ensure_dawdreamer_runtime
        from synth_setter.data.vst.param_map import load_param_map
        from synth_setter.resources import as_file, param_map

        ensure_dawdreamer_runtime(backend)
        with as_file(param_map(render_config.param_spec_name)) as path:
            joint_map = load_param_map(path)
        return DawDreamerRenderer(
            plugin_path=render_config.plugin_path,
            sample_rate=render_config.sample_rate,
            channels=render_config.channels,
            signal_duration_seconds=render_config.signal_duration_seconds,
            plugin_state_path=render_config.plugin_state_path,
            parameter_map=joint_map,
            reload_plugin_each_render=render_config.plugin_reload_cadence == "render",
        )
    if backend == "torchsynth":
        return TorchSynthRenderer(
            plugin_path=render_config.plugin_path,
            sample_rate=render_config.sample_rate,
            channels=render_config.channels,
            signal_duration_seconds=render_config.signal_duration_seconds,
        )

    if backend == "pedalboard":
        plugin = None
        if render_config.plugin_reload_cadence == "once":
            from synth_setter.data.vst.core import load_plugin, load_preset

            plugin = load_plugin(render_config.plugin_path)
            load_preset(plugin, render_config.plugin_state_path)
        return PedalboardRenderer(
            plugin_path=render_config.plugin_path,
            sample_rate=render_config.sample_rate,
            channels=render_config.channels,
            signal_duration_seconds=render_config.signal_duration_seconds,
            plugin_state_path=render_config.plugin_state_path,
            plugin=plugin,
            process_reset_mode=render_config.process_reset_mode,
        )
    assert_never(backend)
