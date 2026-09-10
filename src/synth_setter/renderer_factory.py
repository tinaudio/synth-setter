"""Construct offline audio renderers from the shared render configuration.

Typical usage passes the same validated ``RenderConfig`` to generation and
evaluation before calling ``renderer.render(...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import assert_never

from synth_setter.data.vst.renderers import (
    AudioRenderer,
    DawDreamerFaustRenderer,
    DawDreamerRenderer,
    PedalboardRenderer,
    SurgePyRenderer,
    TorchSynthRenderer,
)
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.synth_spec import SynthSpec
from synth_setter.workspace import operator_workspace


def anchor_render_preset(render: RenderConfig) -> RenderConfig:
    """Anchor a relative preset path to the operator workspace.

    :param render: Composed render configuration.
    :returns: Configuration with a concrete preset path.
    """
    preset = Path(render.plugin_state_path).expanduser()
    if preset.is_absolute() or not render.plugin_state_path:
        return render
    synth_values = render.synth.model_dump()
    synth_values["plugin_state_path"] = str(operator_workspace() / preset)
    return render.model_copy(update={"synth": SynthSpec.model_validate(synth_values)})


def make_audio_renderer(render_config: RenderConfig) -> AudioRenderer:
    """Construct one renderer session for the configured backend.

    :param render_config: Backend identity and host lifecycle shared across pipeline stages.
    :returns: Renderer whose native-host lifetime follows the configured reload cadence.
    :raises AssertionError: A DawDreamer config bypassed format/backend validation.
    """
    backend = render_config.renderer_backend
    synth_format = render_config.synth.format
    if backend == "pyfdn":
        from synth_setter.data.pyfdn_instrument import PyFDNRenderer

        return PyFDNRenderer(
            excitation=render_config.pyfdn_excitation or "impulse",
            param_spec_name=render_config.param_spec_name,
            synth_version=render_config.synth.synth_version,
            plugin_path=render_config.plugin_path,
            sample_rate=render_config.sample_rate,
            channels=render_config.channels,
            signal_duration_seconds=render_config.signal_duration_seconds,
            plugin_state_path=render_config.plugin_state_path,
        )
    if backend == "dawdreamer":
        from synth_setter.data.vst.dawdreamer_runtime import ensure_dawdreamer_runtime

        if synth_format == "faust":
            ensure_dawdreamer_runtime(backend, backend_version=render_config.backend_version)
            return DawDreamerFaustRenderer(
                plugin_path=render_config.plugin_path,
                sample_rate=render_config.sample_rate,
                channels=render_config.channels,
                signal_duration_seconds=render_config.signal_duration_seconds,
                plugin_state_path=render_config.plugin_state_path,
                param_spec_name=render_config.param_spec_name,
                source_sha256=render_config.synth.source_sha256 or "",
                reload_processor_each_render=render_config.plugin_reload_cadence == "render",
            )
        if synth_format == "vst3":
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
                flush_blocks=render_config.flush_blocks,
            )
        raise AssertionError(f"unsupported DawDreamer synth format {synth_format!r}")
    if backend == "surgepy":
        from synth_setter.data.vst.param_map import load_param_map
        from synth_setter.data.vst.surgepy_runtime import ensure_surgepy_runtime
        from synth_setter.resources import as_file, param_map

        ensure_surgepy_runtime(backend, render_config.synth.synth_version)
        with as_file(param_map(render_config.param_spec_name)) as path:
            joint_map = load_param_map(path)
        return SurgePyRenderer(
            plugin_path=render_config.plugin_path,
            sample_rate=render_config.sample_rate,
            channels=render_config.channels,
            signal_duration_seconds=render_config.signal_duration_seconds,
            plugin_state_path=render_config.plugin_state_path,
            parameter_map=joint_map,
        )
    if backend == "torchsynth":
        return TorchSynthRenderer(
            plugin_path=render_config.plugin_path,
            sample_rate=render_config.sample_rate,
            channels=render_config.channels,
            signal_duration_seconds=render_config.signal_duration_seconds,
        )

    if backend == "faustwasm":
        from synth_setter.data.vst.faustwasm_renderer import FaustWasmRenderer

        if render_config.block_size is None:
            raise AssertionError("validated FaustWasm config has no block_size")
        return FaustWasmRenderer(
            plugin_path=render_config.plugin_path,
            sample_rate=render_config.sample_rate,
            channels=render_config.channels,
            signal_duration_seconds=render_config.signal_duration_seconds,
            plugin_state_path=render_config.plugin_state_path,
            block_size=render_config.block_size,
            param_spec_name=render_config.param_spec_name,
            source_sha256=render_config.synth.source_sha256 or "",
            backend_version=render_config.backend_version or "",
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
            flush_blocks=render_config.flush_blocks,
        )
    assert_never(backend)
