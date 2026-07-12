"""Runtime DawDreamer plugin introspection."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from synth_setter.data.vst.dawdreamer_map import DawDreamerPluginMap, build_dawdreamer_map


def dump_dawdreamer_plugin(
    plugin_path: Path,
    *,
    sample_rate: float = 44100,
    block_size: int = 2048,
) -> DawDreamerPluginMap:
    """Load a plugin in DawDreamer and return its normalized parameter map.

    :param plugin_path: VST3 or supported plugin bundle path.
    :param sample_rate: Temporary introspection engine sample rate.
    :param block_size: Temporary introspection engine block size.
    :returns: Name-keyed parameter map with host indices and metadata.
    """
    resolved_path = plugin_path.expanduser().resolve()
    daw: Any = import_module("dawdreamer")
    engine = daw.RenderEngine(sample_rate, block_size)
    plugin = engine.make_plugin_processor("introspect", str(resolved_path))
    descriptions = plugin.get_parameters_description()
    return build_dawdreamer_map(resolved_path, descriptions)
