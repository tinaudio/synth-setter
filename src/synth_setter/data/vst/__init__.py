"""Public re-exports for the ``synth_setter.data.vst`` package.

Importing this package is intentionally pedalboard-free: callers that need
``load_plugin`` / ``load_preset`` / ``render_params`` import from
``synth_setter.data.vst.core`` directly. Registry dictionaries are loaded lazily
so parameter specifications can import ``vst.param_spec`` without a registry cycle.
"""

from typing import TYPE_CHECKING

from synth_setter.data.vst.param_map import DawDreamerParamRef, SynthParamMap
from synth_setter.data.vst.param_spec import ParamSpec
from synth_setter.data.vst.renderers import (
    AudioRenderer,
    DawDreamerRenderer,
    PedalboardRenderer,
    TorchSynthRenderer,
)
from synth_setter.data.vst.surge_xt_param_spec import (
    SURGE_4_PARAM_SPEC,
    SURGE_SIMPLE_PARAM_SPEC,
    SURGE_XT_PARAM_SPEC,
)

if TYPE_CHECKING:
    from synth_setter.data.vst.param_spec_registry import param_specs, plugin_state_paths


def __getattr__(name: str) -> object:
    """Load registry dictionaries without creating a ParamSpec import cycle.

    :param name: Requested package attribute.
    :returns: Requested registry dictionary.
    :raises AttributeError: The package does not export ``name``.
    """
    if name not in {"param_specs", "plugin_state_paths"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from synth_setter.data.vst import param_spec_registry

    return getattr(param_spec_registry, name)


__all__ = [
    "ParamSpec",
    "AudioRenderer",
    "DawDreamerRenderer",
    "DawDreamerParamRef",
    "SynthParamMap",
    "PedalboardRenderer",
    "TorchSynthRenderer",
    "SURGE_4_PARAM_SPEC",
    "SURGE_SIMPLE_PARAM_SPEC",
    "SURGE_XT_PARAM_SPEC",
    "param_specs",
    "plugin_state_paths",
]
