"""Public re-exports for the ``synth_setter.data.vst`` package.

Importing this package is intentionally pedalboard-free: callers that need
``load_plugin`` / ``load_preset`` / ``render_params`` import from
``synth_setter.data.vst.core`` directly. The registry dicts (``param_specs``,
``plugin_state_paths``) live in ``synth_setter.data.vst.param_spec_registry`` and are
re-exported lazily here for backward compatibility.
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

_LAZY_REGISTRY_EXPORTS = frozenset({"param_specs", "plugin_state_paths"})


def __getattr__(name: str) -> object:
    """Lazily resolve registry exports to keep leaf-module imports acyclic.

    :param name: Package attribute requested by an importer.
    :returns: Registry mapping for a supported lazy export.
    :raises AttributeError: If ``name`` is not a lazy package export.
    """
    if name not in _LAZY_REGISTRY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from synth_setter.data.vst import param_spec_registry

    return getattr(param_spec_registry, name)


def __dir__() -> list[str]:
    """List eager and lazy package exports for runtime introspection.

    :returns: Sorted package attribute names.
    """
    return sorted({*globals(), *_LAZY_REGISTRY_EXPORTS})


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
