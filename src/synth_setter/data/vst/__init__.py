"""Public re-exports for the ``synth_setter.data.vst`` package.

Importing this package is intentionally pedalboard-free: callers that need
``load_plugin`` / ``load_preset`` / ``render_params`` import from
``synth_setter.data.vst.core`` directly. The registry dicts (``param_specs``,
``preset_paths``) live in ``synth_setter.data.vst.param_spec_registry`` and are
re-exported here for backward compat.
"""

from synth_setter.data.vst.param_spec import ParamSpec
from synth_setter.data.vst.param_spec_registry import param_specs, preset_paths
from synth_setter.data.vst.surge_xt_param_spec import (
    SURGE_4_PARAM_SPEC,
    SURGE_SIMPLE_PARAM_SPEC,
    SURGE_XT_PARAM_SPEC,
)

__all__ = [
    "ParamSpec",
    "SURGE_4_PARAM_SPEC",
    "SURGE_SIMPLE_PARAM_SPEC",
    "SURGE_XT_PARAM_SPEC",
    "param_specs",
    "preset_paths",
]
