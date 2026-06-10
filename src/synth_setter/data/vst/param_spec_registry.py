"""Pedalboard-free registry of ``ParamSpec`` objects keyed by name.

Importing this module pulls only ``param_spec`` and ``surge_xt_param_spec`` —
both of which are pure-Python and free of pedalboard / VST3 native deps. This
is the canonical pedalboard-free entrypoint for interpreter-only contexts
(the SkyPilot launcher, spec construction in ``synth_setter.pipeline.schemas.spec``);
``synth_setter.data.vst`` re-exports the same names for backward compat, but importing
``synth_setter.data.vst.core`` directly is what pulls pedalboard.
"""

from __future__ import annotations

import os

from synth_setter.data.vst.param_spec import ParamSpec
from synth_setter.data.vst.surge_xt_param_spec import (
    SURGE_4_PARAM_SPEC,
    SURGE_SIMPLE_PARAM_SPEC,
    SURGE_XT_PARAM_SPEC,
)

param_specs: dict[str, ParamSpec] = {
    "surge_xt": SURGE_XT_PARAM_SPEC,
    "surge_simple": SURGE_SIMPLE_PARAM_SPEC,
    "surge_4": SURGE_4_PARAM_SPEC,
}

preset_paths: dict[str, str] = {
    "surge_xt": "presets/surge-base.vstpreset",
    "surge_simple": "presets/surge-simple.vstpreset",
    "surge_4": "presets/surge-mini.vstpreset",
}


def default_plugin_path() -> str:
    """Return ``$SYNTH_SETTER_PLUGIN_PATH`` if set and non-empty, else the bundled Surge XT path.

    ``or`` (not a ``get`` default) so an empty override also falls back to the bundle.

    :returns: Resolved VST3 plugin path.
    """
    return os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"
