"""Pedalboard-free registry of ``ParamSpec`` objects keyed by name.

Importing this module pulls only the pure-Python ``*_param_spec`` modules named
in the import block below, all free of pedalboard / VST3 native deps. This is
the canonical pedalboard-free entrypoint for interpreter-only contexts
(the SkyPilot launcher, spec construction in ``synth_setter.pipeline.schemas.spec``);
``synth_setter.data.vst`` re-exports the same names for backward compat, but importing
``synth_setter.data.vst.core`` directly is what pulls pedalboard.

``synth-setter-introspect-plugin --register`` inserts entries here by line anchor
(``synth_setter.data.vst.registration.registry_with_spec``): keep the import block
contiguous and each registry dict's assignment / closing brace intact when editing by hand.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from synth_setter.data.vst.obxf_param_spec import OBXF_PARAM_SPEC
from synth_setter.data.vst.param_spec import ParamSpec
from synth_setter.data.vst.surge_xt_param_spec import (
    SURGE_4_PARAM_SPEC,
    SURGE_SIMPLE_PARAM_SPEC,
    SURGE_XT_PARAM_SPEC,
)
from synth_setter.data.vst.torchsynth_param_spec import (
    TORCHSYNTH_ADSR_PARAM_SPEC,
    TORCHSYNTH_FULL_PARAM_SPEC,
    TORCHSYNTH_SIMPLE_PARAM_SPEC,
)
from synth_setter.param_spec_name import ParamSpecName

_param_specs: dict[ParamSpecName, ParamSpec] = {
    ParamSpecName("surge_xt"): SURGE_XT_PARAM_SPEC,
    ParamSpecName("surge_simple"): SURGE_SIMPLE_PARAM_SPEC,
    ParamSpecName("surge_4"): SURGE_4_PARAM_SPEC,
    ParamSpecName("obxf"): OBXF_PARAM_SPEC,
    ParamSpecName("torchsynth_adsr"): TORCHSYNTH_ADSR_PARAM_SPEC,
    ParamSpecName("torchsynth_full"): TORCHSYNTH_FULL_PARAM_SPEC,
    ParamSpecName("torchsynth_simple"): TORCHSYNTH_SIMPLE_PARAM_SPEC,
}
param_specs = cast(Mapping[str, ParamSpec], MappingProxyType(_param_specs))

plugin_state_paths: dict[str, str] = {
    "surge_xt": "presets/surge-base.vstpreset",
    "surge_simple": "presets/surge-simple.vstpreset",
    "surge_4": "presets/surge-mini.vstpreset",
    "obxf": "presets/obxf-base.vstpreset",
    # Python backends have no preset file; the baseline patch lives in the spec module.
    "torchsynth_adsr": "",
    "torchsynth_full": "",
    "torchsynth_simple": "",
}


def resolve_param_spec(param_spec_name: ParamSpecName) -> ParamSpec:
    """Resolve a domain-typed name against the runtime-extensible registry.

    :param param_spec_name: Runtime registry key; dynamically registered names are valid.
    :returns: The exact registered specification object, without copying it.
    :raises KeyError: If the name is not registered.
    """
    try:
        return _param_specs[param_spec_name]
    except KeyError:
        raise KeyError(param_spec_name) from None


def default_plugin_path() -> str:
    """Return ``$SYNTH_SETTER_PLUGIN_PATH`` if set and non-empty, else the bundled Surge XT path.

    ``or`` (not a ``get`` default) so an empty override also falls back to the bundle.

    :returns: Resolved VST3 plugin path.
    """
    return os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"
