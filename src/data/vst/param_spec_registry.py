"""Pedalboard-free registry of ``ParamSpec`` objects keyed by name.

Importing this module pulls only ``param_spec`` and ``surge_xt_param_spec`` —
both of which are pure-Python and free of pedalboard / VST3 native deps. Use
this entrypoint from interpreter-only contexts (the SkyPilot launcher, spec
construction in ``src.pipeline.schemas.spec``) so importing the spec model
does not transitively pull pedalboard via ``src.data.vst.__init__``'s
``from src.data.vst.core import ...``.
"""

from __future__ import annotations

from src.data.vst.param_spec import ParamSpec
from src.data.vst.surge_xt_param_spec import (
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
