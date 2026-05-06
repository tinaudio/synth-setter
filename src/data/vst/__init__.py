from src.data.vst.core import load_plugin, load_preset, render_params
from src.data.vst.surge_xt_param_spec import (
    SURGE_SIMPLE_PARAM_SPEC,
    SURGE_XT_PARAM_SPEC,
    SURGE_4_PARAM_SPEC,
)

param_specs = {"surge_xt": SURGE_XT_PARAM_SPEC, "surge_simple": SURGE_SIMPLE_PARAM_SPEC, "surge_4": SURGE_4_PARAM_SPEC}

preset_paths = {
    "surge_xt": "presets/surge-base.vstpreset",
    "surge_simple": "presets/surge-simple.vstpreset",
    "surge_4": "presets/surge-mini.vstpreset",
}
