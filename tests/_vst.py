"""Single source of truth for VST plugin discovery in tests.

``SYNTH_SETTER_TEST_SYNTH`` (a ``preset_paths`` key, default ``surge_xt``)
drives ``TEST_SYNTH`` / ``TEST_PARAM_SPEC_NAME`` / ``TEST_PRESET_PATH`` so a CI
cell can target a second synth without hardcoding. The plugin binary resolves
separately via ``SYNTH_SETTER_PLUGIN_PATH`` (``PLUGIN_PATH`` / ``VST_AVAILABLE``);
``conftest.pytest_collection_modifyitems`` consults ``VST_AVAILABLE`` to
auto-skip ``requires_vst`` tests.
"""

from __future__ import annotations

import os
from pathlib import Path

from synth_setter.data.vst.param_spec_registry import default_plugin_path, preset_paths

# ``or`` (not a ``get`` default) so an empty override also falls back to Surge XT.
TEST_SYNTH = os.environ.get("SYNTH_SETTER_TEST_SYNTH") or "surge_xt"
# Registry key doubles as the render CLI's ``--param_spec_name``.
TEST_PARAM_SPEC_NAME = TEST_SYNTH

# Eager lookup so an unregistered TEST_SYNTH raises KeyError at import rather
# than letting a downstream render test skip or fail opaquely.
TEST_PRESET_PATH = preset_paths[TEST_SYNTH]

PLUGIN_PATH = default_plugin_path()

# Probed once at import: a filesystem stat, no plugin load and no network hit.
VST_AVAILABLE = Path(PLUGIN_PATH).exists()

# Ceiling for any VST-driving subprocess a test spawns; generous because no
# per-call tuning exists and a hung plugin load should fail, not wedge CI.
VST_SUBPROCESS_TIMEOUT_SECONDS = 600
