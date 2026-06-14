"""Single source of truth for VST plugin discovery in tests.

The target synth defaults to Surge XT and is overridable via
``SYNTH_SETTER_TEST_SYNTH`` (a key into
:data:`synth_setter.data.vst.preset_paths`), so a CI cell can point the slow
render/round-trip suite at a second synth without hardcoding. ``TEST_SYNTH``
drives ``TEST_PARAM_SPEC_NAME`` / ``TEST_PRESET_PATH``; the plugin binary is
resolved separately via ``SYNTH_SETTER_PLUGIN_PATH`` (set by CI and the
devcontainer), falling back to the in-repo Surge bundle. Importers use
``PLUGIN_PATH`` for the path and ``VST_AVAILABLE`` for the presence check that
``conftest.pytest_collection_modifyitems`` consults when auto-skipping
``requires_vst`` tests.
"""

from __future__ import annotations

import os
from pathlib import Path

from synth_setter.data.vst.param_spec_registry import default_plugin_path, preset_paths

# ``or`` (not a ``get`` default) so an empty override also falls back to Surge XT.
# The key doubles as the ``--param_spec_name`` the render CLI takes.
TEST_SYNTH = os.environ.get("SYNTH_SETTER_TEST_SYNTH") or "surge_xt"
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
