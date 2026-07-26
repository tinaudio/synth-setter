"""Single source of truth for VST plugin discovery in tests.

``SYNTH_SETTER_TEST_SYNTH`` (a ``plugin_state_paths`` key, default ``surge_xt``)
drives ``TEST_SYNTH`` / ``TEST_PARAM_SPEC_NAME`` / ``TEST_PRESET_PATH`` /
``TEST_SYNTH_VERSION`` so a CI cell can target a second synth without
hardcoding. The plugin binary resolves
separately via ``SYNTH_SETTER_PLUGIN_PATH`` (``PLUGIN_PATH`` / ``VST_AVAILABLE``);
``conftest.pytest_collection_modifyitems`` consults ``VST_AVAILABLE`` to
auto-skip ``requires_vst`` tests.
"""

from __future__ import annotations

import os
from pathlib import Path

from hydra import compose, initialize_config_module

from synth_setter.data.vst.param_spec_registry import default_plugin_path, plugin_state_paths

# ``or`` (not a ``get`` default) so an empty override also falls back to Surge XT.
TEST_SYNTH = os.environ.get("SYNTH_SETTER_TEST_SYNTH") or "surge_xt"
# Registry key doubles as the render CLI's ``--param_spec_name``.
TEST_PARAM_SPEC_NAME = TEST_SYNTH

# Eager lookup so an unregistered TEST_SYNTH raises KeyError at import rather
# than letting a downstream render test skip or fail opaquely.
TEST_PRESET_PATH = plugin_state_paths[TEST_SYNTH]


def _composed_synth_version(synth: str) -> str:
    """Read one synth group's ``synth_version`` pin through Hydra.

    :param synth: Render group name, matching the registry key.
    :returns: The ``synth_version`` that group composes to.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return str(compose(config_name=f"render/{synth}").render.synth.synth_version)


TEST_SYNTH_VERSION = _composed_synth_version(TEST_SYNTH)

PLUGIN_PATH = default_plugin_path()

# Probed once at import: a filesystem stat, no plugin load and no network hit.
VST_AVAILABLE = Path(PLUGIN_PATH).exists()

# Flat ceiling for the single-shot VST load check — no per-sample work, so it
# does not scale; dataset-building subprocesses instead use the sample-scaled
# helper in tests/conftest.py.
VST_SUBPROCESS_TIMEOUT_SECONDS = 600
