"""Single source of truth for Surge XT VST plugin discovery in tests.

The plugin path is overridable via ``SYNTH_SETTER_PLUGIN_PATH`` (set by CI and the
devcontainer); absent or empty, it falls back to the in-repo bundle. Importers use
``PLUGIN_PATH`` for the path and ``VST_AVAILABLE`` for the presence check that
``conftest.pytest_collection_modifyitems`` consults when auto-skipping
``requires_vst`` tests.
"""

from __future__ import annotations

from pathlib import Path

from synth_setter.data.vst.param_spec_registry import default_plugin_path

PLUGIN_PATH = default_plugin_path()

# Probed once at import: a filesystem stat, no plugin load and no network hit.
VST_AVAILABLE = Path(PLUGIN_PATH).exists()
