"""Single source of truth for Surge XT VST plugin discovery in tests.

The plugin path is overridable via ``SYNTH_SETTER_PLUGIN_PATH`` (set by CI and the
devcontainer); absent or empty, it falls back to the in-repo bundle. Importers use
``PLUGIN_PATH`` for the path and ``VST_AVAILABLE`` for the presence check that
``conftest.pytest_collection_modifyitems`` consults when auto-skipping
``requires_vst`` tests.
"""

from __future__ import annotations

import os
from pathlib import Path

# ``or`` (not a get default) so an empty override also falls back to the bundle.
PLUGIN_PATH = os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"

# Probed once at import: a filesystem stat, no plugin load and no network hit.
VST_AVAILABLE = Path(PLUGIN_PATH).exists()
