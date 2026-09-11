"""Emit one installed VST version for the parent version-pin test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from synth_setter.data.vst.core import extract_renderer_version

RESULT_PREFIX = "SYNTH_SETTER_VST_VERSION="


def main() -> None:
    """Read one plugin path and emit its JSON-encoded version."""
    plugin_path = Path(sys.argv[1])
    version = extract_renderer_version(plugin_path)
    print(f"{RESULT_PREFIX}{json.dumps(version)}", flush=True)


if __name__ == "__main__":
    main()
