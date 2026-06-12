"""Load a VST3 bundle and exit non-zero unless it exposes parameters.

Single source for the per-synth load check: the Docker image build runs it
(under ``run-linux-vst-headless.sh``) to validate baked-in synths, and
``tests/docker/test_smoke.py`` runs it per plugin as the in-image smoke test.
"""

import sys

from pedalboard import VST3Plugin


def main() -> None:
    """Load the bundle at ``argv[1]``, instantiating plugin ``argv[2]`` if given.

    An empty or missing ``argv[2]`` loads the bundle's sole plugin; bundles
    exposing several plugins require it.

    :raises SystemExit: No bundle argument was given, or the bundle loaded
        but exposes no parameters (a load failure raises pedalboard's own
        ImportError instead).
    """
    if len(sys.argv) < 2:
        raise SystemExit("usage: load_vst3_check.py BUNDLE [PLUGIN_NAME]")
    bundle_path = sys.argv[1]
    plugin_name = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    plugin = VST3Plugin(bundle_path, plugin_name=plugin_name)
    param_count = len(plugin.parameters)  # type: ignore[attr-defined]
    if param_count == 0:
        raise SystemExit(f"{bundle_path}: loaded but exposes no parameters")
    print(f"{bundle_path}: param_count={param_count}")  # noqa: T201 — CLI check: stdout is its product


if __name__ == "__main__":
    main()
