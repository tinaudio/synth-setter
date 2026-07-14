"""Load a VST3 bundle and exit non-zero unless it exposes parameters.

Single source for the per-synth load check: the Docker image build runs it
(under ``run-linux-vst-headless.sh``) to validate baked-in synths, and
``tests/docker/test_smoke.py`` runs it per plugin as the in-image smoke test.
"""

import sys
from collections.abc import Sequence

from pedalboard import VST3Plugin


def main(argv: Sequence[str] | None = None) -> None:
    """Load the bundle at ``argv[0]``, instantiating plugin ``argv[1]`` if given.

    An empty or missing ``argv[1]`` loads the bundle's sole plugin; bundles
    exposing several plugins require it.

    :param argv: Arguments after the program name; defaults to ``sys.argv[1:]``.
    :raises SystemExit: No bundle argument was given, or the bundle loaded
        but exposes no parameters (a load failure raises pedalboard's own
        ImportError instead).
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit("usage: load_vst3_check.py BUNDLE [PLUGIN_NAME]")
    bundle_path = args[0]
    plugin_name = args[1] if len(args) > 1 and args[1] else None
    plugin = VST3Plugin(bundle_path, plugin_name=plugin_name)
    param_count = len(plugin.parameters)  # type: ignore[attr-defined]
    if param_count == 0:
        raise SystemExit(f"{bundle_path}: loaded but exposes no parameters")
    # CLI check: stdout is its product.
    print(f"{bundle_path}: param_count={param_count}")  # noqa: T201


if __name__ == "__main__":
    main()
