"""Export a registered Faust source as a persistent FaustWasm artifact."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_module

from synth_setter.data.vst.faustwasm_artifacts import export_faustwasm_artifact
from synth_setter.synth_spec import SYNTHS, SynthName

_FAUST_SYNTH_NAMES = tuple(
    sorted(name for name, synth in SYNTHS.items() if synth.format == "faust")
)


def _parser() -> argparse.ArgumentParser:
    """Build the registry-constrained artifact export parser.

    :returns: Parser requiring a registered Faust synth and absent destination.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth", choices=_FAUST_SYNTH_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _configured_backend_version() -> str:
    """Resolve the authored FaustWasm package pin through Hydra.

    :returns: Configured backend version.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="render/faustwasm")
    return str(cfg.render.backend_version)


def main(argv: Sequence[str] | None = None) -> None:
    """Compile and atomically publish one registry-backed FaustWasm artifact.

    :param argv: Optional command-line arguments excluding the executable name.
    """
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    export_faustwasm_artifact(
        SynthName(args.synth),
        output,
        backend_version=_configured_backend_version(),
    )


if __name__ == "__main__":
    main()
