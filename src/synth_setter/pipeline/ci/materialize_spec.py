#!/usr/bin/env python3
"""Compose a DatasetSpec from a Hydra experiment and write it to disk as JSON.

Used by CI smoke workflows that need an on-disk ``input_spec.json`` before running the
generator or validating spec structure. Replaces the legacy YAML-path interface with an
experiment-name interface that drives Hydra compose under the hood.

Usage::

    python -m synth_setter.pipeline.ci.materialize_spec <experiment> <output_dir>

The composed spec is written to ``<output_dir>/input_spec.json`` and echoed on stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_module
from hydra.errors import HydraException

from synth_setter.cli.generate_dataset import spec_from_cfg
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.workspace import operator_workspace

# Operator-side anchor for ``paths.*`` Hydra interpolations. Under CI
# (the only caller today) the checkout is on disk and resolves to its
# root; under a wheel install ``operator_workspace()`` falls back to
# ``$SYNTH_SETTER_WORKSPACE`` or ``Path.cwd()``.
_OPERATOR_WORKSPACE = operator_workspace()


def main() -> None:
    """Compose the named experiment + write the JSON spec under ``output_dir``."""
    if len(sys.argv) < 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <experiment> <output_dir>\n")
        sys.exit(1)

    experiment = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(config_name="dataset", overrides=[f"experiment={experiment}"])
    except HydraException as exc:
        sys.stderr.write(f"error: Hydra compose failed for experiment {experiment!r}: {exc}\n")
        sys.exit(2)
    cfg.paths.root_dir = str(_OPERATOR_WORKSPACE)
    cfg.paths.output_dir = str(_OPERATOR_WORKSPACE)
    cfg.paths.work_dir = str(_OPERATOR_WORKSPACE)
    spec = spec_from_cfg(cfg)

    spec_json = spec.model_dump_json(indent=2)
    output_path = output_dir / INPUT_SPEC_FILENAME
    output_path.write_text(spec_json)

    sys.stdout.write(spec_json + "\n")
    sys.stderr.write(f"\nSpec written to {output_path}\n")


if __name__ == "__main__":
    main()
