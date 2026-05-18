"""Print the launcher-uploaded spec's R2 URI for a given materialized spec + job name.

Used by ``.github/workflows/generate-dataset-shards.yaml`` to surface the URI
as a job/workflow output (so ``validate-dataset-shards.yaml`` can pull the
spec from R2). The previous shape — bash + ``python3 -c "import json, sys; ..."``
— inlined the URI shape in a workflow YAML; centralizing it here keeps the
``r2://{bucket}/{LAUNCHER_SPEC_R2_PREFIX}/{job_name}.json`` convention in one
Python module alongside ``upload_spec_to_r2``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from synth_setter.pipeline.constants import (
    LAUNCHER_SPEC_R2_PREFIX,
    R2_URI_SCHEME,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec


def compute_spec_uri(spec_path: Path, job_name: str) -> str:
    """Read the spec at ``spec_path`` and return the launcher-staging R2 URI for ``job_name``.

    Uses ``DatasetSpec.model_validate_json`` so legacy flat ``r2_bucket`` keys
    in pre-PR materialized specs still resolve to a valid bucket via the
    back-compat shim in ``DatasetSpec._normalize_r2``.

    :param spec_path: Local filesystem path to ``input_spec.json``.
    :param job_name: SkyPilot job name; becomes the per-job R2 key under
        ``LAUNCHER_SPEC_R2_PREFIX``.
    :returns: Fully-qualified ``r2://...`` URI string.
    :rtype: str
    """
    spec = DatasetSpec.model_validate_json(spec_path.read_text())
    return f"{R2_URI_SCHEME}{spec.r2.bucket}/{LAUNCHER_SPEC_R2_PREFIX}/{job_name}.json"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the ``synth-setter-spec-uri`` console script.

    :returns: Configured ``argparse.ArgumentParser`` for ``main`` to call ``parse_args`` on.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="synth-setter-spec-uri",
        description=(
            "Print the launcher-uploaded spec's R2 URI for a given materialized spec and job name."
        ),
    )
    parser.add_argument(
        "--spec",
        required=True,
        type=Path,
        help="Path to the materialized input_spec.json on local disk.",
    )
    parser.add_argument(
        "--job-name",
        required=True,
        help="SkyPilot job name (the per-job R2 key segment).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse argv, write the URI to stdout, return an exit code.

    Writes via ``sys.stdout.write`` (not ``print``) so the bash caller in
    ``generate-dataset-shards.yaml`` can interpolate the output verbatim with
    no need to trim a trailing newline beyond ``$( ... )``'s default behavior.

    :param argv: Argument list (omit to use ``sys.argv[1:]``).
    :returns: Process exit code (0 on success).
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    uri = compute_spec_uri(args.spec, args.job_name)
    sys.stdout.write(uri + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
