#!/usr/bin/env python3
"""Print the launcher's per-job R2 spec URI for a materialized input_spec.json.

Replaces the inline ``python -c`` one-liner that ``generate-dataset-shards.yaml``
used to read ``r2_bucket`` from the spec and concatenate the
``skypilot-launcher-specs/{cluster_name}.json`` URI by hand. Keeping this in
Python rather than bash means the legacy-key back-compat shim and the canonical
URI scheme constants (``R2_URI_SCHEME``, ``RCLONE_REMOTE``) stay the single
source of truth.

Usage::

    synth-setter-spec-uri <input_spec.json> <cluster_name>

Prints the resulting ``r2://bucket/skypilot-launcher-specs/<cluster>.json`` URI
to stdout; exits non-zero on missing args, missing/unreadable spec, or invalid
spec content.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

from synth_setter.pipeline.constants import LAUNCHER_SPEC_R2_PREFIX
from synth_setter.pipeline.schemas.spec import DatasetSpec

# Distinct exit codes so a GitHub Actions log scanner (or a human reading the
# step output) can tell argv / fs / parse failures apart without grepping the
# stderr message text.
_EXIT_USAGE = 1
_EXIT_MISSING_FILE = 2
_EXIT_INVALID_SPEC = 3


def compute_spec_uri(spec_path: Path, cluster_name: str) -> str:
    """Read ``spec_path`` and return the launcher's R2 URI for ``cluster_name``.

    The URI follows the launcher's own convention exactly so the worker
    pod's ``WORKER_SPEC_URI`` env var lines up with the workflow's exported
    output.

    :param spec_path: Local path to a materialized ``input_spec.json``.
    :param cluster_name: SkyPilot managed-job name forwarded as ``--job-name``.
    :returns: ``r2://<bucket>/skypilot-launcher-specs/<cluster>.json`` URI string.
    """
    spec = DatasetSpec.model_validate_json(spec_path.read_text())
    return spec.r2.uri(f"{LAUNCHER_SPEC_R2_PREFIX}/{cluster_name}.json")


def main() -> None:
    """CLI entry: ``synth-setter-spec-uri <spec.json> <cluster_name>``."""
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <input_spec.json> <cluster_name>\n")
        sys.exit(_EXIT_USAGE)
    spec_path = Path(sys.argv[1])
    cluster_name = sys.argv[2]
    if not spec_path.is_file():
        sys.stderr.write(f"error: spec file not found: {spec_path}\n")
        sys.exit(_EXIT_MISSING_FILE)
    try:
        uri = compute_spec_uri(spec_path, cluster_name)
    except (OSError, ValueError, ValidationError) as exc:
        # ValueError covers Pydantic's JSON decode error; OSError covers
        # read-time fs failures (permission denied, mid-read truncation).
        # Collapse the traceback into one stderr line so the GitHub Actions
        # step output is interpretable at a glance.
        sys.stderr.write(f"error: failed to parse spec {spec_path}: {exc}\n")
        sys.exit(_EXIT_INVALID_SPEC)
    sys.stdout.write(uri + "\n")


if __name__ == "__main__":
    main()
