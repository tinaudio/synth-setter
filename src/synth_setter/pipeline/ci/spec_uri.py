#!/usr/bin/env python3
"""Print the canonical R2 URI of a materialized ``input_spec.json``.

Reads the local spec file, parses it as a ``DatasetSpec``, and emits
``spec.r2.input_spec_uri()`` — the under-prefix URI where the spec itself
lives (``r2://<bucket>/<prefix>input_spec.json``).

Keeping this in Python rather than bash lets the URI be parsed by the real
``DatasetSpec`` model (so any schema drift fails loud at the validator
rather than silently in jq/sed) and lets argv / fs / parse failures map
to distinct exit codes for log scanners.

Usage::

    synth-setter-spec-uri <input_spec.json>

Prints the resulting URI to stdout; exits non-zero on wrong arity, missing
or unreadable spec, or invalid spec content.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

from synth_setter.pipeline.schemas.spec import DatasetSpec

# Distinct exit codes so a GitHub Actions log scanner (or a human reading the
# step output) can tell argv / fs / parse failures apart without grepping the
# stderr message text.
_EXIT_USAGE = 1
_EXIT_MISSING_FILE = 2
_EXIT_INVALID_SPEC = 3


def compute_spec_uri(spec_path: Path) -> str:
    """Read ``spec_path`` and return the spec's canonical input_spec R2 URI.

    :param spec_path: Local path to a materialized ``input_spec.json``.
    :returns: ``spec.r2.input_spec_uri()`` —
        ``r2://<bucket>/<prefix>input_spec.json`` URI string.
    """
    spec = DatasetSpec.model_validate_json(spec_path.read_text())
    return spec.r2.input_spec_uri()


def main() -> None:
    """CLI entry: ``synth-setter-spec-uri <spec.json>``."""
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <input_spec.json>\n")
        sys.exit(_EXIT_USAGE)
    spec_path = Path(sys.argv[1])
    if not spec_path.is_file():
        sys.stderr.write(f"error: spec file not found: {spec_path}\n")
        sys.exit(_EXIT_MISSING_FILE)
    try:
        uri = compute_spec_uri(spec_path)
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
