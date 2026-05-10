#!/usr/bin/env python3
"""Validate a materialized DatasetSpec JSON.

Provides structural validation (required fields, git_sha format, etc.) and optional test-value
validation for ci-materialize-test.yaml expectations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from src.pipeline.r2_io import downloaded_to_tempfile, is_r2_uri
from src.pipeline.schemas.spec import (
    OUTPUT_FORMAT_TO_EXTENSION,
    DatasetSpec,
    RenderConfig,
)

_GIT_SHA_HEX_LEN = 40

# Required keys are derived from the model so adding a field to ``DatasetSpec``
# (including computed_fields, which serialize on dump) automatically tightens
# the structural check on the next CI run — no parallel list to update.
_REQUIRED_TOP_LEVEL_FIELDS: tuple[str, ...] = tuple(
    sorted(set(DatasetSpec.model_fields) | set(DatasetSpec.model_computed_fields))
)
_REQUIRED_RENDER_FIELDS: tuple[str, ...] = tuple(sorted(RenderConfig.model_fields))


def validate_structure(spec: dict[str, Any]) -> list[str]:
    """Validate structural correctness of a spec dict.

    Returns a list of error strings (empty means valid).
    Checks: required fields present, git_sha is 40-char hex,
    renderer_version non-empty, shards non-empty.
    """
    errors: list[str] = []

    missing = [f for f in _REQUIRED_TOP_LEVEL_FIELDS if f not in spec]
    if missing:
        errors.append(f"missing required fields: {missing}")

    render = spec.get("render") or {}
    if not isinstance(render, dict):
        errors.append("render must be a mapping")
        render = {}
    missing_render = [f for f in _REQUIRED_RENDER_FIELDS if f not in render]
    if missing_render:
        errors.append(f"missing required render fields: {missing_render}")

    git_sha = spec.get("git_sha", "")
    if not (len(git_sha) == _GIT_SHA_HEX_LEN and all(c in "0123456789abcdef" for c in git_sha)):
        errors.append(f"git_sha is not a valid {_GIT_SHA_HEX_LEN}-char hex SHA: {git_sha!r}")

    if not render.get("renderer_version"):
        errors.append("render.renderer_version is empty")

    if not spec.get("shards"):
        errors.append("shards is empty")

    return errors


def validate_test_values(spec: dict[str, Any]) -> list[str]:
    """Validate test-specific values expected from ci-materialize-test.yaml.

    Returns a list of error strings (empty means valid).
    Checks: 3 shards, seeds [42,43,44], filenames zero-padded,
    config passthrough (param_spec_name, sample_rate, batch_per_shard, base_seed, velocity).
    """
    errors: list[str] = []

    shards = spec.get("shards", [])
    if len(shards) != 3:
        errors.append(f"expected 3 shards, got {len(shards)}")

    seeds = [s["seed"] for s in shards]
    if seeds != [42, 43, 44]:
        errors.append(f"expected seeds [42, 43, 44], got {seeds}")

    filenames = [s["filename"] for s in shards]
    ext = OUTPUT_FORMAT_TO_EXTENSION[spec.get("output_format", "hdf5")]
    expected_filenames = [f"shard-{i:06d}{ext}" for i in range(3)]
    if filenames != expected_filenames:
        errors.append(f"expected filenames {expected_filenames}, got {filenames}")

    render = spec.get("render") or {}
    top_passthrough = {
        "base_seed": 42,
    }
    render_passthrough = {
        "param_spec_name": "surge_simple",
        "sample_rate": 16000,
        "batch_per_shard": 32,
        "velocity": 100,
    }
    for field, expected in top_passthrough.items():
        actual = spec.get(field)
        if actual != expected:
            errors.append(f"{field}: expected {expected!r}, got {actual!r}")
    for field, expected in render_passthrough.items():
        actual = render.get(field) if isinstance(render, dict) else None
        if actual != expected:
            errors.append(f"render.{field}: expected {expected!r}, got {actual!r}")

    return errors


def _read_spec_text(spec_arg: str) -> str:
    """Read spec JSON text from a local path or `r2://bucket/key` URI."""
    if is_r2_uri(spec_arg):
        with downloaded_to_tempfile(spec_arg) as local_path:
            return local_path.read_text()
    return Path(spec_arg).read_text()


def main() -> None:
    """CLI entry point: validate a spec JSON file (local path or r2:// URI)."""
    if len(sys.argv) < 2:
        sys.stderr.write(
            f"Usage: {sys.argv[0]} <spec.json|r2://bucket/key.json> [--test-values]\n"
        )
        sys.exit(1)

    spec_arg = sys.argv[1]
    run_test_values = "--test-values" in sys.argv

    spec = json.loads(_read_spec_text(spec_arg))

    errors = validate_structure(spec)
    if not errors:
        render = spec.get("render", {})
        sys.stdout.write("All structural checks passed:\n")
        sys.stdout.write(f"  git_sha:          {spec['git_sha']}\n")
        sys.stdout.write(f"  renderer_version: {render.get('renderer_version')}\n")
        sys.stdout.write(f"  num_params:       {spec['num_params']}\n")
        sys.stdout.write(f"  num_shards:       {len(spec['shards'])}\n")

    if run_test_values:
        errors.extend(validate_test_values(spec))
        if not errors:
            seeds = [s["seed"] for s in spec["shards"]]
            filenames = [s["filename"] for s in spec["shards"]]
            sys.stdout.write(f"  num_shards: {len(spec['shards'])} (expected 3)\n")
            sys.stdout.write(f"  seeds: {seeds} (expected [42, 43, 44])\n")
            sys.stdout.write(f"  filenames: {filenames}\n")
            sys.stdout.write("  config passthrough: all correct\n")
            sys.stdout.write("All test assertions passed.\n")

    if errors:
        for error in errors:
            sys.stderr.write(f"FAIL: {error}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
