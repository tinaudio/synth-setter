#!/usr/bin/env python3
"""Validate a materialized DatasetPipelineSpec JSON.

Provides structural validation (required fields, code_version format, etc.) and optional test-value
validation for ci-materialize-test.yaml expectations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REQUIRED_FIELDS = [
    "base_seed",
    "channels",
    "code_version",
    "created_at",
    "is_repo_dirty",
    "min_loudness",
    "num_params",
    "output_format",
    "param_spec",
    "plugin_path",
    "preset_path",
    "r2_prefix",
    "renderer_version",
    "run_id",
    "sample_batch_size",
    "sample_rate",
    "shard_size",
    "shards",
    "signal_duration_seconds",
    "splits",
    "velocity",
]


def validate_structure(spec: dict) -> list[str]:
    """Validate structural correctness of a spec dict.

    Returns a list of error strings (empty means valid).
    Checks: required fields present, code_version is 40-char hex,
    renderer_version non-empty, shards non-empty.
    """
    errors: list[str] = []

    missing = [f for f in _REQUIRED_FIELDS if f not in spec]
    if missing:
        errors.append(f"missing required fields: {missing}")

    cv = spec.get("code_version", "")
    if not (len(cv) == 40 and all(c in "0123456789abcdef" for c in cv)):
        errors.append(f"code_version is not a valid 40-char hex SHA: {cv!r}")

    if not spec.get("renderer_version"):
        errors.append("renderer_version is empty")

    if not spec.get("shards"):
        errors.append("shards is empty")

    return errors


def validate_test_values(spec: dict) -> list[str]:
    """Validate test-specific values expected from ci-materialize-test.yaml.

    Returns a list of error strings (empty means valid).
    Checks: 3 shards, seeds [42,43,44], filenames zero-padded,
    config passthrough (param_spec, sample_rate, shard_size, base_seed, velocity).
    """
    errors: list[str] = []

    shards = spec.get("shards", [])
    if len(shards) != 3:
        errors.append(f"expected 3 shards, got {len(shards)}")

    seeds = [s["seed"] for s in shards]
    if seeds != [42, 43, 44]:
        errors.append(f"expected seeds [42, 43, 44], got {seeds}")

    filenames = [s["filename"] for s in shards]
    expected_filenames = ["shard-000000.h5", "shard-000001.h5", "shard-000002.h5"]
    if filenames != expected_filenames:
        errors.append(f"expected filenames {expected_filenames}, got {filenames}")

    passthrough = {
        "param_spec": "surge_simple",
        "sample_rate": 16000,
        "shard_size": 32,
        "base_seed": 42,
        "velocity": 100,
    }
    for field, expected in passthrough.items():
        actual = spec.get(field)
        if actual != expected:
            errors.append(f"{field}: expected {expected!r}, got {actual!r}")

    return errors


def main() -> None:
    """CLI entry point: validate a spec JSON file."""
    if len(sys.argv) < 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <spec.json> [--test-values]\n")
        sys.exit(1)

    spec_path = Path(sys.argv[1])
    run_test_values = "--test-values" in sys.argv

    spec = json.loads(spec_path.read_text())

    errors = validate_structure(spec)
    if not errors:
        sys.stdout.write("All structural checks passed:\n")
        sys.stdout.write(f"  code_version:     {spec['code_version']}\n")
        sys.stdout.write(f"  renderer_version: {spec['renderer_version']}\n")
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
