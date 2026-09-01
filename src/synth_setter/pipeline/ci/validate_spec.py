#!/usr/bin/env python3
"""Validate a materialized DatasetSpec JSON.

Provides structural validation (required fields, git_sha format, etc.) and optional test-value
validation for generate_dataset/ci-materialize-test.yaml expectations.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from synth_setter.pipeline.schemas.spec import (
    DatasetSpec,
    OutputFormat,
    RenderConfig,
)
from synth_setter.pipeline.spec_io import read_spec_text
from synth_setter.synth_spec import SynthSpec

# Required keys are derived from the model so adding a field to ``DatasetSpec``
# (including computed_fields, which serialize on dump) automatically tightens
# the structural check on the next CI run — no parallel list to update.
_REQUIRED_TOP_LEVEL_FIELDS: tuple[str, ...] = tuple(
    sorted(set(DatasetSpec.model_fields) | set(DatasetSpec.model_computed_fields))
)
_BACKWARD_COMPATIBLE_OPTIONAL_RENDER_FIELDS = frozenset(
    {"audio_dtype", "mel_spec_dtype", "pyfdn_effect", "retain_local_shards"}
)
# ``synth`` is checked shape-aware below so its required version is validated too.
_REQUIRED_RENDER_FIELDS: tuple[str, ...] = tuple(
    sorted(
        set(RenderConfig.model_fields) - _BACKWARD_COMPATIBLE_OPTIONAL_RENDER_FIELDS - {"synth"}
    )
)
_REQUIRED_SYNTH_FIELDS: tuple[str, ...] = tuple(sorted(SynthSpec.model_fields))


def _render_param_spec_name(render: dict[str, Any]) -> str | None:
    """Read the param-spec name from canonical nested synth identity.

    :param render: Raw ``render`` mapping from a spec dict.
    :returns: The declared param-spec name, or ``None`` when identity is absent.
    """
    synth = render.get("synth")
    if not isinstance(synth, dict):
        return None
    name = synth.get("param_spec_name")
    return None if name is None else str(name)


def _parse_output_format(value: Any) -> OutputFormat | None:
    """Coerce a raw spec-dict ``output_format`` value to ``OutputFormat``, or ``None``.

    Validates the raw JSON value without constructing a ``DatasetSpec`` (these
    checks run on the spec dict before model construction).

    :param value: Raw ``output_format`` value from a spec dict.
    :returns: The matching format, or ``None`` when ``value`` is not a known token.
    """
    try:
        return OutputFormat(value)
    except ValueError:
        return None


def validate_structure(spec: dict[str, Any]) -> list[str]:
    """Validate structural correctness of a spec dict.

    Returns a list of error strings (empty means valid).
    Checks: required fields present, git_sha is 40-char hex,
    synth_version non-empty, shards non-empty.
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

    synth = render.get("synth")
    if not isinstance(synth, dict):
        errors.append("render.synth must be a mapping")
    else:
        missing_synth = [field for field in _REQUIRED_SYNTH_FIELDS if field not in synth]
        if missing_synth:
            errors.append(f"missing required synth fields: {missing_synth}")
        synth_version = synth.get("synth_version")
        if "synth_version" in synth and (
            not isinstance(synth_version, str) or not synth_version.strip()
        ):
            errors.append("render.synth.synth_version must be a non-empty string")

    cv = spec.get("git_sha", "")
    if not (len(cv) == 40 and all(c in "0123456789abcdef" for c in cv)):
        errors.append(f"git_sha is not a valid 40-char hex SHA: {cv!r}")

    raw_format = spec.get("output_format")
    if raw_format is not None and _parse_output_format(raw_format) is None:
        errors.append(
            f"output_format {raw_format!r} is not one of {sorted(f.value for f in OutputFormat)}"
        )

    if not spec.get("shards"):
        errors.append("shards is empty")

    return errors


def validate_test_values(spec: dict[str, Any]) -> list[str]:
    """Validate test-specific values expected from generate_dataset/ci-materialize-test.yaml.

    Returns a list of error strings (empty means valid).
    Checks: 3 shards, seeds [42,43,44], filenames zero-padded,
    config passthrough (param_spec_name, sample_rate, samples_per_shard, base_seed, velocity).
    """
    errors: list[str] = []

    shards = spec.get("shards", [])
    if len(shards) != 3:
        errors.append(f"expected 3 shards, got {len(shards)}")

    seeds = [s["seed"] for s in shards]
    if seeds != [42, 43, 44]:
        errors.append(f"expected seeds [42, 43, 44], got {seeds}")

    filenames = [s["filename"] for s in shards]
    raw_format = spec.get("output_format", OutputFormat.LANCE.value)
    output_format = _parse_output_format(raw_format)
    if output_format is None:
        errors.append(
            f"cannot compute expected filenames: output_format {raw_format!r} is not one of "
            f"{sorted(f.value for f in OutputFormat)}"
        )
    else:
        ext = output_format.extension
        expected_filenames = [f"shard-{i:06d}{ext}" for i in range(3)]
        if filenames != expected_filenames:
            errors.append(f"expected filenames {expected_filenames}, got {filenames}")

    render = spec.get("render") or {}
    top_passthrough = {
        "base_seed": 42,
    }
    render_passthrough = {
        "sample_rate": 44100,
        "samples_per_shard": 32,
        "velocity": 100,
    }
    actual_spec_name = _render_param_spec_name(render)
    if actual_spec_name != "surge_simple":
        errors.append(f"render.param_spec_name: expected 'surge_simple', got {actual_spec_name!r}")
    for field, expected in top_passthrough.items():
        actual = spec.get(field)
        if actual != expected:
            errors.append(f"{field}: expected {expected!r}, got {actual!r}")
    for field, expected in render_passthrough.items():
        actual = render.get(field) if isinstance(render, dict) else None
        if actual != expected:
            errors.append(f"render.{field}: expected {expected!r}, got {actual!r}")

    return errors


def main() -> None:
    """CLI entry point: validate a spec JSON file (local path, file:// URI, or r2:// URI)."""
    if len(sys.argv) < 2:
        sys.stderr.write(
            f"Usage: {sys.argv[0]} "
            "<spec.json|file:///abs/path/spec.json|r2://bucket/key.json> [--test-values]\n"
        )
        sys.exit(1)

    spec_arg = sys.argv[1]
    run_test_values = "--test-values" in sys.argv

    spec = json.loads(read_spec_text(spec_arg))

    errors = validate_structure(spec)
    if not errors:
        render = spec.get("render", {})
        sys.stdout.write("All structural checks passed:\n")
        sys.stdout.write(f"  git_sha:          {spec['git_sha']}\n")
        synth = render.get("synth", {})
        sys.stdout.write(f"  synth_version:    {synth.get('synth_version')}\n")
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
