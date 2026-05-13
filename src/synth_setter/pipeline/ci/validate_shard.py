#!/usr/bin/env python3
"""Validate HDF5 shards against a DatasetSpec.

Checks that each shard file is a valid HDF5 file, contains the per-row
arrays the writer emits (``synth_setter.data.vst.shapes.DATASET_FIELD_NAMES``),
and that each dataset's full ``.shape`` matches what those shape helpers
predict for ``spec.render`` — i.e. ``(N, C, time)`` for audio,
``(N, C, n_mels, n_frames)`` for the mel spectrogram, and
``(N, num_params)`` for the param array.

CLI usage:
    python3 -m synth_setter.pipeline.ci.validate_shard <spec.json|r2://bucket/spec.json>

Iterates `spec.shards` and downloads each shard from R2 (under
`r2://{spec.r2_bucket}/{spec.r2_prefix}{shard.filename}`) before validating.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import h5py

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    audio_dataset_shape,
    mel_dataset_shape,
    param_array_dataset_shape,
)
from synth_setter.pipeline.r2_io import downloaded_to_tempfile, is_r2_uri, shard_uri
from synth_setter.pipeline.schemas.spec import DatasetSpec


def _expected_dataset_shapes(spec: DatasetSpec) -> dict[str, tuple[int, ...]]:
    """Full per-field shapes (N + inner) the writer emits for ``spec``.

    Keys match ``DATASET_FIELD_NAMES``; values come from the writer's own
    shape helpers in ``synth_setter.data.vst.shapes`` so a future renderer
    change that drifts the audio / mel / param shapes fails fast here.
    """
    render = spec.render
    num_samples = render.batch_per_shard
    return {
        AUDIO_FIELD: audio_dataset_shape(
            num_samples, render.channels, render.sample_rate, render.signal_duration_seconds
        ),
        MEL_SPEC_FIELD: mel_dataset_shape(
            num_samples, render.channels, render.sample_rate, render.signal_duration_seconds
        ),
        PARAM_ARRAY_FIELD: param_array_dataset_shape(num_samples, spec.num_params),
    }


def validate_shard(shard_path: Path, spec: DatasetSpec) -> list[str]:
    """Validate one HDF5 shard against a DatasetSpec.

    Checks:
    1. File opens as HDF5
    2. Contains every dataset named in ``DATASET_FIELD_NAMES``
    3. Each dataset's full ``.shape`` matches what
       ``_expected_dataset_shapes`` predicts for ``spec``

    Returns list of error strings (empty = valid).
    """
    if not shard_path.exists():
        return [f"shard file not found: {shard_path}"]

    try:
        f = h5py.File(shard_path, "r")
    except OSError:
        return [f"file is not valid HDF5: {shard_path}"]

    expected_shapes = _expected_dataset_shapes(spec)
    errors: list[str] = []
    with f:
        for name in DATASET_FIELD_NAMES:
            if name not in f:
                errors.append(f"missing dataset: {name!r}")
                continue

            actual = cast(h5py.Dataset, f[name]).shape
            expected = expected_shapes[name]
            if actual != expected:
                errors.append(f"dataset {name!r} has shape {actual}, expected {expected}")

    return errors


def _load_spec(spec_arg: str) -> DatasetSpec:
    """Load a spec from a local path or `r2://bucket/key` URI."""
    if is_r2_uri(spec_arg):
        with downloaded_to_tempfile(spec_arg) as local_path:
            return DatasetSpec.model_validate_json(local_path.read_text())
    return DatasetSpec.model_validate_json(Path(spec_arg).read_text())


def validate_all_shards_from_r2(spec: DatasetSpec) -> list[str]:
    """Validate every shard in `spec.shards` by downloading from R2.

    Returns aggregated error strings across all shards (empty = all valid). Each error is prefixed
    with the shard filename so the source is obvious.
    """
    errors: list[str] = []
    for shard in spec.shards:
        shard_object_uri = shard_uri(spec.r2_bucket, spec.r2_prefix, shard.filename)
        with downloaded_to_tempfile(shard_object_uri) as local_shard:
            shard_errors = validate_shard(local_shard, spec)
        for err in shard_errors:
            errors.append(f"{shard.filename}: {err}")
    return errors


def main() -> None:
    """CLI entry point: validate every shard referenced by a spec.

    The single argument is a spec JSON path or `r2://bucket/key.json` URI.
    Each shard listed in `spec.shards` is fetched from R2 and validated.
    """
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <spec.json|r2://bucket/spec.json>\n")
        sys.exit(1)

    spec_arg = sys.argv[1]
    spec = _load_spec(spec_arg)

    errors = validate_all_shards_from_r2(spec)

    if errors:
        for error in errors:
            sys.stderr.write(f"FAIL: {error}\n")
        sys.exit(1)

    sys.stdout.write(f"OK: all {len(spec.shards)} shards valid\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
