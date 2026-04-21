"""Spec-driven single-shard generate_dataset runner.

Public API:
    run(spec): Full flow — upload spec to R2, generate shard, upload shard to R2.
    build_generate_args(spec, shard, output_dir): Build CLI args for
        src/data/vst/generate_vst_dataset.py.

This module is no longer invocable via ``python -m``; the container's CLI
entrypoint (``scripts/docker_entrypoint.py generate_dataset --spec <path>``)
parses the spec and calls ``run(spec)`` in-process.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline.constants import INPUT_SPEC_FILENAME
from pipeline.schemas.spec import DatasetPipelineSpec, ShardSpec

# Bootstraps Xvfb + xsettingsd + dbus for VST3 plugin init; resolved relative
# to the container WORKDIR (``/home/build/synth-setter``) baked in the image.
# X11 wrapping lives at the audio-rendering boundary (this subprocess call),
# not at the container entrypoint — the click CLI stays X11-agnostic so idle
# and passthrough don't pay the Xvfb startup cost.
VST_HEADLESS_WRAPPER = "scripts/run-linux-vst-headless.sh"


def _rclone_copy(src: str, dest: str) -> None:
    """Upload a file to R2 via rclone with checksum verification."""
    subprocess.check_call(["rclone", "copy", "--checksum", src, dest])  # noqa: S603, S607


def build_generate_args(
    spec: DatasetPipelineSpec, shard: ShardSpec, output_dir: Path
) -> list[str]:
    """Build CLI args for generate_vst_dataset.py from a spec and shard.

    Args:
        spec: Materialized pipeline spec (dataset-level parameters).
        shard: The specific shard to generate (owns filename).
        output_dir: Directory for the output HDF5 file.

    Returns:
        List of CLI arguments for generate_vst_dataset.py.
    """
    output_file = output_dir / shard.filename

    options = {
        "plugin_path": spec.plugin_path,
        "preset_path": spec.preset_path,
        "sample_rate": spec.sample_rate,
        "channels": spec.channels,
        "velocity": spec.velocity,
        "signal_duration_seconds": spec.signal_duration_seconds,
        "min_loudness": spec.min_loudness,
        "param_spec": spec.param_spec,
        "sample_batch_size": spec.sample_batch_size,
    }

    args = [
        sys.executable,
        "src/data/vst/generate_vst_dataset.py",
        str(output_file),
        str(spec.shard_size),
    ]
    for key, value in options.items():
        args.extend([f"--{key}", str(value)])

    return args


def run(spec: DatasetPipelineSpec) -> None:
    """Materialized-spec driven run: upload spec to R2, generate shard, upload shard.

    Single-shard only. Multi-shard generation is tracked in
    https://github.com/tinaudio/synth-setter/issues/407 (``generate-shards``)
    and requires the distributed pipeline. This entrypoint fails fast with
    ``NotImplementedError`` on ``spec.num_shards > 1`` so callers don't
    silently get a partial dataset.

    HDF5-only. Other output formats (e.g. WebDataset) are not yet wired into
    the downstream generator.

    Args:
        spec: Pre-materialized DatasetPipelineSpec.

    Raises:
        NotImplementedError: If ``spec.num_shards > 1`` (tracked in #407).
        ValueError: If ``spec.output_format != "hdf5"``.
    """
    # Single-shard fail-fast. The downstream generator, the shard-upload
    # step, and build_generate_args all assume exactly one shard; see #407
    # for the multi-shard rewrite.
    if spec.num_shards > 1:
        raise NotImplementedError(
            f"num_shards > 1 not yet supported (got {spec.num_shards}). "
            "Multi-shard generation is tracked in #407 "
            "(https://github.com/tinaudio/synth-setter/issues/407)."
        )

    if spec.output_format != "hdf5":
        raise ValueError(
            f"generate_vst_dataset.py only supports hdf5 output, got: {spec.output_format}"
        )

    r2_dest_prefix = f"r2:{spec.r2_bucket}/{spec.r2_prefix}"

    with tempfile.TemporaryDirectory() as work_dir_str:
        work_dir = Path(work_dir_str)

        # `rclone copy` treats the destination as a directory and preserves
        # the source's basename. The local spec file is already named
        # INPUT_SPEC_FILENAME, so uploading to the prefix directory lands
        # the object at `{prefix}{INPUT_SPEC_FILENAME}` without the
        # double-name issue a full object-key destination would cause.
        spec_path = work_dir / INPUT_SPEC_FILENAME
        spec_path.write_text(spec.model_dump_json(indent=2))
        _rclone_copy(str(spec_path), r2_dest_prefix)

        # Single-shard only: picks spec.shards[0] unconditionally. Guarded by
        # the fail-fast check above; multi-shard support tracked in #407.
        shard = spec.shards[0]
        args = [VST_HEADLESS_WRAPPER, *build_generate_args(spec, shard, work_dir)]
        subprocess.check_call(args)  # noqa: S603 — args built from validated spec
        shard_path = work_dir / shard.filename
        _rclone_copy(str(shard_path), r2_dest_prefix)


if __name__ == "__main__":
    raise SystemExit(
        "pipeline.entrypoints.generate_dataset is no longer invocable via `python -m`. "
        "Use `scripts/docker_entrypoint.py generate_dataset --spec <path>` instead."
    )
