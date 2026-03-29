"""Entrypoint helper for MODE=generate_dataset.

Materializes a DatasetPipelineSpec from config, uploads spec to R2, generates a
single shard, and uploads the shard to R2.

Public API:
    run(): full flow — materialize spec, upload, generate, upload shard
    build_generate_args(): builds CLI args from a spec + shard + output_dir
    main(): reads env vars and delegates to run()

Expected env vars:
    DATASET_CONFIG   (required): Path to dataset config YAML inside the container.
    RUN_METADATA_DIR (optional): Dir for input_spec.json output. Default: /run-metadata.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline.constants import INPUT_SPEC_FILENAME
from pipeline.schemas.config import dataset_config_id_from_path, load_dataset_config
from pipeline.schemas.prefix import make_r2_prefix
from pipeline.schemas.spec import DatasetPipelineSpec, ShardSpec, materialize_spec


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


def run(config_path: Path, metadata_dir: Path) -> None:
    """Full generate_dataset flow: materialize, upload spec, generate, upload shard.

    Args:
        config_path: Path to dataset config YAML.
        metadata_dir: Dir for input_spec.json (bind-mounted, host reads this).

    Raises:
        NotImplementedError: If num_shards > 1.
        ValueError: If output_format is not 'hdf5'.
    """
    cfg = load_dataset_config(config_path)
    config_id = dataset_config_id_from_path(config_path)

    if cfg.num_shards > 1:
        raise NotImplementedError(
            f"num_shards > 1 not yet supported (got {cfg.num_shards}). "
            "Multi-shard generation requires the distributed pipeline."
        )

    if cfg.output_format != "hdf5":
        raise ValueError(
            f"generate_vst_dataset.py only supports hdf5 output, got: {cfg.output_format}"
        )

    spec = materialize_spec(cfg, config_id)

    # Write spec to metadata dir (host reads this via bind mount)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    spec_path = metadata_dir / INPUT_SPEC_FILENAME
    spec_path.write_text(spec.model_dump_json(indent=2))

    # Upload spec to R2 before generation
    r2_prefix = make_r2_prefix(config_id, spec.run_id)
    r2_dest = f"r2:intermediate-data/{r2_prefix}"
    _rclone_copy(str(spec_path), r2_dest)

    # Generate shard in temp dir, then upload to R2
    shard = spec.shards[0]
    with tempfile.TemporaryDirectory() as shard_dir:
        args = build_generate_args(spec, shard, Path(shard_dir))
        subprocess.check_call(args)  # noqa: S603 — args built from validated spec
        shard_path = Path(shard_dir) / shard.filename
        _rclone_copy(str(shard_path), r2_dest)


def main() -> None:
    """Read env vars and run."""
    config_path = Path(os.environ["DATASET_CONFIG"])
    metadata_dir = Path(os.environ.get("RUN_METADATA_DIR", "/run-metadata"))

    run(config_path, metadata_dir)


if __name__ == "__main__":
    main()
