"""Entrypoint helper for MODE=generate_dataset.

Reads dataset generation parameters from environment variables, loads the
dataset config YAML, and invokes generate_vst_dataset.py as a subprocess.

Expected env vars:
    DATASET_CONFIG  (required): Path to dataset config YAML inside the container.
    OUTPUT_DIR      (optional): Output directory for HDF5 files. Default: /output.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pipeline.schemas.config import load_dataset_config


def build_generate_args(
    config_path: Path,
    *,
    output_dir: Path = Path("/output"),
) -> list[str]:
    """Build the CLI args for generate_vst_dataset.py from a dataset config.

    Generates a single shard (shard_size samples) per invocation. The output
    file is always named shard-000000.hdf5 in the given output_dir. Multi-shard
    generation is not yet supported — pass num_shards=1 in the config.

    Args:
        config_path: Path to dataset config YAML.
        output_dir: Directory for the output HDF5 file.

    Returns:
        List of CLI arguments for generate_vst_dataset.py.

    Raises:
        ValueError: If config output_format is not 'hdf5'.
        NotImplementedError: If config num_shards > 1.
    """
    cfg = load_dataset_config(config_path)

    if cfg.output_format != "hdf5":
        raise ValueError(
            f"generate_vst_dataset.py only supports hdf5 output, got: {cfg.output_format}"
        )

    if cfg.num_shards > 1:
        raise NotImplementedError(
            f"num_shards > 1 not yet supported (got {cfg.num_shards}). "
            "Multi-shard generation requires the distributed pipeline."
        )

    output_file = output_dir / "shard-000000.hdf5"

    return [
        sys.executable,
        "src/data/vst/generate_vst_dataset.py",
        str(output_file),
        str(cfg.shard_size),
        "--plugin_path",
        cfg.plugin_path,
        "--preset_path",
        cfg.preset_path,
        "--sample_rate",
        str(cfg.sample_rate),
        "--channels",
        str(cfg.channels),
        "--velocity",
        str(cfg.velocity),
        "--signal_duration_seconds",
        str(cfg.signal_duration_seconds),
        "--min_loudness",
        str(cfg.min_loudness),
        "--param_spec",
        cfg.param_spec,
        "--sample_batch_size",
        str(cfg.sample_batch_size),
    ]


def main() -> None:
    """Read env vars, build CLI args, and exec generate_vst_dataset.py."""
    config_path = Path(os.environ["DATASET_CONFIG"])

    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    args = build_generate_args(config_path, output_dir=output_dir)

    sys.exit(subprocess.call(args))  # noqa: S603 — args built from validated config, not user input


if __name__ == "__main__":
    main()
