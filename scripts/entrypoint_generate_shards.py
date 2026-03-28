"""Entrypoint helper for MODE=generate_shards.

Reads dataset generation parameters from environment variables, loads the
dataset config YAML, and invokes generate_vst_dataset.py as a subprocess.

Expected env vars:
    DATASET_CONFIG  (required): Path to dataset config YAML inside the container.
    NUM_SAMPLES     (optional): Override for shard_size * num_shards.
    OUTPUT_DIR      (optional): Output directory for HDF5 files. Default: /output.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pipeline.schemas.config import dataset_config_id_from_path, load_dataset_config


def build_generate_args(
    config_path: Path,
    *,
    num_samples_override: int | None = None,
    output_dir: Path = Path("/output"),
) -> list[str]:
    """Build the CLI args for generate_vst_dataset.py from a dataset config.

    Args:
        config_path: Path to dataset config YAML.
        num_samples_override: If set, overrides shard_size * num_shards.
        output_dir: Directory for the output HDF5 file.

    Returns:
        List of CLI arguments for generate_vst_dataset.py.
    """
    cfg = load_dataset_config(config_path)
    config_id = dataset_config_id_from_path(config_path)

    num_samples = num_samples_override or (cfg.shard_size * cfg.num_shards)
    output_file = output_dir / f"{config_id}.hdf5"

    return [
        sys.executable,
        "src/data/vst/generate_vst_dataset.py",
        str(output_file),
        str(num_samples),
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

    num_samples_str = os.environ.get("NUM_SAMPLES")
    num_samples_override = int(num_samples_str) if num_samples_str else None

    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    args = build_generate_args(
        config_path,
        num_samples_override=num_samples_override,
        output_dir=output_dir,
    )

    sys.exit(subprocess.call(args))  # noqa: S603 — args built from validated config, not user input


if __name__ == "__main__":
    main()
