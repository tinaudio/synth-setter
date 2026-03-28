"""Write dataset config fields to GITHUB_OUTPUT for Actions workflows."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pipeline.schemas.config import dataset_config_id_from_path, load_dataset_config
from pipeline.schemas.prefix import make_dataset_wandb_run_id, make_r2_prefix


def main() -> None:
    """Load dataset config from YAML and write fields to GITHUB_OUTPUT."""
    parser = argparse.ArgumentParser(description="Load dataset config and emit to GITHUB_OUTPUT")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_dataset_config(config_path)
    config_id = dataset_config_id_from_path(config_path)

    run_id = make_dataset_wandb_run_id(config_id)
    r2_prefix = make_r2_prefix(config_id, run_id)

    fields = {
        "dataset_config_id": config_id,
        "param_spec": cfg.param_spec,
        "plugin_path": cfg.plugin_path,
        "preset_path": cfg.preset_path,
        "sample_rate": cfg.sample_rate,
        "channels": cfg.channels,
        "velocity": cfg.velocity,
        "signal_duration_seconds": cfg.signal_duration_seconds,
        "min_loudness": cfg.min_loudness,
        "sample_batch_size": cfg.sample_batch_size,
        "num_samples": cfg.shard_size * cfg.num_shards,
        "run_id": run_id,
        "r2_prefix": r2_prefix,
    }

    output_path = os.environ.get("GITHUB_OUTPUT")
    dest = open(output_path, "a") if output_path else sys.stdout  # noqa: SIM115

    try:
        for field_name, value in fields.items():
            dest.write(f"{field_name}={value}\n")
    finally:
        if dest is not sys.stdout:
            dest.close()


if __name__ == "__main__":
    main()
