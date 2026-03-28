"""Resolve dataset generation run parameters and write to GITHUB_OUTPUT.

Replaces the shell echo block in the dataset-generation workflow with a testable Python script. All
values are derived from CLI input or the dataset config — no hardcoded magic numbers.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pipeline.schemas.config import load_dataset_config

_DEFAULT_DATASET_CONFIG = "configs/dataset/surge-simple-480k-10k.yaml"
_DEFAULT_DOCKER_TAG = "dev-snapshot"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for run parameter resolution."""
    parser = argparse.ArgumentParser(
        description="Resolve dataset generation run parameters for CI"
    )
    parser.add_argument(
        "--event-name",
        required=True,
        help="GitHub event name (e.g., pull_request, workflow_dispatch)",
    )
    parser.add_argument(
        "--dataset-config",
        default="",
        help="Path to dataset config YAML (empty = use default)",
    )
    parser.add_argument(
        "--num-samples",
        default="",
        help="Number of samples override (empty = derive from config)",
    )
    parser.add_argument(
        "--docker-tag",
        default="",
        help="Docker image tag (empty = use default)",
    )
    parser.add_argument(
        "--upload-to-r2",
        default="",
        help="Whether to upload to R2 (empty = derive from event type)",
    )
    return parser.parse_args()


def resolve_params(
    event_name: str,
    dataset_config: str,
    num_samples: str,
    docker_tag: str,
    upload_to_r2: str,
) -> dict[str, str]:
    """Resolve run parameters from CLI inputs and dataset config.

    For pull_request events: uses sample_batch_size as num_samples (one batch =
    minimum meaningful smoke test), disables R2 upload.

    For workflow_dispatch events: uses provided values with config-derived
    fallbacks (num_samples = shard_size * num_shards).
    """
    resolved_config = dataset_config or _DEFAULT_DATASET_CONFIG
    cfg = load_dataset_config(Path(resolved_config))

    if event_name == "pull_request":
        return {
            "dataset_config": resolved_config,
            "num_samples": str(cfg.sample_batch_size),
            "docker_tag": _DEFAULT_DOCKER_TAG,
            "upload_to_r2": "false",
        }

    resolved_num_samples = num_samples or str(cfg.shard_size * cfg.num_shards)
    resolved_docker_tag = docker_tag or _DEFAULT_DOCKER_TAG
    resolved_upload = upload_to_r2 or "true"

    return {
        "dataset_config": resolved_config,
        "num_samples": resolved_num_samples,
        "docker_tag": resolved_docker_tag,
        "upload_to_r2": resolved_upload,
    }


def main() -> None:
    """Parse args, resolve parameters, and write to GITHUB_OUTPUT or stdout."""
    args = _parse_args()

    fields = resolve_params(
        event_name=args.event_name,
        dataset_config=args.dataset_config,
        num_samples=args.num_samples,
        docker_tag=args.docker_tag,
        upload_to_r2=args.upload_to_r2,
    )

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
