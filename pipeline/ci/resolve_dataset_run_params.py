"""Resolve dataset generation run parameters and write to GITHUB_OUTPUT.

Fills in defaults for empty CLI inputs. The dataset config YAML is the single source of truth for
all generation parameters — this script only resolves workflow-level concerns (which config, which
image tag, whether to upload).
"""

from __future__ import annotations

import argparse
import os
import sys

_DEFAULT_DATASET_CONFIG = "configs/dataset/surge-simple-480k-10k.yaml"
_DEFAULT_DOCKER_TAG = "dev-snapshot"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for run parameter resolution."""
    parser = argparse.ArgumentParser(
        description="Resolve dataset generation run parameters for CI"
    )
    parser.add_argument(
        "--dataset-config",
        default="",
        help="Path to dataset config YAML (empty = use default)",
    )
    parser.add_argument(
        "--docker-tag",
        default="",
        help="Docker image tag (empty = use default)",
    )
    parser.add_argument(
        "--upload-to-r2",
        default="",
        help="Whether to upload to R2: 'true' or 'false' (empty = true)",
    )
    return parser.parse_args()


def resolve_params(
    dataset_config: str,
    docker_tag: str,
    upload_to_r2: str,
) -> dict[str, str]:
    """Resolve run parameters, filling empty inputs with defaults.

    Args:
        dataset_config: Config YAML path (empty = default config).
        docker_tag: Docker image tag (empty = dev-snapshot).
        upload_to_r2: 'true', 'false', or empty (empty = true).

    Returns:
        Dict of resolved parameter key-value pairs.

    Raises:
        ValueError: If upload_to_r2 is not empty, 'true', or 'false'.
    """
    if upload_to_r2 and upload_to_r2.lower() not in ("true", "false"):
        raise ValueError(f"upload_to_r2 must be 'true' or 'false', got: {upload_to_r2!r}")

    return {
        "dataset_config": dataset_config or _DEFAULT_DATASET_CONFIG,
        "docker_tag": docker_tag or _DEFAULT_DOCKER_TAG,
        "upload_to_r2": upload_to_r2.lower() if upload_to_r2 else "true",
    }


def main() -> None:
    """Parse args, resolve parameters, and write to GITHUB_OUTPUT or stdout."""
    args = _parse_args()

    fields = resolve_params(
        dataset_config=args.dataset_config,
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
