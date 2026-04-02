"""Write image config fields to GITHUB_OUTPUT for Actions workflows."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pipeline.schemas.image_config import load_image_config


def main() -> None:
    """Load image config from YAML and write fields to GITHUB_OUTPUT."""
    parser = argparse.ArgumentParser(description="Load image config and emit to GITHUB_OUTPUT")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--github-sha", required=True, help="40-char commit SHA")
    parser.add_argument("--issue-number", required=True, type=int, help="GitHub issue number")
    args = parser.parse_args()

    cfg = load_image_config(
        Path(args.config),
        github_sha=args.github_sha,
        issue_number=args.issue_number,
    )

    output_path = os.environ.get("GITHUB_OUTPUT")
    dest = open(output_path, "a") if output_path else sys.stdout  # noqa: SIM115

    try:
        for field_name, value in cfg.model_dump().items():
            str_value = str(value)
            if "\n" in str_value or "\r" in str_value:
                raise ValueError(
                    f"GITHUB_OUTPUT value for '{field_name}' contains a newline character; "
                    "this would inject extra output keys"
                )
            dest.write(f"{field_name}={str_value}\n")
    finally:
        if dest is not sys.stdout:
            dest.close()


if __name__ == "__main__":
    main()
