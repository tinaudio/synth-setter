"""Image creation config schema and loader.

Defines the inputs needed to create a Docker image build, validated at the YAML trust boundary via
Pydantic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator

_HEX_CHARS = frozenset("0123456789abcdef")


class ImageConfig(BaseModel, strict=True, extra="forbid"):
    """Validated image creation config: static settings + runtime build inputs.

    Static fields come from the YAML config file.
    Runtime fields (github_sha, issue_number) come from the caller.
    image_config_id is derived from the config filename stem.
    """

    # --- Static fields (from YAML config, all required) ---
    dockerfile: str
    image: str
    base_image: str
    base_image_tag: str
    build_mode: Literal["source", "prebuilt"]
    target_platform: Literal["linux/amd64", "linux/arm64"]
    torch_index_url: str
    r2_endpoint: str
    r2_bucket: str

    # --- Runtime fields (from caller, no defaults) ---
    github_sha: str
    issue_number: int
    image_config_id: str

    @field_validator("github_sha")
    @classmethod
    def github_sha_must_be_40_hex(cls, v: str) -> str:
        """Reject anything that isn't a full lowercase commit SHA."""
        if len(v) != 40 or not _HEX_CHARS.issuperset(v):
            raise ValueError("github_sha must be a 40-character lowercase hex string")
        return v

    @field_validator("issue_number")
    @classmethod
    def issue_number_must_be_positive(cls, v: int) -> int:
        """Reject zero or negative issue numbers."""
        if v <= 0:
            raise ValueError("issue_number must be a positive integer")
        return v


def load_image_config(
    config_path: Path,
    *,
    github_sha: str,
    issue_number: int,
) -> ImageConfig:
    """Load image config from YAML and merge with runtime inputs.

    Reads static fields from the YAML config file, merges with runtime
    inputs, validates via Pydantic, and derives image_config_id from
    the config filename stem.

    Args:
        config_path: Path to YAML config under configs/image/.
        github_sha: 40-char lowercase hex commit SHA.
        issue_number: Positive GitHub issue number.

    Returns:
        Validated ImageConfig with all fields populated.

    Raises:
        FileNotFoundError: config_path doesn't exist or isn't a file.
        ValueError: top-level YAML is not a mapping.
        pydantic.ValidationError: invalid field values.
    """
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    raw = yaml.safe_load(config_path.read_text())

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"Top-level YAML in {config_path} must be a mapping, got {type(raw).__name__}"
        )

    raw.update(
        {
            "github_sha": github_sha,
            "issue_number": issue_number,
            "image_config_id": config_path.stem,
        }
    )

    return ImageConfig(**raw)
