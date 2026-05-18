"""Image creation config schema and loader.

Defines the inputs needed to create a Docker image build, validated at the YAML trust boundary via
Pydantic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

_HEX_CHARS = frozenset("0123456789abcdef")


class ImageConfig(BaseModel, strict=True, extra="forbid"):
    """Validated image creation config: static YAML settings + runtime build inputs from the caller."""

    # Static fields (from YAML)
    dockerfile: str = Field(description="Repo-relative path to the Dockerfile to build.")
    image: str = Field(description="Final image reference (`registry/repo:tag`) the build pushes.")
    base_image: str = Field(
        description="Base image reference the Dockerfile's `FROM` line is rewritten to."
    )
    base_image_tag: str = Field(
        description="Base image tag pinned at build time (paired with `base_image`)."
    )
    build_mode: Literal["source", "prebuilt"] = Field(
        description=(
            "`source` builds the full stack from the Dockerfile; `prebuilt` rewrites the base "
            "reference and skips rebuilding upstream layers."
        )
    )
    target_platform: Literal["linux/amd64", "linux/arm64"] = Field(
        description="Docker BuildKit target platform."
    )
    torch_backend: str = Field(
        description="Torch wheel selector (e.g. `cu121`, `cpu`) that picks the PyTorch index."
    )

    # Runtime fields (from caller)
    github_sha: str = Field(
        description="Commit SHA the build is tied to (provenance label and tag input)."
    )
    issue_number: int = Field(
        description="GitHub issue this build is tied to; appears in image tags and OCI labels."
    )
    image_config_id: str = Field(
        description="Derived from the YAML config filename stem (e.g. `runpod-h100.yaml` → `runpod-h100`)."
    )

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

    Reads static fields from the YAML config file, merges with runtime inputs, validates via
    Pydantic, and derives image_config_id from the config filename stem.

    :param config_path: Path to YAML config under configs/image/.
    :param github_sha: 40-char lowercase hex commit SHA.
    :param issue_number: Positive GitHub issue number.
    :return: Validated ImageConfig with all fields populated.
    :raises FileNotFoundError: config_path doesn't exist or isn't a file.
    :raises ValueError: top-level YAML is not a mapping.
    :raises pydantic.ValidationError: invalid field values.
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
