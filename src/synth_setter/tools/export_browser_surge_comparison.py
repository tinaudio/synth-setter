"""Export the four requested checkpoint snapshots and their browser catalog."""

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator

from synth_setter.cli.clap_render import resolve_inverse_checkpoint
from synth_setter.cli.sketch_render import _producer_revision, _resolve_stats
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.resources import configs_dir
from synth_setter.tools.export_browser_surge_bundle import export_browser_surge_bundle


class _CheckpointSpec(BaseModel, strict=True, extra="forbid"):
    """Pin checkpoint bytes and the normalization selected for their evaluation.

    .. attribute :: id
        Stable browser model identifier.

    .. attribute :: param_spec
        Registered output encoding.

    .. attribute :: sketch
        Whether sketch conditioning was trained.

    .. attribute :: step
        Required checkpoint global step.

    .. attribute :: epoch
        Required checkpoint epoch.

    .. attribute :: checkpoint
        Original R2 checkpoint object.

    .. attribute :: sha256
        Required checkpoint SHA-256.

    .. attribute :: md5
        Owner-supplied secondary checksum.

    .. attribute :: bytes
        Required checkpoint byte count.

    .. attribute :: stats
        Evaluation normalization source.

    .. attribute :: stats_sha256
        Required statistics SHA-256.
    """

    id: Literal["simple-baseline", "simple-sketch", "xt-baseline", "xt-sketch"]
    param_spec: Literal["surge_simple", "surge_xt"]
    sketch: bool
    step: int = Field(ge=0)
    epoch: int = Field(ge=0)
    checkpoint: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    md5: str = Field(pattern=r"^[a-f0-9]{32}$")
    bytes: int = Field(gt=0)
    stats: str
    stats_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class _Settings(BaseModel, strict=True, extra="forbid"):
    """Require exactly one of each requested comparison arm.

    .. attribute :: models
        Four distinct digest-pinned checkpoint definitions.
    """

    models: list[_CheckpointSpec] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_arms(self) -> Self:
        """Reject duplicate or mislabeled model definitions.

        :returns: The validated catalog settings.
        :raises ValueError: Model identifiers disagree with their capabilities.
        """
        if len({model.id for model in self.models}) != 4:
            raise ValueError("comparison requires four distinct models")
        for model in self.models:
            if (
                model.id.startswith("simple") != (model.param_spec == "surge_simple")
                or model.id.endswith("sketch") != model.sketch
            ):
                raise ValueError("model identifier does not match its capabilities")
        return self


def _checkpoint(row: _CheckpointSpec) -> Path:
    """Resolve trusted checkpoint bytes and verify their recorded training identity.

    :param row: Required checkpoint identity and checksums.
    :returns: Verified local checkpoint.
    :raises ValueError: Byte count, secondary checksum, or metadata disagrees.
    """
    path = resolve_inverse_checkpoint(row.checkpoint, row.sha256)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "md5").hexdigest()
    if path.stat().st_size != row.bytes or digest != row.md5:
        raise ValueError(f"checkpoint size/MD5 mismatch: {row.id}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    if (
        checkpoint["global_step"] != row.step
        or checkpoint["epoch"] != row.epoch
        or hparams["param_spec"] != row.param_spec
        or (hparams["sketch_controls"] is not None) != row.sketch
    ):
        raise ValueError(f"checkpoint training identity mismatch: {row.id}")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    """Publish four executable model directories without overwriting existing output.

    :param argv: Optional CLI arguments excluding the executable name.
    :raises FileExistsError: Output already exists.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"comparison destination already exists: {output}")
    with configs_dir().joinpath("browser_surge_comparison.yaml").open("r") as source:
        settings = _Settings.model_validate(
            OmegaConf.to_container(OmegaConf.load(source), resolve=True)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    revision = _producer_revision()
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary)
        entries = []
        sampling = None
        for row in settings.models:
            checkpoint = _checkpoint(row)
            stats = _resolve_stats(row.stats, row.stats_sha256)
            model = VSTFlowMatchingModule.load_from_checkpoint(
                checkpoint, map_location="cpu", weights_only=False
            ).eval()
            manifest = export_browser_surge_bundle(
                model,
                stats,
                staging / row.id,
                checkpoint_sha256=row.sha256,
                git_revision=revision,
            )
            if sampling is None:
                sampling = manifest["sampling"]
            entries.append(
                {
                    "id": row.id,
                    "label": f"{row.param_spec} {'sketch' if row.sketch else 'baseline'} — {row.step:,} steps",
                    "sketch": row.sketch,
                    "checkpointSha256": row.sha256,
                    "checkpointSource": row.checkpoint,
                    "trainingStep": row.step,
                    "epoch": row.epoch,
                    "encodedWidth": manifest["encodedWidth"],
                    "statsSha256": row.stats_sha256,
                    "statsSource": row.stats,
                }
            )
        (staging / "catalog.json").write_text(
            json.dumps({"models": entries, "sampling": sampling}, indent=2, allow_nan=False)
        )
        os.rename(staging, output)


if __name__ == "__main__":
    main()
