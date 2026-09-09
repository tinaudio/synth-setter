"""Archive inline dataset oracle-evaluation artifacts without eval lineage.

Publish a completed split with its source/candidate provenance::

    uri = upload_oracle_probe(
        eval_dir, r2=spec.r2, launch_id=launch_id, provenance=provenance
    )
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.r2_location import R2Location
from synth_setter.pipeline.schemas.spec import RenderConfig, Split
from synth_setter.utils.logging_utils import resolve_git_sha

_PROVENANCE_FILENAME = "provenance.json"
_UPLOAD_EXCLUDE = "predictions/**"

logger = structlog.get_logger(__name__)


class OracleProbeProvenance(BaseModel):
    """Describe the dataset and render configurations compared by an oracle probe.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.

    .. attribute :: source_dataset_uri

        URI of the finalized source split consumed by evaluation.

    .. attribute :: source_dataset_task

        Dataset task identity used in the probe namespace.

    .. attribute :: source_split

        Finalized dataset split consumed by evaluation.

    .. attribute :: source_run_id

        Dataset run identity used in the probe namespace.

    .. attribute :: source_render

        Render configuration that produced the source dataset.

    .. attribute :: candidate_render

        Render configuration evaluated against the source.

    .. attribute :: evaluation_git_sha

        Current synth-setter revision that executed the evaluation.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    source_dataset_uri: str = Field(description="Finalized source split URI.")
    source_dataset_task: str = Field(min_length=1, description="Source dataset task identity.")
    source_split: Split = Field(description="Source dataset split.")
    source_run_id: str = Field(min_length=1, description="Source dataset run identity.")
    source_render: RenderConfig = Field(description="Source dataset render configuration.")
    candidate_render: RenderConfig = Field(description="Candidate render configuration.")
    evaluation_git_sha: str = Field(
        default_factory=resolve_git_sha,
        min_length=1,
        description="Current synth-setter git revision that executed the evaluation.",
    )

    @field_validator("source_dataset_uri")
    @classmethod
    def _source_dataset_uri_must_be_r2(cls, value: str) -> str:
        """Reject provenance that cannot identify an R2 source dataset.

        :param value: Candidate source dataset URI.
        :returns: The validated R2 URI.
        :raises ValueError: ``value`` is not an ``r2://`` URI.
        """
        if not r2_io.is_r2_uri(value):
            raise ValueError("source_dataset_uri must be an r2:// URI")
        return value


def new_oracle_probe_launch_id() -> str:
    """Mint a UUID so concurrent inline invocations use disjoint prefixes.

    :returns: Hexadecimal UUID without path separators.
    """
    return uuid4().hex


def _copy_probe_artifacts(eval_dir: Path, archive_dir: Path) -> None:
    """Stage the config, rendered audio, and metrics while omitting run logs.

    :param eval_dir: Completed Hydra eval run directory.
    :param archive_dir: Empty directory receiving the durable artifacts.
    :raises FileNotFoundError: A required eval artifact is absent.
    """
    config_path = eval_dir / ".hydra" / "config.yaml"
    artifact_dirs = (eval_dir / "audio", eval_dir / "metrics")
    missing = [path for path in (config_path, *artifact_dirs) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"oracle probe artifacts missing from {eval_dir}: {missing}")

    staged_config = archive_dir / ".hydra" / "config.yaml"
    staged_config.parent.mkdir(parents=True)
    shutil.copy2(config_path, staged_config)
    for source in artifact_dirs:
        shutil.copytree(source, archive_dir / source.name)


def upload_oracle_probe(
    eval_dir: Path,
    *,
    r2: R2Location,
    launch_id: str,
    provenance: OracleProbeProvenance,
) -> str:
    """Upload one eval split's durable probe artifacts and return its R2 URI.

    :param eval_dir: Completed Hydra eval run directory.
    :param r2: Source dataset storage location; its bucket owns the probe.
    :param launch_id: Identity shared by all splits in one inline invocation.
    :param provenance: Source and candidate identities persisted with the probe.
    :returns: Destination URI containing config, audio, metrics, and provenance.
    """
    destination = r2.uri(
        "probes/dataset-oracle/"
        f"{provenance.source_dataset_task}/{provenance.source_run_id}/"
        f"{launch_id}/{provenance.source_split}"
    )
    with tempfile.TemporaryDirectory(prefix="synth-setter-oracle-probe-") as temp_dir:
        archive_dir = Path(temp_dir)
        _copy_probe_artifacts(eval_dir, archive_dir)
        r2_io.upload_dir(archive_dir, destination, exclude=_UPLOAD_EXCLUDE)
        provenance_path = archive_dir / _PROVENANCE_FILENAME
        provenance_path.write_text(provenance.model_dump_json(indent=2) + "\n", encoding="utf-8")
        r2_io.upload_to_uri(provenance_path, f"{destination}/{_PROVENANCE_FILENAME}")
    logger.info("oracle_probe_uploaded", uri=destination)
    return destination
