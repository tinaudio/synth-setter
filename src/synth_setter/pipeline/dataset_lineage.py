"""Discover immutable W&B dataset artifacts from finalized dataset provenance.

For example, ``dataset_artifact_ref("r2://bucket/run")`` returns the
``data-<task>:<run_id>`` artifact reference declared by that run's frozen spec.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import structlog
from pydantic import BaseModel, StringConstraints

from synth_setter.param_spec_name import is_onset_duration_param_spec_name
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.spec_io import join_uri, read_spec_text

log = structlog.get_logger(__name__)

type _LineageId = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class _DatasetLineageIdentity(BaseModel, extra="ignore", strict=True):
    """Stable identity projection from a frozen dataset spec.

    .. attribute :: task_name
        :type: str

        Dataset task used in the W&B artifact name.

    .. attribute :: run_id
        :type: str

        Immutable W&B artifact alias for the finalized dataset.
    """

    task_name: _LineageId
    run_id: _LineageId


class _DatasetSynthIdentity(BaseModel, extra="ignore", strict=True):
    param_spec_name: _LineageId


class _DatasetRenderIdentity(BaseModel, extra="ignore", strict=True):
    synth: _DatasetSynthIdentity


class _DatasetParamSpecIdentity(BaseModel, extra="ignore", strict=True):
    render: _DatasetRenderIdentity


def dataset_artifact_ref(
    dataset_root: str | Path | None, download_dataset_root_uri: str | None = None
) -> tuple[str, str] | None:
    """Return the immutable W&B dataset artifact declared by a finalized root.

    :param dataset_root: Optional local finalized dataset directory.
    :param download_dataset_root_uri: Optional R2 or file URI; preferred over
        ``dataset_root`` so lineage discovery does not hydrate a Lightning datamodule.
    :returns: Canonical ``(artifact_name, immutable_run_id)`` pair, or ``None``
        when no root has a readable frozen spec.
    """
    identity = _frozen_spec_projection(
        _DatasetLineageIdentity, dataset_root, download_dataset_root_uri
    )
    return None if identity is None else (f"data-{identity.task_name}", identity.run_id)


def validate_dataset_param_spec(
    dataset_root: str | Path | None,
    download_dataset_root_uri: str | None,
    param_spec_name: str,
) -> None:
    """Reject a run that would decode dataset rows under the other note-timing encoding.

    An endpoint-timed spec and its onset-duration twin share one width, so the name in the
    frozen spec is the only thing that tells their rows apart. Roots that predate frozen
    specs hold endpoint-timed rows, so only an endpoint-timed run accepts an unverifiable
    one. Two endpoint-timed names are not compared: their rows differ in width, and a
    stale frozen spec must not block a run that needs no timing check.

    :param dataset_root: Optional local finalized dataset directory.
    :param download_dataset_root_uri: Optional R2 or file URI, preferred over ``dataset_root``.
    :param param_spec_name: Spec the run encodes and decodes parameter rows with.
    :raises ValueError: Either side selects onset-duration timing and the dataset does
        not name the run's spec.
    """
    identity = _frozen_spec_projection(
        _DatasetParamSpecIdentity, dataset_root, download_dataset_root_uri
    )
    dataset_spec_name = None if identity is None else identity.render.synth.param_spec_name
    if dataset_spec_name == param_spec_name:
        return
    if dataset_spec_name is None:
        if not is_onset_duration_param_spec_name(param_spec_name):
            return
        root = describe_unresolved_dataset_root(dataset_root, download_dataset_root_uri)
        raise ValueError(
            f"cannot verify that {root or 'the dataset'} holds "
            f"param_spec_name={param_spec_name!r} rows: no readable {INPUT_SPEC_FILENAME} "
            "names its parameter spec"
        )
    if is_onset_duration_param_spec_name(param_spec_name) or is_onset_duration_param_spec_name(
        dataset_spec_name
    ):
        raise ValueError(
            f"dataset rows are encoded with param_spec_name={dataset_spec_name!r}, "
            f"run requested {param_spec_name!r}"
        )


def validate_datamodule_dataset(datamodule: Mapping[str, object]) -> None:
    """Check a composed datamodule node against the dataset it would read.

    :param datamodule: Composed ``datamodule`` config node; rejected like
        :func:`validate_dataset_param_spec` when its dataset names another spec.
    """
    param_spec_name = datamodule.get("param_spec_name")
    # Fake mode synthesizes rows at the spec's width and reads no dataset.
    if param_spec_name is None or datamodule.get("fake"):
        return
    dataset_root = datamodule.get("dataset_root")
    download_uri = datamodule.get("download_dataset_root_uri")
    validate_dataset_param_spec(
        None if dataset_root is None else str(dataset_root),
        None if download_uri is None else str(download_uri),
        str(param_spec_name),
    )


def _frozen_spec_projection[ProjectionT: BaseModel](
    projection: type[ProjectionT],
    dataset_root: str | Path | None,
    download_dataset_root_uri: str | None,
) -> ProjectionT | None:
    """Project the first readable frozen spec, preferring the remote root.

    :param projection: Model selecting the frozen-spec fields the caller needs.
    :param dataset_root: Optional local finalized dataset directory.
    :param download_dataset_root_uri: Optional R2 or file URI, preferred over ``dataset_root``.
    :returns: Validated projection, or ``None`` when no root has a readable matching spec.
    """
    if download_dataset_root_uri is not None:
        remote = None
        if r2_io.is_r2_uri(download_dataset_root_uri):
            try:
                r2_io.ensure_r2_env_loaded()
            except RuntimeError as exc:
                log.warning(
                    "dataset_lineage_remote_unavailable",
                    dataset_root=download_dataset_root_uri,
                    error=str(exc),
                )
            else:
                remote = _projection_from_root(projection, download_dataset_root_uri)
        else:
            remote = _projection_from_root(projection, download_dataset_root_uri)
        if remote is not None:
            return remote
    return _projection_from_root(projection, dataset_root)


def describe_unresolved_dataset_root(
    dataset_root: str | Path | None, download_dataset_root_uri: str | None = None
) -> str | None:
    """Name the configured dataset root that :func:`dataset_artifact_ref` could not resolve.

    Callers use this only after that function returned ``None``, to say in the
    run's durable lineage marker *which* input has no edge (#2424).

    :param dataset_root: Optional local finalized dataset directory.
    :param download_dataset_root_uri: Optional R2 or file URI, preferred as the
        root lineage discovery reads.
    :returns: A human-readable description of the configured root, or ``None``
        when no root is configured at all (nothing was expected to resolve).
    """
    root = download_dataset_root_uri or dataset_root
    return f"dataset root {root}" if root else None


def _projection_from_root[ProjectionT: BaseModel](
    projection: type[ProjectionT], dataset_root: str | Path | None
) -> ProjectionT | None:
    """Project the frozen spec under one local or remote root.

    :param projection: Model selecting the frozen-spec fields the caller needs.
    :param dataset_root: Optional root containing a frozen ``input_spec.json``.
    :returns: Validated projection, or ``None`` when the root has no readable matching spec.
    """
    if dataset_root is None:
        return None
    try:
        spec_uri = join_uri(str(dataset_root), INPUT_SPEC_FILENAME)
        return projection.model_validate_json(read_spec_text(spec_uri))
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        log.warning("dataset_lineage_unavailable", dataset_root=str(dataset_root), error=str(exc))
        return None
