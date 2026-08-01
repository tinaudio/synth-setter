"""Discover immutable W&B dataset artifacts from finalized dataset provenance.

For example, ``dataset_artifact_ref("r2://bucket/run")`` returns the
``data-<task>:<run_id>`` artifact reference declared by that run's frozen spec.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import structlog
from pydantic import BaseModel, StringConstraints

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.spec_io import join_uri, read_spec_text

log = structlog.get_logger(__name__)

_LineageId = Annotated[str, StringConstraints(min_length=1)]


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
    if download_dataset_root_uri is not None:
        remote_ref = None
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
                remote_ref = _artifact_ref_from_root(download_dataset_root_uri)
        else:
            remote_ref = _artifact_ref_from_root(download_dataset_root_uri)
        if remote_ref is not None:
            return remote_ref
    return _artifact_ref_from_root(dataset_root)


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


def _artifact_ref_from_root(dataset_root: str | Path | None) -> tuple[str, str] | None:
    """Load an immutable dataset artifact reference from one local or remote root.

    :param dataset_root: Optional root containing a frozen ``input_spec.json``.
    :returns: Canonical ``(artifact_name, immutable_run_id)`` pair, or ``None``
        when the root has no readable frozen spec.
    """
    if dataset_root is None:
        return None
    try:
        spec_uri = join_uri(str(dataset_root), INPUT_SPEC_FILENAME)
        identity = _DatasetLineageIdentity.model_validate_json(read_spec_text(spec_uri))
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        log.warning("dataset_lineage_unavailable", dataset_root=str(dataset_root), error=str(exc))
        return None
    return (f"data-{identity.task_name}", identity.run_id)
