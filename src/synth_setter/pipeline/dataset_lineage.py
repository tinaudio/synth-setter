"""Discover immutable W&B dataset artifacts from finalized dataset provenance.

For example, ``dataset_artifact_ref("r2://bucket/run")`` returns the
``data-<task>:<run_id>`` artifact reference declared by that run's frozen spec.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import structlog

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.spec_io import load_spec_from_root

log = structlog.get_logger(__name__)


def dataset_artifact_ref(
    dataset_root: str | Path | None, download_dataset_root_uri: str | None = None
) -> tuple[str, str] | None:
    """Return the immutable W&B dataset artifact declared by a finalized root.

    :param dataset_root: Optional local finalized dataset directory.
    :param download_dataset_root_uri: Optional R2 dataset root; preferred over
        ``dataset_root`` so lineage discovery does not hydrate a Lightning datamodule.
    :returns: Canonical ``(artifact_name, immutable_run_id)`` pair, or ``None``
        when no root has a readable frozen spec.
    """
    if download_dataset_root_uri is not None:
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
            if remote_ref is not None:
                return remote_ref
    return _artifact_ref_from_root(dataset_root)


def _artifact_ref_from_root(dataset_root: str | Path | None) -> tuple[str, str] | None:
    """Load an immutable dataset artifact reference from one local or remote root.

    :param dataset_root: Optional root containing a frozen ``input_spec.json``.
    :returns: Canonical ``(artifact_name, immutable_run_id)`` pair, or ``None``
        when the root has no readable frozen spec.
    """
    if dataset_root is None:
        return None
    try:
        spec = load_spec_from_root(str(dataset_root))
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        log.warning("dataset_lineage_unavailable", dataset_root=str(dataset_root), error=str(exc))
        return None
    return (f"data-{spec.task_name}", spec.run_id)
