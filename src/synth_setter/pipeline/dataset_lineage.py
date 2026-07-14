"""Discover W&B dataset artifacts from a local finalized dataset root."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.spec_io import load_spec_from_root


def dataset_artifact_ref(dataset_root: str | Path) -> tuple[str, str] | None:
    """Return the W&B dataset artifact declared by a local dataset root.

    :param dataset_root: Local finalized dataset directory containing ``input_spec.json``.
    :returns: The canonical ``(artifact_name, alias)`` pair, or ``None`` when
        the root has no readable frozen spec.
    """
    spec_path = Path(dataset_root) / INPUT_SPEC_FILENAME
    if not spec_path.is_file():
        return None
    try:
        spec = load_spec_from_root(str(dataset_root))
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        logger.warning("dataset lineage unavailable for {}: {}", dataset_root, exc)
        return None
    return (f"data-{spec.task_name}", "latest")
