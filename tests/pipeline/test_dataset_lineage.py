"""Behavioral tests for local dataset-to-W&B lineage discovery."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from synth_setter.pipeline.dataset_lineage import dataset_artifact_ref
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import write_spec_to_path


def test_dataset_artifact_ref_valid_local_spec_returns_dataset_artifact(
    tmp_path: Path, dataset_spec_factory: Callable[..., DatasetSpec]
) -> None:
    """A copied finalized dataset root links to its declared W&B dataset artifact.

    :param tmp_path: Local dataset root containing its persisted input spec.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    """
    spec = dataset_spec_factory(
        task_name="surge-simple-lance",
        train_val_test_sizes=[4, 4, 0],
        r2={"bucket": "intermediate-data"},
        render={"samples_per_shard": 4},
    )
    write_spec_to_path(spec, tmp_path / "input_spec.json")

    assert dataset_artifact_ref(tmp_path) == ("data-surge-simple-lance", "latest")


def test_dataset_artifact_ref_missing_spec_returns_none(tmp_path: Path) -> None:
    """A local dataset without generation provenance remains usable without a link.

    :param tmp_path: Local dataset root without an input spec.
    """
    assert dataset_artifact_ref(tmp_path) is None


def test_dataset_artifact_ref_invalid_spec_returns_none(tmp_path: Path) -> None:
    """Malformed local provenance cannot create an untrusted lineage edge.

    :param tmp_path: Local dataset root containing an invalid input spec.
    """
    (tmp_path / "input_spec.json").write_text("{}", encoding="utf-8")

    assert dataset_artifact_ref(tmp_path) is None
