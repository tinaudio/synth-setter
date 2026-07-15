"""Behavioral tests for local dataset-to-W&B lineage discovery."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synth_setter.pipeline import r2_io
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

    assert dataset_artifact_ref(tmp_path) == (
        "data-surge-simple-lance",
        "surge-simple-lance-20260520T000000000Z",
    )


def test_dataset_artifact_ref_repeated_task_uses_frozen_run_id(
    tmp_path: Path, dataset_spec_factory: Callable[..., DatasetSpec]
) -> None:
    """A retained dataset links the immutable artifact version from its own spec.

    :param tmp_path: Local dataset root containing the first finalized run's spec.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    """
    spec = dataset_spec_factory(
        task_name="surge-simple-lance",
        run_id="surge-simple-lance-20260713T120000000Z",
        train_val_test_sizes=[4, 4, 0],
        r2={"bucket": "intermediate-data"},
        render={"samples_per_shard": 4},
    )
    write_spec_to_path(spec, tmp_path / "input_spec.json")

    assert dataset_artifact_ref(tmp_path) == (
        "data-surge-simple-lance",
        "surge-simple-lance-20260713T120000000Z",
    )


def test_dataset_artifact_ref_remote_root_returns_frozen_run_id(
    fake_r2_remote: Path, dataset_spec_factory: Callable[..., DatasetSpec]
) -> None:
    """A remote dataset root supplies lineage without hydrating the datamodule.

    :param fake_r2_remote: Local filesystem backing the fake ``r2:`` remote.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    """
    spec = dataset_spec_factory(
        task_name="surge-simple-lance",
        run_id="surge-simple-lance-20260713T130000000Z",
        train_val_test_sizes=[4, 4, 0],
        r2={"bucket": "intermediate-data"},
        render={"samples_per_shard": 4},
    )
    dataset_root_uri = "r2://intermediate-data/lineage-run"
    write_spec_to_path(
        spec,
        fake_r2_remote / "intermediate-data" / "lineage-run" / "input_spec.json",
    )

    assert dataset_artifact_ref(dataset_root_uri) == (
        "data-surge-simple-lance",
        "surge-simple-lance-20260713T130000000Z",
    )


def test_dataset_artifact_ref_remote_root_precedes_conflicting_local_spec(
    tmp_path: Path,
    fake_r2_remote: Path,
    dataset_spec_factory: Callable[..., DatasetSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured remote root wins over stale local provenance.

    :param tmp_path: Local root containing a conflicting frozen spec.
    :param fake_r2_remote: Local filesystem backing the fake ``r2:`` remote.
    :param dataset_spec_factory: Factory producing valid frozen dataset specs.
    :param monkeypatch: Replaces R2 credential initialization for the local remote.
    """
    local_spec = dataset_spec_factory(
        task_name="local-lineage",
        run_id="local-lineage-20260713T140000000Z",
        train_val_test_sizes=[4, 4, 0],
        r2={"bucket": "intermediate-data"},
        render={"samples_per_shard": 4},
    )
    remote_spec = dataset_spec_factory(
        task_name="remote-lineage",
        run_id="remote-lineage-20260713T150000000Z",
        train_val_test_sizes=[4, 4, 0],
        r2={"bucket": "intermediate-data"},
        render={"samples_per_shard": 4},
    )
    remote_root_uri = "r2://intermediate-data/remote-lineage"
    write_spec_to_path(local_spec, tmp_path / "input_spec.json")
    write_spec_to_path(
        remote_spec,
        fake_r2_remote / "intermediate-data" / "remote-lineage" / "input_spec.json",
    )
    ensure_r2 = MagicMock()
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", ensure_r2)

    assert dataset_artifact_ref(tmp_path, remote_root_uri) == (
        "data-remote-lineage",
        "remote-lineage-20260713T150000000Z",
    )
    ensure_r2.assert_called_once_with()


def test_dataset_artifact_ref_remote_failure_falls_back_to_local_spec(
    tmp_path: Path,
    dataset_spec_factory: Callable[..., DatasetSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed remote lookup retains lineage from a readable local root.

    :param tmp_path: Local root containing a frozen spec.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    :param monkeypatch: Replaces R2 credential initialization for the failed remote.
    """
    local_spec = dataset_spec_factory(
        task_name="local-lineage",
        run_id="local-lineage-20260713T160000000Z",
        train_val_test_sizes=[4, 4, 0],
        r2={"bucket": "intermediate-data"},
        render={"samples_per_shard": 4},
    )
    write_spec_to_path(local_spec, tmp_path / "input_spec.json")
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", MagicMock())

    assert dataset_artifact_ref(tmp_path, "r2://intermediate-data/missing") == (
        "data-local-lineage",
        "local-lineage-20260713T160000000Z",
    )


def test_dataset_artifact_ref_credential_failure_falls_back_to_local_spec(
    tmp_path: Path,
    dataset_spec_factory: Callable[..., DatasetSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable R2 credentials retain lineage from a readable local root.

    :param tmp_path: Local root containing a frozen spec.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    :param monkeypatch: Makes R2 credential initialization fail.
    """
    local_spec = dataset_spec_factory(
        task_name="local-lineage",
        run_id="local-lineage-20260713T170000000Z",
        train_val_test_sizes=[4, 4, 0],
        r2={"bucket": "intermediate-data"},
        render={"samples_per_shard": 4},
    )
    write_spec_to_path(local_spec, tmp_path / "input_spec.json")
    monkeypatch.setattr(
        r2_io,
        "ensure_r2_env_loaded",
        MagicMock(side_effect=RuntimeError("R2 credentials unavailable")),
    )

    assert dataset_artifact_ref(tmp_path, "r2://intermediate-data/missing") == (
        "data-local-lineage",
        "local-lineage-20260713T170000000Z",
    )


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
