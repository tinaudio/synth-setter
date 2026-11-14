"""Behavioral tests for local dataset-to-W&B lineage discovery."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.dataset_lineage import (
    dataset_artifact_ref,
    describe_unresolved_dataset_root,
)
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


def test_dataset_artifact_ref_legacy_local_spec_returns_dataset_artifact(tmp_path: Path) -> None:
    """Lineage identity survives unrelated frozen-spec schema drift.

    :param tmp_path: Local dataset root containing a legacy input spec.
    """
    legacy_spec = {
        "task_name": "surge-simple-lance-440k-20k-20k",
        "run_id": "surge-simple-lance-440k-20k-20k-20260706T005448315Z",
        "copy_dataset_root_uri": "r2://obsolete/source",
        "render": {"synth": {"name": "surge_simple"}},
    }
    (tmp_path / "input_spec.json").write_text(json.dumps(legacy_spec), encoding="utf-8")

    assert dataset_artifact_ref(tmp_path) == (
        "data-surge-simple-lance-440k-20k-20k",
        "surge-simple-lance-440k-20k-20k-20260706T005448315Z",
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


def test_dataset_artifact_ref_legacy_remote_spec_returns_dataset_artifact(
    fake_r2_remote: Path,
) -> None:
    """Remote lineage identity survives unrelated frozen-spec schema drift.

    :param fake_r2_remote: Local filesystem backing the fake ``r2:`` remote.
    """
    legacy_spec = {
        "task_name": "surge-simple-lance-440k-20k-20k",
        "run_id": "surge-simple-lance-440k-20k-20k-20260706T005448315Z",
        "render": {"synth": {"name": "surge_simple"}},
    }
    spec_path = fake_r2_remote / "intermediate-data" / "legacy-lineage-run" / "input_spec.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(json.dumps(legacy_spec), encoding="utf-8")

    assert dataset_artifact_ref("r2://intermediate-data/legacy-lineage-run") == (
        "data-surge-simple-lance-440k-20k-20k",
        "surge-simple-lance-440k-20k-20k-20260706T005448315Z",
    )


def test_dataset_artifact_ref_file_uri_skips_r2_preflight(
    tmp_path: Path,
    dataset_spec_factory: Callable[..., DatasetSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mounted source URI resolves lineage without requiring rclone.

    :param tmp_path: Local source root containing its persisted input spec.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    :param monkeypatch: Fails the test if the R2 preflight is called.
    """
    spec = dataset_spec_factory(
        task_name="mounted-lineage",
        run_id="mounted-lineage-20260724T000000000Z",
        train_val_test_sizes=[4, 4, 0],
        r2={"bucket": "intermediate-data"},
        render={"samples_per_shard": 4},
    )
    write_spec_to_path(spec, tmp_path / "input_spec.json")
    ensure_r2 = MagicMock(side_effect=AssertionError("R2 preflight called"))
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", ensure_r2)

    assert dataset_artifact_ref(None, tmp_path.as_uri()) == (
        "data-mounted-lineage",
        "mounted-lineage-20260724T000000000Z",
    )
    ensure_r2.assert_not_called()


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
    download = MagicMock(side_effect=subprocess.CalledProcessError(1, ["rclone", "copyto"]))
    monkeypatch.setattr(r2_io, "download_to_path", download)

    assert dataset_artifact_ref(tmp_path, "r2://intermediate-data/missing") == (
        "data-local-lineage",
        "local-lineage-20260713T160000000Z",
    )
    download.assert_called_once_with("r2://intermediate-data/missing/input_spec.json", ANY)


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_name", ""),
        ("run_id", ""),
        ("task_name", " "),
        ("run_id", "\t"),
        ("task_name", 440),
        ("run_id", 440),
    ],
)
def test_dataset_artifact_ref_invalid_identity_returns_none(
    tmp_path: Path, field: str, value: str | int
) -> None:
    """Blank or non-string identity fields cannot create a lineage edge.

    :param tmp_path: Local dataset root containing malformed lineage identity.
    :param field: Identity field made invalid.
    :param value: Invalid frozen value.
    """
    identity: dict[str, object] = {
        "task_name": "surge-simple-lance-440k-20k-20k",
        "run_id": "surge-simple-lance-440k-20k-20k-20260706T005448315Z",
    }
    identity[field] = value
    (tmp_path / "input_spec.json").write_text(json.dumps(identity), encoding="utf-8")

    assert dataset_artifact_ref(tmp_path) is None


def test_describe_unresolved_dataset_root_prefers_the_remote_uri() -> None:
    """The description names the root discovery actually reads — the remote URI."""
    assert (
        describe_unresolved_dataset_root("/datasets/local", "r2://intermediate-data/run")
        == "dataset root r2://intermediate-data/run"
    )


def test_describe_unresolved_dataset_root_falls_back_to_the_local_root() -> None:
    """With no remote URI configured, the local root is what failed to resolve."""
    assert describe_unresolved_dataset_root("/datasets/local") == "dataset root /datasets/local"


def test_describe_unresolved_dataset_root_without_any_root_returns_none() -> None:
    """No configured root means nothing was expected to resolve, so nothing is reported."""
    assert describe_unresolved_dataset_root(None, None) is None
