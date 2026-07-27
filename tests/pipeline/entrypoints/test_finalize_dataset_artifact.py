"""Unit tests for ``finalize_dataset.build_dataset_artifact``.

Asserts on a real ``wandb.Artifact`` object (no run, no network): name,
type, R2 references, and metadata are all observable on the returned
artifact, so these tests exercise the real construction rather than a
mock of it. The offline end-to-end logging path is covered separately in
``test_finalize_dataset.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock

import pytest
from lightning.pytorch.loggers import Logger

from synth_setter.cli import finalize_dataset
from synth_setter.cli.finalize_dataset import build_dataset_artifact
from synth_setter.pipeline.schemas.spec import DatasetSpec

_RUN_ID = "finalize-art-20260520T000000000Z"
_BUCKET = "intermediate-data"
_PREFIX = f"data/finalize-art/{_RUN_ID}/"


def _spec(
    factory: Callable[..., DatasetSpec],
    train_val_test_sizes: tuple[int, int, int] = (8, 4, 0),
) -> DatasetSpec:
    """Build a finalize-artifact Lance spec with a fixed run id, bucket, and prefix.

    :param factory: Shared ``conftest`` ``dataset_spec_factory``.
    :param train_val_test_sizes: Split sample counts (multiples of 4); the
        default leaves ``test`` empty so the empty-split omission is testable.
    :returns: A frozen ``DatasetSpec`` whose ``num_shards`` and ``r2`` are
        deterministic.
    """
    return factory(
        task_name="finalize-art",
        run_id=_RUN_ID,
        train_val_test_sizes=list(train_val_test_sizes),
        r2={"bucket": _BUCKET, "prefix": _PREFIX},
        render={"samples_per_render_batch": 4, "samples_per_shard": 4},
    )


def test_build_dataset_artifact_name_is_data_prefixed_config_id(
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """The artifact name is ``data-{task_name}`` per storage-provenance-spec §4.

    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    artifact = build_dataset_artifact(_spec(dataset_spec_factory))
    assert artifact.name == "data-finalize-art"


def test_build_dataset_artifact_type_is_dataset(
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """The artifact type is ``dataset`` per storage-provenance-spec §4.

    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    artifact = build_dataset_artifact(_spec(dataset_spec_factory))
    assert artifact.type == "dataset"


def test_log_dataset_artifact_aliases_frozen_run_id(
    dataset_spec_factory: Callable[..., DatasetSpec], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finalized dataset can be consumed through its immutable run-id alias.

    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    :param monkeypatch: Replaces the Lightning logger type at the W&B boundary.
    """

    class FakeWandbLogger:
        """Minimal W&B logger boundary fake."""

        def __init__(self) -> None:
            self.experiment = MagicMock()

    monkeypatch.setattr(finalize_dataset, "WandbLogger", FakeWandbLogger)
    logger = FakeWandbLogger()

    finalize_dataset._log_dataset_artifact(
        cast(list[Logger], [logger]), _spec(dataset_spec_factory)
    )

    (artifact,) = logger.experiment.log_artifact.call_args.args
    assert artifact.name == "data-finalize-art"
    assert logger.experiment.log_artifact.call_args.kwargs == {"aliases": [_RUN_ID]}


def test_build_dataset_artifact_metadata_carries_shard_count_n_samples_git_sha(
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """Metadata records shard_count, n_samples, and git_sha per storage-provenance-spec §6.

    (8 train + 4 val) / 4 per shard = 3 shards; 12 total samples; the factory
    pins ``git_sha`` to 40 zeros.

    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    artifact = build_dataset_artifact(_spec(dataset_spec_factory))
    assert artifact.metadata == {"shard_count": 3, "n_samples": 12, "git_sha": "0" * 40}


def test_build_dataset_artifact_references_nonempty_lance_splits_and_stats(
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """Lance references each non-empty split ``.lance`` plus ``stats.npz`` as ``s3://`` URIs.

    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    artifact = build_dataset_artifact(_spec(dataset_spec_factory))
    refs = {entry.ref for entry in artifact.manifest.entries.values()}
    assert refs == {
        f"s3://{_BUCKET}/{_PREFIX}train.lance",
        f"s3://{_BUCKET}/{_PREFIX}val.lance",
        f"s3://{_BUCKET}/{_PREFIX}stats.npz",
    }


def test_build_dataset_artifact_omits_empty_lance_split_reference(
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """An empty split (``test`` size 0) contributes no reference — nothing was finalized there.

    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    artifact = build_dataset_artifact(_spec(dataset_spec_factory))
    refs = {entry.ref for entry in artifact.manifest.entries.values()}
    assert f"s3://{_BUCKET}/{_PREFIX}test.lance" not in refs
