"""Unit tests for ``finalize_dataset.build_dataset_artifact``.

Asserts on a real ``wandb.Artifact`` object (no run, no network): name,
type, R2 references, and metadata are all observable on the returned
artifact, so these tests exercise the real construction rather than a
mock of it. The offline end-to-end logging path is covered separately in
``test_finalize_dataset.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from synth_setter.cli.finalize_dataset import _r2_to_s3_uri, build_dataset_artifact
from synth_setter.pipeline.schemas.spec import DatasetSpec

_RUN_ID = "finalize-art-20260520T000000000Z"
_BUCKET = "intermediate-data"
_PREFIX = f"data/finalize-art/{_RUN_ID}/"


def _spec(
    factory: Callable[..., DatasetSpec],
    output_format: str = "hdf5",
    train_val_test_sizes: tuple[int, int, int] = (8, 4, 0),
) -> DatasetSpec:
    """Build a finalize-artifact spec with a fixed run id, bucket, and prefix.

    :param factory: Shared ``conftest`` ``dataset_spec_factory``.
    :param output_format: ``hdf5`` or ``wds``; selects the reference shape.
    :param train_val_test_sizes: Split sample counts (multiples of 4); the
        default leaves ``test`` empty so the empty-split omission is testable.
    :returns: A frozen ``DatasetSpec`` whose ``num_shards`` and ``r2`` are
        deterministic.
    """
    return factory(
        task_name="finalize-art",
        run_id=_RUN_ID,
        output_format=output_format,
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


def test_build_dataset_artifact_hdf5_references_nonempty_splits_and_stats(
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """hdf5 references each non-empty split ``.h5`` plus ``stats.npz`` as ``s3://`` URIs.

    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    artifact = build_dataset_artifact(_spec(dataset_spec_factory))
    refs = {entry.ref for entry in artifact.manifest.entries.values()}
    assert refs == {
        f"s3://{_BUCKET}/{_PREFIX}train.h5",
        f"s3://{_BUCKET}/{_PREFIX}val.h5",
        f"s3://{_BUCKET}/{_PREFIX}stats.npz",
    }


def test_build_dataset_artifact_hdf5_omits_empty_split_reference(
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """An empty split (``test`` size 0) contributes no reference — nothing was finalized there.

    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    artifact = build_dataset_artifact(_spec(dataset_spec_factory))
    refs = {entry.ref for entry in artifact.manifest.entries.values()}
    assert f"s3://{_BUCKET}/{_PREFIX}test.h5" not in refs


def test_build_dataset_artifact_wds_references_run_prefix_and_stats(
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """Wds references the run prefix dir (carrying the shard tars) plus ``stats.npz``.

    wds keeps shards in place rather than resharding into split ``.h5`` files,
    so the dataset footprint is the prefix dir plus the derived ``stats.npz``.

    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    artifact = build_dataset_artifact(_spec(dataset_spec_factory, output_format="wds"))
    refs = {entry.ref for entry in artifact.manifest.entries.values()}
    assert refs == {
        f"s3://{_BUCKET}/{_PREFIX}",
        f"s3://{_BUCKET}/{_PREFIX}stats.npz",
    }


def test_r2_to_s3_uri_rejects_non_r2_scheme() -> None:
    """A URI without the ``r2://`` scheme raises ValueError rather than silently passing through.

    Guards against a future reference source handing ``build_dataset_artifact``
    a non-R2 URI, which would otherwise log a malformed lineage reference.
    """
    with pytest.raises(ValueError, match="r2://"):
        _r2_to_s3_uri("s3://bucket/already-s3.h5")
