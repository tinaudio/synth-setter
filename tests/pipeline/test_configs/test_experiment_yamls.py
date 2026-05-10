"""Round-trip every datagen experiment YAML through DatasetSpec validation.

Each experiment under ``configs/experiment/datagen/*.yaml`` (enumerated in
``DATASET_EXPERIMENTS`` below — train-side experiment YAMLs at the top of
``configs/experiment/`` and under ``kosc/``, ``ksin/``, etc. are out of scope
here and tested separately) composes via the production
``compose_dataset_spec`` helper, must validate as ``DatasetSpec``, and must
JSON-round-trip without drift.
"""

from __future__ import annotations

import pytest

from src.generate_dataset import compose_dataset_spec
from src.pipeline.schemas.spec import DatasetSpec

# Datagen experiments live under configs/experiment/datagen/ to keep them
# separate from the training experiments at the top of configs/experiment/
# and under its task subdirectories (kosc/, ksin/, surge/, etc.).
DATASET_EXPERIMENTS: tuple[str, ...] = (
    "datagen/10-1k-shards",
    "datagen/ci-materialize-test",
    "datagen/ci-materialize-test-wds",
    "datagen/runpod-smoke-shard",
    "datagen/runpod-smoke-shard-wds",
    "datagen/surge-simple-480k-10k",
)


@pytest.mark.parametrize("experiment", DATASET_EXPERIMENTS)
def test_experiment_yaml_validates_as_dataset_spec(experiment: str) -> None:
    """Each composed experiment validates as DatasetSpec."""
    spec = compose_dataset_spec(experiment)
    # task_name is the experiment file's stem, not the full Hydra group path.
    assert spec.task_name == experiment.removeprefix("datagen/")
    assert spec.num_shards >= 1
    assert spec.num_params > 0


@pytest.mark.parametrize("experiment", DATASET_EXPERIMENTS)
def test_experiment_yaml_json_round_trips(experiment: str) -> None:
    """model_dump_json → model_validate_json yields an equal model."""
    spec = compose_dataset_spec(experiment)
    restored = DatasetSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


@pytest.mark.parametrize(
    ("experiment", "expected_format"),
    [
        ("datagen/ci-materialize-test", "hdf5"),
        ("datagen/ci-materialize-test-wds", "wds"),
        ("datagen/runpod-smoke-shard", "hdf5"),
        ("datagen/runpod-smoke-shard-wds", "wds"),
    ],
)
def test_wds_variant_inherits_from_hdf5_base(experiment: str, expected_format: str) -> None:
    """The wds variants inherit layout from their hdf5 siblings via Hydra defaults."""
    spec = compose_dataset_spec(experiment)
    assert spec.output_format == expected_format
    expected_ext = ".tar" if expected_format == "wds" else ".h5"
    assert all(s.filename.endswith(expected_ext) for s in spec.shards)
