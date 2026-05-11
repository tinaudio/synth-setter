"""Round-trip every dataset experiment YAML through DatasetSpec validation.

Each datagen experiment under ``configs/experiment/*.yaml`` (excluding the
train-side experiments nested under ``kosc/``, ``ksin/``, etc.) composes via
Hydra's ``initialize_config_dir`` + ``compose``. The composed dict — minus
the ``data:``, ``r2:``, and ``hydra:`` group sub-trees that are lifted to
top-level via interpolation — must validate as ``DatasetSpec`` and JSON
round-trip without drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pipeline.schemas.spec import DatasetSpec

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

# Datagen experiments — flat at the top of configs/experiment/. Excludes the
# train-side experiment subdirectories (kosc/, ksin/, surge/, etc.) which use
# train.yaml's composition.
DATASET_EXPERIMENTS: tuple[str, ...] = (
    "10-1k-shards",
    "ci-materialize-test",
    "runpod-smoke-shard",
    "surge-simple-480k-10k",
)


def _compose_dataset_spec(experiment: str) -> DatasetSpec:
    """Compose ``configs/dataset.yaml`` with the named experiment override."""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="dataset",
            overrides=[f"experiment={experiment}", "+dataset_root=/dev/null"],
        )
    raw: Any = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict), f"composed config is not a mapping: {type(raw).__name__}"
    raw.pop("data", None)
    raw.pop("r2", None)
    raw.pop("hydra", None)
    raw.pop("dataset_root", None)
    return DatasetSpec(**raw)


@pytest.mark.parametrize("experiment", DATASET_EXPERIMENTS)
def test_experiment_yaml_validates_as_dataset_spec(experiment: str) -> None:
    """Each composed experiment validates as DatasetSpec."""
    spec = _compose_dataset_spec(experiment)
    assert spec.task_name == experiment
    assert spec.num_shards >= 1
    assert spec.num_params > 0


@pytest.mark.parametrize("experiment", DATASET_EXPERIMENTS)
def test_experiment_yaml_json_round_trips(experiment: str) -> None:
    """``model_dump_json`` → ``model_validate_json`` yields an equal model."""
    spec = _compose_dataset_spec(experiment)
    restored = DatasetSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec
