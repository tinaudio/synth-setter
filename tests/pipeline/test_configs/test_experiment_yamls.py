"""Round-trip an explicit allowlist of dataset experiment YAMLs through DatasetSpec.

Each entry in :data:`DATASET_EXPERIMENTS` names a top-level datagen experiment under
``configs/experiment/`` that composes ``dataset.yaml`` via Hydra's
``initialize_config_dir`` + ``compose``. The composed dict — minus the ``data:``,
``r2:``, and ``hydra:`` group sub-trees that are lifted to top-level via interpolation
— must validate as ``DatasetSpec`` and JSON round-trip without drift.

The list is curated rather than auto-discovered: ``configs/experiment/`` also holds
train-side configs (top-level files like ``time_weighting.yaml`` and the nested
``kosc/``, ``ksin/``, ``surge/``, … subdirectories) that compose ``train.yaml``, not
``dataset.yaml``, and would not validate as ``DatasetSpec``. Add a new entry here when
landing a new datagen experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pipeline.schemas.spec import DatasetSpec

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

# Curated list of datagen experiments (those that compose dataset.yaml). See the
# module docstring for why this is an allowlist rather than a directory scan.
DATASET_EXPERIMENTS: tuple[str, ...] = (
    "10-1k-shards",
    "ci-materialize-test",
    "runpod-smoke-shard",
    "surge-simple-480k-10k",
)


def _compose_dataset_spec(experiment: str) -> DatasetSpec:
    """Compose ``configs/dataset.yaml`` with the named experiment override."""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="dataset", overrides=[f"experiment={experiment}"])
    # ``configs/paths/default.yaml`` interpolates ``${oc.env:PROJECT_ROOT}`` and
    # ``${hydra:runtime.output_dir}``; the latter is only set under @hydra.main,
    # not bare ``compose()``. Pin both so ``resolve=True`` doesn't trip in unit
    # tests (mirrors the train/eval conftest pattern in ``tests/conftest.py``).
    cfg.paths.root_dir = str(CONFIG_DIR.parent)
    cfg.paths.output_dir = str(CONFIG_DIR.parent)
    cfg.paths.work_dir = str(CONFIG_DIR.parent)
    raw: Any = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict), f"composed config is not a mapping: {type(raw).__name__}"
    raw.pop("data", None)
    raw.pop("r2", None)
    raw.pop("hydra", None)
    raw.pop("paths", None)
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
