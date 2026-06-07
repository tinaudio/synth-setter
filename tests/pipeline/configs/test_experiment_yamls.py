"""Round-trip an explicit allowlist of dataset experiment YAMLs through DatasetSpec.

Each entry in :data:`DATASET_EXPERIMENTS` names a datagen experiment under
``configs/experiment/generate_dataset/`` that composes ``dataset.yaml`` via Hydra's
``initialize_config_module`` + ``compose``. The composed dict — minus the non-``DatasetSpec``
group sub-trees (``data:``, ``r2:``, ``paths:``, and ``hydra:``) that are either lifted
to top-level via interpolation or only exist for Hydra runtime — must validate as
``DatasetSpec`` and JSON round-trip without drift.

The list is curated rather than auto-discovered: ``configs/experiment/`` also holds
train-side configs (top-level files like ``time_weighting.yaml`` and the nested
``kosc/``, ``ksin/``, ``surge/``, … subdirectories) that compose ``train.yaml``, not
``dataset.yaml``, and would not validate as ``DatasetSpec``. Add a new entry here when
landing a new datagen experiment under ``configs/experiment/generate_dataset/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module

from synth_setter.cli.generate_dataset import spec_from_cfg
from synth_setter.pipeline.schemas.spec import DatasetSpec

# Local checkout root — the test pins ``cfg.paths.*`` to this anchor so the
# composed ``${oc.env:PROJECT_ROOT}`` / ``${hydra:runtime.output_dir}``
# interpolations resolve to a real on-disk path during unit tests.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Curated allowlist — see module docstring for why. ``smoke-shard-with-*`` configs
# inherit ``smoke-shard``'s task_name via ``@_global_`` defaults chaining.
DATASET_EXPERIMENTS: dict[str, str] = {
    "generate_dataset/10-1k-shards": "10-1k-shards",
    "generate_dataset/ci-materialize-test": "ci-materialize-test",
    "generate_dataset/ci-materialize-test-wds": "ci-materialize-test-wds",
    "generate_dataset/nightly-parallel-smoke": "nightly-parallel-smoke",
    "generate_dataset/smoke-shard": "smoke-shard",
    "generate_dataset/smoke-shard-wds": "smoke-shard-wds",
    "generate_dataset/surge-simple-480k-10k": "surge-simple-480k-10k",
    "generate_dataset/smoke-shard-with-finalize": "smoke-shard",
    "generate_dataset/smoke-shard-with-oracle-eval": "smoke-shard",
    "generate_dataset/reference-surge-xt-paired": "ref-surge-xt-489",
    "generate_dataset/copy-paired-surge-xt": "copy-paired-surge-xt",
}


def _compose_dataset_spec(experiment: str) -> DatasetSpec:
    """Compose ``configs/dataset.yaml`` with the named experiment override."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="dataset", overrides=[f"experiment={experiment}"])
    # ``configs/paths/default.yaml`` interpolates ``${oc.env:PROJECT_ROOT}`` and
    # ``${hydra:runtime.output_dir}``; the latter is only set under @hydra.main,
    # not bare ``compose()``. Pin both so ``resolve=True`` doesn't trip in unit
    # tests (mirrors the train/eval conftest pattern in ``tests/conftest.py``).
    cfg.paths.root_dir = str(REPO_ROOT)
    cfg.paths.output_dir = str(REPO_ROOT)
    cfg.paths.work_dir = str(REPO_ROOT)
    return spec_from_cfg(cfg)


@pytest.mark.parametrize(("experiment", "expected_task_name"), DATASET_EXPERIMENTS.items())
def test_experiment_yaml_validates_as_dataset_spec(
    experiment: str, expected_task_name: str
) -> None:
    """Each composed experiment validates as DatasetSpec with its expected task_name.

    :param experiment: Hydra experiment id under ``configs/experiment/generate_dataset/``.
    :param expected_task_name: ``task_name`` the experiment composes to — the file
        stem for standalone configs, or the inherited ``smoke-shard`` for the
        ``smoke-shard-with-*`` configs that chain it via ``@_global_`` defaults.
    """
    spec = _compose_dataset_spec(experiment)
    assert spec.task_name == expected_task_name
    assert spec.num_shards >= 1
    assert spec.num_params > 0


@pytest.mark.parametrize("experiment", DATASET_EXPERIMENTS)
def test_experiment_yaml_json_round_trips(experiment: str) -> None:
    """``model_dump_json`` → ``model_validate_json`` yields an equal model."""
    spec = _compose_dataset_spec(experiment)
    restored = DatasetSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


def test_reference_surge_xt_paired_pins_deterministic_uri_fields() -> None:
    """The paired-copy reference pins the fixed task_name + run_id that make its URI stable.

    The #489 copy sweeps pin that URI; drift on task_name/run_id moves the dataset out from under
    them. param_spec_name / samples_per_shard / train_val_test_sizes are the copy-preflight match
    set, checked end-to-end by ``test_copy_base_matches_reference_for_copy_preflight``.
    """
    spec = _compose_dataset_spec("generate_dataset/reference-surge-xt-paired")
    assert spec.task_name == "ref-surge-xt-489"
    assert spec.run_id == "paired-ref-v1"
    assert spec.render.param_spec_name == "surge_xt"
    assert spec.render.samples_per_shard == 20
    assert spec.train_val_test_sizes == (40, 40, 40)


def test_copy_base_matches_reference_for_copy_preflight() -> None:
    """The copy base passes ``validate_copy_source`` against the reference — the real preflight.

    A paired copy replays the reference's patches only if this check accepts the pair, so drift on
    the match set (param_spec_name / samples_per_shard / train_val_test_sizes / output_format) in
    either config raises here exactly as it would abort a real copy run. Also pins the gate-off
    sentinel so a tidy-up back to the inherited default can't silently re-arm the loudness gate.
    """
    reference = _compose_dataset_spec("generate_dataset/reference-surge-xt-paired")
    copy = _compose_dataset_spec("generate_dataset/copy-paired-surge-xt")
    copy.validate_copy_source(reference)  # raises ValueError on any match-set mismatch
    assert copy.render.min_loudness == -1000.0
