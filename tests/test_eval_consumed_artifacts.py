"""Unit tests for ``eval._consumed_artifact_refs``.

The pure helper maps the opt-in ``consumed_*`` cfg fields to the
``(name, alias)`` lineage edges eval feeds to ``use_input_artifacts`` — both
the model and the dataset it consumes (``storage-provenance-spec.md`` §5).
Tested in isolation from the ``evaluate`` entrypoint per
``tests/_meta/test_entrypoint_test_modules.py``.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from synth_setter.cli.eval import _consumed_artifact_refs


def test_consumed_artifact_refs_both_ids_set_returns_model_then_data_edges() -> None:
    """Set train + dataset ids yield the model edge first, then the dataset edge."""
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": "flow-simple",
            "consumed_dataset_config_id": "diva-v1",
            "consumed_artifact_alias": "latest",
        }
    )

    assert _consumed_artifact_refs(cfg) == [
        ("model-flow-simple", "latest"),
        ("data-diva-v1", "latest"),
    ]


def test_consumed_artifact_refs_only_model_id_set_returns_model_edge_only() -> None:
    """A set train id with a null dataset id yields the model edge alone."""
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": "flow-simple",
            "consumed_dataset_config_id": None,
            "consumed_artifact_alias": "latest",
        }
    )

    assert _consumed_artifact_refs(cfg) == [("model-flow-simple", "latest")]


def test_consumed_artifact_refs_only_dataset_id_set_returns_data_edge_only() -> None:
    """A set dataset id with a null train id yields the dataset edge alone."""
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": None,
            "consumed_dataset_config_id": "diva-v1",
            "consumed_artifact_alias": "latest",
        }
    )

    assert _consumed_artifact_refs(cfg) == [("data-diva-v1", "latest")]


def test_consumed_artifact_refs_both_ids_null_returns_empty() -> None:
    """Both ids null yields no edges — the opt-out no-op path."""
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": None,
            "consumed_dataset_config_id": None,
            "consumed_artifact_alias": "latest",
        }
    )

    assert _consumed_artifact_refs(cfg) == []
