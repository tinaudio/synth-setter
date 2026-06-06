"""Unit tests for ``train._consumed_artifact_refs``.

The pure helper maps the opt-in ``consumed_*`` cfg fields to the
``(name, alias)`` lineage edges training feeds to ``use_input_artifacts``
(``storage-provenance-spec.md`` §5). Tested in isolation from the full
``train`` entrypoint per ``tests/_meta/test_entrypoint_test_modules.py``.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from synth_setter.cli.train import _consumed_artifact_refs


def test_consumed_artifact_refs_dataset_id_set_returns_data_edge() -> None:
    """A set dataset config_id yields one ``data-{id}`` edge at the alias."""
    cfg = OmegaConf.create(
        {"consumed_dataset_config_id": "diva-v1", "consumed_artifact_alias": "latest"}
    )

    assert _consumed_artifact_refs(cfg) == [("data-diva-v1", "latest")]


def test_consumed_artifact_refs_dataset_id_null_returns_empty() -> None:
    """A null dataset config_id yields no edges — the opt-out no-op path."""
    cfg = OmegaConf.create(
        {"consumed_dataset_config_id": None, "consumed_artifact_alias": "latest"}
    )

    assert _consumed_artifact_refs(cfg) == []


def test_consumed_artifact_refs_alias_override_is_honored() -> None:
    """A non-default alias flows into the edge verbatim."""
    cfg = OmegaConf.create(
        {"consumed_dataset_config_id": "diva-v1", "consumed_artifact_alias": "v3"}
    )

    assert _consumed_artifact_refs(cfg) == [("data-diva-v1", "v3")]
