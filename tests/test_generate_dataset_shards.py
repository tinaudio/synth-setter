"""Tests for the ``synth-setter-generate-dataset`` CLI entrypoint."""

from __future__ import annotations

from omegaconf import DictConfig

from synth_setter.cli.generate_dataset import spec_from_cfg
from synth_setter.pipeline.schemas.spec import DatasetSpec


def test_cfg_dataset_composes_and_validates_as_dataset_spec(
    cfg_dataset: DictConfig,
) -> None:
    """The new fixture composes ``dataset.yaml`` and round-trips through ``DatasetSpec``.

    :param cfg_dataset: Function-scoped fixture composing ``dataset.yaml`` with the
        ``generate_dataset/smoke-shard`` experiment and ``tmp_path``-pinned paths.
    """
    spec = spec_from_cfg(cfg_dataset)
    assert isinstance(spec, DatasetSpec)
    assert spec.num_shards >= 1
    assert spec.render.samples_per_shard >= 1
