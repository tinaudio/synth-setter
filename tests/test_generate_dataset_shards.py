"""Tests for the ``synth-setter-generate-dataset`` CLI entrypoint.

Covers two shapes: a cheap Hydra-compose round-trip through ``DatasetSpec``, and
a VST-gated end-to-end render that drives ``from_hydra`` against ``cfg_dataset``
and asserts every shard lands at the spec-derived R2 URI in the fake-local
rclone remote.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from synth_setter.cli.generate_dataset import from_hydra, spec_from_cfg
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.conftest import _validate_surge_dataset


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


@pytest.mark.slow
@pytest.mark.requires_vst
def test_generate_dataset_renders_shards_to_fake_r2(
    cfg_dataset: DictConfig,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_hydra`` renders every shard in ``spec.shards`` and uploads to fake R2.

    :param cfg_dataset: Hydra DictConfig composed with the
        ``generate_dataset/smoke-shard`` experiment (paths pinned to ``tmp_path``).
    :param fake_r2_remote: Local-typed rclone remote rooted at ``tmp_path``; a
        URI ``r2://<bucket>/<key>`` materializes at ``<root>/<bucket>/<key>``.
    :param monkeypatch: Used to pin the single-worker rank/world env contract.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    from_hydra(cfg_dataset)

    spec = spec_from_cfg(cfg_dataset)
    landed_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    for shard in spec.shards:
        shard_path = landed_root / shard.filename
        assert shard_path.is_file(), f"shard missing: {shard_path}"
        _validate_surge_dataset(shard_path, num_samples=spec.render.samples_per_shard)
