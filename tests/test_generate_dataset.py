"""Tests for the ``synth-setter-generate-dataset`` CLI entrypoint.

Covers two shapes: a cheap Hydra-compose round-trip through ``DatasetSpec``,
and an ``integration_r2``-gated end-to-end render that drives ``from_hydra``
against ``cfg_dataset`` and asserts every shard lands at the spec-derived R2
URI in real Cloudflare R2 (auto-skips when ``rclone`` / R2 creds are absent).

Keep this module to cfg-entrypoint tests; direct-call unit tests for
``generate`` / ``main`` and the arg-builders live in
``tests/pipeline/entrypoints/test_generate_dataset_unit.py``.
``tests/_meta/test_entrypoint_test_modules.py`` enforces that no private
``synth_setter.cli`` helper is imported here.
"""

from __future__ import annotations

import uuid

import h5py
import numpy as np
import pytest
from omegaconf import DictConfig, open_dict

from synth_setter.cli.generate_dataset import from_hydra, spec_from_cfg
from synth_setter.pipeline import r2_io
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


@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.requires_vst
@pytest.mark.slow
def test_generate_dataset_renders_shards_to_r2(
    cfg_dataset: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_hydra`` renders every shard in ``spec.shards`` and uploads to real R2.

    The unique-per-run ``r2.prefix`` override keeps concurrent runs isolated; a
    best-effort ``rclone purge`` in ``finally`` removes the prefix even on test
    failure so we don't leak shards. Auto-skips when ``rclone`` is missing or
    ``rclone lsd r2:`` fails (contributor laptops, fork PRs without secrets).

    :param cfg_dataset: Hydra DictConfig composed with the
        ``generate_dataset/smoke-shard`` experiment.
    :param monkeypatch: Pins the single-worker rank/world env contract.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or `rclone lsd r2:` failed)")

    unique_prefix = (
        f"test-runs/test_generate_dataset_renders_shards_to_r2/{uuid.uuid4().hex[:12]}/"
    )
    with open_dict(cfg_dataset):
        cfg_dataset.r2.prefix = unique_prefix

    spec = spec_from_cfg(cfg_dataset)
    try:
        from_hydra(cfg_dataset)
        for shard in spec.shards:
            size = r2_io.object_size(spec.r2.shard_uri(shard))
            assert size is not None and size > 0, f"shard missing in R2: {shard.filename}"
    finally:
        r2_io.purge_prefix(spec.r2.bucket, spec.r2.prefix)


@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.requires_vst
@pytest.mark.slow
def test_generate_dataset_shard_cadence_renders_one_identical_patch_per_shard(
    cfg_dataset: DictConfig,
) -> None:
    """``render.param_sample_cadence="shard"`` makes every sample in a shard share one patch.

    Drives the real ``generate_dataset`` entrypoint (``from_hydra``) end-to-end
    under shard cadence, then downloads each shard and asserts its ``param_array``
    rows are all identical — the one-patch-per-shard invariant the #489 variance
    probe relies on. Auto-skips without R2; purges the unique prefix in
    ``finally`` so a failure can't leak shards.

    :param cfg_dataset: Hydra DictConfig composed with the
        ``generate_dataset/smoke-shard`` experiment.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or `rclone lsd r2:` failed)")

    unique_prefix = f"test-runs/test_generate_dataset_shard_cadence/{uuid.uuid4().hex[:12]}/"
    with open_dict(cfg_dataset):
        cfg_dataset.r2.prefix = unique_prefix
        cfg_dataset.render.param_sample_cadence = "shard"

    spec = spec_from_cfg(cfg_dataset)
    assert spec.render.param_sample_cadence == "shard"
    try:
        from_hydra(cfg_dataset)
        for shard in spec.shards:
            with r2_io.downloaded_to_tempfile(spec.r2.shard_uri(shard)) as local:
                with h5py.File(local, "r") as f:
                    param_dataset = f["param_array"]
                    assert isinstance(param_dataset, h5py.Dataset)
                    params = param_dataset[:]
            assert params.shape[0] == spec.render.samples_per_shard
            assert np.array_equal(params, np.broadcast_to(params[0], params.shape)), (
                f"shard {shard.filename} has non-identical param rows under shard cadence"
            )
    finally:
        r2_io.purge_prefix(spec.r2.bucket, spec.r2.prefix)
