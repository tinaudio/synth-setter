"""Tests for the ``synth-setter-generate-dataset`` CLI entrypoint.

Covers two shapes: a cheap Hydra-compose round-trip through ``DatasetSpec``, and
a VST-gated end-to-end render that drives ``from_hydra`` against ``cfg_dataset``
and asserts every shard lands at the spec-derived R2 URI in the fake-local
rclone remote.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from omegaconf import DictConfig

from synth_setter.cli.generate_dataset import from_hydra, spec_from_cfg
from synth_setter.pipeline.schemas.spec import DatasetSpec

_PLUGIN_PATH = Path(os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3")


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
@pytest.mark.skipif(
    not _PLUGIN_PATH.exists(),
    reason=f"VST plugin not found at {_PLUGIN_PATH}",
)
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

    # generate_dataset.VST_HEADLESS_WRAPPER and the rendered subprocess args are
    # repo-relative; fake_r2_remote chdirs into tmp_path, so re-anchor CWD at the
    # project root before from_hydra shells out.
    monkeypatch.chdir(cfg_dataset.paths.root_dir)

    from_hydra(cfg_dataset)

    spec = spec_from_cfg(cfg_dataset)
    expected_clip_samples = int(spec.render.sample_rate * spec.render.signal_duration_seconds)
    expected_audio_shape = (
        spec.render.samples_per_shard,
        spec.render.channels,
        expected_clip_samples,
    )
    landed_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    for shard in spec.shards:
        shard_path = landed_root / shard.filename
        assert shard_path.is_file(), f"shard missing: {shard_path}"
        with h5py.File(shard_path, "r") as f:
            audio = f["audio"]
            params = f["param_array"]
            assert isinstance(audio, h5py.Dataset), f"'audio' not a Dataset in {shard_path}"
            assert isinstance(params, h5py.Dataset), f"'param_array' not a Dataset in {shard_path}"
            assert audio.shape == expected_audio_shape, (
                f"audio shape {audio.shape} != expected {expected_audio_shape}"
            )
            assert params.shape == (spec.render.samples_per_shard, spec.num_params), (
                f"param_array shape {params.shape} != "
                f"({spec.render.samples_per_shard}, {spec.num_params})"
            )
            assert np.isfinite(audio[...]).all(), f"audio in {shard_path} contains NaN/Inf"
