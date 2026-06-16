"""Writer-level determinism from per-sample seeding (#884).

Drives the real ``make_hdf5_dataset`` (real sampler, real loudness path) with only
the VST3 binary faked, and pins the headline guarantee: a row's content is a pure
function of ``(base_seed, row_index)`` — identical across runs and independent of
shard size, hence of worker/order/sharding.
"""

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import pytest

from synth_setter.cli.generate_dataset import build_generate_args
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst.seeding import rng_for_sample
from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.data.vst.writers import make_hdf5_dataset
from tests.data.vst._fake_plugin import FakeVST3Plugin
from tests.data.vst.test_fake_plugin_e2e import _fake_render_cfg
from tests.data.vst.test_generate_vst_dataset import _SPEC_NAME
from tests.helpers.finalize_shards import build_hdf5_smoke_spec

_BASE_SEED = 20260615


def _render_param_array(out: Path, *, base_seed: int, num_samples: int) -> np.ndarray:
    # min_loudness=-inf accepts attempt 0 unconditionally, so param_array reflects
    # the seeded draw alone (independent of whatever audio the fake plugin emits).
    cfg = _fake_render_cfg(num_samples=num_samples, min_loudness=float("-inf")).model_copy(
        update={"base_seed": base_seed}
    )
    make_hdf5_dataset(hdf5_file=out, render_cfg=cfg)
    with h5py.File(out, "r") as f:
        param_array = f[PARAM_ARRAY_FIELD]
        assert isinstance(param_array, h5py.Dataset)
        return param_array[...]


@pytest.mark.fake_vst
def test_same_base_seed_yields_identical_param_arrays(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    a = _render_param_array(tmp_path / "a.h5", base_seed=_BASE_SEED, num_samples=4)
    b = _render_param_array(tmp_path / "b.h5", base_seed=_BASE_SEED, num_samples=4)
    assert np.array_equal(a, b)


@pytest.mark.fake_vst
def test_different_base_seed_yields_different_param_arrays(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    a = _render_param_array(tmp_path / "a.h5", base_seed=_BASE_SEED, num_samples=4)
    b = _render_param_array(tmp_path / "b.h5", base_seed=_BASE_SEED + 1, num_samples=4)
    assert not np.array_equal(a, b)


@pytest.mark.fake_vst
def test_prefix_rows_independent_of_shard_size(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    # Sample i is the same whether the shard holds 2 rows or 4 — a direct proxy
    # for sharding/order independence.
    small = _render_param_array(tmp_path / "small.h5", base_seed=_BASE_SEED, num_samples=2)
    large = _render_param_array(tmp_path / "large.h5", base_seed=_BASE_SEED, num_samples=4)
    assert np.array_equal(small, large[:2])


@pytest.mark.fake_vst
def test_row_params_are_pure_function_of_seed_and_index(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    num_samples = 4
    got = _render_param_array(tmp_path / "a.h5", base_seed=_BASE_SEED, num_samples=num_samples)
    spec = param_specs[_SPEC_NAME]
    for i in range(num_samples):
        synth, note = spec.sample(rng_for_sample(_BASE_SEED, i, 0))
        assert np.array_equal(got[i], spec.encode(synth, note))


@pytest.mark.fake_vst
def test_distinct_shard_seeds_render_distinct_reproducible_rows(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    # End-to-end backing for the launcher's per-shard seed assignment: distinct
    # shard seeds drive distinct yet individually-reproducible renders (#884).
    spec = build_hdf5_smoke_spec(train_val_test_sizes=(2, 0, 0), samples_per_shard=1)
    seeds = [
        int(args[args.index("--base_seed") + 1])
        for args in (build_generate_args(spec, shard, tmp_path) for shard in spec.shards)
    ]
    assert seeds[0] != seeds[1]

    shard0 = _render_param_array(tmp_path / "s0.h5", base_seed=seeds[0], num_samples=2)
    shard1 = _render_param_array(tmp_path / "s1.h5", base_seed=seeds[1], num_samples=2)
    assert not np.array_equal(shard0, shard1)

    shard0_rerun = _render_param_array(tmp_path / "s0b.h5", base_seed=seeds[0], num_samples=2)
    assert np.array_equal(shard0, shard0_rerun)


@pytest.mark.fake_vst
def test_rendered_shard_has_finite_audio_mel_and_param_datasets(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    # A schema/finiteness guard so a missing-field or NaN regression fails here
    # self-documentingly, not as a downstream shape error.
    out = tmp_path / "shard.h5"
    _render_param_array(out, base_seed=_BASE_SEED, num_samples=2)
    with h5py.File(out, "r") as f:
        for field in (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD):
            dataset = f[field]
            assert isinstance(dataset, h5py.Dataset), f"missing dataset {field}"
            assert np.isfinite(dataset[...]).all(), f"non-finite values in {field}"
