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
from synth_setter.pipeline.partitioning import get_my_shards
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.data.vst._fake_plugin import FakeVST3Plugin
from tests.data.vst.test_fake_plugin_e2e import _fake_render_cfg
from tests.data.vst.test_generate_vst_dataset import _SPEC_NAME
from tests.helpers.finalize_shards import build_hdf5_smoke_spec

_BASE_SEED = 20260615


def _read_param_array(out: Path) -> np.ndarray:
    """Read a rendered shard's encoded parameter rows.

    :param out: Rendered HDF5 shard path.
    :returns: ``param_array`` rows from the shard.
    """
    with h5py.File(out, "r") as f:
        param_array = f[PARAM_ARRAY_FIELD]
        assert isinstance(param_array, h5py.Dataset)
        return param_array[...]


def _read_int_attr(attrs: h5py.AttributeManager, key: str) -> int:
    """Return an integer HDF5 attr, failing if the stored value is not scalar.

    :param attrs: HDF5 attribute manager to read from.
    :param key: Attribute key.
    :returns: Scalar integer attribute value.
    """
    value = attrs[key]
    assert isinstance(value, int | np.integer)
    return int(value)


def _read_seed_metadata(out: Path) -> dict[str, int]:
    """Read seed provenance attrs from a rendered HDF5 shard.

    :param out: Rendered HDF5 shard path.
    :returns: Seed provenance attrs relevant to repeat-run determinism.
    """
    with h5py.File(out, "r") as f:
        audio = f[AUDIO_FIELD]
        assert isinstance(audio, h5py.Dataset)
        return {
            "base_seed": _read_int_attr(audio.attrs, "base_seed"),
            "attempts_per_sample": _read_int_attr(audio.attrs, "attempts_per_sample"),
        }


def _render_param_array(out: Path, *, base_seed: int, num_samples: int) -> np.ndarray:
    """Render a fake-plugin HDF5 shard and return its parameter array.

    :param out: Destination HDF5 path.
    :param base_seed: Master seed passed into the renderer.
    :param num_samples: Number of rows to render.
    :returns: Rendered ``param_array`` dataset.
    """
    # min_loudness=-inf accepts attempt 0 unconditionally, so param_array reflects
    # the seeded draw alone (independent of whatever audio the fake plugin emits).
    cfg = _fake_render_cfg(num_samples=num_samples, min_loudness=float("-inf")).model_copy(
        update={"base_seed": base_seed}
    )
    make_hdf5_dataset(hdf5_file=out, render_cfg=cfg)
    return _read_param_array(out)


def _render_worker_layout(
    out_dir: Path, spec: DatasetSpec, *, world: int
) -> dict[int, dict[str, np.ndarray]]:
    """Render shards owned by ``world`` workers and return rows keyed by global index.

    :param out_dir: Directory receiving the worker-owned HDF5 shards.
    :param spec: Dataset spec whose shard seeds and row counts define the render.
    :param world: Simulated worker count.
    :returns: Per-global-index arrays for every dataset field.
    """
    rows: dict[int, dict[str, np.ndarray]] = {}
    fields = (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD)
    for rank in range(world):
        for shard_id in get_my_shards(spec.num_shards, rank=rank, world=world):
            shard = spec.shards[shard_id]
            shard_path = out_dir / f"worker-{rank}" / shard.filename
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            render_cfg = _fake_render_cfg(
                num_samples=spec.render.samples_per_shard,
                min_loudness=float("-inf"),
            ).model_copy(update={"base_seed": shard.seed})
            make_hdf5_dataset(hdf5_file=shard_path, render_cfg=render_cfg)
            with h5py.File(shard_path, "r") as f:
                for local_idx in range(spec.render.samples_per_shard):
                    global_idx = shard_id * spec.render.samples_per_shard + local_idx
                    rows[global_idx] = {}
                    for field in fields:
                        dataset = f[field]
                        assert isinstance(dataset, h5py.Dataset), f"missing dataset {field}"
                        rows[global_idx][field] = dataset[local_idx]
    return rows


@pytest.mark.fake_vst
def test_same_base_seed_yields_identical_param_arrays(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Same seed renders byte-identical parameter rows across two runs.

    :param tmp_path: Pytest fixture providing destination paths.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    a = _render_param_array(tmp_path / "a.h5", base_seed=_BASE_SEED, num_samples=4)
    b = _render_param_array(tmp_path / "b.h5", base_seed=_BASE_SEED, num_samples=4)
    assert np.array_equal(a, b)


@pytest.mark.fake_vst
def test_same_render_config_yields_identical_params_and_seed_metadata(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Same experiment renders identical params and seed provenance twice.

    :param tmp_path: Pytest fixture providing destination paths.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    cfg = _fake_render_cfg(num_samples=4, min_loudness=float("-inf")).model_copy(
        update={"base_seed": _BASE_SEED, "attempts_per_sample": 3}
    )

    make_hdf5_dataset(hdf5_file=first, render_cfg=cfg)
    make_hdf5_dataset(hdf5_file=second, render_cfg=cfg)

    assert np.array_equal(_read_param_array(first), _read_param_array(second))
    assert _read_seed_metadata(first) == _read_seed_metadata(second) == {
        "base_seed": _BASE_SEED,
        "attempts_per_sample": 3,
    }


@pytest.mark.fake_vst
def test_different_base_seed_yields_different_param_arrays(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Different seeds produce different parameter rows.

    :param tmp_path: Pytest fixture providing destination paths.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    a = _render_param_array(tmp_path / "a.h5", base_seed=_BASE_SEED, num_samples=4)
    b = _render_param_array(tmp_path / "b.h5", base_seed=_BASE_SEED + 1, num_samples=4)
    assert not np.array_equal(a, b)


@pytest.mark.fake_vst
def test_prefix_rows_independent_of_shard_size(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Rows shared by two shard sizes render identically.

    :param tmp_path: Pytest fixture providing destination paths.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    # Sample i is the same whether the shard holds 2 rows or 4 — a direct proxy
    # for sharding/order independence.
    small = _render_param_array(tmp_path / "small.h5", base_seed=_BASE_SEED, num_samples=2)
    large = _render_param_array(tmp_path / "large.h5", base_seed=_BASE_SEED, num_samples=4)
    assert np.array_equal(small, large[:2])


@pytest.mark.fake_vst
def test_per_index_content_identical_across_worker_counts(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """One-worker and two-worker shard ownership produce byte-identical rows.

    :param tmp_path: Pytest fixture providing output directories for each worker layout.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    spec = build_hdf5_smoke_spec(train_val_test_sizes=(8, 0, 0), samples_per_shard=2)

    single_worker = _render_worker_layout(tmp_path / "world-1", spec, world=1)
    two_workers = _render_worker_layout(tmp_path / "world-2", spec, world=2)

    assert single_worker.keys() == two_workers.keys()
    for global_idx, fields in single_worker.items():
        for field, expected in fields.items():
            assert np.array_equal(two_workers[global_idx][field], expected), (
                f"{field} row {global_idx} differed across worker counts"
            )


@pytest.mark.fake_vst
def test_row_params_are_pure_function_of_seed_and_index(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Each row matches the direct ``rng_for_sample(seed, index, 0)`` draw.

    :param tmp_path: Pytest fixture providing destination paths.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
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
    """Launcher-derived shard seeds are distinct and reproducible.

    :param tmp_path: Pytest fixture providing destination paths.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
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
    """Rendered fake-plugin shards contain finite values for every dataset field.

    :param tmp_path: Pytest fixture providing destination paths.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    # A schema/finiteness guard so a missing-field or NaN regression fails here
    # self-documentingly, not as a downstream shape error.
    out = tmp_path / "shard.h5"
    _render_param_array(out, base_seed=_BASE_SEED, num_samples=2)
    with h5py.File(out, "r") as f:
        for field in (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD):
            dataset = f[field]
            assert isinstance(dataset, h5py.Dataset), f"missing dataset {field}"
            assert np.isfinite(dataset[...]).all(), f"non-finite values in {field}"
