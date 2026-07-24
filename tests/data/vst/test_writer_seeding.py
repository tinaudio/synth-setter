"""Writer-level determinism from per-sample seeding (#884).

Drives the real ``make_lance_dataset`` (real sampler, real loudness path) with only
the VST3 binary faked, and pins the headline guarantee: a row's content is a pure
function of ``(base_seed, row_index)`` — identical across runs and independent of
shard size, hence of worker/order/sharding.
"""

import json
import sys
from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.cli.generate_dataset import build_generate_args
from synth_setter.data.vst.generate_vst_dataset import SampleSeed, VSTDataSample, main
from synth_setter.data.vst.param_spec import NoteParams, ParamSpec
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst.renderers import AudioRenderer
from synth_setter.data.vst.seeding import rng_for_sample, seed_for_sample
from synth_setter.data.vst.shapes import AUDIO_FIELD, DEBUG_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.pipeline.data.lance_shard import SHARD_METADATA_SCHEMA_KEY
from synth_setter.pipeline.partitioning import get_my_shards
from synth_setter.pipeline.schemas.seed_debug import SeedDebugDocument
from synth_setter.pipeline.schemas.spec import DatasetSpec, Split
from tests.data.vst._fake_plugin import FakeVST3Plugin
from tests.data.vst.test_fake_plugin_e2e import _fake_render_cfg
from tests.data.vst.test_generate_vst_dataset import _SPEC_NAME
from tests.helpers.finalize_shards import _LANCE_SMOKE_RENDER, build_lance_smoke_spec

_BASE_SEED = 20260615


def _multishard_lance_spec(
    train_val_test_sizes: tuple[int, int, int],
    samples_per_shard: int,
    train_val_test_seeds: tuple[int, int, int] | None = None,
) -> DatasetSpec:
    """Build a lance ``DatasetSpec`` split into shards of ``samples_per_shard`` rows.

    The shared ``build_lance_smoke_spec`` fixes a single-shard render, so its
    render is overridden here to size shards for the worker-layout tests.

    :param train_val_test_sizes: Three-tuple of sample counts across splits.
    :param samples_per_shard: Per-shard row count driving shard-count derivation.
    :param train_val_test_seeds: Optional independent split master seeds.
    :returns: A frozen lance ``DatasetSpec`` whose shards are deterministic.
    """
    render = {
        **_LANCE_SMOKE_RENDER,
        "samples_per_shard": samples_per_shard,
        "samples_per_render_batch": samples_per_shard,
    }
    return build_lance_smoke_spec(
        train_val_test_sizes=train_val_test_sizes,
        render=render,
        train_val_test_seeds=train_val_test_seeds,
    )


def _read_column(out: Path, field: str) -> np.ndarray:
    """Read one fixed-shape tensor column from a rendered Lance shard.

    :param out: Rendered Lance shard path.
    :param field: Column name to read.
    :returns: The column stacked into a ``(num_rows, *shape)`` array.
    """
    chunk = lance.dataset(str(out)).to_table(columns=[field]).column(field).combine_chunks()
    return chunk.to_numpy_ndarray()


def _read_param_array(out: Path) -> np.ndarray:
    """Read a rendered shard's encoded parameter rows.

    :param out: Rendered Lance shard path.
    :returns: ``param_array`` rows from the shard.
    """
    return _read_column(out, PARAM_ARRAY_FIELD)


def _read_debug_rows(out: Path) -> list[SeedDebugDocument]:
    """Read and validate row-level seed provenance from a rendered Lance shard.

    :param out: Rendered Lance shard path.
    :returns: Typed debug documents in row order.
    """
    documents = lance.dataset(str(out)).to_table(columns=[DEBUG_FIELD]).column(0).to_pylist()
    return [SeedDebugDocument.model_validate_json(document) for document in documents]


def _read_seed_metadata(out: Path) -> dict[str, int]:
    """Read seed provenance from a rendered Lance shard's schema metadata.

    Raw JSON preserves seed positions when ``min_loudness=-inf`` serializes to
    ``null`` and cannot pass strict ``ShardMetadata`` validation.

    :param out: Rendered Lance shard path.
    :returns: Seed provenance attrs relevant to repeat-run determinism.
    """
    payload = lance.dataset(str(out)).schema.metadata[SHARD_METADATA_SCHEMA_KEY]
    data = json.loads(payload)
    return {
        "base_seed": data["base_seed"],
        "sample_offset": data["sample_offset"],
        "attempts_per_sample": data["attempts_per_sample"],
    }


def _render_param_array(out: Path, *, base_seed: int, num_samples: int) -> np.ndarray:
    """Render a fake-plugin Lance shard and return its parameter array.

    :param out: Destination Lance path.
    :param base_seed: Master seed passed into the renderer.
    :param num_samples: Number of rows to render.
    :returns: Rendered ``param_array`` column.
    """
    # min_loudness=-inf accepts attempt 0 unconditionally, so param_array reflects
    # the seeded draw alone (independent of whatever audio the fake plugin emits).
    cfg = _fake_render_cfg(num_samples=num_samples, min_loudness=float("-inf")).model_copy(
        update={"base_seed": base_seed}
    )
    make_lance_dataset(out, cfg)
    return _read_param_array(out)


def _render_split_params(out_dir: Path, spec: DatasetSpec, split: Split) -> np.ndarray:
    """Render and concatenate one split's parameter rows in logical order.

    :param out_dir: Directory receiving the split's Lance shards.
    :param spec: Dataset spec defining split-local seed streams.
    :param split: Split whose rows are rendered.
    :returns: Concatenated encoded parameter rows.
    """
    lo, hi = spec.split_shard_ranges[split]
    rows: list[np.ndarray] = []
    for shard in spec.shards[lo:hi]:
        shard_path = out_dir / shard.filename
        # Fake-plugin fields replace spec.render, so inject the shard's seed position directly.
        render_cfg = _fake_render_cfg(
            num_samples=spec.render.samples_per_shard,
            min_loudness=float("-inf"),
        ).model_copy(
            update={"base_seed": shard.seed, "sample_offset": shard.sample_offset}
        )
        make_lance_dataset(shard_path, render_cfg)
        rows.append(_read_param_array(shard_path))
    return np.concatenate(rows)


def _render_worker_layout(
    out_dir: Path, spec: DatasetSpec, *, world: int
) -> dict[int, dict[str, np.ndarray]]:
    """Render shards owned by ``world`` workers and return rows keyed by global index.

    :param out_dir: Directory receiving the worker-owned Lance shards.
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
            # Fake-plugin fields replace spec.render, so inject the shard's seed position directly.
            render_cfg = _fake_render_cfg(
                num_samples=spec.render.samples_per_shard,
                min_loudness=float("-inf"),
            ).model_copy(
                update={"base_seed": shard.seed, "sample_offset": shard.sample_offset}
            )
            make_lance_dataset(shard_path, render_cfg)
            columns = {field: _read_column(shard_path, field) for field in fields}
            for local_idx in range(spec.render.samples_per_shard):
                global_idx = shard_id * spec.render.samples_per_shard + local_idx
                rows[global_idx] = {field: columns[field][local_idx] for field in fields}
    return rows


@pytest.mark.fake_vst
def test_same_base_seed_yields_identical_param_arrays(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Same seed renders byte-identical parameter rows across two runs.

    :param tmp_path: Pytest fixture providing destination paths.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    a = _render_param_array(tmp_path / "a.lance", base_seed=_BASE_SEED, num_samples=4)
    b = _render_param_array(tmp_path / "b.lance", base_seed=_BASE_SEED, num_samples=4)
    assert np.array_equal(a, b)


@pytest.mark.fake_vst
def test_same_render_config_yields_identical_params_and_seed_metadata(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Same experiment renders identical params and seed provenance twice.

    :param tmp_path: Pytest fixture providing destination paths.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    first = tmp_path / "first.lance"
    second = tmp_path / "second.lance"
    cfg = _fake_render_cfg(num_samples=4, min_loudness=float("-inf")).model_copy(
        update={"base_seed": _BASE_SEED, "sample_offset": 12, "attempts_per_sample": 3}
    )

    make_lance_dataset(first, cfg)
    make_lance_dataset(second, cfg)

    assert np.array_equal(_read_param_array(first), _read_param_array(second))
    assert (
        _read_seed_metadata(first)
        == _read_seed_metadata(second)
        == {
            "base_seed": _BASE_SEED,
            "sample_offset": 12,
            "attempts_per_sample": 3,
        }
    )


@pytest.mark.fake_vst
def test_launcher_args_through_renderer_main_persist_shard_id(
    tmp_path: Path,
    install_fake_plugin: FakeVST3Plugin,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launcher argv survives CLI parsing and reaches persisted row provenance.

    :param tmp_path: Pytest fixture providing the destination path.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    :param monkeypatch: Pytest fixture used to install launcher-produced argv.
    """
    spec = _multishard_lance_spec((1, 0, 0), samples_per_shard=1)
    shard = spec.shards[0]
    args = build_generate_args(spec, shard, tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate_vst_dataset", *args[2:]])

    main()

    out = tmp_path / shard.filename
    assert _read_debug_rows(out)[0].shard_id == shard.shard_id


@pytest.mark.fake_vst
def test_writer_populates_debug_json_with_seed_and_inputs(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Rendered JSON documents expose the concrete seed and every derivation input.

    :param tmp_path: Pytest fixture providing the destination path.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    out = tmp_path / "debug.lance"
    cfg = _fake_render_cfg(num_samples=2, min_loudness=float("-inf")).model_copy(
        update={"base_seed": _BASE_SEED, "sample_offset": 12}
    )

    make_lance_dataset(out, cfg, shard_id=7)

    assert _read_debug_rows(out) == [
        SeedDebugDocument(
            seed=seed_for_sample(_BASE_SEED, 12, 0),
            master_seed=_BASE_SEED,
            sample_idx=12,
            attempt=0,
            shard_id=7,
            parameter_source="sampled",
        ),
        SeedDebugDocument(
            seed=seed_for_sample(_BASE_SEED, 13, 0),
            master_seed=_BASE_SEED,
            sample_idx=13,
            attempt=0,
            shard_id=7,
            parameter_source="sampled",
        ),
    ]


@pytest.mark.fake_vst
def test_writer_marks_external_fixed_parameter_provenance(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Fixed input rows do not claim a sampled parameter seed.

    :param tmp_path: Pytest fixture providing the destination path.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    out = tmp_path / "fixed.lance"
    cfg = _fake_render_cfg(num_samples=1, min_loudness=float("-inf"))
    synth_params, note_params = param_specs[_SPEC_NAME].sample(rng_for_sample(_BASE_SEED, 0))

    make_lance_dataset(
        out,
        cfg,
        shard_id=7,
        fixed_synth_params_list=[synth_params],
        fixed_note_params_list=[note_params],
    )

    row = _read_debug_rows(out)[0]
    assert row.parameter_source == "fixed"
    assert row.parameter_seed is None


@pytest.mark.fake_vst
def test_writer_marks_mixed_parameter_provenance(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """A fixed synth patch plus sampled note is identified as mixed provenance.

    :param tmp_path: Pytest fixture providing the destination path.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    out = tmp_path / "mixed.lance"
    cfg = _fake_render_cfg(num_samples=1, min_loudness=float("-inf"))
    synth_params, _ = param_specs[_SPEC_NAME].sample(rng_for_sample(_BASE_SEED, 0))

    make_lance_dataset(out, cfg, shard_id=7, fixed_synth_params_list=[synth_params])

    assert _read_debug_rows(out)[0].parameter_source == "mixed"


@pytest.mark.fake_vst
def test_writer_persists_shard_cadence_parameter_seed_across_batches(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Every shard-cadence row identifies the first row's reused patch seed.

    :param tmp_path: Pytest fixture providing the destination path.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    out = tmp_path / "shard-cadence.lance"
    cfg = _fake_render_cfg(num_samples=3, min_loudness=float("-inf")).model_copy(
        update={
            "base_seed": _BASE_SEED,
            "sample_offset": 12,
            "param_sample_cadence": "shard",
            "samples_per_render_batch": 1,
        }
    )

    make_lance_dataset(out, cfg, shard_id=7)

    rows = _read_debug_rows(out)
    for sample_idx, row in enumerate(rows, start=12):
        assert row.sample_idx == sample_idx
        assert row.parameter_seed == seed_for_sample(_BASE_SEED, 12, 0)
        assert row.parameter_sample_idx == 12
        assert row.parameter_attempt == 0
        assert row.parameter_source == "sampled"


@pytest.mark.fake_vst
def test_writer_persists_nonzero_accepted_attempt(
    tmp_path: Path,
    install_fake_plugin: FakeVST3Plugin,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writer debug uses the accepted attempt reported by sample generation.

    :param tmp_path: Pytest fixture providing the destination path.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    :param monkeypatch: Pytest fixture used to force a nonzero accepted attempt.
    """
    from synth_setter.data.vst import writers

    original_generate_sample = writers.generate_sample

    def _generate_after_retries(
        renderer: AudioRenderer,
        velocity: int,
        min_loudness: float,
        param_spec: ParamSpec,
        fixed_synth_params: dict[str, float] | None = None,
        fixed_note_params: NoteParams | None = None,
        *,
        warmup: bool = False,
        seed: SampleSeed | None = None,
    ) -> VSTDataSample:
        sample = original_generate_sample(
            renderer,
            velocity,
            min_loudness,
            param_spec,
            fixed_synth_params,
            fixed_note_params,
            warmup=warmup,
            seed=seed,
        )
        sample.attempt = 2
        return sample

    monkeypatch.setattr(writers, "generate_sample", _generate_after_retries)
    out = tmp_path / "retried.lance"
    cfg = _fake_render_cfg(num_samples=1, min_loudness=float("-inf")).model_copy(
        update={"base_seed": _BASE_SEED, "sample_offset": 12}
    )

    make_lance_dataset(out, cfg, shard_id=7)

    assert _read_debug_rows(out)[0] == SeedDebugDocument(
        seed=seed_for_sample(_BASE_SEED, 12, 2),
        master_seed=_BASE_SEED,
        sample_idx=12,
        attempt=2,
        shard_id=7,
        parameter_source="sampled",
    )


@pytest.mark.fake_vst
def test_different_base_seed_yields_different_param_arrays(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Different seeds produce different parameter rows.

    :param tmp_path: Pytest fixture providing destination paths.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    a = _render_param_array(tmp_path / "a.lance", base_seed=_BASE_SEED, num_samples=4)
    b = _render_param_array(tmp_path / "b.lance", base_seed=_BASE_SEED + 1, num_samples=4)
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
    small = _render_param_array(tmp_path / "small.lance", base_seed=_BASE_SEED, num_samples=2)
    large = _render_param_array(tmp_path / "large.lance", base_seed=_BASE_SEED, num_samples=4)
    assert np.array_equal(small, large[:2])


@pytest.mark.fake_vst
def test_split_rows_stable_across_train_and_shard_sizes(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """Train rows nest and held-out rows stay fixed across dataset scaling.

    :param tmp_path: Pytest fixture providing output directories for both layouts.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    split_seeds = (101, 202, 303)
    small = _multishard_lance_spec((4, 4, 4), 2, split_seeds)
    large = _multishard_lance_spec((8, 4, 4), 4, split_seeds)

    small_train = _render_split_params(tmp_path / "small-train", small, "train")
    large_train = _render_split_params(tmp_path / "large-train", large, "train")
    assert np.array_equal(small_train, large_train[:4])
    for split in ("val", "test"):
        assert np.array_equal(
            _render_split_params(tmp_path / f"small-{split}", small, split),
            _render_split_params(tmp_path / f"large-{split}", large, split),
        )


@pytest.mark.fake_vst
def test_per_index_content_identical_across_worker_counts(
    tmp_path: Path, install_fake_plugin: FakeVST3Plugin
) -> None:
    """One-worker and two-worker shard ownership produce byte-identical rows.

    :param tmp_path: Pytest fixture providing output directories for each worker layout.
    :param install_fake_plugin: Swaps the VST loader for a deterministic fake plugin.
    """
    spec = _multishard_lance_spec(train_val_test_sizes=(8, 0, 0), samples_per_shard=2)

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
    got = _render_param_array(tmp_path / "a.lance", base_seed=_BASE_SEED, num_samples=num_samples)
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
    spec = _multishard_lance_spec(train_val_test_sizes=(2, 0, 0), samples_per_shard=1)
    seeds = [
        int(args[args.index("--base_seed") + 1])
        for args in (build_generate_args(spec, shard, tmp_path) for shard in spec.shards)
    ]
    assert seeds[0] != seeds[1]

    shard0 = _render_param_array(tmp_path / "s0.lance", base_seed=seeds[0], num_samples=2)
    shard1 = _render_param_array(tmp_path / "s1.lance", base_seed=seeds[1], num_samples=2)
    assert not np.array_equal(shard0, shard1)

    shard0_rerun = _render_param_array(tmp_path / "s0b.lance", base_seed=seeds[0], num_samples=2)
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
    out = tmp_path / "shard.lance"
    _render_param_array(out, base_seed=_BASE_SEED, num_samples=2)
    dataset = lance.dataset(str(out))
    for field in (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD):
        assert field in dataset.schema.names, f"missing column {field}"
        assert np.isfinite(_read_column(out, field)).all(), f"non-finite values in {field}"
