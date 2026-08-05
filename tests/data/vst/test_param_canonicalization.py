"""Tests for symmetric-block canonicalization of encoded param rows."""

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from synth_setter.data.vst.param_canonicalization import (
    SYMMETRIC_BLOCK_REGISTRY,
    CanonicalBlocks,
    block_indices_by_prefix,
    canonicalize_blocks,
    resolve_canonical_blocks,
)
from synth_setter.data.vst.param_spec import (
    ContinuousParameter,
    ParamSpec,
    decode_model_output,
)
from synth_setter.data.vst.param_spec_registry import param_specs, plugin_state_paths
from synth_setter.data.vst.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC
from synth_setter.data.vst_datamodule import RawBatch, prepare_batch
from synth_setter.evaluation.compute_audio_metrics import compute_mss
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from tests._vst import PLUGIN_PATH, _composed_synth_version


def _spec(names: list[str]) -> ParamSpec:
    """Build a flat all-continuous spec from bare param names.

    :param names: Param names in encoding order.
    :returns: Spec with one scalar continuous param per name and no note params.
    """
    return ParamSpec([ContinuousParameter(name=n, min=0.0, max=1.0) for n in names], [])


# Hand-checkable layouts exercising second- and first-offset sort keys.
THREE_BLOCKS = CanonicalBlocks(indices=((0, 1), (2, 3), (4, 5)), key_offset=1)
TWO_BLOCKS = CanonicalBlocks(indices=((0, 1), (2, 3)), key_offset=0)

# surge_simple's per-osc volume dims (block offset 4 in each osc index run).
OSC_VOLUME_DIMS = (20, 27, 34)


class TestBlockIndicesByPrefix:
    def test_surge_simple_osc_blocks_are_aligned_and_volume_keyed(self) -> None:
        """surge_simple's three osc blocks resolve to aligned index runs keyed on volume."""
        blocks = block_indices_by_prefix(
            SURGE_SIMPLE_PARAM_SPEC,
            prefixes=("a_osc_1_", "a_osc_2_", "a_osc_3_"),
            key_suffix="volume",
        )
        assert blocks.indices == (
            (16, 17, 18, 19, 20, 21, 22),
            (23, 24, 25, 26, 27, 28, 29),
            (30, 31, 32, 33, 34, 35, 36),
        )
        assert blocks.key_offset == 4

    def test_mismatched_block_suffixes_raise_value_error(self) -> None:
        """Blocks whose param suffix sequences differ are rejected."""
        spec = _spec(["x_1_a", "x_1_b", "x_2_a", "x_2_c"])
        with pytest.raises(ValueError, match="suffix"):
            block_indices_by_prefix(spec, prefixes=("x_1_", "x_2_"), key_suffix="a")

    def test_key_suffix_absent_raises_value_error(self) -> None:
        """A key_suffix not present in the blocks is rejected."""
        spec = _spec(["x_1_a", "x_2_a"])
        with pytest.raises(ValueError, match="key_suffix"):
            block_indices_by_prefix(spec, prefixes=("x_1_", "x_2_"), key_suffix="volume")

    def test_prefix_without_params_raises_value_error(self) -> None:
        """A prefix matching no spec params is rejected."""
        spec = _spec(["x_1_a", "x_2_a"])
        with pytest.raises(ValueError, match="x_9_"):
            block_indices_by_prefix(spec, prefixes=("x_1_", "x_9_"), key_suffix="a")

    def test_onehot_param_in_block_raises_value_error(self) -> None:
        """Non-scalar (onehot) params cannot form canonical blocks."""
        from synth_setter.data.vst.param_spec import CategoricalParameter

        spec = ParamSpec(
            [
                CategoricalParameter(name="x_1_a", values=[0.0, 1.0], encoding="onehot"),
                CategoricalParameter(name="x_2_a", values=[0.0, 1.0], encoding="onehot"),
            ],
            [],
        )
        with pytest.raises(ValueError, match="non-scalar"):
            block_indices_by_prefix(spec, prefixes=("x_1_", "x_2_"), key_suffix="a")


class TestCanonicalizeBlocks:
    def test_blocks_sorted_by_descending_key(self) -> None:
        """Blocks come back ordered by descending key value."""
        row = np.array([[0.9, 0.1, 0.8, 0.5, 0.7, 0.3]], dtype=np.float32)
        out = canonicalize_blocks(row, THREE_BLOCKS)
        assert np.allclose(out, [[0.8, 0.5, 0.7, 0.3, 0.9, 0.1]])

    def test_other_dims_and_row_order_untouched(self) -> None:
        """Dims outside the blocks and the row order are preserved."""
        blocks = CanonicalBlocks(indices=((1, 2), (3, 4)), key_offset=0)
        rows = np.array(
            [
                [0.11, 0.2, 0.6, 0.9, 0.5, 0.99],
                [0.22, 0.9, 0.5, 0.2, 0.6, 0.88],
            ],
            dtype=np.float32,
        )
        out = canonicalize_blocks(rows, blocks)
        assert np.allclose(out[:, 0], [0.11, 0.22])
        assert np.allclose(out[:, 5], [0.99, 0.88])
        assert np.allclose(out[0], [0.11, 0.9, 0.5, 0.2, 0.6, 0.99])
        assert np.allclose(out[1], [0.22, 0.9, 0.5, 0.2, 0.6, 0.88])

    def test_permuted_blocks_map_to_same_canonical_row(self) -> None:
        """All block permutations of one row share one canonical form."""
        base = np.array([[0.9, 0.1, 0.8, 0.5, 0.7, 0.3]], dtype=np.float32)
        permuted = np.array([[0.7, 0.3, 0.9, 0.1, 0.8, 0.5]], dtype=np.float32)
        assert np.allclose(
            canonicalize_blocks(base, THREE_BLOCKS), canonicalize_blocks(permuted, THREE_BLOCKS)
        )

    def test_canonical_input_is_fixed_point(self) -> None:
        """An already-canonical row is returned unchanged."""
        row = np.array([[0.8, 0.5, 0.7, 0.3, 0.6, 0.1]], dtype=np.float32)
        out = canonicalize_blocks(row, THREE_BLOCKS)
        assert np.allclose(out, row)

    def test_tied_keys_use_block_values_as_deterministic_tiebreaker(self) -> None:
        """Permuted blocks with equal keys share one canonical form."""
        rows = np.array(
            [
                [0.1, 0.5, 0.2, 0.5, 0.3, 0.5],
                [0.3, 0.5, 0.1, 0.5, 0.2, 0.5],
            ],
            dtype=np.float32,
        )
        out = canonicalize_blocks(rows, THREE_BLOCKS)
        expected = np.array(
            [
                [0.3, 0.5, 0.2, 0.5, 0.1, 0.5],
                [0.3, 0.5, 0.2, 0.5, 0.1, 0.5],
            ],
            dtype=np.float32,
        )
        assert np.array_equal(out, expected)

    def test_input_array_is_not_mutated(self) -> None:
        """The input array is left untouched (pure function)."""
        row = np.array([[0.9, 0.1, 0.8, 0.5, 0.7, 0.3]], dtype=np.float32)
        original = row.copy()
        canonicalize_blocks(row, THREE_BLOCKS)
        assert np.array_equal(row, original)

    def test_stored_float32_dtype_survives_canonicalization(self) -> None:
        """Sorting keeps the stored width — a float64 result would double batch memory."""
        rows = np.array([[0.9, 0.1, 0.8, 0.5, 0.7, 0.3]], dtype=np.float32)
        assert canonicalize_blocks(rows, THREE_BLOCKS).dtype == np.float32

    def test_empty_batch_returns_empty_array_of_same_shape(self) -> None:
        """A zero-row batch canonicalizes to an empty array, not a reshape error."""
        rows = np.zeros((0, 6), dtype=np.float32)
        out = canonicalize_blocks(rows, THREE_BLOCKS)
        assert out.shape == (0, 6)
        assert out.dtype == np.float32


class TestResolveCanonicalBlocks:
    def test_surge_simple_resolves_osc_blocks(self) -> None:
        """surge_simple resolves to its osc blocks keyed on volume."""
        blocks = resolve_canonical_blocks(ParamSpecName("surge_simple"))
        assert blocks.indices[0] == (16, 17, 18, 19, 20, 21, 22)
        assert blocks.key_offset == 4

    def test_unregistered_spec_raises_key_error(self) -> None:
        """Specs without registered symmetric blocks raise KeyError."""
        with pytest.raises(KeyError):
            resolve_canonical_blocks(ParamSpecName("surge_xt"))


class TestDataModuleWiring:
    def _write_dataset(self, root: Path) -> None:
        """Write tiny surge_simple-width train/val/test Lance splits.

        :param root: Dataset root receiving the three ``*.lance`` splits.
        """
        from tests.helpers.lance_fixtures import make_shard_columns, write_lance_shard

        for split, seed in (("train", 0), ("val", 1), ("test", 2)):
            columns = make_shard_columns(8, num_params=92, seed=seed)
            write_lance_shard(root / f"{split}.lance", columns)

    def test_flag_canonicalizes_train_and_val_batches(self, tmp_path: Path) -> None:
        """With the flag on, train and val batches sort osc blocks by volume.

        :param tmp_path: Pytest per-test directory the splits are written under.
        """
        from synth_setter.data.lance_datamodule import LanceVSTDataModule

        self._write_dataset(tmp_path)
        module = LanceVSTDataModule(
            tmp_path,
            param_spec_name=ParamSpecName("surge_simple"),
            batch_size=4,
            ot=False,
            use_saved_mean_and_variance=False,
            canonicalize_symmetric_blocks=True,
        )
        module.setup("fit")
        for batch in (next(iter(module.train_dataloader())), next(iter(module.val_dataloader()))):
            volumes = batch["params"][:, list(OSC_VOLUME_DIMS)].numpy()
            assert (np.diff(volumes, axis=1) <= 0).all()

    def test_flag_off_leaves_stored_row_order(self, tmp_path: Path) -> None:
        """Default flag leaves stored param rows untouched.

        :param tmp_path: Pytest per-test directory the splits are written under.
        """
        from synth_setter.data.lance_datamodule import LanceVSTDataModule
        from tests.helpers.lance_fixtures import make_shard_columns, write_lance_shard

        for split, seed in (("train", 0), ("val", 1), ("test", 2)):
            columns = make_shard_columns(8, num_params=92, seed=seed)
            write_lance_shard(tmp_path / f"{split}.lance", columns)
        module = LanceVSTDataModule(
            tmp_path,
            param_spec_name=ParamSpecName("surge_simple"),
            batch_size=4,
            ot=False,
            use_saved_mean_and_variance=False,
        )
        module.setup("fit")
        batch = next(iter(module.val_dataloader()))
        expected = make_shard_columns(8, num_params=92, seed=1)["param_array"][:4] * 2 - 1
        assert np.allclose(batch["params"].numpy(), expected, atol=1e-6)

    @pytest.mark.parametrize("value", ["typo", 1, None])
    def test_non_bool_flag_is_rejected_at_construction(self, value: object) -> None:
        """A non-bool override fails loudly instead of silently enabling the transform.

        Hydra composes ``datamodule.canonicalize_symmetric_blocks=typo`` as the
        truthy string ``"typo"``, which would otherwise rewrite every training
        target rather than failing the run.

        :param value: Non-bool override reaching the datamodule.
        """
        from synth_setter.data.lance_datamodule import LanceVSTDataModule

        with pytest.raises(TypeError, match="canonicalize_symmetric_blocks"):
            LanceVSTDataModule(
                Path("unused"),
                param_spec_name=ParamSpecName("surge_simple"),
                canonicalize_symmetric_blocks=value,  # type: ignore[arg-type]
            )

    def _fake_train_batch(self, *, canonicalize: bool, seed: int) -> np.ndarray:
        """Draw one fake train batch under a pinned global RNG.

        :param canonicalize: Whether the datamodule canonicalizes its draws.
        :param seed: Global torch seed the fake dataset draws from.
        :returns: The batch's ``params`` rows.
        """
        from synth_setter.data.lance_datamodule import LanceVSTDataModule

        module = LanceVSTDataModule(
            Path("unused"),
            param_spec_name=ParamSpecName("surge_simple"),
            batch_size=16,
            ot=False,
            fake=True,
            use_saved_mean_and_variance=False,
            canonicalize_symmetric_blocks=canonicalize,
        )
        module.setup("fit")
        torch.manual_seed(seed)
        return next(iter(module.train_dataloader()))["params"].numpy()

    @pytest.mark.parametrize(
        ("stage", "loader"),
        [("fit", "train"), ("fit", "val"), ("test", "test"), ("predict", "predict")],
    )
    def test_flag_canonicalizes_every_served_split(
        self, tmp_path: Path, stage: str, loader: str
    ) -> None:
        """Every stage's loader serves canonical blocks, not just the fit splits.

        :param tmp_path: Pytest per-test directory the splits are written under.
        :param stage: Lightning stage passed to ``setup``.
        :param loader: Dataloader the stage exposes.
        """
        from synth_setter.data.lance_datamodule import LanceVSTDataModule

        self._write_dataset(tmp_path)
        blocks = resolve_canonical_blocks(ParamSpecName("surge_simple"))
        sort_keys = [block[blocks.key_offset] for block in blocks.indices]
        module = LanceVSTDataModule(
            tmp_path,
            param_spec_name=ParamSpecName("surge_simple"),
            batch_size=4,
            ot=False,
            use_saved_mean_and_variance=False,
            canonicalize_symmetric_blocks=True,
            predict_file=tmp_path / "test.lance",
        )
        module.setup(stage)
        batch = next(iter(getattr(module, f"{loader}_dataloader")()))
        assert (np.diff(batch["params"][:, sort_keys].numpy(), axis=1) <= 0).all()

    def test_fake_mode_canonicalizes_exactly_the_drawn_rows(self) -> None:
        """Fake splits canonicalize too, so smoke runs cannot silently drop the flag.

        Pinning the global seed makes both datamodules draw identical rows, so the flag-on batch
        must be exactly the flag-off batch with its blocks sorted — stronger than asserting that
        some row came out ordered.
        """
        blocks = resolve_canonical_blocks(ParamSpecName("surge_simple"))
        drawn = self._fake_train_batch(canonicalize=False, seed=1234)
        canonicalized = self._fake_train_batch(canonicalize=True, seed=1234)
        assert np.array_equal(canonicalized, canonicalize_blocks(drawn, blocks))

    def test_fake_mode_flag_off_leaves_drawn_order(self) -> None:
        """Without the flag the drawn block order survives, so the pair is not vacuous."""
        blocks = resolve_canonical_blocks(ParamSpecName("surge_simple"))
        drawn = self._fake_train_batch(canonicalize=False, seed=1234)
        assert not np.array_equal(drawn, canonicalize_blocks(drawn, blocks))


class TestPrepareBatchCanonicalization:
    """canonical_blocks plumbing through prepare_batch."""

    def _raw(self, param_array: np.ndarray) -> RawBatch:
        """Wrap one param array as a params-only RawBatch.

        :param param_array: Encoded ``(batch, num_params)`` rows in ``[0, 1]``.
        :returns: Raw batch whose optional modalities are all ``None``.
        """
        return cast(
            RawBatch,
            {
                "param_array": param_array,
                "audio": None,
                "mel_spec": None,
                "music2latent": None,
            },
        )

    def test_canonical_blocks_sorts_params_in_output(self) -> None:
        """prepare_batch canonicalizes before the [-1, 1] rescale."""
        raw = self._raw(np.array([[0.2, 0.9, 0.6, 0.1]], dtype=np.float32))
        generator = torch.Generator().manual_seed(0)
        batch = prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=True,
            ot=False,
            generator=generator,
            canonical_blocks=TWO_BLOCKS,
        )
        params = batch["params"]
        assert params is not None
        expected = np.array([[0.6, 0.1, 0.2, 0.9]], dtype=np.float32) * 2 - 1
        assert np.allclose(params.numpy(), expected)

    def test_ot_matching_preserves_canonical_blocks_and_row_alignment(self) -> None:
        """Under OT the emitted blocks stay canonical and keep their mel companion.

        Shipped training runs with ``ot=true``; a regression where
        ``_hungarian_match`` reordered params independently of the other
        modalities would train canonical targets against the wrong conditioning.
        """
        param_array = np.array(
            [[0.2, 0.9, 0.6, 0.1], [0.7, 0.3, 0.4, 0.8], [0.1, 0.2, 0.5, 0.6]],
            dtype=np.float32,
        )
        # Each row's mel is a constant plane carrying that row's index, so the
        # pairing survives any row permutation OT applies.
        mel = np.stack(
            [np.full((1, 2, 2), float(i), dtype=np.float32) for i in range(len(param_array))]
        )
        raw = cast(
            RawBatch,
            {
                "param_array": param_array,
                "audio": None,
                "mel_spec": mel,
                "music2latent": None,
            },
        )
        batch = prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=True,
            ot=True,
            generator=torch.Generator().manual_seed(0),
            canonical_blocks=TWO_BLOCKS,
        )
        params, out_mel = batch["params"], batch["mel"]
        assert params is not None
        assert out_mel is not None

        keys = params.numpy()[:, [TWO_BLOCKS.indices[0][0], TWO_BLOCKS.indices[1][0]]]
        assert (np.diff(keys, axis=1) <= 0).all()

        canonical = canonicalize_blocks(param_array, TWO_BLOCKS) * 2 - 1
        for row, plane in zip(params.numpy(), out_mel.numpy(), strict=True):
            source = int(plane.flat[0])
            assert np.allclose(row, canonical[source])

    def test_canonical_blocks_none_leaves_params_unchanged(self) -> None:
        """Without canonical_blocks the params pass through as stored."""
        raw = self._raw(np.array([[0.2, 0.9, 0.6, 0.1]], dtype=np.float32))
        generator = torch.Generator().manual_seed(0)
        batch = prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=True,
            ot=False,
            generator=generator,
        )
        params = batch["params"]
        assert params is not None
        expected = np.array([[0.2, 0.9, 0.6, 0.1]], dtype=np.float32) * 2 - 1
        assert np.allclose(params.numpy(), expected)


def _real_render_config(param_spec_name: str) -> RenderConfig:
    """Build the real-plugin render config for one registered spec.

    :param param_spec_name: Registry key naming the spec and its preset.
    :returns: Validated headless render configuration.
    """
    values: dict[str, object] = {
        "synth": {
            "name": param_spec_name,
            "param_spec_name": param_spec_name,
            "plugin_path": PLUGIN_PATH,
            "plugin_state_path": plugin_state_paths[param_spec_name],
            "synth_version": _composed_synth_version(param_spec_name),
        },
        "sample_rate": 44100,
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "samples_per_render_batch": 2,
        "samples_per_shard": 4,
        "gui_toggle_cadence": "never",
        "plugin_reload_cadence": "once",
    }
    return RenderConfig(**values)  # type: ignore[arg-type]


def _block_permutations(row: np.ndarray, indices: np.ndarray) -> Iterator[np.ndarray]:
    """Yield each non-identity block permutation of one encoded row.

    The identity is excluded: rendering it measures the plugin's own phase
    variance, which the calibration already samples, not block interchange.

    :param row: Encoded row to permute.
    :param indices: ``(blocks, width)`` encoded-dim layout to permute.
    :yields: One row per non-identity block ordering.
    :ytype: np.ndarray
    """
    identity = tuple(range(len(indices)))
    for order in itertools.permutations(identity):
        if order == identity:
            continue
        candidate = row.copy()
        candidate[indices.reshape(-1)] = row[indices[list(order)].reshape(-1)]
        yield candidate


@dataclass(frozen=True)
class _RenderProbe:
    """Renders encoded rows and draws fresh ones for one spec under test.

    .. attribute :: render

        Renders one encoded row to ``(channels, samples)`` audio.

    .. attribute :: spec

        Spec whose encoded width fresh rows are drawn at.

    .. attribute :: rng

        Draw source for fresh rows.
    """

    render: Callable[[np.ndarray], np.ndarray]
    spec: ParamSpec
    rng: np.random.Generator

    def draw(self) -> np.ndarray:
        """Draw one fresh encoded row.

        :returns: Encoded ``[0, 1]`` row at the spec's width.
        """
        return self.rng.random(len(self.spec))


def _audible_rows(probe: _RenderProbe, *, count: int, min_loudness: float) -> list[np.ndarray]:
    """Draw encoded rows that render above the generator's loudness gate.

    :param probe: Render context for the spec under test.
    :param count: Rows required.
    :param min_loudness: dBFS gate, matching the dataset generator's own.
    :returns: Exactly ``count`` audible rows.
    """
    floor = 10 ** (min_loudness / 20)
    candidates = (probe.draw() for _ in range(80))
    rows = list(
        itertools.islice(
            (c for c in candidates if np.sqrt((probe.render(c) ** 2).mean()) > floor), count
        )
    )
    assert len(rows) == count, "fixture drew too few audible rows to test"
    return rows


def _permutation_scores(
    probe: _RenderProbe, row: np.ndarray, indices: np.ndarray
) -> tuple[float, float, float]:
    """Score one row's block permutations against two reference populations.

    Scored with MSS, not sample-wise: the Surge render randomizes phase, so a
    waveform metric saturates on repeat renders of one row and resolves nothing.

    :param probe: Render context for the spec under test.
    :param row: Encoded row whose blocks are permuted.
    :param indices: ``(blocks, width)`` encoded-dim layout to permute.
    :returns: Worst non-identity permutation score, the closer of two unrelated
        rows, and the largest of three repeat renders of ``row``.
    """
    base = probe.render(row)
    unrelated = min(compute_mss(base, probe.render(probe.draw())) for _ in range(2))
    repeat = max(compute_mss(base, probe.render(row)) for _ in range(3))
    worst = max(compute_mss(base, probe.render(c)) for c in _block_permutations(row, indices))
    return worst, unrelated, repeat


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.parametrize("param_spec_name", sorted(SYMMETRIC_BLOCK_REGISTRY))
def test_registered_blocks_render_interchangeably(param_spec_name: str) -> None:
    """Verify registered block permutations render within the repeat-render MSS floor (#1886).

    :param param_spec_name: Registry key whose block group is under test.
    """
    spec = param_specs[param_spec_name]
    blocks = resolve_canonical_blocks(ParamSpecName(param_spec_name))
    indices = np.array(blocks.indices)
    config = _real_render_config(param_spec_name)
    renderer = make_audio_renderer(config)

    def render(encoded: np.ndarray) -> np.ndarray:
        """Render one encoded row through the real plugin.

        :param encoded: Encoded ``[0, 1]`` row for the spec under test.
        :returns: ``(channels, samples)`` audio.
        """
        synth_params, note = decode_model_output(encoded * 2 - 1, spec)
        start, end = sorted(note["note_start_and_end"])
        return renderer.render(synth_params, int(note["pitch"]), config.velocity, (start, end))

    probe = _RenderProbe(render=render, spec=spec, rng=np.random.default_rng(11))
    # Several rows: oscillator identity could matter only under a routing or
    # mode combination one row never selects.
    rows = _audible_rows(probe, count=3, min_loudness=config.min_loudness)

    for index, row in enumerate(rows):
        worst, unrelated, repeat = _permutation_scores(probe, row, indices)
        # Repeat renders must remain far below unrelated rows for this tolerance
        # to distinguish permutations — see #1886.
        assert repeat < 0.1 * unrelated, (
            f"{param_spec_name} render is too nondeterministic to test on row {index}: "
            f"repeat render scores MSS {repeat:.3f} against {unrelated:.3f} for an "
            "unrelated row, so the tolerance below would not discriminate"
        )
        assert worst < max(0.1 * unrelated, 2 * repeat), (
            f"{param_spec_name} blocks are not interchangeable on row {index}: worst "
            f"permutation scores MSS {worst:.3f}, against {unrelated:.3f} for the closer "
            f"of two unrelated rows and {repeat:.3f} for a repeat render of the same row"
        )
