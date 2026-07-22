"""Tests for symmetric-block canonicalization of encoded param rows."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from synth_setter.data.vst.param_canonicalization import (
    CanonicalBlocks,
    block_indices_by_prefix,
    canonicalize_blocks,
    resolve_canonical_blocks,
)
from synth_setter.data.vst.param_spec import ContinuousParameter, ParamSpec
from synth_setter.data.vst.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC
from synth_setter.data.vst_datamodule import RawBatch, prepare_batch
from synth_setter.param_spec_name import ParamSpecName


def _spec(names: list[str]) -> ParamSpec:
    """Build a flat all-continuous spec from bare param names.

    :param names: Param names in encoding order.
    :returns: Spec with one scalar continuous param per name and no note params.
    """
    return ParamSpec([ContinuousParameter(name=n, min=0.0, max=1.0) for n in names], [])


# Three two-dim blocks keyed on the second dim, and two two-dim blocks keyed
# on the first — small layouts the sorting tests assert against by hand.
THREE_BLOCKS = CanonicalBlocks(indices=((0, 1), (2, 3), (4, 5)), key_offset=1)
TWO_BLOCKS = CanonicalBlocks(indices=((0, 1), (2, 3)), key_offset=0)

# surge_simple's per-osc volume dims (block offset 4 in each osc index run).
OSC_VOLUME_DIMS = (20, 27, 34)


class TestBlockIndicesByPrefix:
    """block_indices_by_prefix derivation and validation."""
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
    """canonicalize_blocks row-wise sorting semantics."""

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

    def test_tied_keys_keep_original_block_order(self) -> None:
        """Equal keys keep the stored block order (stable sort)."""
        row = np.array([[0.1, 0.5, 0.2, 0.5, 0.3, 0.5]], dtype=np.float32)
        out = canonicalize_blocks(row, THREE_BLOCKS)
        assert np.allclose(out, row)

    def test_input_array_is_not_mutated(self) -> None:
        """The input array is left untouched (pure function)."""
        row = np.array([[0.9, 0.1, 0.8, 0.5, 0.7, 0.3]], dtype=np.float32)
        original = row.copy()
        canonicalize_blocks(row, THREE_BLOCKS)
        assert np.array_equal(row, original)


class TestResolveCanonicalBlocks:
    """Registry resolution for specs with symmetric blocks."""
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
    """canonicalize_symmetric_blocks flag through LanceVSTDataModule."""

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
        from tests.helpers.lance_fixtures import make_shard_columns, write_lance_shard

        from synth_setter.data.lance_datamodule import LanceVSTDataModule

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
