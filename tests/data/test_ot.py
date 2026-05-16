"""Behavioral tests for :mod:`synth_setter.data.ot`.

Covers the optimal-transport helpers used by the surge / kosc / ksin
datamodules to align random ``noise`` samples to ground-truth ``params`` (and
the rest of the batch) via the Hungarian algorithm before passing them to the
flow-matching loss:

* :func:`_hungarian_match` — pairs ``noise`` to ``params`` by minimum
  pairwise Euclidean distance, reorders both, and reorders extra arrays
  by the ``params`` permutation so the whole batch stays consistent.
* :func:`concatenate`, :func:`stack` — type-dispatched dim-0 joins that
  always return ``torch.Tensor``.
* :func:`_collate_tuple`, :func:`_collate_dict` — produce flow-matching
  batches: stack inputs, draw fresh ``noise`` from ``torch.randn_like``.
* :func:`regular_collate_fn` — dispatches tuple/list/dict to the
  corresponding ``_collate_*``.
* :func:`_ot_collate_tuple`, :func:`_ot_collate_dict` — collate then
  ``_hungarian_match`` so ``noise`` and ``params`` are paired.
* :func:`ot_collate_fn` — dispatches tuple/list/dict to the OT path.

The tests pin observable behavior: shape contracts, type promotion
(numpy → torch), the optimality of the returned permutation, and the
``arg is None → reordered_arg is None`` passthrough that lets callers
drop unused modalities.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import numpy as np
import pytest
import torch

from synth_setter.data.ot import (
    _collate_dict,
    _collate_tuple,
    _hungarian_match,
    _ot_collate_dict,
    _ot_collate_tuple,
    concatenate,
    ot_collate_fn,
    regular_collate_fn,
    stack,
)

# A row-swapped pair: noise[0] is nearest to params[1] and noise[1] is nearest
# to params[0], so the Hungarian-optimal assignment is the swap [1, 0]. Used by
# every test that only needs "an input where the LAP returns a non-identity
# permutation" without depending on which permutation it is.
_NOISE_SWAPPED = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
_PARAMS_SWAPPED = torch.tensor([[1.0, 1.0], [0.0, 0.0]])


@pytest.fixture(autouse=True)
def _isolate_torch_rng() -> Iterator[None]:
    """Snapshot+restore the global torch RNG around every test in this module.

    Several tests call ``torch.manual_seed(0)`` to make ``torch.randn_like``
    deterministic. Without isolation, that seeding would leak into later
    tests when the full suite runs. ``torch.random.fork_rng()`` snapshots
    the CPU (and CUDA, if available) RNG state on entry and restores it on
    exit, so each test sees the same RNG state as if run alone.

    :yields None: Control returns to the test under the forked RNG context.
    """
    with torch.random.fork_rng(devices=[]):
        yield


# --------------------------------------------------------------------------- #
# _hungarian_match                                                            #
# --------------------------------------------------------------------------- #


class TestHungarianMatch:
    """Pair ``noise`` to ``params`` via scipy's ``linear_sum_assignment`` on Euclidean cdist."""

    def test_returns_pair_when_no_extra_args(self) -> None:
        """With no ``*args``, the return is a 2-tuple of ``(noise, params)`` only."""
        out = _hungarian_match(_NOISE_SWAPPED, _PARAMS_SWAPPED)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_pairs_each_noise_row_to_nearest_param_row(self) -> None:
        """Identity-permutation cost: each noise row's nearest param IS itself."""
        # noise == params, so the Hungarian-optimal assignment is the identity.
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 4)).astype(np.float32)
        noise = torch.from_numpy(data.copy())
        params = torch.from_numpy(data.copy())
        out_noise, out_params = _hungarian_match(noise, params)
        assert torch.allclose(out_noise, out_params)

    def test_minimizes_total_pairwise_distance(self) -> None:
        """Reordered noise + params must beat the input ordering on total cdist."""
        # Construct a pathological input where rows are deliberately shuffled
        # so identity is non-optimal. We seed both tensors to make the
        # optimal pairing exact: noise rows = [a, b, c]; params rows = [b, c, a].
        # The Hungarian optimum is noise[0]<->params[2] (both 'a'),
        # noise[1]<->params[0] (both 'b'), noise[2]<->params[1] (both 'c').
        a, b, c = (
            torch.tensor([10.0, 0.0]),
            torch.tensor([0.0, 10.0]),
            torch.tensor([-10.0, 0.0]),
        )
        noise = torch.stack([a, b, c])
        params = torch.stack([b, c, a])
        out_noise, out_params = _hungarian_match(noise, params)
        assert torch.allclose(torch.cdist(out_noise, out_params).diag().sum(), torch.tensor(0.0))

    def test_numpy_noise_converted_to_torch(self) -> None:
        """``np.ndarray`` noise becomes a ``torch.Tensor`` on the way out."""
        noise = _NOISE_SWAPPED.numpy()
        # _hungarian_match accepts ndarray at runtime (it calls torch.from_numpy);
        # the source signature is too narrow — see pyrightconfig.json exclude.
        out_noise, _ = _hungarian_match(noise, _PARAMS_SWAPPED)  # type: ignore[arg-type]
        assert isinstance(out_noise, torch.Tensor)

    def test_numpy_params_converted_to_torch(self) -> None:
        """``np.ndarray`` params become a ``torch.Tensor`` on the way out."""
        params = _PARAMS_SWAPPED.numpy()
        _, out_params = _hungarian_match(_NOISE_SWAPPED, params)  # type: ignore[arg-type]
        assert isinstance(out_params, torch.Tensor)

    def test_extra_arg_reordered_by_col_ind(self) -> None:
        """The trailing ``*args`` get permuted by ``col_ind`` (the params permutation)."""
        # When params are shuffled into [b, c, a], the optimal col_ind that
        # restores the identity diagonal is [2, 0, 1]. The extra arg should
        # come out reordered the same way.
        a, b, c = (
            torch.tensor([10.0, 0.0]),
            torch.tensor([0.0, 10.0]),
            torch.tensor([-10.0, 0.0]),
        )
        noise = torch.stack([a, b, c])
        params = torch.stack([b, c, a])
        extra = torch.tensor([100, 200, 300])
        _, _, out_extra = _hungarian_match(noise, params, extra)
        assert torch.equal(out_extra, torch.tensor([300, 100, 200]))

    def test_extra_none_arg_passes_through_as_none(self) -> None:
        """``arg is None`` is preserved as ``None`` instead of being indexed."""
        _, _, out_extra = _hungarian_match(_NOISE_SWAPPED, _PARAMS_SWAPPED, None)
        assert out_extra is None

    def test_multiple_extra_args_each_handled_independently(self) -> None:
        """Mixed extras: tensors reordered, ``None``s pass through, in original order."""
        a = torch.tensor([10, 20])
        b = torch.tensor([100, 200])
        result = _hungarian_match(_NOISE_SWAPPED, _PARAMS_SWAPPED, a, None, b)
        assert len(result) == 5
        assert result[2] is not None
        assert result[3] is None
        assert result[4] is not None

    def test_deterministic_for_same_input(self) -> None:
        """Same inputs ⇒ same outputs (scipy LAP is deterministic; we don't reseed RNG)."""
        noise = torch.tensor([[3.0, 1.0], [0.0, 2.0], [-1.0, 4.0]])
        params = torch.tensor([[0.0, 2.0], [-1.0, 4.0], [3.0, 1.0]])
        out1 = _hungarian_match(noise, params)
        out2 = _hungarian_match(noise, params)
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])


# --------------------------------------------------------------------------- #
# concatenate                                                                 #
# --------------------------------------------------------------------------- #


class TestConcatenate:
    """Type-dispatched dim-0 concat that always returns ``torch.Tensor``.

    ``concatenate`` is annotated as ``Union[Tensor, ndarray]`` in the source,
    but the body indexes ``list_of_arrays[0]`` — it really takes a *list*.
    ``# type: ignore[arg-type]`` acknowledges the mismatch (``ot.py`` is in the
    pyright exclude list, so we can't widen the source signature in this PR).
    """

    def test_concat_torch_tensors_along_dim_0(self) -> None:
        """``[tensor(2,3), tensor(2,3)] -> tensor(4,3)``."""
        result = concatenate([torch.zeros(2, 3), torch.ones(2, 3)])  # type: ignore[arg-type]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 3)

    def test_concat_torch_preserves_values(self) -> None:
        """Concatenation preserves block ordering (no shuffling)."""
        result = concatenate([torch.zeros(2, 3), torch.ones(2, 3)])  # type: ignore[arg-type]
        assert torch.all(result[:2] == 0.0)
        assert torch.all(result[2:] == 1.0)

    def test_concat_numpy_arrays_returns_torch_tensor(self) -> None:
        """``np.ndarray`` list dispatches through ``np.concatenate`` then ``torch.from_numpy``."""
        arrays = [np.zeros((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)]
        result = concatenate(arrays)  # type: ignore[arg-type]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 3)

    def test_concat_torch_preserves_dtype(self) -> None:
        """Float32 inputs come back as float32 (no implicit upcast)."""
        result = concatenate([torch.zeros(2, 3, dtype=torch.float32)] * 2)  # type: ignore[arg-type]
        assert result.dtype == torch.float32

    def test_concat_single_element_list(self) -> None:
        """One-element input list returns the input shape unchanged."""
        result = concatenate([torch.zeros(3, 4)])  # type: ignore[arg-type]
        assert result.shape == (3, 4)

    def test_concat_three_arrays(self) -> None:
        """N-array concat sums the leading axis."""
        result = concatenate([torch.zeros(2, 4)] * 3)  # type: ignore[arg-type]
        assert result.shape == (6, 4)


# --------------------------------------------------------------------------- #
# stack                                                                       #
# --------------------------------------------------------------------------- #


class TestStack:
    """Type-dispatched dim-0 stack that always returns ``torch.Tensor``.

    Same ``# type: ignore[arg-type]`` rationale as ``TestConcatenate``.
    """

    def test_stack_torch_tensors_inserts_new_axis_at_zero(self) -> None:
        """``[tensor(3), tensor(3)] -> tensor(2, 3)``."""
        result = stack([torch.zeros(3), torch.ones(3)])  # type: ignore[arg-type]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)

    def test_stack_numpy_arrays_returns_torch_tensor(self) -> None:
        """Numpy input goes through ``np.stack`` then ``torch.from_numpy``."""
        arrays = [np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
        result = stack(arrays)  # type: ignore[arg-type]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)

    def test_stack_torch_preserves_dtype(self) -> None:
        """Float32 inputs come back as float32."""
        result = stack([torch.zeros(3, dtype=torch.float32)] * 2)  # type: ignore[arg-type]
        assert result.dtype == torch.float32

    def test_stack_preserves_values(self) -> None:
        """Stacking preserves block ordering."""
        result = stack([torch.zeros(3), torch.ones(3)])  # type: ignore[arg-type]
        assert torch.all(result[0] == 0.0)
        assert torch.all(result[1] == 1.0)


# --------------------------------------------------------------------------- #
# _collate_tuple                                                              #
# --------------------------------------------------------------------------- #


class TestCollateTuple:
    """Collate ``(sins, params, sin_fn)`` tuples; build ``noise`` via ``randn_like(params)``."""

    @staticmethod
    def _make_batch(num_items: int = 3, sin_fn: object = "fake_fn") -> list[tuple]:
        """Return a fresh ``[(sins, params, sin_fn), ...]`` batch.

        :param num_items: Number of per-item tuples in the batch.
        :param sin_fn: Sentinel value attached to every item (only the first is read).

        :return: A list of ``(sins, params, sin_fn)`` tuples ready for collate.
        :rtype: list[tuple]
        """
        return [
            (
                torch.full((2, 5), float(i)),
                torch.full((2, 4), float(i)),
                sin_fn,
            )
            for i in range(num_items)
        ]

    def test_returns_four_element_tuple(self) -> None:
        """Output is exactly ``(sins, params, noise, sin_fn)``."""
        out = _collate_tuple(self._make_batch())
        assert isinstance(out, tuple)
        assert len(out) == 4

    def test_concatenates_sins_and_params_along_dim_0(self) -> None:
        """Per-item ``(2, 5)`` ``sins`` from 3 items → ``(6, 5)`` after concat."""
        sins, params, _, _ = _collate_tuple(self._make_batch(num_items=3))
        assert sins.shape == (6, 5)
        assert params.shape == (6, 4)

    def test_noise_shape_matches_params(self) -> None:
        """``noise = torch.randn_like(params)`` ⇒ identical shape and dtype."""
        torch.manual_seed(0)
        _, params, noise, _ = _collate_tuple(self._make_batch())
        assert noise.shape == params.shape
        assert noise.dtype == params.dtype

    def test_sin_fn_taken_from_first_item(self) -> None:
        """The returned ``sin_fn`` is the first tuple's, not e.g. a re-collected list."""
        batch = [
            (torch.zeros(2, 5), torch.zeros(2, 4), "first_fn"),
            (torch.zeros(2, 5), torch.zeros(2, 4), "second_fn"),
        ]
        _, _, _, sin_fn = _collate_tuple(batch)
        assert sin_fn == "first_fn"


# --------------------------------------------------------------------------- #
# _collate_dict                                                               #
# --------------------------------------------------------------------------- #


class TestCollateDict:
    """Collate ``{params, mel_spec, audio}`` dicts; ``audio=None`` falls through."""

    @staticmethod
    def _make_batch(num_items: int = 3, audio: bool = True) -> list[dict]:
        """Return a fresh batch of per-item dicts.

        :param num_items: Number of per-item dicts in the batch.
        :param audio: When False, every per-item dict has ``audio=None``.

        :return: A list of ``{params, mel_spec, audio}`` dicts ready for collate.
        :rtype: list[dict]
        """
        return [
            {
                "params": torch.full((4,), float(i)),
                "mel_spec": torch.full((2, 8), float(i)),
                "audio": torch.full((10,), float(i)) if audio else None,
            }
            for i in range(num_items)
        ]

    def test_returns_dict_with_required_keys(self) -> None:
        """Output dict carries exactly ``{params, noise, mel_spec, audio}``."""
        torch.manual_seed(0)
        out = _collate_dict(self._make_batch())
        assert set(out.keys()) == {"params", "noise", "mel_spec", "audio"}

    def test_stacks_params_and_mel_spec_along_new_dim_0(self) -> None:
        """3 items of per-item ``params(4,)`` ⇒ output ``params(3, 4)``."""
        torch.manual_seed(0)
        out = _collate_dict(self._make_batch(num_items=3))
        params = out["params"]
        mel_spec = out["mel_spec"]
        assert params is not None and mel_spec is not None
        assert params.shape == (3, 4)
        assert mel_spec.shape == (3, 2, 8)

    def test_audio_stacked_when_first_item_has_it(self) -> None:
        """``audio[0] is not None`` ⇒ ``audio`` is stacked; ``None`` is the fall-through."""
        torch.manual_seed(0)
        out = _collate_dict(self._make_batch(num_items=3, audio=True))
        assert out["audio"] is not None
        assert out["audio"].shape == (3, 10)

    def test_audio_none_when_first_item_is_none(self) -> None:
        """``audio[0] is None`` ⇒ output ``audio`` is ``None``."""
        torch.manual_seed(0)
        out = _collate_dict(self._make_batch(num_items=3, audio=False))
        assert out["audio"] is None

    def test_noise_shape_matches_params(self) -> None:
        """``noise = torch.randn_like(params)`` ⇒ identical shape/dtype."""
        torch.manual_seed(0)
        out = _collate_dict(self._make_batch())
        noise = out["noise"]
        params = out["params"]
        assert noise is not None and params is not None
        assert noise.shape == params.shape
        assert noise.dtype == params.dtype


# --------------------------------------------------------------------------- #
# regular_collate_fn                                                          #
# --------------------------------------------------------------------------- #


class TestRegularCollateFn:
    """Type-dispatched collate: tuple/list go to ``_collate_tuple``, dict to ``_collate_dict``."""

    def test_dispatch_tuple_item_to_collate_tuple(self) -> None:
        """A batch of tuples returns the 4-tuple shape of ``_collate_tuple``."""
        torch.manual_seed(0)
        batch = [
            (torch.zeros(2, 5), torch.zeros(2, 4), "fake_fn"),
            (torch.zeros(2, 5), torch.zeros(2, 4), "fake_fn"),
        ]
        out = regular_collate_fn(batch)
        assert isinstance(out, tuple)
        assert len(out) == 4

    def test_dispatch_list_item_to_collate_tuple(self) -> None:
        """A batch of *lists* (not tuples) also routes through ``_collate_tuple``."""
        torch.manual_seed(0)
        batch = [
            [torch.zeros(2, 5), torch.zeros(2, 4), "fake_fn"],
            [torch.zeros(2, 5), torch.zeros(2, 4), "fake_fn"],
        ]
        out = regular_collate_fn(batch)
        assert isinstance(out, tuple)
        assert len(out) == 4

    def test_dispatch_dict_item_to_collate_dict(self) -> None:
        """A batch of dicts returns a dict from ``_collate_dict``."""
        torch.manual_seed(0)
        batch = [
            {"params": torch.zeros(4), "mel_spec": torch.zeros(2, 8), "audio": None},
        ] * 2
        out = regular_collate_fn(batch)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"params", "noise", "mel_spec", "audio"}

    def test_raises_not_implemented_for_unknown_item_type(self) -> None:
        """A batch of e.g. raw tensors (not tuple/list/dict) raises ``NotImplementedError``."""
        batch = [torch.zeros(4), torch.zeros(4)]
        with pytest.raises(NotImplementedError, match="Expected tuple or dict"):
            regular_collate_fn(batch)


# --------------------------------------------------------------------------- #
# _ot_collate_tuple                                                           #
# --------------------------------------------------------------------------- #


class TestOtCollateTuple:
    """``_collate_tuple`` + ``_hungarian_match`` reordering."""

    @staticmethod
    def _make_batch() -> list[tuple]:
        """Return a small tuple batch for OT collation tests.

        :return: Three ``(sins, params, sin_fn)`` tuples with distinct rows.
        :rtype: list[tuple]
        """
        return [
            (torch.tensor([[1.0, 2.0]]), torch.tensor([[10.0, 0.0]]), "fn"),
            (torch.tensor([[3.0, 4.0]]), torch.tensor([[0.0, 10.0]]), "fn"),
            (torch.tensor([[5.0, 6.0]]), torch.tensor([[-10.0, 0.0]]), "fn"),
        ]

    def test_returns_four_element_tuple(self) -> None:
        """OT variant preserves the regular collator's 4-tuple shape."""
        torch.manual_seed(0)
        out = _ot_collate_tuple(self._make_batch())
        assert isinstance(out, tuple)
        assert len(out) == 4

    def test_routes_through_hungarian_match(self) -> None:
        """``_hungarian_match`` is invoked exactly once per OT collate call."""
        with patch(
            "synth_setter.data.ot._hungarian_match",
            side_effect=lambda noise, params, *args: (noise, params, *args),
        ) as mock_match:
            torch.manual_seed(0)
            _ot_collate_tuple(self._make_batch())
        mock_match.assert_called_once()

    def test_hungarian_match_called_with_noise_params_sins(self) -> None:
        """The positional contract is ``(noise, params, sins)``."""
        with patch(
            "synth_setter.data.ot._hungarian_match",
            side_effect=lambda noise, params, *args: (noise, params, *args),
        ) as mock_match:
            torch.manual_seed(0)
            _ot_collate_tuple(self._make_batch())
        positional = mock_match.call_args.args
        assert len(positional) == 3


# --------------------------------------------------------------------------- #
# _ot_collate_dict                                                            #
# --------------------------------------------------------------------------- #


class TestOtCollateDict:
    """``_collate_dict`` + ``_hungarian_match`` over noise/params/mel_spec/audio."""

    @staticmethod
    def _make_batch(audio: bool = True) -> list[dict]:
        """Return a small dict batch for OT collation tests.

        :param audio: When False, every per-item dict has ``audio=None``.

        :return: A list of ``{params, mel_spec, audio}`` dicts.
        :rtype: list[dict]
        """
        return [
            {
                "params": torch.tensor([10.0, 0.0]),
                "mel_spec": torch.full((2, 4), 1.0),
                "audio": torch.full((6,), 1.0) if audio else None,
            },
            {
                "params": torch.tensor([0.0, 10.0]),
                "mel_spec": torch.full((2, 4), 2.0),
                "audio": torch.full((6,), 2.0) if audio else None,
            },
        ]

    def test_returns_dict_with_required_keys(self) -> None:
        """Output dict has ``{params, noise, mel_spec, audio}`` regardless of OT reorder."""
        torch.manual_seed(0)
        out = _ot_collate_dict(self._make_batch())
        assert set(out.keys()) == {"params", "noise", "mel_spec", "audio"}

    def test_routes_through_hungarian_match(self) -> None:
        """``_hungarian_match`` is invoked exactly once per OT-dict collate call."""
        with patch(
            "synth_setter.data.ot._hungarian_match",
            side_effect=lambda noise, params, *args: (noise, params, *args),
        ) as mock_match:
            torch.manual_seed(0)
            _ot_collate_dict(self._make_batch())
        mock_match.assert_called_once()

    def test_hungarian_match_called_with_four_positional_args(self) -> None:
        """Positional contract is ``(noise, params, mel_spec, audio)``."""
        with patch(
            "synth_setter.data.ot._hungarian_match",
            side_effect=lambda noise, params, *args: (noise, params, *args),
        ) as mock_match:
            torch.manual_seed(0)
            _ot_collate_dict(self._make_batch())
        positional = mock_match.call_args.args
        assert len(positional) == 4

    def test_audio_none_passes_through_hungarian_match(self) -> None:
        """When the batch has ``audio=None``, the OT path returns ``audio=None``."""
        torch.manual_seed(0)
        out = _ot_collate_dict(self._make_batch(audio=False))
        assert out["audio"] is None


# --------------------------------------------------------------------------- #
# ot_collate_fn                                                               #
# --------------------------------------------------------------------------- #


class TestOtCollateFn:
    """OT dispatcher: tuple/list → ``_ot_collate_tuple``, dict → ``_ot_collate_dict``."""

    def test_dispatch_tuple_item_to_ot_tuple(self) -> None:
        """Tuple items route through the OT tuple path and return a 4-tuple."""
        torch.manual_seed(0)
        batch = [
            (torch.tensor([[1.0]]), torch.tensor([[1.0]]), "fn"),
            (torch.tensor([[2.0]]), torch.tensor([[2.0]]), "fn"),
        ]
        out = ot_collate_fn(batch)
        assert isinstance(out, tuple)
        assert len(out) == 4

    def test_dispatch_list_item_to_ot_tuple(self) -> None:
        """List items (not tuples) also route through the OT tuple path."""
        torch.manual_seed(0)
        batch = [
            [torch.tensor([[1.0]]), torch.tensor([[1.0]]), "fn"],
            [torch.tensor([[2.0]]), torch.tensor([[2.0]]), "fn"],
        ]
        out = ot_collate_fn(batch)
        assert isinstance(out, tuple)
        assert len(out) == 4

    def test_dispatch_dict_item_to_ot_dict(self) -> None:
        """Dict items route through the OT dict path and return a 4-key dict."""
        torch.manual_seed(0)
        batch = [
            {"params": torch.tensor([1.0]), "mel_spec": torch.zeros(2, 4), "audio": None},
            {"params": torch.tensor([2.0]), "mel_spec": torch.zeros(2, 4), "audio": None},
        ]
        out = ot_collate_fn(batch)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"params", "noise", "mel_spec", "audio"}

    def test_raises_not_implemented_for_unknown_item_type(self) -> None:
        """Raw-tensor items raise ``NotImplementedError`` — same contract as regular_collate_fn."""
        batch = [torch.zeros(4), torch.zeros(4)]
        with pytest.raises(NotImplementedError, match="Expected tuple or dict"):
            ot_collate_fn(batch)
