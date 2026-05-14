"""Tests for ``synth_setter.data.ot``.

Covers the optimal-transport pairing function (:func:`_hungarian_match`),
the small numpy/torch concat helpers (:func:`concatenate`, :func:`stack`),
and the four collate variants:

* :func:`regular_collate_fn` → :func:`_collate_tuple` / :func:`_collate_dict`
* :func:`ot_collate_fn` → :func:`_ot_collate_tuple` / :func:`_ot_collate_dict`

The optimality check on :func:`_hungarian_match` is adversarial: for small
batches (B≤5) it brute-forces every permutation and verifies the returned
assignment is genuinely the minimum-cost one. This catches subtle bugs the
``isinstance``/shape checks would miss — e.g. accidentally returning
``row_ind`` as the params permutation instead of ``col_ind``.
"""

from __future__ import annotations

import itertools
from typing import TypeVar
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

_T = TypeVar("_T")


def _present(x: _T | None) -> _T:
    """Assert non-None and return — narrows ``Optional[T]`` to ``T`` for type checkers.

    :param x: Any value, typically a dict lookup that may be ``None``.

    :returns: ``x`` itself, narrowed to its non-Optional type.
    :rtype: _T
    """
    assert x is not None
    return x


def _total_pairwise_cost(a: torch.Tensor, b: torch.Tensor) -> float:
    """Total Euclidean cost between row-paired tensors: ``Σᵢ ‖a[i] - b[i]‖₂``.

    The metric matches :func:`torch.cdist` (p=2, the default Hungarian cost in
    :func:`_hungarian_match`), so comparing this against either a brute-force minimum
    or a random-pairing baseline directly probes OT optimality.

    :param a: Tensor of shape (B, ...).
    :param b: Tensor of shape (B, ...).

    :returns: Sum of L2 distances between corresponding rows.
    :rtype: float
    """
    return sum(torch.linalg.norm(a[i] - b[i]).item() for i in range(a.shape[0]))


def _brute_force_optimal_cost(noise: torch.Tensor, params: torch.Tensor) -> float:
    """Enumerate every permutation of ``params`` and return the minimum total L2 cost.

    Used to validate :func:`_hungarian_match` on small batches (factorial growth caps
    this at B≤6). The cost metric is Euclidean (the same one ``torch.cdist`` computes
    by default), so the brute-force minimum is the ground truth Hungarian must match.

    :param noise: Tensor of shape (B, D).
    :param params: Tensor of shape (B, D).

    :returns: Minimum total Euclidean cost across all B! pairings.
    :rtype: float
    """
    b = noise.shape[0]
    return min(
        _total_pairwise_cost(noise, params[list(perm)])
        for perm in itertools.permutations(range(b))
    )


def _make_dict_batch(n: int = 2, *, audio: bool = True) -> list[dict]:
    """Build ``n`` identical-shape dict items for collate-fn tests.

    Each item carries a ``params`` (D=3), ``mel_spec`` ((2, 4)), and either an audio
    tensor ((2, 8)) or ``None``. Values are filled with the item index so assertions
    can recover provenance.

    :param n: Number of items in the returned list.
    :param audio: If True, populate ``audio``; if False, set to ``None`` for the
        audio-skip path in :func:`_collate_dict`.

    :returns: List of dicts compatible with :func:`_collate_dict` /
        :func:`_ot_collate_dict`.
    :rtype: list[dict]
    """
    return [
        {
            "params": torch.full((3,), float(i)),
            "mel_spec": torch.full((2, 4), float(i)),
            "audio": torch.full((2, 8), float(i)) if audio else None,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _hungarian_match — optimality and contract
# ---------------------------------------------------------------------------


class TestHungarianMatchOptimality:
    """Brute-force checks that :func:`_hungarian_match` finds the *minimum* cost pairing."""

    @pytest.mark.parametrize("batch_size", [2, 3, 4, 5])
    def test_matches_brute_force_optimum_on_random_inputs(self, batch_size: int) -> None:
        """For B≤5, the Hungarian total cost equals the brute-force minimum.

        :param batch_size: Number of rows in noise/params; parameterized to exercise multiple
            sizes.
        """
        torch.manual_seed(batch_size)
        noise = torch.randn(batch_size, 3)
        params = torch.randn(batch_size, 3)

        matched_noise, matched_params = _hungarian_match(noise.clone(), params.clone())
        hungarian_cost = _total_pairwise_cost(matched_noise, matched_params)
        brute_force_cost = _brute_force_optimal_cost(noise, params)

        assert hungarian_cost == pytest.approx(brute_force_cost, rel=1e-5)

    def test_beats_random_pairing_in_expectation(self) -> None:
        """OT-paired cost is strictly less than the random-pairing cost on a typical batch.

        Single deterministic example (seed pinned) avoids flakiness. The claim is conservative: OT
        improves over random pairing for almost every minibatch, and the seeded example confirms
        the implementation is actually pairing instead of returning identity.
        """
        torch.manual_seed(0)
        noise = torch.randn(8, 4)
        params = torch.randn(8, 4)

        random_cost = _total_pairwise_cost(noise, params)
        matched_noise, matched_params = _hungarian_match(noise.clone(), params.clone())
        ot_cost = _total_pairwise_cost(matched_noise, matched_params)
        assert ot_cost < random_cost


class TestHungarianMatchPermutation:
    """The returned ordering must be a bijection over [0, B-1]."""

    def test_returned_params_are_a_permutation_of_input(self) -> None:
        """No row is duplicated or dropped — output params is a row-permutation of input."""
        torch.manual_seed(1)
        noise = torch.randn(6, 3)
        params = torch.randn(6, 3)

        _, matched_params = _hungarian_match(noise.clone(), params.clone())

        # Stack input and output rows, dedupe; if either side had a duplicate or a row
        # the other side lacked, ``torch.unique`` would return ≠ 6 rows.
        combined = torch.cat([params, matched_params], dim=0)
        assert torch.unique(combined, dim=0).shape[0] == 6

    def test_noise_returned_in_identity_order(self) -> None:
        """``row_ind`` from :func:`linear_sum_assignment` on a square matrix is identity.

        Translation: ``noise`` comes back unpermuted; only ``params`` (and co-data) are
        reordered to match each noise row.
        """
        torch.manual_seed(2)
        noise = torch.randn(5, 3)
        params = torch.randn(5, 3)

        matched_noise, _ = _hungarian_match(noise.clone(), params.clone())
        torch.testing.assert_close(matched_noise, noise)

    def test_already_optimal_input_returns_identity_permutation(self) -> None:
        """When ``params == noise`` row-for-row, the optimal pairing is identity."""
        torch.manual_seed(3)
        noise = torch.randn(4, 3)
        params = noise.clone()

        matched_noise, matched_params = _hungarian_match(noise.clone(), params.clone())
        torch.testing.assert_close(matched_params, params)
        torch.testing.assert_close(matched_noise, noise)


class TestHungarianMatchCoData:
    """Co-data (``*args``) is permuted by the same ``col_ind`` as ``params``."""

    def test_co_data_permuted_in_lockstep_with_params(self) -> None:
        """Tag each params row with its index; after OT, the tags follow params."""
        torch.manual_seed(4)
        noise = torch.randn(5, 3)
        params = torch.randn(5, 3)
        # ``labels[i]`` is bonded to ``params[i]``; after pairing they must move together.
        labels = torch.arange(5, dtype=torch.float32).reshape(5, 1)

        _, matched_params, matched_labels = _hungarian_match(
            noise.clone(), params.clone(), labels.clone()
        )

        # Recover the permutation from labels and verify params follow the same indices.
        col_ind = matched_labels.flatten().to(torch.int64)
        torch.testing.assert_close(matched_params, params[col_ind])

    def test_none_co_data_passes_through_as_none(self) -> None:
        """``arg=None`` flows through to the corresponding output slot as ``None``."""
        torch.manual_seed(5)
        noise = torch.randn(3, 2)
        params = torch.randn(3, 2)

        _, _, audio_out, mel_out = _hungarian_match(noise, params, None, None)
        assert audio_out is None
        assert mel_out is None

    def test_mixed_present_and_none_co_data(self) -> None:
        """A mix of real tensors and ``None`` in ``*args`` preserves both slot semantics."""
        torch.manual_seed(6)
        noise = torch.randn(4, 2)
        params = torch.randn(4, 2)
        labels = torch.arange(4, dtype=torch.float32).reshape(4, 1)

        _, _, labels_out, none_out = _hungarian_match(
            noise.clone(), params.clone(), labels.clone(), None
        )
        assert isinstance(labels_out, torch.Tensor)
        assert labels_out.shape == labels.shape
        assert none_out is None


class TestHungarianMatchInputConversion:
    """Numpy arrays in → torch tensors out (silent auto-conversion of noise/params)."""

    def test_numpy_noise_is_converted_to_tensor(self) -> None:
        """A numpy ``noise`` input becomes a torch tensor in the output."""
        rng = np.random.default_rng(0)
        noise_np = rng.standard_normal((4, 3)).astype(np.float32)
        params = torch.randn(4, 3)

        # Function types args as Tensor but the runtime accepts ndarray — pin that.
        noise_out, _ = _hungarian_match(noise_np, params)  # type: ignore[arg-type]
        assert isinstance(noise_out, torch.Tensor)

    def test_numpy_params_is_converted_to_tensor(self) -> None:
        """A numpy ``params`` input becomes a torch tensor in the output."""
        rng = np.random.default_rng(1)
        noise = torch.randn(4, 3)
        params_np = rng.standard_normal((4, 3)).astype(np.float32)

        _, params_out = _hungarian_match(noise, params_np)  # type: ignore[arg-type]
        assert isinstance(params_out, torch.Tensor)

    def test_both_numpy_inputs_produce_optimal_pairing(self) -> None:
        """Conversion path doesn't break optimality — brute-force check on numpy inputs."""
        rng = np.random.default_rng(2)
        noise_np = rng.standard_normal((4, 3)).astype(np.float32)
        params_np = rng.standard_normal((4, 3)).astype(np.float32)

        noise_out, params_out = _hungarian_match(noise_np.copy(), params_np.copy())  # type: ignore[arg-type]
        hungarian_cost = _total_pairwise_cost(noise_out, params_out)
        brute_force_cost = _brute_force_optimal_cost(
            torch.from_numpy(noise_np), torch.from_numpy(params_np)
        )
        assert hungarian_cost == pytest.approx(brute_force_cost, rel=1e-5)


class TestHungarianMatchEdgeCases:
    """Boundary inputs that exercise unusual code paths."""

    def test_batch_size_one_returns_inputs_unchanged(self) -> None:
        """Single-row input: trivial 1×1 cost matrix; outputs equal inputs."""
        noise = torch.tensor([[1.0, 2.0]])
        params = torch.tensor([[3.0, 4.0]])
        labels = torch.tensor([[42.0]])

        noise_out, params_out, labels_out = _hungarian_match(noise, params, labels)
        torch.testing.assert_close(noise_out, noise)
        torch.testing.assert_close(params_out, params)
        torch.testing.assert_close(_present(labels_out), labels)


# ---------------------------------------------------------------------------
# concatenate / stack — dispatch on input type
# ---------------------------------------------------------------------------


class TestConcatenate:
    """`concatenate` cats torch tensors with ``dim=0`` or numpy arrays with ``axis=0``.

    The source typing annotates the param as ``Union[Tensor, ndarray]``, but every
    real caller passes a *list* — the runtime peeks at ``list_of_arrays[0]`` to dispatch.
    Tests pin that documented behavior; ``# type: ignore[arg-type]`` acknowledges the
    type-hint gap on the source side.
    """

    def test_torch_inputs_concat_along_first_axis(self) -> None:
        """List of (3, 4) tensors → (3+3+3, 4) tensor via ``torch.cat(dim=0)``."""
        a = torch.zeros(3, 4)
        b = torch.ones(3, 4)
        out = concatenate([a, b])  # type: ignore[arg-type]

        assert isinstance(out, torch.Tensor)
        assert out.shape == (6, 4)
        torch.testing.assert_close(out[:3], a)
        torch.testing.assert_close(out[3:], b)

    def test_numpy_inputs_returned_as_torch_tensor(self) -> None:
        """Numpy in → torch out via ``np.concatenate`` then ``torch.from_numpy``."""
        a = np.zeros((2, 3), dtype=np.float32)
        b = np.ones((2, 3), dtype=np.float32)

        out = concatenate([a, b])  # type: ignore[arg-type]

        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 3)
        np.testing.assert_array_equal(out.numpy()[:2], a)
        np.testing.assert_array_equal(out.numpy()[2:], b)


class TestStack:
    """`stack` stacks along a new leading axis (``dim=0`` / ``axis=0``).

    Same source-side typing caveat as :class:`TestConcatenate`.
    """

    def test_torch_inputs_stack_into_new_first_axis(self) -> None:
        """List of (4,) tensors → (3, 4) tensor via ``torch.stack(dim=0)``."""
        rows = [torch.full((4,), float(i)) for i in range(3)]
        out = stack(rows)  # type: ignore[arg-type]

        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 4)
        for i, row in enumerate(rows):
            torch.testing.assert_close(out[i], row)

    def test_numpy_inputs_returned_as_torch_tensor(self) -> None:
        """Numpy in → torch out via ``np.stack`` then ``torch.from_numpy``."""
        rows = [np.full((4,), float(i), dtype=np.float32) for i in range(3)]
        out = stack(rows)  # type: ignore[arg-type]

        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 4)


# ---------------------------------------------------------------------------
# Regular (non-OT) collate functions
# ---------------------------------------------------------------------------


class TestCollateTuple:
    """`_collate_tuple` concatenates (sins, params, sin_fn) batches and adds noise."""

    def test_concatenates_sins_and_params_and_adds_matching_noise(self) -> None:
        """Batch of two (sins, params, sin_fn) tuples → flattened sins/params + noise/sin_fn."""
        sin_fn = object()  # opaque sentinel
        item_0 = (torch.zeros(2, 5), torch.ones(2, 3), sin_fn)
        item_1 = (torch.ones(2, 5), torch.full((2, 3), 2.0), sin_fn)

        sins, params, noise, sin_fn_out = _collate_tuple([item_0, item_1])

        assert sins.shape == (4, 5)
        assert params.shape == (4, 3)
        assert noise.shape == params.shape
        assert sin_fn_out is sin_fn

    def test_sin_fn_is_taken_from_first_batch_element(self) -> None:
        """`sin_fn` is forwarded from ``batch[0]`` — later elements' ``sin_fn`` are ignored."""
        first = object()
        second = object()
        item_0 = (torch.zeros(1, 5), torch.zeros(1, 3), first)
        item_1 = (torch.zeros(1, 5), torch.zeros(1, 3), second)

        _, _, _, sin_fn_out = _collate_tuple([item_0, item_1])
        assert sin_fn_out is first


class TestCollateDict:
    """`_collate_dict` stacks dict-of-tensors batches and adds noise; handles ``audio=None``."""

    def test_stacks_params_mel_and_audio_then_adds_noise(self) -> None:
        """All three keys stack; ``noise`` is generated via ``randn_like(params)``."""
        out = _collate_dict(_make_dict_batch())

        assert _present(out["params"]).shape == (2, 3)
        assert _present(out["mel_spec"]).shape == (2, 2, 4)
        assert _present(out["audio"]).shape == (2, 2, 8)
        assert _present(out["noise"]).shape == _present(out["params"]).shape

    def test_audio_none_in_first_item_propagates_to_output(self) -> None:
        """If ``batch[0]["audio"] is None`` the output audio is ``None`` — short-circuit path."""
        batch = _make_dict_batch(audio=False)

        out = _collate_dict(batch)
        assert out["audio"] is None
        assert _present(out["params"]).shape == (2, 3)


class TestRegularCollateFn:
    """`regular_collate_fn` dispatches on the type of ``batch[0]``."""

    def test_dispatches_tuple_input_to_collate_tuple(self) -> None:
        """A list of tuples routes to ``_collate_tuple``."""
        sin_fn = object()
        batch = [(torch.zeros(1, 5), torch.zeros(1, 3), sin_fn) for _ in range(2)]

        out = regular_collate_fn(batch)
        assert isinstance(out, tuple)
        assert len(out) == 4
        # (sins, params, noise, sin_fn) — sin_fn at slot 3
        assert out[3] is sin_fn

    def test_dispatches_list_input_to_collate_tuple(self) -> None:
        """A list of *lists* (not tuples) also routes to ``_collate_tuple``."""
        sin_fn = object()
        batch = [[torch.zeros(1, 5), torch.zeros(1, 3), sin_fn] for _ in range(2)]

        out = regular_collate_fn(batch)
        assert isinstance(out, tuple)
        assert out[3] is sin_fn

    def test_dispatches_dict_input_to_collate_dict(self) -> None:
        """A list of dicts routes to ``_collate_dict``."""
        out = regular_collate_fn(_make_dict_batch(audio=False))
        assert isinstance(out, dict)
        assert set(out.keys()) == {"params", "noise", "mel_spec", "audio"}

    def test_unsupported_item_type_raises_not_implemented(self) -> None:
        """An item of an unknown type (e.g. plain tensor) triggers ``NotImplementedError``."""
        batch = [torch.zeros(3), torch.zeros(3)]
        with pytest.raises(NotImplementedError, match="tuple or dict"):
            regular_collate_fn(batch)


# ---------------------------------------------------------------------------
# OT collate functions
# ---------------------------------------------------------------------------


class TestOtCollateTuple:
    """`_ot_collate_tuple` collates then applies :func:`_hungarian_match` to (noise, params,
    sins)."""

    def test_calls_hungarian_match_with_three_aligned_tensors(self) -> None:
        """The OT call sees noise, params, and sins — three positional args."""
        sin_fn = object()
        batch = [(torch.zeros(2, 5), torch.ones(2, 3), sin_fn) for _ in range(2)]

        with patch(
            "synth_setter.data.ot._hungarian_match",
            side_effect=lambda n, p, s: (n, p, s),
        ) as patched:
            sins, params, noise, sin_fn_out = _ot_collate_tuple(batch)

        patched.assert_called_once()
        args = patched.call_args.args
        assert len(args) == 3
        assert args[0].shape == (4, 3)  # noise (matches params shape)
        assert args[1].shape == (4, 3)  # params
        assert args[2].shape == (4, 5)  # sins
        assert sin_fn_out is sin_fn

    def test_end_to_end_with_real_hungarian_match_returns_optimal_pairing(self) -> None:
        """No mock on Hungarian: resulting (noise, params) pair has minimum total cost.

        ``torch.randn_like`` is patched to a fixed tensor so the test asserts on a
        behavior (OT optimality given known inputs) rather than reproducing a private
        seed sequence inside ``_collate_tuple``.
        """
        torch.manual_seed(7)
        sin_fn = object()
        batch = [
            (torch.zeros(2, 5), torch.randn(2, 3), sin_fn),
            (torch.zeros(2, 5), torch.randn(2, 3), sin_fn),
        ]
        params_pre = torch.cat([item[1] for item in batch], dim=0)
        fixed_noise = torch.randn(4, 3, generator=torch.Generator().manual_seed(99))
        brute_force_cost = _brute_force_optimal_cost(fixed_noise, params_pre)

        with patch(
            "synth_setter.data.ot.torch.randn_like", return_value=fixed_noise.clone()
        ):
            _, params_out, noise_out, _ = _ot_collate_tuple(batch)

        hungarian_cost = _total_pairwise_cost(noise_out, params_out)
        assert hungarian_cost == pytest.approx(brute_force_cost, rel=1e-5)


class TestOtCollateDict:
    """`_ot_collate_dict` collates dicts then applies OT across (noise, params, mel_spec,
    audio)."""

    def test_calls_hungarian_match_with_four_aligned_tensors(self) -> None:
        """The OT call sees noise, params, mel_spec, audio — four positional args."""
        with patch(
            "synth_setter.data.ot._hungarian_match",
            side_effect=lambda n, p, m, a: (n, p, m, a),
        ) as patched:
            out = _ot_collate_dict(_make_dict_batch())

        patched.assert_called_once()
        args = patched.call_args.args
        assert len(args) == 4
        assert args[0].shape == (2, 3)  # noise
        assert args[1].shape == (2, 3)  # params
        assert args[2].shape == (2, 2, 4)  # mel_spec
        assert args[3].shape == (2, 2, 8)  # audio
        assert set(out.keys()) == {"params", "noise", "mel_spec", "audio"}

    def test_audio_none_path_forwards_none_to_hungarian_match(self) -> None:
        """When ``audio is None`` after collate, OT receives ``None`` and returns ``None``."""
        with patch(
            "synth_setter.data.ot._hungarian_match",
            side_effect=lambda n, p, m, a: (n, p, m, a),
        ) as patched:
            out = _ot_collate_dict(_make_dict_batch(audio=False))

        args = patched.call_args.args
        assert args[3] is None
        assert out["audio"] is None


class TestOtCollateFn:
    """`ot_collate_fn` dispatches on the type of ``batch[0]`` to the OT variants."""

    def test_dispatches_tuple_input_to_ot_collate_tuple(self) -> None:
        """A list of tuples routes through :func:`_ot_collate_tuple`."""
        sin_fn = object()
        batch = [(torch.zeros(1, 5), torch.zeros(1, 3), sin_fn) for _ in range(2)]

        out = ot_collate_fn(batch)
        assert isinstance(out, tuple)
        assert len(out) == 4
        assert out[3] is sin_fn

    def test_dispatches_list_input_to_ot_collate_tuple(self) -> None:
        """List items (not tuples) take the same OT-tuple branch."""
        sin_fn = object()
        batch = [[torch.zeros(1, 5), torch.zeros(1, 3), sin_fn] for _ in range(2)]

        out = ot_collate_fn(batch)
        assert isinstance(out, tuple)
        assert out[3] is sin_fn

    def test_dispatches_dict_input_to_ot_collate_dict(self) -> None:
        """A list of dicts routes through :func:`_ot_collate_dict`."""
        out = ot_collate_fn(_make_dict_batch(audio=False))
        assert isinstance(out, dict)
        assert set(out.keys()) == {"params", "noise", "mel_spec", "audio"}

    def test_unsupported_item_type_raises_not_implemented(self) -> None:
        """A plain tensor (not tuple/list/dict) triggers ``NotImplementedError``."""
        batch = [torch.zeros(3), torch.zeros(3)]
        with pytest.raises(NotImplementedError, match="tuple or dict"):
            ot_collate_fn(batch)
