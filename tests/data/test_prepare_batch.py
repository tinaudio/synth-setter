"""Pinning tests for :func:`synth_setter.data.vst_datamodule.prepare_batch`.

Each test compares against :func:`_reference_prepare_batch`, an independent
golden that applies the batch math (mel norm, param rescale, seeded noise,
Hungarian match) drawing noise via ``torch.randn`` on its own seeded
``torch.Generator``. ``prepare_batch`` draws via ``empty_like().normal_()``;
the two APIs (and the seeded global RNG) are bit-identical on CPU, and
:func:`test_noise_draw_apis_same_seed_bit_identical` guards that equivalence
loudly so a PyTorch upgrade that breaks it fails one test, not every golden.
"""

from __future__ import annotations

import itertools
from typing import cast

import numpy as np
import pytest
import torch

from synth_setter.data.ot import _hungarian_match
from synth_setter.data.vst_datamodule import RawBatch, prepare_batch

_NUM_PARAMS = 5
_BATCH = 8
_MEL_SHAPE = (_BATCH, 2, 4, 6)
_AUDIO_SHAPE = (_BATCH, 2, 16)
_M2L_SHAPE = (_BATCH, 3, 7)


def _unwrap(maybe_tensor: torch.Tensor | None) -> torch.Tensor:
    """Assert ``maybe_tensor`` is non-``None`` and narrow it for pyright.

    :param maybe_tensor: The dict value to narrow.
    :returns: The same tensor, typed as non-``None``.
    """
    assert maybe_tensor is not None
    return maybe_tensor


def _unwrap_array(maybe_array: np.ndarray | None) -> np.ndarray:
    """Assert ``maybe_array`` is non-``None`` and narrow it for pyright.

    :param maybe_array: The ``raw`` dict value to narrow.
    :returns: The same array, typed as non-``None``.
    """
    assert maybe_array is not None
    return maybe_array


def _reference_prepare_batch(
    raw: RawBatch,
    *,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    rescale_params: bool,
    ot: bool,
    seed: int,
) -> dict[str, torch.Tensor | None]:
    """Apply the batch math independently of ``prepare_batch`` as the golden.

    Draws noise via ``torch.randn`` on a fresh generator seeded by ``seed``;
    ``prepare_batch`` under a same-seeded generator reproduces this bit-for-bit
    (equivalence guarded by :func:`test_noise_draw_apis_same_seed_bit_identical`).

    :param raw: Read shard columns; see :class:`RawBatch`.
    :param mean: Mel mean, or ``None`` to skip normalization.
    :param std: Mel std, or ``None`` to skip normalization.
    :param rescale_params: Whether to map params ``[0, 1] -> [-1, 1]``.
    :param ot: Whether to Hungarian-match noise to params.
    :param seed: Seed for the golden's own noise generator.
    :returns: ``{"mel_spec", "m2l", "params", "noise", "audio"}`` tensors.
    """
    audio_raw = raw.get("audio")
    if audio_raw is not None:
        audio = torch.from_numpy(audio_raw).to(dtype=torch.float32)
    else:
        audio = None

    mel_raw = raw.get("mel_spec")
    if mel_raw is not None:
        mel_spec = mel_raw
        if mean is not None and std is not None:
            mel_spec = (mel_spec - mean) / std
        mel_spec = torch.from_numpy(mel_spec).to(dtype=torch.float32)
    else:
        mel_spec = None

    m2l_raw = raw.get("music2latent")
    if m2l_raw is not None:
        m2l = torch.from_numpy(m2l_raw).to(dtype=torch.float32)
    else:
        m2l = None

    param_raw = _unwrap_array(raw["param_array"])
    if rescale_params:
        param_raw = param_raw * 2 - 1
    param_array = torch.from_numpy(param_raw).to(dtype=torch.float32)
    noise = torch.randn(param_array.shape, generator=torch.Generator().manual_seed(seed))
    if ot:
        noise, param_array, mel_spec, m2l, audio = _hungarian_match(
            noise, param_array, mel_spec, m2l, audio
        )

    return dict(
        mel_spec=mel_spec.contiguous() if mel_spec is not None else None,
        m2l=m2l.contiguous() if m2l is not None else None,
        params=param_array.contiguous(),
        noise=noise.contiguous(),
        audio=audio.contiguous() if audio is not None else None,
    )


def _make_raw(
    *,
    read_mel: bool = True,
    read_m2l: bool = False,
    read_audio: bool = False,
    seed: int = 7,
) -> RawBatch:
    """Build a deterministic ``raw`` batch for the given read flags.

    :param read_mel: Whether to include a ``mel_spec`` array.
    :param read_m2l: Whether to include a ``music2latent`` array.
    :param read_audio: Whether to include an ``audio`` array.
    :param seed: Seed for the NumPy generator backing every array.
    :returns: ``raw`` batch with ``param_array`` always present.
    """
    rng = np.random.default_rng(seed)
    raw: RawBatch = {
        "param_array": rng.random((_BATCH, _NUM_PARAMS)).astype(np.float32),
    }
    if read_mel:
        raw["mel_spec"] = rng.standard_normal(_MEL_SHAPE).astype(np.float32)
    if read_m2l:
        raw["music2latent"] = rng.standard_normal(_M2L_SHAPE).astype(np.float32)
    if read_audio:
        raw["audio"] = rng.uniform(-1.0, 1.0, _AUDIO_SHAPE).astype(np.float32)
    return raw


@pytest.mark.parametrize("column", ["param_array", "mel_spec", "music2latent", "audio"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_prepare_batch_nonfinite_column_raises_value_error(
    column: str, value: float
) -> None:
    """Non-finite stored values fail before model-facing transformation.

    :param column: Raw column corrupted with a non-finite value.
    :param value: Non-finite value injected into the raw column.
    """
    raw = _make_raw(read_mel=True, read_m2l=True, read_audio=True)
    arrays = {
        "param_array": raw["param_array"],
        "mel_spec": raw.get("mel_spec"),
        "music2latent": raw.get("music2latent"),
        "audio": raw.get("audio"),
    }
    array = arrays[column]
    assert array is not None
    array.flat[0] = value

    with pytest.raises(ValueError, match=rf"{column} contains non-finite values"):
        prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=True,
            ot=False,
            generator=torch.Generator(),
        )


def test_prepare_batch_normalization_overflow_raises_value_error() -> None:
    """Finite normalization operands may not produce non-finite model input."""
    raw = _make_raw()
    mel = _unwrap_array(raw.get("mel_spec"))
    mean = np.zeros(mel.shape[1:], dtype=np.float32)
    std = np.ones(mel.shape[1:], dtype=np.float32)
    mel.flat[0] = np.finfo(np.float32).max
    mean.flat[0] = -np.finfo(np.float32).max

    with pytest.raises(ValueError, match="normalization produced non-finite values"):
        prepare_batch(
            raw,
            mean=mean,
            std=std,
            rescale_params=True,
            ot=False,
            generator=torch.Generator(),
        )


def test_prepare_batch_float32_cast_overflow_raises_value_error() -> None:
    """Finite normalized values must remain finite at the model-facing dtype."""
    raw = _make_raw()
    mel = _unwrap_array(raw.get("mel_spec")).astype(np.float64)
    mel.flat[0] = float(np.finfo(np.float32).max) * 2
    raw["mel_spec"] = mel

    with pytest.raises(ValueError, match="float32 conversion produced non-finite values"):
        prepare_batch(
            raw,
            mean=np.zeros(mel.shape[1:], dtype=np.float64),
            std=np.ones(mel.shape[1:], dtype=np.float64),
            rescale_params=True,
            ot=False,
            generator=torch.Generator(),
        )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_prepare_batch_parameter_out_of_range_raises_value_error(value: float) -> None:
    """Stored parameters outside their normalized range fail before rescaling.

    :param value: Invalid parameter value injected into the raw batch.
    """
    raw = _make_raw()
    raw["param_array"][0, 0] = value

    with pytest.raises(ValueError, match="param_array values must be within \\[0, 1\\]"):
        prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=True,
            ot=False,
            generator=torch.Generator(),
        )


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_prepare_batch_parameter_range_endpoints_are_valid(value: float) -> None:
    """The normalized parameter interval includes both endpoints.

    :param value: Inclusive endpoint placed in the raw parameter batch.
    """
    raw = _make_raw()
    raw["param_array"][0, 0] = value

    batch = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=False,
        ot=False,
        generator=torch.Generator(),
    )

    assert _unwrap(batch["params"])[0, 0].item() == value


@pytest.mark.parametrize("value", [-1.01, 1.01])
def test_prepare_batch_audio_out_of_range_raises_value_error(value: float) -> None:
    """Stored audio outside full scale fails before tensor conversion.

    :param value: Invalid audio sample injected into the raw batch.
    """
    raw = _make_raw(read_audio=True)
    audio = raw.get("audio")
    assert audio is not None
    audio.flat[0] = value

    with pytest.raises(ValueError, match="audio values must be within \\[-1, 1\\]"):
        prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=False,
            ot=False,
            generator=torch.Generator(),
        )


@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_prepare_batch_audio_range_endpoints_are_valid(value: float) -> None:
    """The full-scale audio interval includes both endpoints.

    :param value: Inclusive endpoint placed in the raw audio batch.
    """
    raw = _make_raw(read_audio=True)
    audio = raw.get("audio")
    assert audio is not None
    audio.flat[0] = value

    batch = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=False,
        ot=False,
        generator=torch.Generator(),
    )

    assert _unwrap(batch["audio"]).flatten()[0].item() == value


def test_prepare_batch_is_pure_and_pinned() -> None:
    """``prepare_batch`` reproduces the frozen pre-refactor golden bit-for-bit."""
    seed = 0
    raw = _make_raw(read_mel=True, read_audio=True, seed=3)
    mean = np.zeros((2, 4, 6), dtype=np.float32)
    std = np.full((2, 4, 6), 2.0, dtype=np.float32)

    golden = _reference_prepare_batch(
        raw, mean=mean, std=std, rescale_params=True, ot=True, seed=seed
    )
    out = prepare_batch(
        raw,
        mean=mean,
        std=std,
        rescale_params=True,
        ot=True,
        generator=torch.Generator().manual_seed(seed),
    )

    for key in ("mel_spec", "params", "noise", "audio"):
        # atol=rtol=0: every step (affine mel norm, x*2-1, seeded noise, integer
        # row permutation from Hungarian) is exact, so bit-equality must hold.
        torch.testing.assert_close(out[key], golden[key], atol=0.0, rtol=0.0)
        assert _unwrap(out[key]).is_contiguous()
        assert _unwrap(out[key]).dtype == torch.float32
    assert out["m2l"] is None


@pytest.mark.parametrize("ot", [False, True])
def test_prepare_batch_same_seed_yields_identical_output(ot: bool) -> None:
    """Two calls with the same seed produce identical outputs (deterministic purity).

    :param ot: Whether to exercise the Hungarian-match path; ``ot=False`` isolates
        determinism from any per-call generator consumption inside the matcher.
    """
    raw = _make_raw(seed=11)
    first = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=True,
        ot=ot,
        generator=torch.Generator().manual_seed(42),
    )
    second = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=True,
        ot=ot,
        generator=torch.Generator().manual_seed(42),
    )
    for key in ("params", "noise", "mel_spec"):
        torch.testing.assert_close(first[key], second[key], atol=0.0, rtol=0.0)
    # The unread slots must stay None across calls, not silently materialize.
    assert first["audio"] is None and second["audio"] is None
    assert first["m2l"] is None and second["m2l"] is None


def test_prepare_batch_does_not_mutate_raw_inputs() -> None:
    """``prepare_batch`` performs no in-place writes to the ``raw`` arrays."""
    raw = _make_raw(read_mel=True, read_audio=True, seed=5)
    snapshot = {
        "param_array": _unwrap_array(raw["param_array"]).copy(),
        "mel_spec": _unwrap_array(raw.get("mel_spec")).copy(),
        "audio": _unwrap_array(raw.get("audio")).copy(),
    }
    prepare_batch(
        raw,
        mean=np.zeros((2, 4, 6), dtype=np.float32),
        std=np.ones((2, 4, 6), dtype=np.float32),
        rescale_params=True,
        ot=True,
        generator=torch.Generator().manual_seed(0),
    )
    np.testing.assert_array_equal(_unwrap_array(raw["param_array"]), snapshot["param_array"])
    np.testing.assert_array_equal(_unwrap_array(raw.get("mel_spec")), snapshot["mel_spec"])
    np.testing.assert_array_equal(_unwrap_array(raw.get("audio")), snapshot["audio"])


@pytest.mark.parametrize(
    ("read_mel", "read_m2l", "read_audio"),
    [(True, False, False), (False, True, False), (True, False, True)],
)
def test_prepare_batch_modality_slots_match_read_flags(
    read_mel: bool, read_m2l: bool, read_audio: bool
) -> None:
    """Each modality slot is non-``None`` exactly when its source array is present.

    :param read_mel: Whether the ``mel_spec`` source array is present.
    :param read_m2l: Whether the ``music2latent`` source array is present.
    :param read_audio: Whether the ``audio`` source array is present.
    """
    raw = _make_raw(read_mel=read_mel, read_m2l=read_m2l, read_audio=read_audio)
    out = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=True,
        ot=False,
        generator=torch.Generator().manual_seed(0),
    )
    assert set(out.keys()) == {
        "mel_spec",
        "m2l",
        "conditioning",
        "params",
        "noise",
        "audio",
    }
    assert out["conditioning"] is None
    assert (out["mel_spec"] is not None) == read_mel
    assert (out["m2l"] is not None) == read_m2l
    assert (out["audio"] is not None) == read_audio
    assert _unwrap(out["params"]).shape == (_BATCH, _NUM_PARAMS)
    assert _unwrap(out["noise"]).shape == (_BATCH, _NUM_PARAMS)


def test_prepare_batch_ot_keeps_m2l_aligned_with_params() -> None:
    """Hungarian matching applies the parameter-row permutation to M2L conditioning."""
    row_ids = np.linspace(0.0, 1.0, _BATCH, dtype=np.float32)
    raw = _make_raw(read_mel=False, read_m2l=True)
    raw["param_array"] = np.repeat(row_ids[:, None], _NUM_PARAMS, axis=1)
    raw["music2latent"] = np.broadcast_to(
        row_ids[:, None, None], _M2L_SHAPE
    ).copy()

    out = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=False,
        ot=True,
        generator=torch.Generator().manual_seed(17),
    )

    params = _unwrap(out["params"])
    m2l = _unwrap(out["m2l"])
    assert torch.equal(m2l[:, 0, 0], params[:, 0])
    assert not torch.equal(params[:, 0], torch.from_numpy(row_ids))


@pytest.mark.parametrize(
    ("mean_set", "std_set"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_prepare_batch_normalizes_mel_only_when_mean_and_std_set(
    mean_set: bool, std_set: bool
) -> None:
    """Mel normalization applies iff both ``mean`` and ``std`` are provided.

    The mixed cases pin the guard's ``and`` semantics: one-sided stats must
    leave the mel untouched, not half-normalize or raise.

    :param mean_set: Whether to pass a mel mean.
    :param std_set: Whether to pass a mel std.
    """
    raw = _make_raw(read_mel=True)
    raw["mel_spec"] = np.full(_MEL_SHAPE, 3.0, dtype=np.float32)

    out = _unwrap(
        prepare_batch(
            raw,
            mean=np.full((2, 4, 6), 1.0, dtype=np.float32) if mean_set else None,
            std=np.full((2, 4, 6), 2.0, dtype=np.float32) if std_set else None,
            rescale_params=False,
            ot=False,
            generator=torch.Generator().manual_seed(0),
        )["mel_spec"]
    )
    expected = (3.0 - 1.0) / 2.0 if (mean_set and std_set) else 3.0
    assert torch.allclose(out, torch.full_like(out, expected))


def test_prepare_batch_rescale_toggle() -> None:
    """``rescale_params`` maps params ``[0, 1] -> [-1, 1]`` exactly as ``x * 2 - 1``."""
    raw = _make_raw(read_mel=False)
    raw_params = torch.from_numpy(_unwrap_array(raw["param_array"])).to(dtype=torch.float32)

    not_rescaled = _unwrap(
        prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=False,
            ot=False,
            generator=torch.Generator().manual_seed(0),
        )["params"]
    )
    assert not_rescaled.min() >= 0.0
    assert not_rescaled.max() <= 1.0
    torch.testing.assert_close(not_rescaled, raw_params, atol=0.0, rtol=0.0)

    rescaled = _unwrap(
        prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=True,
            ot=False,
            generator=torch.Generator().manual_seed(0),
        )["params"]
    )
    assert rescaled.min() >= -1.0
    assert rescaled.max() <= 1.0
    torch.testing.assert_close(rescaled, raw_params * 2 - 1, atol=0.0, rtol=0.0)


def test_prepare_batch_ot_true_matches_reference_hungarian() -> None:
    """``ot=True`` pairs noise/params exactly as the reference ``_hungarian_match``."""
    seed = 0
    raw = _make_raw(read_mel=True, read_audio=True)
    golden = _reference_prepare_batch(
        raw, mean=None, std=None, rescale_params=True, ot=True, seed=seed
    )
    out = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=True,
        ot=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for key in ("noise", "params", "mel_spec", "audio"):
        torch.testing.assert_close(out[key], golden[key], atol=0.0, rtol=0.0)


def test_prepare_batch_ot_false_passes_through_unpermuted() -> None:
    """``ot=False`` leaves every modality in its read order (no Hungarian permutation)."""
    seed = 0
    raw = _make_raw(read_mel=True, read_audio=True)
    out = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=True,
        ot=False,
        generator=torch.Generator().manual_seed(seed),
    )
    expected_params = torch.from_numpy(_unwrap_array(raw["param_array"]) * 2 - 1).to(
        dtype=torch.float32
    )
    expected_mel = torch.from_numpy(_unwrap_array(raw.get("mel_spec"))).to(dtype=torch.float32)
    expected_audio = torch.from_numpy(_unwrap_array(raw.get("audio"))).to(dtype=torch.float32)
    torch.testing.assert_close(out["params"], expected_params, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out["mel_spec"], expected_mel, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out["audio"], expected_audio, atol=0.0, rtol=0.0)
    # randn here vs empty_like().normal_() in production: bit-equality is
    # guarded by test_noise_draw_apis_same_seed_bit_identical.
    expected_noise = torch.randn(
        expected_params.shape, generator=torch.Generator().manual_seed(seed)
    )
    torch.testing.assert_close(out["noise"], expected_noise, atol=0.0, rtol=0.0)


def test_noise_draw_apis_same_seed_bit_identical() -> None:
    """The three normal-draw APIs agree bit-for-bit under one seed on CPU.

    The goldens draw via ``torch.randn(generator=...)``, production draws via
    ``empty_like().normal_(generator=...)``, and the pre-refactor code drew from
    the seeded global RNG. All ``atol=rtol=0`` pinning in this module rests on
    these being the same stream; this test makes a PyTorch upgrade that breaks
    the equivalence fail here loudly instead of as a confusing mass golden diff.
    """
    seed = 1234
    shape = (_BATCH, _NUM_PARAMS)
    via_randn = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    via_normal_inplace = torch.empty(shape).normal_(generator=torch.Generator().manual_seed(seed))
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        via_global_rng = torch.randn(shape)
    torch.testing.assert_close(via_randn, via_normal_inplace, atol=0.0, rtol=0.0)
    torch.testing.assert_close(via_randn, via_global_rng, atol=0.0, rtol=0.0)


def test_prepare_batch_missing_param_array_raises_key_error() -> None:
    """``param_array`` is required; an absent key surfaces a ``KeyError``, not noise."""
    # cast: deliberately omit the required key to pin the missing-column contract.
    raw = cast(RawBatch, {"mel_spec": np.zeros(_MEL_SHAPE, dtype=np.float32)})
    with pytest.raises(KeyError, match="param_array"):
        prepare_batch(
            raw,
            mean=None,
            std=None,
            rescale_params=False,
            ot=False,
            generator=torch.Generator().manual_seed(0),
        )


def test_prepare_batch_ot_pairs_minimize_total_noise_to_param_distance() -> None:
    """OT pairing minimizes summed noise-param L2 — an independent optimality check.

    Hardcoded, reference-free: any valid permutation's total cost must be >= the
    cost ``prepare_batch`` produces, so a mis-paired extraction is caught without
    trusting ``_reference_prepare_batch``.
    """

    # 5! = 120 permutations keeps this brute force fast (8! would be 40320); the
    # pairing is batch-size-independent, so a small batch still exercises it.
    n_small = 5
    rng = np.random.default_rng(7)
    raw: RawBatch = {"param_array": rng.random((n_small, _NUM_PARAMS)).astype(np.float32)}
    out = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=True,
        ot=True,
        generator=torch.Generator().manual_seed(0),
    )
    noise = _unwrap(out["noise"])
    params = _unwrap(out["params"])
    chosen_cost = torch.linalg.vector_norm(noise - params, dim=1).sum().item()
    for perm in itertools.permutations(range(n_small)):
        cost = torch.linalg.vector_norm(noise[list(perm)] - params, dim=1).sum().item()
        assert chosen_cost <= cost + 1e-5


def test_prepare_batch_mean_none_alias_branch_does_not_mutate_raw() -> None:
    """The ``mean=None`` mel path aliases the input array without writing back to it."""
    raw = _make_raw(read_mel=True)
    snapshot = _unwrap_array(raw.get("mel_spec")).copy()
    prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=True,
        ot=True,
        generator=torch.Generator().manual_seed(0),
    )
    np.testing.assert_array_equal(_unwrap_array(raw.get("mel_spec")), snapshot)
