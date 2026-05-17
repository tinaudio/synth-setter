"""Unit tests for ``synth_setter.evaluation.compute_audio_metrics``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pedalboard.io import AudioFile

import synth_setter.evaluation.compute_audio_metrics as cam
from synth_setter.evaluation.compute_audio_metrics import (
    MEL_PARAMS,
    batched_wasserstein_distance_np,
    compute_f0,
    compute_jtfs,
    compute_jtfs_distance,
    compute_mel_specs,
    compute_metrics,
    compute_metrics_on_dir,
    compute_mfcc,
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
    find_possible_subdirs,
    get_pesto_activations,
    get_stft,
    subdir_matches_pattern,
)
from synth_setter.evaluation.compute_audio_metrics import (
    main as compute_audio_metrics_main,
)

_SR = 44100


def _sine(seconds: float = 1.0, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    """Generate a mono ``(1, N)`` sine — the shape compute-audio-metrics expects."""
    t = np.arange(int(seconds * _SR), dtype=np.float32) / _SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32).reshape(1, -1)


def _write_wav(path: Path, audio: np.ndarray) -> None:  # noqa: DOC101,DOC103
    """Write a ``(channels, N)`` array to ``path`` as a 44.1 kHz WAV file."""
    channels = audio.shape[0]
    with AudioFile(str(path), "w", _SR, channels) as f:
        f.write(audio)


def _make_sample_dir(  # noqa: DOC101,DOC103,DOC201,DOC203
    parent: Path, name: str, target: np.ndarray, pred: np.ndarray
) -> Path:
    """Create a ``sample_<name>`` directory with ``target.wav`` and ``pred.wav``."""
    sample_dir = parent / f"sample_{name}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(sample_dir / "target.wav", target)
    _write_wav(sample_dir / "pred.wav", pred)
    return sample_dir


@pytest.fixture(autouse=True)
def _reset_module_caches(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: DOC101,DOC103
    """Reset module-level ``scatter`` and ``pesto_model`` caches per test."""
    monkeypatch.setattr(cam, "scatter", None, raising=True)
    monkeypatch.setattr(cam, "pesto_model", None, raising=True)


# ---------------------------------------------------------------------------
# compute_rms — regression guards for #899.
# ---------------------------------------------------------------------------


def test_compute_rms_identical_signal_returns_one() -> None:
    """``cosine_sim(x, x)`` of a non-silent signal is ``1.0``."""
    audio = _sine()
    rms = compute_rms(audio, audio)
    assert np.isfinite(rms)
    assert rms == pytest.approx(1.0, abs=1e-6)


def test_compute_rms_silent_pred_returns_zero_not_nan() -> None:
    """Silent pred → ``pred_norm == 0`` → clamped denominator → ``cosine_sim = 0``.

    Regression guard: prior to the denominator clamp, this produced ``0/0 = NaN`` and
    poisoned downstream metric aggregation. See the MPS flake on
    ``test_train_eval_surge_xt[mps]`` where a 1-step-trained model can predict params
    that Surge XT renders as bit-silent audio.
    """
    target = _sine()
    pred = np.zeros_like(target)
    rms = compute_rms(target, pred)
    assert np.isfinite(rms), f"compute_rms produced non-finite {rms!r} for silent pred"
    assert rms == 0.0


def test_compute_rms_quiet_nonzero_inputs_return_zero() -> None:
    """Quiet (but non-zero) inputs whose ``target_norm * pred_norm < 1e-12`` return 0.

    Without the explicit short-circuit, the pre-fix path of
    ``dot(target_rms, pred_rms) / np.clip(denom, 1e-12, None)`` would return ~``0.4``
    here (numerator and clamped denominator both ≈ ``4e-13``), contradicting the
    warning's "returning 0" claim — see the Copilot review on PR #899.
    """
    quiet = np.full((1, _SR), 1e-7, dtype=np.float64)
    rms = compute_rms(quiet, quiet)
    assert rms == 0.0


# ---------------------------------------------------------------------------
# subdir_matches_pattern
# ---------------------------------------------------------------------------


def test_subdir_matches_pattern_with_both_files_returns_true(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """Both ``target.wav`` and ``pred.wav`` present → True."""
    (tmp_path / "target.wav").touch()
    (tmp_path / "pred.wav").touch()
    assert subdir_matches_pattern(tmp_path) is True


def test_subdir_matches_pattern_missing_target_returns_false(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """Missing ``target.wav`` → False."""
    (tmp_path / "pred.wav").touch()
    assert subdir_matches_pattern(tmp_path) is False


def test_subdir_matches_pattern_missing_pred_returns_false(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """Missing ``pred.wav`` → False."""
    (tmp_path / "target.wav").touch()
    assert subdir_matches_pattern(tmp_path) is False


def test_subdir_matches_pattern_empty_dir_returns_false(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """Empty directory → False."""
    assert subdir_matches_pattern(tmp_path) is False


# ---------------------------------------------------------------------------
# find_possible_subdirs
# ---------------------------------------------------------------------------


def test_find_possible_subdirs_returns_only_matching_dirs(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """Returns only subdirectories that contain both ``target.wav`` and ``pred.wav``."""
    good = tmp_path / "sample_0"
    good.mkdir()
    (good / "target.wav").touch()
    (good / "pred.wav").touch()

    bad_missing = tmp_path / "sample_1"
    bad_missing.mkdir()
    (bad_missing / "target.wav").touch()

    result = find_possible_subdirs(tmp_path)
    assert result == [good]


def test_find_possible_subdirs_skips_files(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """Files in ``audio_dir`` are not considered candidate subdirectories."""
    (tmp_path / "stray.wav").touch()
    good = tmp_path / "sample_0"
    good.mkdir()
    (good / "target.wav").touch()
    (good / "pred.wav").touch()

    result = find_possible_subdirs(tmp_path)
    assert result == [good]


def test_find_possible_subdirs_empty_dir_returns_empty_list(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """No subdirectories → empty list."""
    assert find_possible_subdirs(tmp_path) == []


# ---------------------------------------------------------------------------
# compute_mel_specs
# ---------------------------------------------------------------------------


def test_compute_mel_specs_returns_one_spec_per_mel_param() -> None:
    """Cardinality contract: one spec per entry in ``MEL_PARAMS``."""
    audio = _sine(seconds=0.5)
    specs = compute_mel_specs(audio[0])
    assert len(specs) == len(MEL_PARAMS)


@pytest.mark.parametrize(
    ("idx", "expected_n_mels"),
    [(0, 32), (1, 64), (2, 128)],
)
def test_compute_mel_specs_spec_has_expected_n_mels(  # noqa: DOC101,DOC103
    idx: int, expected_n_mels: int
) -> None:
    """Each spec has finite values and the expected number of mel rows."""
    audio = _sine(seconds=0.5)
    specs = compute_mel_specs(audio[0])
    assert specs[idx].shape[-2] == expected_n_mels
    assert np.isfinite(specs[idx]).all()


def test_compute_mel_specs_is_deterministic() -> None:
    """Identical inputs yield identical outputs."""
    audio = _sine(seconds=0.5)
    first = compute_mel_specs(audio[0])
    second = compute_mel_specs(audio[0])
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# compute_mss
# ---------------------------------------------------------------------------


def test_compute_mss_identical_inputs_returns_zero() -> None:
    """``compute_mss(x, x)`` is exactly 0."""
    audio = _sine(seconds=0.5)[0]
    assert compute_mss(audio, audio) == pytest.approx(0.0, abs=1e-9)


def test_compute_mss_different_inputs_is_positive() -> None:
    """Distinct signals produce a strictly positive distance."""
    target = _sine(seconds=0.5, freq=440.0)[0]
    pred = _sine(seconds=0.5, freq=880.0)[0]
    dist = compute_mss(target, pred)
    assert dist > 0
    assert np.isfinite(dist)


def test_compute_mss_is_symmetric() -> None:
    """``compute_mss(a, b) == compute_mss(b, a)``."""
    a = _sine(seconds=0.5, freq=440.0)[0]
    b = _sine(seconds=0.5, freq=880.0)[0]
    assert compute_mss(a, b) == pytest.approx(compute_mss(b, a), abs=1e-9)


# ---------------------------------------------------------------------------
# compute_mfcc
# ---------------------------------------------------------------------------


def test_compute_mfcc_returns_20_coefficients() -> None:
    """``compute_mfcc`` returns a finite ``(20, n_frames)`` array."""
    audio = _sine(seconds=0.5)[0]
    mfcc = compute_mfcc(audio)
    assert mfcc.shape[0] == 20
    assert mfcc.shape[1] > 0
    assert np.isfinite(mfcc).all()


def test_compute_mfcc_is_deterministic() -> None:
    """Identical inputs yield identical MFCCs."""
    audio = _sine(seconds=0.5)[0]
    np.testing.assert_array_equal(compute_mfcc(audio), compute_mfcc(audio))


# ---------------------------------------------------------------------------
# compute_wmfcc
# ---------------------------------------------------------------------------


def test_compute_wmfcc_identical_inputs_returns_zero() -> None:
    """DTW-normalized distance of identical signals is 0."""
    audio = _sine(seconds=0.5)[0]
    assert compute_wmfcc(audio, audio) == pytest.approx(0.0, abs=1e-9)


def test_compute_wmfcc_different_inputs_is_positive() -> None:
    """Distinct signals produce a strictly positive distance."""
    target = _sine(seconds=0.5, freq=440.0)[0]
    pred = _sine(seconds=0.5, freq=880.0)[0]
    dist = compute_wmfcc(target, pred)
    assert dist > 0
    assert np.isfinite(dist)


# ---------------------------------------------------------------------------
# get_stft
# ---------------------------------------------------------------------------


def test_get_stft_returns_2d_magnitude() -> None:
    """``get_stft`` returns a non-negative ``(n_frames, n_bins)`` magnitude array."""
    audio = _sine(seconds=0.5)
    stft = get_stft(audio)
    assert stft.ndim == 2
    assert (stft >= 0).all()
    assert np.isfinite(stft).all()


def test_get_stft_averages_channels() -> None:
    """Stereo STFT equals the STFT of the per-sample channel mean."""
    ch0 = _sine(seconds=0.5, freq=440.0)
    ch1 = _sine(seconds=0.5, freq=880.0)
    stereo = np.concatenate([ch0, ch1], axis=0)
    expected_mono = (ch0 + ch1) / 2
    np.testing.assert_allclose(get_stft(stereo), get_stft(expected_mono), atol=1e-6)


# ---------------------------------------------------------------------------
# batched_wasserstein_distance_np
# ---------------------------------------------------------------------------


def test_batched_wasserstein_distance_identical_hists_returns_zero() -> None:
    """Identical histograms have exactly zero Wasserstein distance."""
    hist = np.array([[0.25, 0.25, 0.25, 0.25]])
    np.testing.assert_array_equal(batched_wasserstein_distance_np(hist, hist), [0.0])


def test_batched_wasserstein_distance_handcrafted_case() -> None:
    """Hand-computed 2-bin case validates the CDF-difference formula.

    ``hist1 = [1, 0]``, ``hist2 = [0, 1]``, ``bin_width = 0.5``:
    CDFs are ``[1, 1]`` vs ``[0, 1]``, ``|diff| = [1, 0]``, ``sum = 1``, ``* 0.5 = 0.5``.
    """
    hist1 = np.array([[1.0, 0.0]])
    hist2 = np.array([[0.0, 1.0]])
    assert batched_wasserstein_distance_np(hist1, hist2) == pytest.approx([0.5])


def test_batched_wasserstein_distance_preserves_batch_dim() -> None:
    """Leading batch dimensions are preserved in the output shape."""
    hist1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    hist2 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    result = batched_wasserstein_distance_np(hist1, hist2)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [0.0, 0.0])


# ---------------------------------------------------------------------------
# compute_sot
# ---------------------------------------------------------------------------


def test_compute_sot_identical_inputs_returns_zero() -> None:
    """Identical signals have zero spectral optimal-transport distance."""
    audio = _sine(seconds=0.5)
    assert compute_sot(audio, audio) == pytest.approx(0.0, abs=1e-9)


def test_compute_sot_different_inputs_is_finite_and_nonnegative() -> None:
    """Distinct signals yield a finite, non-negative distance."""
    target = _sine(seconds=0.5, freq=440.0)
    pred = _sine(seconds=0.5, freq=1760.0)
    dist = compute_sot(target, pred)
    assert np.isfinite(dist)
    assert dist >= 0


# ---------------------------------------------------------------------------
# compute_metrics_on_dir
# ---------------------------------------------------------------------------


def test_compute_metrics_on_dir_returns_expected_keys(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """End-to-end on a single sample dir returns finite ``mss/wmfcc/sot/rms``."""
    sample_dir = _make_sample_dir(tmp_path, "0", _sine(seconds=0.5), _sine(seconds=0.5))
    metrics = compute_metrics_on_dir(sample_dir)
    assert set(metrics.keys()) == {"mss", "wmfcc", "sot", "rms"}
    for value in metrics.values():
        assert np.isfinite(value)


def test_compute_metrics_on_dir_identical_files_yields_perfect_scores(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """Identical target/pred → mss/wmfcc/sot ≈ 0 and rms == 1."""
    audio = _sine(seconds=0.5)
    sample_dir = _make_sample_dir(tmp_path, "0", audio, audio)
    metrics = compute_metrics_on_dir(sample_dir)
    assert metrics["mss"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["wmfcc"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["sot"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["rms"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_writes_csv_with_expected_index_and_columns(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """Writes a ``metrics-<pid>.csv`` with the trailing ``_N`` suffix as the row index."""
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    sample_dirs = [
        _make_sample_dir(audio_root, "0", _sine(seconds=0.3), _sine(seconds=0.3)),
        _make_sample_dir(
            audio_root,
            "1",
            _sine(seconds=0.3, freq=440.0),
            _sine(seconds=0.3, freq=880.0),
        ),
    ]

    metric_file = compute_metrics(sample_dirs, output_dir)
    assert metric_file.is_file()

    df = pd.read_csv(metric_file, index_col=0)
    assert sorted(str(i) for i in df.index) == ["0", "1"]
    assert {"mss", "wmfcc", "sot", "rms"}.issubset(df.columns)
    assert np.isfinite(df.to_numpy()).all()


# ---------------------------------------------------------------------------
# main (Click CLI)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_main_writes_metrics_and_aggregated_csvs(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """End-to-end Click CLI run produces ``metrics.csv`` and ``aggregated_metrics.csv``."""
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    metrics_dir = tmp_path / "metrics"

    _make_sample_dir(audio_root, "0", _sine(seconds=0.3), _sine(seconds=0.3))
    _make_sample_dir(
        audio_root,
        "1",
        _sine(seconds=0.3, freq=440.0),
        _sine(seconds=0.3, freq=880.0),
    )

    runner = CliRunner()
    result = runner.invoke(
        compute_audio_metrics_main,
        [str(audio_root), str(metrics_dir), "-w", "1"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    metrics_df = pd.read_csv(metrics_dir / "metrics.csv")
    assert len(metrics_df) == 2
    assert {"mss", "wmfcc", "sot", "rms"}.issubset(metrics_df.columns)

    agg_df = pd.read_csv(metrics_dir / "aggregated_metrics.csv", index_col=0)
    assert {"mean", "std"}.issubset(agg_df.columns)
    assert {"mss", "wmfcc", "sot", "rms"}.issubset(set(agg_df.index))


# ---------------------------------------------------------------------------
# compute_jtfs / compute_jtfs_distance — exercise the real Scattering1D
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compute_jtfs_first_call_constructs_scatter_and_returns_array() -> None:
    """First call populates the module-level ``scatter`` cache and returns coefficients."""
    audio = _sine(seconds=0.5)[0]
    assert cam.scatter is None
    result = compute_jtfs(audio, J=6, Q=8)
    assert cam.scatter is not None
    assert isinstance(result, np.ndarray)
    assert result.ndim >= 1


@pytest.mark.slow
def test_compute_jtfs_distance_identical_inputs_returns_zero() -> None:
    """Identical signals → JTFS L1 distance is 0."""
    audio = _sine(seconds=0.5)[0]
    dist = compute_jtfs_distance(audio, audio, J=6, Q=8)
    assert dist == pytest.approx(0.0, abs=1e-9)


@pytest.mark.slow
def test_compute_jtfs_distance_different_inputs_is_positive() -> None:
    """Distinct signals → JTFS L1 distance is strictly positive and finite."""
    target = _sine(seconds=0.5, freq=440.0)[0]
    pred = _sine(seconds=0.5, freq=880.0)[0]
    dist = compute_jtfs_distance(target, pred, J=6, Q=8)
    assert np.isfinite(dist)
    assert dist > 0


@pytest.mark.slow
def test_compute_jtfs_cache_is_shape_keyed_not_param_keyed() -> None:
    """Quirk: the module-level cache is keyed only on first-call shape.

    Subsequent calls reuse the cached ``Scattering1D`` even when ``J``/``Q`` change,
    as long as the input shape is unchanged. This test pins that behavior so any
    future refactor that re-keys the cache (e.g. on ``(shape, J, Q)``) trips here
    and prompts an intentional update.
    """
    audio = _sine(seconds=0.5)[0]
    compute_jtfs(audio, J=6, Q=8)
    cached = cam.scatter
    compute_jtfs(audio, J=4, Q=4)
    assert cam.scatter is cached


# ---------------------------------------------------------------------------
# get_pesto_activations / compute_f0 — exercise the real pesto model
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_get_pesto_activations_returns_matched_shape_arrays() -> None:
    """F0 arrays for target and pred share a shape (joint confidence mask)."""
    target = _sine(seconds=1.0, freq=440.0)
    pred = _sine(seconds=1.0, freq=440.0)
    assert cam.pesto_model is None
    target_f0, pred_f0 = get_pesto_activations(target, pred)
    assert cam.pesto_model is not None
    assert target_f0.shape == pred_f0.shape
    assert target_f0.ndim == 1


@pytest.mark.slow
def test_compute_f0_identical_inputs_returns_zero() -> None:
    """Identical signals → mean abs f0 difference is 0."""
    audio = _sine(seconds=1.0, freq=440.0)
    dist = compute_f0(audio, audio)
    assert dist == pytest.approx(0.0, abs=1e-6)


@pytest.mark.slow
def test_compute_f0_different_inputs_is_finite() -> None:
    """Distinct tones produce a finite mean abs f0 difference."""
    target = _sine(seconds=1.0, freq=440.0)
    pred = _sine(seconds=1.0, freq=880.0)
    dist = compute_f0(target, pred)
    assert np.isfinite(dist)
