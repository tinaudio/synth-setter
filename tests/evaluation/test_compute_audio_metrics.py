"""Unit tests for ``synth_setter.evaluation.compute_audio_metrics``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

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
from tests.helpers.audio_utils import sine, write_wav

_SR = 44100


def _sine(seconds: float = 1.0, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    return sine(freq=freq, amplitude=amplitude, channels=1, sr=_SR, seconds=seconds)


def _make_sample_dir(parent: Path, name: str, target: np.ndarray, pred: np.ndarray) -> Path:
    """Create a ``sample_<name>`` directory with ``target.wav`` and ``pred.wav``.

    :param parent: Parent directory under which the sample directory is created.
    :param name: Sample identifier (becomes the ``sample_<name>`` directory suffix).
    :param target: ``target.wav`` audio array.
    :param pred: ``pred.wav`` audio array.
    :return: Path of the created sample directory.
    """
    sample_dir = parent / f"sample_{name}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_wav(sample_dir / "target.wav", target)
    write_wav(sample_dir / "pred.wav", pred)
    return sample_dir


@pytest.fixture(autouse=True)
def _reset_module_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level ``scatter`` and ``pesto_model`` caches per test.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
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


def test_subdir_matches_pattern_with_both_files_returns_true(tmp_path: Path) -> None:
    """Both ``target.wav`` and ``pred.wav`` present → True.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    (tmp_path / "target.wav").touch()
    (tmp_path / "pred.wav").touch()
    assert subdir_matches_pattern(tmp_path) is True


def test_subdir_matches_pattern_missing_target_returns_false(tmp_path: Path) -> None:
    """Missing ``target.wav`` → False.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    (tmp_path / "pred.wav").touch()
    assert subdir_matches_pattern(tmp_path) is False


def test_subdir_matches_pattern_missing_pred_returns_false(tmp_path: Path) -> None:
    """Missing ``pred.wav`` → False.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    (tmp_path / "target.wav").touch()
    assert subdir_matches_pattern(tmp_path) is False


def test_subdir_matches_pattern_empty_dir_returns_false(tmp_path: Path) -> None:
    """Empty directory → False.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    assert subdir_matches_pattern(tmp_path) is False


# ---------------------------------------------------------------------------
# find_possible_subdirs
# ---------------------------------------------------------------------------


def test_find_possible_subdirs_returns_only_matching_dirs(tmp_path: Path) -> None:
    """Returns only subdirectories that contain both ``target.wav`` and ``pred.wav``.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    good = tmp_path / "sample_0"
    good.mkdir()
    (good / "target.wav").touch()
    (good / "pred.wav").touch()

    bad_missing = tmp_path / "sample_1"
    bad_missing.mkdir()
    (bad_missing / "target.wav").touch()

    result = find_possible_subdirs(tmp_path)
    assert result == [good]


def test_find_possible_subdirs_skips_files(tmp_path: Path) -> None:
    """Files in ``audio_dir`` are not considered candidate subdirectories.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    (tmp_path / "stray.wav").touch()
    good = tmp_path / "sample_0"
    good.mkdir()
    (good / "target.wav").touch()
    (good / "pred.wav").touch()

    result = find_possible_subdirs(tmp_path)
    assert result == [good]


def test_find_possible_subdirs_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    """No subdirectories → empty list.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
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
def test_compute_mel_specs_spec_has_expected_n_mels(idx: int, expected_n_mels: int) -> None:
    """Each spec has finite values and the expected number of mel rows.

    :param idx: Parametrized ``idx`` value under test.
    :param expected_n_mels: Parametrized ``expected_n_mels`` value under test.
    """
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


def test_compute_mss_grows_with_frequency_separation() -> None:
    """A tone two octaves away scores further from the reference than one octave.

    Monotonicity holds over this range but not globally — the mel-scale distance saturates above
    ~3.5 kHz, so widening the sweep would invert the comparison.
    """
    reference = _sine(seconds=0.5, freq=440.0)[0]
    one_octave = _sine(seconds=0.5, freq=880.0)[0]
    two_octaves = _sine(seconds=0.5, freq=1760.0)[0]

    assert compute_mss(reference, one_octave) < compute_mss(reference, two_octaves)


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


def test_compute_mfcc_multichannel_input_returns_channel_leading_shape() -> None:
    """``compute_mfcc`` with ``(C, T)`` input returns ``(C, 20, frames)`` from librosa.

    The caller (``compute_wmfcc``) reshapes this to ``(-1, frames)`` before DTW,
    so channels are NOT averaged — they are passed through as-is.
    """
    audio = _sine(seconds=0.5)  # shape (1, T)
    mfcc = compute_mfcc(audio)
    # librosa returns (C, n_mfcc, frames) for (C, T) input
    assert mfcc.shape[0] == 1  # C=1 preserved
    assert mfcc.shape[1] == 20
    assert mfcc.shape[2] > 0
    assert np.isfinite(mfcc).all()


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


def test_compute_wmfcc_is_symmetric() -> None:
    """``compute_wmfcc(a, b) == compute_wmfcc(b, a)`` despite the DTW alignment.

    DTW distance is only symmetric when the local cost and step pattern are; the length
    normalisation applied here preserves that, so a step-pattern change that broke it would surface
    as an argument-order dependence.

    Unlike the other distances, wMFCC is *not* monotonic in frequency separation — it dips around
    3.5 kHz — so no ordering is asserted.
    """
    a = _sine(seconds=0.5, freq=440.0)[0]
    b = _sine(seconds=0.5, freq=880.0)[0]

    assert compute_wmfcc(a, b) == pytest.approx(compute_wmfcc(b, a), rel=1e-9)


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
    assert dist > 0


def test_compute_sot_is_symmetric() -> None:
    """``compute_sot(a, b) == compute_sot(b, a)`` — transport cost carries no direction."""
    a = _sine(seconds=0.5, freq=440.0)
    b = _sine(seconds=0.5, freq=880.0)

    assert compute_sot(a, b) == pytest.approx(compute_sot(b, a), rel=1e-9)


def test_compute_sot_grows_with_frequency_separation() -> None:
    """Transport cost rises as the spectra move further apart."""
    reference = _sine(seconds=0.5, freq=440.0)
    one_octave = _sine(seconds=0.5, freq=880.0)
    three_octaves = _sine(seconds=0.5, freq=3520.0)

    assert compute_sot(reference, one_octave) < compute_sot(reference, three_octaves)


# ---------------------------------------------------------------------------
# compute_metrics_on_dir
# ---------------------------------------------------------------------------


def test_compute_metrics_on_dir_returns_expected_keys(tmp_path: Path) -> None:
    """End-to-end on a single sample dir returns finite ``mss/wmfcc/sot/rms``.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    sample_dir = _make_sample_dir(tmp_path, "0", _sine(seconds=0.5), _sine(seconds=0.5))
    metrics = compute_metrics_on_dir(sample_dir)
    assert set(metrics.keys()) == {"mss", "wmfcc", "sot", "rms"}
    for value in metrics.values():
        assert np.isfinite(value)


def test_compute_metrics_on_dir_identical_files_yields_perfect_scores(tmp_path: Path) -> None:
    """Identical target/pred → mss/wmfcc/sot ≈ 0 and rms == 1.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
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


def test_compute_metrics_writes_csv_with_expected_index_and_columns(tmp_path: Path) -> None:
    """Writes a ``metrics-<pid>.csv`` with the trailing ``_N`` suffix as the row index.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
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
def test_main_writes_metrics_and_aggregated_csvs(tmp_path: Path) -> None:
    """End-to-end Click CLI run produces ``metrics.csv`` and ``aggregated_metrics.csv``.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
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
    """An octave apart produces a finite, strictly positive mean abs f0 difference."""
    target = _sine(seconds=1.0, freq=440.0)
    pred = _sine(seconds=1.0, freq=880.0)
    dist = compute_f0(target, pred)
    assert np.isfinite(dist)
    assert dist > 0


@pytest.mark.slow
def test_compute_f0_without_confident_frames_returns_nan() -> None:
    """No frame clearing the 0.85 gate on both signals yields NaN, not an error.

    Pins current behaviour, not desired behaviour: the pinned ``mir-1k_g7``
    model is vocal-trained, so a 1760 Hz tone leaves the mask empty and the
    mean over zero frames is NaN, which then propagates into eval aggregates.
    Tracked in #2634 — update this test with the fix.
    """
    target = _sine(seconds=1.0, freq=440.0)
    pred = _sine(seconds=1.0, freq=1760.0)

    confident_target, _ = get_pesto_activations(target, pred)
    assert confident_target.size == 0

    assert np.isnan(compute_f0(target, pred))


_UNIFORM_PARAMS_CSV = ",pred,target\ncutoff,0.5,0.5\nresonance,0.2,0.2\n"


def _make_uniform_sample_dir(
    parent: Path,
    name: str,
    target: np.ndarray,
    pred: np.ndarray,
    *,
    params_csv: str = _UNIFORM_PARAMS_CSV,
) -> Path:
    """Create a ``sample_<name>`` dir with target/pred wavs and a ``params.csv``.

    :param parent: Parent directory under which the sample directory is created.
    :param name: Sample identifier (becomes the ``sample_<name>`` directory suffix).
    :param target: ``target.wav`` audio array.
    :param pred: ``pred.wav`` audio array.
    :param params_csv: Body written to ``params.csv``; vary it to break the gate.
    :return: Path of the created sample directory.
    """
    sample_dir = _make_sample_dir(parent, name, target, pred)
    (sample_dir / "params.csv").write_text(params_csv)
    return sample_dir


@pytest.mark.slow
def test_main_uniform_params_writes_only_standard_metric_outputs(tmp_path: Path) -> None:
    """Uniform parameters do not trigger an additional shuffled evaluation pass.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "aggregated_metrics_shuffled.csv").write_text("stale")
    (metrics_dir / "shuffle_permutation.csv").write_text("stale")
    sample_0 = _make_uniform_sample_dir(audio_root, "0", _sine(seconds=0.3), _sine(seconds=0.3))
    _make_uniform_sample_dir(
        audio_root, "1", _sine(seconds=0.3, freq=440.0), _sine(seconds=0.3, freq=880.0)
    )
    legacy_sample = metrics_dir / "shuffled_audio" / "sample_0"
    legacy_sample.mkdir(parents=True)
    (legacy_sample / "pred.wav").symlink_to(sample_0 / "pred.wav")
    (legacy_sample / "target.wav").symlink_to(sample_0 / "target.wav")

    result = CliRunner().invoke(
        compute_audio_metrics_main,
        [str(audio_root), str(metrics_dir), "-w", "1"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert (metrics_dir / "aggregated_metrics.csv").is_file()
    assert (metrics_dir / "metrics.csv").is_file()
    assert not (metrics_dir / "aggregated_metrics_shuffled.csv").exists()
    assert not (metrics_dir / "shuffle_permutation.csv").exists()
    assert not (metrics_dir / "shuffled_audio").exists()


@pytest.mark.slow
def test_main_preserves_unowned_legacy_named_directory(tmp_path: Path) -> None:
    """Metric cleanup does not recursively delete an unowned output directory.

    :param tmp_path: Root containing rendered audio and a user-owned metrics artifact.
    """
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    _make_sample_dir(audio_root, "0", _sine(seconds=0.2), _sine(seconds=0.2))
    metrics_dir = tmp_path / "metrics"
    retained_file = metrics_dir / "shuffled_audio" / "notes.txt"
    retained_file.parent.mkdir(parents=True)
    retained_file.write_text("keep")

    result = CliRunner().invoke(
        compute_audio_metrics_main,
        [str(audio_root), str(metrics_dir), "-w", "1"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert retained_file.read_text() == "keep"


def test_main_output_dir_equal_to_audio_dir_raises(tmp_path: Path) -> None:
    """Metric outputs cannot overwrite or remove source audio artifacts.

    :param tmp_path: Root containing one valid rendered sample.
    """
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    _make_sample_dir(audio_root, "0", _sine(seconds=0.2), _sine(seconds=0.2))

    result = CliRunner().invoke(
        compute_audio_metrics_main,
        [str(audio_root), str(audio_root), "-w", "1"],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "must not equal, contain, or be contained" in str(result.exception)
    assert (audio_root / "sample_0" / "pred.wav").is_file()


def test_main_output_dir_nested_in_audio_dir_raises_without_mutation(tmp_path: Path) -> None:
    """Containment validation runs before creating a nested output directory.

    :param tmp_path: Root containing one valid rendered sample.
    """
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    _make_sample_dir(audio_root, "0", _sine(seconds=0.2), _sine(seconds=0.2))
    metrics_dir = audio_root / "metrics"

    result = CliRunner().invoke(
        compute_audio_metrics_main,
        [str(audio_root), str(metrics_dir), "-w", "1"],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert not metrics_dir.exists()


def test_main_output_dir_ancestor_of_audio_dir_raises(tmp_path: Path) -> None:
    """Metric cleanup cannot remove a nested source audio directory.

    :param tmp_path: Output root containing source audio under a cleanup target name.
    """
    output_root = tmp_path / "metrics"
    audio_root = output_root / "shuffled_audio"
    audio_root.mkdir(parents=True)
    _make_sample_dir(audio_root, "0", _sine(seconds=0.2), _sine(seconds=0.2))

    result = CliRunner().invoke(
        compute_audio_metrics_main,
        [str(audio_root), str(output_root), "-w", "1"],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "must not equal, contain, or be contained" in str(result.exception)
    assert (audio_root / "sample_0" / "pred.wav").is_file()


def test_main_num_workers_zero_raises_usage_error(tmp_path: Path) -> None:
    """``--num_workers 0`` is rejected at the CLI boundary before any IO.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    runner = CliRunner()
    result = runner.invoke(
        compute_audio_metrics_main,
        [str(tmp_path / "audio"), str(tmp_path / "metrics"), "-w", "0"],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# sample-rate forwarding
# ---------------------------------------------------------------------------

_RATE_AWARE_METRICS = (compute_mss, compute_sot, compute_wmfcc)


@pytest.mark.parametrize("metric", _RATE_AWARE_METRICS, ids=lambda fn: fn.__name__)
def test_metric_sample_rate_reaches_the_analysis_window(
    metric: Callable[..., float],
) -> None:
    """A caller's sample rate changes the score, so it must be reaching the transform.

    Each of these metrics sizes its window and hop from the sample rate. If the argument were
    dropped on the way to the helper — the defect this parametrisation guards — every rate would
    collapse onto the 44.1 kHz default and score identically.

    :param metric: Metric under test.
    """
    target = _sine(seconds=0.5, freq=440.0)
    pred = _sine(seconds=0.5, freq=880.0)

    assert metric(target, pred, _SR / 2) != pytest.approx(metric(target, pred, _SR))


@pytest.mark.parametrize("metric", _RATE_AWARE_METRICS, ids=lambda fn: fn.__name__)
def test_metric_default_sample_rate_matches_an_explicit_44100(
    metric: Callable[..., float],
) -> None:
    """Omitting the sample rate keeps the historical 44.1 kHz behaviour exactly.

    Every pre-existing call site relies on this: the parameter is additive, not a change.

    :param metric: Metric under test.
    """
    target = _sine(seconds=0.5, freq=440.0)
    pred = _sine(seconds=0.5, freq=880.0)

    assert metric(target, pred) == pytest.approx(metric(target, pred, 44100.0))
