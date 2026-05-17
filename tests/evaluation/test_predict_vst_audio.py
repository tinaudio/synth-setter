"""Behavioral tests for ``synth_setter.evaluation.predict_vst_audio``.

These tests pin the public contract of the module so the refactor in this PR
can run safely. The CLI flow is exercised end-to-end with ``render_params``
monkey-patched to return deterministic fake audio — no real VST is loaded.
"""

from __future__ import annotations

import os

# Pin the headless backend before ``predict_vst_audio`` (or any sibling test
# module via collection order — e.g. ``tests/test_callbacks.py`` pulls in
# ``synth_setter.utils.callbacks`` which imports pyplot at module load) can
# trigger ``matplotlib.pyplot`` initialization. ``matplotlib.use("Agg")`` raises
# ``ValueError`` if pyplot is already imported; setting ``MPLBACKEND`` first
# is the safe equivalent. ``setdefault`` respects a CI-provided override.
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path  # noqa: E402

import librosa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from click.testing import CliRunner, Result  # noqa: E402

from synth_setter.data.vst import param_specs  # noqa: E402
from synth_setter.evaluation import predict_vst_audio  # noqa: E402

# Small/fast audio dimensions for tests — full 4 s @ 44.1 kHz is unnecessary
# here and slows mel-spectrogram computation. ``hop_length=512`` (the source's
# hardcoded value) means we get ~20 frames for 0.25 s.
_SR = 8000.0
_CHANNELS = 2
_DURATION_S = 0.25
_N_SAMPLES = int(_SR * _DURATION_S)


def _fake_audio(channels: int = _CHANNELS, n: int = _N_SAMPLES) -> np.ndarray:
    """Return a deterministic non-silent signal — same shape ``render_params`` produces.

    :param channels: Number of channels to stack.
    :param n: Number of samples per channel.
    :returns: ``(channels, n)`` float32 sine wave.
    :rtype: np.ndarray
    """
    t = np.arange(n, dtype=np.float32) / _SR
    sine = (0.25 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return np.stack([sine] * channels, axis=0)


# ---------------------------------------------------------------------------
# make_spectrogram
# ---------------------------------------------------------------------------


def test_make_spectrogram_returns_one_2d_array_per_channel() -> None:
    """Stereo input → list of length 2; each entry is a 2-D mel-spec."""
    audio = _fake_audio(channels=2)
    specs = predict_vst_audio.make_spectrogram(audio, _SR)
    assert len(specs) == 2
    for spec in specs:
        assert spec.ndim == 2
        assert spec.shape[0] == predict_vst_audio._MEL_N_MELS


def test_make_spectrogram_mono_returns_single_entry() -> None:
    """Mono input → list of length 1."""
    audio = _fake_audio(channels=1)
    specs = predict_vst_audio.make_spectrogram(audio, _SR)
    assert len(specs) == 1


def test_make_spectrogram_output_is_in_decibels() -> None:
    """``power_to_db(ref=np.max)`` peaks at 0 dB and otherwise is non-positive."""
    audio = _fake_audio()
    specs = predict_vst_audio.make_spectrogram(audio, _SR)
    spec = specs[0]
    assert np.isfinite(spec).all()
    assert spec.max() == pytest.approx(0.0, abs=1e-6)
    assert spec.min() <= 0.0


def test_make_spectrogram_pure_tone_peaks_near_expected_mel_bin() -> None:
    """A pure tone peaks in a mel bin close to its frequency — guards a zeros-mutant.

    Ported from #1060; uses ``librosa.mel_frequencies`` to compute the expected
    bin under the same defaults ``make_spectrogram`` uses (``fmin=0``,
    ``fmax=sr/2``).
    """
    sr = 16000.0
    freq = 1000.0
    n = 8192
    t = np.arange(n, dtype=np.float32) / sr
    sine = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32).reshape(1, -1)
    specs = predict_vst_audio.make_spectrogram(sine, sr)
    spec = specs[0]
    peak_bin = int(np.argmax(spec.mean(axis=1)))
    mel_centers = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=sr / 2)
    expected_bin = int(np.argmin(np.abs(mel_centers - freq)))
    # Mel filterbank smears narrowband content across neighbouring bins.
    assert abs(peak_bin - expected_bin) <= 5, f"peak at bin {peak_bin}, expected ~{expected_bin}"


# ---------------------------------------------------------------------------
# write_spectrograms
# ---------------------------------------------------------------------------


def test_write_spectrograms_creates_a_nonempty_png(tmp_path: Path) -> None:
    """Side-effect contract: writes a PNG file containing both pred and target panels.

    :param tmp_path: pytest tmp dir fixture.
    """
    pred = _fake_audio()
    target = _fake_audio()
    out = tmp_path / "spec.png"

    predict_vst_audio.write_spectrograms(pred, target, _SR, str(out))

    assert out.is_file()
    assert out.stat().st_size > 0


def test_write_spectrograms_closes_figure_to_avoid_leaks(tmp_path: Path) -> None:
    """Each call closes its figure — otherwise the render loop leaks one per sample.

    Ported from #1060.

    :param tmp_path: pytest tmp dir fixture.
    """
    plt.close("all")
    predict_vst_audio.write_spectrograms(
        _fake_audio(), _fake_audio(), _SR, str(tmp_path / "spec.png")
    )
    assert plt.get_fignums() == []


def test_write_spectrograms_single_panel_does_not_crash(tmp_path: Path) -> None:
    """Mono pred + zero-channel target → single panel; ``plt.subplots(1, 1)`` returns one Axes.

    Regression guard: without the ``np.atleast_1d(axs)`` normalization in
    ``write_spectrograms``, the single-panel case TypeErrored on
    ``zip(axs, panels)`` because the function tried to iterate a scalar Axes.

    :param tmp_path: pytest tmp dir fixture.
    """
    pred = _fake_audio(channels=1)
    # An empty target produces no target panels; only the pred panel remains.
    target = np.zeros((0, _N_SAMPLES), dtype=np.float32)
    out = tmp_path / "spec_mono.png"

    predict_vst_audio.write_spectrograms(pred, target, _SR, str(out))

    assert out.is_file()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# params_to_csv
# ---------------------------------------------------------------------------


def _two_row_synth_params() -> tuple[dict[str, float], dict[str, float]]:
    pred = {"cutoff": 0.5, "resonance": 0.25}
    target = {"cutoff": 0.4, "resonance": 0.30}
    return pred, target


def _two_row_note_params() -> tuple[dict[str, float], dict[str, float]]:
    pred: dict[str, float] = {"pitch": 60.0, "velocity": 100.0}
    target: dict[str, float] = {"pitch": 60.0, "velocity": 100.0}
    return pred, target


def test_params_to_csv_writes_pred_and_target_columns(tmp_path: Path) -> None:
    """The CSV has exactly the columns ``pred`` and ``target`` and one row per param.

    :param tmp_path: pytest tmp dir fixture.
    """
    pred_synth, target_synth = _two_row_synth_params()
    pred_note, target_note = _two_row_note_params()
    out = tmp_path / "params.csv"

    predict_vst_audio.params_to_csv(
        target_synth_params=target_synth,
        target_note_params=target_note,
        pred_synth_params=pred_synth,
        pred_note_params=pred_note,
        save_path=str(out),
    )

    df = pd.read_csv(out, index_col=0)
    assert sorted(df.columns.tolist()) == ["pred", "target"]
    assert set(df.index) == {"cutoff", "resonance", "pitch", "velocity"}
    assert df.loc["cutoff", "pred"] == pytest.approx(0.5)
    assert df.loc["cutoff", "target"] == pytest.approx(0.4)


def test_params_to_csv_with_none_targets_writes_pred_only(tmp_path: Path) -> None:
    """When ``--no-params`` / no rerender, targets are ``None`` and only ``pred`` is populated.

    Pins the behavior the ``main`` flow relies on at the no-targets call site: pandas
    accepts ``None`` for the ``target`` column and renders it as an empty/NaN column.

    :param tmp_path: pytest tmp dir fixture.
    """
    pred_synth, _ = _two_row_synth_params()
    pred_note, _ = _two_row_note_params()
    out = tmp_path / "params.csv"

    predict_vst_audio.params_to_csv(
        target_synth_params=None,
        target_note_params=None,
        pred_synth_params=pred_synth,
        pred_note_params=pred_note,
        save_path=str(out),
    )

    df = pd.read_csv(out, index_col=0)
    assert "pred" in df.columns
    assert "target" in df.columns
    assert bool(df["target"].isna().all())


# ---------------------------------------------------------------------------
# main() — CLI integration
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_pred_dir(tmp_path: Path) -> Path:
    """Create a ``pred-N.pt`` / ``target-audio-N.pt`` / ``target-params-N.pt`` set.

    Builds a small surge_xt-shaped param vector (length matches the registered
    spec) so ``param_spec.decode`` works in-process.

    :param tmp_path: pytest tmp dir fixture.
    :returns: Path to the populated pred directory.
    :rtype: Path
    """
    pred_dir = tmp_path / "pred_dir"
    pred_dir.mkdir()

    spec = param_specs["surge_xt"]
    param_len = len(spec)
    batch_size = 2

    # Values in [-1, 1] — main() rescales to [0, 1] before decode.
    pred_params = torch.zeros((batch_size, param_len), dtype=torch.float32)
    target_params = torch.zeros((batch_size, param_len), dtype=torch.float32)
    target_audio = torch.from_numpy(np.stack([_fake_audio() for _ in range(batch_size)], axis=0))

    torch.save(pred_params, pred_dir / "pred-0.pt")
    torch.save(target_params, pred_dir / "target-params-0.pt")
    torch.save(target_audio, pred_dir / "target-audio-0.pt")
    return pred_dir


@pytest.fixture
def patched_render(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace ``render_params`` with a recorder returning deterministic audio.

    :param monkeypatch: pytest monkeypatch fixture.
    :returns: A list that accumulates one entry per call to the patched renderer.
    :rtype: list[dict]
    """
    calls: list[dict] = []

    def _fake_render(
        plugin_path: str,
        params: dict[str, float],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        signal_duration_seconds: float,
        sample_rate: float,
        channels: int,
        preset_path: str | None = None,
    ) -> np.ndarray:
        calls.append(
            {
                "plugin_path": plugin_path,
                "midi_note": midi_note,
                "preset_path": preset_path,
            }
        )
        return _fake_audio(channels=channels)

    monkeypatch.setattr(predict_vst_audio, "render_params", _fake_render)
    return calls


def _invoke_main(args: list[str]) -> Result:
    """Wrap ``CliRunner.invoke`` with ``standalone_mode=False`` so exceptions surface.

    :param args: argv list passed to ``predict_vst_audio.main``.
    :returns: The click ``Result`` object.
    :rtype: Result
    """
    runner = CliRunner()
    return runner.invoke(predict_vst_audio.main, args, standalone_mode=False)


def test_main_writes_pred_and_target_wav_per_sample(
    fake_pred_dir: Path,
    tmp_path: Path,
    patched_render: list[dict],
) -> None:
    """End-to-end: ``main`` materializes ``sample_N/{pred,target}.wav`` for each row.

    :param fake_pred_dir: pre-populated pred directory.
    :param tmp_path: pytest tmp dir fixture.
    :param patched_render: render-recorder fixture.
    """
    out_dir = tmp_path / "out"
    result = _invoke_main(
        [
            str(fake_pred_dir),
            str(out_dir),
            "--skip-spectrogram",
            "--no-params",
        ]
    )
    assert result.exit_code == 0, result.output

    for i in (0, 1):
        sample_dir = out_dir / f"sample_{i}"
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "target.wav").is_file()


def test_main_writes_spectrogram_when_enabled(
    fake_pred_dir: Path,
    tmp_path: Path,
    patched_render: list[dict],
) -> None:
    """Default (no ``--skip-spectrogram``) writes ``spec.png`` per sample.

    :param fake_pred_dir: pre-populated pred directory.
    :param tmp_path: pytest tmp dir fixture.
    :param patched_render: render-recorder fixture.
    """
    out_dir = tmp_path / "out"
    result = _invoke_main(
        [
            str(fake_pred_dir),
            str(out_dir),
            "--no-params",
        ]
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "sample_0" / "spec.png").is_file()


def test_main_rerender_target_calls_render_for_pred_and_target(
    fake_pred_dir: Path,
    tmp_path: Path,
    patched_render: list[dict],
) -> None:
    """``--rerender_target`` doubles the render-param calls (one pred + one target per row).

    :param fake_pred_dir: pre-populated pred directory.
    :param tmp_path: pytest tmp dir fixture.
    :param patched_render: render-recorder fixture.
    """
    out_dir = tmp_path / "out"
    result = _invoke_main(
        [
            str(fake_pred_dir),
            str(out_dir),
            "--skip-spectrogram",
            "--rerender_target",
        ]
    )
    assert result.exit_code == 0, result.output
    # 2 rows × (pred + target) = 4 calls
    assert len(patched_render) == 4


def test_main_writes_params_csv_when_rerender_target(
    fake_pred_dir: Path,
    tmp_path: Path,
    patched_render: list[dict],
) -> None:
    """``--rerender_target`` populates both ``pred`` and ``target`` columns of the CSV.

    :param fake_pred_dir: pre-populated pred directory.
    :param tmp_path: pytest tmp dir fixture.
    :param patched_render: render-recorder fixture.
    """
    out_dir = tmp_path / "out"
    result = _invoke_main(
        [
            str(fake_pred_dir),
            str(out_dir),
            "--skip-spectrogram",
            "--rerender_target",
        ]
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(out_dir / "sample_0" / "params.csv", index_col=0)
    assert {"pred", "target"} <= set(df.columns)
    assert bool(df["target"].notna().any())


def test_main_writes_params_csv_when_no_params(
    fake_pred_dir: Path,
    tmp_path: Path,
    patched_render: list[dict],
) -> None:
    """``--no-params`` still writes a params CSV (with empty target column).

    :param fake_pred_dir: pre-populated pred directory.
    :param tmp_path: pytest tmp dir fixture.
    :param patched_render: render-recorder fixture.
    """
    out_dir = tmp_path / "out"
    result = _invoke_main(
        [
            str(fake_pred_dir),
            str(out_dir),
            "--skip-spectrogram",
            "--no-params",
        ]
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(out_dir / "sample_0" / "params.csv", index_col=0)
    assert bool(df["target"].isna().all())


def test_main_target_params_present_but_no_rerender_does_not_crash(
    fake_pred_dir: Path,
    tmp_path: Path,
    patched_render: list[dict],
) -> None:
    """Targets on disk + ``rerender_target=False`` must not crash.

    Regression guard for the latent ``NameError`` at the old ``params_to_csv`` call
    site: when ``rerender_target=False`` and ``target_params is not None``,
    ``target_synth_params``/``target_note_params`` were never bound in the loop
    iteration but were still referenced. The fix decouples the CSV column from
    the rerender flag — see PR description.

    :param fake_pred_dir: pre-populated pred directory.
    :param tmp_path: pytest tmp dir fixture.
    :param patched_render: render-recorder fixture.
    """
    out_dir = tmp_path / "out"
    result = _invoke_main(
        [
            str(fake_pred_dir),
            str(out_dir),
            "--skip-spectrogram",
        ]
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(out_dir / "sample_0" / "params.csv", index_col=0)
    # The original target-on-disk row should now produce a populated target
    # column (no NameError, CSV still emitted).
    assert bool(df["target"].notna().any())


def test_main_handles_multiple_pred_files_in_numeric_not_lex_order(
    tmp_path: Path,
    patched_render: list[dict],
) -> None:
    """File indices are sorted numerically (not lex), and offsets accumulate.

    Non-monotonic file creation (``pred-10``, ``pred-2``, ``pred-0``) tests the
    ``_list_pred_files`` sort key: lexicographic order would produce
    ``[pred-0, pred-10, pred-2]`` whereas the contract is ``[pred-0, pred-2, pred-10]``.
    With three pred files of 2 rows each, ``sample_0..sample_5`` must all exist.

    :param tmp_path: pytest tmp dir fixture.
    :param patched_render: render-recorder fixture.
    """
    pred_dir = tmp_path / "pred_dir"
    pred_dir.mkdir()

    spec = param_specs["surge_xt"]
    # Deliberately create files out of order to confirm the sort key.
    for i in (10, 2, 0):
        pred_params = torch.zeros((2, len(spec)), dtype=torch.float32)
        target_audio = torch.from_numpy(np.stack([_fake_audio()] * 2, axis=0))
        torch.save(pred_params, pred_dir / f"pred-{i}.pt")
        torch.save(target_audio, pred_dir / f"target-audio-{i}.pt")

    out_dir = tmp_path / "out"
    result = _invoke_main(
        [
            str(pred_dir),
            str(out_dir),
            "--skip-spectrogram",
            "--no-params",
        ]
    )
    assert result.exit_code == 0, result.output

    # Three pred files × two rows each → samples 0..5.
    for i in range(6):
        assert (out_dir / f"sample_{i}" / "pred.wav").is_file()
