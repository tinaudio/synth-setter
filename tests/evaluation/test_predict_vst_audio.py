"""Unit tests for ``synth_setter.evaluation.predict_vst_audio``.

Covers the three pure helpers (``make_spectrogram``, ``write_spectrograms``,
``params_to_csv``) and the click ``main`` entrypoint with the VST3 render call
patched out — so the suite stays CPU-only and deterministic and runs under
``make test-fast``.
"""

from __future__ import annotations

import os

# Pin the headless backend before ``predict_vst_audio`` triggers ``pyplot`` import.
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from click.testing import CliRunner, Result  # noqa: E402

from synth_setter.data.vst import param_specs  # noqa: E402
from synth_setter.data.vst.param_spec import NoteParams  # noqa: E402
from synth_setter.evaluation import predict_vst_audio  # noqa: E402
from synth_setter.evaluation.predict_vst_audio import (  # noqa: E402
    main,
    make_spectrogram,
    params_to_csv,
    write_spectrograms,
)
from tests.helpers.audio_utils import noise as _noise  # noqa: E402
from tests.helpers.audio_utils import sine  # noqa: E402

_SR = 8000.0
_PARAM_SPEC_NAME = "surge_simple"
_PARAM_SPEC = param_specs[_PARAM_SPEC_NAME]
_CHANNELS = 2
_SAMPLES = 1024


def _sine(channels: int, samples: int, *, freq: float, sr: float) -> np.ndarray:
    return sine(freq=freq, channels=channels, sr=sr, samples=samples)


# ---------- make_spectrogram ----------


def test_make_spectrogram_returns_one_db_array_per_channel() -> None:
    """Runtime contract: returns ``list[ndarray]`` despite the source's ``np.ndarray`` annotation."""
    specs = make_spectrogram(_noise(channels=2, samples=4096), _SR)

    assert isinstance(specs, list)
    assert len(specs) == 2
    for spec in specs:
        assert spec.shape[0] == 128
        assert spec.max() <= 0.0


def test_make_spectrogram_mono_input_returns_singleton_list() -> None:
    """Mono ``(1, N)`` input → one-element list, not a bare array."""
    specs = make_spectrogram(_noise(channels=1, samples=4096), _SR)

    assert isinstance(specs, list)
    assert len(specs) == 1


def test_make_spectrogram_pure_tone_peaks_near_expected_mel_bin() -> None:
    """A 1 kHz sine should peak in a mel bin close to 1 kHz — guards against a zeros-mutant."""
    import librosa

    sr = 44100.0
    freq = 1000.0
    specs = make_spectrogram(_sine(channels=1, samples=8192, freq=freq, sr=sr), sr)
    spec = specs[0]
    peak_bin = int(np.argmax(spec.mean(axis=1)))
    # Match the melspectrogram defaults (fmin=0, fmax=sr/2) so we resolve the same bin grid.
    mel_centers = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=sr / 2)
    expected_bin = int(np.argmin(np.abs(mel_centers - freq)))
    # Allow a few bins of slop — the mel filterbank smears narrowband content across neighbours.
    assert abs(peak_bin - expected_bin) <= 5, f"peak at bin {peak_bin}, expected ~{expected_bin}"


# ---------- write_spectrograms ----------


def test_write_spectrograms_writes_png_to_disk(tmp_path: Path) -> None:
    """A non-empty PNG (PNG magic bytes) should appear at ``save_path``.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    out = tmp_path / "spec.png"

    write_spectrograms(_noise(2, 4096, seed=1), _noise(2, 4096, seed=2), _SR, str(out))

    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_spectrograms_closes_figure_to_avoid_leaks(tmp_path: Path) -> None:
    """Each call closes its figure — otherwise the render loop leaks one per sample.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    plt.close("all")
    write_spectrograms(
        _noise(2, 4096, seed=1), _noise(2, 4096, seed=2), _SR, str(tmp_path / "spec.png")
    )

    assert plt.get_fignums() == []


# ---------- params_to_csv ----------


def _sample_param_dicts(seed: int = 0) -> tuple[dict[str, float], NoteParams]:
    """Deterministic ``(synth_params, note_params)`` pair via ``_PARAM_SPEC.decode``.

    :param seed: Seed for the per-call RNG.
    :return: ``(synth_params, note_params)`` pair decoded from a random encoding.
    """
    rng = np.random.default_rng(seed)
    encoded = rng.random(len(_PARAM_SPEC)).astype(np.float32)
    return _PARAM_SPEC.decode(encoded)


def test_params_to_csv_writes_pred_and_target_columns(tmp_path: Path) -> None:
    """Both dicts populated → CSV holds a row per pred key with finite values in both columns.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    pred_s, pred_n = _sample_param_dicts(seed=0)
    tgt_s, tgt_n = _sample_param_dicts(seed=1)
    out = tmp_path / "params.csv"

    params_to_csv(tgt_s, tgt_n, pred_s, pred_n, str(out), _PARAM_SPEC)

    df = pd.read_csv(out, index_col=0)
    assert list(df.columns) == ["pred", "target"]
    assert set(df.index) == set(pred_s) | set(pred_n)
    assert bool(df["pred"].notna().all())
    assert bool(df["target"].notna().all())


def test_params_to_csv_none_target_leaves_target_column_nan(tmp_path: Path) -> None:
    """``None`` target params (the CLI's ``--no-params`` path) leave an all-NaN target column.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    pred_s, pred_n = _sample_param_dicts()
    out = tmp_path / "params.csv"

    params_to_csv(None, None, pred_s, pred_n, str(out), _PARAM_SPEC)

    df = pd.read_csv(out, index_col=0)
    assert bool(df["pred"].notna().all())
    assert bool(df["target"].isna().all())


# ---------- main (click CLI) ----------


def _fake_render(*_args: object, **_kwargs: object) -> np.ndarray:
    """Stand-in for ``render_params`` — only the ``(channels, samples)`` shape contract matters.

    :param \\*_args: Ignored positional arguments forwarded by callers.
    :param \\*\\*_kwargs: Ignored keyword arguments forwarded by callers.
    :return: ``(_CHANNELS, _SAMPLES)`` float32 audio array.
    """
    rng = np.random.default_rng(42)
    return rng.standard_normal((_CHANNELS, _SAMPLES)).astype(np.float32)


def _write_batch(
    pred_dir: Path,
    *,
    index: int,
    batch_size: int,
    with_target_params: bool,
) -> None:
    """Write the ``.pt`` files one ``PredictionWriter`` batch would produce.

    :param pred_dir: Destination directory for the ``.pt`` files.
    :param index: Batch index that becomes the ``pred-<index>.pt`` suffix.
    :param batch_size: Number of rows per batch tensor.
    :param with_target_params: When True, also write ``target-params-<index>.pt``.
    """
    rng = np.random.default_rng(index)
    # ``main`` rescales pred params via ``(x + 1) / 2`` — so the fixture must live on [-1, 1].
    encoded = (rng.random((batch_size, len(_PARAM_SPEC))) * 2 - 1).astype(np.float32)
    torch.save(torch.from_numpy(encoded), pred_dir / f"pred-{index}.pt")

    target_audio = rng.standard_normal((batch_size, _CHANNELS, _SAMPLES)).astype(np.float32)
    torch.save(torch.from_numpy(target_audio), pred_dir / f"target-audio-{index}.pt")

    if with_target_params:
        torch.save(torch.from_numpy(encoded.copy()), pred_dir / f"target-params-{index}.pt")


@pytest.fixture()
def runner() -> CliRunner:
    """Fresh click ``CliRunner`` per test.

    :return: A fresh ``CliRunner`` instance.
    """
    return CliRunner()


@pytest.fixture()
def pred_dir(tmp_path: Path) -> Path:
    """Empty ``preds/`` subdirectory ready for ``_write_batch`` calls.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :return: Path of the created ``preds/`` directory.
    """
    d = tmp_path / "preds"
    d.mkdir()
    return d


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    """``out/`` path the CLI will create.

    Not pre-created — the CLI does that.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :return: Path of the planned ``out/`` directory.
    """
    return tmp_path / "out"


@pytest.fixture(autouse=True)
def _patch_render_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the VST3 render call with the in-process ``_fake_render`` stub.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
    monkeypatch.setattr(predict_vst_audio, "render_params", _fake_render)


def _invoke_main(runner: CliRunner, pred_dir: Path, out_dir: Path, *extra: str) -> Result:
    """Invoke ``main`` with the standard small-audio test options plus any ``extra`` flags.

    :param runner: Click ``CliRunner`` driving the invocation.
    :param pred_dir: Directory passed as the first CLI positional.
    :param out_dir: Directory passed as the second CLI positional.
    :param \\*extra: Additional CLI flags appended verbatim.
    :return: The ``Result`` produced by ``runner.invoke``.
    """
    return runner.invoke(
        main,
        [
            str(pred_dir),
            str(out_dir),
            f"--param_spec={_PARAM_SPEC_NAME}",
            f"--sample_rate={int(_SR)}",
            f"--channels={_CHANNELS}",
            "--signal_duration_seconds=0.1",
            *extra,
        ],
        catch_exceptions=False,
    )


def test_main_no_params_writes_pred_target_csv_and_spectrogram(
    runner: CliRunner, pred_dir: Path, out_dir: Path
) -> None:
    """``--no-params`` path produces pred.wav, target.wav, spec.png, and params.csv per sample.

    :param runner: Parametrized ``runner`` value under test.
    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """
    _write_batch(pred_dir, index=0, batch_size=2, with_target_params=False)

    result = _invoke_main(runner, pred_dir, out_dir, "--no-params")

    assert result.exit_code == 0, result.output
    for j in range(2):
        sample_dir = out_dir / f"sample_{j}"
        for name in ("pred.wav", "target.wav", "spec.png", "params.csv"):
            assert (sample_dir / name).is_file(), f"missing {name} under {sample_dir}"


def test_main_skip_spectrogram_suppresses_png(
    runner: CliRunner, pred_dir: Path, out_dir: Path
) -> None:
    """``--skip-spectrogram`` keeps the wav/csv outputs but skips the matplotlib render.

    :param runner: Parametrized ``runner`` value under test.
    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """
    _write_batch(pred_dir, index=0, batch_size=1, with_target_params=False)
    plt.close("all")

    result = _invoke_main(runner, pred_dir, out_dir, "--no-params", "--skip-spectrogram")

    assert result.exit_code == 0, result.output
    sample = out_dir / "sample_0"
    assert (sample / "pred.wav").is_file()
    assert (sample / "target.wav").is_file()
    assert (sample / "params.csv").is_file()
    assert not (sample / "spec.png").exists()
    # Stronger guarantee than the missing-file assert: no matplotlib figure was ever created.
    assert plt.get_fignums() == []


def test_main_rerender_target_renders_pred_and_target_per_sample(
    runner: CliRunner, pred_dir: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-t`` triggers a second ``render_params`` call per sample to re-synthesise the target.

    :param runner: Parametrized ``runner`` value under test.
    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
    calls: list[object] = []

    def _counting_render(*args: object, **_kwargs: object) -> np.ndarray:
        calls.append(args)
        return _fake_render()

    monkeypatch.setattr(predict_vst_audio, "render_params", _counting_render)

    batch_size = 3
    _write_batch(pred_dir, index=0, batch_size=batch_size, with_target_params=True)

    result = _invoke_main(runner, pred_dir, out_dir, "--rerender_target", "--skip-spectrogram")

    assert result.exit_code == 0, result.output
    # One render for pred + one for the re-synthesised target, per sample.
    assert len(calls) == batch_size * 2
    for j in range(batch_size):
        df = pd.read_csv(out_dir / f"sample_{j}" / "params.csv", index_col=0)
        assert bool(df["pred"].notna().all())
        assert bool(df["target"].notna().all())


def test_main_target_params_on_disk_without_rerender_does_not_crash(
    runner: CliRunner, pred_dir: Path, out_dir: Path
) -> None:
    """Targets-on-disk + ``rerender_target=False`` must complete without crashing.

    Regression guard for ``UnboundLocalError`` at the ``params_to_csv`` call
    site: when ``target-params-{i}.pt`` is present but ``--rerender_target``
    is not passed, ``target_synth_params`` / ``target_note_params`` were never
    bound but were still referenced by the ``target_params is not None`` arm
    of the call-site conditional.

    :param runner: Parametrized ``runner`` value under test.
    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """
    _write_batch(pred_dir, index=0, batch_size=2, with_target_params=True)

    result = _invoke_main(runner, pred_dir, out_dir, "--skip-spectrogram")

    assert result.exit_code == 0, result.output
    for j in range(2):
        sample_dir = out_dir / f"sample_{j}"
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "target.wav").is_file()
        assert (sample_dir / "params.csv").is_file()


def test_main_multiple_batches_produce_contiguous_sample_indices(
    runner: CliRunner, pred_dir: Path, out_dir: Path
) -> None:
    """``current_offset`` accumulates across pred files so sample dirs don't collide.

    :param runner: Parametrized ``runner`` value under test.
    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """
    _write_batch(pred_dir, index=0, batch_size=2, with_target_params=False)
    _write_batch(pred_dir, index=1, batch_size=3, with_target_params=False)

    result = _invoke_main(runner, pred_dir, out_dir, "--no-params", "--skip-spectrogram")

    assert result.exit_code == 0, result.output
    # Set compare avoids the lexicographic ``sample_10`` ordering trap once batches grow.
    sample_dirs = {d.name for d in out_dir.iterdir() if d.is_dir()}
    assert sample_dirs == {f"sample_{i}" for i in range(5)}
