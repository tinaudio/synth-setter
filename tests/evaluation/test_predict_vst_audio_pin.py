"""Pin tests for ``predict_vst_audio`` CLI — wandb-metrics plan Phase 0.

Locks the per-sample output layout the Phase 2 ``render_predictions`` library
extraction must preserve: every ``sample_<i>/`` directory must contain
``pred.wav``, ``target.wav``, ``params.csv`` (``target`` column is NaN when
``--no-params`` is set; the file is still written), and ``spec.png``
(unless ``--skip-spectrogram``).

``render_params`` is patched to a deterministic in-process stub so the suite
stays CPU-only and runs without a Surge XT plugin binary present — matching
the convention already used by ``tests/evaluation/test_predict_vst_audio.py``.
The contract being pinned is file *layout*, not audio content.
"""

from __future__ import annotations

import os

# Pin headless matplotlib before predict_vst_audio's transitive ``pyplot`` import.
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from synth_setter.evaluation import predict_vst_audio  # noqa: E402
from synth_setter.evaluation.predict_vst_audio import main as predict_vst_audio_main  # noqa: E402

_PARAM_SPEC_NAME = "surge_simple"
_CHANNELS = 2
_SAMPLE_RATE = 8000
_SIGNAL_DURATION_SECONDS = 0.1


def _fake_render(
    _plugin_path: object,
    _synth_params: object,
    _pitch: object,
    _velocity: object,
    _note_start_and_end: object,
    signal_duration_seconds: float,
    sample_rate: int,
    channels: int,
    **_kwargs: object,
) -> np.ndarray:
    """Deterministic ``(channels, sample_rate * signal_duration_seconds)`` float32 noise.

    Honors the CLI's ``--sample_rate`` / ``--signal_duration_seconds`` / ``--channels``
    flags so the stub output shape tracks the real ``render_params`` contract.

    :param _plugin_path: Ignored; mirrors the real ``render_params`` plugin-path arg.
    :param _synth_params: Ignored; mirrors the synth-param dict.
    :param _pitch: Ignored; mirrors the MIDI pitch arg.
    :param _velocity: Ignored; mirrors the MIDI velocity arg.
    :param _note_start_and_end: Ignored; mirrors the ``(start, end)`` tuple.
    :param signal_duration_seconds: Render length in seconds.
    :param sample_rate: Render sample rate in Hz.
    :param channels: Number of audio channels.
    :param \\*\\*_kwargs: Ignored; absorbs ``preset_path`` and any future kwargs.
    :return: ``(channels, int(sample_rate * signal_duration_seconds))`` float32 audio.
    """
    num_samples = int(sample_rate * signal_duration_seconds)
    rng = np.random.default_rng(42)
    return rng.standard_normal((channels, num_samples)).astype(np.float32)


def _invoke_predict_cli(pred_dir: Path, out_dir: Path, *extra: str) -> None:
    """Invoke the CLI on ``pred_dir`` / ``out_dir`` with the small-fixture flag set.

    :param pred_dir: Directory of ``pred-*.pt`` / ``target-*-*.pt`` tensors.
    :param out_dir: CLI-created output directory.
    :param \\*extra: Additional CLI flags appended verbatim.
    """
    runner = CliRunner()
    result = runner.invoke(
        predict_vst_audio_main,
        [
            str(pred_dir),
            str(out_dir),
            f"--param_spec={_PARAM_SPEC_NAME}",
            f"--sample_rate={_SAMPLE_RATE}",
            f"--channels={_CHANNELS}",
            f"--signal_duration_seconds={_SIGNAL_DURATION_SECONDS}",
            *extra,
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output


@pytest.fixture(autouse=True)
def _patch_render_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real VST render with the in-process stub.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
    monkeypatch.setattr(predict_vst_audio, "render_params", _fake_render)


def test_cli_writes_expected_wav_layout(fixture_pred_dir: Path, tmp_path: Path) -> None:
    """Every input row produces a ``sample_<i>/`` dir with both ``pred.wav`` and ``target.wav``.

    Asserts the *negative* effects of the two suppression flags as well: under
    ``--skip-spectrogram`` no ``spec.png`` appears, and under ``--no-params`` the
    ``target`` column in ``params.csv`` is NaN (the file itself is still written —
    pinned by ``test_cli_writes_params_csv``-adjacent contracts).

    :param fixture_pred_dir: Session-scoped pred-tensor dir from ``conftest.py``.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    out_dir = tmp_path / "out"

    _invoke_predict_cli(fixture_pred_dir, out_dir, "--skip-spectrogram", "--no-params")

    sample_dirs = sorted(d for d in out_dir.iterdir() if d.is_dir())
    assert [d.name for d in sample_dirs] == ["sample_0", "sample_1"]
    for sample in sample_dirs:
        assert (sample / "pred.wav").is_file(), f"missing pred.wav under {sample}"
        assert (sample / "target.wav").is_file(), f"missing target.wav under {sample}"
        assert not (sample / "spec.png").exists(), (
            f"--skip-spectrogram should suppress spec.png; found at {sample / 'spec.png'}"
        )
        params_df = pd.read_csv(sample / "params.csv", index_col=0)
        assert bool(params_df["target"].isna().all()), (
            f"--no-params should leave the target column NaN; got {params_df['target'].tolist()}"
        )


def test_cli_writes_params_csv(fixture_pred_dir: Path, tmp_path: Path) -> None:
    """Default-flag path writes a per-sample ``params.csv`` with the ``pred``/``target`` columns.

    :param fixture_pred_dir: Session-scoped pred-tensor dir from ``conftest.py``.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    out_dir = tmp_path / "out"

    _invoke_predict_cli(fixture_pred_dir, out_dir, "--skip-spectrogram")

    for sample_name in ("sample_0", "sample_1"):
        csv_path = out_dir / sample_name / "params.csv"
        assert csv_path.is_file(), f"missing params.csv at {csv_path}"
        df = pd.read_csv(csv_path, index_col=0)
        assert list(df.columns) == ["pred", "target"]


def test_cli_writes_spectrogram_when_enabled(fixture_pred_dir: Path, tmp_path: Path) -> None:
    """``spec.png`` lands per sample when the spectrogram flag is on (default).

    :param fixture_pred_dir: Session-scoped pred-tensor dir from ``conftest.py``.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    out_dir = tmp_path / "out"

    _invoke_predict_cli(fixture_pred_dir, out_dir, "--no-params")

    for sample_name in ("sample_0", "sample_1"):
        png = out_dir / sample_name / "spec.png"
        assert png.is_file(), f"missing spec.png at {png}"
        # PNG magic-byte check guards against the file-handle leaking an empty file.
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
