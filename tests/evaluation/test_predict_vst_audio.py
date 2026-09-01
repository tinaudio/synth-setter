"""Unit tests for ``synth_setter.evaluation.predict_vst_audio``.

Covers the three pure helpers (``make_spectrogram``, ``write_spectrograms``,
``params_to_csv``) and the process CLI with renderer construction isolated for
CPU-fast artifact checks. Exact decoded values are pinned by
``tests/data/vst/test_param_spec.py``, not here — ``main`` tests assert only
file shape and finiteness.
"""

from __future__ import annotations

import os
import sys

# Pin the headless backend before ``predict_vst_audio`` triggers ``pyplot`` import.
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Literal  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from pydantic_settings import CliApp  # noqa: E402

from synth_setter.data.vst import param_specs  # noqa: E402
from synth_setter.data.vst.param_spec import NoteParams  # noqa: E402
from synth_setter.evaluation import predict_vst_audio  # noqa: E402
from synth_setter.evaluation.predict_vst_audio import (  # noqa: E402
    main,
    make_spectrogram,
    params_to_csv,
    write_spectrograms,
)
from synth_setter.param_spec_name import ParamSpecName  # noqa: E402
from synth_setter.pipeline.schemas.spec import RenderConfig  # noqa: E402
from synth_setter.synth_spec import SynthName, SynthSpec  # noqa: E402
from tests.helpers.audio_utils import noise as _noise  # noqa: E402
from tests.helpers.audio_utils import sine  # noqa: E402

_SR = 8000.0
_PARAM_SPEC_NAME = ParamSpecName("surge_simple")
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


# ---------- main (process CLI) ----------


def _render_config(
    gui_toggle_cadence: Literal["never", "once", "render"] = "never",
) -> RenderConfig:
    """Return the complete render config transported to the process CLI.

    :param gui_toggle_cadence: GUI warm-up lifecycle under test.
    :returns: Validated config for the test renderer session.
    """
    return RenderConfig(
        synth=SynthSpec(
            name=SynthName("surge_simple"),
            param_spec_name=_PARAM_SPEC_NAME,
            plugin_path="plugins/Surge XT.vst3",
            plugin_state_path="presets/surge-simple.vstpreset",
            synth_version="1.3.4",
        ),
        sample_rate=int(_SR),
        channels=_CHANNELS,
        velocity=100,
        signal_duration_seconds=0.1,
        min_loudness=-55.0,
        samples_per_render_batch=2,
        samples_per_shard=4,
        plugin_reload_cadence="render",
        gui_toggle_cadence=gui_toggle_cadence,
    )


def _fake_render(*_args: object, **_kwargs: object) -> np.ndarray:
    """Stand-in for ``AudioRenderer.render`` with the configured output shape.

    :param \\*_args: Ignored positional arguments forwarded by callers.
    :param \\*\\*_kwargs: Ignored keyword arguments forwarded by callers.
    :return: ``(_CHANNELS, _SAMPLES)`` float32 audio array.
    """
    rng = np.random.default_rng(42)
    return (0.1 * rng.standard_normal((_CHANNELS, _SAMPLES))).astype(np.float32)


def _write_batch(
    pred_dir: Path,
    *,
    index: int,
    batch_size: int,
    with_target_params: bool,
    with_target_audio: bool = True,
) -> None:
    """Write the ``.pt`` files one ``PredictionWriter`` batch would produce.

    :param pred_dir: Destination directory for the ``.pt`` files.
    :param index: Batch index that becomes the ``pred-<index>.pt`` suffix.
    :param batch_size: Number of rows per batch tensor.
    :param with_target_params: When True, also write ``target-params-<index>.pt``.
    :param with_target_audio: When False, omit ``target-audio-<index>.pt`` — the
        layout ``ValAudioProbe`` stages, since training val batches carry no raw audio.
    """
    rng = np.random.default_rng(index)
    # decode_model_output rescales pred params from [-1, 1] — the fixture must live on that range.
    encoded = (rng.random((batch_size, len(_PARAM_SPEC))) * 2 - 1).astype(np.float32)
    torch.save(torch.from_numpy(encoded), pred_dir / f"pred-{index}.pt")

    if with_target_audio:
        target_audio = rng.standard_normal((batch_size, _CHANNELS, _SAMPLES)).astype(np.float32)
        torch.save(torch.from_numpy(target_audio), pred_dir / f"target-audio-{index}.pt")

    if with_target_params:
        torch.save(torch.from_numpy(encoded.copy()), pred_dir / f"target-params-{index}.pt")


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
def fake_renderer(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace native host construction with one renderer test double.

    :param monkeypatch: Pytest fixture used to patch the factory boundary.
    :returns: Renderer whose calls remain available for behavioral assertions.
    """
    renderer = MagicMock(name="renderer")
    renderer.render.side_effect = _fake_render
    renderer.factory_configs = []

    def capture_config(config: RenderConfig) -> MagicMock:
        renderer.factory_configs.append(config)
        return renderer

    monkeypatch.setattr(predict_vst_audio, "make_audio_renderer", capture_config)
    return renderer


def _invoke_main(
    pred_dir: Path,
    out_dir: Path,
    operation_flags: tuple[str, ...],
    render_config: RenderConfig | None = None,
) -> SimpleNamespace:
    """Invoke ``main`` with a serialized complete render config.

    :param pred_dir: Directory passed as the first CLI positional.
    :param out_dir: Directory passed as the second CLI positional.
    :param operation_flags: Operation flags enabled for this invocation.
    :param render_config: Optional non-default renderer lifecycle under test.
    :returns: Success result compatible with the existing artifact assertions.
    """
    enabled = set(operation_flags)
    config = render_config if render_config is not None else _render_config()
    argv = [str(pred_dir), str(out_dir), *CliApp.serialize(config)]
    for flag in ("--rerender-target", "--no-params", "--skip-spectrogram"):
        if flag in enabled:
            argv.extend([flag, "True"])
    main(argv)
    return SimpleNamespace(exit_code=0, output="")


@pytest.mark.parametrize("amplitude", [-1.01, 1.01])
def test_main_out_of_range_render_raises_before_writing_audio(
    pred_dir: Path,
    out_dir: Path,
    fake_renderer: MagicMock,
    amplitude: float,
) -> None:
    """Evaluation rejects over-range renderer output instead of clipping it silently.

    :param pred_dir: Prediction batch destination.
    :param out_dir: Planned artifact destination.
    :param fake_renderer: Renderer boundary configured with over-range output.
    :param amplitude: Positive or negative out-of-range sample value.
    """
    _write_batch(
        pred_dir,
        index=0,
        batch_size=1,
        with_target_params=False,
    )
    fake_renderer.render.side_effect = None
    fake_renderer.render.return_value = np.full((_CHANNELS, _SAMPLES), amplitude, dtype=np.float32)

    with pytest.raises(ValueError, match=r"outside \[-1, 1\]"):
        _invoke_main(pred_dir, out_dir, ("--no-params", "--skip-spectrogram"))


def test_main_no_params_writes_pred_target_csv_and_spectrogram(
    pred_dir: Path, out_dir: Path
) -> None:
    """``--no-params`` path produces pred.wav, target.wav, spec.png, and params.csv per sample.

    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """
    _write_batch(pred_dir, index=0, batch_size=2, with_target_params=False)

    result = _invoke_main(pred_dir, out_dir, ("--no-params",))

    assert result.exit_code == 0, result.output
    for j in range(2):
        sample_dir = out_dir / f"sample_{j}"
        for name in ("pred.wav", "target.wav", "spec.png", "params.csv"):
            assert (sample_dir / name).is_file(), f"missing {name} under {sample_dir}"


def test_main_skip_spectrogram_suppresses_png(pred_dir: Path, out_dir: Path) -> None:
    """``--skip-spectrogram`` keeps the wav/csv outputs but skips the matplotlib render.

    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """
    _write_batch(pred_dir, index=0, batch_size=1, with_target_params=False)
    plt.close("all")

    result = _invoke_main(pred_dir, out_dir, ("--no-params", "--skip-spectrogram"))

    assert result.exit_code == 0, result.output
    sample = out_dir / "sample_0"
    assert (sample / "pred.wav").is_file()
    assert (sample / "target.wav").is_file()
    assert (sample / "params.csv").is_file()
    assert not (sample / "spec.png").exists()
    # Stronger guarantee than the missing-file assert: no matplotlib figure was ever created.
    assert plt.get_fignums() == []


def test_main_rerender_target_renders_pred_and_target_per_sample(
    pred_dir: Path,
    out_dir: Path,
    fake_renderer: MagicMock,
) -> None:
    """Target rerendering uses the same configured renderer session as prediction.

    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    :param fake_renderer: Renderer installed at the native-host factory boundary.
    """
    batch_size = 3
    _write_batch(pred_dir, index=0, batch_size=batch_size, with_target_params=True)

    result = _invoke_main(pred_dir, out_dir, ("--rerender-target", "--skip-spectrogram"))

    assert result.exit_code == 0, result.output
    assert len(fake_renderer.factory_configs) == 1
    factory_config = fake_renderer.factory_configs[0]
    assert factory_config.renderer_backend == _render_config().renderer_backend
    assert factory_config.plugin_reload_cadence == _render_config().plugin_reload_cadence
    assert fake_renderer.render.call_count == batch_size * 2
    for call in fake_renderer.render.call_args_list:
        assert call.args[2] == _render_config().velocity
    for j in range(batch_size):
        df = pd.read_csv(out_dir / f"sample_{j}" / "params.csv", index_col=0)
        assert bool(df["pred"].notna().all())
        assert bool(df["target"].notna().all())


@pytest.mark.parametrize(
    ("cadence", "expected_warmups"),
    [
        ("never", [False, False]),
        ("once", [True, False]),
        pytest.param(
            "render",
            [True, True],
            marks=pytest.mark.skipif(
                sys.platform == "darwin",
                reason='gui_toggle_cadence="render" is unsupported on Darwin',
            ),
        ),
    ],
)
def test_main_forwards_gui_warmup_cadence_to_renderer(
    pred_dir: Path,
    out_dir: Path,
    fake_renderer: MagicMock,
    cadence: Literal["never", "once", "render"],
    expected_warmups: list[bool],
) -> None:
    """Prediction and target renders honor capture-time GUI warm-up cadence.

    :param pred_dir: Directory for staged prediction tensors.
    :param out_dir: Destination for rendered artifacts.
    :param fake_renderer: Renderer recording production-interface calls.
    :param cadence: Configured GUI lifecycle.
    :param expected_warmups: Warm-up flags for prediction then target rendering.
    """
    _write_batch(pred_dir, index=0, batch_size=1, with_target_params=True)

    _invoke_main(
        pred_dir,
        out_dir,
        ("--rerender-target", "--skip-spectrogram"),
        _render_config(cadence),
    )

    assert [call.kwargs["warmup"] for call in fake_renderer.render.call_args_list] == (
        expected_warmups
    )


def test_main_rerender_target_accepts_float64_target_params(pred_dir: Path, out_dir: Path) -> None:
    """A float64 target-params tensor still decodes — the call site casts to float32.

    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """

    _write_batch(pred_dir, index=0, batch_size=1, with_target_params=True)
    target_path = pred_dir / "target-params-0.pt"
    torch.save(torch.load(target_path, weights_only=True).to(torch.float64), target_path)

    result = _invoke_main(pred_dir, out_dir, ("--rerender-target", "--skip-spectrogram"))

    assert result.exit_code == 0, result.output
    df = pd.read_csv(out_dir / "sample_0" / "params.csv", index_col=0)
    assert bool(df["target"].notna().all())


def test_main_target_params_on_disk_without_rerender_does_not_crash(
    pred_dir: Path, out_dir: Path
) -> None:
    """Targets-on-disk + ``rerender_target=False`` must complete without crashing.

    Regression guard for ``UnboundLocalError`` at the ``params_to_csv`` call
    site: when ``target-params-{i}.pt`` is present but ``--rerender-target``
    is not passed, ``target_synth_params`` / ``target_note_params`` were never
    bound but were still referenced by the ``target_params is not None`` arm
    of the call-site conditional.

    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """
    _write_batch(pred_dir, index=0, batch_size=2, with_target_params=True)

    result = _invoke_main(pred_dir, out_dir, ("--skip-spectrogram",))

    assert result.exit_code == 0, result.output
    for j in range(2):
        sample_dir = out_dir / f"sample_{j}"
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "target.wav").is_file()
        assert (sample_dir / "params.csv").is_file()


def test_main_multiple_batches_produce_contiguous_sample_indices(
    pred_dir: Path, out_dir: Path
) -> None:
    """``current_offset`` accumulates across pred files so sample dirs don't collide.

    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """
    _write_batch(pred_dir, index=0, batch_size=2, with_target_params=False)
    _write_batch(pred_dir, index=1, batch_size=3, with_target_params=False)

    result = _invoke_main(pred_dir, out_dir, ("--no-params", "--skip-spectrogram"))

    assert result.exit_code == 0, result.output
    # Set compare avoids the lexicographic ``sample_10`` ordering trap once batches grow.
    sample_dirs = {d.name for d in out_dir.iterdir() if d.is_dir()}
    assert sample_dirs == {f"sample_{i}" for i in range(5)}


def test_main_rerender_target_tolerates_missing_target_audio_file(
    pred_dir: Path, out_dir: Path
) -> None:
    """``--rerender-target`` renders both wavs when no ``target-audio-*.pt`` was staged.

    ``ValAudioProbe`` stages only ``pred`` and ``target-params`` tensors (training
    val batches carry no raw audio), so the rerender path must not require the
    target-audio file.

    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """
    _write_batch(pred_dir, index=0, batch_size=2, with_target_params=True, with_target_audio=False)

    result = _invoke_main(pred_dir, out_dir, ("--rerender-target",))

    assert result.exit_code == 0
    for sample in range(2):
        sample_dir = out_dir / f"sample_{sample}"
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "target.wav").is_file()
        assert (sample_dir / "spec.png").is_file()


def test_main_missing_target_audio_without_rerender_fails(pred_dir: Path, out_dir: Path) -> None:
    """Without ``--rerender-target`` there is no target source, so the CLI must fail loudly.

    :param pred_dir: Parametrized ``pred_dir`` value under test.
    :param out_dir: Parametrized ``out_dir`` value under test.
    """
    _write_batch(pred_dir, index=0, batch_size=1, with_target_params=True, with_target_audio=False)

    argv = [str(pred_dir), str(out_dir), *CliApp.serialize(_render_config())]
    with pytest.raises(ValueError, match="rerender-target"):
        main(argv)
