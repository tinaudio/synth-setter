"""predict_vst_audio renders torchsynth-backend predictions in-process — no plugin host."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pedalboard.io import AudioFile
from pydantic_settings import CliApp

from synth_setter.data.vst import param_specs
from synth_setter.evaluation.predict_vst_audio import main
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig

_SR = 22_050.0
_DURATION_SECONDS = 4.0  # matches the spec note window — the midpoint note starts at 2.0s
_SAMPLES = int(_SR * _DURATION_SECONDS)
_CHANNELS = 2
_PARAM_SPEC_NAME = ParamSpecName("torchsynth_simple")


def _stage_predictions(pred_dir: Path, batch_size: int = 1) -> None:
    """Write one ``PredictionWriter`` batch of model-space rows for the torchsynth spec.

    Stages pred + target-params only (the ``ValAudioProbe`` layout), so the CLI's
    ``--rerender-target`` path renders both rows through the same backend.

    :param pred_dir: Destination directory for the ``.pt`` files.
    :param batch_size: Number of rows in the staged batch.
    """
    # Mid-range model-space rows decode to the spec midpoint patch — deterministic
    # and audible, unlike random rows which can decode to silent patches.
    encoded = np.zeros((batch_size, len(param_specs[_PARAM_SPEC_NAME])), dtype=np.float32)
    torch.save(torch.from_numpy(encoded), pred_dir / "pred-0.pt")
    torch.save(torch.from_numpy(encoded.copy()), pred_dir / "target-params-0.pt")


def _read_wav(path: Path) -> np.ndarray:
    """Read a rendered wav back as a ``(channels, samples)`` array.

    :param path: Wav file written by the CLI.
    :returns: Decoded audio with channels on the first axis.
    """
    with AudioFile(str(path)) as f:
        return f.read(f.frames)


def test_main_torchsynth_plugin_path_renders_without_plugin_host(tmp_path: Path) -> None:
    """The TorchSynth backend renders predictions and targets in-process.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    _stage_predictions(pred_dir)
    out_dir = tmp_path / "out"

    render_config = RenderConfig(
        plugin_path="torchsynth",
        plugin_state_path="",
        param_spec_name=_PARAM_SPEC_NAME,
        renderer_version="1.0.2",
        renderer_backend="torchsynth",
        sample_rate=int(_SR),
        channels=_CHANNELS,
        velocity=100,
        signal_duration_seconds=_DURATION_SECONDS,
        min_loudness=-55.0,
        samples_per_render_batch=1,
        samples_per_shard=1,
        gui_toggle_cadence="never",
    )
    main(
        [
            str(pred_dir),
            str(out_dir),
            *CliApp.serialize(render_config),
            "--rerender-target",
            "True",
            "--skip-spectrogram",
            "True",
        ]
    )

    sample_dir = out_dir / "sample_0"
    pred_audio = _read_wav(sample_dir / "pred.wav")
    target_audio = _read_wav(sample_dir / "target.wav")
    assert pred_audio.shape == (_CHANNELS, _SAMPLES)
    assert np.abs(pred_audio).max() > 0.0
    # Identical staged pred/target rows must render identical audio through the backend.
    np.testing.assert_array_equal(pred_audio, target_audio)
    assert (sample_dir / "params.csv").is_file()
