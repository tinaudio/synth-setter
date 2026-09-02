"""Behavior tests for sketch-conditioned rendering."""

from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import pytest

from synth_setter.cli import sketch_render
from synth_setter.cli.sketch_render import cfg_grid, load_audio_file


def test_cfg_grid_repeated_strengths_returns_argument_order_product() -> None:
    """Repeated strengths expand content-major into every requested arm."""
    assert cfg_grid([0.0, 2.0], [1.0, 3.0]) == (
        (0.0, 1.0),
        (0.0, 3.0),
        (2.0, 1.0),
        (2.0, 3.0),
    )


def test_cfg_grid_nonfinite_strength_raises() -> None:
    """A non-finite guidance strength cannot identify a valid arm."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        cfg_grid([float("nan")], [1.0])


def test_load_audio_file_unsupported_codec_uses_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codec unsupported by pedalboard falls back to FFmpeg decoding.

    :param tmp_path: Temporary encoded-audio path.
    :param monkeypatch: Decoder and subprocess patch fixture.
    """
    source = tmp_path / "gsm.wav"
    source.write_bytes(b"source")
    decoded = np.arange(12, dtype=np.float32).reshape(6, 2) / 20
    decoded[0, 0] = 1.2
    monkeypatch.setattr(
        sketch_render,
        "decode_clip",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("not supported")),
    )
    monkeypatch.setattr(
        sketch_render.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, decoded.tobytes(), b""),
    )

    audio = load_audio_file(source, sample_rate=44_100, channels=2, num_samples=4)

    assert audio.shape == (2, 4)
    np.testing.assert_array_equal(audio, np.clip(decoded[:4].T, -1.0, 1.0))
