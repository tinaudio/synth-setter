"""Behavior tests for sketch-conditioned rendering."""

import csv
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from pedalboard.io import AudioFile

from synth_setter.cli import sketch_render
from synth_setter.cli.sketch_render import cfg_arm_name, cfg_grid, load_audio_file, main
from synth_setter.conditioning import SketchControlSpec
from synth_setter.data.third_party_datamodule import AudioDecodeError, decode_clip
from synth_setter.data.vst.core import write_wav
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule


def test_cfg_grid_repeated_strengths_returns_argument_order_product() -> None:
    """Repeated strengths expand content-major into every requested arm."""
    assert cfg_grid([0.0, 2.0], [1.0, 3.0]) == (
        (0.0, 1.0),
        (0.0, 3.0),
        (2.0, 1.0),
        (2.0, 3.0),
    )


def test_cfg_arm_name_close_strengths_remain_distinct() -> None:
    """Distinct representable CFG values cannot alias one artifact directory."""
    assert cfg_arm_name(1.0000001, 1.0) != cfg_arm_name(1.0000002, 1.0)
    assert cfg_arm_name(1.0, 2.0) == "cfg-c1-s2"


def test_cfg_grid_duplicate_arm_raises() -> None:
    """Repeated identical strengths cannot request the same arm twice."""
    with pytest.raises(ValueError, match="duplicate"):
        cfg_grid([1.0, 1.0], [2.0])


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
        lambda *args, **kwargs: (_ for _ in ()).throw(AudioDecodeError("not supported")),
    )
    command: list[str] = []

    def run_ffmpeg(args: list[str], **kwargs: object) -> CompletedProcess[bytes]:
        del kwargs
        command.extend(args)
        return CompletedProcess(args, 0, decoded.tobytes(), b"")

    monkeypatch.setattr(sketch_render.subprocess, "run", run_ffmpeg)

    audio = load_audio_file(source, sample_rate=44_100, channels=2, num_samples=4)

    assert command[command.index("-t") + 1] == str(4 / 44_100)
    assert audio.shape == (2, 4)
    np.testing.assert_array_equal(audio, np.clip(decoded[:4].T, -1.0, 1.0))


def test_load_audio_file_real_wavpack_uses_ffmpeg_fallback() -> None:
    """A real unsupported WavPack fixture decodes through the shipped FFmpeg path."""
    source = Path(__file__).parent / "fixtures" / "audio" / "ffmpeg-fallback.wv"
    expected = (
        np.array(
            [
                (-30000, -20000, -10000, 0, 10000, 20000, 30000, 12345),
                (30000, 20000, 10000, 0, -10000, -20000, -30000, -12345),
            ],
            dtype=np.float32,
        )
        / 32768
    )
    with pytest.raises(AudioDecodeError):
        decode_clip(
            source.read_bytes(), sample_rate=8000, channels=2, num_samples=8, amplitude_scale=1.0
        )

    audio = load_audio_file(source, sample_rate=8000, channels=2, num_samples=8)

    np.testing.assert_array_equal(audio, expected)


def test_load_audio_file_contract_error_does_not_run_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render-contract failures are not reinterpreted by another decoder.

    :param tmp_path: Temporary encoded-audio path.
    :param monkeypatch: Decoder and subprocess patch fixture.
    """
    source = tmp_path / "loud.wav"
    source.write_bytes(b"source")
    monkeypatch.setattr(
        sketch_render,
        "decode_clip",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("outside [-1, 1]")),
    )
    monkeypatch.setattr(
        sketch_render.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("FFmpeg should not run"),
    )

    with pytest.raises(ValueError, match="outside"):
        load_audio_file(source, sample_rate=44_100, channels=2, num_samples=4)


def test_load_audio_file_pcm_uses_primary_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A supported file returns pedalboard decoding without spawning FFmpeg.

    :param tmp_path: Temporary encoded-audio path.
    :param monkeypatch: Decoder and subprocess patch fixture.
    """
    source = tmp_path / "pcm.wav"
    source.write_bytes(b"source")
    expected = np.ones((2, 4), dtype=np.float32)
    monkeypatch.setattr(sketch_render, "decode_clip", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        sketch_render.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("FFmpeg should not run"),
    )

    assert load_audio_file(source, sample_rate=44_100, channels=2, num_samples=4) is expected


def test_resolve_stats_r2_uri_materializes_cached_bytes(
    fake_r2_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2 mel statistics are downloaded through rclone into the model cache.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param monkeypatch: Cache-home override fixture.
    """
    source = fake_r2_remote / "models" / "stats.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"mel statistics")
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))

    resolved = sketch_render._resolve_stats("r2://models/stats.npz")

    assert resolved.read_bytes() == b"mel statistics"
    assert resolved.is_relative_to(fake_r2_remote / "cache" / "synth-setter")


def test_load_model_matching_checkpoint_returns_evaluation_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compatible checkpoint is moved to the requested evaluation device.

    :param tmp_path: Temporary checkpoint path.
    :param monkeypatch: Lightning checkpoint loader patch fixture.
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    render = sketch_render._load_settings().render

    class CompatibleModel:
        hparams = {
            "conditioning": "mel",
            "sketch_controls": SketchControlSpec(num_frames=32),
            "param_spec": "surge_simple",
            "num_params": 92,
        }
        device: torch.device | None = None
        evaluating = False

        def to(self, device: torch.device) -> "CompatibleModel":
            self.device = device
            return self

        def eval(self) -> "CompatibleModel":
            self.evaluating = True
            return self

    model = CompatibleModel()
    monkeypatch.setattr(
        sketch_render.VSTFlowMatchingModule,
        "load_from_checkpoint",
        lambda *args, **kwargs: model,
    )

    loaded = sketch_render._load_model(checkpoint, render, torch.device("cpu"))

    assert loaded is model
    assert model.device == torch.device("cpu")
    assert model.evaluating


def test_load_model_matching_width_wrong_param_spec_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal vector widths cannot substitute different parameter semantics.

    :param tmp_path: Temporary checkpoint path.
    :param monkeypatch: Lightning checkpoint loader patch fixture.
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    render = sketch_render._load_settings().render
    model = SimpleNamespace(
        hparams={
            "conditioning": "mel",
            "sketch_controls": SketchControlSpec(num_frames=32),
            "param_spec": "surge_4",
            "num_params": 92,
        }
    )
    monkeypatch.setattr(
        sketch_render.VSTFlowMatchingModule,
        "load_from_checkpoint",
        lambda *args, **kwargs: model,
    )

    with pytest.raises(ValueError, match="surge_4.*surge_simple"):
        sketch_render._load_model(checkpoint, render, torch.device("cpu"))


def test_prepare_inputs_normalizes_mel_and_zeros_weak_pitch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Content and sketch waveforms become the checkpoint's two input streams.

    :param tmp_path: Temporary mel-statistics path.
    :param monkeypatch: Feature extractor patch fixture.
    """
    render = sketch_render._load_settings().render
    shape = (2, 128, 401)
    stats = tmp_path / "stats.npz"
    np.savez(
        stats,
        mean=np.full(shape, 2.0, dtype=np.float32),
        std=np.full(shape, 2.0, dtype=np.float32),
    )
    controls = torch.full((1, 386, 32), 0.05)
    controls[:, 0] = 0.25
    controls[:, 1] = 0.5
    controls[:, 2, 0] = 0.1
    controls[:, 2, 1] = 0.11
    monkeypatch.setattr(sketch_render, "make_spectrogram", lambda *args: np.full(shape, 4.0))
    monkeypatch.setattr(
        sketch_render,
        "extract_sketch_controls",
        lambda *args: torch.ones((386, 401)),
    )
    monkeypatch.setattr(sketch_render, "pool_sketch_controls", lambda *args: controls)
    model = cast(
        VSTFlowMatchingModule,
        SimpleNamespace(
            hparams={"sketch_controls": SketchControlSpec(num_frames=32, pitch_zero_threshold=0.1)}
        ),
    )

    batch = sketch_render._prepare_inputs(
        np.zeros((2, 176400), dtype=np.float32),
        np.zeros((2, 176400), dtype=np.float32),
        stats,
        model,
        render,
        torch.device("cpu"),
    )

    assert batch["mel"].shape == (1, *shape)
    assert batch["mel"].dtype is torch.float32
    assert torch.equal(batch["mel"], torch.ones((1, *shape)))
    assert batch["sketch_ctrl"].shape == (1, 386, 32)
    assert batch["sketch_ctrl"].dtype is torch.float32
    assert torch.equal(batch["sketch_ctrl"][:, 0], torch.full((1, 32), 0.25))
    assert torch.equal(batch["sketch_ctrl"][:, 1], torch.full((1, 32), 0.5))
    assert batch["sketch_ctrl"][0, 2, 0] == pytest.approx(0.1)
    assert batch["sketch_ctrl"][0, 2, 1] == pytest.approx(0.11)
    assert torch.count_nonzero(batch["sketch_ctrl"][:, 2:, 2:]) == 0


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--sample-steps", "0"], "sample-steps must be positive"),
        (["--upload-prefix", "local/path"], "upload-prefix must use r2://"),
        (
            ["--upload-prefix", "r2://bucket/path", "--no-upload"],
            "upload-prefix cannot be combined",
        ),
    ],
)
def test_cli_inconsistent_options_fail_before_inference(
    tmp_path: Path, extra_args: list[str], message: str
) -> None:
    """Contradictory CLI controls fail without loading the model.

    :param tmp_path: Temporary argument paths.
    :param extra_args: Invalid option combination.
    :param message: Expected validation detail.
    """
    sketch = tmp_path / "sketch.wav"
    content = tmp_path / "content.wav"
    sketch.touch()
    content.touch()

    result = CliRunner().invoke(main, [str(sketch), str(content), *extra_args])

    assert result.exit_code != 0
    assert message in result.output


def test_cli_local_grid_writes_every_arm_with_shared_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public CLI renders a Cartesian grid without changing pair noise.

    :param tmp_path: Temporary input and output paths.
    :param monkeypatch: Model, feature, renderer, and metric boundary patches.
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    stats = tmp_path / "stats.npz"
    stats.write_bytes(b"stats")
    sketch = tmp_path / "sketch.wav"
    content = tmp_path / "content.wav"
    silence = np.zeros((2, 176400), dtype=np.float32)
    write_wav(silence, str(sketch), 44_100, 2)
    write_wav(silence, str(content), 44_100, 2)

    class RecordingModel:
        hparams = {"num_params": 92}
        calls: list[tuple[float, float, torch.Tensor]] = []

        def sample_batch(self, batch: object, **kwargs: object) -> torch.Tensor:
            del batch
            noise = kwargs["noise"]
            assert isinstance(noise, torch.Tensor)
            content_strength = kwargs["content_cfg_strength"]
            sketch_strength = kwargs["sketch_cfg_strength"]
            assert isinstance(content_strength, float)
            assert isinstance(sketch_strength, float)
            self.calls.append((content_strength, sketch_strength, noise.clone()))
            return torch.zeros((1, 92))

    model = RecordingModel()
    monkeypatch.setattr(sketch_render, "_load_model", lambda *args: model)
    monkeypatch.setattr(
        sketch_render,
        "_prepare_inputs",
        lambda *args: {"mel": torch.zeros((1, 2, 128, 401))},
    )
    monkeypatch.setattr(sketch_render, "_resolve_stats", lambda *args: stats)
    monkeypatch.setattr(
        sketch_render,
        "_render_wav",
        lambda prediction, render, path: write_wav(
            silence, str(path), render.sample_rate, render.channels
        ),
    )
    monkeypatch.setattr(
        sketch_render,
        "params_to_csv",
        lambda *args, **kwargs: Path(args[4]).write_text("parameter,pred\n", encoding="utf-8"),
    )
    monkeypatch.setattr(
        sketch_render,
        "compute_metrics_on_dir",
        lambda *args: {"mss": 1.0, "wmfcc": 2.0, "sot": 3.0, "rms": 0.5},
    )
    output = tmp_path / "output"

    result = CliRunner().invoke(
        main,
        [
            str(sketch),
            str(content),
            "--checkpoint",
            str(checkpoint),
            "--stats",
            str(stats),
            "--content-cfg",
            "0",
            "--content-cfg",
            "2",
            "--sketch-cfg",
            "1",
            "--sample-steps",
            "2",
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [(content_cfg, sketch_cfg) for content_cfg, sketch_cfg, _ in model.calls] == [
        (0.0, 1.0),
        (2.0, 1.0),
    ]
    assert torch.equal(model.calls[0][2], model.calls[1][2])
    for arm in ("cfg-c0-s1", "cfg-c2-s1"):
        arm_dir = output / "arms" / arm
        assert {path.name for path in arm_dir.iterdir()} == {
            "metrics.csv",
            "params.csv",
            "pred.wav",
            "sketch.wav",
            "target.wav",
        }
        with (arm_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            assert next(csv.DictReader(stream))["r2_uri"] == ""
        with AudioFile(str(arm_dir / "pred.wav")) as audio:
            assert audio.frames == 176400


def test_console_script_is_installed_and_callable() -> None:
    """The package exposes the sketch renderer executable."""
    executable = Path(sys.executable).with_name("synth-setter-sketch")

    result = subprocess.run(  # noqa: S603 — fixed package entrypoint
        [str(executable), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "SKETCH_WAV CONTENT_WAV" in result.stdout
    assert "--content-cfg" in result.stdout
