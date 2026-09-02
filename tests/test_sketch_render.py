"""Behavior tests for sketch-conditioned rendering."""

import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from click.testing import CliRunner, Result
from pedalboard.io import AudioFile

from synth_setter.cli import sketch_render
from synth_setter.cli.sketch_render import cfg_arm_name, cfg_grid, load_audio_file, main
from synth_setter.conditioning import SketchControlSpec
from synth_setter.data.third_party_datamodule import AudioDecodeError, decode_clip
from synth_setter.data.vst.core import write_wav
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule


def test_noise_source_device_mps_uses_supported_cpu_generator() -> None:
    """MPS sampling draws seeded noise on CPU before device transfer."""
    assert sketch_render._noise_source_device(torch.device("mps")) == torch.device("cpu")
    assert sketch_render._noise_source_device(torch.device("cuda")) == torch.device("cuda")


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


def test_cfg_grid_empty_axis_raises() -> None:
    """A Cartesian grid requires both guidance axes."""
    with pytest.raises(ValueError, match="non-empty"):
        cfg_grid([], [1.0])


def test_cfg_grid_nonfinite_strength_raises() -> None:
    """A non-finite guidance strength cannot identify a valid arm."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        cfg_grid([float("nan")], [1.0])


def test_load_audio_file_unsupported_codec_uses_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fall back to FFmpeg when pedalboard rejects an audio codec.

    :param tmp_path: Isolates the encoded source from checked-in fixtures.
    :param monkeypatch: Forces the unsupported-codec boundary and captures FFmpeg argv.
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


def test_load_audio_file_short_ffmpeg_output_pads_zero_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pad a short fallback decode on the time axis.

    :param tmp_path: Isolates the encoded source.
    :param monkeypatch: Supplies a short decoded FFmpeg stream.
    """
    source = tmp_path / "short.wv"
    source.write_bytes(b"source")
    decoded = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    monkeypatch.setattr(
        sketch_render,
        "decode_clip",
        lambda *args, **kwargs: (_ for _ in ()).throw(AudioDecodeError("not supported")),
    )
    monkeypatch.setattr(
        sketch_render.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, decoded.tobytes(), b""),
    )

    audio = load_audio_file(source, sample_rate=8000, channels=2, num_samples=4)

    assert audio.shape == (2, 4)
    np.testing.assert_array_equal(audio[:, :2], decoded.T)
    np.testing.assert_array_equal(audio[:, 2:], np.zeros((2, 2), dtype=np.float32))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for codec fallback")
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


def test_resolve_stats_local_path_returns_existing_file(tmp_path: Path) -> None:
    """A local statistics override is resolved without R2.

    :param tmp_path: Temporary statistics path.
    """
    stats = tmp_path / "stats.npz"
    stats.write_bytes(b"stats")

    digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    assert sketch_render._resolve_stats(str(stats), digest) == stats.resolve()


def test_resolve_stats_missing_local_path_raises(tmp_path: Path) -> None:
    """A missing local statistics override fails before inference.

    :param tmp_path: Temporary missing path root.
    """
    with pytest.raises(FileNotFoundError, match="mel statistics"):
        sketch_render._resolve_stats(str(tmp_path / "missing.npz"), "a" * 64)


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

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    resolved = sketch_render._resolve_stats("r2://models/stats.npz", digest)
    cached = sketch_render._resolve_stats("r2://models/stats.npz", digest)

    assert cached == resolved
    assert resolved.read_bytes() == b"mel statistics"
    assert resolved.is_relative_to(fake_r2_remote / "cache" / "synth-setter")


@pytest.mark.parametrize("digest", ["short", "z" * 64])
def test_settings_invalid_checkpoint_digest_raises(digest: str) -> None:
    """Configured checkpoint trust requires a complete hexadecimal digest.

    :param digest: Invalid digest value.
    """
    values = sketch_render._load_settings().model_dump()
    values["checkpoint_sha256"] = digest

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        sketch_render._SketchRenderSettings.model_validate(values)


def test_resolve_stats_changed_r2_bytes_refreshes_uri_cache(
    fake_r2_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new trusted digest replaces stale bytes cached under the same URI.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param monkeypatch: Cache-home override fixture.
    """
    source = fake_r2_remote / "models" / "stats.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"old stats")
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))
    old_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    sketch_render._resolve_stats("r2://models/stats.npz", old_digest)
    source.write_bytes(b"corrected stats")
    new_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    refreshed = sketch_render._resolve_stats("r2://models/stats.npz", new_digest)

    assert refreshed.read_bytes() == b"corrected stats"


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
        """Minimal device/evaluation interface returned by the patched loader.

        .. attribute :: hparams

            Compatible checkpoint metadata.

        .. attribute :: device

            Device selected by the loader.

        .. attribute :: evaluating

            Whether evaluation mode was selected.
        """

        hparams = {
            "conditioning": "mel",
            "sketch_controls": SketchControlSpec(num_frames=32),
            "param_spec": "surge_simple",
            "num_params": 92,
        }
        device: torch.device | None = None
        evaluating = False

        def to(self, device: torch.device) -> "CompatibleModel":
            """Record the requested device.

            :param device: Inference device.
            :returns: This test model.
            """
            self.device = device
            return self

        def eval(self) -> "CompatibleModel":
            """Record evaluation-mode selection.

            :returns: This test model.
            """
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
    controls = torch.full((386, 401), 0.05)
    controls[0] = 0.25
    controls[1] = 0.5
    controls[2, 0] = 0.1
    controls[2, 13] = 0.11
    monkeypatch.setattr(sketch_render, "make_spectrogram", lambda *args: np.full(shape, 4.0))
    monkeypatch.setattr(
        sketch_render,
        "extract_sketch_controls",
        lambda *args: controls,
    )
    model = cast(
        VSTFlowMatchingModule,
        SimpleNamespace(
            hparams={"sketch_controls": SketchControlSpec(num_frames=32, pitch_zero_threshold=0.1)}
        ),
    )

    batch = sketch_render._prepare_inputs(
        sketch_audio=np.zeros((2, 176400), dtype=np.float32),
        content_audio=np.zeros((2, 176400), dtype=np.float32),
        stats_path=stats,
        model=model,
        render=render,
        device=torch.device("cpu"),
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


@pytest.mark.parametrize("wrong_field", ["mean", "std"])
def test_prepare_inputs_broadcastable_mel_statistics_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong_field: str
) -> None:
    """Reject statistics that NumPy could broadcast over the mel grid.

    :param tmp_path: Temporary statistics archive.
    :param monkeypatch: Mel front-end patch fixture.
    :param wrong_field: Archive member with a broadcastable wrong shape.
    """
    shape = (2, 128, 401)
    mean: np.ndarray = np.zeros(shape, dtype=np.float32)
    std: np.ndarray = np.ones(shape, dtype=np.float32)
    if wrong_field == "mean":
        mean = mean[:1]
    else:
        std = std[:1]
    stats = tmp_path / "stats.npz"
    np.savez(stats, mean=mean, std=std)
    monkeypatch.setattr(sketch_render, "make_spectrogram", lambda *args: np.ones(shape))
    render = sketch_render._load_settings().render
    model = cast(
        VSTFlowMatchingModule,
        SimpleNamespace(hparams={"sketch_controls": SketchControlSpec(num_frames=32)}),
    )

    with pytest.raises(ValueError, match="mel statistics must match mel shape"):
        sketch_render._prepare_inputs(
            sketch_audio=np.zeros((2, 176400), dtype=np.float32),
            content_audio=np.zeros((2, 176400), dtype=np.float32),
            stats_path=stats,
            model=model,
            render=render,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--sample-steps", "0"], "sample-steps must be positive"),
        (["--upload-prefix", "local/path"], "upload-prefix must use r2://"),
        (["--checkpoint", "model.ckpt"], "checkpoint-sha256 is required"),
        (["--checkpoint-sha256", "short"], "must contain 64 hex characters"),
        (["--checkpoint-sha256", "z" * 64], "must be hexadecimal"),
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
        """Capture guidance and noise supplied by the CLI.

        .. attribute :: hparams

            Parameter width consumed by the CLI.

        .. attribute :: calls

            Guidance strengths and noise recorded for each arm.
        """

        hparams = {"num_params": 92}
        calls: list[tuple[float, float, torch.Tensor]] = []

        def sample_batch(
            self,
            batch: object,
            *,
            noise: torch.Tensor,
            content_cfg_strength: float,
            sketch_cfg_strength: float,
            sample_steps: int,
        ) -> torch.Tensor:
            """Record one arm and return a valid parameter row.

            :param batch: Prepared model input.
            :param noise: Shared initial flow state.
            :param content_cfg_strength: Content guidance scale.
            :param sketch_cfg_strength: Sketch guidance scale.
            :param sample_steps: Flow integration steps.
            :returns: One zero-valued model output row.
            """
            del batch, sample_steps
            self.calls.append((content_cfg_strength, sketch_cfg_strength, noise.clone()))
            return torch.zeros((1, 92))

    model = RecordingModel()
    monkeypatch.setattr(sketch_render, "_load_model", lambda *args: model)
    monkeypatch.setattr(
        sketch_render,
        "_prepare_inputs",
        lambda **kwargs: {"mel": torch.zeros((1, 2, 128, 401))},
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

    def invoke(seed: int, output: Path) -> Result:
        return CliRunner().invoke(
            main,
            [
                str(sketch),
                str(content),
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-sha256",
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "--stats",
                str(stats),
                "--stats-sha256",
                hashlib.sha256(stats.read_bytes()).hexdigest(),
                "--content-cfg",
                "0",
                "--content-cfg",
                "2",
                "--sketch-cfg",
                "1",
                "--sample-steps",
                "2",
                "--seed",
                str(seed),
                "--output-dir",
                str(output),
                "--device",
                "cpu",
            ],
        )

    output = tmp_path / "output"
    result = invoke(123, output)
    repeated = invoke(123, tmp_path / "repeated")
    changed = invoke(124, tmp_path / "changed")

    assert result.exit_code == 0, result.output
    assert repeated.exit_code == 0, repeated.output
    assert changed.exit_code == 0, changed.output
    assert [(content_cfg, sketch_cfg) for content_cfg, sketch_cfg, _ in model.calls] == [
        (0.0, 1.0),
        (2.0, 1.0),
    ] * 3
    assert torch.equal(model.calls[0][2], model.calls[1][2])
    assert torch.equal(model.calls[0][2], model.calls[2][2])
    assert not torch.equal(model.calls[0][2], model.calls[4][2])
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
