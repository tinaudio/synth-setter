"""Sketch-control extraction dispatches on the checkpoint's profile."""

import numpy as np
import pytest
import torch

from synth_setter.conditioning import SketchControlSpec, SketchTemporalResampling, TIVBackend
from synth_setter.data.vst.shapes import mel_hop_length
from synth_setter.features import profile_controls
from synth_setter.features.profile_controls import (
    extract_music_sketch_controls,
    extract_profile_controls,
)
from synth_setter.features.pyfdn_controls import extract_reverb_sketch
from synth_setter.features.tiv import extract_tiv_batch
from synth_setter.sketch import prepare_music_sketch_controls, resample_sketch_controls

_SAMPLE_RATE = 44_100


def _decaying_noise(seed: int, channels: int = 1) -> np.ndarray:
    """Return a synthetic exponentially decaying noise burst that resembles a reverb tail.

    :param seed: Generator seed.
    :param channels: Channel count.
    :returns: Channel-first float32 audio of four seconds.
    """
    rng = np.random.default_rng(seed)
    envelope = np.exp(-np.arange(4 * _SAMPLE_RATE) / (0.4 * _SAMPLE_RATE))
    return (rng.standard_normal((channels, envelope.size)) * envelope).astype(np.float32)


def test_reverb_profile_returns_reverb_sketch_on_control_grid() -> None:
    """The pyFDN profile feeds the mono waveform straight to the reverb-sketch extractor."""
    audio = _decaying_noise(3)
    spec = SketchControlSpec(
        profile="pyfdn_reverb", column="pyfdn_sketch", num_frames=32, num_control_tokens=32
    )
    controls = extract_profile_controls(audio, _SAMPLE_RATE, spec)
    assert controls.shape == (1, 10, 32)
    assert controls.dtype is torch.float32
    np.testing.assert_array_equal(
        controls[0].numpy(), extract_reverb_sketch(audio[0], _SAMPLE_RATE)
    )


def test_reverb_profile_multichannel_audio_rejected() -> None:
    """A stereo file cannot be silently downmixed into an impulse response."""
    spec = SketchControlSpec(
        profile="pyfdn_reverb", column="pyfdn_sketch", num_frames=32, num_control_tokens=32
    )
    with pytest.raises(ValueError, match="mono"):
        extract_profile_controls(_decaying_noise(3, channels=2), _SAMPLE_RATE, spec)


def test_music_profile_pools_tracks_and_zeros_weak_pitch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The music profile keeps the production pooling and pitch zero-binning.

    :param monkeypatch: Replaces the PESTO-backed extractor with a fixed control matrix.
    """
    controls = torch.full((386, 401), 0.05)
    controls[0] = 0.25
    controls[1] = 0.5
    # Pooling 401 source frames to 32 averages source 0-12 into output 0 and 12-25 into
    # output 1, so the spikes are sized by window mean, clear of the threshold either way.
    controls[2, :26] = 0.0
    controls[2, 0] = 0.12 * 13
    controls[2, 13] = 0.08 * 14
    monkeypatch.setattr(
        profile_controls, "extract_sketch_controls_batch", lambda *args: controls[None]
    )
    spec = SketchControlSpec(num_frames=32, pitch_zero_threshold=0.1)
    pooled = extract_profile_controls(np.zeros((2, 176400), np.float32), _SAMPLE_RATE, spec)
    assert pooled.shape == (1, 386, 32)
    assert pooled.dtype is torch.float32
    assert torch.equal(pooled[:, 0], torch.full((1, 32), 0.25))
    assert torch.equal(pooled[:, 1], torch.full((1, 32), 0.5))
    assert pooled[0, 2, 0] == pytest.approx(0.12)
    assert pooled[0, 2, 1] == 0.0
    assert torch.count_nonzero(pooled[:, 2:, 2:]) == 0


def test_music_preparation_thresholds_after_frame_resampling() -> None:
    """Pitch thresholding applies on the configured frame grid before tokenization."""
    controls = torch.zeros(1, 386, 5)
    controls[0, 2, 2] = 0.3
    spec = SketchControlSpec(
        num_frames=2,
        num_control_tokens=1,
        pitch_zero_threshold=0.15,
    )

    prepared = prepare_music_sketch_controls(controls, spec)

    assert prepared.dtype is torch.float32
    assert prepared.shape == (1, 386, 2)
    assert torch.count_nonzero(prepared) == 0


def test_music_preparation_sparser_extraction_than_frame_grid_rejected() -> None:
    """A frame grid denser than the extractor emits is refused instead of fabricated."""
    controls = torch.zeros(1, 386, 3)
    spec = SketchControlSpec(num_frames=4, num_control_tokens=2)

    with pytest.raises(ValueError, match="at least 4 frames"):
        prepare_music_sketch_controls(controls, spec)


def test_music_preparation_returns_float32_independent_storage() -> None:
    """Preparation normalizes dtype without retaining writable source storage."""
    controls = torch.ones(1, 386, 2, dtype=torch.float64)
    spec = SketchControlSpec(num_frames=2, num_control_tokens=2)

    prepared = prepare_music_sketch_controls(controls, spec)
    prepared.zero_()

    assert controls.dtype is torch.float64
    assert torch.count_nonzero(controls) == controls.numel()


def test_music_preparation_output_supports_trainable_projection_backward() -> None:
    """Prepared tensors remain usable by trainable projections under autograd."""
    controls = torch.ones(1, 386, 4, requires_grad=True)
    spec = SketchControlSpec(num_frames=2, num_control_tokens=2)
    projection = torch.nn.Linear(386, 1, bias=False)

    prepared = prepare_music_sketch_controls(controls, spec)
    projection(prepared.transpose(1, 2)).sum().backward()

    assert projection.weight.grad is not None
    assert controls.grad is not None


@pytest.mark.gpu
def test_music_preparation_preserves_cuda_device() -> None:
    """Preparation keeps extracted controls on their source device."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    controls = torch.ones(1, 386, 4, device="cuda")
    spec = SketchControlSpec(num_frames=2, num_control_tokens=2)

    prepared = prepare_music_sketch_controls(controls, spec)

    assert prepared.device.type == "cuda"


def test_tiv_profile_rejected() -> None:
    """Profiles without an offline extractor fail loudly instead of feeding zeros."""
    spec = SketchControlSpec(
        profile="tiv", column="tiv", source="online", sample_rate=_SAMPLE_RATE, num_frames=1
    )
    with pytest.raises(ValueError, match="tiv"):
        extract_profile_controls(_decaying_noise(3), _SAMPLE_RATE, spec)


def test_music_batch_extraction_matches_single_clip_rows_without_autograd() -> None:
    """The shared batch helper is the per-clip path, row for row, and never records a graph."""
    time = torch.arange(_SAMPLE_RATE, dtype=torch.float32) / _SAMPLE_RATE
    audio = torch.stack(
        [torch.sin(2.0 * torch.pi * 440.0 * time), torch.sin(2.0 * torch.pi * 660.0 * time)]
    )[:, None]
    spec = SketchControlSpec(num_frames=32, num_control_tokens=32)

    controls = extract_music_sketch_controls(audio.requires_grad_(True), _SAMPLE_RATE, spec)

    assert controls.shape == (2, 386, 32)
    assert controls.dtype is torch.float32
    assert controls.requires_grad is False
    # PESTO's batched convolutions differ from single-row ones by float noise, so the
    # exact parity check feeds both paths the same one-row batch.
    single = extract_profile_controls(audio[1].detach().numpy(), _SAMPLE_RATE, spec)
    one_row = extract_music_sketch_controls(audio[1:].detach(), _SAMPLE_RATE, spec)
    torch.testing.assert_close(one_row, single, rtol=0, atol=0)


@pytest.fixture
def music_tiv_spec() -> SketchControlSpec:
    """Return the combined contract with a strict pitch threshold.

    :returns: Online 96-frame music and TIV contract with a pitch threshold of one.
    """
    return SketchControlSpec(
        profile="music_tiv",
        source="online",
        sample_rate=_SAMPLE_RATE,
        num_frames=96,
        num_control_tokens=96,
        pitch_zero_threshold=1.0,
    )


@pytest.fixture
def music_tiv_audio() -> torch.Tensor:
    """Return a one-second A4 waveform with nonzero signed TIV coordinates.

    :returns: Float32 mono audio with one batch row at 44.1 kHz.
    """
    time = torch.arange(_SAMPLE_RATE, dtype=torch.float32) / _SAMPLE_RATE
    return torch.sin(2.0 * torch.pi * 440.0 * time)[None, None]


@pytest.mark.parametrize("mode", ["average", "linear"])
@pytest.mark.parametrize("backend", ["torch", "essentia"])
def test_music_tiv_extraction_preserves_music_and_signed_tiv(
    backend: TIVBackend,
    mode: SketchTemporalResampling,
    music_tiv_audio: torch.Tensor,
    music_tiv_spec: SketchControlSpec,
) -> None:
    """The combined profile appends unthresholded TIV on the music frame grid.

    :param backend: TIV chroma frontend.
    :param mode: Temporal reduction shared by the four groups.
    :param music_tiv_audio: Tone with signed TIV coordinates.
    :param music_tiv_spec: Combined extraction contract.
    """
    spec = music_tiv_spec.model_copy(update={"temporal_resampling": mode, "tiv_backend": backend})

    controls = extract_music_sketch_controls(music_tiv_audio, _SAMPLE_RATE, spec)

    music = extract_music_sketch_controls(
        music_tiv_audio, _SAMPLE_RATE, spec.model_copy(update={"profile": "music"})
    )
    centered_audio = (
        torch.nn.functional.pad(music_tiv_audio, (1024, 1536))
        if backend == "torch"
        else music_tiv_audio
    )
    raw_tiv = extract_tiv_batch(centered_audio, _SAMPLE_RATE, backend=backend)[0].numpy()
    source_samples = np.arange(raw_tiv.shape[-1]) * 512
    target_samples = np.arange(0, music_tiv_audio.shape[-1] + 1, mel_hop_length(_SAMPLE_RATE))
    aligned_tiv = np.stack([np.interp(target_samples, source_samples, row) for row in raw_tiv])
    tiv = resample_sketch_controls(torch.from_numpy(aligned_tiv).float()[None], 96, mode)
    torch.testing.assert_close(controls[:, :386], music)
    torch.testing.assert_close(controls[:, 386:], tiv)
    assert (controls[:, 386:] < 0).any()
    assert controls.shape == (1, 398, 96)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_music_tiv_extraction_preserves_audio_device(
    device: str, music_tiv_audio: torch.Tensor, music_tiv_spec: SketchControlSpec
) -> None:
    """All combined controls stay on the waveform device.

    :param device: Waveform and extraction device.
    :param music_tiv_audio: Tone waveform.
    :param music_tiv_spec: Combined extraction contract.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    audio = music_tiv_audio.to(device)

    controls = extract_music_sketch_controls(audio, _SAMPLE_RATE, music_tiv_spec)

    assert controls.device == audio.device


def test_music_tiv_extraction_requires_grad_audio_detaches_controls(
    music_tiv_audio: torch.Tensor, music_tiv_spec: SketchControlSpec
) -> None:
    """Feature extraction never records an autograd graph.

    :param music_tiv_audio: Tone waveform.
    :param music_tiv_spec: Combined extraction contract.
    """
    controls = extract_music_sketch_controls(
        music_tiv_audio.requires_grad_(True), _SAMPLE_RATE, music_tiv_spec
    )

    assert controls.requires_grad is False


def test_music_tiv_extraction_offline_matches_batch(
    music_tiv_audio: torch.Tensor, music_tiv_spec: SketchControlSpec
) -> None:
    """Native inference extracts the same controls as online training.

    :param music_tiv_audio: Tone waveform.
    :param music_tiv_spec: Combined extraction contract.
    """
    offline = extract_profile_controls(music_tiv_audio[0].numpy(), _SAMPLE_RATE, music_tiv_spec)
    batch = extract_music_sketch_controls(music_tiv_audio, _SAMPLE_RATE, music_tiv_spec)

    torch.testing.assert_close(offline, batch)


@pytest.mark.parametrize("backend", ["torch", "essentia"])
def test_music_tiv_final_five_millisecond_tone_retains_boundary_energy(
    backend: TIVBackend, music_tiv_spec: SketchControlSpec
) -> None:
    """TIV must analyze the same trailing audio that reaches the music controls.

    :param backend: TIV chroma frontend.
    :param music_tiv_spec: Combined extraction contract.
    """
    samples = torch.arange(4 * _SAMPLE_RATE, dtype=torch.float32)
    audio = torch.sin(2 * torch.pi * 440 * samples / _SAMPLE_RATE)[None, None]
    audio[..., :-220] = 0
    spec = music_tiv_spec.model_copy(update={"tiv_backend": backend})

    controls = extract_music_sketch_controls(audio, _SAMPLE_RATE, spec)

    assert torch.count_nonzero(controls[:, 386:, -1]) > 0


@pytest.mark.parametrize("backend", ["torch", "essentia"])
@pytest.mark.parametrize("transition_time", [0.1, 3.9])
def test_music_tiv_note_transition_preserves_waveform_timestamp(
    backend: TIVBackend, transition_time: float, music_tiv_spec: SketchControlSpec
) -> None:
    """Changing notes near either boundary retain their physical transition time.

    :param backend: TIV chroma frontend.
    :param transition_time: A4-to-C5 transition in seconds.
    :param music_tiv_spec: Combined extraction contract.
    """
    time = torch.arange(4 * _SAMPLE_RATE, dtype=torch.float32) / _SAMPLE_RATE
    frequency = torch.where(time < transition_time, 440.0, 523.251)
    audio = torch.sin(2 * torch.pi * frequency * time)[None, None]
    spec = music_tiv_spec.model_copy(update={"tiv_backend": backend, "num_frames": 401})

    tiv = extract_music_sketch_controls(audio, _SAMPLE_RATE, spec)[0, 386:]

    before = tiv[:, 2:5].mean(dim=1)
    after = tiv[:, 396:399].mean(dim=1)
    direction = after - before
    progress = ((tiv - before[:, None]) * direction[:, None]).sum(dim=0) / direction.square().sum()
    crossing = int(torch.nonzero(progress >= 0.5)[0, 0])
    fraction = (0.5 - progress[crossing - 1]) / (progress[crossing] - progress[crossing - 1])
    observed_time = (crossing - 1 + float(fraction)) * 0.01
    assert observed_time == pytest.approx(transition_time, abs=0.012)
