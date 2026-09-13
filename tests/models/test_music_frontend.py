"""Browser music front ends reproduce the native feature pipeline."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from synth_setter.cli.clap_render import resolve_inverse_checkpoint
from synth_setter.cli.sketch_render import (
    _prepare_inputs,
    _resolve_stats,
    load_render_config,
)
from synth_setter.conditioning import resolve_sketch_controls
from synth_setter.data.vst.shapes import make_spectrogram
from synth_setter.features.profile_controls import extract_profile_controls
from synth_setter.models.music_frontend import (
    MusicSketchFrontend,
    StereoMelFrontend,
    export_music_sketch_onnx,
    export_stereo_mel_onnx,
)
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.tools.export_browser_surge_bundle import _load_settings

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration_r2,
    pytest.mark.r2,
    pytest.mark.requires_surgepy,
]
_BROWSER_ONNX_OPS = {
    "Abs",
    "Add",
    "Clip",
    "Concat",
    "Conv",
    "Div",
    "Equal",
    "Gather",
    "GreaterOrEqual",
    "IsNaN",
    "LayerNormalization",
    "LeakyRelu",
    "Log",
    "MatMul",
    "Max",
    "Mul",
    "Pad",
    "Pow",
    "ReduceMax",
    "ReduceMean",
    "ReduceSum",
    "Reshape",
    "Slice",
    "Softmax",
    "Split",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
    "Where",
}


def _real_audio(midi_note: int = 72) -> np.ndarray:
    """Render an isolated native note through the configured production synth.

    :param midi_note: Pitch distinguishing the input fixtures.
    :returns: Stereo waveform on the checkpoint grid.
    """
    return make_audio_renderer(load_render_config()).render(
        {}, midi_note=midi_note, velocity=100, note_start_and_end=(0.5, 2.0)
    )


@pytest.fixture(scope="module")
def stats() -> tuple[np.ndarray, np.ndarray]:
    """Load the digest-verified statistics selected by sketch rendering.

    :returns: Per-channel training means and standard deviations.
    """
    settings = _load_settings()
    with np.load(_resolve_stats(settings.stats, settings.stats_sha256)) as archive:
        return archive["mean"], archive["std"]


@pytest.fixture(scope="module")
def real_model() -> VSTFlowMatchingModule:
    """Load the digest-pinned production checkpoint on CPU.

    :returns: Real evaluation model selected by sketch rendering.
    """
    settings = _load_settings()
    checkpoint = resolve_inverse_checkpoint(settings.checkpoint, settings.checkpoint_sha256)
    return VSTFlowMatchingModule.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    ).eval()


@pytest.fixture(scope="module")
def sketch_frontend() -> MusicSketchFrontend:
    """Load the production PESTO checkpoint once for this module.

    :returns: Evaluation front end on the browser audio grid.
    """
    return MusicSketchFrontend(
        sample_rate=44_100, output_frames=32, pitch_zero_threshold=0.1
    ).eval()


def test_stereo_mel_real_audio_matches_native_spectrogram(
    stats: tuple[np.ndarray, np.ndarray],
) -> None:
    """Channel-specific normalization matches a real native render.

    :param stats: Production mel statistics.
    """
    audio = _real_audio()
    expected = ((make_spectrogram(audio, 44_100) - stats[0]) / stats[1])[None]
    frontend = StereoMelFrontend(sample_rate=44_100, mean=stats[0], std=stats[1]).eval()
    with torch.no_grad():
        actual = frontend(torch.from_numpy(audio[None])).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.slow
def test_frontends_real_audio_match_native_prepare_inputs(
    stats: tuple[np.ndarray, np.ndarray],
    sketch_frontend: MusicSketchFrontend,
    real_model: VSTFlowMatchingModule,
) -> None:
    """Both browser outputs match the production preparation path on real audio.

    :param stats: Production mel statistics.
    :param sketch_frontend: Production PESTO front end.
    :param real_model: Digest-pinned production checkpoint.
    """
    audio = _real_audio()
    render = load_render_config()
    settings = _load_settings()
    native = _prepare_inputs(
        sketch_audio=audio,
        content_audio=audio,
        stats_path=_resolve_stats(settings.stats, settings.stats_sha256),
        model=real_model,
        render=render,
        device=torch.device("cpu"),
    )
    mel_frontend = StereoMelFrontend(sample_rate=44_100, mean=stats[0], std=stats[1]).eval()
    with torch.no_grad():
        actual_mel = mel_frontend(torch.from_numpy(audio[None]))
        actual_sketch = sketch_frontend(torch.from_numpy(audio[None]))
    torch.testing.assert_close(actual_mel, native["mel"], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_sketch, native["sketch_ctrl"], rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("midi_note", [60, 72])
def test_music_sketch_real_audio_matches_native_profile_extractor(
    sketch_frontend: MusicSketchFrontend, midi_note: int
) -> None:
    """Activation-only PESTO and scalar tracks match the native extractor.

    :param sketch_frontend: Production checkpoint front end.
    :param midi_note: Pitch of the native input fixture.
    """
    audio = _real_audio(midi_note)
    spec = resolve_sketch_controls({"profile": "music", "num_frames": 32})
    assert spec is not None
    expected = extract_profile_controls(audio, 44_100, spec)
    with torch.no_grad():
        actual = sketch_frontend(torch.from_numpy(audio[None]))
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_music_sketch_silence_matches_native_profile_extractor(
    sketch_frontend: MusicSketchFrontend,
) -> None:
    """Silence preserves native floors and pitch thresholding.

    :param sketch_frontend: Production checkpoint front end.
    """
    audio = np.zeros((2, 176_400), dtype=np.float32)
    spec = resolve_sketch_controls({"profile": "music", "num_frames": 32})
    assert spec is not None
    expected = extract_profile_controls(audio, 44_100, spec)
    with torch.no_grad():
        actual = sketch_frontend(torch.from_numpy(audio[None]))
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_music_sketch_distinct_native_notes_change_pitch(
    sketch_frontend: MusicSketchFrontend,
) -> None:
    """The two native Surge notes retain different activation profiles.

    :param sketch_frontend: Production checkpoint front end.
    """
    baseline = torch.from_numpy(_real_audio(60)[None])
    content = torch.from_numpy(_real_audio(72)[None])
    with torch.no_grad():
        baseline_pitch = sketch_frontend(baseline)[:, 2:]
        content_pitch = sketch_frontend(content)[:, 2:]
    assert not torch.equal(baseline_pitch, content_pitch)


@pytest.mark.slow
def test_music_frontends_onnx_match_torch(
    tmp_path: Path,
    stats: tuple[np.ndarray, np.ndarray],
    sketch_frontend: MusicSketchFrontend,
) -> None:
    """ONNX Runtime executes both browser graphs with native-equivalent outputs.

    :param tmp_path: Graph destination.
    :param stats: Production mel statistics.
    :param sketch_frontend: Production checkpoint front end.
    """
    mel_frontend = StereoMelFrontend(sample_rate=44_100, mean=stats[0], std=stats[1]).eval()
    mel_path, sketch_path = tmp_path / "frontend.onnx", tmp_path / "sketch.onnx"
    export_stereo_mel_onnx(mel_frontend, mel_path)
    export_music_sketch_onnx(sketch_frontend, sketch_path)
    waveform = _real_audio()[None]
    with torch.no_grad():
        expected_mel = mel_frontend(torch.from_numpy(waveform)).numpy()
        expected_sketch = sketch_frontend(torch.from_numpy(waveform)).numpy()
    (actual_mel,) = ort.InferenceSession(str(mel_path)).run(None, {"waveform": waveform})
    (actual_sketch,) = ort.InferenceSession(str(sketch_path)).run(None, {"waveform": waveform})
    assert isinstance(actual_mel, np.ndarray)
    assert isinstance(actual_sketch, np.ndarray)
    np.testing.assert_allclose(actual_mel, expected_mel, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(actual_sketch, expected_sketch, rtol=3e-4, atol=3e-4)
    graph_ops = {
        node.op_type for path in (mel_path, sketch_path) for node in onnx.load(path).graph.node
    }
    assert graph_ops <= _BROWSER_ONNX_OPS
