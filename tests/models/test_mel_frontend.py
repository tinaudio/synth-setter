"""The exportable mel front end must reproduce the production librosa features."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.shapes import make_spectrogram
from synth_setter.models.mel_frontend import NormalizedMelFrontend, export_frontend_onnx
from synth_setter.param_spec_name import ParamSpecName

_SAMPLE_RATE = 44_100
_FRAMES = 4 * _SAMPLE_RATE
_SPEC_NAME = ParamSpecName("pyfdn_n8_mono_householder")


@pytest.fixture(scope="module")
def impulse_response() -> np.ndarray:
    """Render one real order-8 Householder pyFDN impulse response.

    :returns: Channel-first float32 audio shaped ``(1, 176400)``.
    """
    spec = resolve_param_spec(_SPEC_NAME)
    renderer = PyFDNRenderer(
        excitation="impulse",
        param_spec_name=_SPEC_NAME,
        plugin_path="pyfdn",
        sample_rate=_SAMPLE_RATE,
        channels=1,
        signal_duration_seconds=4.0,
        plugin_state_path="",
    )
    synth_params, _ = spec.sample(np.random.default_rng(3))
    return renderer.render(synth_params, 60, 100, (0.0, 4.0))


@pytest.fixture(scope="module")
def stats() -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic non-trivial mel statistics on the production grid.

    :returns: Mean and positive standard deviation shaped ``(1, 128, 401)``.
    """
    rng = np.random.default_rng(11)
    mean = rng.normal(-40.0, 10.0, size=(1, 128, 401)).astype(np.float32)
    std = rng.uniform(2.0, 12.0, size=(1, 128, 401)).astype(np.float32)
    return mean, std


def _expected(audio: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Return the sketch CLI's normalized mel for one waveform.

    :param audio: Channel-first mono waveform.
    :param stats: Mean and standard deviation on the mel grid.
    :returns: Normalized mel shaped ``(1, 128, 401)``.
    """
    mean, std = stats
    return np.asarray((make_spectrogram(audio, _SAMPLE_RATE) - mean) / std, dtype=np.float32)


def test_frontend_forward_matches_librosa_pipeline_on_real_impulse_response(
    impulse_response: np.ndarray, stats: tuple[np.ndarray, np.ndarray]
) -> None:
    """The torch front end reproduces librosa mel, dB scaling, and statistics normalization.

    :param impulse_response: Real pyFDN render.
    :param stats: Normalization statistics.
    """
    frontend = NormalizedMelFrontend(sample_rate=_SAMPLE_RATE, mean=stats[0], std=stats[1]).eval()
    with torch.no_grad():
        actual = frontend(torch.from_numpy(impulse_response)).numpy()
    assert actual.shape == (1, 1, 128, 401)
    np.testing.assert_allclose(actual[0], _expected(impulse_response, stats), atol=1e-3)


def test_frontend_forward_silence_matches_librosa_floor(
    stats: tuple[np.ndarray, np.ndarray],
) -> None:
    """Digital silence hits the amin floor identically instead of producing NaN or -inf.

    :param stats: Normalization statistics.
    """
    silence = np.zeros((1, _FRAMES), dtype=np.float32)
    frontend = NormalizedMelFrontend(sample_rate=_SAMPLE_RATE, mean=stats[0], std=stats[1]).eval()
    with torch.no_grad():
        actual = frontend(torch.from_numpy(silence)).numpy()
    np.testing.assert_allclose(actual[0], _expected(silence, stats), atol=1e-3)


def test_exported_frontend_onnx_matches_librosa_pipeline(
    tmp_path: Path, impulse_response: np.ndarray, stats: tuple[np.ndarray, np.ndarray]
) -> None:
    """The ONNX graph runs on the CPU runtime with the same result as the torch front end.

    :param tmp_path: Export destination.
    :param impulse_response: Real pyFDN render.
    :param stats: Normalization statistics.
    """
    frontend = NormalizedMelFrontend(sample_rate=_SAMPLE_RATE, mean=stats[0], std=stats[1]).eval()
    graph = tmp_path / "frontend.onnx"
    export_frontend_onnx(frontend, graph)
    session = ort.InferenceSession(str(graph))
    assert [item.name for item in session.get_inputs()] == ["waveform"]
    assert session.get_inputs()[0].shape == [1, _FRAMES]
    (actual,) = session.run(None, {"waveform": impulse_response})
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual[0], _expected(impulse_response, stats), atol=1e-3)


@pytest.mark.parametrize("shape", [(128, 401), (1, 128, 400), (2, 128, 401)])
def test_frontend_stats_off_grid_rejected(shape: tuple[int, ...]) -> None:
    """Statistics must match the fixed ``(1, 128, 401)`` mel grid exactly.

    :param shape: Statistics shape that does not match the grid.
    """
    with pytest.raises(ValueError, match="statistics"):
        NormalizedMelFrontend(
            sample_rate=_SAMPLE_RATE,
            mean=np.zeros(shape, np.float32),
            std=np.ones(shape, np.float32),
        )


def test_frontend_nonpositive_std_rejected() -> None:
    """A zero standard deviation would divide by zero in the browser."""
    std = np.ones((1, 128, 401), np.float32)
    std[0, 5, 7] = 0.0
    with pytest.raises(ValueError, match="positive"):
        NormalizedMelFrontend(
            sample_rate=_SAMPLE_RATE, mean=np.zeros((1, 128, 401), np.float32), std=std
        )
