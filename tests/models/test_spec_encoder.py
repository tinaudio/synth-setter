"""Behavioral tests for the log-mel front end and the frontend/backbone composition."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

import hydra
import jaxtyping
import librosa
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from synth_setter.models.components.cnn import MelCNN
from synth_setter.models.components.spec_encoder import LogMelFrontend, SpecEncoder
from synth_setter.models.components.transformer import AudioSpectrogramTransformer
from synth_setter.resources import configs_dir

_frontend = partial(LogMelFrontend, in_dim=4_410, sample_rate=44_100)


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Keep model initialization and synthetic waveforms deterministic."""
    torch.manual_seed(0)


def test_log_mel_frontend_returns_single_channel_spectrogram_grid() -> None:
    """The front end emits the channel axis spectrogram backbones consume."""
    features = _frontend()(torch.zeros(2, 4_410))

    assert features.shape == (2, 1, 128, 11)


def test_log_mel_frontend_matches_dataset_frontend() -> None:
    """All frames preserve the stored-mel frontend's numeric contract."""
    audio = torch.randn(1, 4_410)
    expected = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio[0].numpy(),
            sr=44_100,
            n_fft=1_102,
            hop_length=441,
            n_mels=128,
            window="hamming",
        ),
        ref=np.max,
    )

    actual = _frontend()(audio)[0, 0].detach().numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_frontend_power_one_matches_amplitude_decibels() -> None:
    """Magnitude spectrograms use amplitude rather than power decibel scaling."""
    audio = torch.randn(1, 4_410)
    expected = librosa.amplitude_to_db(
        librosa.feature.melspectrogram(
            y=audio[0].numpy(),
            sr=44_100,
            n_fft=1_102,
            hop_length=441,
            n_mels=128,
            window="hamming",
            power=1.0,
        ),
        ref=np.max,
    )

    actual = _frontend(power=1.0)(audio)[0, 0].detach().numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_frontend_htk_scale_matches_dataset_frontend() -> None:
    """The HTK mel scale matches the stored-feature reference."""
    audio = torch.randn(1, 4_410)
    expected = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio[0].numpy(),
            sr=44_100,
            n_fft=1_102,
            hop_length=441,
            n_mels=128,
            window="hamming",
            htk=True,
        ),
        ref=np.max,
    )

    actual = _frontend(mel_scale="htk")(audio)[0, 0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_frontend_hann_window_matches_dataset_frontend() -> None:
    """The Hann window matches the stored-feature reference."""
    audio = torch.randn(1, 4_410)
    expected = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio[0].numpy(),
            sr=44_100,
            n_fft=1_102,
            hop_length=441,
            n_mels=128,
            window="hann",
        ),
        ref=np.max,
    )

    actual = _frontend(window="hann")(audio)[0, 0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_frontend_top_db_clips_relative_dynamic_range() -> None:
    """The dynamic-range option clips values relative to each waveform peak."""
    audio = torch.randn(1, 4_410)
    unclipped = _frontend(top_db=None)(audio)
    clipped = _frontend(top_db=10.0)(audio)

    torch.testing.assert_close(clipped, torch.clamp(unclipped, min=-10.0))


def test_log_mel_frontend_sign_inversion_returns_same_spectrogram() -> None:
    """A pi phase shift leaves the magnitude-based features unchanged."""
    frontend = _frontend()
    audio = torch.randn(2, 4_410)

    torch.testing.assert_close(frontend(-audio), frontend(audio))


@pytest.mark.parametrize("audio", [torch.zeros(2, 1, 4_410), torch.zeros(2, 4_409)])
def test_log_mel_frontend_invalid_waveform_shape_raises(audio: torch.Tensor) -> None:
    """Malformed waveform batches fail at the front-end boundary.

    :param audio: Wrong-rank or wrong-length waveform batch.
    """
    with pytest.raises(ValueError, match="Expected waveform shape"):
        _frontend()(audio)


@pytest.mark.parametrize("amin", [0.0, -1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_amin_raises(amin: float) -> None:
    """A non-positive or non-finite logarithm floor is rejected.

    :param amin: Invalid power floor.
    """
    with pytest.raises(ValueError, match="amin"):
        _frontend(amin=amin)


@pytest.mark.parametrize("power", [0.0, -1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_power_raises(power: float) -> None:
    """A non-positive or non-finite magnitude exponent is rejected.

    :param power: Invalid spectrogram exponent.
    """
    with pytest.raises(ValueError, match="power"):
        _frontend(power=power)


@pytest.mark.parametrize("top_db", [-1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_top_db_raises(top_db: float) -> None:
    """A negative or non-finite dynamic range is rejected.

    :param top_db: Invalid dynamic range.
    """
    with pytest.raises(ValueError, match="top_db"):
        _frontend(top_db=top_db)


@pytest.mark.parametrize(
    ("factory", "frequency_name"),
    [
        pytest.param(partial(_frontend, f_min=-1.0), "f_min", id="negative-f-min"),
        pytest.param(partial(_frontend, f_min=float("inf")), "f_min", id="infinite-f-min"),
        pytest.param(partial(_frontend, f_min=float("nan")), "f_min", id="nan-f-min"),
        pytest.param(partial(_frontend, f_min=22_050.0), "f_min", id="f-min-at-nyquist"),
        pytest.param(partial(_frontend, f_max=0.0), "f_max", id="f-max-not-greater-than-f-min"),
        pytest.param(partial(_frontend, f_max=float("inf")), "f_max", id="infinite-f-max"),
        pytest.param(partial(_frontend, f_max=float("nan")), "f_max", id="nan-f-max"),
        pytest.param(partial(_frontend, f_max=22_051.0), "f_max", id="f-max-above-nyquist"),
    ],
)
def test_log_mel_frontend_invalid_frequency_bound_raises(
    factory: Callable[[], LogMelFrontend], frequency_name: str
) -> None:
    """Invalid mel-frequency bounds fail before producing non-finite features.

    :param factory: Front-end factory containing the invalid bound.
    :param frequency_name: Constructor argument receiving the invalid bound.
    """
    with pytest.raises(ValueError, match=frequency_name):
        factory()


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        pytest.param(partial(_frontend, hop_length=0), "hop_length", id="zero-hop-length"),
        pytest.param(partial(_frontend, n_fft=0), "n_fft", id="zero-n-fft"),
        pytest.param(partial(_frontend, n_mels=0), "n_mels", id="zero-n-mels"),
    ],
)
def test_log_mel_frontend_non_positive_geometry_raises(
    factory: Callable[[], LogMelFrontend], field: str
) -> None:
    """Non-positive frontend geometry fails at the configuration boundary.

    :param factory: Front-end factory containing the zero size.
    :param field: Constructor argument receiving the zero size.
    """
    with pytest.raises(ValueError, match=field):
        factory()


def test_log_mel_frontend_unknown_window_raises() -> None:
    """An unsupported Fourier window fails before transform construction."""
    with pytest.raises(jaxtyping.TypeCheckError):
        _frontend(window="blackman")  # type: ignore[arg-type]


def test_log_mel_frontend_normalization_standardizes_raw_features() -> None:
    """Configured statistics are applied once after decibel conversion."""
    frontend = _frontend(normalization_mean=[-10.0], normalization_std=[2.0])
    audio = torch.randn(1, 4_410)

    raw = frontend.forward_raw(audio)

    assert frontend.normalization_enabled
    torch.testing.assert_close(frontend(audio), (raw + 10.0) / 2.0)
    assert not frontend.normalization_mean.requires_grad
    assert not frontend.normalization_std.requires_grad


def test_log_mel_frontend_without_normalization_preserves_raw_output() -> None:
    """The default front end retains the existing unnormalized contract."""
    frontend = _frontend()
    audio = torch.randn(1, 4_410)

    assert not frontend.normalization_enabled
    torch.testing.assert_close(frontend(audio), frontend.forward_raw(audio))


def test_log_mel_frontend_hydra_inline_nested_statistics_normalize_output() -> None:
    """The shipped Hydra config converts nested inline statistics to plain lists."""
    with (configs_dir() / "model/frontend/log_mel.yaml").open() as config_file:
        frontend_config = OmegaConf.load(config_file)
    config = OmegaConf.create(
        {
            "datamodule": {"signal_length": 4_410, "sample_rate": 44_100},
            "frontend": frontend_config,
        }
    )
    config.frontend.normalization_mean = [[-10.0]]
    config.frontend.normalization_std = [[2.0]]
    frontend = hydra.utils.instantiate(config.frontend)
    audio = torch.randn(1, 4_410)

    torch.testing.assert_close(frontend(audio), (frontend.forward_raw(audio) + 10.0) / 2.0)


def test_log_mel_frontend_statistics_file_normalizes_real_features(tmp_path: Path) -> None:
    """A local dataset statistics artifact configures feature normalization.

    :param tmp_path: Isolated directory for the statistics artifact.
    """
    stats_path = tmp_path / "stats.npz"
    np.savez(
        stats_path,
        mean=np.full((1, 128, 11), -20.0, dtype=np.float32),
        std=np.full((1, 128, 11), 4.0, dtype=np.float32),
    )
    frontend = _frontend(normalization_stats_path=str(stats_path))
    audio = torch.randn(1, 4_410)

    torch.testing.assert_close(frontend(audio), (frontend.forward_raw(audio) + 20.0) / 4.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"normalization_mean": 0.0}, id="mean-without-std"),
        pytest.param({"normalization_std": 1.0}, id="std-without-mean"),
        pytest.param(
            {
                "normalization_mean": 0.0,
                "normalization_std": 1.0,
                "normalization_stats_path": "stats.npz",
            },
            id="inline-and-file",
        ),
    ],
)
def test_log_mel_frontend_incomplete_or_ambiguous_normalization_raises(
    kwargs: dict[str, object],
) -> None:
    """Normalization requires exactly one complete statistics source.

    :param kwargs: Invalid normalization constructor arguments.
    """
    with pytest.raises(ValueError, match="normalization"):
        _frontend(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mean", "std", "message"),
    [
        pytest.param(float("nan"), 1.0, "mean", id="nonfinite-mean"),
        pytest.param(0.0, float("inf"), "std", id="nonfinite-std"),
        pytest.param(0.0, 0.0, "positive", id="zero-std"),
        pytest.param(1e100, 1.0, "mean", id="mean-overflows-frontend-dtype"),
        pytest.param(0.0, 1e-100, "positive", id="std-underflows-frontend-dtype"),
        pytest.param([0.0, 1.0], 1.0, "broadcast", id="wrong-mean-geometry"),
        pytest.param(0.0, [1.0, 2.0], "broadcast", id="wrong-std-geometry"),
    ],
)
def test_log_mel_frontend_invalid_normalization_values_raise(
    mean: float | list[float], std: float | list[float], message: str
) -> None:
    """Invalid normalization values fail at the frontend boundary.

    :param mean: Candidate normalization mean.
    :param std: Candidate normalization standard deviation.
    :param message: Expected validation error fragment.
    """
    with pytest.raises(ValueError, match=message):
        _frontend(normalization_mean=mean, normalization_std=std)


def test_log_mel_frontend_missing_statistics_file_raises(tmp_path: Path) -> None:
    """An explicitly configured missing statistics artifact is not ignored.

    :param tmp_path: Isolated directory that does not contain the requested artifact.
    """
    with pytest.raises(FileNotFoundError):
        _frontend(normalization_stats_path=str(tmp_path / "missing.npz"))


def test_log_mel_frontend_odd_fft_statistics_match_actual_frame_geometry() -> None:
    """Normalization geometry follows centered STFT padding for odd FFT sizes."""
    frontend = LogMelFrontend(
        in_dim=22_000,
        sample_rate=22_050,
        n_fft=551,
        hop_length=220,
        n_mels=8,
        normalization_mean=np.zeros((1, 8, 100)).tolist(),
        normalization_std=np.ones((1, 8, 100)).tolist(),
    )

    assert frontend(torch.zeros(1, 22_000)).shape == (1, 1, 8, 100)


def test_log_mel_frontend_setter_updates_normalization() -> None:
    """Calibration can enable normalization after frontend construction."""
    frontend = _frontend()
    audio = torch.randn(1, 4_410)
    raw = frontend.forward_raw(audio)

    frontend.set_normalization_statistics(torch.tensor(-5.0), torch.tensor(2.0))

    torch.testing.assert_close(frontend(audio), (raw + 5.0) / 2.0)


@pytest.mark.parametrize(
    ("mean", "std", "message"),
    [
        pytest.param(torch.tensor(1e100, dtype=torch.float64), torch.tensor(1.0), "mean"),
        pytest.param(torch.tensor(0.0), torch.tensor(1e-100, dtype=torch.float64), "positive"),
    ],
)
def test_log_mel_frontend_setter_rejects_values_unsafe_after_dtype_cast(
    mean: torch.Tensor, std: torch.Tensor, message: str
) -> None:
    """Calibration values must remain valid in the frontend buffer dtype.

    :param mean: Candidate normalization mean.
    :param std: Candidate normalization standard deviation.
    :param message: Expected validation error fragment.
    """
    with pytest.raises(ValueError, match=message):
        _frontend().set_normalization_statistics(mean, std)


def test_log_mel_frontend_checkpoint_restores_statistics_without_source(tmp_path: Path) -> None:
    """Nested checkpoint buffers restore without reopening their source artifact.

    :param tmp_path: Isolated directory for the removable source artifact.
    """
    stats_path = tmp_path / "stats.npz"
    np.savez(
        stats_path,
        mean=np.full((1, 128, 11), -5.0, dtype=np.float32),
        std=np.full((1, 128, 11), 2.0, dtype=np.float32),
    )
    configured = SpecEncoder(
        frontend=_frontend(normalization_stats_path=str(stats_path)),
        backbone=torch.nn.Identity(),
    )
    state = configured.state_dict()
    stats_path.unlink()
    restored_frontend = _frontend()
    restored = SpecEncoder(frontend=restored_frontend, backbone=torch.nn.Identity())

    restored.load_state_dict(state)

    assert restored_frontend.normalization_enabled
    audio = torch.randn(1, 4_410)
    torch.testing.assert_close(restored(audio), configured(audio))


@pytest.mark.parametrize("buffer_name", ["normalization_mean", "normalization_std"])
def test_log_mel_frontend_empty_checkpoint_preserves_destination_dtype(buffer_name: str) -> None:
    """Loading disabled statistics preserves the destination module's precision.

    :param buffer_name: Normalization buffer whose destination precision must survive.
    """
    frontend = _frontend().double()

    frontend.load_state_dict(_frontend().state_dict())

    assert getattr(frontend, buffer_name).dtype == torch.float64


def test_log_mel_frontend_legacy_checkpoint_loads_without_normalization_keys() -> None:
    """Strict legacy loading disables constructor-provided normalization."""
    frontend = _frontend(normalization_mean=-5.0, normalization_std=2.0)
    legacy_state = {
        name: value
        for name, value in _frontend().state_dict().items()
        if name not in {"normalization_mean", "normalization_std"}
    }

    frontend.load_state_dict(legacy_state)

    assert not frontend.normalization_enabled


def test_log_mel_frontend_normalization_preserves_waveform_gradients() -> None:
    """Standardization remains differentiable with respect to input waveforms."""
    frontend = _frontend(normalization_mean=-20.0, normalization_std=4.0)
    audio = torch.randn(2, 4_410, requires_grad=True)

    frontend(audio).square().mean().backward()

    assert audio.grad is not None
    assert torch.count_nonzero(audio.grad)


def test_normalized_online_and_stored_mel_paths_match_ast_outputs() -> None:
    """Online normalization reproduces the stored-mel path through shared AST weights."""
    time = torch.arange(4_410) / 44_100
    audio = torch.sin(2 * torch.pi * 440 * time).unsqueeze(0)
    stored_raw = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio[0].numpy(),
            sr=44_100,
            n_fft=1_102,
            hop_length=441,
            n_mels=128,
            window="hamming",
        ),
        ref=np.max,
    )
    mean = np.linspace(-30.0, -10.0, stored_raw.size, dtype=np.float32).reshape(1, 128, 11)
    std = np.linspace(2.0, 4.0, stored_raw.size, dtype=np.float32).reshape(1, 128, 11)
    stored_features = torch.from_numpy((stored_raw[None] - mean) / std).unsqueeze(0)
    ast = AudioSpectrogramTransformer(
        d_model=8,
        n_heads=2,
        n_layers=1,
        n_conditioning_outputs=2,
        patch_size=4,
        patch_stride=2,
        input_channels=1,
        spec_shape=cast(tuple[int], (128, 11)),
    ).eval()
    online_encoder = SpecEncoder(
        frontend=_frontend(normalization_mean=mean.tolist(), normalization_std=std.tolist()),
        backbone=ast,
    ).eval()

    with torch.no_grad():
        stored_output = ast(stored_features)
        online_output = online_encoder(audio)

    torch.testing.assert_close(online_output, stored_output, atol=1e-3, rtol=1e-3)


def test_spec_encoder_with_cnn_backbone_returns_pooled_embedding() -> None:
    """The CNN backbone reduces the front end's grid to one vector per row."""
    encoder = SpecEncoder(
        frontend=_frontend(),
        backbone=MelCNN(hidden_dim=4, out_dim=5, num_blocks=1, kernel_size=3),
    )

    assert encoder(torch.zeros(2, 4_410)).shape == (2, 5)


def test_spec_encoder_distinct_spectra_return_distinct_embeddings() -> None:
    """The composed encoder responds to spectral content instead of returning a constant."""
    time = torch.arange(4_410) / 44_100
    audio = torch.stack(
        [torch.sin(2 * torch.pi * 220 * time), torch.sin(2 * torch.pi * 1_760 * time)]
    )
    encoder = SpecEncoder(
        frontend=_frontend(),
        backbone=MelCNN(hidden_dim=4, out_dim=5, num_blocks=1, kernel_size=3),
    )
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(audio)

    assert torch.isfinite(embeddings).all()
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_spec_encoder_distinct_envelopes_return_distinct_embeddings() -> None:
    """The composed encoder preserves temporal-envelope information for one carrier."""
    time = torch.arange(4_410) / 44_100
    carrier = torch.sin(2 * torch.pi * 440 * time)
    audio = torch.stack(
        [carrier * torch.linspace(0, 1, 4_410), carrier * torch.linspace(1, 0, 4_410)]
    )
    encoder = SpecEncoder(
        frontend=_frontend(),
        backbone=MelCNN(hidden_dim=4, out_dim=5, num_blocks=1, kernel_size=3),
    )
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(audio)

    assert torch.isfinite(embeddings).all()
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_spec_encoder_backward_reaches_the_backbone_and_the_waveform() -> None:
    """Gradients survive the front end, so waveform-side terms stay trainable."""
    encoder = SpecEncoder(
        frontend=_frontend(),
        backbone=MelCNN(hidden_dim=4, out_dim=5, num_blocks=1, kernel_size=3),
    )
    audio = torch.randn(2, 4_410, requires_grad=True)

    encoder(audio).square().mean().backward()

    assert audio.grad is not None
    assert torch.count_nonzero(audio.grad)
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad), name
