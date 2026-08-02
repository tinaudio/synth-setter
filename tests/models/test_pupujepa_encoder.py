"""Behavior tests for the shared PupuJEPA Tiny waveform encoder."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml
from safetensors.torch import save_file

from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.models.components.pretrained_encoder import PretrainedConditioningEncoder
from synth_setter.models.components.pupujepa_encoder import (
    PupuJepaAudioEncoder,
    PupuJepaMelFrontend,
)
from synth_setter.pupujepa import PUPUJEPA_TINY_CONFIG, PupuJepaConfig


def _tiny_config() -> PupuJepaConfig:
    """Return an inexpensive explicit PupuJEPA geometry.

    :returns: Small architecture retaining PupuJEPA's patch and RoPE contracts.
    """
    return PupuJepaConfig(
        sample_rate=8_000,
        n_fft=64,
        win_length=64,
        hop_length=16,
        n_mels=32,
        fmin=0.0,
        fmax=4_000.0,
        mel_mean=-4.089994845986366,
        mel_std=2.0242277159094813,
        patch_time=4,
        patch_frequency=8,
        embed_dim=12,
        depth=1,
        num_heads=3,
        mlp_ratio=4.0,
        use_swiglu=True,
        qk_norm=True,
    )


def test_frontend_four_second_audio_produces_400_frames() -> None:
    """The pinned reflection padding retains the upstream 25 Hz frame grid."""
    frontend = PupuJepaMelFrontend(PUPUJEPA_TINY_CONFIG)

    features = frontend(torch.zeros(1, 4 * PUPUJEPA_TINY_CONFIG.sample_rate))

    assert features.shape == (1, 1, 400, 128)


def test_frontend_silence_uses_upstream_log_mel_floor() -> None:
    """Zero spectra clamp after mel projection without an artificial FFT floor."""
    config = PUPUJEPA_TINY_CONFIG

    features = PupuJepaMelFrontend(config)(torch.zeros(1, 4 * config.sample_rate))

    expected = (math.log(1e-5) - config.mel_mean) / config.mel_std
    torch.testing.assert_close(features, torch.full_like(features, expected))


def test_frontend_sine_matches_upstream_known_values() -> None:
    """Magnitude mel scaling, natural log, and normalization match upstream."""
    time = torch.arange(4 * PUPUJEPA_TINY_CONFIG.sample_rate) / PUPUJEPA_TINY_CONFIG.sample_rate
    audio = (0.25 * torch.sin(2 * torch.pi * 440 * time)).unsqueeze(0)
    bins = torch.tensor([0, 1, 4, 5, 6, 7, 10, 20, 30, 40, 60, 100])

    first_frame = PupuJepaMelFrontend(PUPUJEPA_TINY_CONFIG)(audio)[0, 0, 0, bins]

    torch.testing.assert_close(
        first_frame,
        torch.tensor(
            [
                1.11500394,
                1.12037933,
                1.16001856,
                1.18105197,
                1.21695352,
                1.28553438,
                1.40300822,
                1.39413226,
                0.70392591,
                0.29106188,
                -0.28137225,
                -1.30110860,
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_encoder_variable_length_audio_returns_frequency_concatenated_sequence() -> None:
    """Complete time patches remain variable while frequency patches form the width."""
    torch.manual_seed(0)
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)

    embeddings = encoder(torch.randn(2, 256).clamp(-1.0, 1.0))

    assert embeddings.shape == (2, 48, 4)
    assert torch.isfinite(embeddings).all()
    assert not torch.equal(embeddings[0], embeddings[1])


def test_encoder_half_rate_input_preserves_duration_geometry() -> None:
    """Resampling maps equal-duration inputs onto the same patch grid."""
    torch.manual_seed(0)
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)
    time = torch.arange(256) / config.sample_rate
    audio = torch.sin(2 * torch.pi * 220 * time).unsqueeze(0)

    native_embeddings = encoder(audio)
    resampled_embeddings = encoder(audio[:, ::2].contiguous(), sample_rate=config.sample_rate // 2)

    assert resampled_embeddings.shape == native_embeddings.shape
    assert torch.isfinite(resampled_embeddings).all()


def test_encoder_stereo_input_matches_channel_mean() -> None:
    """Online Lance waveforms use the same downmix as cached embeddings."""
    torch.manual_seed(0)
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)
    stereo = torch.randn(2, 2, 256).clamp(-1.0, 1.0)

    actual = encoder(stereo)
    expected = encoder(stereo.mean(dim=1))

    torch.testing.assert_close(actual, expected)


def test_encoder_autocast_preserves_float32_embedding_space() -> None:
    """Mixed-precision training cannot move the frozen teacher representation."""
    torch.manual_seed(0)
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)
    audio = torch.randn(2, 256).clamp(-1.0, 1.0)
    expected = encoder(audio)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = encoder(audio)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_encoder_is_frozen_and_stays_in_eval_mode() -> None:
    """Training a conditioning head cannot move or stochasticize the teacher."""
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)

    encoder.train(True)

    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_encoder_waveform_gradient_is_finite_and_nonzero() -> None:
    """Frozen teacher parameters do not sever the online waveform gradient."""
    torch.manual_seed(0)
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)
    audio = torch.randn(2, 256).clamp(-1.0, 1.0).requires_grad_()

    encoder(audio).square().mean().backward()

    assert audio.grad is not None
    assert torch.isfinite(audio.grad).all()
    assert torch.count_nonzero(audio.grad).item() > 0


def test_encoder_batch_rows_have_isolated_waveform_gradients() -> None:
    """One row's teacher sequence is independent of every other batch row."""
    torch.manual_seed(0)
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)
    audio = torch.randn(2, 256).clamp(-1.0, 1.0).requires_grad_()

    (gradient,) = torch.autograd.grad(encoder(audio)[0].square().sum(), audio)

    assert torch.count_nonzero(gradient[0]).item() > 0
    assert torch.equal(gradient[1], torch.zeros_like(gradient[1]))


def _checkpoint_args() -> dict[str, object]:
    """Return the complete pinned checkpoint configuration.

    :returns: Shape-defining PupuJEPA Tiny args mapping.
    """
    config = PUPUJEPA_TINY_CONFIG
    return {
        "preprocess": {
            "sample_rate": config.sample_rate,
            "cut_mel_frame": 1_024,
            "hop_size": config.hop_length,
            "n_fft": config.n_fft,
            "win_size": config.win_length,
            "fmin": config.fmin,
            "fmax": config.fmax,
            "n_mels": config.n_mels,
            "normalize": True,
            "flip_ft": True,
        },
        "model": {
            "image_size": [1_024, config.n_mels],
            "patch_size": [config.patch_time, config.patch_frequency],
            "embed_dim": config.embed_dim,
            "depth": config.depth,
            "num_heads": config.num_heads,
            "mlp_ratio": config.mlp_ratio,
            "drop_path_rate": 0.0,
            "drop_path_uniform": False,
            "use_swiglu": config.use_swiglu,
            "layer_scale_init_value": None,
            "qk_norm": config.qk_norm,
            "norm_layer": "layer",
            "frequency_first": False,
            "in_chans": 1,
        },
    }


def test_from_pretrained_incomplete_teacher_state_raises(tmp_path: Path) -> None:
    """A partial teacher checkpoint cannot silently leave random parameters loaded.

    :param tmp_path: Temporary local checkpoint directory.
    """
    config = PUPUJEPA_TINY_CONFIG
    (tmp_path / "args.json").write_text(yaml.safe_dump(_checkpoint_args()))
    save_file(
        {"teacher.norm.weight": torch.ones(config.embed_dim)}, tmp_path / "model.safetensors"
    )

    with pytest.raises(RuntimeError, match="strict PupuJEPA teacher state"):
        PupuJepaAudioEncoder.from_pretrained(
            sample_rate=config.sample_rate,
            checkpoint=str(tmp_path),
        )


def test_from_pretrained_disabled_normalization_raises(tmp_path: Path) -> None:
    """A checkpoint requesting another frontend cannot reuse the pinned weights.

    :param tmp_path: Temporary local checkpoint directory.
    """
    args = _checkpoint_args()
    preprocess = cast("dict[str, object]", args["preprocess"])
    preprocess["normalize"] = False
    (tmp_path / "args.json").write_text(yaml.safe_dump(args))
    save_file({"teacher.norm.weight": torch.ones(192)}, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="frontend contract"):
        PupuJepaAudioEncoder.from_pretrained(sample_rate=24_000, checkpoint=str(tmp_path))


def test_conditioning_training_step_updates_pool_not_teacher() -> None:
    """A real optimization step trains the pool while leaving the teacher frozen."""
    torch.manual_seed(0)
    config = _tiny_config()
    backbone = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)
    head = EmbeddingPool(
        embed_dim=config.output_dim,
        d_model=8,
        num_heads=1,
        max_seq_len=4,
    )
    encoder = PretrainedConditioningEncoder(backbone=backbone, head=head, out_dim=8)
    original_query = head.query.detach().clone()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)

    optimizer.zero_grad()
    encoder(torch.randn(2, 256).clamp(-1.0, 1.0)).square().mean().backward()
    optimizer.step()

    assert not torch.equal(head.query, original_query)
    assert all(parameter.grad is None for parameter in backbone.parameters())


@pytest.mark.slow
def test_pupujepa_online_conditioning_overfits_fixed_batch() -> None:
    """The trainable pool learns a fixed mapping from frozen teacher states."""
    torch.manual_seed(0)
    config = _tiny_config()
    encoder = PretrainedConditioningEncoder(
        backbone=PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config),
        head=EmbeddingPool(
            embed_dim=config.output_dim,
            d_model=8,
            num_heads=1,
            max_seq_len=4,
        ),
        out_dim=8,
    )
    predictor = torch.nn.Linear(8, 2)
    audio = torch.randn(2, 256).clamp(-1.0, 1.0)
    with torch.no_grad():
        embeddings = encoder.embed(audio)
    targets = torch.tensor(((-1.0, 1.0), (1.0, -1.0)))
    optimizer = torch.optim.Adam((*encoder.head.parameters(), *predictor.parameters()), lr=3e-3)

    initial_loss = torch.nn.functional.mse_loss(predictor(encoder.project(embeddings)), targets)
    loss = initial_loss
    for _ in range(1_000):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(predictor(encoder.project(embeddings)), targets)
        loss.backward()
        optimizer.step()

    assert loss.item() < initial_loss.item() / 100
    assert loss.item() < 0.01


def test_encoder_empty_waveform_raises_value_error() -> None:
    """Empty online waveforms retain the public validation error contract."""
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)

    with pytest.raises(ValueError, match="positive num_samples"):
        encoder(torch.empty(1, 0))


def test_encoder_empty_channel_axis_raises_value_error() -> None:
    """A channel-first batch must contain one or two channels."""
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)

    with pytest.raises(ValueError, match="1 or 2 channels"):
        encoder(torch.empty(1, 0, 256))


def test_encoder_opposed_out_of_range_stereo_raises_value_error() -> None:
    """Online bounds reject malformed channels even when their downmix is in range."""
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)
    opposed = torch.stack([torch.full((256,), 1.1), torch.full((256,), -1.1)])[None, ...]

    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        encoder(opposed)


def test_encoder_out_of_range_waveform_raises() -> None:
    """Online waveforms outside the normalized audio contract fail early."""
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        encoder(torch.full((1, 256), 1.01))


def test_encoder_too_short_for_one_time_patch_raises() -> None:
    """Audio without four complete mel frames fails before the patch convolution."""
    config = _tiny_config()
    encoder = PupuJepaAudioEncoder(sample_rate=config.sample_rate, config=config)

    with pytest.raises(ValueError, match="one complete time patch"):
        encoder(torch.zeros(1, 63))
