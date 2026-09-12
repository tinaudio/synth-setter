"""Contract tests for official MeanAudio-S-Full latent generation."""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.pipeline.data.meanaudio import (
    MEANAUDIO_CHECKPOINT_REVISION,
    MEANAUDIO_PACKAGE_COMMIT,
)
from synth_setter.pipeline.data.meanaudio_generation import (
    MEANAUDIO_DURATION_SECONDS,
    MEANAUDIO_LATENT_SHAPE,
    MEANAUDIO_S_FULL_CHECKPOINT_NAME,
    MEANAUDIO_S_FULL_CHECKPOINT_SHA256,
    MEANAUDIO_STEPS,
    meanaudio_s_full_provenance,
    validate_meanaudio_s_full_latent,
)


def test_validate_meanaudio_s_full_latent_channel_major_returns_float32() -> None:
    """Official unnormalized channel-major latents retain the inverse input shape."""
    latent = np.full((1, 20, 125), 0.25, dtype=np.float16)

    validated = validate_meanaudio_s_full_latent(latent)

    assert validated.shape == (1, *MEANAUDIO_LATENT_SHAPE)
    assert validated.dtype == np.float32
    assert validated.flags.c_contiguous
    np.testing.assert_array_equal(validated, np.full((1, 20, 125), 0.25, dtype=np.float32))


def test_validate_meanaudio_s_full_latent_frame_major_rejects_decode_layout() -> None:
    """Frame-major upstream state cannot silently enter the channel-major inverse model."""
    with pytest.raises(ValueError, match=r"\(1, 20, 125\)"):
        validate_meanaudio_s_full_latent(np.zeros((1, 125, 20), dtype=np.float32))


def test_validate_meanaudio_s_full_latent_nonfinite_rejects_conditioning() -> None:
    """Non-finite generated state fails before inverse inference."""
    latent = np.zeros((1, 20, 125), dtype=np.float32)
    latent[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        validate_meanaudio_s_full_latent(latent)


def test_meanaudio_s_full_provenance_pins_official_generation_assets() -> None:
    """Generation provenance names immutable source, model, and text assets."""
    provenance = meanaudio_s_full_provenance()

    assert provenance["meanaudio_upstream_revision"] == MEANAUDIO_PACKAGE_COMMIT
    assert provenance["meanaudio_checkpoint_revision"] == MEANAUDIO_CHECKPOINT_REVISION
    assert provenance["meanaudio_model_checkpoint_name"] == MEANAUDIO_S_FULL_CHECKPOINT_NAME
    assert provenance["meanaudio_model_checkpoint_sha256"] == MEANAUDIO_S_FULL_CHECKPOINT_SHA256
    assert len(provenance["meanaudio_clap_package_revision"]) == 40
    assert len(provenance["meanaudio_clap_checkpoint_sha256"]) == 64
    assert len(provenance["meanaudio_t5_revision"]) == 40
    assert MEANAUDIO_STEPS == 25
    assert MEANAUDIO_DURATION_SECONDS == 4.0
