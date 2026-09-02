"""Real-weight production path for the MeanAudio-S-Full comparison candidate."""

from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.dev.run_clap_comparison import (
    DEFAULT_MEANAUDIO_INVERSE_CHECKPOINT,
    render_meanaudio_candidate,
)
from synth_setter.pipeline.data.meanaudio_generation import (
    MEANAUDIO_DURATION_SECONDS,
    MEANAUDIO_STEPS,
    load_meanaudio_s_full_generator,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.network,
    pytest.mark.meanaudio_e2e,
    pytest.mark.requires_surgepy,
    pytest.mark.integration_r2,
    pytest.mark.r2,
]


def test_meanaudio_s_full_prompt_inverse_surge_clap_production_path(tmp_path: Path) -> None:
    """A real S-Full latent drives the immutable inverse checkpoint through CLAP scoring.

    :param tmp_path: Isolated candidate WAV and metric CSV destination.
    """
    if not torch.cuda.is_available():
        pytest.skip("MeanAudio-S-Full E2E needs CUDA for the official real-weight generation path")

    prompt = "A single frog croaking beside a quiet pond"
    generator = load_meanaudio_s_full_generator(
        steps=MEANAUDIO_STEPS,
        duration_seconds=MEANAUDIO_DURATION_SECONDS,
        device="cuda",
    )
    latent = generator(prompt, 0)
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    output = tmp_path / "candidate.wav"
    row = render_meanaudio_candidate(
        prompt,
        latent,
        checkpoint=DEFAULT_MEANAUDIO_INVERSE_CHECKPOINT,
        output=output,
        wav_r2_uri="r2://production-e2e-not-uploaded/candidate.wav",
        device="cuda",
        seed=0,
    )

    audio, sample_rate = sf.read(output, dtype="float32", always_2d=True)
    assert sample_rate == 44_100
    assert audio.shape == (176_400, 2)
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 1e-4
    assert output.with_suffix(".csv").is_file()
    assert math.isfinite(float(row["cosine_similarity"]))
    assert math.isfinite(float(row["cosine_distance"]))
    assert float(row["cosine_distance"]) == pytest.approx(1.0 - float(row["cosine_similarity"]))
