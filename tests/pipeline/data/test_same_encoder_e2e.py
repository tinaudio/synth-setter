"""Real-weights e2e tests for :func:`load_same_audio_encoder` (SAME-S).

Unlike the stub-driven registry suite in ``test_add_embeddings.py``, these tests
download the public ``stabilityai/SAME-S`` checkpoint (~433 MB, no credentials)
through ``huggingface_hub``'s default cache and run the real encoder on CPU, so
they carry ``same_e2e`` and run in their own GHA workflow, not the fast lane.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_audio_tools")

from synth_setter.pipeline.data.add_embeddings import (  # noqa: E402
    SAME_EMBEDDING_DIM,
    SAME_SAMPLE_RATE,
    SameEncodeFn,
    load_same_audio_encoder,
    same_num_latent_frames,
)

pytestmark = [pytest.mark.slow, pytest.mark.network, pytest.mark.same_e2e]

_SAME_S_HF_REPO = "stabilityai/SAME-S"
_FIXTURE_SECONDS = 1.0
_FIXTURE_ROWS = 2
_GOLDEN_PATH = Path(__file__).parents[2] / "fixtures" / "same" / "same_s_golden_latents.npz"


def sine_sweep_fixture() -> np.ndarray:
    """Build the deterministic stereo sine-sweep batch both e2e tests encode.

    :returns: ``(2, 2, 44100)`` float32 batch at ``SAME_SAMPLE_RATE``; row 0
        sweeps 110→880 Hz, row 1 sweeps 1760→220 Hz, right channel detuned so
        the stereo field is non-trivial.
    """
    num_samples = int(_FIXTURE_SECONDS * SAME_SAMPLE_RATE)
    time = np.arange(num_samples, dtype=np.float64) / SAME_SAMPLE_RATE
    rows = []
    for start_hz, end_hz in ((110.0, 880.0), (1760.0, 220.0)):
        # Linear chirp: phase integrates the linearly interpolated frequency.
        freq = start_hz + (end_hz - start_hz) * time / _FIXTURE_SECONDS
        phase = 2.0 * np.pi * np.cumsum(freq) / SAME_SAMPLE_RATE
        left = np.sin(phase)
        right = np.sin(phase * 1.01)
        rows.append(np.stack([left, right]))
    return (0.5 * np.stack(rows)).astype(np.float32)


@pytest.fixture(scope="module")
def same_s_encode() -> SameEncodeFn:
    """Load the real SAME-S encoder once per module, pinned to CPU.

    :returns: Encode callable over prepared stereo 44.1 kHz batches.
    """
    return load_same_audio_encoder(_SAME_S_HF_REPO, device="cpu")


def test_same_s_real_weights_encode_matches_contract(same_s_encode: SameEncodeFn) -> None:
    """Real SAME-S latents honor the (rows, 256, T) float32 finite contract.

    :param same_s_encode: Real CPU encoder over the public checkpoint.
    """
    audio = sine_sweep_fixture()
    expected_frames = same_num_latent_frames(audio.shape[2], SAME_SAMPLE_RATE)

    latents = same_s_encode(audio)

    assert latents.shape == (_FIXTURE_ROWS, SAME_EMBEDDING_DIM, expected_frames)
    assert latents.dtype == np.float32
    assert np.isfinite(latents).all()
    # Degenerate (constant/zero) output would satisfy shape+finite alone.
    assert latents.std() > 0.0
    assert not np.allclose(latents[0], latents[1])


def test_same_s_real_weights_encode_matches_golden_latents(
    same_s_encode: SameEncodeFn,
) -> None:
    """Real SAME-S latents match the committed golden within loose tolerances.

    Guards against silent numerical drift of the encode path under future lock
    or upstream bumps. The golden was generated on this repo's lock (torch
    2.11.0+cu128 wheel running on CPU, Linux x86_64) via
    ``tests/pipeline/data/generate_same_golden.py``; using a current-lock
    golden as a proxy for upstream-pinned behavior is licensed by an
    independent measurement that SAME-S latents agree between upstream's
    torch 2.7.1 pin and this lock (max_abs_diff 9.5e-5, mean_abs_diff 8.7e-6
    on latents with std 1.795; allclose at rtol=atol=1e-4). Tolerances are
    calibrated for cross-platform CPU variance (the workflow may run on
    arm64 macOS), not ulp noise: the failure mode guarded is gross drift.

    :param same_s_encode: Real CPU encoder over the public checkpoint.
    """
    golden = np.load(_GOLDEN_PATH)["latents"]

    latents = same_s_encode(sine_sweep_fixture())

    assert latents.shape == golden.shape
    np.testing.assert_allclose(latents, golden, rtol=1e-2, atol=1e-3)
