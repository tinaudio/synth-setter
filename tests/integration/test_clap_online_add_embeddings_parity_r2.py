"""Parity test for the pinned online and add-embeddings CLAP consumers."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest
import torch

from synth_setter.clap import DEFAULT_CLAP_TRAINING_CHECKPOINT, resolve_clap_checkpoint
from synth_setter.models.components.pretrained_encoder import ClapAudioEncoder
from synth_setter.pipeline.data.add_embeddings import load_clap_audio_encoder

pytestmark = [pytest.mark.slow, pytest.mark.integration_r2, pytest.mark.r2]

_SAMPLE_RATE = 44_100


def test_pinned_clap_online_matches_add_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both real consumers produce the same embedding for one waveform.

    The temporary XDG root removes the downloaded checkpoint after the test.

    :param monkeypatch: Fixture routing the shared checkpoint cache under ``tmp_path``.
    :param tmp_path: Self-cleaning checkpoint-cache root.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    samples = np.arange(_SAMPLE_RATE, dtype=np.float32)
    waveform = np.sin(samples * 0.01, dtype=np.float32)[None, :]

    checkpoint_dir = resolve_clap_checkpoint(DEFAULT_CLAP_TRAINING_CHECKPOINT)
    add_embeddings_encoder = load_clap_audio_encoder(checkpoint_dir, "cpu")
    expected = add_embeddings_encoder(waveform, _SAMPLE_RATE)
    del add_embeddings_encoder
    gc.collect()

    online_encoder = ClapAudioEncoder.from_pretrained(
        sample_rate=_SAMPLE_RATE,
        checkpoint=checkpoint_dir,
    )
    with torch.no_grad():
        actual = online_encoder(torch.from_numpy(waveform)).numpy()

    assert expected.shape == actual.shape == (1, 512)
    assert np.isfinite(expected).all()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
