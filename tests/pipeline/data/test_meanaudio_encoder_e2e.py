"""Real upstream parity and production-path coverage for MeanAudio embeddings."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

import hydra
import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from synth_setter.data.vst.shapes import AUDIO_FIELD, MEANAUDIO_16K_FIELD
from synth_setter.pipeline.data.lance_shard import (
    SHARD_METADATA_SCHEMA_KEY,
    tensor_array,
    write_lance_dataset,
)
from synth_setter.pipeline.data.meanaudio import (
    MEANAUDIO_EMBEDDING_DIM,
    resolve_meanaudio_checkpoint,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.subprocess_stream import check_call_streamed

pytestmark = [pytest.mark.slow, pytest.mark.network, pytest.mark.meanaudio_e2e]

_SAMPLE_RATE = 44_100
_CLIP_SECONDS = 4
_CLIP_SAMPLES = _SAMPLE_RATE * _CLIP_SECONDS
_PARITY_ATOL = 1e-6
_PARITY_RTOL = 1e-6


def _deterministic_audio() -> np.ndarray:
    """Return two normalized four-second clips with distinct spectra.

    :returns: Contiguous float32 ``(2, 1, 176400)`` audio.
    """
    time = np.arange(_CLIP_SAMPLES, dtype=np.float32) / _SAMPLE_RATE
    clips = np.stack(
        [
            0.4 * np.sin(2 * np.pi * 220 * time),
            0.4 * np.sin(2 * np.pi * 880 * time),
        ]
    )
    return np.ascontiguousarray(clips[:, None, :], dtype=np.float32)


def _write_audio_lance(destination: Path) -> None:
    """Persist deterministic clips as production-schema Lance audio tensors.

    :param destination: Local Lance dataset destination.
    """
    audio = _deterministic_audio().astype(np.float16)
    tensor = tensor_array(audio, np.dtype("float16"), audio.shape[1:])
    metadata = ShardMetadata(
        velocity=100,
        signal_duration_seconds=float(_CLIP_SECONDS),
        sample_rate=_SAMPLE_RATE,
        channels=1,
        min_loudness=-60.0,
    )
    schema = pa.schema(
        [pa.field(AUDIO_FIELD, tensor.type, nullable=False)],
        metadata={SHARD_METADATA_SCHEMA_KEY: metadata.model_dump_json().encode()},
    )
    write_lance_dataset(destination, schema, [pa.record_batch([tensor], schema=schema)])


def _conditioning_encoder() -> torch.nn.Module:
    """Compose the shipped MeanAudio profile and instantiate its real EmbeddingPool.

    :returns: Evaluation-mode generic sequence-conditioning encoder.
    """
    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            config = compose(
                config_name="train.yaml",
                overrides=[
                    "datamodule=surge_lance",
                    "datamodule.param_spec_name=surge_xt",
                    "model=vst_flow",
                    "conditioning=meanaudio_16k",
                    "trainer=cpu",
                    "paths.output_dir=/tmp/synth-setter-meanaudio-e2e",
                ],
            )
        return cast("torch.nn.Module", hydra.utils.instantiate(config.model.encoder)).eval()
    finally:
        GlobalHydra.instance().clear()


def test_meanaudio_public_adapter_matches_direct_upstream_posterior_mean(
    tmp_path: Path,
) -> None:
    """The public adapter equals direct upstream mel, VAE, and posterior-mean inference.

    Separate processes release the large VAE state between paths without changing either
    implementation under comparison.

    :param tmp_path: Isolated audio and output-array location.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = resolve_meanaudio_checkpoint()
    audio_path = tmp_path / "audio.npy"
    direct_path = tmp_path / "direct.npy"
    adapter_path = tmp_path / "adapter.npy"
    np.save(audio_path, _deterministic_audio()[:1], allow_pickle=False)
    direct_program = """
import sys

import numpy as np
import torch
import torchaudio.functional as audio_fn
from meanaudio.ext.autoencoder.vae import get_my_vae
from meanaudio.ext.mel_converter import get_mel_converter

audio = np.load(sys.argv[1], allow_pickle=False)
checkpoint, output, device = sys.argv[2:]
mono = audio_fn.resample(torch.from_numpy(audio[:, 0]), 44_100, 16_000)
mel_converter = get_mel_converter("16k").to(device).eval()
with torch.device("meta"):
    vae = get_my_vae("16k")
state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
vae.load_state_dict(state, strict=True, assign=True)
del vae.decoder
del state
vae.remove_weight_norm()
vae = vae.to(device).eval().requires_grad_(False)
with torch.inference_mode():
    latents = vae.encode(mel_converter(mono.to(device))).mode().float().cpu().numpy()
np.save(output, latents, allow_pickle=False)
"""
    adapter_program = """
import sys

import numpy as np
import torch
import torchaudio.functional as audio_fn

from synth_setter.pipeline.data.meanaudio import load_meanaudio_audio_encoder

audio = np.load(sys.argv[1], allow_pickle=False)
checkpoint, output, device = sys.argv[2:]
encode = load_meanaudio_audio_encoder(checkpoint, device=device)
first = encode(audio, 44_100)
second = encode(audio, 44_100)
np.testing.assert_array_equal(second, first)
native_mono = audio_fn.resample(torch.from_numpy(audio[:, 0]), 44_100, 16_000).numpy()[:, None]
native_stereo = np.repeat(native_mono, 2, axis=1)
np.testing.assert_allclose(encode(native_mono, 16_000), first, rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(encode(native_stereo, 16_000), first, rtol=1e-6, atol=1e-6)
np.save(output, first, allow_pickle=False)
"""
    for program, output in ((direct_program, direct_path), (adapter_program, adapter_path)):
        check_call_streamed(
            [sys.executable, "-c", program, str(audio_path), str(checkpoint), str(output), device],
            timeout=1_800,
        )

    direct = np.load(direct_path, allow_pickle=False)
    actual = np.load(adapter_path, allow_pickle=False)
    assert actual.shape == direct.shape == (1, MEANAUDIO_EMBEDDING_DIM, 125)
    # Both paths execute the same pinned float32 kernels; tolerance admits only kernel-order jitter.
    np.testing.assert_allclose(actual, direct, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)


def test_add_embeddings_real_meanaudio_lance_conditions_embedding_pool(
    tmp_path: Path,
) -> None:
    """The public CLI persists real latents consumed by the shipped EmbeddingPool.

    :param tmp_path: Isolated Lance and Hydra output location.
    """
    dataset_path = tmp_path / "meanaudio-real.lance"
    _write_audio_lance(dataset_path)
    command = Path(sys.executable).with_name("synth-setter-add-embeddings")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_call_streamed(
        [
            str(command),
            "logger=[]",
            f"lance_uri={dataset_path}",
            "embeddings=[meanaudio_16k]",
            f"device={device}",
            "batch_size=1",
            "build_index=false",
            f"paths.log_dir={tmp_path / 'logs'}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
        env=os.environ | {"PROJECT_ROOT": str(Path(__file__).resolve().parents[3])},
        timeout=1_800,
    )

    dataset = lance.dataset(dataset_path)
    vector_column = f"{MEANAUDIO_16K_FIELD}_vec"
    assert dataset.count_rows() == 2
    assert {MEANAUDIO_16K_FIELD, vector_column} <= set(dataset.schema.names)
    table = dataset.to_table(columns=[MEANAUDIO_16K_FIELD, vector_column])
    sequence = table.column(MEANAUDIO_16K_FIELD).combine_chunks().to_numpy_ndarray()
    vectors = np.stack(table.column(vector_column).to_numpy(zero_copy_only=False))

    assert sequence.shape == (2, MEANAUDIO_EMBEDDING_DIM, 125)
    assert sequence.dtype == np.float32
    assert np.isfinite(sequence).all()
    assert sequence.std() > 0.0
    assert not np.array_equal(sequence[0], sequence[1])
    np.testing.assert_allclose(vectors, sequence.mean(axis=-1), rtol=1e-5, atol=1e-6)

    encoder = _conditioning_encoder()
    with torch.inference_mode():
        conditioned = encoder(torch.from_numpy(sequence))
    assert conditioned.shape == (2, 8, 512)
    assert torch.isfinite(conditioned).all()
    assert conditioned.std() > 0.0
    assert not torch.allclose(conditioned[0], conditioned[1])
