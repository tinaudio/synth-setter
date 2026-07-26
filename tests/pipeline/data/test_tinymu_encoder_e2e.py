"""Production-path E2E coverage for the pinned TinyMU MATPAC representation."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import lance
import numpy as np
import pyarrow as pa
import pytest
import soundfile as sf
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

pytest.importorskip("timm")

from synth_setter.data.vst.shapes import AUDIO_FIELD, TINYMU_FIELD  # noqa: E402
from synth_setter.pipeline.data.add_embeddings import add_embeddings  # noqa: E402
from synth_setter.pipeline.data.lance_shard import (  # noqa: E402
    SHARD_METADATA_SCHEMA_KEY,
    tensor_array,
    write_lance_dataset,
)
from synth_setter.pipeline.data.tinymu import (  # noqa: E402
    TINYMU_EMBEDDING_DIM,
    TINYMU_SOURCE_DIR_ENV,
    load_tinymu_audio_encoder,
    tinymu_num_latent_frames,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig  # noqa: E402
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.integration_r2, pytest.mark.r2]

_SOURCE_AUDIO_RELATIVE_PATH = Path("resource/example1.wav")
_CLIP_SECONDS = 4


def _tinymu_source_dir() -> Path:
    """Resolve the explicitly supplied unlicensed TinyMU checkout for this E2E test.

    :returns: Pinned local TinyMU checkout.
    """
    configured = os.environ.get(TINYMU_SOURCE_DIR_ENV)
    if configured is None:
        pytest.skip(f"set {TINYMU_SOURCE_DIR_ENV} to the pinned TinyMU checkout")
    return Path(configured)


def _write_real_audio_lance(source_dir: Path, destination: Path) -> tuple[int, int]:
    """Persist four seconds of upstream's real example audio as a production Lance tensor.

    :param source_dir: Pinned TinyMU checkout containing the example WAV.
    :param destination: Local Lance dataset destination.
    :returns: Source sample rate and sample count.
    :raises ValueError: The real audio is shorter than four seconds.
    """
    audio, sample_rate = sf.read(
        source_dir / _SOURCE_AUDIO_RELATIVE_PATH,
        always_2d=True,
        dtype="float32",
    )
    clip_samples = sample_rate * _CLIP_SECONDS
    if len(audio) < clip_samples:
        raise ValueError(f"real TinyMU audio has only {len(audio)} samples, need {clip_samples}")
    batch = np.ascontiguousarray(audio[:clip_samples].T[None], dtype=np.float16)
    tensor = tensor_array(batch, np.dtype("float16"), batch.shape[1:])
    metadata = ShardMetadata(
        velocity=100,
        signal_duration_seconds=float(_CLIP_SECONDS),
        sample_rate=sample_rate,
        channels=batch.shape[1],
        min_loudness=-60.0,
    )
    schema = pa.schema(
        [pa.field(AUDIO_FIELD, tensor.type, nullable=False)],
        metadata={SHARD_METADATA_SCHEMA_KEY: metadata.model_dump_json().encode()},
    )
    write_lance_dataset(destination, schema, [pa.record_batch([tensor], schema=schema)])
    return sample_rate, clip_samples


def _tinymu_conditioning_encoder() -> torch.nn.Module:
    """Instantiate the generic conditioning encoder through the shipped Hydra profile.

    :returns: Generic sequence-conditioning encoder configured for TinyMU.
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
                    "conditioning=tinymu",
                    "trainer=cpu",
                    "paths.output_dir=/tmp/synth-setter-tinymu-e2e",
                ],
            )
        return hydra.utils.instantiate(config.model.encoder).eval()
    finally:
        GlobalHydra.instance().clear()


def test_tinymu_real_encoder_same_audio_is_bitwise_deterministic() -> None:
    """Frozen eval-mode MATPAC returns identical sequences for identical real audio."""
    source_dir = _tinymu_source_dir()
    audio, sample_rate = sf.read(
        source_dir / _SOURCE_AUDIO_RELATIVE_PATH,
        always_2d=True,
        dtype="float32",
    )
    clip = np.ascontiguousarray(audio[: sample_rate * _CLIP_SECONDS].T[None])
    encode = load_tinymu_audio_encoder(source_dir=source_dir, device="cpu")

    first = encode(clip, sample_rate)
    second = encode(clip, sample_rate)

    np.testing.assert_array_equal(first, second)


def test_add_embeddings_real_tinymu_checkpoint_audio_conditions_generic_encoder(
    tmp_path: Path,
) -> None:
    """Real audio traverses MATPAC, Lance persistence, and generic conditioning.

    :param tmp_path: Isolated cache and Lance location.
    """
    source_dir = _tinymu_source_dir()
    dataset_path = tmp_path / "tinymu-real.lance"
    sample_rate, clip_samples = _write_real_audio_lance(source_dir, dataset_path)

    previous_cache_home = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(tmp_path / "empty-cache")
    try:
        add_embeddings(
            AddEmbeddingsConfig(
                lance_uri=str(dataset_path),
                embeddings=("tinymu",),
                tinymu_source_dir=source_dir,
                device="cpu",
                batch_size=1,
                build_index=False,
            )
        )
    finally:
        if previous_cache_home is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = previous_cache_home

    dataset = lance.dataset(dataset_path)
    assert dataset.count_rows() == 1
    assert {TINYMU_FIELD, f"{TINYMU_FIELD}_vec"} <= set(dataset.schema.names)
    values = dataset.to_table(columns=[TINYMU_FIELD, f"{TINYMU_FIELD}_vec"])
    sequence = values.column(TINYMU_FIELD).combine_chunks().to_numpy_ndarray()
    expected_frames = tinymu_num_latent_frames(clip_samples, sample_rate)
    assert sequence.shape == (1, TINYMU_EMBEDDING_DIM, expected_frames)
    assert sequence.dtype == np.float32
    assert np.isfinite(sequence).all()
    assert sequence.std() > 0.0

    vectors = np.stack(values.column(f"{TINYMU_FIELD}_vec").to_numpy(zero_copy_only=False))
    np.testing.assert_allclose(vectors, sequence.mean(axis=-1), rtol=1e-5, atol=1e-6)

    encoder = _tinymu_conditioning_encoder()
    with torch.inference_mode():
        conditioned = encoder(torch.from_numpy(sequence))
    assert conditioned.shape == (1, 512)
    assert torch.isfinite(conditioned).all()
