"""Production-path E2E coverage for the pinned TinyMU MATPAC representation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

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
from synth_setter.pipeline.data.add_embeddings import MIN_ROWS_FOR_INDEX  # noqa: E402
from synth_setter.pipeline.data.lance_shard import (  # noqa: E402
    SHARD_METADATA_SCHEMA_KEY,
    tensor_array,
    write_lance_dataset,
)
from synth_setter.pipeline.data.tinymu import (  # noqa: E402
    TINYMU_CHECKPOINT_REVISION,
    TINYMU_CHECKPOINT_SHA256,
    TINYMU_FRONTEND,
    TINYMU_SOURCE_COMMIT,
    TINYMU_SOURCE_DIR_ENV,
    load_tinymu_audio_encoder,
    tinymu_num_latent_frames,
)
from synth_setter.pipeline.schemas.lance_attempt import LanceDatasetCard  # noqa: E402
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
    assert configured is not None
    return Path(configured)


def _write_real_audio_lance(
    source_dir: Path, destination: Path, rows: int = 2
) -> tuple[int, int]:
    """Persist two distinct real four-second clips as production Lance tensors.

    :param source_dir: Pinned TinyMU checkout containing the example WAV.
    :param destination: Local Lance dataset destination.
    :param rows: Number of alternating real clips to persist.
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
    clips = np.stack([audio[:clip_samples].T, audio[-clip_samples:].T])
    batch = np.ascontiguousarray(clips[np.arange(rows) % len(clips)], dtype=np.float16)
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


def test_tinymu_real_encoder_distinct_audio_is_distinct_and_deterministic() -> None:
    """MATPAC distinguishes two real clips while repeated inference stays bitwise stable."""
    source_dir = _tinymu_source_dir()
    audio, sample_rate = sf.read(
        source_dir / _SOURCE_AUDIO_RELATIVE_PATH,
        always_2d=True,
        dtype="float32",
    )
    clip_samples = sample_rate * _CLIP_SECONDS
    clips = np.ascontiguousarray(
        np.stack([audio[:clip_samples].T, audio[-clip_samples:].T]),
        dtype=np.float32,
    )
    encode = load_tinymu_audio_encoder(source_dir=source_dir, device="cpu")

    first = encode(clips, sample_rate)
    second = encode(clips, sample_rate)

    assert np.isfinite(first).all()
    assert not np.allclose(first[0], first[1])
    np.testing.assert_array_equal(first, second)


def test_add_embeddings_real_tinymu_checkpoint_audio_conditions_generic_encoder(
    tmp_path: Path,
) -> None:
    """Real audio traverses MATPAC, Lance persistence, and generic conditioning.

    :param tmp_path: Isolated cache and Lance location.
    """
    source_dir = _tinymu_source_dir()
    dataset_root = tmp_path / "tinymu-real"
    dataset_root.mkdir()
    split_paths = {
        split: dataset_root / f"{split}.lance" for split in ("train", "val", "test")
    }
    sample_rate, clip_samples = _write_real_audio_lance(source_dir, split_paths["train"])
    for split in ("val", "test"):
        _write_real_audio_lance(source_dir, split_paths[split])
    card = LanceDatasetCard(
        schema_version=1,
        run_id="tinymu-real-e2e",
        finalized_at="2026-07-28T00:00:00+00:00",
        selected_attempts=(),
    )
    (dataset_root / "dataset.json").write_text(card.model_dump_json(indent=2))
    (dataset_root / "dataset.complete").touch()

    command = Path(sys.executable).with_name("synth-setter-add-embeddings")
    environment = os.environ | {
        "PROJECT_ROOT": str(Path(__file__).resolve().parents[3]),
        "XDG_CACHE_HOME": str(tmp_path / "empty-cache"),
    }
    subprocess.run(  # noqa: S603 — installed public CLI with test-owned arguments
        [
            str(command),
            f"dataset_root_uri={dataset_root}",
            "embeddings=[tinymu]",
            f"tinymu_source_dir={source_dir}",
            "device=cpu",
            "batch_size=1",
            "build_index=false",
            f"paths.log_dir={tmp_path / 'logs'}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
        check=True,
        cwd=tmp_path,
        env=environment,
        timeout=1_800,
    )

    for split_path in split_paths.values():
        split_dataset = lance.dataset(split_path)
        assert split_dataset.count_rows() == 2
        assert {TINYMU_FIELD, f"{TINYMU_FIELD}_vec"} <= set(split_dataset.schema.names)

    persisted = LanceDatasetCard.model_validate_json((dataset_root / "dataset.json").read_text())
    assert persisted.schema_version == 2
    provenance = persisted.embeddings[0]
    assert provenance.name == "tinymu"
    assert len(provenance.producer_git_sha) == 40
    assert len(provenance.producer_transform_sha256) == 64
    assert provenance.source_commit == TINYMU_SOURCE_COMMIT
    assert provenance.checkpoint_revision == TINYMU_CHECKPOINT_REVISION
    assert provenance.checkpoint_sha256 == TINYMU_CHECKPOINT_SHA256
    assert tuple(result.split for result in provenance.splits) == ("train", "val", "test")

    dataset = lance.dataset(split_paths["train"])
    values = dataset.to_table(columns=[TINYMU_FIELD, f"{TINYMU_FIELD}_vec"])
    sequence = values.column(TINYMU_FIELD).combine_chunks().to_numpy_ndarray()
    expected_frames = tinymu_num_latent_frames(clip_samples, sample_rate)
    assert sequence.shape == (2, TINYMU_FRONTEND.embedding_dim, expected_frames)
    assert sequence.dtype == np.float32
    assert np.isfinite(sequence).all()
    assert sequence.std() > 0.0

    vectors = np.stack(values.column(f"{TINYMU_FIELD}_vec").to_numpy(zero_copy_only=False))
    np.testing.assert_allclose(vectors, sequence.mean(axis=-1), rtol=1e-5, atol=1e-6)

    encoder = _tinymu_conditioning_encoder()
    with torch.inference_mode():
        conditioned = encoder(torch.from_numpy(sequence))
    assert conditioned.shape == (2, 512)
    assert torch.isfinite(conditioned).all()
    assert not torch.allclose(conditioned[0], conditioned[1])


def test_tinymu_real_embeddings_build_searchable_ivf_pq_index(tmp_path: Path) -> None:
    """Real MATPAC vectors build and serve the declared Lance ANN index.

    :param tmp_path: Isolated dataset and Hydra output root.
    """
    source_dir = _tinymu_source_dir()
    dataset_path = tmp_path / "tinymu-indexed.lance"
    _write_real_audio_lance(source_dir, dataset_path, rows=MIN_ROWS_FOR_INDEX)
    command = Path(sys.executable).with_name("synth-setter-add-embeddings")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subprocess.run(  # noqa: S603 — installed public CLI with test-owned arguments
        [
            str(command),
            f"lance_uri={dataset_path}",
            "embeddings=[tinymu]",
            f"tinymu_source_dir={source_dir}",
            f"device={device}",
            "batch_size=16",
            "build_index=true",
            "num_partitions=4",
            "num_sub_vectors=16",
            f"paths.log_dir={tmp_path / 'logs'}",
            f"hydra.run.dir={tmp_path / 'index-run'}",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[3],
        env=os.environ | {"PROJECT_ROOT": str(Path(__file__).resolve().parents[3])},
        timeout=1_800,
    )

    dataset = lance.dataset(dataset_path)
    vector_column = f"{TINYMU_FIELD}_vec"
    indices = cast("list[dict[str, Any]]", dataset.list_indices())
    assert any(index["fields"] == [vector_column] for index in indices)
    vectors = np.stack(
        dataset.to_table(columns=[vector_column]).column(vector_column).to_numpy(zero_copy_only=False)
    )
    hits = dataset.to_table(
        nearest={"column": vector_column, "q": vectors[0].astype(np.float32), "k": 1}
    )
    assert hits.num_rows == 1
    assert hits.column("_distance")[0].as_py() < 1e-2
