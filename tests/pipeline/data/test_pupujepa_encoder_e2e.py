"""Real pinned-weight parity coverage for PupuJEPA Tiny consumers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import hydra
import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    PUPUJEPA_LARGE_FIELD,
    PUPUJEPA_TINY_FIELD,
)
from synth_setter.pipeline.data.lance_shard import (
    SHARD_METADATA_SCHEMA_KEY,
    tensor_array,
    write_lance_dataset,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pupujepa import (
    PUPUJEPA_LARGE_EMBEDDING_DIM,
    PUPUJEPA_TINY_EMBEDDING_DIM,
    PupuJepaVariant,
    resolve_pupujepa_checkpoint,
)

pytestmark = [pytest.mark.slow, pytest.mark.network]


class _PupuJepaE2ECase(NamedTuple):
    """Describe one released teacher's production-path contract.

    .. attribute :: variant

        Released teacher size.

    .. attribute :: profile

        Registry and conditioning profile name.

    .. attribute :: field

        Lance sequence field.

    .. attribute :: embedding_dim

        Frequency-concatenated teacher width.

    .. attribute :: device

        Inference device sized for the teacher.
    """

    variant: PupuJepaVariant
    profile: str
    field: str
    embedding_dim: int
    device: str


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _PupuJepaE2ECase(
                "tiny",
                "pupujepa_tiny",
                PUPUJEPA_TINY_FIELD,
                PUPUJEPA_TINY_EMBEDDING_DIM,
                "cpu",
            ),
            id="tiny",
        ),
        pytest.param(
            _PupuJepaE2ECase(
                "large",
                "pupujepa_large",
                PUPUJEPA_LARGE_FIELD,
                PUPUJEPA_LARGE_EMBEDDING_DIM,
                "cpu",
            ),
            id="large",
        ),
    ],
)
def test_real_pupujepa_weights_add_embeddings_and_online_consumers_match(
    tmp_path: Path,
    case: _PupuJepaE2ECase,
) -> None:
    """The public CLI artifact matches online inference and conditions the flow.

    :param tmp_path: Isolated Lance dataset and Hydra output root.
    :param case: Released teacher's profile, field, shape, and device contract.
    """
    variant, profile, field, embedding_dim, device = case
    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            online_config = compose(
                config_name="train.yaml",
                overrides=[
                    "experiment=torchsynth/flow",
                    f"conditioning={profile}_online",
                    "trainer=cpu",
                ],
            )
            cached_config = compose(
                config_name="train.yaml",
                overrides=[
                    "experiment=torchsynth/flow",
                    f"conditioning={profile}",
                    "trainer=cpu",
                ],
            )
        cached = hydra.utils.instantiate(cached_config.model.encoder).eval().to(device)
    finally:
        GlobalHydra.instance().clear()

    sample_rate = int(online_config.datamodule.sample_rate)
    time = torch.arange(4 * sample_rate) / sample_rate
    mono = torch.stack(
        [
            0.25 * torch.sin(2 * torch.pi * 440 * time),
            0.25 * torch.sin(2 * torch.pi * 880 * time),
        ]
    )
    stereo = np.repeat(mono.numpy()[:, None, :], 2, axis=1).astype(np.float16)
    mono = torch.from_numpy(stereo.mean(axis=1, dtype=np.float32))
    tensor = tensor_array(stereo, np.dtype("float16"), stereo.shape[1:])
    metadata = ShardMetadata(
        velocity=100,
        signal_duration_seconds=4.0,
        sample_rate=sample_rate,
        channels=2,
        min_loudness=-60.0,
    )
    schema = pa.schema(
        [pa.field(AUDIO_FIELD, tensor.type, nullable=False)],
        metadata={SHARD_METADATA_SCHEMA_KEY: metadata.model_dump_json().encode()},
    )
    dataset_path = tmp_path / f"{profile}-real.lance"
    write_lance_dataset(dataset_path, schema, [pa.record_batch([tensor], schema=schema)])

    command = Path(sys.executable).with_name("synth-setter-add-embeddings")
    checkpoint = resolve_pupujepa_checkpoint(variant=variant)
    subprocess.run(  # noqa: S603 — installed public CLI with test-owned arguments
        [
            str(command),
            f"lance_uri={dataset_path}",
            f"embeddings=[{profile}]",
            f"checkpoints.{profile}={checkpoint}",
            f"device={device}",
            "batch_size=1",
            "build_index=false",
            f"paths.log_dir={tmp_path / 'logs'}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
        check=True,
        cwd=tmp_path,
        env=os.environ | {"PROJECT_ROOT": str(Path(__file__).resolve().parents[3])},
        timeout=1_800,
    )

    table = lance.dataset(dataset_path).to_table(columns=[field, f"{field}_vec"])
    offline_sequence = table.column(field).combine_chunks().to_numpy_ndarray()
    offline_vector = np.stack(table.column(f"{field}_vec").to_numpy(zero_copy_only=False))
    online = hydra.utils.instantiate(online_config.model.encoder).eval().to(device)
    shared_head_state = {
        key: value
        for key, value in online.head.state_dict().items()
        if key != "positional_encoding.pe"
    }
    cached.load_state_dict(shared_head_state, strict=False)
    mono = mono.to(device)
    with torch.inference_mode():
        online_sequence = online.embed(mono).cpu()
        online_conditioning = online(mono).cpu()
        cached_conditioning = cached(torch.from_numpy(offline_sequence).to(device)).cpu()

    assert offline_sequence.shape == (2, embedding_dim, 100)
    assert np.isfinite(offline_sequence).all()
    assert not np.array_equal(offline_sequence[0], offline_sequence[1])
    np.testing.assert_allclose(offline_sequence, online_sequence.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(offline_vector, offline_sequence.mean(axis=-1), rtol=1e-5, atol=1e-6)
    assert online_conditioning.shape == cached_conditioning.shape == (2, 8, 512)
    assert torch.isfinite(online_conditioning).all()
    assert torch.isfinite(cached_conditioning).all()
    assert not torch.equal(online_conditioning[0], online_conditioning[1])
    torch.testing.assert_close(online_conditioning, cached_conditioning, rtol=1e-5, atol=1e-5)
