"""Integration tests for SLAP retrieval export."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Literal, cast
from uuid import UUID, uuid4

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from synth_setter.data.vst_datamodule import prepare_batch
from synth_setter.models.components.slap import BYOLLoss, SiameseArm
from synth_setter.models.slap_module import SLAPModule
from synth_setter.pipeline.data import export_slap as export_slap_module
from synth_setter.pipeline.data.export_slap import (
    SLAP_EXPORT_METADATA_KEY,
    SLAP_SOURCE_POINTER_KEY,
    _ensure_row_uuids,
    _existing_completed,
    _ExportMetadata,
    _normalize,
    _validate_projection_pair,
    export_slap,
)
from synth_setter.pipeline.data.lance_shard import SHARD_METADATA_SCHEMA_KEY
from synth_setter.pipeline.schemas.export_slap_config import ExportSlapConfig
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata


def _arm(input_dim: int, output_dim: int = 2) -> SiameseArm:
    return SiameseArm(
        encoder=nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(input_dim, 4)),
        projector=nn.Linear(4, output_dim),
        transform=nn.Linear(output_dim, output_dim),
        normalize_projections=True,
    )


def _model(
    output_dim: int = 2, *, audio_input_key: Literal["audio", "mel"] = "audio"
) -> SLAPModule:
    return SLAPModule(
        audio_encoder=_arm(128 if audio_input_key == "mel" else 5, output_dim),
        text_encoder=_arm(2, output_dim),
        loss_fn=BYOLLoss(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        audio_input_key=audio_input_key,
    )


def _model_config(
    output_dim: int = 2, *, audio_input_key: Literal["audio", "mel"] = "audio"
) -> dict[str, object]:
    def arm(input_dim: int) -> dict[str, object]:
        return {
            "_target_": "synth_setter.models.components.slap.SiameseArm",
            "encoder": {
                "_target_": "torch.nn.Sequential",
                "_args_": [
                    {"_target_": "torch.nn.Flatten", "start_dim": 1},
                    {"_target_": "torch.nn.Linear", "in_features": input_dim, "out_features": 4},
                ],
            },
            "projector": {
                "_target_": "torch.nn.Linear",
                "in_features": 4,
                "out_features": output_dim,
            },
            "transform": {
                "_target_": "torch.nn.Linear",
                "in_features": output_dim,
                "out_features": output_dim,
            },
            "normalize_projections": True,
        }

    return {
        "_target_": "synth_setter.models.slap_module.SLAPModule",
        "audio_encoder": arm(128 if audio_input_key == "mel" else 5),
        "text_encoder": arm(2),
        "loss_fn": {"_target_": "synth_setter.models.components.slap.BYOLLoss"},
        "optimizer": {"_target_": "torch.optim.SGD", "_partial_": True, "lr": 0.1},
        "compile": False,
        "audio_input_key": audio_input_key,
    }


def _source(
    path: Path,
    rows: int = 2,
    *,
    row_uuid: list[str] | None = None,
    payload_offset: int = 0,
    audio_row: list[float] | None = None,
    audio_rows: list[list[float]] | None = None,
    mel_row: list[float] | None = None,
    param_row: list[float] | None = None,
    param_rows: list[list[float]] | None = None,
) -> int:
    audio_row = audio_row or [0.1, 0.2, 0.3, 0.4, 0.5]
    mel_row = mel_row or [-0.5] * 128
    param_row = param_row or [0.75, 0.25]
    audio = (
        np.asarray(audio_rows, dtype=np.float32).reshape(rows, 1, 5)
        if audio_rows is not None
        else np.tile(np.array([[audio_row]], dtype=np.float32), (rows, 1, 1))
    )
    mel = np.tile(np.array(mel_row, dtype=np.float32).reshape(1, 1, 128, 1), (rows, 1, 1, 1))
    params = (
        np.asarray(param_rows, dtype=np.float32)
        if param_rows is not None
        else np.tile(np.array([param_row], dtype=np.float32), (rows, 1))
    )
    if rows:
        audio_column = pa.FixedShapeTensorArray.from_numpy_ndarray(audio)
        mel_column = pa.FixedShapeTensorArray.from_numpy_ndarray(mel)
        param_column = pa.FixedShapeTensorArray.from_numpy_ndarray(params)
    else:
        audio_column = pa.array([], pa.fixed_shape_tensor(pa.float32(), [1, 5]))
        mel_column = pa.array([], pa.fixed_shape_tensor(pa.float32(), [1, 128, 1]))
        param_column = pa.array([], pa.fixed_shape_tensor(pa.float32(), [2]))
    columns: dict[str, pa.Array] = {
        "audio": audio_column,
        "mel_spec": mel_column,
        "param_array": param_column,
        "payload": pa.array(np.arange(rows, dtype=np.int64) + payload_offset),
    }
    if row_uuid is not None:
        columns["row_uuid"] = pa.array(row_uuid, pa.string())
    shard_metadata = ShardMetadata(
        velocity=100,
        signal_duration_seconds=5 / 16_000,
        sample_rate=16_000,
        channels=1,
        min_loudness=-60.0,
    )
    metadata = {
        SHARD_METADATA_SCHEMA_KEY: shard_metadata.model_dump_json().encode(),
        b"unrelated": b"keep",
    }
    return lance.write_dataset(pa.table(columns).replace_schema_metadata(metadata), path).version


def _checkpoint(
    path: Path,
    output_dim: int = 2,
    *,
    audio_input_key: Literal["audio", "mel"] = "audio",
    training_audio: torch.Tensor | None = None,
    training_params: torch.Tensor | None = None,
) -> SLAPModule:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = _model(output_dim, audio_input_key=audio_input_key)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        default_audio = (
            torch.full((1, 128, 1), -0.5)
            if audio_input_key == "mel"
            else torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        )
        rows = [
            {
                audio_input_key: default_audio if training_audio is None else training_audio,
                "params": (
                    torch.tensor([0.5, -0.5]) if training_params is None else training_params
                ),
            }
        ]
        loader = DataLoader(cast(Dataset[dict[str, torch.Tensor]], rows), batch_size=1)
        trainer.fit(model, train_dataloaders=loader)
        trainer.save_checkpoint(path)
    return model


def _config(
    source: Path, output: Path, checkpoint: Path, version: int, **updates: object
) -> ExportSlapConfig:
    values: dict[str, object] = {
        "source_root_uri": str(source),
        "output_root_uri": str(output),
        "splits": ("train",),
        "source_versions": {"train": version},
        "ckpt_path": checkpoint,
        "model": _model_config(),
        "batch_size": 2,
        "build_index": False,
        "metric": "cosine",
        "num_partitions": 1,
        "num_sub_vectors": 1,
        "use_saved_mean_and_variance": False,
    }
    values.update(updates)
    return ExportSlapConfig.model_validate(values)


def test_export_slap_real_checkpoint_writes_linked_normalized_ema_rows(tmp_path: Path) -> None:
    """Export only linked, normalized EMA projections.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance")
    checkpoint = tmp_path / "model.ckpt"
    trained = _checkpoint(checkpoint)
    with torch.inference_mode():
        _, expected_audio, _ = trained.audio_ema(
            torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
        )
        _, expected_param, _ = trained.text_ema(torch.tensor([[0.5, -0.5]], dtype=torch.float32))
        _, online_audio, _ = trained.audio_encoder(
            torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
        )

    result = export_slap(_config(source, output, checkpoint, version))["train"]

    exported = lance.dataset(output / "train.lance")
    table = exported.to_table()
    expected_schema = pa.schema(
        [
            pa.field("row_uuid", pa.string(), nullable=False),
            pa.field("is_param_embedding", pa.bool_(), nullable=False),
            pa.field("slap", pa.list_(pa.float32(), 2), nullable=False),
        ]
    )
    assert exported.schema.remove_metadata().equals(expected_schema)
    assert table.num_rows == 4
    assert table["is_param_embedding"].to_pylist() == [True, False, True, False]
    uuids = table["row_uuid"].to_pylist()
    assert uuids[0] == uuids[1] and uuids[2] == uuids[3] and uuids[0] != uuids[2]
    assert all(UUID(value) for value in uuids)
    vectors = table["slap"].combine_chunks().values.to_numpy().reshape(4, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(vectors[0], expected_param.numpy()[0], atol=1e-6)
    np.testing.assert_allclose(vectors[1], expected_audio.numpy()[0], atol=1e-6)
    assert not np.allclose(vectors[1], online_audio.numpy()[0])
    source_ds = lance.dataset(source / "train.lance")
    assert source_ds.to_table(columns=["payload"])["payload"].to_pylist() == [0, 1]
    assert source_ds.schema.metadata[b"unrelated"] == b"keep"
    pointer = json.loads(source_ds.schema.metadata[SLAP_SOURCE_POINTER_KEY])
    provenance = json.loads(exported.schema.metadata[SLAP_EXPORT_METADATA_KEY])
    assert pointer["output_version"] == exported.version == result.output_version
    assert provenance["source_version"] == result.source_version
    assert provenance["checkpoint_sha256"]


def test_export_slap_distinct_rows_preserve_projection_pairing(tmp_path: Path) -> None:
    """Each source row keeps its own parameter and waveform projections.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    source_uuids = [str(uuid4()), str(uuid4()), str(uuid4())]
    audio_rows = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [-0.2, -0.1, 0.0, 0.1, 0.2],
        [0.7, 0.5, 0.3, 0.1, -0.1],
    ]
    param_rows = [[0.75, 0.25], [0.1, 0.9], [0.6, 0.4]]
    version = _source(
        source / "train.lance",
        rows=3,
        row_uuid=source_uuids,
        audio_rows=audio_rows,
        param_rows=param_rows,
    )
    checkpoint = tmp_path / "model.ckpt"
    trained = _checkpoint(checkpoint)
    prepared = prepare_batch(
        {
            "audio": np.asarray(audio_rows, dtype=np.float32).reshape(3, 1, 5),
            "param_array": np.asarray(param_rows, dtype=np.float32),
        },
        mean=None,
        std=None,
        rescale_params=True,
        ot=False,
        generator=torch.Generator().manual_seed(23),
    )
    with torch.inference_mode():
        _, expected_audio, _ = trained.audio_ema(cast(torch.Tensor, prepared["audio"]))
        _, expected_params, _ = trained.text_ema(cast(torch.Tensor, prepared["params"]))

    export_slap(_config(source, output, checkpoint, version, batch_size=2))

    table = lance.dataset(output / "train.lance").to_table()
    vectors = table["slap"].combine_chunks().values.to_numpy().reshape(6, 2)
    np.testing.assert_array_equal(table["row_uuid"].to_pylist(), np.repeat(source_uuids, 2))
    assert table["is_param_embedding"].to_pylist() == [True, False] * 3
    np.testing.assert_allclose(vectors[::2], expected_params.numpy(), atol=1e-6)
    np.testing.assert_allclose(vectors[1::2], expected_audio.numpy(), atol=1e-6)


def test_export_slap_real_mel_checkpoint_uses_stored_mel_for_ema_projection(
    tmp_path: Path,
) -> None:
    """Mel-configured exports ignore distinct stored waveform values.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    mel = [-0.5] * 128
    audio = [0.5, 0.4, 0.3, 0.2, 0.1]
    version = _source(source / "train.lance", rows=1, audio_row=audio, mel_row=mel)
    checkpoint = tmp_path / "model.ckpt"
    trained = _checkpoint(checkpoint, audio_input_key="mel")
    config = _config(source, output, checkpoint, version).model_copy(
        update={"model": _model_config(audio_input_key="mel")}
    )
    with torch.inference_mode():
        _, expected_mel, _ = trained.audio_ema(torch.tensor(mel).reshape(1, 1, 128, 1))

    export_slap(config)

    vectors = (
        lance.dataset(output / "train.lance")
        .to_table(columns=["slap"])["slap"]
        .combine_chunks()
        .values.to_numpy()
        .reshape(2, 2)
    )
    np.testing.assert_allclose(vectors[1], expected_mel.numpy()[0], atol=1e-6)


def test_export_slap_normalized_mel_matches_training_preparation(tmp_path: Path) -> None:
    """Mel and parameter EMA inputs match the training preparation contract.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    mel_row = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    version = _source(
        source / "train.lance",
        rows=1,
        mel_row=mel_row.tolist(),
        param_row=[0.75, 0.25],
    )
    mean = np.linspace(-0.25, 0.25, 128, dtype=np.float32).reshape(1, 128, 1)
    std = np.linspace(0.5, 1.5, 128, dtype=np.float32).reshape(1, 128, 1)
    stats_path = tmp_path / "training-stats.npz"
    np.savez(stats_path, mean=mean, std=std)
    prepared = prepare_batch(
        {
            "mel_spec": mel_row.reshape(1, 1, 128, 1),
            "param_array": np.array([[0.75, 0.25]], dtype=np.float32),
        },
        mean=mean,
        std=std,
        rescale_params=True,
        ot=False,
        generator=torch.Generator().manual_seed(17),
    )
    prepared_mel = prepared["mel"]
    prepared_params = prepared["params"]
    assert isinstance(prepared_mel, torch.Tensor)
    assert isinstance(prepared_params, torch.Tensor)
    checkpoint = tmp_path / "model.ckpt"
    trained = _checkpoint(
        checkpoint,
        audio_input_key="mel",
        training_audio=prepared_mel[0],
        training_params=prepared_params[0],
    )
    config = _config(
        source,
        output,
        checkpoint,
        version,
        model=_model_config(audio_input_key="mel"),
        use_saved_mean_and_variance=True,
        mel_stats_path=stats_path,
    )
    with torch.inference_mode():
        _, expected_audio, _ = trained.audio_ema(prepared_mel)
        _, expected_params, _ = trained.text_ema(prepared_params)

    export_slap(config)

    exported = lance.dataset(output / "train.lance")
    vectors = (
        exported.to_table(columns=["slap"])["slap"]
        .combine_chunks()
        .values.to_numpy()
        .reshape(2, 2)
    )
    np.testing.assert_allclose(vectors[0], expected_params.numpy()[0], atol=1e-6)
    np.testing.assert_allclose(vectors[1], expected_audio.numpy()[0], atol=1e-6)
    metadata = json.loads(exported.schema.metadata[SLAP_EXPORT_METADATA_KEY])
    assert metadata["parameter_preprocessing"] == "stored_[0,1]_to_model_[-1,1]"
    assert metadata["mel_preprocessing"] == "normalize_with_saved_mean_and_variance"
    assert metadata["mel_stats_sha256"] == export_slap_module._file_sha256(stats_path)


def test_export_slap_mel_missing_required_stats_fails_before_uuid_migration(
    tmp_path: Path,
) -> None:
    """A normalized mel export requires explicit local training statistics.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "train.lance"
    version = _source(source_path, rows=1)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint, audio_input_key="mel")
    config = _config(
        source,
        tmp_path / "output",
        checkpoint,
        version,
        model=_model_config(audio_input_key="mel"),
        use_saved_mean_and_variance=True,
    )

    with pytest.raises(ValueError, match="mel_stats_path"):
        export_slap(config)

    assert lance.dataset(source_path).version == version
    assert "row_uuid" not in lance.dataset(source_path).schema.names


@pytest.mark.parametrize(
    ("mean", "std", "message"),
    [
        (np.array([np.nan]), np.array([1.0]), "mean.*finite"),
        (np.array([0.0]), np.array([np.inf]), "std.*finite"),
        (np.array([0.0]), np.array([0.0]), "std.*positive"),
    ],
)
def test_export_slap_invalid_mel_stats_fails_before_uuid_migration(
    tmp_path: Path, mean: np.ndarray, std: np.ndarray, message: str
) -> None:
    """Invalid mel statistics cannot mutate a source.

    :param tmp_path: Isolated dataset root.
    :param mean: Invalid or paired mean statistic.
    :param std: Invalid or paired standard-deviation statistic.
    :param message: Expected validation category.
    """
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "train.lance"
    version = _source(source_path, rows=1)
    stats_path = tmp_path / "stats.npz"
    np.savez(stats_path, mean=mean, std=std)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint, audio_input_key="mel")
    config = _config(
        source,
        tmp_path / "output",
        checkpoint,
        version,
        model=_model_config(audio_input_key="mel"),
        use_saved_mean_and_variance=True,
        mel_stats_path=stats_path,
    )

    with pytest.raises(ValueError, match=message):
        export_slap(config)

    assert lance.dataset(source_path).version == version
    assert "row_uuid" not in lance.dataset(source_path).schema.names


def test_export_slap_mel_stats_broadcast_expansion_fails_before_uuid_migration(
    tmp_path: Path,
) -> None:
    """Statistics cannot expand the stored mel geometry.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "train.lance"
    version = _source(source_path, rows=1)
    stats_path = tmp_path / "stats.npz"
    np.savez(
        stats_path,
        mean=np.zeros((2, 128, 1), dtype=np.float32),
        std=np.ones((2, 128, 1), dtype=np.float32),
    )
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint, audio_input_key="mel")
    config = _config(
        source,
        tmp_path / "output",
        checkpoint,
        version,
        model=_model_config(audio_input_key="mel"),
        use_saved_mean_and_variance=True,
        mel_stats_path=stats_path,
    )

    with pytest.raises(ValueError, match="broadcast|shape|geometry"):
        export_slap(config)

    assert lance.dataset(source_path).version == version
    assert "row_uuid" not in lance.dataset(source_path).schema.names


def test_export_slap_mel_preprocessing_selection_conflicts_with_existing_output(
    tmp_path: Path,
) -> None:
    """Normalized and raw-mel exports cannot share one destination.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance", rows=1)
    stats_path = tmp_path / "stats.npz"
    np.savez(stats_path, mean=np.array([0.0]), std=np.array([1.0]))
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint, audio_input_key="mel")
    normalized = _config(
        source,
        output,
        checkpoint,
        version,
        model=_model_config(audio_input_key="mel"),
        use_saved_mean_and_variance=True,
        mel_stats_path=stats_path,
    )
    export_slap(normalized)

    with pytest.raises(ValueError, match="conflicting request"):
        export_slap(
            normalized.model_copy(
                update={"use_saved_mean_and_variance": False, "mel_stats_path": None}
            )
        )


def test_export_slap_stats_changed_while_loading_fails_before_uuid_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A statistics replacement during loading cannot create mixed provenance.

    :param tmp_path: Isolated dataset root.
    :param monkeypatch: Pytest patching fixture.
    """
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "train.lance"
    version = _source(source_path, rows=1)
    stats_path = tmp_path / "stats.npz"
    np.savez(stats_path, mean=np.array([0.0]), std=np.array([1.0]))
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint, audio_input_key="mel")
    config = _config(
        source,
        tmp_path / "output",
        checkpoint,
        version,
        model=_model_config(audio_input_key="mel"),
        use_saved_mean_and_variance=True,
        mel_stats_path=stats_path,
    )
    original_load = export_slap_module.load_mel_statistics

    def load_then_replace(path: Path) -> tuple[np.ndarray, np.ndarray]:
        loaded = original_load(path)
        np.savez(path, mean=np.array([0.25]), std=np.array([1.0]))
        return loaded

    monkeypatch.setattr(export_slap_module, "load_mel_statistics", load_then_replace)

    with pytest.raises(ValueError, match="statistics changed while loading"):
        export_slap(config)

    assert lance.dataset(source_path).version == version
    assert "row_uuid" not in lance.dataset(source_path).schema.names


def test_export_slap_changed_mel_stats_conflicts_with_existing_output(tmp_path: Path) -> None:
    """Changed statistics content cannot reuse an existing output.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    source_path = source / "train.lance"
    version = _source(source_path, rows=1)
    stats_path = tmp_path / "stats.npz"
    np.savez(stats_path, mean=np.array([0.0]), std=np.array([1.0]))
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint, audio_input_key="mel")
    config = _config(
        source,
        output,
        checkpoint,
        version,
        model=_model_config(audio_input_key="mel"),
        use_saved_mean_and_variance=True,
        mel_stats_path=stats_path,
    )
    export_slap(config)
    source_version = lance.dataset(source_path).version
    np.savez(stats_path, mean=np.array([0.25]), std=np.array([1.0]))

    with pytest.raises(ValueError, match="conflicting request"):
        export_slap(config)

    assert lance.dataset(source_path).version == source_version


def test_export_slap_audio_model_default_normalization_requires_no_stats(tmp_path: Path) -> None:
    """Waveform models ignore the mel-statistics default.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    source.mkdir()
    version = _source(source / "train.lance", rows=1)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    values = _config(source, tmp_path / "output", checkpoint, version).model_dump()
    values.pop("use_saved_mean_and_variance")

    export_slap(ExportSlapConfig.model_validate(values))

    assert lance.dataset(tmp_path / "output" / "train.lance").count_rows() == 2


def test_export_slap_incompatible_parameter_width_fails_before_uuid_migration(
    tmp_path: Path,
) -> None:
    """Model compatibility is checked before a UUID-less source is changed.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    source.mkdir()
    train_path = source / "train.lance"
    val_path = source / "val.lance"
    train_version = _source(train_path, rows=2)
    val_version = _source(val_path, rows=2, param_row=[0.1, 0.2, 0.3])
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, tmp_path / "output", checkpoint, train_version).model_copy(
        update={
            "splits": ("train", "val"),
            "source_versions": {"train": train_version, "val": val_version},
        }
    )

    with pytest.raises(ValueError, match="source.*param_array.*shape.*incompatible.*model") as exc:
        export_slap(config)

    assert isinstance(exc.value.__cause__, RuntimeError)
    for path, version in [(train_path, train_version), (val_path, val_version)]:
        unchanged = lance.dataset(path)
        assert unchanged.version == version
        assert "row_uuid" not in unchanged.schema.names


def test_export_slap_no_index_allows_width_not_divisible_by_pq_subvectors(
    tmp_path: Path,
) -> None:
    """PQ divisibility is irrelevant when no index is requested.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance", rows=1)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint, output_dim=3)
    config = _config(source, output, checkpoint, version).model_copy(
        update={"model": _model_config(output_dim=3), "num_sub_vectors": 16}
    )

    export_slap(config)

    assert lance.dataset(output / "train.lance").schema.field("slap").type.list_size == 3


def test_export_slap_uuid_migration_failure_same_request_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed first export can reuse its attributable UUID migration.

    :param tmp_path: Isolated dataset root.
    :param monkeypatch: Pytest patching fixture.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    version = _source(path)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, version)
    original_project_batch = export_slap_module._project_batch

    def fail_projection_once(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("transient projection failure")

    with monkeypatch.context() as context:
        context.setattr(export_slap_module, "_project_batch", fail_projection_once)
        with pytest.raises(OSError, match="transient projection failure"):
            export_slap(config)

    migrated = lance.dataset(path)
    migrated_uuids = migrated.to_table(columns=["row_uuid"])["row_uuid"].to_pylist()
    assert not (output / "train.lance").exists()

    export_slap(config)

    assert export_slap_module._project_batch is original_project_batch
    assert (
        lance.dataset(path).to_table(columns=["row_uuid"])["row_uuid"].to_pylist()
        == migrated_uuids
    )
    assert lance.dataset(output / "train.lance").count_rows() == 4


def test_export_slap_historical_uuid_snapshot_publishes_selected_version_pointer(
    tmp_path: Path,
) -> None:
    """A historical UUID snapshot remains a deliberate export source.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    historical_version = _source(path, row_uuid=[str(uuid4()), str(uuid4())])
    lance.dataset(path).update_schema_metadata({"later": "metadata"})
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, historical_version)

    first = export_slap(config)["train"]
    second = export_slap(config)["train"]

    source_dataset = lance.dataset(path)
    pointer = json.loads(source_dataset.schema.metadata[SLAP_SOURCE_POINTER_KEY])
    assert first == second
    assert pointer["input_source_version"] == historical_version
    assert source_dataset.schema.metadata[b"later"] == b"metadata"


def test_export_slap_identical_rerun_reuses_output_and_source_versions(tmp_path: Path) -> None:
    """An identical completed request changes neither dataset.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance")
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, version)
    first = export_slap(config)["train"]
    source_version = lance.dataset(source / "train.lance").version

    second = export_slap(config)["train"]

    assert second == first
    assert lance.dataset(source / "train.lance").version == source_version
    assert lance.dataset(output / "train.lance").version == first.output_version


def test_export_slap_source_recreated_before_pointer_rejects_wrong_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointer publication rejects a source recreated after output creation.

    :param tmp_path: Isolated dataset root.
    :param monkeypatch: Pytest patching fixture.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    version = _source(path, row_uuid=[str(uuid4()), str(uuid4())])
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, version)
    original_complete = export_slap_module._complete_output

    def complete_then_recreate(
        dataset: lance.LanceDataset,
        metadata: _ExportMetadata,
        *,
        config: ExportSlapConfig,
        output_uri: str,
    ) -> tuple[lance.LanceDataset, _ExportMetadata]:
        completed = original_complete(dataset, metadata, config=config, output_uri=output_uri)
        shutil.rmtree(path)
        _source(path, row_uuid=[str(uuid4()), str(uuid4())], payload_offset=100)
        return completed

    monkeypatch.setattr(export_slap_module, "_complete_output", complete_then_recreate)

    with pytest.raises(ValueError, match="transaction identity"):
        export_slap(config)

    assert SLAP_SOURCE_POINTER_KEY not in (lance.dataset(path).schema.metadata or {})
    assert lance.dataset(output / "train.lance").count_rows() == 4


def test_export_slap_completed_reuse_rejects_recreated_source_identity(tmp_path: Path) -> None:
    """A recreated source cannot reuse vectors from the former transaction lineage.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    version = _source(path)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, version)
    export_slap(config)

    shutil.rmtree(path)
    replacement_uuids = [str(uuid4()), str(uuid4())]
    _source(path, row_uuid=replacement_uuids, payload_offset=100)
    lance.dataset(path).update_schema_metadata({"replacement": "different-lineage"})

    with pytest.raises(ValueError, match="conflicting request|transaction identity"):
        export_slap(config)

    assert SLAP_SOURCE_POINTER_KEY not in (lance.dataset(path).schema.metadata or {})


def test_ensure_row_uuids_returns_uuid_commit_not_concurrent_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UUID migration pins its own commit when another writer advances the head.

    :param tmp_path: Isolated dataset root.
    :param monkeypatch: Pytest patching fixture.
    """
    path = tmp_path / "source.lance"
    requested_version = _source(path, rows=1)
    head = lance.dataset(path)
    original_add_columns = head.add_columns

    def add_then_advance(
        transforms: (
            dict[str, str]
            | Callable[[pa.RecordBatch], pa.RecordBatch]
            | pa.RecordBatchReader
            | pa.Table
            | pa.Field
            | list[pa.Field]
            | pa.Schema
        ),
        read_columns: list[str] | None = None,
        reader_schema: pa.Schema | None = None,
        batch_size: int | None = None,
    ) -> None:
        original_add_columns(
            transforms,
            read_columns=read_columns,
            reader_schema=reader_schema,
            batch_size=batch_size,
        )
        lance.dataset(path).update_schema_metadata({"concurrent": "metadata"})

    monkeypatch.setattr(head, "add_columns", add_then_advance)
    open_count = 0

    def open_source(uri: str) -> lance.LanceDataset:
        nonlocal open_count
        del uri
        open_count += 1
        return head if open_count == 1 else lance.dataset(path)

    monkeypatch.setattr(export_slap_module, "_open", open_source)

    migrated = _ensure_row_uuids(str(path), requested_version, batch_size=1)

    assert migrated.version == 2
    assert lance.dataset(path).version == 3
    assert "row_uuid" in migrated.schema.names


def test_export_slap_rejects_duplicate_existing_uuid(tmp_path: Path) -> None:
    """Duplicate source UUIDs cannot enter a retrieval dataset.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    duplicate = "12345678-1234-5678-9234-567812345678"
    version = _source(source / "train.lance", row_uuid=[duplicate, duplicate])
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match="row_uuid"):
        export_slap(_config(source, output, checkpoint, version))

    assert not (output / "train.lance").exists()


@pytest.mark.parametrize("bad_uuid", [None, "not-a-uuid"])
def test_export_slap_rejects_null_or_invalid_existing_uuid(
    tmp_path: Path, bad_uuid: str | None
) -> None:
    """Null and malformed source UUIDs fail validation.

    :param tmp_path: Isolated dataset root.
    :param bad_uuid: Invalid UUID value under test.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    _source(path, row_uuid=[str(uuid4()), str(uuid4())])
    dataset = lance.dataset(path)
    dataset.update(
        {"row_uuid": "NULL" if bad_uuid is None else f"'{bad_uuid}'"}, where="payload = 0"
    )
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match="row_uuid"):
        export_slap(_config(source, output, checkpoint, lance.dataset(path).version))


def test_export_slap_historical_source_rejects_unattributed_uuid_head(
    tmp_path: Path,
) -> None:
    """An unrelated UUID-bearing successor is not adopted as a migration retry.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    historical_version = _source(path)
    head = lance.dataset(path)
    head.add_columns(pa.field("row_uuid", pa.string(), nullable=True))
    head.update({"row_uuid": f"'{uuid4()}'"})
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match="historical pinned source lacks row_uuid"):
        export_slap(_config(source, output, checkpoint, historical_version))

    assert not (output / "train.lance").exists()


def test_export_slap_historical_source_without_uuid_refuses_new_head_mutation(
    tmp_path: Path,
) -> None:
    """A historical UUID-less pin cannot mutate a newer head.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    historical_version = _source(path)
    lance.dataset(path).update_schema_metadata({"later": "metadata"})
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match="historical pinned source lacks row_uuid"):
        export_slap(_config(source, output, checkpoint, historical_version))

    assert "row_uuid" not in lance.dataset(path).schema.names


def test_export_slap_index_failure_retains_incomplete_output_without_pointer(
    tmp_path: Path,
) -> None:
    """Index failure retains data without publishing completion.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance")
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(RuntimeError, match="training|centroids|rows|PQ"):
        export_slap(_config(source, output, checkpoint, version, build_index=True))

    exported = lance.dataset(output / "train.lance")
    assert json.loads(exported.schema.metadata[SLAP_EXPORT_METADATA_KEY])["completed"] is False
    assert SLAP_SOURCE_POINTER_KEY not in (
        lance.dataset(source / "train.lance").schema.metadata or {}
    )


def test_export_slap_index_failure_same_request_retry_builds_real_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incomplete 256-row output resumes through real index construction.

    :param tmp_path: Isolated dataset root.
    :param monkeypatch: Pytest patching fixture.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance", rows=256)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, version, build_index=True)

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("transient index failure")

    with monkeypatch.context() as context:
        context.setattr(lance.LanceDataset, "create_index", fail_index)
        with pytest.raises(RuntimeError, match="transient index failure"):
            export_slap(config)

    incomplete = lance.dataset(output / "train.lance")
    assert incomplete.count_rows() == 512
    assert json.loads(incomplete.schema.metadata[SLAP_EXPORT_METADATA_KEY])["completed"] is False

    export_slap(config)

    completed = lance.dataset(output / "train.lance")
    assert completed.count_rows() == 512
    assert completed.list_indices()[0]["fields"] == ["slap"]
    assert json.loads(completed.schema.metadata[SLAP_EXPORT_METADATA_KEY])["completed"] is True


def test_export_slap_missing_checkpoint_state_fails_before_source_mutation(tmp_path: Path) -> None:
    """Strict checkpoint loading precedes source mutation.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    version = _source(path)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["state_dict"].pop("audio_ema.projector.weight")
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="Missing key"):
        export_slap(_config(source, output, checkpoint, version))

    assert "row_uuid" not in lance.dataset(path).schema.names


@pytest.mark.parametrize("modality", ["audio EMA", "parameter EMA"])
@pytest.mark.parametrize(
    ("projection", "message"),
    [
        (torch.ones(2), "invalid shape"),
        (torch.empty((1, 0)), "invalid shape"),
        (torch.tensor([[float("nan"), 1.0]]), "nonfinite"),
        (torch.zeros((1, 2)), "zero or invalid"),
    ],
)
def test_normalize_invalid_projection_rejects_modality_contract(
    modality: str, projection: torch.Tensor, message: str
) -> None:
    """Both EMA modalities enforce rank, width, finiteness, and nonzero norms.

    :param modality: Projection arm under test.
    :param projection: Invalid projection tensor.
    :param message: Expected validation category.
    """
    with pytest.raises(ValueError, match=message):
        _normalize(projection, modality)


@pytest.mark.parametrize("magnitude", [1e20, 1e-30])
def test_normalize_extreme_finite_vectors_returns_unit_float32(magnitude: float) -> None:
    """Finite float32 magnitudes normalize without overflow or underflow.

    :param magnitude: Large or tiny finite component magnitude.
    """
    normalized = _normalize(torch.tensor([[magnitude, magnitude]], dtype=torch.float32), "EMA")

    np.testing.assert_allclose(normalized, [[np.sqrt(0.5), np.sqrt(0.5)]], atol=1e-6)
    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous


def test_validate_projection_pair_different_width_rejects_batch() -> None:
    """EMA modalities must share a projection width."""
    with pytest.raises(ValueError, match="dimensions differ"):
        _validate_projection_pair(torch.ones((1, 2)), torch.ones((1, 3)), batch_rows=1)


def test_validate_projection_pair_wrong_cardinality_rejects_batch() -> None:
    """EMA modalities must preserve source batch cardinality."""
    with pytest.raises(ValueError, match="cardinality"):
        _validate_projection_pair(torch.ones((1, 2)), torch.ones((1, 2)), batch_rows=2)


@pytest.mark.parametrize("missing_field", ["audio", "param_array"])
def test_export_slap_missing_input_field_fails_before_uuid_mutation(
    tmp_path: Path, missing_field: str
) -> None:
    """Required model inputs are checked before UUID migration.

    :param tmp_path: Isolated dataset root.
    :param missing_field: Required field removed from the source.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    _source(path)
    table = lance.dataset(path).to_table().drop([missing_field])
    shutil.rmtree(path)
    version = lance.write_dataset(table, path).version
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match=missing_field):
        export_slap(_config(source, output, checkpoint, version))

    assert "row_uuid" not in lance.dataset(path).schema.names


@pytest.mark.parametrize("wrong_field", ["audio", "param_array"])
def test_export_slap_plain_list_input_field_fails_before_uuid_mutation(
    tmp_path: Path, wrong_field: str
) -> None:
    """Plain Arrow lists are not accepted as shaped model tensors.

    :param tmp_path: Isolated dataset root.
    :param wrong_field: Tensor field replaced by a plain list field.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    _source(path)
    table = lance.dataset(path).to_table()
    index = table.schema.get_field_index(wrong_field)
    width = 5 if wrong_field == "audio" else 2
    plain_lists = pa.array([[0.1] * width, [0.2] * width])
    table = table.set_column(index, wrong_field, plain_lists)
    shutil.rmtree(path)
    version = lance.write_dataset(table, path).version
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match=f"{wrong_field}.*FixedShapeTensor"):
        export_slap(_config(source, output, checkpoint, version))

    assert "row_uuid" not in lance.dataset(path).schema.names


@pytest.mark.parametrize(
    ("audio_row", "param_row", "message"),
    [
        ([float("nan"), 0.0, 0.0, 0.0, 0.0], None, "audio.*nonfinite"),
        (None, [float("inf"), 0.0], "param_array.*nonfinite"),
        (None, [1.01, 0.0], "param_array.*within.*1"),
        ([1.01, 0.0, 0.0, 0.0, 0.0], None, "within.*1"),
    ],
)
def test_export_slap_invalid_stored_input_fails_before_uuid_mutation(
    tmp_path: Path,
    audio_row: list[float] | None,
    param_row: list[float] | None,
    message: str,
) -> None:
    """Nonfinite tensors and out-of-range waveforms never reach inference.

    :param tmp_path: Isolated dataset root.
    :param audio_row: Optional invalid waveform row.
    :param param_row: Optional invalid parameter row.
    :param message: Expected validation category.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    version = _source(path, audio_row=audio_row, param_row=param_row)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match=message):
        export_slap(_config(source, output, checkpoint, version))

    assert "row_uuid" not in lance.dataset(path).schema.names


def test_export_slap_zero_ema_projection_publishes_no_output_pointer(tmp_path: Path) -> None:
    """Invalid EMA vectors publish neither output nor pointer.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    path = source / "train.lance"
    version = _source(path)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["state_dict"]["audio_ema.projector.weight"].zero_()
    payload["state_dict"]["audio_ema.projector.bias"].zero_()
    torch.save(payload, checkpoint)

    with pytest.raises(OSError, match="zero or invalid vectors"):
        export_slap(_config(source, output, checkpoint, version))

    assert not (output / "train.lance").exists()
    assert SLAP_SOURCE_POINTER_KEY not in (lance.dataset(path).schema.metadata or {})


def test_existing_completed_unrelated_not_found_error_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arbitrary storage error mentioning not found is not treated as absence.

    :param monkeypatch: Pytest patching fixture.
    """

    def fail_open(uri: str) -> lance.LanceDataset:
        raise ValueError(f"credentials not found while opening {uri}")

    monkeypatch.setattr(export_slap_module, "_open", fail_open)

    with pytest.raises(ValueError, match="credentials not found"):
        _existing_completed("unused", "request")


def test_export_slap_conflicting_destination_is_never_overwritten(tmp_path: Path) -> None:
    """An output owned by another request is preserved.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    version = _source(source / "train.lance")
    lance.write_dataset(pa.table({"owner": ["other"]}), output / "train.lance")
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match="conflicting request"):
        export_slap(_config(source, output, checkpoint, version))

    assert lance.dataset(output / "train.lance").schema.names == ["owner"]


def test_export_slap_multiple_splits_writes_separate_datasets(tmp_path: Path) -> None:
    """Each selected split receives an independent dataset.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    train_version = _source(source / "train.lance", rows=1)
    val_version = _source(source / "val.lance", rows=1)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, train_version).model_copy(
        update={
            "splits": ("train", "val"),
            "source_versions": {"train": train_version, "val": val_version},
        }
    )

    results = export_slap(config)

    assert set(results) == {"train", "val"}
    assert lance.dataset(output / "train.lance").count_rows() == 2
    assert lance.dataset(output / "val.lance").count_rows() == 2


def test_export_slap_config_rejects_local_file_uri_alias(tmp_path: Path) -> None:
    """Bare paths and equivalent file URIs cannot alias.

    :param tmp_path: Isolated dataset root.
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    values = _config(tmp_path / "source", tmp_path / "output", checkpoint, 1).model_dump()
    values.update(
        source_root_uri=str(tmp_path / "same"), output_root_uri=(tmp_path / "same").as_uri()
    )

    with pytest.raises(ValueError, match="must differ"):
        ExportSlapConfig.model_validate(values)


def test_export_slap_config_rejects_r2_s3_alias(tmp_path: Path) -> None:
    """Canonical R2 and S3-compatible forms cannot alias.

    :param tmp_path: Isolated dataset root.
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    values = _config(tmp_path / "source", tmp_path / "output", checkpoint, 1).model_dump()
    values.update(
        source_root_uri="r2://test-bucket/shared",
        output_root_uri="s3://test-bucket/shared",
    )

    with pytest.raises(ValueError, match="must differ"):
        ExportSlapConfig.model_validate(values)


def test_export_slap_empty_split_publishes_completed_empty_dataset(tmp_path: Path) -> None:
    """An empty split has an explicit completed dataset.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance", rows=0)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    export_slap(_config(source, output, checkpoint, version))

    exported = lance.dataset(output / "train.lance")
    assert exported.count_rows() == 0
    assert json.loads(exported.schema.metadata[SLAP_EXPORT_METADATA_KEY])["completed"] is True


def test_export_slap_cli_trained_checkpoint_supports_ann_search_and_source_join(
    tmp_path: Path,
) -> None:
    """The installed CLI produces an ANN-searchable, source-joinable export.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance", rows=256)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = {
        **_config(source, output, checkpoint, version, build_index=True).model_dump(mode="json"),
        "synth": {"name": "test"},
    }
    config_path = tmp_path / "export.yaml"
    config_path.write_text(OmegaConf.to_yaml(config))

    completed = subprocess.run(  # noqa: S603 - trusted interpreter and fixed module
        [
            str(Path(sys.executable).with_name("synth-setter-export-slap")),
            "--config-dir",
            str(tmp_path),
            "--config-name",
            "export",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    exported = lance.dataset(output / "train.lance")
    query = np.asarray(exported.to_table(columns=["slap"])["slap"][0].as_py(), dtype=np.float32)
    nearest = exported.to_table(nearest={"column": "slap", "q": query, "k": 1})
    matched_uuid = nearest["row_uuid"][0].as_py()
    source_match = lance.dataset(source / "train.lance").to_table(
        filter=f"row_uuid = '{matched_uuid}'", columns=["payload"]
    )
    assert source_match.num_rows == 1
    assert exported.list_indices()[0]["fields"] == ["slap"]


@pytest.mark.parametrize("model", ["slap_ast_audio_mlp_param", "slap_ast_audio_transformer_param"])
def test_export_slap_cli_shipped_model_resolves_parameter_dimensions(model: str) -> None:
    """Shipped model groups resolve synth-dependent construction settings.

    :param model: Shipped SLAP model group.
    """
    completed = subprocess.run(  # noqa: S603 - trusted interpreter and registered model groups
        [
            sys.executable,
            "-m",
            "synth_setter.cli.export_slap",
            f"model={model}",
            "--cfg",
            "job",
            "--resolve",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "${param_spec_width:" not in completed.stdout


@pytest.mark.parametrize("audio_input_key", ["audio", "mel"])
def test_export_slap_metadata_shape_mismatch_rejects_before_uuid_migration(
    tmp_path: Path, audio_input_key: Literal["audio", "mel"]
) -> None:
    """Source tensor geometry must agree with its declared audio frontend.

    :param tmp_path: Isolated dataset root.
    :param audio_input_key: Stored model input under test.
    """
    source = tmp_path / "source"
    source.mkdir()
    path = source / "train.lance"
    _source(path, rows=1)
    dataset = lance.dataset(path)
    metadata = json.loads(dataset.schema.metadata[SHARD_METADATA_SCHEMA_KEY])
    metadata["signal_duration_seconds"] = 1.0
    dataset.update_schema_metadata({SHARD_METADATA_SCHEMA_KEY.decode(): json.dumps(metadata)})
    version = dataset.version
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint, audio_input_key=audio_input_key)
    config = _config(source, tmp_path / "output", checkpoint, version).model_copy(
        update={"model": _model_config(audio_input_key=audio_input_key)}
    )

    with pytest.raises(ValueError, match="shape.*metadata"):
        export_slap(config)

    unchanged = lance.dataset(path)
    assert unchanged.version == version
    assert "row_uuid" not in unchanged.schema.names


@pytest.mark.parametrize("batch_size", [1, 3])
def test_export_slap_transient_later_batch_resumes_without_replaying_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    """A transient scan failure resumes at the first unconsumed source row.

    :param tmp_path: Isolated dataset root.
    :param monkeypatch: Scoped Lance scan-boundary failure injection.
    :param batch_size: Failure after a partial scan or immediately before exhaustion.
    """
    source = tmp_path / "source"
    source.mkdir()
    source_uuids = [str(uuid4()), str(uuid4()), str(uuid4())]
    audio_rows = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [-0.5, -0.25, 0.0, 0.25, 0.5],
    ]
    param_rows = [[0.75, 0.25], [0.2, 0.8], [0.6, 0.1]]
    version = _source(
        source / "train.lance",
        rows=3,
        row_uuid=source_uuids,
        audio_rows=audio_rows,
        param_rows=param_rows,
    )
    checkpoint = tmp_path / "model.ckpt"
    trained = _checkpoint(checkpoint)
    original_to_batches = cast(
        Callable[..., Iterator[pa.RecordBatch]], lance.LanceDataset.to_batches
    )
    injected = False

    def fail_after_first_output_batch(
        dataset: lance.LanceDataset, *args: object, **kwargs: object
    ) -> Iterator[pa.RecordBatch]:
        nonlocal injected
        batches = original_to_batches(dataset, *args, **kwargs)
        columns = kwargs.get("columns")
        if injected or columns != ["row_uuid", "audio", "param_array"]:
            return batches

        def transient_batches() -> Iterator[pa.RecordBatch]:
            nonlocal injected
            yield next(batches)
            injected = True
            raise TimeoutError("transient object-store read")

        return transient_batches()

    monkeypatch.setattr(lance.LanceDataset, "to_batches", fail_after_first_output_batch)

    export_slap(_config(source, tmp_path / "output", checkpoint, version, batch_size=batch_size))

    table = lance.dataset(tmp_path / "output" / "train.lance").to_table()
    prepared = prepare_batch(
        {
            "audio": np.asarray(audio_rows, dtype=np.float32).reshape(3, 1, 5),
            "param_array": np.asarray(param_rows, dtype=np.float32),
        },
        mean=None,
        std=None,
        rescale_params=True,
        ot=False,
        generator=torch.Generator().manual_seed(29),
    )
    with torch.inference_mode():
        _, expected_audio, _ = trained.audio_ema(cast(torch.Tensor, prepared["audio"]))
        _, expected_params, _ = trained.text_ema(cast(torch.Tensor, prepared["params"]))
    vectors = table["slap"].combine_chunks().values.to_numpy().reshape(6, 2)
    assert injected
    np.testing.assert_array_equal(table["row_uuid"].to_pylist(), np.repeat(source_uuids, 2))
    np.testing.assert_allclose(vectors[::2], expected_params.numpy(), atol=1e-6)
    np.testing.assert_allclose(vectors[1::2], expected_audio.numpy(), atol=1e-6)


def test_export_slap_transient_source_open_retries_to_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient source read failure does not abort a valid export.

    :param tmp_path: Isolated dataset root.
    :param monkeypatch: Scoped object-store boundary failure injection.
    """
    source = tmp_path / "source"
    source.mkdir()
    version = _source(source / "train.lance")
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, tmp_path / "output", checkpoint, version)
    open_dataset = cast(Callable[..., lance.LanceDataset], lance.dataset)
    failed = False

    def transient_open(*args: object, **kwargs: object) -> lance.LanceDataset:
        nonlocal failed
        if not failed:
            failed = True
            raise TimeoutError("transient object-store read")
        return open_dataset(*args, **kwargs)

    monkeypatch.setattr(lance, "dataset", transient_open)

    result = export_slap(config)["train"]

    assert lance.dataset(result.output_uri).count_rows() == 4
