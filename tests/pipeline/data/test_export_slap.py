"""Integration tests for SLAP retrieval export."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any, cast
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

from synth_setter.models.components.slap import BYOLLoss, SiameseArm
from synth_setter.models.slap_module import SLAPModule
from synth_setter.pipeline.data import export_slap as export_slap_module
from synth_setter.pipeline.data.export_slap import (
    SLAP_EXPORT_METADATA_KEY,
    SLAP_SOURCE_POINTER_KEY,
    _ensure_row_uuids,
    export_slap,
)
from synth_setter.pipeline.data.lance_shard import SHARD_METADATA_SCHEMA_KEY
from synth_setter.pipeline.schemas.export_slap_config import ExportSlapConfig
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata


def _arm(input_dim: int, output_dim: int = 2) -> SiameseArm:
    return SiameseArm(
        encoder=nn.Linear(input_dim, 4),
        projector=nn.Linear(4, output_dim),
        transform=nn.Linear(output_dim, output_dim),
        normalize_projections=True,
    )


def _model(output_dim: int = 2) -> SLAPModule:
    return SLAPModule(
        audio_encoder=_arm(5, output_dim),
        text_encoder=_arm(2, output_dim),
        loss_fn=BYOLLoss(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
    )


def _model_config(output_dim: int = 2) -> dict[str, object]:
    def arm(input_dim: int) -> dict[str, object]:
        return {
            "_target_": "synth_setter.models.components.slap.SiameseArm",
            "encoder": {
                "_target_": "torch.nn.Linear",
                "in_features": input_dim,
                "out_features": 4,
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
        "audio_encoder": arm(5),
        "text_encoder": arm(2),
        "loss_fn": {"_target_": "synth_setter.models.components.slap.BYOLLoss"},
        "optimizer": {"_target_": "torch.optim.SGD", "_partial_": True, "lr": 0.1},
        "compile": False,
        "audio_input_key": "audio",
    }


def _source(
    path: Path,
    rows: int = 2,
    *,
    row_uuid: list[str] | None = None,
    payload_offset: int = 0,
) -> int:
    audio = np.tile(np.array([[1, 2, 3, 4, 5]], dtype=np.float32), (rows, 1))
    params = np.tile(np.array([[2, 1]], dtype=np.float32), (rows, 1))
    if rows:
        audio_column = pa.FixedShapeTensorArray.from_numpy_ndarray(audio)
        mel_column = pa.FixedShapeTensorArray.from_numpy_ndarray(audio[:, None, :])
        param_column = pa.FixedShapeTensorArray.from_numpy_ndarray(params)
    else:
        audio_column = pa.array([], pa.fixed_shape_tensor(pa.float32(), [5]))
        mel_column = pa.array([], pa.fixed_shape_tensor(pa.float32(), [1, 5]))
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
        signal_duration_seconds=1.0,
        sample_rate=16_000,
        channels=1,
        min_loudness=-60.0,
    )
    metadata = {
        SHARD_METADATA_SCHEMA_KEY: shard_metadata.model_dump_json().encode(),
        b"unrelated": b"keep",
    }
    return lance.write_dataset(pa.table(columns).replace_schema_metadata(metadata), path).version


def _checkpoint(path: Path, output_dim: int = 2) -> SLAPModule:
    model = _model(output_dim)
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    rows = [{"audio": torch.arange(5, dtype=torch.float32), "params": torch.tensor([2.0, 1.0])}]
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
            torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
        )
        _, expected_param, _ = trained.text_ema(torch.tensor([[2, 1]], dtype=torch.float32))
        _, online_audio, _ = trained.audio_encoder(
            torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
        )

    result = export_slap(_config(source, output, checkpoint, version))["train"]

    exported = lance.dataset(output / "train.lance")
    table = exported.to_table()
    assert exported.schema.names == ["row_uuid", "is_param_embedding", "slap"]
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

    def add_then_advance(*args: object, **kwargs: object) -> object:
        result = cast(Any, original_add_columns)(*args, **kwargs)
        lance.dataset(path).update_schema_metadata({"concurrent": "metadata"})
        return result

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
    values.update(source_root_uri=str(tmp_path / "same"), output_root_uri=(tmp_path / "same").as_uri())

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
    """The module CLI produces an ANN-searchable, source-joinable export.

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
            sys.executable,
            "-m",
            "synth_setter.cli.export_slap",
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
