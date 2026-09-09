"""Behavioral contracts for SLAP retrieval export boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import lance
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.cli.export_slap import _hydra_main
from synth_setter.pipeline.data import export_slap as export_slap_module
from synth_setter.pipeline.data.export_slap import (
    SLAP_EXPORT_METADATA_KEY,
    SLAP_SOURCE_POINTER_KEY,
    _normalize,
    export_slap,
)
from synth_setter.pipeline.data.lance_shard import SHARD_METADATA_SCHEMA_KEY
from synth_setter.pipeline.schemas.export_slap_config import ExportSlapConfig
from tests.pipeline.data.test_export_slap import _checkpoint, _config, _model_config, _source


def _valid_values(tmp_path: Path) -> dict[str, object]:
    """Build the smallest valid configuration mapping.

    :param tmp_path: Isolated filesystem root.
    :returns: Strictly valid exporter fields.
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    return _config(tmp_path / "source", tmp_path / "output", checkpoint, 1).model_dump()


def _align_output_metadata_version(output: Path) -> None:
    """Make tampered output provenance name the next real Lance version.

    :param output: Existing output dataset path.
    """
    dataset = lance.dataset(output)
    metadata = json.loads(dataset.schema.metadata[SLAP_EXPORT_METADATA_KEY])
    metadata["output_version"] = dataset.version + 1
    dataset.update_schema_metadata(
        {SLAP_EXPORT_METADATA_KEY.decode(): json.dumps(metadata)}, replace=False
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"splits": (), "source_versions": {}}, "at least one"),
        ({"splits": ("train", "train")}, "duplicates"),
        ({"source_versions": {"val": 1}}, "exactly match"),
        ({"source_versions": {"train": 0}}, "positive Lance versions"),
        ({"ckpt_path": Path("missing.ckpt")}, "not a file"),
        ({"model": {}}, "must contain _target_"),
    ],
)
def test_export_slap_config_invalid_request_rejects_at_boundary(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    """Invalid requests fail strict validation before export.

    :param tmp_path: Isolated filesystem root.
    :param updates: Invalid field replacement.
    :param message: Expected validation category.
    """
    values = _valid_values(tmp_path)
    values.update(updates)

    with pytest.raises(ValidationError, match=message):
        ExportSlapConfig.model_validate(values)


@pytest.mark.parametrize("field", ["source_root_uri", "output_root_uri"])
def test_export_slap_config_unsupported_root_scheme_rejects_at_boundary(
    tmp_path: Path, field: str
) -> None:
    """Unsupported storage schemes fail before any dataset access.

    :param tmp_path: Isolated filesystem root.
    :param field: Root URI field receiving an unsupported scheme.
    """
    values = _valid_values(tmp_path)
    values[field] = "https://example.test/dataset"

    with pytest.raises(ValidationError, match="unsupported dataset URI scheme"):
        ExportSlapConfig.model_validate(values)


def test_export_slap_config_from_hydra_resolves_fields_and_ignores_composition_keys(
    tmp_path: Path,
) -> None:
    """Hydra composition produces one strict request without unrelated groups.

    :param tmp_path: Isolated filesystem root.
    """
    values = _valid_values(tmp_path)
    values.update(splits=["train"], synth={"name": "surge_xt"})
    cfg = OmegaConf.create(values)

    config = ExportSlapConfig.from_hydra_cfg(cfg)

    assert config.splits == ("train",)
    assert config.ckpt_path == values["ckpt_path"]
    assert "synth" not in config.model_dump()


def test_export_slap_file_uri_roots_write_real_linked_dataset(tmp_path: Path) -> None:
    """File URI roots support the same persisted export contract as bare paths.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance", rows=1)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, version).model_copy(
        update={"source_root_uri": source.as_uri(), "output_root_uri": output.as_uri()}
    )

    result = export_slap(config)["train"]

    assert result.output_row_count == 2
    assert lance.dataset(output / "train.lance").count_rows() == 2


def test_export_slap_malformed_checkpoint_fails_before_source_mutation(tmp_path: Path) -> None:
    """A non-Lightning checkpoint cannot trigger UUID migration.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    source.mkdir()
    path = source / "train.lance"
    version = _source(path)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"weights": {}}, checkpoint)

    with pytest.raises(ValueError, match="Lightning state_dict"):
        export_slap(_config(source, tmp_path / "output", checkpoint, version))

    assert "row_uuid" not in lance.dataset(path).schema.names


def test_export_slap_non_slap_model_fails_before_source_mutation(tmp_path: Path) -> None:
    """A model target outside SLAP cannot trigger UUID migration.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    source.mkdir()
    path = source / "train.lance"
    version = _source(path)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"unused")
    config = _config(source, tmp_path / "output", checkpoint, version).model_copy(
        update={"model": {"_target_": "builtins.dict"}}
    )

    with pytest.raises(ValueError, match="instantiate SLAPModule"):
        export_slap(config)

    assert "row_uuid" not in lance.dataset(path).schema.names


def test_export_slap_noncanonical_uuid_rejects_before_output_creation(tmp_path: Path) -> None:
    """Equivalent but noncanonical UUID text cannot become a join key.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    source.mkdir()
    path = source / "train.lance"
    version = _source(path, rows=1, row_uuid=[str(uuid4()).upper()])
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match="not canonical"):
        export_slap(_config(source, tmp_path / "output", checkpoint, version))

    assert not (tmp_path / "output" / "train.lance").exists()


def test_export_slap_index_requires_projection_width_divisibility(tmp_path: Path) -> None:
    """An indexed export rejects a PQ layout that cannot represent its vectors.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    source.mkdir()
    path = source / "train.lance"
    version = _source(path, rows=1)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint, output_dim=3)
    config = _config(source, tmp_path / "output", checkpoint, version).model_copy(
        update={
            "build_index": True,
            "model": _model_config(output_dim=3),
            "num_sub_vectors": 16,
        }
    )

    with pytest.raises(ValueError, match="must divide"):
        export_slap(config)

    assert not (tmp_path / "output" / "train.lance").exists()


def test_normalize_nonunit_vector_returns_unit_float32_values() -> None:
    """Normalization converts a known non-unit vector without changing direction."""
    normalized = _normalize(torch.tensor([[3.0, 4.0]], dtype=torch.float64), "audio EMA")

    np.testing.assert_array_equal(normalized, np.array([[0.6, 0.8]], dtype=np.float32))
    assert normalized.dtype == np.float32


def test_export_slap_checkpoint_changed_after_hash_rejects_before_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint replacement during startup cannot export unverified weights.

    :param tmp_path: Isolated dataset root.
    :param monkeypatch: Pytest patching fixture.
    """
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "train.lance"
    version = _source(source_path)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    replacement = tmp_path / "replacement.ckpt"
    payload = torch.load(checkpoint, weights_only=False)
    payload["state_dict"]["audio_ema.projector.bias"].add_(0.25)
    torch.save(payload, replacement)
    original_hash = export_slap_module._checkpoint_sha256

    def hash_then_replace(path: Path) -> str:
        digest = original_hash(path)
        path.write_bytes(replacement.read_bytes())
        return digest

    monkeypatch.setattr(export_slap_module, "_checkpoint_sha256", hash_then_replace)

    with pytest.raises(ValueError, match="checkpoint.*chang|hash|SHA"):
        export_slap(_config(source, tmp_path / "output", checkpoint, version))

    assert "row_uuid" not in lance.dataset(source_path).schema.names
    assert not (tmp_path / "output" / "train.lance").exists()


def test_export_slap_empty_source_zero_width_projection_rejects_without_pointer(
    tmp_path: Path,
) -> None:
    """An empty split cannot conceal an unusable zero-width embedding model.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "train.lance"
    version = _source(source_path, rows=0)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint, output_dim=0)
    config = _config(source, tmp_path / "output", checkpoint, version).model_copy(
        update={"model": _model_config(output_dim=0)}
    )

    with pytest.raises(ValueError, match="dimension|width|nonempty|zero"):
        export_slap(config)

    assert not (tmp_path / "output" / "train.lance").exists()
    assert SLAP_SOURCE_POINTER_KEY not in (lance.dataset(source_path).schema.metadata or {})


@pytest.mark.parametrize("metadata", [None, "{"])
def test_export_slap_invalid_shard_metadata_rejects_before_uuid_migration(
    tmp_path: Path, metadata: str | None
) -> None:
    """Missing or malformed shard metadata cannot mutate a UUID-less source.

    :param tmp_path: Isolated dataset root.
    :param metadata: Missing or malformed shard metadata payload.
    """
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "train.lance"
    _source(source_path)
    replacement: dict[str, str | None] = (
        {} if metadata is None else {SHARD_METADATA_SCHEMA_KEY.decode(): metadata}
    )
    dataset = lance.dataset(source_path)
    dataset.update_schema_metadata(replacement, replace=True)
    version = lance.dataset(source_path).version
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError):
        export_slap(_config(source, tmp_path / "output", checkpoint, version))

    current = lance.dataset(source_path)
    assert current.version == version
    assert "row_uuid" not in current.schema.names
    assert SLAP_SOURCE_POINTER_KEY not in (current.schema.metadata or {})


def test_export_slap_distinct_splits_preserve_uuid_and_projection_isolation(
    tmp_path: Path,
) -> None:
    """Each split exports only its own UUIDs and model inputs.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    train_uuid = "11111111-1111-4111-8111-111111111111"
    val_uuid = "22222222-2222-4222-8222-222222222222"
    train_version = _source(
        source / "train.lance",
        rows=1,
        row_uuid=[train_uuid],
        audio_row=[0.1, 0.2, 0.3, 0.4, 0.5],
        param_row=[0.75, 0.25],
    )
    val_version = _source(
        source / "val.lance",
        rows=1,
        row_uuid=[val_uuid],
        audio_row=[-0.1, -0.2, -0.3, -0.4, -0.5],
        param_row=[0.25, 0.75],
    )
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, train_version).model_copy(
        update={
            "splits": ("train", "val"),
            "source_versions": {"train": train_version, "val": val_version},
        }
    )

    export_slap(config)

    train = lance.dataset(output / "train.lance").to_table()
    val = lance.dataset(output / "val.lance").to_table()
    assert train["row_uuid"].to_pylist() == [train_uuid, train_uuid]
    assert val["row_uuid"].to_pylist() == [val_uuid, val_uuid]
    assert train["slap"].to_pylist() != val["slap"].to_pylist()


def test_export_slap_reuse_rejects_output_advanced_past_completion(tmp_path: Path) -> None:
    """A completed output advanced by another writer is not silently reused.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance")
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, version)
    export_slap(config)
    lance.dataset(output / "train.lance").update_schema_metadata({"foreign": "writer"})

    with pytest.raises(ValueError, match="completion version does not match"):
        export_slap(config)

    assert lance.dataset(output / "train.lance").schema.metadata[b"foreign"] == b"writer"


def test_export_slap_reuse_rejects_tampered_output_schema(tmp_path: Path) -> None:
    """A matching output with an added column fails closed on reuse.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance")
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, version)
    export_slap(config)
    output_path = output / "train.lance"
    lance.dataset(output_path).add_columns({"foreign": "1"})
    _align_output_metadata_version(output_path)

    with pytest.raises(ValueError, match="schema differs"):
        export_slap(config)

    assert "foreign" in lance.dataset(output_path).schema.names


def test_export_slap_reuse_rejects_missing_persisted_rows(tmp_path: Path) -> None:
    """A matching output with deleted retrieval rows fails closed on reuse.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance")
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(source, output, checkpoint, version)
    export_slap(config)
    output_path = output / "train.lance"
    lance.dataset(output_path).delete("is_param_embedding = true")
    _align_output_metadata_version(output_path)

    with pytest.raises(ValueError, match="row count differs"):
        export_slap(config)

    assert lance.dataset(output_path).count_rows() == 2


def test_export_slap_hydra_boundary_runs_real_export(tmp_path: Path) -> None:
    """The Hydra boundary validates and runs the real exporter path.

    :param tmp_path: Isolated dataset root.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    version = _source(source / "train.lance", rows=1)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    values = _config(source, output, checkpoint, version).model_dump(mode="json")
    values["synth"] = {"name": "test"}

    _hydra_main.__wrapped__(OmegaConf.create(values))

    assert lance.dataset(output / "train.lance").count_rows() == 2


def test_export_slap_hydra_boundary_reports_invalid_request(tmp_path: Path) -> None:
    """The Hydra boundary translates validation failure to process status one.

    :param tmp_path: Isolated filesystem root.
    """
    values = _valid_values(tmp_path)
    values["splits"] = []
    values["source_versions"] = {}

    with pytest.raises(SystemExit) as raised:
        _hydra_main.__wrapped__(OmegaConf.create(values))

    assert raised.value.code == 1
