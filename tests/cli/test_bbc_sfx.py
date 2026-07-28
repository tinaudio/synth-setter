"""Behavior tests for the BBC Sound Effects importer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import soundfile as sf
from click.testing import CliRunner
from pydantic import ValidationError

from synth_setter.cli.bbc_sfx import main
from synth_setter.pipeline.data import bbc_sfx as bbc_sfx_module
from synth_setter.pipeline.data.bbc_sfx import (
    IA_METADATA_FILENAME,
    LOCAL_MANIFEST_FILENAME,
    IAInventoryEntry,
    convert_release,
    download_source,
    load_ia_metadata,
    partition_inventory,
    upload_release,
    verify_release,
)

_requires_rclone = pytest.mark.skipif(
    shutil.which("rclone") is None, reason="rclone binary not available on PATH"
)


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 8_000,
    channels: int = 1,
    subtype: str = "PCM_16",
) -> bytes:
    """Write a deterministic short WAV fixture.

    :param path: Destination WAV path.
    :param sample_rate: Sample rate in hertz.
    :param channels: Number of interleaved channels.
    :param subtype: SoundFile WAV subtype.
    :returns: Exact encoded WAV bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.25, 0.25, 16 * channels, dtype=np.float32).reshape(16, channels)
    sf.write(path, samples, sample_rate, subtype=subtype)
    return path.read_bytes()


def _metadata(files: list[tuple[str, bytes]], **overrides: object) -> dict[str, object]:
    records = [
        {
            "name": name,
            "size": str(len(payload)),
            "md5": hashlib.md5(payload).hexdigest(),  # noqa: S324 -- IA inventory contract.
            "sha1": hashlib.sha1(payload).hexdigest(),  # noqa: S324 -- IA inventory contract.
            "crc32": "1234abcd",
            "mtime": "1700000000",
            "length": "0.002",
        }
        for name, payload in files
    ]
    result: dict[str, object] = {
        "metadata": {"identifier": "BBCSoundEffectsComplete"},
        "files": records,
    }
    result.update(overrides)
    return result


def _source_fixture(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """Create mixed-format WAV sources and matching authoritative metadata.

    :param tmp_path: Per-test scratch directory.
    :returns: Source root and item-relative encoded WAV payloads.
    """
    source_root = tmp_path / "source"
    item_root = source_root / "BBCSoundEffectsComplete"
    payloads = {
        "sounds/A crowd & birds.wav": _write_wav(
            item_root / "sounds" / "A crowd & birds.wav", sample_rate=8_000, channels=1
        ),
        "sounds/Long path's punctuation (take 2)!.wav": _write_wav(
            item_root / "sounds" / "Long path's punctuation (take 2)!.wav",
            sample_rate=11_025,
            channels=2,
            subtype="FLOAT",
        ),
    }
    (item_root / IA_METADATA_FILENAME).write_text(
        json.dumps(_metadata(list(payloads.items()))), encoding="utf-8"
    )
    return source_root, payloads


def test_load_ia_metadata_valid_inventory_preserves_authoritative_fields(tmp_path: Path) -> None:
    """A valid snapshot preserves every authoritative inventory field.

    :param tmp_path: Per-test scratch directory.
    """
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(_metadata([("sounds/A crowd & birds.wav", b"wave")])), encoding="utf-8"
    )

    snapshot = load_ia_metadata(metadata_path)

    assert snapshot.files[0].path == "sounds/A crowd & birds.wav"
    assert snapshot.files[0].size == 4
    assert snapshot.files[0].md5 == "b2d7d7656eb4e5153688637c8fbf7b49"
    assert snapshot.files[0].sha1 == "d7ce74466d54133cbc5e0e85ef3be8e8d840790a"
    assert snapshot.files[0].crc32 == "1234abcd"
    assert snapshot.files[0].mtime == "1700000000"
    assert snapshot.files[0].length == "0.002"


@pytest.mark.parametrize(
    "record_update",
    [
        {"name": "/sounds/absolute.wav"},
        {"name": "sounds/../escape.wav"},
        {"name": "other/file.wav"},
        {"name": "sounds/not-wave.mp3"},
        {"size": "0"},
        {"size": "not-an-int"},
        {"md5": "missing"},
        {"md5": None},
    ],
)
def test_load_ia_metadata_invalid_inventory_rejected(
    tmp_path: Path, record_update: dict[str, object]
) -> None:
    """Malformed inventory fields fail strict metadata validation.

    :param tmp_path: Per-test scratch directory.
    :param record_update: Invalid field replacement under test.
    """
    document = _metadata([("sounds/safe.wav", b"wave")])
    files = document["files"]
    assert isinstance(files, list)
    files[0].update(record_update)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises((ValidationError, ValueError)):
        load_ia_metadata(metadata_path)


def test_load_ia_metadata_missing_identifier_rejected(tmp_path: Path) -> None:
    """Metadata without an item identifier is rejected.

    :param tmp_path: Per-test scratch directory.
    """
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"files": []}), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_ia_metadata(metadata_path)


def test_load_ia_metadata_wrong_identifier_rejected(tmp_path: Path) -> None:
    """Metadata for another Internet Archive item is rejected.

    :param tmp_path: Per-test scratch directory.
    """
    document = _metadata([("sounds/safe.wav", b"wave")])
    document["metadata"] = {"identifier": "another-item"}
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValidationError, match="identifier"):
        load_ia_metadata(metadata_path)


def test_load_ia_metadata_duplicate_wav_path_rejected(tmp_path: Path) -> None:
    """Duplicate authoritative WAV paths are rejected.

    :param tmp_path: Per-test scratch directory.
    """
    document = _metadata([("sounds/same.wav", b"wave"), ("sounds/same.wav", b"wave")])
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_ia_metadata(metadata_path)


def test_partition_inventory_nonpositive_target_rejected(tmp_path: Path) -> None:
    """A nonpositive shard target is rejected.

    :param tmp_path: Per-test scratch directory.
    """
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(_metadata([("sounds/safe.wav", b"wave")])), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="positive"):
        partition_inventory(load_ia_metadata(metadata_path).files, target_bytes=0)


def test_partition_inventory_unsorted_input_uses_path_order_and_source_bytes(
    tmp_path: Path,
) -> None:
    """Partitioning sorts paths and keeps each source file whole.

    :param tmp_path: Per-test scratch directory.
    """
    document = _metadata(
        [("sounds/c.wav", b"1234567"), ("sounds/a.wav", b"1234"), ("sounds/b.wav", b"123456")]
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    inventory = load_ia_metadata(metadata_path).files

    shards = partition_inventory(inventory, target_bytes=10)

    assert [[entry.path for entry in shard] for shard in shards] == [
        ["sounds/a.wav", "sounds/b.wav"],
        ["sounds/c.wav"],
    ]


@pytest.mark.parametrize("failure", ["missing", "extra", "corrupt"])
def test_convert_release_invalid_source_inventory_refused(tmp_path: Path, failure: str) -> None:
    """Conversion refuses missing, extra, and corrupt source files.

    :param tmp_path: Per-test scratch directory.
    :param failure: Source-inventory violation under test.
    """
    source_root, payloads = _source_fixture(tmp_path)
    item_root = source_root / "BBCSoundEffectsComplete"
    if failure == "missing":
        (item_root / next(iter(payloads))).unlink()
    elif failure == "extra":
        _write_wav(item_root / "sounds" / "unexpected.wav")
    else:
        (item_root / next(iter(payloads))).write_bytes(b"corrupt")

    with pytest.raises(ValueError, match=failure):
        convert_release(source_root, tmp_path / "release", shard_target_bytes=1_000_000)


def test_convert_release_source_changed_after_preflight_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shard ingestion revalidates bytes after the collection-wide preflight.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Mutation injected between preflight and shard ingestion.
    """
    source_root, payloads = _source_fixture(tmp_path)
    original_validate = bbc_sfx_module._validate_source_inventory

    def mutate_after_validation(item_root: Path, entries: Iterable[IAInventoryEntry]) -> None:
        original_validate(item_root, entries)
        changed_path = item_root / next(iter(payloads))
        changed_path.write_bytes(b"x" * changed_path.stat().st_size)

    monkeypatch.setattr(bbc_sfx_module, "_validate_source_inventory", mutate_after_validation)

    with pytest.raises(OSError, match="corrupt MD5"):
        convert_release(source_root, tmp_path / "release", shard_target_bytes=1_000_000)


def test_convert_release_mixed_wavs_writes_typed_blob_v2_rows(tmp_path: Path) -> None:
    """Conversion writes typed metadata and exact Blob v2 WAV bytes.

    :param tmp_path: Per-test scratch directory.
    """
    source_root, payloads = _source_fixture(tmp_path)
    release_root = tmp_path / "release"

    manifest = convert_release(source_root, release_root, shard_target_bytes=1)

    assert manifest.total_rows == 2
    assert manifest.total_source_bytes == sum(map(len, payloads.values()))
    assert len(manifest.shards) == 2
    first = lance.dataset(release_root / manifest.shards[0].dataset_path)
    second = lance.dataset(release_root / manifest.shards[1].dataset_path)
    expected_schema = pa.schema(
        [
            pa.field("path", pa.string(), nullable=False),
            pa.field("size", pa.uint64(), nullable=False),
            pa.field("ia_md5", pa.string(), nullable=False),
            pa.field("ia_sha1", pa.string()),
            pa.field("ia_crc32", pa.string()),
            pa.field("ia_mtime", pa.string()),
            pa.field("ia_length", pa.string()),
            pa.field("sample_rate", pa.uint32(), nullable=False),
            pa.field("channels", pa.uint16(), nullable=False),
            pa.field("frames", pa.uint64(), nullable=False),
            pa.field("format", pa.string(), nullable=False),
            pa.field("subtype", pa.string(), nullable=False),
            pa.field("endian", pa.string(), nullable=False),
            pa.field("duration_seconds", pa.float64(), nullable=False),
            lance.blob_field("audio", nullable=False),
        ]
    )
    assert first.schema.equals(expected_schema, check_metadata=True)
    assert second.schema.equals(expected_schema, check_metadata=True)
    assert first.schema.field("audio").type.extension_name == "lance.blob.v2"
    first_row = first.to_table(
        columns=[
            "path",
            "size",
            "ia_md5",
            "ia_sha1",
            "ia_crc32",
            "ia_mtime",
            "ia_length",
            "sample_rate",
            "channels",
            "frames",
            "format",
            "subtype",
            "endian",
            "duration_seconds",
        ]
    ).to_pylist()[0]
    second_row = second.to_table(
        columns=[
            "path",
            "sample_rate",
            "channels",
            "frames",
            "format",
            "subtype",
            "endian",
            "duration_seconds",
        ]
    ).to_pylist()[0]
    first_payload = payloads["sounds/A crowd & birds.wav"]
    assert first_row == {
        "path": "sounds/A crowd & birds.wav",
        "size": len(first_payload),
        "ia_md5": hashlib.md5(first_payload).hexdigest(),  # noqa: S324 -- IA inventory contract.
        "ia_sha1": hashlib.sha1(first_payload).hexdigest(),  # noqa: S324 -- IA inventory contract.
        "ia_crc32": "1234abcd",
        "ia_mtime": "1700000000",
        "ia_length": "0.002",
        "sample_rate": 8_000,
        "channels": 1,
        "frames": 16,
        "format": "WAV",
        "subtype": "PCM_16",
        "endian": "FILE",
        "duration_seconds": 0.002,
    }
    assert second_row["path"] == "sounds/Long path's punctuation (take 2)!.wav"
    assert second_row["sample_rate"] == 11_025
    assert second_row["channels"] == 2
    assert second_row["frames"] == 16
    assert second_row["subtype"] == "FLOAT"
    assert second_row["duration_seconds"] == pytest.approx(16 / 11_025)
    assert first.take_blobs("audio", indices=[0])[0].read() == payloads[first_row["path"]]
    assert second.take_blobs("audio", indices=[0])[0].read() == payloads[second_row["path"]]


def test_convert_release_existing_final_output_refused(tmp_path: Path) -> None:
    """Conversion never overwrites a completed release.

    :param tmp_path: Per-test scratch directory.
    """
    source_root, _ = _source_fixture(tmp_path)
    (tmp_path / "release").mkdir()

    with pytest.raises(FileExistsError):
        convert_release(source_root, tmp_path / "release", shard_target_bytes=1_000_000)


def test_convert_release_corrupt_partial_metadata_rejected(tmp_path: Path) -> None:
    """Resume fails closed when pinned partial metadata differs from the source.

    :param tmp_path: Per-test scratch directory.
    """
    source_root, _ = _source_fixture(tmp_path)
    metadata_root = tmp_path / "release.partial" / "metadata"
    metadata_root.mkdir(parents=True)
    (metadata_root / IA_METADATA_FILENAME).write_bytes(b"different")

    with pytest.raises(ValueError, match="partial release artifact mismatch"):
        convert_release(source_root, tmp_path / "release", shard_target_bytes=1)


def test_convert_release_valid_partial_resumes_completed_shards(tmp_path: Path) -> None:
    """Conversion reuses verified shards from an interrupted partial release.

    :param tmp_path: Per-test scratch directory.
    """
    source_root, _ = _source_fixture(tmp_path)
    completed_root = tmp_path / "completed"
    completed = convert_release(source_root, completed_root, shard_target_bytes=1)
    partial_root = tmp_path / "release.partial"
    completed_root.rename(partial_root)
    (partial_root / LOCAL_MANIFEST_FILENAME).unlink()
    shutil.rmtree(partial_root / completed.shards[1].dataset_path)

    resumed = convert_release(source_root, tmp_path / "release", shard_target_bytes=1)

    assert resumed.total_rows == completed.total_rows
    assert resumed.total_source_bytes == completed.total_source_bytes
    assert resumed.shards == completed.shards
    first_shard_prefix = f"{completed.shards[0].dataset_path}/"
    assert [
        item for item in resumed.release_files if item.path.startswith(first_shard_prefix)
    ] == [item for item in completed.release_files if item.path.startswith(first_shard_prefix)]
    assert not partial_root.exists()


def test_download_source_runs_literal_ia_argv_and_snapshots_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download passes literal argv and atomically saves metadata stdout.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Isolated executable path and command environment.
    """
    source_root = tmp_path / "source root"
    source_root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "argv.jsonl"
    document = json.dumps(_metadata([("sounds/a & b.wav", b"wave")]))
    script = bin_dir / "ia"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "with pathlib.Path(os.environ['IA_ARGV_LOG']).open('a') as f:\n"
        "    f.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')\n"
        "if sys.argv[1] == 'metadata':\n"
        "    print(os.environ['IA_METADATA'])\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("IA_ARGV_LOG", str(log_path))
    monkeypatch.setenv("IA_METADATA", document)

    snapshot = download_source(source_root)

    calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert calls == [
        {
            "argv": [
                "download",
                "BBCSoundEffectsComplete",
                "--glob=*.wav",
                "--retries",
                "10",
                "--checksum",
            ],
            "cwd": str(source_root),
        },
        {"argv": ["metadata", "BBCSoundEffectsComplete"], "cwd": str(source_root)},
    ]
    assert snapshot.files[0].path == "sounds/a & b.wav"
    assert (source_root / "BBCSoundEffectsComplete" / IA_METADATA_FILENAME).is_file()
    assert not list((source_root / "BBCSoundEffectsComplete").glob("*.partial"))


def test_download_source_malformed_metadata_does_not_poison_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed successful response is rejected before snapshot publication.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Isolated executable path.
    """
    source_root = tmp_path / "source"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ia"
    script.write_text(
        "#!/bin/sh\nif [ \"$1\" = metadata ]; then printf '{}\\n'; fi\nexit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(ValidationError):
        download_source(source_root)

    assert not (source_root / "BBCSoundEffectsComplete" / IA_METADATA_FILENAME).exists()


def test_download_source_metadata_transient_failures_are_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata fetch retries transient command failures within a wall-clock bound.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Isolated executable path and command environment.
    """
    source_root = tmp_path / "source"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    attempts_path = tmp_path / "attempts"
    document = json.dumps(_metadata([("sounds/a.wav", b"wave")]))
    script = bin_dir / "ia"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1] == 'metadata':\n"
        "    path = pathlib.Path(os.environ['IA_ATTEMPTS'])\n"
        "    attempts = int(path.read_text()) + 1 if path.exists() else 1\n"
        "    path.write_text(str(attempts))\n"
        "    if attempts < 3:\n"
        "        raise SystemExit(1)\n"
        "    print(os.environ['IA_METADATA'])\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("IA_ATTEMPTS", str(attempts_path))
    monkeypatch.setenv("IA_METADATA", document)

    snapshot = download_source(source_root)

    assert attempts_path.read_text() == "3"
    assert snapshot.files[0].path == "sounds/a.wav"


def test_download_source_existing_authoritative_snapshot_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing authoritative snapshot prevents a metadata refetch.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Isolated executable path.
    """
    source_root = tmp_path / "source"
    item_root = source_root / "BBCSoundEffectsComplete"
    item_root.mkdir(parents=True)
    (item_root / IA_METADATA_FILENAME).write_text(
        json.dumps(_metadata([("sounds/a.wav", b"wave")])), encoding="utf-8"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "argv.txt"
    script = bin_dir / "ia"
    script.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log_path!s}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    download_source(source_root)

    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "download BBCSoundEffectsComplete --glob=*.wav --retries 10 --checksum"
    ]


def test_upload_release_reserved_completion_manifest_in_inventory_rejected(
    tmp_path: Path,
) -> None:
    """A forged release inventory cannot publish completion during upload.

    :param tmp_path: Per-test scratch directory.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    convert_release(source_root, release_root, shard_target_bytes=1_000_000)
    manifest_path = release_root / LOCAL_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text())
    payload["release_files"].append(
        {
            "path": "manifest.json",
            "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
    )
    (release_root / "manifest.json").write_bytes(b"")
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="manifest.json"):
        upload_release(release_root)


@_requires_rclone
def test_upload_release_custom_remote_uri_targets_requested_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit remote URI publishes data below only that prefix.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local rclone remote and working-directory isolation.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    convert_release(source_root, release_root, shard_target_bytes=1_000_000)
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)

    upload_release(release_root, remote_uri="r2://custom-bucket/bbc")

    assert (remote_root / "custom-bucket" / "bbc" / "metadata" / "inventory.jsonl").is_file()
    assert not (remote_root / "experiments").exists()


@_requires_rclone
def test_upload_release_real_rclone_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real rclone upload preserves remote bytes after local mutation.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local rclone remote and working-directory isolation.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    convert_release(source_root, release_root, shard_target_bytes=1_000_000)
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)

    upload_release(release_root)
    uploaded = remote_root / "experiments" / "third_party" / "BBCSoundEffectsComplete"
    original = (uploaded / "metadata" / "inventory.jsonl").read_bytes()
    (release_root / "metadata" / "inventory.jsonl").write_bytes(b"changed")

    with pytest.raises(ValueError, match="release file mismatch"):
        upload_release(release_root)
    assert (uploaded / "metadata" / "inventory.jsonl").read_bytes() == original
    assert not (uploaded / "manifest.json").exists()
    assert not (uploaded / "release-manifest.json").exists()


@_requires_rclone
def test_verify_release_remote_release_manifest_extra_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification rejects a remotely uploaded local manifest candidate.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local rclone remote and working-directory isolation.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    convert_release(source_root, release_root, shard_target_bytes=1_000_000)
    manifest_path = release_root / LOCAL_MANIFEST_FILENAME
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)
    upload_release(release_root)
    remote = remote_root / "experiments" / "third_party" / "BBCSoundEffectsComplete"
    shutil.copy2(manifest_path, remote / LOCAL_MANIFEST_FILENAME)

    with pytest.raises(ValueError, match="extra=.*release-manifest.json"):
        verify_release(source_root, tmp_path / "download", manifest_path)
    assert not (remote / "manifest.json").exists()


@_requires_rclone
def test_verify_release_shard_source_byte_boundary_mismatch_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest byte ranges must match the source rows assigned to each shard.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local rclone remote and working-directory isolation.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    manifest = convert_release(source_root, release_root, shard_target_bytes=1)
    manifest_path = release_root / LOCAL_MANIFEST_FILENAME
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)
    upload_release(release_root)
    payload = manifest.model_dump()
    shifted_boundary = payload["shards"][0]["source_byte_end"] + 1
    payload["shards"][0]["source_byte_end"] = shifted_boundary
    payload["shards"][1]["source_byte_start"] = shifted_boundary
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="shard source byte range"):
        verify_release(source_root, tmp_path / "download", manifest_path)


@_requires_rclone
def test_verify_release_existing_matching_download_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification resumes a partial download whose existing bytes match R2.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local rclone remote and working-directory isolation.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    manifest = convert_release(source_root, release_root, shard_target_bytes=1_000_000)
    manifest_path = release_root / LOCAL_MANIFEST_FILENAME
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)
    upload_release(release_root)
    remote = remote_root / "experiments" / "third_party" / "BBCSoundEffectsComplete"
    download_root = tmp_path / "download"
    partial_file = manifest.release_files[0].path
    (download_root / partial_file).parent.mkdir(parents=True)
    shutil.copy2(remote / partial_file, download_root / partial_file)

    result = verify_release(source_root, download_root, manifest_path)

    assert result.rows == 2
    assert (remote / "manifest.json").is_file()

    second_result = verify_release(source_root, tmp_path / "second-download", manifest_path)

    assert second_result == result
    assert (tmp_path / "second-download" / "metadata" / "inventory.jsonl").is_file()


@_requires_rclone
def test_verify_release_downloaded_ia_snapshot_must_match_authoritative_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forged manifest cannot bless a different remote IA metadata snapshot.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local rclone remote and working-directory isolation.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    manifest = convert_release(source_root, release_root, shard_target_bytes=1_000_000)
    manifest_path = release_root / LOCAL_MANIFEST_FILENAME
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)
    upload_release(release_root)
    remote = remote_root / "experiments" / "third_party" / "BBCSoundEffectsComplete"
    remote_snapshot = remote / "metadata" / IA_METADATA_FILENAME
    remote_snapshot.write_bytes(b"{}")
    payload = manifest.model_dump()
    snapshot_record = next(
        item for item in payload["release_files"] if item["path"] == "metadata/ia-metadata.json"
    )
    snapshot_record["size"] = 2
    snapshot_record["sha256"] = hashlib.sha256(b"{}").hexdigest()
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="downloaded IA metadata snapshot mismatch"):
        verify_release(source_root, tmp_path / "download", manifest_path)


@_requires_rclone
def test_verify_release_publishes_captured_manifest_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest publication uses bytes captured before the verification gate.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local remote plus a mutation injected at publication time.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    convert_release(source_root, release_root, shard_target_bytes=1_000_000)
    manifest_path = release_root / LOCAL_MANIFEST_FILENAME
    expected = manifest_path.read_bytes()
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)
    upload_release(release_root)
    original_upload = bbc_sfx_module.upload_to_uri_immutable

    def mutate_retained_then_upload(local_path: Path, remote_uri: str) -> None:
        manifest_path.write_bytes(b"changed after verification")
        original_upload(local_path, remote_uri)

    monkeypatch.setattr(bbc_sfx_module, "upload_to_uri_immutable", mutate_retained_then_upload)

    verify_release(source_root, tmp_path / "download", manifest_path)

    remote_manifest = (
        remote_root / "experiments" / "third_party" / "BBCSoundEffectsComplete" / "manifest.json"
    )
    assert remote_manifest.read_bytes() == expected


@_requires_rclone
def test_verify_release_progress_checkpoint_skips_verified_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching durable checkpoint resumes after its verified shard.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local remote plus shard-call observation.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    manifest = convert_release(source_root, release_root, shard_target_bytes=1)
    manifest_path = release_root / LOCAL_MANIFEST_FILENAME
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)
    upload_release(release_root)
    remote = remote_root / "experiments" / "third_party" / "BBCSoundEffectsComplete"
    download_root = tmp_path / "download"
    shutil.copytree(remote, download_root)
    progress_path = download_root / ".bbc-sfx-verify-progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "verified_shards": [manifest.shards[0].dataset_path],
            }
        )
    )
    original_verify = bbc_sfx_module._verify_shard

    def reject_rechecking_first_shard(
        dataset_path: Path,
        item_root: Path,
        *,
        entries: tuple[IAInventoryEntry, ...],
        batch_size: int,
    ) -> None:
        if dataset_path == download_root / manifest.shards[0].dataset_path:
            pytest.fail("verified shard was rechecked")
        original_verify(dataset_path, item_root, entries=entries, batch_size=batch_size)

    monkeypatch.setattr(bbc_sfx_module, "_verify_shard", reject_rechecking_first_shard)

    result = verify_release(source_root, download_root, manifest_path)

    assert result.rows == 2
    assert not progress_path.exists()


@_requires_rclone
@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    [
        ("missing-row", "missing Lance rows"),
        ("typed-metadata", "typed metadata mismatch"),
        ("blob", "blob byte mismatch"),
    ],
)
def test_verify_release_self_consistent_forged_shard_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    expected_error: str,
) -> None:
    """Full verification rejects forged shard content covered by its manifest hash.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local rclone remote and working-directory isolation.
    :param corruption: Shard invariant to forge.
    :param expected_error: Verification failure unique to that forgery.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    manifest = convert_release(source_root, release_root, shard_target_bytes=1_000_000)
    shard_path = release_root / manifest.shards[0].dataset_path
    dataset = lance.dataset(shard_path)
    metadata = dataset.to_table(columns=bbc_sfx_module._METADATA_COLUMNS)
    arrays = [
        metadata.column(field.name).combine_chunks() for field in bbc_sfx_module._METADATA_FIELDS
    ]
    blob_payloads: list[bytes] = []
    for blob in dataset.take_blobs("audio", indices=range(dataset.count_rows())):
        with blob:
            blob_payloads.append(blob.read())

    if corruption == "missing-row":
        arrays = [array.slice(0, 1) for array in arrays]
        blob_payloads = blob_payloads[:1]
    elif corruption == "typed-metadata":
        size_index = bbc_sfx_module._METADATA_COLUMNS.index("size")
        arrays[size_index] = pa.array(
            [value.as_py() + 1 for value in arrays[size_index]], type=pa.uint64()
        )
    else:
        blob_payloads[0] = b"forged blob"

    forged_batch = pa.record_batch(
        [*arrays, lance.blob_array(blob_payloads)], schema=bbc_sfx_module.LANCE_SCHEMA
    )
    shutil.rmtree(shard_path)
    lance.write_dataset(
        iter([forged_batch]),
        shard_path,
        schema=bbc_sfx_module.LANCE_SCHEMA,
        mode="create",
        max_bytes_per_file=bbc_sfx_module.LANCE_MAX_BYTES_PER_FILE,
        max_rows_per_group=1,
        data_storage_version=bbc_sfx_module.LANCE_DATA_STORAGE_VERSION,
    )
    forged_manifest = manifest.model_copy(
        update={"release_files": bbc_sfx_module._release_files(release_root)}
    )
    manifest_path = release_root / LOCAL_MANIFEST_FILENAME
    manifest_path.write_text(forged_manifest.model_dump_json())
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)
    upload_release(release_root)

    with pytest.raises(ValueError, match=expected_error):
        verify_release(source_root, tmp_path / "download", manifest_path)
    assert not (
        remote_root / "experiments" / "third_party" / "BBCSoundEffectsComplete" / "manifest.json"
    ).exists()


@_requires_rclone
def test_verify_release_remote_corruption_reports_mismatch_without_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote corruption aborts verification before completion publication.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local rclone remote and working-directory isolation.
    """
    source_root, _ = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    manifest = convert_release(source_root, release_root, shard_target_bytes=1_000_000)
    manifest_path = release_root / "release-manifest.json"
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)
    upload_release(release_root)
    remote = remote_root / "experiments" / "third_party" / "BBCSoundEffectsComplete"
    release_file = manifest.release_files[0]
    (remote / release_file.path).write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="mismatch"):
        verify_release(source_root, tmp_path / "download", manifest_path)
    assert not (remote / "manifest.json").exists()


def test_click_convert_command_reports_rows_and_bytes(tmp_path: Path) -> None:
    """The convert command reports authoritative row and byte totals.

    :param tmp_path: Per-test scratch directory.
    """
    source_root, payloads = _source_fixture(tmp_path)
    release_root = tmp_path / "release"

    result = CliRunner().invoke(
        main,
        ["convert", str(source_root), str(release_root), "--shard-target-bytes", "1000000"],
    )

    assert result.exit_code == 0, result.output
    assert f"converted 2 rows and {sum(map(len, payloads.values()))} bytes" in result.output


def test_click_download_command_reports_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The download command reports the validated WAV count.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Isolated no-op Internet Archive executable path.
    """
    source_root = tmp_path / "source"
    item_root = source_root / "BBCSoundEffectsComplete"
    item_root.mkdir(parents=True)
    (item_root / IA_METADATA_FILENAME).write_text(
        json.dumps(_metadata([("sounds/a.wav", b"wave")])), encoding="utf-8"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ia"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    result = CliRunner().invoke(main, ["download", str(source_root)])

    assert result.exit_code == 0, result.output
    assert "download inventory: 1 WAV files" in result.output


@_requires_rclone
def test_click_upload_and_verify_commands_publish_completion_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upload withholds completion until verify succeeds through Click.

    :param tmp_path: Per-test scratch directory.
    :param monkeypatch: Local rclone remote and working-directory isolation.
    """
    source_root, payloads = _source_fixture(tmp_path)
    release_root = tmp_path / "release"
    convert_release(source_root, release_root, shard_target_bytes=1_000_000)
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(remote_root)

    uploaded = CliRunner().invoke(main, ["upload", str(release_root)])

    assert uploaded.exit_code == 0, uploaded.output
    assert "immutable release files" in uploaded.output
    remote = remote_root / "experiments" / "third_party" / "BBCSoundEffectsComplete"
    assert not (remote / "manifest.json").exists()

    verified = CliRunner().invoke(
        main,
        [
            "verify",
            str(source_root),
            str(tmp_path / "download"),
            "--release-manifest",
            str(release_root / LOCAL_MANIFEST_FILENAME),
        ],
    )

    assert verified.exit_code == 0, verified.output
    assert (
        f"verified 2 rows and {sum(map(len, payloads.values()))} bytes; 0 mismatches"
        in verified.output
    )
    assert (remote / "manifest.json").is_file()
