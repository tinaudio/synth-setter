"""Stream official NSynth extracts into immutable local Lance datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ijson
import lance
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from synth_setter.pipeline.data.lance_shard import (
    LANCE_DATA_STORAGE_VERSION,
    LANCE_MAX_BYTES_PER_FILE,
)
from synth_setter.pipeline.r2_io import (
    download_dir_no_overwrite,
    upload_dir_immutable,
    upload_file_immutable,
)

SplitName = Literal["train", "valid", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "valid", "test")
OFFICIAL_EXPECTED_COUNTS: dict[str, int] = {
    "train": 289205,
    "valid": 12678,
    "test": 4096,
}
DEFAULT_BATCH_SIZE = 128
DEFAULT_REMOTE_ROOT = "r2://experiments/third_party/NSynth"
MANIFEST_FILENAME = "manifest.json"

_SOURCE_URL = "https://magenta.tensorflow.org/datasets/nsynth"
_LICENSE_NAME = "Creative Commons Attribution 4.0 International"
_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
_SAFE_NOTE_STR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SCHEMA_SOURCE_KEY = b"synth_setter.nsynth.source"
_SCHEMA_PROVENANCE_KEY = b"synth_setter.nsynth.provenance"
_SCHEMA_LICENSE_KEY = b"synth_setter.nsynth.license"
_FILE_SCAN_CHUNK_BYTES = 1024 * 1024
_EXAMPLE_FIELDS = (
    "instrument",
    "instrument_family",
    "instrument_family_str",
    "instrument_source",
    "instrument_source_str",
    "instrument_str",
    "note",
    "note_str",
    "pitch",
    "qualities",
    "qualities_str",
    "sample_rate",
    "velocity",
)
_ROW_FIELDS = (*_EXAMPLE_FIELDS, "wav_size", "wav_sha256")


@dataclass(frozen=True)
class VerificationSummary:
    """Summarize a complete bounded verification scan.

    .. attribute :: split_counts

        Verified row count keyed by official split.

    .. attribute :: total_count

        Verified row count across all splits.

    .. attribute :: mismatches

        Zero when verification returns; mismatches raise immediately.
    """

    split_counts: dict[str, int]
    total_count: int
    mismatches: int = 0


class NSynthExample(BaseModel):
    """Validate one exact official ``examples.json`` metadata record.

    .. attribute :: model_config

        Strict, frozen, extra-forbidding parsing configuration.

    .. attribute :: instrument

        Numeric instrument identifier.

    .. attribute :: instrument_family

        Numeric instrument-family identifier.

    .. attribute :: instrument_family_str

        Instrument-family label.

    .. attribute :: instrument_source

        Numeric instrument-source identifier.

    .. attribute :: instrument_source_str

        Instrument-source label.

    .. attribute :: instrument_str

        Canonical instrument identifier.

    .. attribute :: note

        Numeric note identifier.

    .. attribute :: note_str

        Canonical note identifier and WAV basename.

    .. attribute :: pitch

        MIDI pitch number.

    .. attribute :: qualities

        Ten binary NSynth quality flags.

    .. attribute :: qualities_str

        Active quality labels.

    .. attribute :: sample_rate

        WAV sample rate in hertz.

    .. attribute :: velocity

        MIDI velocity.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    instrument: int
    instrument_family: int
    instrument_family_str: str
    instrument_source: int
    instrument_source_str: str
    instrument_str: str
    note: int
    note_str: str
    pitch: int
    qualities: list[int]
    qualities_str: list[str]
    sample_rate: int
    velocity: int

    @field_validator("note_str")
    @classmethod
    def _validate_note_str(cls, value: str) -> str:
        """Reject identifiers that could escape the split audio directory.

        :param value: Candidate official note identifier.
        :returns: Safe identifier suitable for one WAV basename.
        :raises ValueError: The identifier is empty or contains unsafe characters.
        """
        if _SAFE_NOTE_STR.fullmatch(value) is None:
            raise ValueError("note_str must contain only letters, digits, '_' and '-'")
        return value

    @field_validator("qualities")
    @classmethod
    def _validate_qualities(cls, value: list[int]) -> list[int]:
        """Require the ten binary quality flags defined by NSynth.

        :param value: Candidate quality flags.
        :returns: Exactly ten binary integers.
        :raises ValueError: The vector length differs from ten or a value is not binary.
        """
        if len(value) != 10 or any(flag not in (0, 1) for flag in value):
            raise ValueError("qualities must contain exactly 10 binary integers")
        return value


class NSynthSplitManifest(BaseModel):
    """Describe one imported split and its preserved source sidecar.

    .. attribute :: model_config

        Strict, frozen, extra-forbidding parsing configuration.

    .. attribute :: count

        Exact split row count.

    .. attribute :: dataset_path

        Root-relative Lance dataset path.

    .. attribute :: examples_path

        Root-relative preserved JSON sidecar path.

    .. attribute :: examples_sha256

        Lowercase SHA-256 digest of the sidecar bytes.

    .. attribute :: examples_size

        Sidecar byte size.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    count: int
    dataset_path: str
    examples_path: str
    examples_sha256: str
    examples_size: int


class NSynthManifest(BaseModel):
    """Validate the completion marker written after all import artifacts.

    .. attribute :: model_config

        Strict, frozen, extra-forbidding parsing configuration.

    .. attribute :: format_version

        Import manifest schema version.

    .. attribute :: dataset

        Canonical dataset name.

    .. attribute :: source_url

        Official dataset provenance URL.

    .. attribute :: license_name

        Full source license name.

    .. attribute :: license_url

        Canonical source license URL.

    .. attribute :: remote_root

        Immutable R2 prefix containing this import.

    .. attribute :: total_count

        Aggregate row count across all splits.

    .. attribute :: splits

        Per-split paths, counts, and sidecar digests.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    format_version: Literal[1]
    dataset: Literal["NSynth"]
    source_url: str
    license_name: str
    license_url: str
    remote_root: str
    total_count: int
    splits: dict[SplitName, NSynthSplitManifest]

    @model_validator(mode="after")
    def _validate_split_totals(self) -> NSynthManifest:
        """Require all official splits and an accurate aggregate count.

        :returns: Validated manifest.
        :raises ValueError: Split keys or aggregate count are inconsistent.
        """
        if set(self.splits) != set(SPLITS):
            raise ValueError(f"manifest splits must be exactly {list(SPLITS)}")
        split_total = sum(item.count for item in self.splits.values())
        if self.total_count != split_total:
            raise ValueError(
                f"manifest total_count {self.total_count} does not equal split total {split_total}"
            )
        return self


def _sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory.

    :param path: File to hash.
    :returns: Lowercase SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_FILE_SCAN_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_files(source_path: Path, copy_path: Path, split: SplitName) -> tuple[int, str]:
    """Stream-compare two files while hashing the source.

    :param source_path: Original file.
    :param copy_path: Preserved or downloaded copy.
    :param split: Official split used in mismatch diagnostics.
    :returns: Source byte size and lowercase SHA-256 digest.
    :raises ValueError: The files differ at any byte or in length.
    """
    digest = hashlib.sha256()
    size = 0
    with source_path.open("rb") as source, copy_path.open("rb") as copy:
        while True:
            source_chunk = source.read(_FILE_SCAN_CHUNK_BYTES)
            copy_chunk = copy.read(_FILE_SCAN_CHUNK_BYTES)
            if source_chunk != copy_chunk:
                raise ValueError(f"{split} downloaded JSON sidecar bytes differ from source")
            if not source_chunk:
                break
            digest.update(source_chunk)
            size += len(source_chunk)
    return size, digest.hexdigest()


def _iter_examples(path: Path) -> Iterator[tuple[str, NSynthExample]]:
    """Yield strict records from a top-level JSON object in parser-sized chunks.

    :param path: Official ``examples.json`` path.
    :yields: Top-level key and validated metadata record in source order.
    :ytype: tuple[str, NSynthExample]
    :raises ValueError: The document root is not an object or a key differs from ``note_str``.
    """
    with path.open("rb") as source:
        first_event = next(ijson.parse(source), None)
    if first_event is None or first_event[1] != "start_map":
        raise ValueError(f"{path} must contain one top-level JSON object")

    with path.open("rb") as source:
        for key, payload in ijson.kvitems(source, ""):
            example = NSynthExample.model_validate(payload)
            if key != example.note_str:
                raise ValueError(
                    f"examples.json key {key!r} does not equal note_str {example.note_str!r}"
                )
            yield key, example


def _schema(split: SplitName, examples_sha256: str) -> pa.Schema:
    """Build the typed NSynth schema with source, provenance, and license metadata.

    :param split: Official split represented by the dataset.
    :param examples_sha256: Digest of the preserved source metadata file.
    :returns: Arrow schema using Lance Blob v2 for WAV payloads.
    """
    fields = [
        pa.field("instrument", pa.int64(), nullable=False),
        pa.field("instrument_family", pa.int64(), nullable=False),
        pa.field("instrument_family_str", pa.string(), nullable=False),
        pa.field("instrument_source", pa.int64(), nullable=False),
        pa.field("instrument_source_str", pa.string(), nullable=False),
        pa.field("instrument_str", pa.string(), nullable=False),
        pa.field("note", pa.int64(), nullable=False),
        pa.field("note_str", pa.string(), nullable=False),
        pa.field("pitch", pa.int64(), nullable=False),
        pa.field("qualities", pa.list_(pa.int8(), 10), nullable=False),
        pa.field("qualities_str", pa.list_(pa.string()), nullable=False),
        pa.field("sample_rate", pa.int64(), nullable=False),
        pa.field("velocity", pa.int64(), nullable=False),
        pa.field("wav_size", pa.int64(), nullable=False),
        pa.field("wav_sha256", pa.string(), nullable=False),
        lance.blob_field("audio", nullable=False),
    ]
    metadata = {
        _SCHEMA_SOURCE_KEY: json.dumps(
            {"dataset": "NSynth", "split": split, "url": _SOURCE_URL}, sort_keys=True
        ).encode("utf-8"),
        _SCHEMA_PROVENANCE_KEY: json.dumps(
            {
                "examples_sha256": examples_sha256,
                "source_layout": f"nsynth-{split}/examples.json + audio/<note_str>.wav",
            },
            sort_keys=True,
        ).encode("utf-8"),
        _SCHEMA_LICENSE_KEY: json.dumps(
            {"name": _LICENSE_NAME, "url": _LICENSE_URL}, sort_keys=True
        ).encode("utf-8"),
    }
    return pa.schema(fields, metadata=metadata)


def _record_batch(
    records: list[NSynthExample], wav_payloads: list[bytes], schema: pa.Schema
) -> pa.RecordBatch:
    """Encode one bounded metadata and Blob-v2 batch.

    :param records: Strict source records in row order.
    :param wav_payloads: Corresponding complete WAV files.
    :param schema: Split schema returned by :func:`_schema`.
    :returns: Record batch ready for ``lance.write_dataset``.
    """
    columns: list[pa.Array] = []
    for field in schema:
        if field.name == "audio":
            columns.append(lance.blob_array(wav_payloads))
        elif field.name == "wav_size":
            columns.append(pa.array([len(payload) for payload in wav_payloads], type=field.type))
        elif field.name == "wav_sha256":
            columns.append(
                pa.array(
                    [hashlib.sha256(payload).hexdigest() for payload in wav_payloads],
                    type=field.type,
                )
            )
        else:
            columns.append(
                pa.array([getattr(record, field.name) for record in records], type=field.type)
            )
    return pa.record_batch(columns, schema=schema)


def _split_batches(
    examples_path: Path,
    audio_root: Path,
    schema: pa.Schema,
    *,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Yield validated records and WAV payloads in bounded batches.

    :param examples_path: Official split metadata file.
    :param audio_root: Official split audio directory.
    :param schema: Split schema returned by :func:`_schema`.
    :param batch_size: Maximum rows and WAV payloads held per batch.
    :yields: Streamed record batches in source order.
    :ytype: pa.RecordBatch
    """
    records: list[NSynthExample] = []
    wav_payloads: list[bytes] = []
    for note_str, example in _iter_examples(examples_path):
        wav_path = audio_root / f"{note_str}.wav"
        records.append(example)
        wav_payloads.append(wav_path.read_bytes())
        if len(records) == batch_size:
            yield _record_batch(records, wav_payloads, schema)
            records, wav_payloads = [], []
    if records:
        yield _record_batch(records, wav_payloads, schema)


def _index_unique_note(note_index: sqlite3.Connection, note_str: str) -> None:
    """Record one note identifier in the disk-backed uniqueness index.

    :param note_index: Temporary SQLite uniqueness index.
    :param note_str: Strict note identifier to record.
    :raises ValueError: The identifier already exists in the source JSON.
    """
    try:
        note_index.execute("INSERT INTO notes VALUES (?)", (note_str,))
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"duplicate metadata key {note_str!r}") from exc


def _validate_split_source(
    examples_path: Path,
    audio_root: Path,
    *,
    expected_count: int,
    split: SplitName,
) -> int:
    """Validate uniqueness, WAV coverage, and exact count without in-memory indexes.

    :param examples_path: Official split metadata file.
    :param audio_root: Official split audio directory.
    :param expected_count: Required metadata and WAV count.
    :param split: Official split used in diagnostics.
    :returns: Validated metadata count.
    :raises FileNotFoundError: A metadata row has no corresponding WAV.
    :raises ValueError: A key is duplicated or the metadata/WAV counts disagree.
    """
    count = 0
    with (
        tempfile.TemporaryDirectory(prefix="nsynth-note-index-") as index_dir,
        sqlite3.connect(Path(index_dir) / "notes.sqlite") as note_index,
    ):
        note_index.execute("CREATE TABLE notes (note_str TEXT PRIMARY KEY)")
        for note_str, _example in _iter_examples(examples_path):
            _index_unique_note(note_index, note_str)
            wav_path = audio_root / f"{note_str}.wav"
            if not wav_path.is_file():
                raise FileNotFoundError(f"missing WAV for {note_str!r}: {wav_path}")
            count += 1

    wav_count = sum(1 for path in audio_root.rglob("*.wav") if path.is_file())
    if wav_count != count:
        raise ValueError(f"{split} contains an orphan WAV: {count} examples and {wav_count} WAVs")
    if count != expected_count:
        raise ValueError(f"{split} expected {expected_count} examples and WAVs, found {count}")
    return count


def _write_split(
    source_root: Path,
    output_root: Path,
    split: SplitName,
    *,
    expected_count: int,
    batch_size: int,
) -> NSynthSplitManifest:
    """Write one validated official split into a local Lance dataset.

    :param source_root: Parent of the three extracted source directories.
    :param output_root: Partial import root receiving the split artifacts.
    :param split: Official split to write.
    :param expected_count: Required metadata and WAV count.
    :param batch_size: Maximum records and WAV payloads per Arrow batch.
    :returns: Manifest entry for the completed split.
    :raises FileNotFoundError: The official split metadata or audio directory is absent.
    :raises ValueError: Source counts differ from the expected split count.
    """
    split_root = source_root / f"nsynth-{split}"
    examples_path = split_root / "examples.json"
    audio_root = split_root / "audio"
    if not examples_path.is_file() or not audio_root.is_dir():
        raise FileNotFoundError(
            f"expected {examples_path} and audio directory {audio_root} for split {split}"
        )

    source_count = _validate_split_source(
        examples_path,
        audio_root,
        expected_count=expected_count,
        split=split,
    )
    examples_size = examples_path.stat().st_size
    examples_sha256 = _sha256_file(examples_path)
    schema = _schema(split, examples_sha256)
    dataset_path = output_root / f"{split}.lance"
    lance.write_dataset(
        _split_batches(examples_path, audio_root, schema, batch_size=batch_size),
        dataset_path,
        schema=schema,
        mode="create",
        max_bytes_per_file=LANCE_MAX_BYTES_PER_FILE,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )
    dataset_count = lance.dataset(dataset_path).count_rows()
    if dataset_count != source_count:
        raise ValueError(
            f"{split} Lance dataset contains {dataset_count} rows after writing {source_count}"
        )

    examples_relative = f"metadata/nsynth-{split}.examples.json"
    sidecar = output_root / examples_relative
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(examples_path, sidecar)
    return NSynthSplitManifest(
        count=dataset_count,
        dataset_path=f"{split}.lance",
        examples_path=examples_relative,
        examples_sha256=examples_sha256,
        examples_size=examples_size,
    )


def _write_manifest(path: Path, manifest: NSynthManifest) -> None:
    """Write the strict completion marker as the final local artifact.

    :param path: Manifest destination under the partial import root.
    :param manifest: Validated completion payload.
    """
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def build_local_import(
    source_root: Path,
    output_root: Path,
    *,
    expected_counts: Mapping[str, int] = OFFICIAL_EXPECTED_COUNTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> NSynthManifest:
    """Build all three splits under a partial root and atomically publish it.

    :param source_root: Parent of the official extracted split directories.
    :param output_root: New local directory to publish; existing paths are refused.
    :param expected_counts: Exact required row count for each official split.
    :param batch_size: Maximum records and WAV payloads held in memory per batch.
    :param remote_root: Immutable remote prefix recorded in the manifest.
    :returns: Strict manifest written last inside ``output_root``.
    :raises FileExistsError: ``output_root`` already exists.
    :raises ValueError: Counts are incomplete, non-positive, or ``batch_size`` is invalid.
    """
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root {output_root}")
    if set(expected_counts) != set(SPLITS):
        raise ValueError(f"expected_counts keys must be exactly {list(SPLITS)}")
    if any(count < 1 for count in expected_counts.values()):
        raise ValueError("expected_counts values must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not remote_root.startswith("r2://"):
        raise ValueError(f"remote_root must be an r2:// URI, got {remote_root!r}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.partial-", dir=output_root.parent)
    )
    try:
        splits: dict[SplitName, NSynthSplitManifest] = {
            split: _write_split(
                source_root,
                partial,
                split,
                expected_count=expected_counts[split],
                batch_size=batch_size,
            )
            for split in SPLITS
        }
        manifest = NSynthManifest(
            format_version=1,
            dataset="NSynth",
            source_url=_SOURCE_URL,
            license_name=_LICENSE_NAME,
            license_url=_LICENSE_URL,
            remote_root=remote_root.rstrip("/"),
            total_count=sum(item.count for item in splits.values()),
            splits=splits,
        )
        _write_manifest(partial / MANIFEST_FILENAME, manifest)
        os.replace(partial, output_root)
    finally:
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
    return manifest


def ingest_nsynth(
    source_root: Path,
    output_root: Path,
    *,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    expected_counts: Mapping[str, int] = OFFICIAL_EXPECTED_COUNTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> NSynthManifest:
    """Build the local import, then upload artifacts before the manifest marker.

    :param source_root: Parent of the official extracted split directories.
    :param output_root: New local directory to publish; existing paths are refused.
    :param remote_root: Immutable R2 prefix receiving the import.
    :param expected_counts: Exact required row count for each official split.
    :param batch_size: Maximum records and WAV payloads held per batch.
    :returns: Strict manifest uploaded last.
    """
    manifest = build_local_import(
        source_root,
        output_root,
        expected_counts=expected_counts,
        batch_size=batch_size,
        remote_root=remote_root,
    )
    remote_root = remote_root.rstrip("/")
    upload_dir_immutable(output_root, remote_root, exclude=MANIFEST_FILENAME)
    upload_file_immutable(
        output_root / MANIFEST_FILENAME,
        f"{remote_root}/{MANIFEST_FILENAME}",
    )
    return manifest


def _read_manifest(import_root: Path) -> NSynthManifest:
    """Parse the downloaded completion marker through its strict schema.

    :param import_root: Local import root containing ``manifest.json``.
    :returns: Strict manifest payload.
    """
    return NSynthManifest.model_validate_json((import_root / MANIFEST_FILENAME).read_bytes())


def _verify_manifest_contract(
    manifest: NSynthManifest, expected_counts: Mapping[str, int]
) -> None:
    """Check immutable manifest values and requested split counts.

    :param manifest: Strict downloaded manifest.
    :param expected_counts: Counts required by the verification invocation.
    :raises ValueError: Provenance, license, paths, or counts differ.
    """
    if manifest.source_url != _SOURCE_URL:
        raise ValueError(f"manifest source_url mismatch: {manifest.source_url!r}")
    if manifest.license_name != _LICENSE_NAME or manifest.license_url != _LICENSE_URL:
        raise ValueError("manifest license metadata mismatch")
    if not manifest.remote_root.startswith("r2://"):
        raise ValueError(f"manifest remote_root is not an r2:// URI: {manifest.remote_root!r}")
    for split in SPLITS:
        entry = manifest.splits[split]
        if entry.count != expected_counts[split]:
            raise ValueError(
                f"manifest {split} count {entry.count} differs from expected "
                f"{expected_counts[split]}"
            )
        if entry.dataset_path != f"{split}.lance":
            raise ValueError(f"manifest {split} dataset_path mismatch: {entry.dataset_path!r}")
        expected_sidecar = f"metadata/nsynth-{split}.examples.json"
        if entry.examples_path != expected_sidecar:
            raise ValueError(f"manifest {split} examples_path mismatch: {entry.examples_path!r}")


def _verify_row(
    split: SplitName,
    row_index: int,
    batch: pa.RecordBatch,
    *,
    batch_index: int,
    example: NSynthExample,
    source_wav: bytes,
    stored_wav: bytes,
) -> None:
    """Compare one strict source record and WAV with one downloaded Lance row.

    :param split: Official split used in diagnostics.
    :param row_index: Zero-based split row position.
    :param batch: Projected Lance metadata batch.
    :param batch_index: Row position inside ``batch``.
    :param example: Strict source metadata record.
    :param source_wav: Original source WAV bytes.
    :param stored_wav: Downloaded Lance Blob-v2 bytes.
    :raises ValueError: Any metadata, size, digest, or WAV byte differs.
    """
    for field_name in _EXAMPLE_FIELDS:
        source_value = getattr(example, field_name)
        stored_value = batch.column(field_name)[batch_index].as_py()
        if stored_value != source_value:
            raise ValueError(
                f"{split} row {row_index} metadata mismatch for {field_name}: "
                f"stored {stored_value!r}, source {source_value!r}"
            )
    source_digest = hashlib.sha256(source_wav).hexdigest()
    stored_size = batch.column("wav_size")[batch_index].as_py()
    stored_digest = batch.column("wav_sha256")[batch_index].as_py()
    if stored_size != len(source_wav):
        raise ValueError(
            f"{split} row {row_index} WAV size mismatch: stored {stored_size}, "
            f"source {len(source_wav)}"
        )
    if stored_digest != source_digest:
        raise ValueError(f"{split} row {row_index} WAV SHA-256 mismatch")
    if stored_wav != source_wav:
        raise ValueError(f"{split} row {row_index} WAV blob bytes mismatch")


def _verify_split(
    source_root: Path,
    import_root: Path,
    split: SplitName,
    *,
    entry: NSynthSplitManifest,
    expected_count: int,
    batch_size: int,
) -> int:
    """Stream-compare one downloaded Lance split with its original extract.

    :param source_root: Parent of the official extracted split directories.
    :param import_root: Downloaded import root.
    :param split: Official split to verify.
    :param entry: Strict manifest entry for the split.
    :param expected_count: Required row and WAV count.
    :param batch_size: Maximum Lance rows and blob handles processed together.
    :returns: Verified row count.
    :raises ValueError: Any manifest, schema, metadata, or WAV comparison fails.
    """
    split_root = source_root / f"nsynth-{split}"
    examples_path = split_root / "examples.json"
    audio_root = split_root / "audio"
    source_count = _validate_split_source(
        examples_path,
        audio_root,
        expected_count=expected_count,
        split=split,
    )

    sidecar_path = import_root / entry.examples_path
    source_size, source_json_digest = _compare_files(examples_path, sidecar_path, split)
    if entry.examples_size != source_size:
        raise ValueError(f"{split} manifest examples_size mismatch")
    if entry.examples_sha256 != source_json_digest:
        raise ValueError(f"{split} manifest examples_sha256 mismatch")

    dataset_path = import_root / entry.dataset_path
    dataset = lance.dataset(dataset_path)
    expected_schema = _schema(split, source_json_digest)
    if not dataset.schema.equals(expected_schema, check_metadata=True):
        raise ValueError(f"{split} Lance schema or schema metadata mismatch")
    if dataset.data_storage_version != LANCE_DATA_STORAGE_VERSION:
        raise ValueError(
            f"{split} Lance data storage version {dataset.data_storage_version!r} "
            f"differs from {LANCE_DATA_STORAGE_VERSION!r}"
        )
    if dataset.count_rows() != source_count:
        raise ValueError(
            f"{split} Lance row count {dataset.count_rows()} differs from source {source_count}"
        )

    source_records = _iter_examples(examples_path)
    row_index = 0
    scanner = dataset.scanner(
        columns=list(_ROW_FIELDS), batch_size=batch_size, scan_in_order=True
    )
    for batch in scanner.to_batches():
        indices = list(range(row_index, row_index + batch.num_rows))
        blob_files = dataset.take_blobs("audio", indices=indices)
        for batch_index, blob_file in enumerate(blob_files):
            try:
                _key, example = next(source_records)
            except StopIteration as exc:
                raise ValueError(f"{split} Lance contains extra row {row_index}") from exc
            source_wav = (audio_root / f"{example.note_str}.wav").read_bytes()
            with blob_file:
                stored_wav = blob_file.read()
            _verify_row(
                split,
                row_index,
                batch,
                batch_index=batch_index,
                example=example,
                source_wav=source_wav,
                stored_wav=stored_wav,
            )
            row_index += 1
    sentinel = object()
    if next(source_records, sentinel) is not sentinel:
        raise ValueError(f"{split} Lance is missing source rows after row {row_index}")
    return row_index


def verify_local_import(
    source_root: Path,
    import_root: Path,
    *,
    expected_counts: Mapping[str, int] = OFFICIAL_EXPECTED_COUNTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> VerificationSummary:
    """Prove a downloaded NSynth import matches the original source byte-for-byte.

    :param source_root: Parent of the official extracted split directories.
    :param import_root: Local root containing a downloaded import and manifest.
    :param expected_counts: Exact required row count for each official split.
    :param batch_size: Maximum metadata rows and blob handles scanned together.
    :returns: Zero-mismatch counts after every split passes.
    :raises ValueError: Counts are incomplete, ``batch_size`` is invalid, or any mismatch exists.
    """
    if set(expected_counts) != set(SPLITS):
        raise ValueError(f"expected_counts keys must be exactly {list(SPLITS)}")
    if any(count < 1 for count in expected_counts.values()):
        raise ValueError("expected_counts values must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    manifest = _read_manifest(import_root)
    _verify_manifest_contract(manifest, expected_counts)
    split_counts: dict[str, int] = {
        split: _verify_split(
            source_root,
            import_root,
            split,
            entry=manifest.splits[split],
            expected_count=expected_counts[split],
            batch_size=batch_size,
        )
        for split in SPLITS
    }
    return VerificationSummary(
        split_counts=split_counts,
        total_count=sum(split_counts.values()),
    )


def download_and_verify_nsynth(
    source_root: Path,
    download_root: Path,
    *,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    expected_counts: Mapping[str, int] = OFFICIAL_EXPECTED_COUNTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> VerificationSummary:
    """Download the complete immutable remote prefix, then verify every artifact.

    :param source_root: Parent of the official extracted split directories.
    :param download_root: New local root receiving the entire remote prefix.
    :param remote_root: Immutable R2 prefix to download.
    :param expected_counts: Exact required row count for each official split.
    :param batch_size: Maximum metadata rows and blob handles scanned together.
    :returns: Zero-mismatch counts after the full download passes.
    :raises FileExistsError: ``download_root`` already exists.
    :raises ValueError: The manifest identifies a different remote prefix.
    """
    if download_root.exists():
        raise FileExistsError(f"refusing to merge into existing download root {download_root}")
    normalized_remote_root = remote_root.rstrip("/")
    download_dir_no_overwrite(normalized_remote_root, download_root)
    manifest = _read_manifest(download_root)
    if manifest.remote_root != normalized_remote_root:
        raise ValueError(
            f"manifest remote_root mismatch: {manifest.remote_root!r} != "
            f"{normalized_remote_root!r}"
        )
    return verify_local_import(
        source_root,
        download_root,
        expected_counts=expected_counts,
        batch_size=batch_size,
    )
