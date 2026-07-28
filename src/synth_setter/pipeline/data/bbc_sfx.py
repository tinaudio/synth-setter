"""Import BBC Sound Effects; use ``synth-setter-bbc-sfx --help`` for the workflow."""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Self, TypedDict

import lance
import pyarrow as pa
import soundfile as sf
import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_none

from synth_setter.pipeline.data.lance_shard import (
    LANCE_DATA_STORAGE_VERSION,
    LANCE_MAX_BYTES_PER_FILE,
)
from synth_setter.pipeline.r2_io import (
    download_dir_no_overwrite,
    upload_dir_immutable,
    upload_to_uri_immutable,
)

IA_ITEM = "BBCSoundEffectsComplete"
IA_METADATA_FILENAME = "ia-metadata.json"
LOCAL_MANIFEST_FILENAME = "release-manifest.json"
REMOTE_URI = "r2://experiments/third_party/BBCSoundEffectsComplete"
DEFAULT_SHARD_TARGET_BYTES = 16 * 1024**3
DEFAULT_VERIFY_BATCH_SIZE = 64
_HASH_CHUNK_BYTES = 8 * 1024**2
_IA_METADATA_ATTEMPTS = 3
_IA_METADATA_TIMEOUT_SECONDS = 120
_MANIFEST_SCHEMA_NAME = "synth-setter.bbc-sfx.release"
_MANIFEST_SCHEMA_VERSION = 1
_LANCE_BLOB_ENCODING = "lance.blob.v2"
_VERIFY_PROGRESS_FILENAME = ".bbc-sfx-verify-progress.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_LOG = structlog.get_logger(__name__)


class _StrictModel(BaseModel):
    """Reject coercion and unknown fields at release-data boundaries.

    .. attribute :: model_config

        Strict Pydantic model configuration.
    """

    model_config = ConfigDict(strict=True, extra="forbid")


class _IAFlexibleModel(BaseModel):
    """Reject coercion while accepting unmodeled Internet Archive fields.

    .. attribute :: model_config

        Strict Pydantic model configuration allowing upstream extensions.
    """

    model_config = ConfigDict(strict=True, extra="allow")


class IAItemMetadata(_IAFlexibleModel):
    """Internet Archive item identity.

    .. attribute :: identifier

        Internet Archive item identifier, constrained to the pinned collection.
    """

    identifier: StrictStr

    @field_validator("identifier")
    @classmethod
    def _identifier_is_target(cls, value: str) -> str:
        """Accept only the pinned Internet Archive item identifier.

        :param value: Candidate item identifier.
        :returns: The accepted identifier.
        :raises ValueError: If the identifier names another item.
        """
        if value != IA_ITEM:
            raise ValueError(f"metadata identifier must be {IA_ITEM!r}")
        return value


class IAFileRecord(_IAFlexibleModel):
    """Unfiltered Internet Archive file record.

    .. attribute :: name

        Item-relative file name.

    .. attribute :: size

        Source byte count as returned by Internet Archive.

    .. attribute :: md5

        Authoritative MD5 checksum when present.

    .. attribute :: sha1

        Authoritative SHA1 checksum when present.

    .. attribute :: crc32

        Authoritative CRC32 checksum when present.

    .. attribute :: mtime

        Internet Archive modification timestamp when present.

    .. attribute :: length

        Internet Archive media duration when present.
    """

    name: StrictStr
    size: StrictStr | StrictInt | None = None
    md5: StrictStr | None = None
    sha1: StrictStr | None = None
    crc32: StrictStr | None = None
    mtime: StrictStr | None = None
    length: StrictStr | None = None


class IAMetadataDocument(_IAFlexibleModel):
    """Required outer fields of an ``ia metadata`` response.

    .. attribute :: metadata

        Validated item identity.

    .. attribute :: files

        Unfiltered item file records.
    """

    metadata: IAItemMetadata
    files: list[IAFileRecord]


class IAInventoryEntry(_StrictModel):
    """Validated authoritative WAV inventory row.

    .. attribute :: path

        Safe item-relative WAV path below ``sounds/``.

    .. attribute :: size

        Positive source size in bytes.

    .. attribute :: md5

        Lowercase authoritative MD5 checksum.

    .. attribute :: sha1

        Lowercase authoritative SHA1 checksum when available.

    .. attribute :: crc32

        Authoritative CRC32 checksum when available.

    .. attribute :: mtime

        Internet Archive modification timestamp when available.

    .. attribute :: length

        Internet Archive media duration when available.
    """

    path: StrictStr
    size: StrictInt
    md5: StrictStr
    sha1: StrictStr | None = None
    crc32: StrictStr | None = None
    mtime: StrictStr | None = None
    length: StrictStr | None = None

    @field_validator("path")
    @classmethod
    def _path_is_safe_wav(cls, value: str) -> str:
        """Require a safe item-relative ``.wav`` path below ``sounds/``.

        :param value: Candidate inventory path.
        :returns: The accepted path.
        :raises ValueError: If the path is unsafe or does not name a WAV.
        """
        _validate_relative_path(value, required_prefix="sounds")
        if not value.endswith(".wav"):
            raise ValueError("IA WAV inventory path must end in .wav")
        return value

    @field_validator("size")
    @classmethod
    def _size_is_positive(cls, value: int) -> int:
        """Require a positive source-file byte count.

        :param value: Candidate byte count.
        :returns: The accepted byte count.
        :raises ValueError: If the byte count is not positive.
        """
        if value <= 0:
            raise ValueError("IA WAV size must be positive")
        return value

    @field_validator("md5")
    @classmethod
    def _md5_is_hex(cls, value: str) -> str:
        """Validate and normalize an authoritative MD5 checksum.

        :param value: Candidate checksum.
        :returns: The lowercase checksum.
        :raises ValueError: If the checksum is not 32 hexadecimal characters.
        """
        if not _MD5_RE.fullmatch(value):
            raise ValueError("IA WAV MD5 must contain 32 hexadecimal characters")
        return value.lower()

    @field_validator("sha1")
    @classmethod
    def _sha1_is_hex(cls, value: str | None) -> str | None:
        """Validate and normalize an optional authoritative SHA1 checksum.

        :param value: Candidate checksum or ``None`` when unavailable.
        :returns: The lowercase checksum or ``None``.
        :raises ValueError: If a checksum is not 40 hexadecimal characters.
        """
        if value is not None and not _SHA1_RE.fullmatch(value):
            raise ValueError("IA WAV SHA1 must contain 40 hexadecimal characters")
        return value.lower() if value is not None else None

    @classmethod
    def from_record(cls, record: IAFileRecord) -> Self:
        """Normalize IA's decimal-string size into one validated WAV row.

        :param record: Unfiltered Internet Archive file record.
        :returns: Validated authoritative WAV inventory entry.
        :raises ValueError: If size or MD5 metadata is absent or malformed.
        """
        raw_size = record.size
        if isinstance(raw_size, bool) or not isinstance(raw_size, (str, int)):
            raise ValueError(f"IA WAV {record.name!r} has no valid size")
        if isinstance(raw_size, str):
            if not raw_size.isdecimal():
                raise ValueError(f"IA WAV {record.name!r} has non-decimal size")
            size = int(raw_size)
        else:
            size = raw_size
        if record.md5 is None:
            raise ValueError(f"IA WAV {record.name!r} has no MD5")
        return cls(
            path=record.name,
            size=size,
            md5=record.md5,
            sha1=record.sha1,
            crc32=record.crc32,
            mtime=record.mtime,
            length=record.length,
        )


@dataclass(frozen=True)
class IASnapshot:
    """Validated IA inventory plus the authoritative snapshot hash.

    .. attribute :: files

        Path-sorted authoritative WAV inventory.

    .. attribute :: sha256

        SHA256 of the exact metadata response bytes.

    .. attribute :: raw_bytes

        Exact metadata response bytes retained for the release.
    """

    files: tuple[IAInventoryEntry, ...]
    sha256: str
    raw_bytes: bytes


class ReleaseFile(_StrictModel):
    """One immutable file expected below the remote release prefix.

    .. attribute :: path

        Safe path relative to the release root.

    .. attribute :: size

        Expected file size in bytes.

    .. attribute :: sha256

        Expected lowercase SHA256 checksum.
    """

    path: StrictStr
    size: StrictInt
    sha256: StrictStr

    @field_validator("path")
    @classmethod
    def _path_is_safe(cls, value: str) -> str:
        """Require a safe release-relative path.

        :param value: Candidate release path.
        :returns: The accepted path.
        """
        _validate_relative_path(value)
        return value

    @field_validator("size")
    @classmethod
    def _size_is_nonnegative(cls, value: int) -> int:
        """Require a nonnegative release-file byte count.

        :param value: Candidate byte count.
        :returns: The accepted byte count.
        :raises ValueError: If the byte count is negative.
        """
        if value < 0:
            raise ValueError("release file size cannot be negative")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256_is_hex(cls, value: str) -> str:
        """Require a lowercase hexadecimal SHA256 checksum.

        :param value: Candidate checksum.
        :returns: The accepted checksum.
        :raises ValueError: If the checksum is malformed.
        """
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("release file SHA256 must contain 64 lowercase hex characters")
        return value


class ReleaseShard(_StrictModel):
    """Row and source-byte range for one Lance shard.

    .. attribute :: dataset_path

        Safe release-relative Lance dataset path below ``shards/``.

    .. attribute :: row_start

        Inclusive global row offset.

    .. attribute :: row_end

        Exclusive global row offset.

    .. attribute :: source_byte_start

        Inclusive cumulative source-byte offset.

    .. attribute :: source_byte_end

        Exclusive cumulative source-byte offset.
    """

    dataset_path: StrictStr
    row_start: StrictInt
    row_end: StrictInt
    source_byte_start: StrictInt
    source_byte_end: StrictInt

    @field_validator("dataset_path")
    @classmethod
    def _dataset_path_is_safe(cls, value: str) -> str:
        """Require a safe shard dataset path below ``shards/``.

        :param value: Candidate dataset path.
        :returns: The accepted path.
        """
        _validate_relative_path(value, required_prefix="shards")
        return value

    @model_validator(mode="after")
    def _ranges_are_nonempty(self) -> Self:
        """Require increasing nonnegative row and source-byte ranges.

        :returns: This validated shard.
        :raises ValueError: If either range is empty, decreasing, or starts below zero.
        """
        if self.row_start < 0 or self.row_end <= self.row_start:
            raise ValueError("shard row range must be nonempty and increasing")
        if self.source_byte_start < 0 or self.source_byte_end <= self.source_byte_start:
            raise ValueError("shard byte range must be nonempty and increasing")
        return self


class ReleaseManifest(_StrictModel):
    """Completion candidate used to verify and publish one immutable release.

    .. attribute :: schema_name

        Stable release-manifest schema identifier.

    .. attribute :: schema_version

        Release-manifest schema version.

    .. attribute :: item_identifier

        Pinned Internet Archive item identifier.

    .. attribute :: lance_data_storage_version

        Lance data storage version used by every shard.

    .. attribute :: lance_blob_encoding

        Lance extension encoding used by the audio column.

    .. attribute :: ia_snapshot_sha256

        SHA256 of the authoritative Internet Archive metadata snapshot.

    .. attribute :: total_rows

        Total WAV rows across all shards.

    .. attribute :: total_source_bytes

        Total authoritative WAV bytes across all shards.

    .. attribute :: shards

        Ordered contiguous shard ranges.

    .. attribute :: release_files

        Immutable release files excluding the local completion candidate.
    """

    schema_name: StrictStr = _MANIFEST_SCHEMA_NAME
    schema_version: StrictInt = _MANIFEST_SCHEMA_VERSION
    item_identifier: StrictStr = IA_ITEM
    lance_data_storage_version: StrictStr = LANCE_DATA_STORAGE_VERSION
    lance_blob_encoding: StrictStr = _LANCE_BLOB_ENCODING
    ia_snapshot_sha256: StrictStr
    total_rows: StrictInt
    total_source_bytes: StrictInt
    shards: list[ReleaseShard]
    release_files: list[ReleaseFile]

    @model_validator(mode="after")
    def _manifest_is_consistent(self) -> Self:
        """Require supported constants, positive totals, and contiguous unique entries.

        :returns: This validated manifest.
        :raises ValueError: If the schema, totals, shard ranges, or file paths are inconsistent.
        """
        if (
            self.schema_name != _MANIFEST_SCHEMA_NAME
            or self.schema_version != _MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("unsupported BBC SFX release manifest schema")
        if self.item_identifier != IA_ITEM:
            raise ValueError("release manifest item identifier is incorrect")
        if self.lance_data_storage_version != LANCE_DATA_STORAGE_VERSION:
            raise ValueError("release manifest Lance storage version is incorrect")
        if self.lance_blob_encoding != _LANCE_BLOB_ENCODING:
            raise ValueError("release manifest blob encoding is incorrect")
        if not _SHA256_RE.fullmatch(self.ia_snapshot_sha256):
            raise ValueError("IA snapshot SHA256 is invalid")
        if self.total_rows <= 0 or self.total_source_bytes <= 0:
            raise ValueError("release totals must be positive")
        if not self.shards or not self.release_files:
            raise ValueError("release manifest must list shards and files")
        row_cursor = 0
        byte_cursor = 0
        dataset_paths: set[str] = set()
        for shard in self.shards:
            if shard.dataset_path in dataset_paths:
                raise ValueError("release manifest contains duplicate shard paths")
            if shard.row_start != row_cursor or shard.source_byte_start != byte_cursor:
                raise ValueError("release manifest shard ranges must be contiguous")
            dataset_paths.add(shard.dataset_path)
            row_cursor = shard.row_end
            byte_cursor = shard.source_byte_end
        if row_cursor != self.total_rows or byte_cursor != self.total_source_bytes:
            raise ValueError("release manifest shard ranges do not match totals")
        if len({item.path for item in self.release_files}) != len(self.release_files):
            raise ValueError("release manifest contains duplicate file paths")
        reserved_paths = {LOCAL_MANIFEST_FILENAME, "manifest.json"}
        if any(item.path in reserved_paths for item in self.release_files):
            raise ValueError("release files cannot contain manifest.json completion markers")
        return self


_TechnicalRow = TypedDict(
    "_TechnicalRow",
    {
        "path": str,
        "size": int,
        "ia_md5": str,
        "ia_sha1": str | None,
        "ia_crc32": str | None,
        "ia_mtime": str | None,
        "ia_length": str | None,
        "sample_rate": int,
        "channels": int,
        "frames": int,
        "format": str,
        "subtype": str,
        "endian": str,
        "duration_seconds": float,
    },
)


class _VerifyProgress(_StrictModel):
    """Durable prefix of shards verified against one manifest.

    .. attribute :: manifest_sha256

        SHA256 of the exact retained manifest bytes.

    .. attribute :: verified_shards

        Ordered prefix of manifest shard paths that passed full verification.
    """

    manifest_sha256: StrictStr
    verified_shards: list[StrictStr]

    @field_validator("manifest_sha256")
    @classmethod
    def _manifest_sha256_is_hex(cls, value: str) -> str:
        """Require a lowercase hexadecimal manifest SHA256.

        :param value: Candidate checksum.
        :returns: Accepted checksum.
        :raises ValueError: If the checksum is malformed.
        """
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("verification manifest SHA256 is invalid")
        return value

    @field_validator("verified_shards")
    @classmethod
    def _shard_paths_are_safe(cls, values: list[str]) -> list[str]:
        """Require unique safe shard paths.

        :param values: Candidate verified shard paths.
        :returns: Accepted paths.
        :raises ValueError: If a path is unsafe or duplicated.
        """
        for value in values:
            _validate_relative_path(value, required_prefix="shards")
        if len(set(values)) != len(values):
            raise ValueError("verification progress contains duplicate shards")
        return values


@dataclass(frozen=True)
class VerifyResult:
    """Verified release totals.

    .. attribute :: rows

        Number of verified source rows.

    .. attribute :: source_bytes

        Number of verified source bytes.

    .. attribute :: mismatches

        Number of detected mismatches; zero for a published release.
    """

    rows: int
    source_bytes: int
    mismatches: int


_METADATA_FIELDS = [
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
]
LANCE_SCHEMA = pa.schema([*_METADATA_FIELDS, lance.blob_field("audio", nullable=False)])
_METADATA_COLUMNS = [field.name for field in _METADATA_FIELDS]


def _validate_relative_path(value: str, required_prefix: str | None = None) -> None:
    """Reject absolute, traversal, non-POSIX, and out-of-prefix paths.

    :param value: Candidate POSIX relative path.
    :param required_prefix: Required first path component, if any.
    :raises ValueError: If the path is unsafe or outside the required prefix.
    """
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    if required_prefix is not None and (not path.parts or path.parts[0] != required_prefix):
        raise ValueError(f"path must be below {required_prefix}/: {value!r}")


def _sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA256 checksum of an in-memory payload.

    :param payload: Bytes to hash.
    :returns: Lowercase hexadecimal SHA256 checksum.
    """
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path, algorithm: str) -> str:
    """Hash a file incrementally without loading it into memory.

    :param path: File to hash.
    :param algorithm: Algorithm accepted by :func:`hashlib.new`.
    :returns: Lowercase hexadecimal checksum.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    """Durably replace a path from an exclusive process-scoped partial file.

    :param path: Destination file.
    :param payload: Complete bytes to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _parse_ia_metadata(raw: bytes) -> IASnapshot:
    """Parse exact IA metadata bytes into a validated authoritative snapshot.

    :param raw: Exact ``ia metadata`` response bytes.
    :returns: Path-sorted validated WAV inventory with snapshot bytes and hash.
    :raises ValueError: If the inventory is empty or contains duplicate WAV paths.
    """
    document = IAMetadataDocument.model_validate_json(raw)
    entries: list[IAInventoryEntry] = []
    seen: set[str] = set()
    for record in document.files:
        if not record.name.endswith(".wav"):
            continue
        entry = IAInventoryEntry.from_record(record)
        if entry.path in seen:
            raise ValueError(f"duplicate IA WAV inventory path: {entry.path!r}")
        seen.add(entry.path)
        entries.append(entry)
    if not entries:
        raise ValueError("IA metadata contains no .wav inventory entries")
    entries.sort(key=lambda entry: entry.path)
    return IASnapshot(tuple(entries), _sha256_bytes(raw), raw)


def load_ia_metadata(path: Path) -> IASnapshot:
    """Parse and validate the authoritative WAV inventory in an IA snapshot.

    :param path: Exact ``ia metadata`` JSON snapshot.
    :returns: Path-sorted validated WAV inventory with snapshot bytes and hash.
    """
    return _parse_ia_metadata(path.read_bytes())


@retry(
    retry=retry_if_exception_type((subprocess.CalledProcessError, subprocess.TimeoutExpired)),
    stop=stop_after_attempt(_IA_METADATA_ATTEMPTS),
    wait=wait_none(),
    reraise=True,
)
def _fetch_ia_metadata(source_root: Path) -> bytes:
    """Fetch IA metadata with bounded attempts and wall-clock time per attempt.

    :param source_root: Working directory for the Internet Archive CLI.
    :returns: Exact metadata response bytes.
    """
    result = subprocess.run(  # noqa: S603 -- executes the installed IA CLI.
        ["ia", "metadata", IA_ITEM],  # noqa: S607
        cwd=source_root,
        check=True,
        stdout=subprocess.PIPE,
        timeout=_IA_METADATA_TIMEOUT_SECONDS,
    )
    return result.stdout


def download_source(source_root: Path) -> IASnapshot:
    """Download resumable WAVs and atomically snapshot authoritative IA metadata.

    :param source_root: Destination directory for the Internet Archive item.
    :returns: Validated inventory loaded from the stable local metadata snapshot.
    """
    source_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 -- executes the installed IA CLI.
        [  # noqa: S607
            "ia",
            "download",
            IA_ITEM,
            "--glob=*.wav",
            "--retries",
            "10",
            "--checksum",
        ],
        cwd=source_root,
        check=True,
    )
    item_root = source_root / IA_ITEM
    metadata_path = item_root / IA_METADATA_FILENAME
    if metadata_path.is_file():
        return load_ia_metadata(metadata_path)
    metadata_bytes = _fetch_ia_metadata(source_root)
    snapshot = _parse_ia_metadata(metadata_bytes)
    _atomic_write(metadata_path, metadata_bytes)
    return snapshot


def partition_inventory(
    inventory: Iterable[IAInventoryEntry], target_bytes: int
) -> list[tuple[IAInventoryEntry, ...]]:
    """Partition path-sorted rows without splitting a source file.

    :param inventory: WAV entries; input order is ignored.
    :param target_bytes: Positive approximate source-byte limit per shard.
    :returns: Path-ordered nonempty shards; an oversized file remains whole.
    :raises ValueError: If ``target_bytes`` is not positive.
    """
    if target_bytes <= 0:
        raise ValueError("shard target bytes must be positive")
    shards: list[tuple[IAInventoryEntry, ...]] = []
    current: list[IAInventoryEntry] = []
    current_bytes = 0
    for entry in sorted(inventory, key=lambda item: item.path):
        if current and current_bytes + entry.size > target_bytes:
            shards.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(entry)
        current_bytes += entry.size
    if current:
        shards.append(tuple(current))
    return shards


def _actual_wav_paths(item_root: Path) -> set[str]:
    """List safe regular WAV paths below an extracted item.

    :param item_root: Root of the downloaded Internet Archive item.
    :returns: Item-relative POSIX WAV paths.
    :raises ValueError: If ``sounds/`` is absent or contains a symlinked WAV.
    """
    sounds_root = item_root / "sounds"
    if not sounds_root.is_dir():
        raise ValueError(f"missing sounds directory: {sounds_root}")
    actual: set[str] = set()
    for path in sounds_root.rglob("*.wav"):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe WAV source file: {path}")
        actual.add(path.relative_to(item_root).as_posix())
    return actual


def _validate_source_inventory(item_root: Path, entries: Iterable[IAInventoryEntry]) -> None:
    """Require exact source paths, byte counts, and authoritative MD5 checksums.

    :param item_root: Root of the downloaded Internet Archive item.
    :param entries: Authoritative WAV inventory.
    :raises ValueError: If files are missing, extra, unsafe, or corrupt.
    """
    inventory = tuple(entries)
    expected = {entry.path for entry in inventory}
    actual = _actual_wav_paths(item_root)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"missing WAV source files: {missing[:5]}")
    if extra:
        raise ValueError(f"extra WAV source files: {extra[:5]}")
    for entry in inventory:
        path = item_root / entry.path
        if path.stat().st_size != entry.size:
            raise ValueError(f"corrupt size for {entry.path!r}")
        if _file_hash(path, "md5") != entry.md5:
            raise ValueError(f"corrupt MD5 for {entry.path!r}")


def _source_payload(item_root: Path, entry: IAInventoryEntry) -> bytes:
    """Read and revalidate one source WAV immediately before consumption.

    :param item_root: Root of the downloaded Internet Archive item.
    :param entry: Validated WAV inventory entry.
    :returns: Exact authoritative source bytes.
    :raises ValueError: If the source size or MD5 changed after preflight.
    """
    payload = (item_root / entry.path).read_bytes()
    if len(payload) != entry.size:
        raise ValueError(f"corrupt size for {entry.path!r}")
    if hashlib.md5(payload).hexdigest() != entry.md5:  # noqa: S324 -- IA inventory contract.
        raise ValueError(f"corrupt MD5 for {entry.path!r}")
    return payload


def _technical_row(entry: IAInventoryEntry, payload: bytes) -> _TechnicalRow:
    """Combine authoritative inventory fields with decoded WAV metadata.

    :param entry: Validated WAV inventory entry.
    :param payload: Exact source WAV bytes.
    :returns: Values for every non-blob Lance schema column.
    """
    info = sf.info(io.BytesIO(payload))
    return {
        "path": entry.path,
        "size": entry.size,
        "ia_md5": entry.md5,
        "ia_sha1": entry.sha1,
        "ia_crc32": entry.crc32,
        "ia_mtime": entry.mtime,
        "ia_length": entry.length,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
        "endian": info.endian,
        "duration_seconds": info.duration,
    }


def _record_batch(item_root: Path, entry: IAInventoryEntry) -> pa.RecordBatch:
    """Build one typed Lance row with the source WAV as a blob.

    :param item_root: Root of the downloaded Internet Archive item.
    :param entry: Validated WAV inventory entry.
    :returns: Single-row batch matching :data:`LANCE_SCHEMA`.
    """
    payload = _source_payload(item_root, entry)
    row = _technical_row(entry, payload)
    arrays = [pa.array([row[field.name]], type=field.type) for field in _METADATA_FIELDS]
    arrays.append(lance.blob_array([payload]))
    return pa.record_batch(arrays, schema=LANCE_SCHEMA)


def _write_shard(
    item_root: Path, entries: tuple[IAInventoryEntry, ...], output_path: Path
) -> None:
    """Write one Lance shard through a sibling partial directory.

    :param item_root: Root of the downloaded Internet Archive item.
    :param entries: Nonempty path-ordered shard inventory.
    :param output_path: New final Lance dataset path.
    :raises FileExistsError: If the final shard path already exists.
    """
    partial = output_path.with_name(f"{output_path.name}.partial")
    if output_path.exists():
        raise FileExistsError(f"refusing existing shard output: {output_path}")
    if partial.exists():
        shutil.rmtree(partial)

    batches = (_record_batch(item_root, entry) for entry in entries)
    lance.write_dataset(
        batches,
        partial,
        schema=LANCE_SCHEMA,
        mode="create",
        max_bytes_per_file=LANCE_MAX_BYTES_PER_FILE,
        max_rows_per_group=1,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )
    partial.replace(output_path)


def _write_or_validate_file(path: Path, payload: bytes) -> None:
    """Write a missing resume artifact or require its exact bytes.

    :param path: Resume artifact path.
    :param payload: Expected complete bytes.
    :raises ValueError: If an existing artifact differs.
    """
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"partial release artifact mismatch: {path}")
        return
    _atomic_write(path, payload)


def _release_files(root: Path) -> list[ReleaseFile]:
    """Inventory immutable release files except the local completion candidate.

    :param root: Completed or partial release root.
    :returns: Path-sorted file sizes and SHA256 checksums.
    """
    result = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == LOCAL_MANIFEST_FILENAME:
            continue
        result.append(
            ReleaseFile(path=relative, size=path.stat().st_size, sha256=_file_hash(path, "sha256"))
        )
    return result


def convert_release(
    source_root: Path,
    release_root: Path,
    *,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
) -> ReleaseManifest:
    """Validate source WAVs and create an atomic, immutable Lance release.

    :param source_root: Directory containing the downloaded Internet Archive item.
    :param release_root: New final directory for the completed release.
    :param shard_target_bytes: Positive approximate source-byte limit per shard.
    :returns: Validated manifest for the completed local release.
    :raises FileExistsError: If the final release path already exists.
    """
    partial_root = release_root.with_name(f"{release_root.name}.partial")
    if release_root.exists():
        raise FileExistsError(f"refusing existing release output: {release_root}")
    snapshot_path = source_root / IA_ITEM / IA_METADATA_FILENAME
    snapshot = load_ia_metadata(snapshot_path)
    item_root = source_root / IA_ITEM
    _validate_source_inventory(item_root, snapshot.files)
    partitions = partition_inventory(snapshot.files, shard_target_bytes)
    partial_root.mkdir(parents=True, exist_ok=True)
    (partial_root / "shards").mkdir(exist_ok=True)
    metadata_root = partial_root / "metadata"
    metadata_root.mkdir(exist_ok=True)
    _write_or_validate_file(metadata_root / IA_METADATA_FILENAME, snapshot.raw_bytes)
    inventory_payload = ("\n".join(entry.model_dump_json() for entry in snapshot.files) + "\n").encode(
        "utf-8"
    )
    _write_or_validate_file(metadata_root / "inventory.jsonl", inventory_payload)
    shards: list[ReleaseShard] = []
    row_start = 0
    byte_start = 0
    for index, entries in enumerate(partitions):
        dataset_path = f"shards/shard-{index:05d}.lance"
        output_path = partial_root / dataset_path
        if output_path.exists():
            _verify_shard(output_path, item_root, entries=entries, batch_size=DEFAULT_VERIFY_BATCH_SIZE)
        else:
            _write_shard(item_root, entries, output_path)
        _LOG.info("bbc_sfx_shard_converted", dataset_path=dataset_path, rows=len(entries))
        row_end = row_start + len(entries)
        byte_end = byte_start + sum(entry.size for entry in entries)
        shards.append(
            ReleaseShard(
                dataset_path=dataset_path,
                row_start=row_start,
                row_end=row_end,
                source_byte_start=byte_start,
                source_byte_end=byte_end,
            )
        )
        row_start = row_end
        byte_start = byte_end
    manifest = ReleaseManifest(
        ia_snapshot_sha256=snapshot.sha256,
        total_rows=len(snapshot.files),
        total_source_bytes=sum(entry.size for entry in snapshot.files),
        shards=shards,
        release_files=_release_files(partial_root),
    )
    _atomic_write(
        partial_root / LOCAL_MANIFEST_FILENAME,
        (manifest.model_dump_json(indent=2) + "\n").encode("utf-8"),
    )
    partial_root.replace(release_root)
    return manifest


def _load_manifest(path: Path) -> ReleaseManifest:
    """Load a strict release manifest from JSON bytes.

    :param path: Manifest file to parse.
    :returns: Validated release manifest.
    """
    return ReleaseManifest.model_validate_json(path.read_bytes())


def _assert_release_file_set(
    root: Path,
    expected_files: list[ReleaseFile],
    *,
    allowed_extra_paths: frozenset[str] = frozenset(),
) -> None:
    """Require exact release paths, sizes, and SHA256 checksums.

    :param root: Release directory to inspect.
    :param expected_files: Authoritative immutable file inventory.
    :param allowed_extra_paths: Exact local-only paths permitted outside the inventory.
    :raises ValueError: If any file is missing, extra, or corrupt.
    """
    expected = {entry.path: entry for entry in expected_files}
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    missing = sorted(expected.keys() - actual)
    extra = sorted(actual - expected.keys() - allowed_extra_paths)
    mismatches: list[str] = []
    if missing:
        mismatches.append(f"missing={missing[:5]}")
    if extra:
        mismatches.append(f"extra={extra[:5]}")
    for relative in sorted(actual & expected.keys()):
        path = root / relative
        record = expected[relative]
        if path.stat().st_size != record.size or _file_hash(path, "sha256") != record.sha256:
            mismatches.append(f"hash/size={relative}")
    if mismatches:
        raise ValueError(f"release file mismatch: {'; '.join(mismatches)}")


def upload_release(
    release_root: Path, *, remote_uri: str = REMOTE_URI
) -> ReleaseManifest:
    """Immutably upload release data while withholding the completion manifest.

    :param release_root: Completed local release containing its candidate manifest.
    :param remote_uri: Destination object-store prefix.
    :returns: Validated candidate manifest for later verification.
    """
    manifest = _load_manifest(release_root / LOCAL_MANIFEST_FILENAME)
    _assert_release_file_set(
        release_root,
        manifest.release_files,
        allowed_extra_paths=frozenset({LOCAL_MANIFEST_FILENAME}),
    )
    upload_dir_immutable(release_root, remote_uri, exclude=LOCAL_MANIFEST_FILENAME)
    return manifest


def _verify_blob(
    blob: lance.BlobFile,
    source_payload: bytes,
    source_path: Path,
    expected_md5: str,
) -> None:
    """Compare a Lance blob byte-for-byte with its authoritative source WAV.

    :param blob: Openable Lance blob value.
    :param source_payload: Captured authoritative source WAV bytes.
    :param source_path: Authoritative local WAV path used for diagnostics.
    :param expected_md5: Authoritative lowercase MD5 checksum.
    :raises ValueError: If blob bytes or the source checksum differ.
    """
    digest = hashlib.md5()  # noqa: S324 -- verifies the authoritative IA checksum.
    with io.BytesIO(source_payload) as source, blob:
        while True:
            source_chunk = source.read(_HASH_CHUNK_BYTES)
            blob_chunk = blob.read(_HASH_CHUNK_BYTES)
            if source_chunk != blob_chunk:
                raise ValueError(f"blob byte mismatch for {source_path}")
            if not source_chunk:
                break
            digest.update(source_chunk)
    if digest.hexdigest() != expected_md5:
        raise ValueError(f"blob MD5 mismatch for {source_path}")


def _verify_inventory_sidecar(path: Path, expected: tuple[IAInventoryEntry, ...]) -> None:
    """Require the downloaded JSONL sidecar to match the source snapshot.

    :param path: Downloaded inventory sidecar.
    :param expected: Authoritative path-ordered inventory.
    :raises ValueError: If parsed rows differ from the authoritative inventory.
    """
    rows = [
        IAInventoryEntry.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if rows != list(expected):
        raise ValueError("inventory sidecar mismatch")


def _verify_shard(
    dataset_path: Path,
    item_root: Path,
    *,
    entries: tuple[IAInventoryEntry, ...],
    batch_size: int,
) -> None:
    """Verify one downloaded shard's schema, metadata, and audio blobs.

    :param dataset_path: Downloaded Lance shard path.
    :param item_root: Root of the authoritative Internet Archive item.
    :param entries: Authoritative entries assigned to this shard.
    :param batch_size: Maximum metadata rows read per batch.
    :raises ValueError: If schema, row count, metadata, or blob bytes differ.
    """
    dataset = lance.dataset(dataset_path)
    if not dataset.schema.equals(LANCE_SCHEMA, check_metadata=True):
        raise ValueError(f"Lance schema mismatch in {dataset_path}")
    row_offset = 0
    for batch in dataset.to_batches(columns=_METADATA_COLUMNS, batch_size=batch_size):
        for row in batch.to_pylist():
            if row_offset >= len(entries):
                raise ValueError(f"extra Lance row in {dataset_path}")
            entry = entries[row_offset]
            source_path = item_root / entry.path
            source_payload = _source_payload(item_root, entry)
            if row != _technical_row(entry, source_payload):
                raise ValueError(f"typed metadata mismatch for {entry.path!r}")
            blob = dataset.take_blobs("audio", indices=[row_offset])[0]
            _verify_blob(blob, source_payload, source_path, entry.md5)
            row_offset += 1
    if row_offset != len(entries):
        raise ValueError(f"missing Lance rows in {dataset_path}")


def verify_release(
    source_root: Path,
    download_root: Path,
    release_manifest_path: Path,
    *,
    batch_size: int = DEFAULT_VERIFY_BATCH_SIZE,
    remote_uri: str = REMOTE_URI,
) -> VerifyResult:
    """Download and fully compare a release before publishing its manifest.

    :param source_root: Directory containing the authoritative source item.
    :param download_root: Directory receiving or resuming the remote release.
    :param release_manifest_path: Local completion candidate to verify and publish.
    :param batch_size: Positive maximum metadata rows read per batch.
    :param remote_uri: Source prefix and completion-manifest destination.
    :returns: Verified row, source-byte, and mismatch totals.
    :raises ValueError: If the batch size or any release invariant is invalid.
    """
    if download_root.exists() and not download_root.is_dir():
        raise ValueError(f"download root is not a directory: {download_root}")
    if batch_size <= 0:
        raise ValueError("verification batch size must be positive")
    manifest_bytes = release_manifest_path.read_bytes()
    manifest = ReleaseManifest.model_validate_json(manifest_bytes)
    snapshot = load_ia_metadata(source_root / IA_ITEM / IA_METADATA_FILENAME)
    if manifest.ia_snapshot_sha256 != snapshot.sha256:
        raise ValueError("IA snapshot SHA256 mismatch")
    if manifest.total_rows != len(snapshot.files) or manifest.total_source_bytes != sum(
        entry.size for entry in snapshot.files
    ):
        raise ValueError("release manifest inventory totals mismatch")
    item_root = source_root / IA_ITEM
    _validate_source_inventory(item_root, snapshot.files)
    download_dir_no_overwrite(remote_uri, download_root, exclude="manifest.json")
    progress_path = download_root / _VERIFY_PROGRESS_FILENAME
    _assert_release_file_set(
        download_root,
        manifest.release_files,
        allowed_extra_paths=frozenset({_VERIFY_PROGRESS_FILENAME}),
    )
    downloaded_snapshot = download_root / "metadata" / IA_METADATA_FILENAME
    if downloaded_snapshot.read_bytes() != snapshot.raw_bytes:
        raise ValueError("downloaded IA metadata snapshot mismatch")
    _verify_inventory_sidecar(download_root / "metadata" / "inventory.jsonl", snapshot.files)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    if progress_path.is_file():
        progress = _VerifyProgress.model_validate_json(progress_path.read_bytes())
    else:
        progress = _VerifyProgress(manifest_sha256=manifest_sha256, verified_shards=[])
    expected_shard_paths = [shard.dataset_path for shard in manifest.shards]
    if progress.manifest_sha256 != manifest_sha256 or expected_shard_paths[
        : len(progress.verified_shards)
    ] != progress.verified_shards:
        raise ValueError("verification progress does not match the retained manifest")

    source_byte_start = 0
    for index, shard in enumerate(manifest.shards):
        entries = snapshot.files[shard.row_start : shard.row_end]
        source_byte_end = source_byte_start + sum(entry.size for entry in entries)
        if (
            shard.source_byte_start != source_byte_start
            or shard.source_byte_end != source_byte_end
        ):
            raise ValueError(f"shard source byte range mismatch: {shard.dataset_path}")
        if index >= len(progress.verified_shards):
            _verify_shard(
                download_root / shard.dataset_path,
                item_root,
                entries=entries,
                batch_size=batch_size,
            )
            progress.verified_shards.append(shard.dataset_path)
            _atomic_write(progress_path, progress.model_dump_json().encode("utf-8"))
            _LOG.info("bbc_sfx_shard_verified", dataset_path=shard.dataset_path, rows=len(entries))
        source_byte_start = source_byte_end
    progress_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as temporary_dir:
        publish_path = Path(temporary_dir) / "manifest.json"
        publish_path.write_bytes(manifest_bytes)
        upload_to_uri_immutable(publish_path, f"{remote_uri}/manifest.json")
    return VerifyResult(manifest.total_rows, manifest.total_source_bytes, 0)
