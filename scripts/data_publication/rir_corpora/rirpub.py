#!/usr/bin/env python3
"""Generic immutable Lance publication for third-party RIR corpora.

Stages, each idempotent and reporting JSON under ``<corpus>/state``::

    extract   unpack pinned archives, write the per-object SHA-256 inventory
    build     write publication/all.lance (one row per source object, exact bytes)
              plus metadata/, source/ ancillary copies, and README.md
    upload    rclone copy --checksum --immutable (never the completion marker)
    verify    reopen all.lance directly from R2, compare every payload, decode audio,
              rclone check the prefix, then upload _COMPLETE.json strictly last

Invocation (from the repository root)::

    uv run python -m scripts.data_publication.rir_corpora.rirpub <stage> <corpus>
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import soundfile as sf

from scripts.data_publication.rir_corpora.specs import SPECS, Spec, executable

log = logging.getLogger(__name__)

R2_ROOT = "r2:experiments/third_party"
S3_ROOT = "s3://experiments/third_party"
MIN_FREE_BYTES = 120 * 1024**3
BATCH_BYTES = 1024**3
BATCH_ROWS = 2000
AUDIO_EXTENSIONS = {".wav", ".wave", ".flac", ".aif", ".aiff", ".ogg", ".mp3", ".w64"}
MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".wave": "audio/wav",
    ".flac": "audio/flac",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".ogg": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".w64": "audio/x-w64",
    ".sofa": "application/x-sofa-hdf5",
    ".npy": "application/x-numpy",
    ".npz": "application/x-numpy-zip",
    ".mat": "application/x-matlab-data",
    ".pkl": "application/x-python-pickle",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".zip": "application/zip",
    ".m": "text/x-matlab",
    ".py": "text/x-python",
    ".html": "text/html",
    ".xml": "application/xml",
}
META_COLUMNS = [
    "source_row_index",
    "source_path",
    "media_type",
    "source_num_bytes",
    "source_sha256",
    "audio_decodable",
    "sample_rate",
    "channels",
    "frames",
    "subtype",
]
TOOL_FILES = ("rirpub.py", "specs.py", "fetch_sources.py")
MODULE = "uv run python -m scripts.data_publication.rir_corpora.rirpub"
_UNDECODABLE: dict[str, Any] = {
    "audio_decodable": False,
    "sample_rate": None,
    "channels": None,
    "frames": None,
    "subtype": None,
}


def corpora_root() -> Path:
    """Return the local root holding one directory per corpus.

    :returns: ``RIR_CORPORA_ROOT`` or ``~/datasets/rir-corpora``.
    """
    return Path(os.environ.get("RIR_CORPORA_ROOT", "~/datasets/rir-corpora")).expanduser()


def now() -> str:
    """Return the current UTC time in the completion-marker format.

    :returns: ``YYYY-MM-DDTHH:MM:SSZ``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 of an in-memory payload.

    :param data: Payload bytes.
    :returns: Hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, streamed in 1 MiB blocks.

    :param path: File to hash.
    :returns: Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    """Write a sorted, indented JSON document, creating parent directories.

    :param path: Destination.
    :param payload: JSON-serialisable object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rclone(*args: str, capture: bool = False) -> str:
    """Run rclone with ``--checksum`` appended, as every R2 operation must.

    :param \\*args: rclone sub-command and operands.
    :param capture: Return stdout instead of streaming it.
    :returns: Captured stdout, or an empty string.
    """
    argv = [executable("rclone"), *args, "--checksum"]
    if capture:
        return subprocess.run(  # noqa: S603 — args are literal strings
            argv, capture_output=True, text=True, check=True
        ).stdout
    subprocess.run(argv, check=True)  # noqa: S603 — args are literal strings
    return ""


def media_type(path: Path) -> str:
    """Infer a MIME type from the file extension.

    :param path: Source object path.
    :returns: Known MIME type or ``application/octet-stream``.
    """
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def audio_info(payload: bytes, path: Path) -> dict[str, Any]:
    """Fully decode an audio payload with libsndfile and report its header fields.

    :param payload: Source object bytes.
    :param path: Source object path; only audio extensions are decoded.
    :returns: ``audio_decodable`` plus nullable ``sample_rate``/``channels``/``frames``/``subtype``.
    """
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        return dict(_UNDECODABLE)
    try:
        with sf.SoundFile(io.BytesIO(payload)) as audio:
            samples = audio.read(dtype="float32", always_2d=True)
            if samples.shape != (audio.frames, audio.channels):
                return dict(_UNDECODABLE)
            return {
                "audio_decodable": True,
                "sample_rate": int(audio.samplerate),
                "channels": int(audio.channels),
                "frames": int(audio.frames),
                "subtype": f"{audio.format}/{audio.subtype}",
            }
    except (sf.LibsndfileError, RuntimeError, ValueError):
        return dict(_UNDECODABLE)


def _format_rows(formats: Counter[tuple[object, ...]]) -> list[dict[str, Any]]:
    """Render decoded-format counts for JSON reports.

    :param formats: ``(subtype, sample_rate, channels)`` counts.
    :returns: One record per combination, sorted.
    """
    return [
        {"subtype": key[0], "sample_rate": key[1], "channels": key[2], "count": count}
        for key, count in sorted(formats.items(), key=str)
    ]


class Corpus:
    """Filesystem and R2 layout for one corpus, plus the four stages."""

    def __init__(self, spec: Spec) -> None:
        """Bind a descriptor to its local and remote locations.

        :param spec: Corpus descriptor.
        """
        self.spec = spec
        self.root = corpora_root() / spec.name
        self.source = self.root / "source"
        self.payload_root = self.root / spec.payload_root
        self.state = self.root / "state"
        self.publication = self.root / "publication"
        self.all_path = self.publication / "all.lance"
        self.rclone_prefix = f"{R2_ROOT}/{spec.name}"
        self.s3_all = f"{S3_ROOT}/{spec.name}/all.lance"
        self.state.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- extract
    def check_capacity(self) -> dict[str, int]:
        """Fail closed when free disk is below the publication threshold.

        :returns: Free bytes and the threshold.
        :raises RuntimeError: Free space is below ``MIN_FREE_BYTES``.
        """
        usage = shutil.disk_usage(self.root)
        if usage.free < MIN_FREE_BYTES:
            raise RuntimeError(
                f"free space {usage.free} below fail-closed threshold {MIN_FREE_BYTES}"
            )
        return {"free_bytes": usage.free, "fail_closed_below_bytes": MIN_FREE_BYTES}

    def archive_inventory(self) -> list[dict[str, Any]]:
        """Hash every pinned archive under ``source/``.

        :returns: Archive name, byte length, and SHA-256 per archive.
        """
        items = []
        for name in self.spec.archives:
            path = self.source / name
            items.append(
                {"archive": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        return items

    def extract(self) -> None:
        """Unpack the archives and write the per-object inventory and report."""
        capacity = self.check_capacity()
        archives = self.archive_inventory()
        if self.spec.archives:
            if self.payload_root.exists():
                shutil.rmtree(self.payload_root)
            self.payload_root.mkdir(parents=True)
            self.spec.extract(self.source, self.payload_root)
        objects = self.object_inventory()
        with (self.state / "source-objects.jsonl").open("w", encoding="utf-8") as stream:
            for item in objects:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
        object_bytes = sum(int(item["bytes"]) for item in objects)
        write_json(
            self.state / "extract-report.json",
            {
                "extracted_at": now(),
                "capacity": capacity,
                "archives": archives,
                "objects": len(objects),
                "object_bytes": object_bytes,
                "by_media_type": dict(Counter(str(item["media_type"]) for item in objects)),
            },
        )
        log.info(
            "%s", json.dumps({"stage": "extract", "objects": len(objects), "bytes": object_bytes})
        )

    def object_inventory(self) -> list[dict[str, Any]]:
        """Hash every non-excluded file under the payload root in sorted path order.

        :returns: Path, byte length, SHA-256, and media type per object.
        """
        items = []
        for path in sorted(p for p in self.payload_root.rglob("*") if p.is_file()):
            relative = path.relative_to(self.payload_root).as_posix()
            if self.spec.excluded(relative):
                continue
            items.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "media_type": media_type(path),
                }
            )
        return items

    def load_inventory(self) -> list[dict[str, Any]]:
        """Read the inventory written by ``extract``.

        :returns: Inventory records in row order.
        """
        lines = (self.state / "source-objects.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    # ------------------------------------------------------------------ build
    def schema(self, archives: list[dict[str, Any]]) -> pa.Schema:
        """Build the row schema with provenance pinned in the schema metadata.

        :param archives: Archive inventory recorded in the metadata.
        :returns: Arrow schema with a ``lance.blob.v2`` payload column.
        """
        blob = lance.blob_field("source_bytes", nullable=False).with_metadata(
            {b"content": b"exact source object bytes, untranscoded"}
        )
        metadata = {
            b"dataset": self.spec.title.encode(),
            b"source_urls": json.dumps(self.spec.source_urls).encode(),
            b"source_pin": self.spec.pin.encode(),
            b"source_archives": json.dumps(archives, sort_keys=True).encode(),
            b"license": self.spec.license.encode(),
            b"usage": self.spec.usage.encode(),
            b"source_order": b"sorted POSIX relative path of every source object",
            b"row_granularity": b"one row per source object",
        }
        return pa.schema(
            [
                pa.field("source_row_index", pa.int32(), nullable=False),
                pa.field("source_path", pa.string(), nullable=False),
                pa.field("media_type", pa.string(), nullable=False),
                pa.field("source_num_bytes", pa.int64(), nullable=False),
                pa.field("source_sha256", pa.string(), nullable=False),
                pa.field("audio_decodable", pa.bool_(), nullable=False),
                pa.field("sample_rate", pa.int32(), nullable=True),
                pa.field("channels", pa.int32(), nullable=True),
                pa.field("frames", pa.int64(), nullable=True),
                pa.field("subtype", pa.string(), nullable=True),
                blob,
            ],
            metadata=metadata,
        )

    def batches(
        self, inventory: list[dict[str, Any]], schema: pa.Schema
    ) -> Iterator[pa.RecordBatch]:
        """Stream record batches bounded by ``BATCH_BYTES`` of payload or ``BATCH_ROWS``.

        :param inventory: Source objects in row order.
        :param schema: Row schema.
        :returns: Lazy batch iterator; a single object may exceed the byte bound.
        """
        rows: list[dict[str, Any]] = []
        payloads: list[bytes] = []
        pending = 0

        def flush() -> pa.RecordBatch:
            nonlocal rows, payloads, pending
            arrays = [
                pa.array([r["source_row_index"] for r in rows], pa.int32()),
                pa.array([r["source_path"] for r in rows], pa.string()),
                pa.array([r["media_type"] for r in rows], pa.string()),
                pa.array([r["source_num_bytes"] for r in rows], pa.int64()),
                pa.array([r["source_sha256"] for r in rows], pa.string()),
                pa.array([r["audio_decodable"] for r in rows], pa.bool_()),
                pa.array([r["sample_rate"] for r in rows], pa.int32()),
                pa.array([r["channels"] for r in rows], pa.int32()),
                pa.array([r["frames"] for r in rows], pa.int64()),
                pa.array([r["subtype"] for r in rows], pa.string()),
                lance.blob_array(payloads),
            ]
            batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            rows, payloads, pending = [], [], 0
            return batch

        def generate() -> Iterator[pa.RecordBatch]:
            nonlocal pending
            for index, item in enumerate(inventory):
                path = self.payload_root / str(item["path"])
                payload = path.read_bytes()
                if len(payload) != item["bytes"] or sha256_bytes(payload) != item["sha256"]:
                    raise RuntimeError(f"source object changed since inventory: {item['path']}")
                rows.append(
                    {
                        "source_row_index": index,
                        "source_path": item["path"],
                        "media_type": item["media_type"],
                        "source_num_bytes": len(payload),
                        "source_sha256": item["sha256"],
                        **audio_info(payload, path),
                    }
                )
                payloads.append(payload)
                pending += len(payload)
                if pending >= BATCH_BYTES or len(rows) >= BATCH_ROWS:
                    log.info("  batch through row %d (%d bytes)", index, pending)
                    yield flush()
            if rows:
                yield flush()

        return generate()

    def build(self) -> None:
        """Write the physical Lance dataset, ancillary copies, metadata, and card."""
        capacity = self.check_capacity()
        inventory = self.load_inventory()
        archives = self.archive_inventory()
        if self.publication.exists():
            shutil.rmtree(self.publication)
        self.publication.mkdir(parents=True)
        schema = self.schema(archives)
        reader = pa.RecordBatchReader.from_batches(schema, self.batches(inventory, schema))
        started = time.time()
        dataset = lance.write_dataset(
            reader,
            self.all_path,
            schema=schema,
            mode="create",
            max_rows_per_file=BATCH_ROWS,
            max_rows_per_group=64,
            data_storage_version="2.2",
            commit_message=f"{self.spec.title} physical dataset",
            transaction_properties={"source_pin": self.spec.pin, "usage": self.spec.usage},
        )
        local = self.validate_local(dataset, inventory)
        local["build_seconds"] = round(time.time() - started, 1)
        self.copy_ancillary(inventory)
        self.write_metadata(capacity, archives, inventory, local, dataset)
        self.write_readme(inventory, local)
        log.info(
            "%s",
            json.dumps(
                {
                    "stage": "build",
                    "rows": dataset.count_rows(),
                    "fragments": len(dataset.get_fragments()),
                }
            ),
        )

    def validate_local(
        self, dataset: lance.LanceDataset, inventory: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Check the local dataset's metadata against the inventory and spot-read blobs.

        :param dataset: Freshly written dataset.
        :param inventory: Source objects in row order.
        :returns: Row count, null counts, decoded-format summary, and spot-check rows.
        :raises RuntimeError: Any row disagrees with the inventory.
        """
        if dataset.count_rows() != len(inventory):
            raise RuntimeError("row count mismatch")
        table = dataset.scanner(columns=META_COLUMNS).to_table()
        pyrows = table.to_pylist()
        for index, (actual, expected) in enumerate(zip(pyrows, inventory, strict=True)):
            if (
                actual["source_row_index"] != index
                or actual["source_path"] != expected["path"]
                or actual["source_num_bytes"] != expected["bytes"]
                or actual["source_sha256"] != expected["sha256"]
            ):
                raise RuntimeError(f"local metadata mismatch row {index}")
        formats: Counter[tuple[object, ...]] = Counter()
        decodable = 0
        for row in pyrows:
            if row["audio_decodable"]:
                decodable += 1
                formats[(row["subtype"], row["sample_rate"], row["channels"])] += 1
        # Spot-verify blob bytes locally; the remote pass re-reads every payload.
        sample = sorted({0, len(inventory) - 1, len(inventory) // 2})
        for index in sample:
            payload = dataset.take_blobs("source_bytes", indices=[index])[0].read()
            if sha256_bytes(payload) != inventory[index]["sha256"]:
                raise RuntimeError(f"local blob mismatch row {index}")
        return {
            "rows": len(inventory),
            "null_counts": {name: table.column(name).null_count for name in META_COLUMNS},
            "audio_decodable_rows": decodable,
            "audio_formats": _format_rows(formats),
            "fragments": len(dataset.get_fragments()),
            "local_blob_spot_checks": sample,
        }

    def copy_ancillary(self, inventory: list[dict[str, Any]]) -> None:
        """Copy documentation objects and acquisition records as plain files.

        :param inventory: Source objects in row order.
        """
        for item in inventory:
            relative = str(item["path"])
            if not self.spec.is_ancillary(relative):
                continue
            target = self.publication / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.payload_root / relative, target)
        acquisition_dir = self.publication / "source" / "_acquisition"
        for extra in self.spec.provenance_files:
            src = self.source / extra
            if src.exists():
                acquisition_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, acquisition_dir / extra)
        acquisition = self.root / "acquisition.json"
        if not acquisition.exists():
            acquisition = self.source / "acquisition.json"
        if acquisition.exists():
            acquisition_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(acquisition, acquisition_dir / "acquisition.json")

    def commands(self) -> list[str]:
        """Render the exact publication command lines recorded in the card.

        :returns: Acquisition commands followed by the four stages.
        """
        return [
            *self.spec.acquisition_commands,
            f"{MODULE} extract {self.spec.name}",
            f"{MODULE} build {self.spec.name}",
            f"rclone copy {self.publication}/ {self.rclone_prefix} --checksum --immutable "
            "--exclude _COMPLETE.json",
            f"{MODULE} verify {self.spec.name}",
        ]

    def write_metadata(
        self,
        capacity: dict[str, int],
        archives: list[dict[str, Any]],
        inventory: list[dict[str, Any]],
        local: dict[str, Any],
        dataset: lance.LanceDataset,
    ) -> None:
        """Write inventories, reports, tool copies, and command lines under ``metadata/``.

        :param capacity: Disk preflight result.
        :param archives: Archive inventory.
        :param inventory: Source objects in row order.
        :param local: Local validation report.
        :param dataset: Written dataset (its schema is recorded).
        """
        metadata = self.publication / "metadata"
        metadata.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.state / "source-objects.jsonl", metadata / "source-objects.jsonl")
        shutil.copyfile(self.state / "extract-report.json", metadata / "extract-report.json")
        (metadata / "source-objects.sha256").write_text(
            "".join(f"{item['sha256']}  {item['path']}\n" for item in inventory),
            encoding="utf-8",
        )
        scripts = metadata / "scripts"
        scripts.mkdir(exist_ok=True)
        for name in TOOL_FILES:
            shutil.copyfile(Path(__file__).resolve().parent / name, scripts / name)
        rclone_version = subprocess.run(  # noqa: S603 — args are literal strings
            [executable("rclone"), "version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()[0]
        tool_versions = {
            "python": sys.version.split()[0],
            "pylance": lance.__version__,
            "pyarrow": pa.__version__,
            "soundfile": sf.__version__,
            "rclone": rclone_version,
            "lance_data_storage_version": "2.2",
        }
        write_json(metadata / "tool-versions.json", tool_versions)
        write_json(
            metadata / "build-report.json",
            {
                "built_at": now(),
                "dataset": self.spec.title,
                "source_pin": self.spec.pin,
                "source_urls": self.spec.source_urls,
                "capacity": capacity,
                "archives": archives,
                "objects": len(inventory),
                "object_bytes": sum(int(i["bytes"]) for i in inventory),
                "local_validation": local,
                "schema": dataset.schema.to_string(),
                "tool_versions": tool_versions,
            },
        )
        (metadata / "publication-commands.txt").write_text(
            "\n".join([*self.commands(), ""]), encoding="utf-8"
        )

    def write_readme(self, inventory: list[dict[str, Any]], local: dict[str, Any]) -> None:
        """Render the root data card from the descriptor and build statistics.

        :param inventory: Source objects in row order.
        :param local: Local validation report.
        """
        spec = self.spec
        total_bytes = sum(int(i["bytes"]) for i in inventory)
        by_type = Counter(str(i["media_type"]) for i in inventory)
        type_rows = "\n".join(f"| `{k}` | {v:,} |" for k, v in sorted(by_type.items()))
        audio_formats: list[dict[str, Any]] = local["audio_formats"]
        formats = (
            "\n".join(
                f"- {f['count']:,} × {f['subtype']}, {f['sample_rate']} Hz, {f['channels']} ch"
                for f in audio_formats
            )
            or "- none (no libsndfile-decodable objects)"
        )
        archive_rows = (
            "\n".join(
                f"| `{a['archive']}` | {a['bytes']:,} | `{a['sha256']}` |"
                for a in self.archive_inventory()
            )
            or "| (no archives; see acquisition record) | | |"
        )
        commands = "\n".join(self.commands())
        sources = ", ".join(f"<{u}>" for u in spec.source_urls)
        text = f"""# {spec.title} — private/internal Lance publication

> **Access and use restriction:** {spec.usage}. License: {spec.license}. Do not make this R2 publication public or remove upstream attribution.

## Identity and provenance

{spec.summary}

- Canonical source: {sources}
- Immutable pin: {spec.pin}
- Acquisition record: `source/_acquisition/acquisition.json` (fetch timestamps, per-object hashes, response headers where applicable).
- Per-object SHA-256 inventory of all {len(inventory):,} source objects ({total_bytes:,} bytes): `metadata/source-objects.jsonl` and `metadata/source-objects.sha256`.

Pinned archives:

| Archive | Bytes | SHA-256 |
|---|---|---|
{archive_rows}

## Acquisition and conversion

Exact commands (also in `metadata/publication-commands.txt`), run from the synth-setter repository root:

```bash
{commands}
```

The generic conversion/validation tool and this corpus' descriptor are preserved under `metadata/scripts/`; tool versions are in `metadata/tool-versions.json`. Lance data storage version 2.2, native blob-v2 columns, batches bounded to 1 GiB of payload.

Inclusion/exclusion: {spec.inclusion}

## Layout

- `all.lance/`: the only physical dataset; one row per source object in sorted relative-path order, exact untranscoded bytes in a `lance.blob.v2` column.
- `source/`: byte-exact copies of the small ancillary source objects (documentation, metadata tables, licences) for human reading; they are also rows in `all.lance`.
- `metadata/`: inventories, build/validation reports, scripts, tool versions.
- `_COMPLETE.json`: uploaded strictly last; its absence means the prefix is incomplete and must not be consumed.

No train/validation/test split is asserted or invented. {spec.split_note}

## Schema

| Field | Type | Meaning |
|---|---|---|
| `source_row_index` | `int32` | Zero-based row in sorted source-path order. |
| `source_path` | `string` | Relative POSIX path of the source object within the extracted release. |
| `media_type` | `string` | MIME type inferred from the file extension. |
| `source_num_bytes` | `int64` | Exact source object byte length. |
| `source_sha256` | `string` | SHA-256 of the exact source bytes. |
| `audio_decodable` | `bool` | Whether libsndfile fully decoded the object. |
| `sample_rate` / `channels` / `frames` / `subtype` | nullable `int32` / `int32` / `int64` / `string` | libsndfile header fields for decodable audio; null otherwise. |
| `source_bytes` | `extension<lance.blob.v2>` | Exact, untranscoded source object bytes. |

Objects by media type:

| Media type | Rows |
|---|---|
{type_rows}

Decodable audio formats observed at build time:

{formats}

{spec.format_note}

## License, attribution, and citation

{spec.license_note}

Required citation:

> {spec.citation}

## Upload and privacy flags

Publication uses `rclone copy ... --checksum --immutable`; every rclone operation in the workflow passes `--checksum`. No public-read ACL, paid compute, RunPod, or Docker build is involved.

## Validation

Local build validation: {local["rows"]:,} rows, zero nulls in every non-nullable field, metadata equal to the source inventory row-by-row, {local["audio_decodable_rows"]:,} fully decoded audio objects. The exhaustive production-path validation (`pinned source -> local Lance writer -> rclone --checksum -> direct s3:// R2 reopen -> native blob reader -> libsndfile decoder`) re-reads every payload from R2 and compares byte length and SHA-256 with the source inventory; its report is `metadata/validation/remote-validation.json` and `_COMPLETE.json` is created only after it passes.

## Known source anomalies and caveats

{spec.caveats}
"""
        (self.publication / "README.md").write_text(text, encoding="utf-8")

    # ----------------------------------------------------------------- upload
    def upload(self) -> None:
        """Copy the publication to R2 immutably, never including the completion marker.

        :raises RuntimeError: The publication is not built or the marker already exists.
        """
        if not (self.publication / "README.md").exists():
            raise RuntimeError("publication is not built")
        if rclone("lsf", f"{self.rclone_prefix}/_COMPLETE.json", capture=True).strip():
            raise RuntimeError("completion marker already exists remotely")
        started = now()
        subprocess.run(  # noqa: S603 — args are literal strings
            [
                executable("rclone"),
                "copy",
                f"{self.publication}/",
                self.rclone_prefix,
                "--checksum",
                "--immutable",
                "--exclude",
                "_COMPLETE.json",
                "--transfers",
                "8",
                "--retries",
                "5",
                "--stats",
                "60s",
                "--stats-one-line",
            ],
            check=True,
        )
        write_json(
            self.state / "upload-report.json",
            {"started_at": started, "finished_at": now(), "flags": ["--checksum", "--immutable"]},
        )
        log.info("%s", json.dumps({"stage": "upload", "prefix": self.rclone_prefix}))

    # ----------------------------------------------------------------- verify
    def _verify_payloads(
        self, dataset: lance.LanceDataset, inventory: list[dict[str, Any]], pyrows: list[dict]
    ) -> tuple[int, int, Counter[tuple[object, ...]]]:
        """Re-read every blob from R2 and compare it with the inventory.

        :param dataset: Dataset opened directly from R2.
        :param inventory: Source objects in row order.
        :param pyrows: Metadata rows read from R2.
        :returns: Bytes compared, audio objects decoded, and decoded-format counts.
        :raises RuntimeError: Any payload or decode disagrees with the inventory.
        """
        formats: Counter[tuple[object, ...]] = Counter()
        payload_bytes = 0
        decoded = 0
        index = 0
        # Chunk blob reads by byte budget so multi-GB rows are read one at a time.
        while index < len(inventory):
            chunk = [index]
            budget = int(inventory[index]["bytes"])
            while (
                index + len(chunk) < len(inventory)
                and budget + int(inventory[index + len(chunk)]["bytes"]) < BATCH_BYTES
                and len(chunk) < 256
            ):
                budget += int(inventory[index + len(chunk)]["bytes"])
                chunk.append(index + len(chunk))
            for offset, blob in enumerate(dataset.take_blobs("source_bytes", indices=chunk)):
                row = index + offset
                payload = blob.read()
                payload_bytes += len(payload)
                if (
                    len(payload) != inventory[row]["bytes"]
                    or sha256_bytes(payload) != inventory[row]["sha256"]
                ):
                    raise RuntimeError(
                        f"remote payload mismatch row {row}: {inventory[row]['path']}"
                    )
                if pyrows[row]["audio_decodable"]:
                    info = audio_info(payload, Path(str(inventory[row]["path"])))
                    if not info["audio_decodable"] or info["frames"] != pyrows[row]["frames"]:
                        raise RuntimeError(f"remote decode mismatch row {row}")
                    decoded += 1
                    formats[(info["subtype"], info["sample_rate"], info["channels"])] += 1
            index += len(chunk)
            log.info("  verified through row %d (%d bytes)", index - 1, payload_bytes)
        return payload_bytes, decoded, formats

    def verify(self) -> None:
        """Validate the published prefix from R2, then upload the completion marker last.

        :raises RuntimeError: The marker already exists or any check fails.
        """
        from synth_setter.pipeline import r2_io  # noqa: PLC0415

        if rclone("lsf", f"{self.rclone_prefix}/_COMPLETE.json", capture=True).strip():
            raise RuntimeError("completion marker exists before exhaustive validation")
        r2_io.ensure_r2_env_loaded()
        storage_options = r2_io.r2_storage_options()
        inventory = self.load_inventory()
        dataset = lance.dataset(self.s3_all, storage_options=storage_options)
        if dataset.count_rows() != len(inventory):
            raise RuntimeError("direct-R2 row count mismatch")
        schema_metadata = {
            k.decode(): v.decode() for k, v in (dataset.schema.metadata or {}).items()
        }
        if schema_metadata.get("source_pin") != self.spec.pin:
            raise RuntimeError("remote schema source pin mismatch")
        table = dataset.scanner(columns=META_COLUMNS, batch_size=256).to_table()
        null_counts = {name: table.column(name).null_count for name in META_COLUMNS}
        pyrows = table.to_pylist()
        for index, (actual, expected) in enumerate(zip(pyrows, inventory, strict=True)):
            if (
                actual["source_path"] != expected["path"]
                or actual["source_sha256"] != expected["sha256"]
                or actual["source_num_bytes"] != expected["bytes"]
            ):
                raise RuntimeError(f"remote metadata mismatch row {index}")
        payload_bytes, decoded, formats = self._verify_payloads(dataset, inventory, pyrows)
        # Every uploaded object must hash-match the local publication (one-way check).
        check = subprocess.run(  # noqa: S603 — args are literal strings
            [
                executable("rclone"),
                "check",
                str(self.publication),
                self.rclone_prefix,
                "--checksum",
                "--one-way",
                "--exclude",
                "_COMPLETE.json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            raise RuntimeError(f"rclone check failed:\n{check.stderr}")
        remote_objects = rclone(
            "lsf", self.rclone_prefix, "--recursive", "--files-only", capture=True
        ).splitlines()
        report = {
            "verified_at": now(),
            "physical_dataset": self.s3_all,
            "rows": len(inventory),
            "payload_bytes_compared": payload_bytes,
            "audio_objects_fully_decoded_from_r2": decoded,
            "audio_formats": _format_rows(formats),
            "null_counts": null_counts,
            "fragments": len(dataset.get_fragments()),
            "rclone_check_one_way": "PASS",
            "remote_objects_before_marker": len(remote_objects),
            "status": "PASS",
        }
        validation_dir = self.publication / "metadata" / "validation"
        write_json(validation_dir / "remote-validation.json", report)
        rclone(
            "copy", str(validation_dir), f"{self.rclone_prefix}/metadata/validation", "--immutable"
        )
        card = self.publication / "README.md"
        marker = {
            "status": "COMPLETE",
            "dataset": self.spec.title,
            "r2_root": self.rclone_prefix,
            "physical_dataset": self.s3_all,
            "rows": len(inventory),
            "source_bytes": sum(int(i["bytes"]) for i in inventory),
            "source_pin": self.spec.pin,
            "access": self.spec.usage,
            "license": self.spec.license,
            "card": {"path": "README.md", "sha256": sha256_file(card)},
            "validation": {
                "path": "metadata/validation/remote-validation.json",
                "sha256": sha256_file(validation_dir / "remote-validation.json"),
                "status": "PASS",
            },
            "source_object_inventory_sha256": sha256_file(
                self.publication / "metadata" / "source-objects.jsonl"
            ),
            "upload_flags": ["--checksum", "--immutable"],
            "manifest_last": True,
            "objects_before_marker": len(remote_objects) + 1,
            "all_checks_passed_before_marker": True,
            "completed_at_utc": now(),
            "schema_version": 1,
        }
        write_json(self.publication / "_COMPLETE.json", marker)
        rclone(
            "copyto",
            str(self.publication / "_COMPLETE.json"),
            f"{self.rclone_prefix}/_COMPLETE.json",
            "--immutable",
        )
        shutil.copyfile(self.publication / "_COMPLETE.json", self.state / "_COMPLETE.json")
        shutil.copyfile(
            validation_dir / "remote-validation.json", self.state / "remote-validation.json"
        )
        log.info(
            "%s",
            json.dumps(
                {
                    "stage": "verify",
                    "status": "COMPLETE",
                    "rows": len(inventory),
                    "bytes": payload_bytes,
                }
            ),
        )


def main(argv: list[str]) -> None:
    """Run one stage for one corpus.

    :param argv: ``[stage, corpus]``.
    :raises SystemExit: The stage or corpus is unknown.
    """
    if len(argv) != 2 or argv[0] not in {"extract", "build", "upload", "verify"}:
        raise SystemExit(f"usage: {MODULE} <extract|build|upload|verify> <corpus>")
    stage, name = argv
    if name not in SPECS:
        raise SystemExit(f"unknown corpus {name!r}; known: {', '.join(sorted(SPECS))}")
    corpus = Corpus(SPECS[name])
    getattr(corpus, stage)()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
