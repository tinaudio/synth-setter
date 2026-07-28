# BBC Sound Effects Internet Archive import

This runbook imports the Internet Archive item
[`BBCSoundEffectsComplete`](https://archive.org/details/BBCSoundEffectsComplete)
into immutable Lance Blob-v2 shards. The authoritative snapshot at the time this
workflow was added contains **16,008 WAV files** totaling **304,723,231,452
bytes**, with an IA MD5 for every WAV.

The workflow never deletes source files or remote objects. It publishes
`manifest.json` only after a full remote download has passed file, metadata,
and audio-byte verification.

## Prerequisites

Install the repository environment so the packaged command, `ia`, Lance,
SoundFile, and rclone are available:

```bash
uv sync
```

Configure the rclone `r2:` remote for Cloudflare R2. Upload and verification
accept `--remote-uri`; the default destination is:

```text
r2://experiments/third_party/BBCSoundEffectsComplete
```

Budget for roughly 305 GB of source WAVs, a similar-sized local release, and a
second similar-sized verification download. Lance audio-payload memory is
bounded by the largest WAV rather than the collection size, but all three trees
coexist during the full gate.

## 1. Download and snapshot IA metadata

`SOURCE_ROOT` is the parent under which Internet Archive creates the item
directory:

```bash
synth-setter-bbc-sfx download /data/bbc-sfx-source
```

The command executes the following argv directly, without a shell:

```text
ia download BBCSoundEffectsComplete --glob=*.wav --retries 10 --checksum
```

It preserves IA's layout under:

```text
/data/bbc-sfx-source/BBCSoundEffectsComplete/sounds/...
```

It fetches `ia metadata BBCSoundEffectsComplete` with bounded attempts, validates
the response, then saves it atomically as
`BBCSoundEffectsComplete/ia-metadata.json`. An existing valid snapshot is kept,
which makes repeated downloads resumable without silently changing the
inventory contract.

Before conversion, compare the live snapshot with the expected authoritative
counts:

```bash
uv run python - <<'PY'
from pathlib import Path
from synth_setter.pipeline.data.bbc_sfx import load_ia_metadata

snapshot = load_ia_metadata(
    Path("/data/bbc-sfx-source/BBCSoundEffectsComplete/ia-metadata.json")
)
print("rows:", len(snapshot.files))
print("bytes:", sum(row.size for row in snapshot.files))
PY
```

Expected output for the current snapshot:

```text
rows: 16008
bytes: 304723231452
```

Stop if IA has changed. Preserve the downloaded snapshot and review the new
inventory before creating a release.

## 2. Convert to Lance

Choose a `RELEASE_ROOT` with no completed release. An existing sibling
`RELEASE_ROOT.partial` is treated as resumable work:

```bash
synth-setter-bbc-sfx convert \
  /data/bbc-sfx-source \
  /data/bbc-sfx-release
```

Conversion rejects missing, extra, size-mismatched, or MD5-mismatched WAVs. It
sorts paths byte-for-byte and partitions them at approximately 16 GiB of source
bytes without splitting a file. Every WAV is revalidated immediately before
its captured bytes become metadata and blob data. Each shard is written to a
partial directory, then atomically renamed. A resumed conversion validates the
pinned metadata and every completed shard against the source before reusing
them; an interrupted
per-shard partial directory is rebuilt.

Each Lance row contains:

- exact relative `sounds/...` path and WAV bytes in a `lance.blob.v2` column;
- IA size, MD5, and available SHA1, CRC32, mtime, and length values; and
- SoundFile sample rate, channels, frames, format, subtype, endian, and duration.

The writer pins the repository's Lance data storage version `2.2` and 32 GiB
per-data-file cap. `release-manifest.json` records the IA snapshot SHA256,
release totals, shard row/byte ranges, and every releasable file's SHA256. Keep a
copy of this local manifest outside `RELEASE_ROOT` if the release tree will be
removed after upload:

```bash
cp /data/bbc-sfx-release/release-manifest.json \
  /data/bbc-sfx-release-manifest.json
```

For tiny development fixtures only, override the partition target with
`--shard-target-bytes`.

## 3. Upload immutable release files

```bash
synth-setter-bbc-sfx upload /data/bbc-sfx-release
```

Uploads use real rclone operations with `--checksum`, `--immutable`, and
no-overwrite behavior. Existing matching objects make retries idempotent;
existing different bytes fail. The command does not purge the destination,
excludes local `release-manifest.json`, and rejects any release inventory that
contains the reserved remote completion path `manifest.json`.

Confirm completion is still absent:

```bash
rclone lsf r2:experiments/third_party/BBCSoundEffectsComplete/manifest.json
```

A not-found result is expected before verification.

## 4. Full-download verification and completion publication

Choose a `DOWNLOAD_ROOT` and pass the retained local manifest. A prior partial
download is resumed only when its existing files match the remote bytes:

```bash
synth-setter-bbc-sfx verify \
  /data/bbc-sfx-source \
  /data/bbc-sfx-verification-download \
  --release-manifest /data/bbc-sfx-release-manifest.json
```

Verification performs all of the following before publication:

1. downloads the entire remote prefix with checksum-enabled immutable rclone;
2. checks the exact release file set, sizes, and SHA256 values;
3. streams bounded Lance metadata batches and compares every typed row with IA
   inventory plus fresh `soundfile.info` results from the source WAV;
4. compares each Blob-v2 stream byte-for-byte with its original WAV while
   recomputing the IA MD5; and
5. requires the manifest row and byte totals to match the source inventory.

After each shard passes, verification atomically records a local progress
checkpoint keyed by the retained manifest hash. A retry rechecks the complete
remote file inventory, then resumes after the verified shard prefix; the
checkpoint is removed before completion publication.

Only a zero-mismatch run uploads the retained local manifest as remote
`manifest.json`. Successful output ends with totals in this form:

```text
verified 16008 rows and 304723231452 bytes; 0 mismatches
```

Do not manually upload `manifest.json`. A failed verification deliberately
leaves the remote prefix incomplete so consumers cannot mistake it for a
verified release.
