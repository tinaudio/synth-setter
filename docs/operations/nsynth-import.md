# Import NSynth into Lance

`synth-setter-import-nsynth` converts the three official extracted NSynth splits
into local Lance datasets, uploads an immutable copy to R2, and verifies a fresh
full-prefix download against the original extracts.

NSynth is distributed under the [Creative Commons Attribution 4.0
license](https://creativecommons.org/licenses/by/4.0/). The importer records
that license and the [official NSynth dataset
page](https://magenta.tensorflow.org/datasets/nsynth) in every Lance schema and
in the root manifest.

## Prepare the source

Extract the official archives beneath one source root without renaming their
contents:

```text
<SOURCE_ROOT>/
├── nsynth-train/
│   ├── examples.json
│   └── audio/<note_str>.wav
├── nsynth-valid/
│   ├── examples.json
│   └── audio/<note_str>.wav
└── nsynth-test/
    ├── examples.json
    └── audio/<note_str>.wav
```

The command requires the official split counts:

| Split   | Examples |
| ------- | -------: |
| `train` |  289,205 |
| `valid` |   12,678 |
| `test`  |    4,096 |

Configure the `r2:` rclone remote through the normal synth-setter object-storage
environment before ingesting. The default immutable destination is
`r2://experiments/third_party/NSynth`.

## Ingest and upload

Choose an output path that does not exist:

```bash
uv run synth-setter-import-nsynth ingest \
  /data/nsynth-extracted \
  /data/nsynth-lance
```

Use `--batch-size N` to bound the number of metadata rows and WAV payloads held
for each streamed Arrow batch. The default is 128.

The command validates strict official metadata, top-level key equality,
10 binary quality flags, safe note names, one WAV per row, orphan WAVs, and
split counts. It writes into a sibling partial directory, writes
`manifest.json` only after all datasets and byte-identical JSON sidecars, then
atomically renames the completed local root.

The resulting layout is:

```text
<OUTPUT_ROOT>/
├── train.lance/
├── valid.lance/
├── test.lance/
├── metadata/
│   ├── nsynth-train.examples.json
│   ├── nsynth-valid.examples.json
│   └── nsynth-test.examples.json
└── manifest.json
```

Each Lance row stores every typed official metadata field plus the WAV byte
size, SHA-256 digest, and native Lance Blob-v2 payload. Datasets use Lance data
storage version 2.2 and a 32 GiB data-file cap.

Uploads use `rclone copy --checksum --immutable`. Datasets and JSON sidecars
land first; `manifest.json` is the remote completion marker and lands last. A
changed existing object fails the ingest. The importer never purges or replaces
remote objects, so inspect and resolve a collision instead of deleting a prefix
from this command.

Use `--remote-root r2://bucket/prefix` to target another immutable prefix.

## Verify a fresh download

Choose a download path that does not exist:

```bash
uv run synth-setter-import-nsynth verify \
  /data/nsynth-extracted \
  /data/nsynth-verify-download
```

Verification downloads the complete remote prefix through rclone before opening
Lance. It then checks the strict manifest, exact schema and counts, source and
license metadata, byte-identical JSON sidecars, every typed metadata value,
every WAV byte, and each stored size and SHA-256 digest. Metadata and Blob-v2
reads are batch-bounded; no split is materialized as one table.

Success ends with a summary containing `0 mismatches`. Any mismatch exits
non-zero and identifies the split, row, or artifact that differed.

## Tiny local-backend exercise

The count override exists for deterministic tests and operator smoke checks; do
not use it for an official import:

```bash
export RCLONE_CONFIG_R2_TYPE=local
uv run synth-setter-import-nsynth ingest \
  /tmp/nsynth-tiny \
  /tmp/nsynth-tiny-output \
  --batch-size 1 \
  --expected-counts train=1,valid=1,test=1
uv run synth-setter-import-nsynth verify \
  /tmp/nsynth-tiny \
  /tmp/nsynth-tiny-download \
  --batch-size 1 \
  --expected-counts train=1,valid=1,test=1
```

With rclone's local backend, the default URI materializes beneath the current
working directory as `experiments/third_party/NSynth`.
