# Guide: Browse Lance datasets in SmooSense

> **Status**: Stable
> **Last Updated**: 2026-06-13
> **Source**: [`src/synth_setter/cli/browse_dataset.py`](../../src/synth_setter/cli/browse_dataset.py), [`src/synth_setter/pipeline/data/lance_browse.py`](../../src/synth_setter/pipeline/data/lance_browse.py)

______________________________________________________________________

## What it is

`synth-setter-browse-dataset` opens the pipeline's Lance shards and finalized
splits in [SmooSense](https://smoosense.ai/docs/database-browser/), a desktop-class
GUI for Lance data — row/column inspection, instant visualizations, metadata
and index views, and version comparison.

The pipeline writes the Lance **file** format (one `.lance` file per shard or
split; see [§7.10 of the data-pipeline design](../design/data-pipeline.md#710-output-format-hdf5-webdataset-or-lance)).
SmooSense's `sense db` browses Lance **dataset** directories instead. This
command bridges the two: it re-materializes each source as a Lance dataset
under one browse-db directory — preserving the embedded `ShardMetadata` on the
dataset schema so the browser's metadata view shows the render parameters —
then launches `sense db` on that directory.

## When to use it

- Eyeballing a finalized split (`train.lance` / `val.lance` / `test.lance`) or a
  single shard before training.
- Debugging a suspect dataset — slice rows, scan columns, read the embedded
  render metadata.
- Comparing what landed in R2 against what you expected.

## Prerequisites

SmooSense is **not** a project dependency: it requires Python ≥3.11, above this
project's ≥3.10 floor. Install it once as an isolated tool:

```bash
uv tool install -U smoosense
```

This puts a `sense` binary on your PATH in its own environment. (`uv tool install` is SmooSense's own recommended install path.)

## Usage

Browse a finalized split straight from R2:

```bash
synth-setter-browse-dataset r2://<bucket>/<run-prefix>/train.lance
```

Browse several sources together (each becomes a `<stem>.lance` table directory
— here `train.lance`, `val.lance`, `test.lance`):

```bash
synth-setter-browse-dataset \
  r2://<bucket>/<run-prefix>/train.lance \
  r2://<bucket>/<run-prefix>/val.lance \
  r2://<bucket>/<run-prefix>/test.lance
```

Browse a local shard:

```bash
synth-setter-browse-dataset ./shard-000000.lance
```

### Options

- `--db-dir DIR` — write the browsable datasets under `DIR` (created if
  missing). Defaults to a fresh temp directory, printed on exit.
- `--launch / --no-launch` — whether to launch `sense db` after exporting.
  `--no-launch` only writes the datasets and prints the `sense db` command to
  run later (useful on a headless box, or when SmooSense isn't installed).

`r2://` sources are downloaded to a scratch directory before export and removed
afterward; the exported browse-db persists so `sense db` can read it.

## How it works

1. Each source is resolved to a local `.lance` file (downloading `r2://` URIs
   via `r2_io`).
2. `build_browse_db` exports each into `<db-dir>/<stem>.lance` as a Lance
   dataset (`export_shard_to_dataset` → `lance.write_dataset`), carrying the
   schema's `ShardMetadata` along.
3. Unless `--no-launch`, `sense db <db-dir>` opens the browser.
