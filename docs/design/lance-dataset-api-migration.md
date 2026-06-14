# Lance Dataset API migration

## Context

The Lance integration was ported from the HDF5 design: it uses the low-level
single-file `lance.file` API (`LanceFileWriter`/`LanceFileReader`), writes one
`*.lance` file per shard, and reads everything through an h5py-shaped adapter
after an rclone download. A review against the pinned library
(`pylance 7.0.0`, `pyarrow 24.0.0`) surfaced three things to fix:

1. **Single-file API, not the Dataset API.** The single-file format forgoes
   Lance's headline features (versioning, fragments, compaction, secondary /
   vector indices, predicate push-down). The h5py-shaped read *interface* is
   independent of which Lance API backs it.
2. **Cloud reads go local-first.** Sequential one-pass paths download a full
   shard via rclone before opening it, even though Lance reads object storage
   natively via `storage_options`.
3. **On-disk format version unpinned.** `LanceFileWriter` floats with the
   library default (currently `2.1`); a future pylance bump could silently
   change the on-disk format.

## Decisions

- **#1 → migrate to the Dataset API** (`lance.write_dataset` / `lance.dataset`).
  A shard / split becomes a Lance *dataset directory* (`<name>.lance/` with
  `data/`, `_versions/`, `_transactions/`), not a single file.
- **#2 → direct R2 streaming for sequential paths only.** finalize, stats, and
  validate read directly from R2 via `storage_options`; the training dataloader
  keeps rclone local-first (random per-batch reads across epochs must not hit
  the network).
- **#3 → pin `data_storage_version="2.1"`** on every `write_dataset` call.
  Behavior-preserving today (equals the default), but locks the on-disk format
  across pylance upgrades. (`"next"` panics at the Rust layer; `"2.1"` is the
  stable anchor.)

## Non-goals / cutover

- **No backward compatibility with single-file `.lance` shards.** The Lance
  path is experimental and every dataset is regenerable from its `DatasetSpec`;
  there is no production Lance data and no format-version marker today. This is
  a clean cutover — old single-file shards are simply regenerated.
- **No new versioning / index features in this change.** The migration *unlocks*
  them; actually using `merge`/compaction/vector indices is out of scope (YAGNI).
- **Worker writes stay local-then-upload.** Workers still write a shard locally,
  run the 4-check validation, then upload — now a directory upload. Only *reads*
  on the sequential paths move to direct R2.

## Design

### Format version + storage_options helpers

- `LANCE_DATA_STORAGE_VERSION = "2.1"` constant in `pipeline/data/lance_shard.py`,
  passed to every `write_dataset`.
- New `r2_io.r2_storage_options() -> dict[str, str]`: calls
  `ensure_r2_env_loaded()` then builds the object-store dict from env using the
  canonical keys from the official docs (S3-compatible stores require both
  `region` and `endpoint`):
  `{"access_key_id", "secret_access_key", "endpoint", "region": "auto"}`
  (`endpoint` is `RCLONE_CONFIG_R2_ENDPOINT`, the
  `https://<acct>.r2.cloudflarestorage.com` form). Whether R2 also needs
  `virtual_hosted_style_request="false"` / `allow_http` is confirmed by a live
  R2 round-trip (test gated behind `is_r2_reachable()`); the dict-from-env
  builder is unit-tested with monkeypatched env. Dataset URIs use the `s3://`
  scheme (via `r2_io.to_s3_uri`), which is how Lance selects the S3 backend.

### Write path (worker + finalize)

- Replace `write_lance_file(path, schema, batches)` with a dataset writer:
  `lance.write_dataset(batches, uri, schema=schema, mode="overwrite", data_storage_version=LANCE_DATA_STORAGE_VERSION, storage_options=...)`.
  `uri` is a local dir (worker) or an `s3://` URI (finalize split → writes the
  dataset directory straight to R2, removing the upload step).
- `writers.make_lance_dataset()` writes a local dataset directory; the existing
  schema/`tensor_array`/`record_batch_from_arrays` builders are unchanged.

### Read path — training (local-first, unchanged transport)

- `LanceShardFile` opens `lance.dataset(local_dir)` instead of
  `LanceFileReader`; caches `count_rows()` and per-column tensor shapes from
  `dataset.schema`. Keep the per-PID reopen for fork-safety.
- `LanceColumn.__getitem__`:
  - contiguous slice → `dataset.scanner(columns=[name], offset=start, limit=stop-start).to_table()`
  - stepped slice / fancy indices → `dataset.take(indices, columns=[name])`
  - `dataset.take` accepts **unsorted** indices, so the ascending-order
    constraint is relaxed (true fancy indexing); decode via
    `to_numpy_ndarray()` + writable-copy guard is unchanged.

### Read path — sequential (direct R2, #2)

- `iter_lance_column_rows(uri, column, *, storage_options=None)` →
  `lance.dataset(uri, storage_options=...).scanner(columns=[column]).to_batches()`.
- `stats.stream_stats_lance(...)` threads `storage_options` through to the above.
- `finalize_lance`: open each shard with `lance.dataset(s3_uri, storage_options)`,
  stream `.to_batches()`, and `write_dataset` the split directly to its `s3://`
  URI. Drops the per-shard download and the split upload.
- `validate_shard._validate_lance_shard`: open via `lance.dataset(s3_uri, storage_options)`, validate `schema` (fixed-shape tensor types) + `count_rows`.

### R2 layout

- `shard-XXXXXX.lance` and `<split>.lance` are now directory prefixes, not
  objects, keyed off `OutputFormat.is_directory`. The worker uploads each shard
  tree with `r2_io.upload_dir` and skip-probes the committed dataset via
  `r2_io.r2_directory_exists("<shard>/_versions")` (a crashed render leaves
  orphan `data/` files with no manifest, so probing any object would strand an
  unreadable shard); training download uses `download_dir_no_overwrite` (already
  directory-recursive). `dataset.complete` marker and `stats.npz` are unchanged.
  `OutputFormat.from_extension` still matches on the `.lance` name suffix of the
  directory.

## Affected files

| Concern                                            | File                                                                 |
| -------------------------------------------------- | -------------------------------------------------------------------- |
| version const + dataset writer + sequential reader | `pipeline/data/lance_shard.py`                                       |
| storage_options builder                            | `pipeline/r2_io.py`                                                  |
| worker shard writer                                | `data/vst/writers.py`                                                |
| training h5py adapter                              | `data/lance_datamodule.py`                                           |
| finalize (direct R2 read + write)                  | `cli/finalize_dataset.py`                                            |
| stats streaming                                    | `pipeline/data/stats.py`                                             |
| shard validation                                   | `pipeline/ci/validate_shard.py`                                      |
| R2 URI docstrings / upload-dir wiring              | `pipeline/schemas/r2_location.py`                                    |
| docs                                               | `docs/design/data-pipeline.md`, `pipeline/CLAUDE.md` validation tier |

## Test strategy

- Local round-trip: `write_dataset` → `lance.dataset` preserves fixed-shape
  tensor values, row order, and the `synth_setter.shard_metadata` schema key.
- `LanceColumn`: contiguous slice, stepped slice, unsorted fancy indices,
  writable-copy guard.
- `r2_storage_options()` builds the expected dict from monkeypatched env.
- `iter_lance_column_rows` / validate over a local dataset directory.
- Direct-R2 round-trips (write split to `s3://`, read back) gated behind
  `is_r2_reachable()`, matching existing R2 test conventions.
- `data_storage_version` pin asserted via the written dataset's reported
  storage version.

## Risks

- **object-store key names for R2.** Resolved by a live R2 round-trip test; the
  builder is the single place to adjust.
- **Fork-safety of `LanceDataset`.** Mitigated by preserving the per-PID reopen.
- **Large blast radius.** Sequenced into phases below; can be split into a PR
  chain (write path → read path → cloud/finalize) if preferred.

## Phases

1. `LANCE_DATA_STORAGE_VERSION` + `r2_storage_options()` (+ tests).
2. Write path → `write_dataset` in `lance_shard` + `writers` (+ round-trip, #3).
3. Training read path → `lance.dataset` adapter, relax sorted-index (+ tests).
4. Sequential reads → direct R2 in `stats` / `validate` / `finalize`, write
   split straight to R2 (+ R2-gated tests).
5. R2 layout wiring: `upload_dir` / `download_dir_no_overwrite`, `r2_location`
   docstrings, dispatch check.
6. Docs: this file, `data-pipeline.md` rationale, `pipeline/CLAUDE.md` tier note.
7. Cleanup: remove dead `lance.file` code; `/code-health` + `/simplify`.
