# Legacy Output Formats: HDF5 & WebDataset

> **Status**: Legacy — maintained for coverage, not active development
> **Primary format**: Lance — see [data-pipeline.md §7.10](../data-pipeline.md#710-output-format-lance-primary-hdf5--webdataset-legacy)
> **Tracking**: [#1779](https://github.com/tinaudio/synth-setter/issues/1779) (make Lance the primary/default format everywhere)

______________________________________________________________________

HDF5 and WebDataset remain selectable via `output_format` and keep coverage-level tests, but they are second-class: not the default, and not the formats new pipeline features target first. This document preserves their format-specific detail — both because the code paths still exist and because **existing R2 datasets are laid out in these formats** and someone will need this spec to interpret them.

Everything format-agnostic — the staging protocol, lifecycle markers, `.valid` commit points, winner selection, reconciliation — lives in [data-pipeline.md](../data-pipeline.md) and applies to these formats unchanged. This document covers only what is HDF5- or WDS-specific.

## HDF5 (`output_format: hdf5`)

Finalize output: virtual datasets (`train.h5`, `val.h5`, `test.h5`) that reference promoted shards. Good for local single-GPU training where the full dataset is downloaded to the training machine. Random access, fast local I/O.

**Staging and promotion:** workers upload per-attempt shards to `metadata/workers/shards/shard-{id}/{worker}-{attempt}.h5`; finalize structural-checks the selected attempt (opens with `h5py`, verifies expected datasets and shapes) and **promotes** it — a copy to the canonical `data/shards/shard-{id}.h5`, recorded with a `.promoted` marker. Resharding into split virtual datasets happens on the finalize machine, which is why HDF5 finalize downloads or rewrites all selected shards ([data-pipeline.md §12](../data-pipeline.md#12-open-questions-risks--limitations) — the single-machine bottleneck Lance avoids).

**Resumability:** `make_hdf5_dataset` is resumable — a partially-written file picks up at the first all-zero row, so a crashed worker can re-run with the same render config and only the missing tail is regenerated, except under `render.param_sample_cadence="shard"`, where a partially-written shard is re-rendered from row 0 (a mid-shard resume can't preserve the one-patch-per-shard invariant). `make_wds_dataset` and the Lance fragment writer are not resumable; a crashed wds or lance worker re-renders the whole shard attempt.

**Why HDF5 is insufficient for multi-GPU training:** HDF5 is random-access oriented. Multi-GPU DataLoaders need to stream shards sequentially without coordinating seeks across workers. Streaming HDF5 virtual datasets from R2 during training creates heavy seek traffic and GPU idle time. Downloading the full dataset to every training node wastes storage and time at scale.

**Copying an existing dataset:** when `copy_dataset_root_uri` is set on the spec, generation re-renders the parameters of an existing dataset instead of sampling fresh ones. The root URI may be a bare path, `file://` URI, or `r2://` URI. The launcher forwards `--copy_dataset_root_uri` to the renderer subprocess, which resolves `<copy_dataset_root_uri>/<shard.filename>` to a local file (downloading it from R2 to a tempfile when the root is an `r2://` URI), reads the source shard's `param_array`, decodes each row into fixed synth/note params via `fixed_params_from_dataset` (`param_spec.decode`), and renders those. This is hdf5-only — the source is read as an HDF5 `param_array` of the same shard filename, so a non-hdf5 output with `--copy_dataset_root_uri` raises `SystemExit`. The source must share the target's `render.param_spec_name` (same encoding width) and have row count equal to `samples_per_shard`. Fixed params are indexed by absolute row, so resume re-renders only the missing tail from the matching source rows. Before any render, `generate` preflights the copy against the source's persisted spec: it loads `<copy_dataset_root_uri>/input_spec.json` (which sits beside the shards at the dataset prefix root) and asserts the source matches the target on every copy-relevant value — `param_spec_name`, `samples_per_shard`, `train_val_test_sizes`, and the full shard-filename set (source of truth: `DatasetSpec.validate_copy_source`) — failing once at launch (with all mismatches aggregated) rather than per-shard mid-render. A missing source `input_spec.json` is itself a launch error. `input_spec.json` files materialized before the dataset-copy source became a root URI still load: a `mode="before"` shim promotes the pre-rename flat `copy_dataset_root: …` and the older nested `datasetsrc: {copy_dataset_root: …}` shape (and drops `datasetsrc: null`) to `copy_dataset_root_uri`.

> `copy_dataset_root_uri` is slated for deletion in favor of generating a dataset from an explicit param list; no hdf5→lance copy bridge will be built.

## WebDataset (`output_format: wds`)

Finalize output: sequential `.tar` archives (`train-{shard}.tar`, `val-{shard}.tar`, `test-{shard}.tar`) optimized for streaming. Good for multi-GPU training where streaming from R2 avoids downloading the full dataset to every node.

**Shard structure:** each `.tar` shard groups rows into per-batch tar entries. The tar key is the batch's first logical row index zero-padded to 8 digits (`f"{start_idx:08d}"`) and advances by `samples_per_render_batch`; each `<key>.<field>.npy` member holds the whole batch stacked along axis 0 — not one sample per file. The per-row field names come from `synth_setter.data.vst.shapes.DATASET_FIELD_NAMES` (`audio`, `mel_spec`, `param_array`):

```
train-000000.tar
├── 00000000.audio.npy      # shape (samples_per_render_batch, ...)
├── 00000000.mel_spec.npy
├── 00000000.param_array.npy
├── 00000064.audio.npy      # next batch — key advances by samples_per_render_batch
├── 00000064.mel_spec.npy
├── 00000064.param_array.npy
├── ...
└── metadata.json          # ShardMetadata sidecar — see src/synth_setter/pipeline/schemas/shard_metadata.py
```

Shard count is tuned for GPU worker count — one shard per GPU worker per epoch is ideal; exact sizing depends on batch size and network bandwidth.

**Training integration:** the `webdataset` Python library provides streaming, shuffling, batching, and multi-worker support out of the box. R2 free egress makes streaming from object storage practical. Each GPU worker gets a disjoint subset of `.tar` shards — no coordination needed. Training code must use WebDataset's built-in shuffle (or `shardshuffle`) — finalize writes shards in deterministic order for reproducibility; shuffling is the training loader's responsibility.

## Sample Type

A typed container for individual training samples during finalize's transcode step (HDF5 → WebDataset). This is a `dataclass`, not a Pydantic model — the data is already validated NumPy arrays at this point, so Pydantic's serialization validation is unnecessary overhead (see the validation-boundaries table in [data-pipeline.md §14.5](../data-pipeline.md#145-validation-boundaries)).

```python
@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: int
    audio: np.ndarray       # shape: (channels, samples)
    mel_spec: np.ndarray    # shape: (mels, frames)
    params: np.ndarray      # shape: (num_params,)
```

The `Sample` type ensures the transcode loop reads and writes the correct fields — a bug that drops `mel_spec` or swaps `audio` and `params` is caught by type hints rather than silently producing a broken `.tar` archive. `frozen=True` prevents accidental mutation during transcoding.
