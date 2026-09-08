# SLAP retrieval datasets

SLAP retrieval uses a separate Lance dataset for each source split. Unlike
embedding-column augmentation, which preserves one row per training example,
this export creates two searchable entries per source row without copying
waveforms, spectrograms, or parameter arrays.

## Row contract

| Column               | Meaning                                                               |
| -------------------- | --------------------------------------------------------------------- |
| `row_uuid`           | Persisted source-row UUID, shared by the two retrieval entries        |
| `is_param_embedding` | `true` for the parameter projection; `false` for the audio projection |
| `slap`               | Normalized float32 EMA projection in the shared SLAP space            |

The retrieval key is `(row_uuid, is_param_embedding)`. `row_uuid` alone is
unique in the source, not in the retrieval table. An audio-content UUID is
not a substitute: different source rows can contain identical audio.

Both modalities use the same checkpoint. The export takes projected vectors,
not backbone representations or online predictor outputs. Audio inputs use the
checkpoint's configured input kind; stored mel spectrograms are not recomputed
with a different frontend.

## Provenance and discovery

Each retrieval dataset identifies its source URI, split, exact Lance version
and transaction identity, checkpoint content identity, resolved model
configuration, projection policy, vector dimension, and row counts.

The source receives a discovery pointer only after its retrieval dataset and
any requested index are complete. That pointer records the output URI/version
and the source version used to compute it. Updating source metadata creates a
later source version; that later version is not the input snapshot.

Source UUIDs must be persisted before pinning the export snapshot. Historical
snapshots without UUIDs cannot be retroactively changed. Keep the referenced
source version available: the retrieval table contains no audio or parameters
from which to reconstruct a deleted source snapshot.

Publication is per split, not an atomic transaction across all splits or both
datasets. A failed pointer update can leave a complete retrieval dataset with
no source-side discovery pointer. Matching reruns should repair publication,
not duplicate rows; a conflicting export must not silently overwrite the
existing table.

## Retrieval

Search the `slap` column using a projection from the same checkpoint and
normalization policy. Join returned UUIDs to the recorded source snapshot to
recover parameters or audio. Deduplicate by UUID when the caller wants one
result per source example.

Without a modality filter, parameter and audio entries compete for the same
nearest-neighbor result set. Use `is_param_embedding` when only one modality is
wanted. Train, validation, and test exports remain separate; do not combine
them for training-time retrieval unless cross-split access is deliberate.

## Running an export

```bash
python -m synth_setter.cli.export_slap \
  model=slap_ast_audio_mlp_param \
  synth=surge_xt \
  source_root_uri=r2://bucket/source \
  output_root_uri=r2://bucket/slap \
  ckpt_path=/path/to/model.ckpt \
  source_versions.train=12 \
  source_versions.val=7 \
  source_versions.test=5
```

Use the exact source versions you intend to export. If a selected head lacks
UUIDs, the exporter adds them and records both the requested version and the
resulting UUID-bearing input snapshot. Reuse the same request on reruns.

To select only train, also remove the unused version entries:

```bash
python -m synth_setter.cli.export_slap \
  source_root_uri=/data/source output_root_uri=/data/slap \
  ckpt_path=/path/to/model.ckpt \
  'splits=[train]' source_versions.train=12 \
  '~source_versions.val' '~source_versions.test'
```

`device=null` selects CUDA when available, otherwise CPU; override with
`device=cpu` or `device=cuda`. `batch_size` bounds inference batches, but
UUID uniqueness validation keeps an in-memory set proportional to the number
of source rows.

Set `build_index=true` to build an IVF-PQ index. `metric` defaults to `cosine`;
`num_partitions` can be specified, and `num_sub_vectors` must divide the vector
width. Index training also requires enough rows for Lance's codebook training;
small exports can omit the index and use exact search. An indexing failure
retains an incomplete output without publishing a source pointer.

Checkpoint paths are local files in this first implementation. Dataset roots
accept local paths, file URIs, and configured R2 URIs; S3-form URIs use the R2
backend, not arbitrary AWS credentials.

## Checkpoint configuration

The initial exporter requires the checkpoint and its matching Hydra model
configuration explicitly. The model configuration includes architecture
settings and resolved synth parameter dimensions; choosing the same model
name with different training overrides is not sufficient.

Self-describing checkpoint support is tracked in #3207. After that and the
exporter merge, #3208 will switch export model construction to the shared
checkpoint loader and register the `synth-setter-export-slap` console command.
Until then, the exporter is invoked as a Python module.

Only load trusted checkpoints and model configurations. Hydra model targets
instantiate Python code; these inputs are not a safe format for untrusted
uploads.
