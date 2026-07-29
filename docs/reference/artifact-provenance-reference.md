# W&B Artifact & Provenance Reference

> **Last Updated**: 2026-07-11
> **Tracking**: #1565, #122, #1572

Companion to [storage-provenance-spec.md](../design/storage-provenance-spec.md). The spec is authoritative for names, paths, and conventions; this reference shows the **landed code patterns** behind them — how each artifact is built, how lineage edges are recorded, and how a logged artifact is resolved back to a local checkpoint.

______________________________________________________________________

## Overview

Every run logs the artifacts it produces and consumes the artifacts it reads, so W&B reconstructs a lineage DAG (spec §5) across data generation → training → evaluation → promotion. Three rules hold everywhere:

- Outputs are logged with `run.log_artifact(...)`; inputs are linked with `run.use_artifact(...)` (only `use_artifact` — not `api.artifact()` — creates a lineage edge).
- R2 objects are attached as `s3://` references with `checksum=False`; the URI records lineage, not a content hash (R2's custom S3 endpoint is unreachable by W&B's default reference handler).
- Artifact logging and lineage edges are **best-effort**: a W&B failure warns and is swallowed so it never aborts a run whose real work already succeeded, and a run with no `WandbLogger` is a silent no-op.

______________________________________________________________________

## 1. Artifact Catalog

| Type           | Name pattern               | Built by                                         | R2 reference                                |
| -------------- | -------------------------- | ------------------------------------------------ | ------------------------------------------- |
| `dataset`      | `data-{dataset_config_id}` | `build_dataset_artifact` (`finalize_dataset.py`) | split `.lance` / shard prefix + `stats.npz` |
| `model`        | `model-{train_config_id}`  | `build_model_artifact` (`train.py`)              | best-checkpoint `model.ckpt` (see §3)       |
| `eval-results` | `eval-{eval_config_id}`    | `build_eval_results_artifact` (`eval.py`)        | output-dir prefix                           |

The `{*_config_id}` is the config filename stem, resolved via `resolve_run_config_id(cfg)` for train/eval and `spec.task_name` for datasets. The artifact name carries the config id, not the `{*_wandb_run_id}`; W&B auto-versions (`:v0`, `:v1`, …) so re-running the same config yields the next version, and the producing run — whose id is pinned via `pin_wandb_run_id` — is what W&B links the artifact to for lineage. (The builders below do **not** copy the run id into `artifact.metadata`; spec §4 reserves that, but it is not yet wired.)

______________________________________________________________________

## 2. Reference Pattern

Each builder returns an **unlogged** `wandb.Artifact` with its R2 outputs attached as `s3://` references:

```python
artifact = wandb.Artifact(name=f"data-{spec.task_name}", type="dataset", metadata={...})
for r2_uri in _finalized_reference_uris(spec):
    artifact.add_reference(_r2_to_s3_uri(r2_uri), checksum=False)
```

`checksum=False` is mandatory: W&B cannot reach R2's custom S3 endpoint to hash the object, so a checksummed reference would fail. The reference records *where the bytes live* for lineage, not their integrity.

______________________________________________________________________

## 3. Metadata Convention (spec §6)

`artifact.metadata` holds properties of the artifact itself, never run hyperparameters (those go in `wandb.config`). Final metrics live in `wandb.summary`; the one exception is `eval-results`, which also copies a small **scalar summary** of its metrics into `artifact.metadata` (via `_eval_summary_metrics`) so a result set can be filtered without opening each run.

| Artifact       | Metadata keys                                                          |
| -------------- | ---------------------------------------------------------------------- |
| `dataset`      | `shard_count`, `n_samples`, `git_sha`                                  |
| `model`        | `git_sha`, plus `_checkpoint_metadata` keys when a checkpoint uploaded |
| `eval-results` | scalar summary metrics (`_eval_summary_metrics`) + `git_sha`           |

The `model` artifact always carries `git_sha`; when the best checkpoint uploads, `_checkpoint_metadata` merges in the keys that identify it (§4), since the `s3://` reference itself renders as a 0-byte entry. At train end (rank-zero, with a `WandbLogger`) the best checkpoint uploads to a derived `r2://{r2.bucket}/checkpoints/{train_config_id}/model.ckpt` URI — overridable via `training.upload_checkpoints_uri` — and attaches to the artifact as an `s3://` reference ([#1572](https://github.com/tinaudio/synth-setter/pull/1572), closing [#92](https://github.com/tinaudio/synth-setter/issues/92)). It degrades to a **lineage-only** artifact (no reference) when no checkpoint was written (`fast_dev_run`), R2 is unreachable (local / CI), the upload fails, or the fingerprint guard refuses an architecture-incompatible overwrite of the slot (`checkpoint_fingerprint.py`, [#2588](https://github.com/tinaudio/synth-setter/issues/2588)), so a completed run is never aborted by checkpoint persistence. The fixed `model.ckpt` basename lets the `${wandb:…}` resolver (§5) select the checkpoint unambiguously.

______________________________________________________________________

## 4. Lineage DAG

```
dataset config
  → [data-generation run] → dataset artifact
                               ├→ [training run] → model artifact
                               │                      │
eval dataset artifact ─────────┴→ [evaluation run] ←──┘
                                        │
                                   eval-results artifact
                                        │
                                  [promote workflow] → GitHub Release
```

The data-generation, training, and evaluation edges are landed; the `[promote workflow] → GitHub Release` tail is the spec's target shape — that workflow is not implemented yet ([#1566](https://github.com/tinaudio/synth-setter/issues/1566)).

**Producing an output** — the `_log_*_artifact` helpers iterate the loggers and log on each `WandbLogger`:

```python
for lg in loggers:
    if isinstance(lg, WandbLogger):
        lg.experiment.log_artifact(build_model_artifact(cfg, ckpt_uri, ckpt_metadata))
```

`ckpt_metadata` (`_checkpoint_metadata`) carries `ckpt_uri`, `ckpt_bytes`, `epoch`, `global_step`, `monitor`, and `monitor_score` — W&B cannot reach R2's custom endpoint, so the `checksum=False` reference renders as a 0-byte entry and this metadata is the only in-UI way to identify the checkpoint it points at ([#2424](https://github.com/tinaudio/synth-setter/issues/2424)).

**Consuming an input** — `record_input_lineage` (`utils/logging_utils.py`) records each `(name, alias)` edge via `use_input_artifacts` (rank-zero-gated, so a DDP run records each edge once) and marks the run when any consumed input has no edge:

```python
record_input_lineage(loggers, *_consumed_artifact_refs(cfg))  # ([("data-diva-v1", "diva-v1-20260520T000000000Z")], [])
```

Dataset edges are discovered from `input_spec.json` under `datamodule.download_dataset_root_uri` when configured, otherwise under `datamodule.dataset_root`. Its validated `task_name` and `run_id` resolve to the immutable `data-{task_name}:{run_id}` alias.

A root without a readable frozen spec, or a ref W&B rejects (a dataset finalized before the `run_id` alias landed in [#1881](https://github.com/tinaudio/synth-setter/issues/1881) has no such alias), still leaves the run usable — but no longer silently. `mark_lineage_incomplete` writes `summary.lineage_incomplete = true`, `summary.lineage_missing` naming each input with no edge, and the `lineage-incomplete` run tag ([#2424](https://github.com/tinaudio/synth-setter/issues/2424)). To audit, filter runs on that tag.

______________________________________________________________________

## 5. Resolving an Artifact to a Checkpoint

The `${wandb:<ref>}` OmegaConf resolver (`utils/utils.py`, registered in `register_resolvers`) turns a model-artifact ref into a local checkpoint path. To resume, point `ckpt_path` at a `${wandb:…}` interpolation — the bare `wandb:…` form is passed through literally and never resolved:

```yaml
ckpt_path: ${wandb:model-flow-simple:latest}
```

It downloads the artifact once under `$PROJECT_ROOT/.cache/checkpoints/<key>` and reuses it on later resolutions; a cache dir holding no `.ckpt` is treated as a partial download and refetched. The cache key (`_cache_key`) is a path-safe slug plus a hash, so a hostile ref (`..`, `:`) cannot escape the cache root and distinct refs never collide. `wandb` is imported lazily — importing the module never requires it.

______________________________________________________________________

## 6. Aliases

W&B automatically applies `:latest` to every artifact. Dataset finalization also applies the immutable `:{run_id}` alias, which train and eval use for dataset lineage. Spec §4 reserves two model aliases that are not yet wired:

| Alias         | Set by           | When                                  | Status                                                                                                     |
| ------------- | ---------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `:latest`     | W&B (automatic)  | every `log_artifact` call             | landed                                                                                                     |
| `:{run_id}`   | dataset finalize | when the frozen dataset is finalized  | landed — train/eval use it to preserve dataset-version lineage                                             |
| `:best`       | training script  | when the val metric improves          | planned — `_log_model_artifact` logs with no `aliases=[…]`                                                 |
| `:production` | promote workflow | when a model is promoted to a Release | planned — promote workflow not implemented ([#1566](https://github.com/tinaudio/synth-setter/issues/1566)) |

______________________________________________________________________

## 7. Code Map

| Concern                          | Symbol                                                                | File                                               |
| -------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------- |
| Dataset artifact                 | `build_dataset_artifact` / `_log_dataset_artifact`                    | `src/synth_setter/cli/finalize_dataset.py`         |
| Model artifact                   | `build_model_artifact` / `_log_model_artifact`                        | `src/synth_setter/cli/train.py`                    |
| Best-checkpoint R2 upload        | `_upload_best_checkpoint` / `_derive_checkpoint_uri`                  | `src/synth_setter/cli/train.py`                    |
| Checkpoint fingerprint guard     | `_fingerprint_guard_allows_overwrite` / `_upload_fingerprint_sidecar` | `src/synth_setter/cli/train.py`                    |
| Fingerprint model / sidecar URI  | `CheckpointFingerprint` / `fingerprint_from_cfg`                      | `src/synth_setter/utils/checkpoint_fingerprint.py` |
| Eval-results artifact            | `build_eval_results_artifact` / `_log_eval_results_artifact`          | `src/synth_setter/cli/eval.py`                     |
| Consumed-edge refs (training)    | `_consumed_artifact_refs`                                             | `src/synth_setter/cli/train.py`                    |
| Lineage edge recording           | `record_input_lineage` / `use_input_artifacts`                        | `src/synth_setter/utils/logging_utils.py`          |
| Incomplete-lineage run marker    | `mark_lineage_incomplete`                                             | `src/synth_setter/utils/logging_utils.py`          |
| Referenced-checkpoint metadata   | `_checkpoint_metadata`                                                | `src/synth_setter/cli/train.py`                    |
| `${wandb:…}` resolver            | `_resolve_wandb_checkpoint` / `register_resolvers`                    | `src/synth_setter/utils/utils.py`                  |
| Run id / `job_type` pinning      | `pin_wandb_run_id`                                                    | `src/synth_setter/utils/logging_utils.py`          |
| Provenance fields (`github_sha`) | `log_wandb_provenance`                                                | `src/synth_setter/utils/logging_utils.py`          |
