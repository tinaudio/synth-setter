# W&B Artifact & Provenance Reference

> **Last Updated**: 2026-06-08
> **Tracking**: #1565, #122

Companion to [storage-provenance-spec.md](../design/storage-provenance-spec.md). The spec is authoritative for names, paths, and conventions; this reference shows the **landed code patterns** behind them — how each artifact is built, how lineage edges are recorded, and how a logged artifact is resolved back to a local checkpoint.

______________________________________________________________________

## Overview

Every run logs the artifacts it produces and consumes the artifacts it reads, so W&B reconstructs a lineage DAG (spec §5) across data generation → training → evaluation → promotion. Three rules hold everywhere:

- Outputs are logged with `run.log_artifact(...)`; inputs are linked with `run.use_artifact(...)` (only `use_artifact` — not `api.artifact()` — creates a lineage edge).
- R2 objects are attached as `s3://` references with `checksum=False`; the URI records lineage, not a content hash (R2's custom S3 endpoint is unreachable by W&B's default reference handler).
- Artifact logging and lineage edges are **best-effort**: a W&B failure warns and is swallowed so it never aborts a run whose real work already succeeded, and a run with no `WandbLogger` is a silent no-op.

______________________________________________________________________

## 1. Artifact Catalog

| Type           | Name pattern               | Built by                                         | R2 reference                             |
| -------------- | -------------------------- | ------------------------------------------------ | ---------------------------------------- |
| `dataset`      | `data-{dataset_config_id}` | `build_dataset_artifact` (`finalize_dataset.py`) | split `.h5` / shard prefix + `stats.npz` |
| `model`        | `model-{train_config_id}`  | `build_model_artifact` (`train.py`)              | checkpoint prefix (opt-in, see §3)       |
| `eval-results` | `eval-{eval_config_id}`    | `build_eval_results_artifact` (`eval.py`)        | output-dir prefix                        |

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

| Artifact       | Metadata keys                                                |
| -------------- | ------------------------------------------------------------ |
| `dataset`      | `shard_count`, `n_samples`, `git_sha`                        |
| `model`        | `git_sha`                                                    |
| `eval-results` | scalar summary metrics (`_eval_summary_metrics`) + `git_sha` |

The `model` artifact carries only `git_sha` today; its checkpoint reference is opt-in via `training.upload_checkpoints_uri` (an `r2://` prefix or null). The null default logs a **lineage-only** artifact with no reference, because R2 checkpoint upload is not implemented yet ([#92](https://github.com/tinaudio/synth-setter/issues/92)).

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
        lg.experiment.log_artifact(build_model_artifact(cfg))
```

**Consuming an input** — `use_input_artifacts` (`utils/logging_utils.py`) records each `(name, alias)` edge via `use_artifact`; it is `@rank_zero_only` so a DDP run records each edge once:

```python
use_input_artifacts(loggers, _consumed_artifact_refs(cfg))  # e.g. [("data-diva-v1", "latest")]
```

Consumed edges are opt-in: training reads `consumed_dataset_config_id` (null by default → no edge), so a run without it set records no input lineage and never calls `use_artifact`.

______________________________________________________________________

## 5. Resolving an Artifact to a Checkpoint

The `${wandb:<ref>}` OmegaConf resolver (`utils/utils.py`, registered in `register_resolvers`) turns a model-artifact ref into a local checkpoint path. To resume, point `ckpt_path` at a `${wandb:…}` interpolation — the bare `wandb:…` form is passed through literally and never resolved:

```yaml
ckpt_path: ${wandb:model-flow-simple:best}
```

It downloads the artifact once under `$PROJECT_ROOT/.cache/checkpoints/<key>` and reuses it on later resolutions; a cache dir holding no `.ckpt` is treated as a partial download and refetched. The cache key (`_cache_key`) is a path-safe slug plus a hash, so a hostile ref (`..`, `:`) cannot escape the cache root and distinct refs never collide. `wandb` is imported lazily — importing the module never requires it.

______________________________________________________________________

## 6. Aliases

`:latest` is the only alias the landed code applies — W&B sets it automatically on every `log_artifact` call. Spec §4 reserves two more, neither yet wired:

| Alias         | Set by           | When                                  | Status                                                                                                     |
| ------------- | ---------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `:latest`     | W&B (automatic)  | every `log_artifact` call             | landed                                                                                                     |
| `:best`       | training script  | when the val metric improves          | planned — `_log_model_artifact` logs with no `aliases=[…]`                                                 |
| `:production` | promote workflow | when a model is promoted to a Release | planned — promote workflow not implemented ([#1566](https://github.com/tinaudio/synth-setter/issues/1566)) |

______________________________________________________________________

## 7. Code Map

| Concern                          | Symbol                                                       | File                                       |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| Dataset artifact                 | `build_dataset_artifact` / `_log_dataset_artifact`           | `src/synth_setter/cli/finalize_dataset.py` |
| Model artifact                   | `build_model_artifact` / `_log_model_artifact`               | `src/synth_setter/cli/train.py`            |
| Eval-results artifact            | `build_eval_results_artifact` / `_log_eval_results_artifact` | `src/synth_setter/cli/eval.py`             |
| Consumed-edge refs (training)    | `_consumed_artifact_refs`                                    | `src/synth_setter/cli/train.py`            |
| Lineage edge recording           | `use_input_artifacts`                                        | `src/synth_setter/utils/logging_utils.py`  |
| `${wandb:…}` resolver            | `_resolve_wandb_checkpoint` / `register_resolvers`           | `src/synth_setter/utils/utils.py`          |
| Run id / `job_type` pinning      | `pin_wandb_run_id`                                           | `src/synth_setter/utils/logging_utils.py`  |
| Provenance fields (`github_sha`) | `log_wandb_provenance`                                       | `src/synth_setter/utils/logging_utils.py`  |
