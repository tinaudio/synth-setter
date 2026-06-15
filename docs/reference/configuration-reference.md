# Configuration Reference

> **Code version**: `f97fc7e` (2026-03-29, `main`)
> **Tracking**: #383, #107

______________________________________________________________________

## 1. Configuration Layers

| Layer                         | Tool                                                                                                                                                                                                                                                                                                                | Validation                                                                                                                                                                                                                | Stored In                                                                                                                          | Example                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Experiment config             | Hydra YAML composition                                                                                                                                                                                                                                                                                              | Deferred — class constructors at `hydra.utils.instantiate()`                                                                                                                                                              | git (`src/synth_setter/configs/experiment/`)                                                                                       | `src/synth_setter/configs/experiment/surge/flow_simple.yaml`                      |
| Pipeline input + runtime spec | Pydantic `BaseModel(strict=True, frozen=True, extra="forbid")` — `DatasetSpec` unifies the prior config + materialized-spec split. All three models (`DatasetSpec`, `RenderConfig`, `ShardSpec`) are strict; JSON round-trip coercions (`list→tuple`, `str→datetime`) are handled by explicit per-field validators. | Parse-time — Hydra `compose` → `spec_from_cfg()` (#887, #912, #917 unified the prior `DatasetConfig` + `DatasetPipelineSpec` split into one model that is both the validated input *and* the materialized artifact on R2) | git (`src/synth_setter/configs/experiment/generate_dataset/`) for input; R2 (`{r2.prefix}input_spec.json`) for the serialized JSON | `src/synth_setter/configs/experiment/generate_dataset/surge-simple-480k-10k.yaml` |
| Cloud infrastructure          | SkyPilot Task YAML                                                                                                                                                                                                                                                                                                  | Launcher script (not Hydra)                                                                                                                                                                                               | git (`src/synth_setter/configs/compute/`)                                                                                          | `src/synth_setter/configs/compute/runpod-template.yaml`                           |
| Secrets / credentials         | Environment variables                                                                                                                                                                                                                                                                                               | Runtime                                                                                                                                                                                                                   | `.env` (local), CI secrets                                                                                                         | `WANDB_API_KEY`                                                                   |

### Why These Boundaries

- **Pydantic strict** at trust boundaries — where data enters from external sources (user config YAML, JSON from R2, worker reports). Catches type errors, missing fields, and invalid values at parse time.
- **Hydra DictConfig** for training — composable experiment configs validated by class constructors at instantiation. Hydra handles defaults, overrides, and interpolation natively.
- **Plain YAML for cloud infrastructure** — consumed by a launcher script that calls provider APIs before the training job starts. Different program, different time, no Hydra composition needed.
- **No training input spec** — training is a single long-running job with no distributed coordination. The data pipeline's spec exists for reconciliation across hundreds of parallel workers; training has no equivalent need. Provenance is captured by W&B run metadata + frozen `config.yaml` in R2.

Reference: `data-pipeline.md` §14.4, `training-pipeline.md` §7.1

______________________________________________________________________

## 2. Config Architecture Per Stage

### 2.1 Data Generation

```
src/synth_setter/configs/experiment/generate_dataset/{id}.yaml → Hydra compose against src/synth_setter/configs/dataset.yaml
  → spec_from_cfg(cfg) → DatasetSpec (frozen, Pydantic, the spec ON R2)
    → spec_io.write_spec_locally(spec, Path(cfg.paths.output_dir))
        → <hydra_output_dir>/data/<task_name>/<run_id>/metadata/input_spec.json (operator-side artifact)
    → r2_io.ensure_r2_env_loaded(sky_cfg.env_file)   (dotenv + auth ping)
    → spec_io.upload_spec(spec) → R2 at {r2.prefix}input_spec.json (one canonical write per main())
    → branch on sky_cfg.compute_template:
        ├─ None: generate(spec, Path(cfg.paths.output_dir), loggers) — renders + uploads shards
        └─ set:  dispatch_via_skypilot — injects spec.r2.input_spec_uri() as WORKER_SPEC_URI;
                 worker pod runs generate(spec, work_dir, loggers) which renders + uploads shards (no spec re-upload)
```

- Input is mutable, human-authored YAML under `src/synth_setter/configs/experiment/generate_dataset/`
- `DatasetSpec` is the unified model: the same frozen Pydantic instance is both the validated input and the materialized artifact (`DatasetConfig` + `DatasetPipelineSpec` were unified in #887)
- Runtime state (git SHA, renderer version, per-shard seeds) auto-fills via `default_factory` fields (`git_sha`, `is_repo_dirty`, `created_at`, plus `run_id` and `r2` via the `_default_run_id` / `_default_r2_location` factories; `r2.prefix` is derived by `_fill_default_r2_prefix` in a `mode='before'` model validator)
- Spec is the reproducibility unit and reconciliation target
- **Config drift protection (planned):** the design doc specifies that re-passing `--config` for a `run_id` that already has a spec should error — but this is not yet enforced. The current implementation always generates a new `run_id` and writes a fresh spec. Tracked in [#386](https://github.com/tinaudio/synth-setter/issues/386).
- **Path note:** `storage-provenance-spec.md` §3a documents the target path as `metadata/input_spec.json`, but the current implementation uploads to `{r2.prefix}input_spec.json` (`r2.prefix` already ends in `/` — see `make_r2_prefix` in `src/synth_setter/pipeline/schemas/prefix.py`; no `metadata/` subdirectory). Tracked in [#385](https://github.com/tinaudio/synth-setter/issues/385).
- **Worker env:** `dispatch_via_skypilot` injects the canonical `spec.r2.input_spec_uri()` as `WORKER_SPEC_URI` into each worker pod's env. The canonical provenance copy at `{r2.prefix}input_spec.json` is written by `spec_io.upload_spec`, called once from `main()` on the launcher host before the dispatch branch fires, so the URI resolves before any worker boots. Workers do not re-upload the spec. See `storage-provenance-spec.md` §3a "Materialized spec: two destinations" for the consumer table.

Reference: `data-pipeline.md` §14.5

### 2.2 Data Finalization

```
synth-setter-finalize-dataset dataset_root_uri=r2://…/<task_name>/<run_id>/
  → @hydra.main composes DictConfig from src/synth_setter/configs/finalize_dataset.yaml
    → load_spec_from_root(cfg.dataset_root_uri) → DatasetSpec (joins input_spec.json under the root; the frozen spec generate uploaded)
      → r2_io.object_size(spec.r2.dataset_complete_marker_uri()) probe (idempotency short-circuit)
      → assert_r2_prefix_matches(…) (advisory: warns on a non-canonical prefix, never aborts — custom prefixes like the oracle-eval e2e's test-runs/ are legitimate)
      → branch on spec.output_format:
          ├─ wds:  finalize_wds  — Welford-stream stats over train shards → upload stats.npz
          └─ hdf5: finalize_hdf5 — download every shard → reshard into {train,val,test}.h5 → stats.npz
      → upload dataset.complete marker LAST (R2 source-of-truth resumability invariant)
```

- Single required input: `dataset_root_uri` (the run prefix `.../<task_name>/<run_id>/`). `load_spec_from_root` joins `input_spec.json` under it; the URI scheme is dispatched by `load_spec_from_uri` (`file://`, `r2://`, or bare path)
- `cfg.paths.output_dir` (Hydra's per-run dir under `${paths.log_dir}/finalize_dataset/<timestamp>`) is the scratch work_dir for both branches
- Idempotency: a re-run against a prefix that already has `dataset.complete` exits cleanly without downloads or uploads. R2 is the source of truth (see `pipeline/CLAUDE.md`)
- Marker-last invariant: `dataset.complete` is uploaded strictly after every artifact a downstream consumer expects, so an interrupted run never leaves a marker without its splits / stats

Reference: `data-pipeline.md` §14.5 (finalize stage)

### 2.3 Training

```
train.yaml + defaults (experiment, datamodule, model, trainer, callbacks, logger)
  → Hydra composes DictConfig
    → hydra.utils.instantiate() → LightningModule, DataModule, Trainer
      → trainer.fit(model, datamodule)
```

- No intermediate spec — Hydra instantiates directly to Python objects
- Provenance: W&B config (hyperparams, `github_sha`) + frozen `config.yaml` in R2
- Resume: Lightning native `ckpt_path=` with W&B artifact download
- Single-job model — no reconciliation, no distributed coordination

Reference: `training-pipeline.md` §4–5

### 2.4 Evaluation

```
eval.yaml + experiment config (pins model + data + checkpoint)
  + evaluation: {render_vst, compute_metrics, rerender_target, num_workers, shuffle_seed}
  + render: {param_spec_name, preset_path, plugin_path?}   # required when render_vst=true
  → Hydra composes DictConfig → predict (→ render → metrics if mode=predict and gates on)
```

- Experiment config pins everything: model checkpoint (W&B artifact ref), data config, eval settings
- `evaluation:` block (in `src/synth_setter/configs/eval.yaml`) gates the in-process render and metrics phases — both default off so `mode=test`/`mode=validate` runs are unchanged
- `render:` defaults entry composes a renderer config group (e.g. `render=surge_xt`) and supplies the VST plugin/preset/param-spec that `_run_predict_postprocessing` forwards to the render subprocess
- No eval spec — configs are the source of truth
- Full provenance in R2 path: `eval/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/{eval_config_id}/{eval_wandb_run_id}/`

Reference: `eval-pipeline.md` §4–5

### 2.5 Cloud Infrastructure

`dispatch_via_skypilot` is the single entry point a `synth-setter-*` CLI takes
to dispatch onto SkyPilot. Each console script that supports compute carries
a `skypilot_launch` sub-config (today: `synth-setter-generate-dataset`; more
entrypoints are expected to follow). Setting
`skypilot_launch.compute_template=<path>` flips the command from "run
in-process" to "materialize the spec, then dispatch via SkyPilot" — no
separate launcher invocation is involved.

#### Dispatch flow

```
synth-setter-generate-dataset experiment=… skypilot_launch.compute_template=src/synth_setter/configs/compute/runpod-template.yaml
  → @hydra.main composes DictConfig → spec_from_cfg → DatasetSpec
    → write_spec_locally(spec, Path(cfg.paths.output_dir))
    → upload_spec(spec) → R2 at {r2.prefix}input_spec.json
    → sky_cfg.extra_envs["WORKER_SPEC_URI"] = spec.r2.input_spec_uri()
    → dispatch_via_skypilot(sky_cfg)
      → SkyPilot provisions pod (RunPod, OCI, kubernetes via `sky local up`)
        → pod runs: cd /home/build/synth-setter
                    && bash scripts/sync_worker_checkout.sh
                    && exec synth-setter-generate-dataset-from-hydra <pinned hydra overrides>
```

- `dispatch_via_skypilot` takes a single `SkypilotLaunchConfig` argument and
  knows nothing about `DatasetSpec`. Dataset-specific worker envs (today
  `WORKER_SPEC_URI`) flow through `sky_cfg.extra_envs`, merged into per-rank
  envs after `resolve_worker_env`.
- Hydra composition lives in the entrypoint itself; the launcher module is
  not on this path beyond `dispatch_via_skypilot`.
- `_build_worker_cmd` (in `synth_setter.cli.generate_dataset`) pins the same
  Hydra overrides the operator composed with, and the `from_hydra` entrypoint
  on the worker rebuilds the spec from those — so worker re-execution is
  deterministic regardless of operator argv.
- The canonical `WORKER_SPEC_URI` is forwarded via `task.update_envs(...)`
  primarily for downstream validate-time consumers (validate-spec /
  validate-shard CI jobs read it off the workflow output). The worker itself
  doesn't fetch the JSON. `task.update_file_mounts` is avoided because
  SkyPilot's RunPod backend rejects programmatic file_mounts with a
  pubkey-overflow error (see [#749](https://github.com/tinaudio/synth-setter/issues/749)).

#### Job-name conventions

`sky_cfg.job_name` is optional. When unset, the launcher falls back to
`synth-setter-<8-hex-uuid>` — a domain-neutral default. Callers that want a
domain-shaped stem set `job_name` explicitly before calling the launcher;
`synth-setter-generate-dataset` pins
`synth-setter-smoke-<spec.task_name[:8]>` from `main()`.

Reference: `training-pipeline.md` Appendix D

______________________________________________________________________

## 3. Cloud Provider Comparison

| Concern             | RunPod                                   | Vast.ai                                                                      |
| ------------------- | ---------------------------------------- | ---------------------------------------------------------------------------- |
| API model           | Create pod with explicit GPU type        | Search offers with query filters → rent best match                           |
| GPU selection       | `gpu_type_id` from catalog               | Query: `gpu_name=RTX_4090 num_gpus>=1 gpu_ram>=24`                           |
| Pricing model       | Fixed $/hr per GPU type                  | Market-based: on-demand or bid (interruptible)                               |
| Spot / preemptible  | Community cloud (cheaper, less reliable) | `--type=bid` with custom bid price                                           |
| Persistent storage  | Network volumes (`networkVolumeId`)      | Volumes (create new or attach existing by ID)                                |
| Docker image        | `image_name` parameter                   | `image` parameter                                                            |
| Environment vars    | `env` dict (key-value pairs)             | Docker-flag format: `"-e KEY=VALUE"`                                         |
| Startup command     | `docker_args` string                     | `onstart` script (SSH mode) or `args` array (args mode)                      |
| SSH access          | Always available                         | Runtype-dependent (`ssh_direct`, `jupyter_direct`, `args`)                   |
| Port exposure       | Default open                             | Configurable: direct ports or proxy                                          |
| Auto-terminate      | Open question (#107 §11)                 | Supported via `destroy instance` API call                                    |
| GPU filtering       | Choose from fixed catalog                | Rich query language (40+ fields: FLOPS, bandwidth, reliability, geolocation) |
| Reliability scoring | N/A                                      | `reliability` field (0–1 score per machine)                                  |
| Datacenter option   | All datacenter                           | `datacenter=True` filter (vs community hosts)                                |
| Region selection    | Limited datacenter selection             | `geolocation in [US,CA,DE]` filter                                           |
| Benchmark data      | N/A                                      | `dlperf` (DL perf score), `dlperf_per_dphtotal` (perf/$)                     |

### Config Shape

The RunPod template exists today (data-pipeline smoke). Vast.ai template not yet implemented.

**RunPod** (`src/synth_setter/configs/compute/runpod-template.yaml`) — landed. Abridged
shape (see the file for the full template):

The launcher injects `image_id` per-launch via `sky_cfg.worker_image_tag` (default `"dev-snapshot"`) for non-OCI backends, so the template omits a literal `image_id:` entry and relies on the per-launch injection:

```yaml
resources:
  cloud: runpod
  accelerators: RTXA4000:1
  use_spot: false
  disk_size: 50

envs:
  RCLONE_CONFIG_R2_TYPE: ""           # the 5 RCLONE_CONFIG_R2_* keys + WANDB_API_KEY
  RCLONE_CONFIG_R2_PROVIDER: ""       # are injected at launch time from
  RCLONE_CONFIG_R2_ACCESS_KEY_ID: ""  # .env or process env (see _WORKER_ENV_KEYS).
  RCLONE_CONFIG_R2_SECRET_ACCESS_KEY: ""
  RCLONE_CONFIG_R2_ENDPOINT: ""
  WANDB_API_KEY: ""
  WORKER_GIT_REF: ""                  # PR-CI bake-lag bypass for sync_worker_checkout.sh
  SYNTH_SETTER_WORKER_RANK: ""        # per-rank partition (synthesized per-rank by _launch_one_rank_from_doc)
  SYNTH_SETTER_NUM_WORKERS: ""

# No `run:` block — the launcher's `_build_worker_cmd` constructs the cd +
# sync_worker_checkout.sh + `exec synth-setter-generate-dataset-from-hydra
# <pinned hydra overrides>` one-liner and injects it via the Task's `run`
# field at dispatch time. Adding a `run:` block here is rejected by
# `_load_compute_template_with_cmd`.
```

The canonical spec is uploaded to R2 by `cli/generate_dataset.py`'s `main()`
(via `spec_io.upload_spec`) before dispatch; `task.update_file_mounts(...)`
is avoided per [#749](https://github.com/tinaudio/synth-setter/issues/749).
The canonical URI (`spec.r2.input_spec_uri()`) is forwarded to the worker
pod as the `WORKER_SPEC_URI` env var (consumed by the CI validate-spec /
validate-shard jobs, which read it via the workflow output rather than off
the pod); the worker process itself re-builds the spec via Hydra compose on
the injected overrides rather than fetching the JSON at boot.

**Vast.ai** (`src/synth_setter/configs/compute/vast-template.yaml`) — planned, not implemented:

```yaml
provider: vast
search_query: "gpu_name=RTX_4090 num_gpus>=1 gpu_ram>=24 reliability>=0.99 datacenter=True"
disk: 64
runtype: "args"
image: "tinaudio/synth-setter-train:latest"
```

______________________________________________________________________

## 4. Config Boundary Rules

| Boundary                             | Tool                       | Why                                                                                       |
| ------------------------------------ | -------------------------- | ----------------------------------------------------------------------------------------- |
| External input (config YAML)         | Pydantic `strict=True`     | Untrusted human input — catch type errors, missing fields, invalid values at parse time   |
| Serialization (spec, reports, cards) | Pydantic `strict=True`     | JSON crossing process boundaries (R2 ↔ CLI ↔ workers) — enforce schema on every read      |
| Training experiment config           | Hydra DictConfig           | Composable defaults + overrides; validation deferred to class constructors                |
| HDF5 shard data (NumPy arrays)       | Custom validation function | Pydantic can't validate `ndarray`; custom shape/dtype/value checks required               |
| Internal data transform              | `dataclass(frozen=True)`   | Already validated — typed container prevents field mixups; no runtime validation overhead |
| Cloud infrastructure                 | Plain YAML                 | Different consumer (launcher script), no composition needed, no `_target_` instantiation  |
| Secrets / credentials                | Environment variables      | Never committed to git; loaded from `.env` (local) or CI secrets                          |

Reference: `data-pipeline.md` §14.4

______________________________________________________________________

## 5. Known Gaps

Gaps are configuration inputs that design docs specify or that standard practice recommends, but that don't exist in the codebase yet.

### 5.1 Identity & Provenance

| Input                          | Type   | What's Needed                                                                  | Reference                   |
| ------------------------------ | ------ | ------------------------------------------------------------------------------ | --------------------------- |
| `train_wandb_run_id`           | string | `{train_config_id}-{YYYYMMDDTHHMMSSsssZ}` — structured, reconstructible run ID | storage-provenance-spec §1  |
| `dataset_wandb_run_id` linkage | string | Explicit link to specific dataset run version                                  | storage-provenance-spec §2  |
| `job_type`                     | string | Must be `"training"` in W&B config — currently empty                           | storage-provenance-spec §7  |
| `github_sha` in `wandb.config` | string | Logged via `log_wandb_provenance()` but not in Hydra config                    | wandb-integration.md gap #3 |

### 5.2 W&B / Artifact Lineage

| Input                    | Type   | What's Needed                                                                          | Reference                   |
| ------------------------ | ------ | -------------------------------------------------------------------------------------- | --------------------------- |
| `logger.wandb.log_model` | bool   | `False` — no checkpoint files to W&B; best ckpt goes to R2, referenced by the artifact | training-pipeline.md §6.2   |
| `logger.wandb.id`        | string | `{train_config_id}-{YYYYMMDDTHHMMSSsssZ}` instead of null/random                       | wandb-integration.md gap #8 |
| `logger.wandb.job_type`  | string | `"training"` instead of empty                                                          | storage-provenance-spec §7  |
| `logger.wandb.resume`    | string | `"allow"` for W&B resume support                                                       | training-pipeline.md §5.3   |

Model `run.log_artifact()` lineage is wired via `_log_model_artifact()` (train), which logs the canonical `model-{config_id}` artifact. At train end the best checkpoint is uploaded to R2 (`_upload_best_checkpoint`) at `r2://{r2.bucket}/checkpoints/{config_id}/model.ckpt` and the artifact references it as an `s3://` URI; `training.upload_checkpoints_uri` optionally overrides the target (default `null` = auto-derive). Dataset `run.use_artifact()` lineage is wired via `use_input_artifacts()` (train/eval), activated by the opt-in `consumed_dataset_config_id` / `consumed_train_config_id` config keys (default `null` = no edge; alias from `consumed_artifact_alias`, default `latest`).

### 5.3 Data Portability

| Input                                  | Type           | What's Needed                                                                                                  | Reference                                                   |
| -------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `datamodule.dataset_root`              | string         | Defaults to `${paths.output_dir}/data` (Hydra per-run dir); CLI/experiment override for fixed datasets         | training-pipeline.md §6.1                                   |
| `datamodule.download_dataset_root_uri` | string \| null | Optional `r2://` directory URI; `prepare_data()` no-clobber-copies it into `dataset_root` before training/eval | `src/synth_setter/data/surge_datamodule.py` §`prepare_data` |
| `datamodule.stats_file`                | string         | Hardcoded paths removed (now `???` in `nsynth.yaml`/`fsd.yaml`); replace with run-id-aware default still open  | `nsynth.yaml` / `fsd.yaml`                                  |

### 5.4 Hardware & Compute

| Input                             | Type   | What's Needed                                                       | Reference                   |
| --------------------------------- | ------ | ------------------------------------------------------------------- | --------------------------- |
| `trainer.precision`               | string | Not set in most configs — should be explicit (e.g., `"bf16-mixed"`) | Lightning Trainer option    |
| `trainer.accumulate_grad_batches` | int    | Effective batch size scaling                                        | Lightning Trainer option    |
| `trainer.benchmark`               | bool   | cuDNN benchmark mode for performance                                | Lightning Trainer option    |
| `datamodule.pin_memory`           | bool   | Not set in active configs — should be standard for GPU training     | Lightning DataLoader option |
| `datamodule.persistent_workers`   | bool   | Keep workers alive between epochs — performance optimization        | Lightning DataLoader option |
| `datamodule.prefetch_factor`      | int    | DataLoader prefetch                                                 | Lightning DataLoader option |
| `model.optimizer.fused`           | bool   | CUDA fused Adam — significant speedup on GPU                        | torch.optim.Adam option     |

### 5.5 Cloud Infrastructure

| Input                               | Type               | What's Needed                                                                                                                                          | Reference                                             |
| ----------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| RunPod config                       | SkyPilot Task YAML | Landed for the data pipeline smoke at `src/synth_setter/configs/compute/runpod-template.yaml`; training launcher still uses the legacy RunPod-API path | data-pipeline.md §14, training-pipeline.md Appendix D |
| Vast.ai config                      | SkyPilot Task YAML | Planned — `src/synth_setter/configs/compute/vast-template.yaml` not yet authored                                                                       | new provider                                          |
| `src/synth_setter/configs/compute/` | directory          | SkyPilot Task templates for the data pipeline launcher (RunPod landed; Vast.ai planned)                                                                | —                                                     |
| `make train`                        | Makefile target    | Training shorthand with EXPERIMENT arg                                                                                                                 | training-pipeline.md §2                               |
| `make docker-train`                 | Makefile target    | Docker training shorthand                                                                                                                              | training-pipeline.md §2                               |
| `make runpod-train`                 | Makefile target    | RunPod launcher shorthand                                                                                                                              | training-pipeline.md §2                               |
| `make resume`                       | Makefile target    | Resume from W&B artifact with EXPERIMENT + RUN_ID                                                                                                      | training-pipeline.md §2                               |

### 5.6 Other

| Input                                                                                              | Type       | What's Needed                                                                                                                                                                                                                                                                                                                                                                                                                | Reference                          |
| -------------------------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Training Docker image                                                                              | Dockerfile | Separate from data pipeline image (needs CUDA+torch, not VST+rclone)                                                                                                                                                                                                                                                                                                                                                         | training-pipeline.md §7.4          |
| Training R2 paths                                                                                  | code       | Dataset R2 path, training R2 path, frozen config upload                                                                                                                                                                                                                                                                                                                                                                      | storage-provenance-spec §3b        |
| `RUNPOD_API_KEY`                                                                                   | env var    | RunPod launcher needs API access                                                                                                                                                                                                                                                                                                                                                                                             | storage-provenance-spec §9         |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_ENDPOINT` / `AWS_ENDPOINT_URL` / `AWS_REGION` | env vars   | Global AWS-SDK → R2 redirect for tools that ignore `RCLONE_CONFIG_R2_*`: the SmooSense / DuckDB-`lance` data viewer querying `.lance` datasets referenced by `s3://...` URIs, raw boto3, AWS CLI. Lance's Rust object_store reads `AWS_ENDPOINT`, DuckDB httpfs reads `AWS_ENDPOINT_URL` — set both. Not needed for model checkpoints (the `${wandb:...}` resolver rclone-downloads from R2). Sample values: `.env.example`. | storage-provenance-spec §11        |
| `torch.compile` backend/mode                                                                       | string     | Currently just a bool toggle — no backend, mode, fullgraph, dynamic options                                                                                                                                                                                                                                                                                                                                                  | Lightning/PyTorch option           |
| `PYTHONHASHSEED`                                                                                   | env var    | Fixed hash seed for reproducibility                                                                                                                                                                                                                                                                                                                                                                                          | standard ML practice               |
| `CUBLAS_WORKSPACE_CONFIG`                                                                          | env var    | `:4096:8` for deterministic cuBLAS                                                                                                                                                                                                                                                                                                                                                                                           | required when `deterministic=True` |

______________________________________________________________________

## 6. Cross-References

This document does not duplicate content from authoritative sources. Consult them directly:

| Topic                                 | Authoritative Source         | What It Covers                                                 |
| ------------------------------------- | ---------------------------- | -------------------------------------------------------------- |
| IDs, R2 paths, W&B artifacts, secrets | `storage-provenance-spec.md` | Naming conventions, bucket layout, artifact types, lineage DAG |
| Current W&B integration state         | `wandb-integration.md`       | What gets logged, initialization, 8 known gaps                 |
| Training architecture                 | `training-pipeline.md`       | Phase plan, design decisions, stage definitions                |
| Data pipeline config → spec flow      | `data-pipeline.md` §14       | Schema design, materialization, validation boundaries          |
| Eval pipeline architecture            | `eval-pipeline.md`           | Three-stage pipeline, R2 integration, config pinning           |
| Docker image contract                 | `docker.md`                  | Image targets, env vars, run patterns                          |
