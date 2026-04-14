# SkyPilot Compute Integration Design

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-04-13
> **Tracking**: [#534](https://github.com/tinaudio/synth-setter/issues/534), [#105](https://github.com/tinaudio/synth-setter/issues/105) (Task 4.2: Compute Backend & Worker), [#106](https://github.com/tinaudio/synth-setter/issues/106) (Task 6.1: RunPod Backend & E2E)

## 1. Context

The data pipeline's compute backend abstraction (Phase 4-6 of [data-pipeline-implementation-plan.md](data-pipeline-implementation-plan.md)) is designed but not yet implemented. The original design specifies a `ComputeBackend` protocol with `LocalBackend` and `RunPodBackend` implementations.

Three factors motivate switching to SkyPilot before implementation begins:

1. **Multi-provider flexibility** — Vast.ai + RunPod without writing each backend.
2. **RunPod friction** — sshd-in-Docker requirement, env var bugs in SSH sessions, no native SSH. These are constant paper cuts for a development workflow that relies on Tailscale + VS Code tunnels + Claude Code.
3. **Reduced scope** — SkyPilot handles provisioning, spot recovery, and job lifecycle; less code to write and maintain.

SkyPilot was chosen over:

- **Raw Vast.ai CLI** — no ecosystem of reusable tooling; the people who need orchestration use SkyPilot on top of Vast.ai rather than scripting the CLI directly.
- **Modal** — requires restructuring code around `@modal.function` decorators; SkyPilot runs vanilla Python scripts with no code changes.

The reconciliation-based pipeline design is naturally compatible with SkyPilot managed jobs — R2 markers serve as checkpoints for idempotent resume after spot preemption.

## 2. Architecture Decisions

| Decision          | Choice                                                  | Rationale                                                                                                               |
| ----------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Integration depth | Full managed jobs                                       | Spot recovery + R2 markers as natural checkpoints. Cost savings 3-5x on interruptible instances                         |
| Local dev/test    | Keep LocalBackend                                       | In-process execution for fast unit tests. Two code paths (local vs SkyPilot)                                            |
| Field name        | `compute_config`                                        | Tool-agnostic. Value is a path to a SkyPilot YAML today. Survives tool changes without schema migration                 |
| Backend selection | Presence of `compute_config`                            | `None` → local, path → SkyPilot. No enum, no protocol, no extra plumbing                                                |
| Shard parallelism | `num_workers` in DatasetConfig                          | Parallelism is a reproducible property of the config, not a runtime variable                                            |
| Worker identity   | UUID generated at worker start                          | Decoupled from any provider. Fully portable                                                                             |
| Deployment        | Docker image                                            | Reproducible; aligns with `pipeline/schemas/image_config.py` (see `docs/reference/docker.md`). SkyPilot pulls the image |
| CLI ownership     | `python -m pipeline generate` wraps SkyPilot            | Single entry point. User never touches `sky` CLI directly for generation                                                |
| Frozen spec       | Include `compute_config`                                | For provenance and cost tracking                                                                                        |
| Scope             | Design all three config types, implement pipeline first | DatasetConfig, train, eval all get `compute_config`                                                                     |

## 3. Schema Changes

### 3.1 DatasetConfig (`pipeline/schemas/config.py`)

Add two fields:

```python
class DatasetConfig(BaseModel):
    # ... existing fields ...
    num_workers: int = 1                       # Number of parallel workers for shard generation
    compute_config: str | None = None          # Path to SkyPilot YAML. None = local execution

    @field_validator("num_workers")
    @classmethod
    def _positive_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_workers must be >= 1")
        return v
```

- `num_workers` defaults to 1 (single worker, backward compatible).
- `compute_config` defaults to None (local execution, backward compatible).
- Both fields are optional additions — existing YAML configs continue to work.

### 3.2 DatasetPipelineSpec (`pipeline/schemas/spec.py`)

Add `compute_config` to the frozen spec. The spec is uploaded to R2, so a local file path is meaningless there. The SkyPilot YAML content is resolved and embedded at materialization time:

```python
class DatasetPipelineSpec(BaseModel):
    # ... existing fields ...
    num_workers: int
    compute_config: dict[str, Any] | None = None  # Resolved SkyPilot YAML content, or None for local
```

`materialize_spec()` reads the file and includes its parsed content. SkyPilot YAMLs are small (~20 lines), so embedding preserves full provenance without bloating the spec.

### 3.3 Training config (Hydra — `configs/train.yaml`)

Training uses pure Hydra DictConfig, not Pydantic. Add `compute_config` as a top-level key:

```yaml
# configs/train.yaml
defaults:
  - _self_
  - data: ???
  - model: ???
  - trainer: default
  # ... existing defaults ...

compute_config: null  # Path to SkyPilot YAML. null = local execution
```

The training entrypoint (`src/train.py`) reads `cfg.get("compute_config")` and either trains locally or launches via SkyPilot SDK.

### 3.4 Eval config (Hydra — `configs/eval.yaml`)

Same pattern as training:

```yaml
# configs/eval.yaml
compute_config: null
```

## 4. New Files & Artifacts

### 4.1 SkyPilot YAML configs (`configs/compute/`)

```
configs/compute/
├── vast-spot.yaml          # Vast.ai interruptible instances (cheapest)
└── vast-ondemand.yaml      # Vast.ai on-demand (reliable, more expensive)
```

Example `configs/compute/vast-spot.yaml`:

```yaml
resources:
  cloud: vast
  accelerators: A100:1       # Adjust per workload
  use_spot: true
  disk_size: 100
  # Use the existing CI image name and an immutable tag (for example, a git SHA).
  # In practice, the pinned tag can be injected by CI or at launch time.
  image_id: docker:tinaudio/perm:<git-sha>

setup: |
  # Worker setup runs inside the container
  echo "Worker ready"

run: |
  # Placeholder — overridden by pipeline CLI at launch time
  python -m pipeline.worker
```

The `run:` block is overridden programmatically by the pipeline CLI when launching managed jobs. Each worker gets its shard range injected.

### 4.2 Worker adaptation

The existing worker design (`pipeline/worker.py`, not yet implemented) stays mostly the same. Changes:

- Worker generates a UUID on startup for `worker_id` (instead of reading `RUNPOD_POD_ID`).
- Worker reads its assigned shard range from CLI args or env vars (set by pipeline CLI at SkyPilot launch time).
- Worker still writes `.rendering` → `.valid`/`.invalid` markers to R2.
- Worker is idempotent: on startup, checks R2 for already-valid shards in its range, skips them. This is the key property that makes SkyPilot managed jobs work — R2 markers are the natural checkpoints for spot preemption recovery.

## 5. What Changes in data-pipeline.md

### §7.9 Compute Abstraction

**Before:** `ComputeBackend` protocol with `submit()`, `LocalBackend`, `RunPodBackend`.

**After:**

- Remove `ComputeBackend` protocol and `RunPodBackend`.
- `LocalBackend` stays for dev/testing (in-process worker execution).
- SkyPilot integration via Python SDK in the pipeline CLI.
- `compute_config` field drives backend selection.
- Pattern: `if compute_config: launch_skypilot_jobs() else: run_local()`.

### §2 Workflow overview

Replace RunPod references with SkyPilot/provider-agnostic language.

### §7.3 Worker identity

`worker_id` changes from `RUNPOD_POD_ID` to UUID generated at worker startup.

### Implementation plan phases

- Phase 4 Task 4.2: `ComputeBackend` protocol → thin `if/else` on `compute_config`.
- Phase 6: `RunPodBackend` → SkyPilot managed jobs integration via Python SDK.

### Glossary

- Remove/update RunPod-specific terms.
- Add SkyPilot, managed job, `compute_config` definitions.

## 6. What Stays the Same

- **R2 as storage** — SkyPilot is compute, R2 is storage. Orthogonal concerns.
- **Reconciliation model** — `reconcile(spec, storage)` checks R2 markers. Unchanged.
- **Shard lifecycle markers** — `.rendering`, `.valid`, `.invalid`, `.promoted`. Unchanged.
- **Shard write protocol** — write `.rendering` → render → validate → upload → write `.valid`. Unchanged.
- **Validation tiers** — worker 3-check, finalize structural check. Unchanged.
- **Finalize workflow** — promote staged shards, compute stats, register W&B. Unchanged.
- **Docker image strategy** — build image, push to registry. SkyPilot pulls it (same as RunPod would).
- **`pipeline status`** — still reconciliation from R2 markers, no provider API queries.
- **Deterministic shard IDs** — `shard-{id:06d}`, infrastructure-independent. Unchanged.

## 7. Implementation Sequence

### Phase A: Schema changes (DatasetConfig + spec)

**Files to modify:**

- `pipeline/schemas/config.py` — add `num_workers`, `compute_config` fields + validators
- `pipeline/schemas/spec.py` — add `compute_config` to `DatasetPipelineSpec`, update `materialize_spec()`
- `configs/dataset/surge-simple-480k-10k.yaml` — add optional new fields (or leave defaults)
- Tests: `tests/pipeline/test_schemas/` — add test cases for new fields, backward compat

### Phase B: SkyPilot compute configs

**Files to create:**

- `configs/compute/vast-spot.yaml`
- `configs/compute/vast-ondemand.yaml`

**Files to modify:**

- `pyproject.toml` — add `skypilot[vast]` as optional dependency

### Phase C: Pipeline CLI + SkyPilot integration

**Files to create/modify:**

- `pipeline/worker.py` — worker process with UUID identity, shard range args, idempotent resume
- `pipeline/cli.py` / `pipeline/__main__.py` — CLI that reconciles, partitions, launches
- SkyPilot SDK calls in CLI: `sky.jobs.launch()` per worker batch

This is where the `compute_config` presence/absence drives behavior:

```python
if spec.compute_config:
    # Launch N managed jobs via SkyPilot Python SDK.
    # `spec.compute_config` is an embedded dict (see §3.2), so build the
    # SkyPilot Task programmatically from that dict rather than loading a
    # YAML path.
    for i, shard_batch in enumerate(partitioned_shards):
        task = build_skypilot_task(spec.compute_config)
        task.update_envs(
            {"SHARD_RANGE": f"{shard_batch.start}-{shard_batch.end}", ...}
        )
        sky.jobs.launch(task, name=f"worker-{i}")
else:
    # Run workers in-process (LocalBackend)
    for shard_batch in partitioned_shards:
        run_worker_local(shard_batch, spec, storage)
```

### Phase D: Training/eval integration (design now, implement later)

**Files to modify:**

- `configs/train.yaml` — add `compute_config: null`
- `configs/eval.yaml` — add `compute_config: null`
- `src/train.py` — check `cfg.get("compute_config")`, launch via SkyPilot if set
- `src/eval.py` — same

Training/eval SkyPilot integration is architecturally simpler than the pipeline — it's a single job (not N parallel workers). The entrypoint wraps the existing `train(cfg)` call in a SkyPilot task.

### Phase E: Design doc updates

**Files to modify:**

- `docs/design/data-pipeline.md` — §2, §7.3, §7.9, glossary
- `docs/design/data-pipeline-implementation-plan.md` — Phase 4 Task 4.2, Phase 6

## 8. Open Questions

### 8.1 Launch mode: fire-and-forget vs block

After `pipeline generate` launches managed jobs, should the CLI exit immediately or wait?

| Option                     | Pro                                                                      | Con                                                       |
| -------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------- |
| **Fire and forget**        | Matches "launch before bed" design goal. `pipeline status` checks later. | No inline progress feedback. User must remember to check. |
| **Block with progress**    | See shard completion in real-time. Single command to completion.         | Blocks terminal. Must handle Ctrl-C gracefully.           |
| **Both via `--wait` flag** | Best of both worlds. Default fire-and-forget, `--wait` for interactive.  | More CLI code.                                            |

**Recommendation:** Fire-and-forget as default (matches data-pipeline.md §2 design goal: "Two commands, no babysitting"). Add `--wait` later if needed (YAGNI).

### 8.2 SkyPilot as optional dependency

Should `skypilot` be a required or optional dependency?

**Recommendation:** Optional. `pip install synth-setter[cloud]` or similar extra. Local-only usage shouldn't require SkyPilot. Import lazily in the CLI when `compute_config` is set.

## 9. Verification

### Unit tests

- DatasetConfig accepts/rejects `num_workers` and `compute_config` values.
- Existing configs without new fields still validate (backward compat).
- `materialize_spec()` correctly resolves and embeds compute config content.
- Spec JSON round-trip with `compute_config`.

### Integration tests

- `pipeline generate` with `compute_config: null` runs workers locally (LocalBackend path).
- `pipeline generate` with `compute_config: configs/compute/vast-spot.yaml` calls SkyPilot SDK (mock SkyPilot in tests).
- Worker idempotency: start worker with some shards already `.valid` in R2 → skips them.

### E2E validation

- `sky check` confirms Vast.ai credentials.
- `sky show-gpus` confirms GPU availability.
- `sky launch --dryrun` with a compute config confirms pricing and provisioning.
- Full `pipeline generate` → `pipeline status` → `pipeline finalize` cycle on Vast.ai spot instance.
