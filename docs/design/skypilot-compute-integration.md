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

| Decision          | Choice                                                    | Rationale                                                                                                                                |
| ----------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Integration depth | Full managed jobs                                         | Spot recovery + R2 markers as natural checkpoints. Cost savings 3-5x on interruptible instances                                          |
| Local dev/test    | Keep LocalBackend                                         | In-process execution for fast unit tests. Two code paths (local vs SkyPilot)                                                             |
| Field name        | `compute_config` *(proposed)*                             | Tool-agnostic. Value is the resolved SkyPilot YAML *content* (embedded dict — see §3.1). Survives tool changes without schema migration  |
| Backend selection | Presence of `compute_config` *(proposed)*                 | `None` → local, dict → SkyPilot. No enum, no protocol, no extra plumbing                                                                 |
| Shard parallelism | `--num-workers` CLI flag on the launcher                  | Worker count is a launcher concern, not a `DatasetSpec` field; parallelism is recoverable from the per-job spec upload                   |
| Worker identity   | UUID generated at worker start                            | Decoupled from any provider. Fully portable                                                                                              |
| Deployment        | Docker image                                              | Reproducible; aligns with `src/synth_setter/pipeline/schemas/image_config.py` (see `docs/reference/docker.md`). SkyPilot pulls the image |
| CLI ownership     | `python -m synth_setter.pipeline generate` wraps SkyPilot | Single entry point. User never touches `sky` CLI directly for generation                                                                 |
| Frozen spec       | Include `compute_config` *(proposed)*                     | For provenance and cost tracking                                                                                                         |
| Scope             | Design all three config types, implement pipeline first   | DatasetSpec, train, eval all get `compute_config` *(proposed)*                                                                           |

## 3. Schema Changes

### 3.1 DatasetSpec (`src/synth_setter/pipeline/schemas/spec.py`) — *proposed, not yet implemented*

`DatasetConfig` + `DatasetPipelineSpec` were unified into a single `DatasetSpec` in [#887](https://github.com/tinaudio/synth-setter/pull/887) (the constructed Pydantic model **is** the artifact on R2; `model.model_dump_json()` is the JSON). The current schema (see [data-pipeline.md §14.1](data-pipeline.md#141-input-spec-schema) and `src/synth_setter/pipeline/schemas/spec.py`) has no compute-related field. The SkyPilot integration adds **one** new field:

```python
class DatasetSpec(BaseModel):
    # ... existing fields ...
    compute_config: dict[str, Any] | None = None        # Resolved SkyPilot YAML content, or None for local
```

- `compute_config` defaults to `None` (local execution, backward compatible).
- Optional field — existing dataset YAMLs continue to construct valid specs.
- `num_workers` is **not** a spec field — worker count is a launcher concern (see the `--num-workers` CLI flag in [§2 Architecture Decisions](#2-architecture-decisions)) and would conflate launcher provisioning with the reproducibility unit.

The SkyPilot YAML content is resolved (read from disk) before the dict reaches `DatasetSpec(**kwargs)` so the frozen spec carries a self-contained snapshot rather than a path; SkyPilot YAMLs are small (~20 lines), so embedding preserves full provenance without bloating the spec.

### 3.2 Training config (Hydra — `configs/train.yaml`)

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

The training entrypoint (`src/synth_setter/cli/train.py`) reads `cfg.get("compute_config")` and either trains locally or launches via SkyPilot SDK.

### 3.3 Eval config (Hydra — `configs/eval.yaml`)

Same pattern as training:

```yaml
# configs/eval.yaml
compute_config: null
```

## 4. New Files & Artifacts

### 4.1 SkyPilot YAML configs (`configs/compute/`)

The smoke pipeline ships three real templates:

```
configs/compute/
├── runpod-template.yaml      # RunPod GPU (primary smoke target)
├── oci-cpu-template.yaml     # OCI x86 CPU Flex (second smoke target)
└── local-template.yaml       # kind/kubernetes (sky local up; CI smoke only — see the YAML header for the CI-only resource shrink, PR #876)
```

All three share the launcher (`src/synth_setter/pipeline/skypilot_launch.py`),
the `dev-snapshot` Docker image, the R2-uploaded spec contract, and the
unified click-CLI dispatch (`scripts/docker_entrypoint.py generate_dataset`,
which carries the `os._exit(0)` defensive workaround for #735 inline).
They differ only in the `resources:` block (provider, accelerators vs.
CPU/memory floor, region) and the provider-specific credential setup in CI.
Future targets follow the same pattern.

The launcher (`synth_setter.pipeline.skypilot_launch`) does not override the `run:` block — it instantiates the Task from YAML and only calls `task.update_envs(...)` to inject the per-launch credential set + the spec URI. The `run:` block in each template handles the worker invocation; per-rank shard scoping is forwarded via `SYNTH_SETTER_WORKER_RANK` / `SYNTH_SETTER_NUM_WORKERS` envs.

#### 4.1.1 Launch mode (`--tail` / `--no-tail`)

The launcher accepts `--tail/--no-tail` (default `--no-tail`) — see the Click option in `src/synth_setter/pipeline/skypilot_launch.py` for the live help text and defaults. `--no-tail` waits for `sky.jobs.launch` + `sky.stream_and_get` to return a job_id per rank (the controller has accepted the job), prints `sky jobs logs --name <job_name>` and `sky jobs cancel --name <job_name>` commands the operator can run, and exits without tailing logs and without cancelling successfully-submitted jobs — the controller's terminal-status lifecycle releases the underlying compute on success/fail, so a clean launcher exit doesn't kill in-flight work. Half-submitted jobs (those whose `sky.jobs.launch`/`sky.stream_and_get` raised or yielded no job_id) are still cancelled in `--no-tail` so the controller doesn't accumulate orphan state; sibling jobs that launched cleanly are left running. `--tail` opts into `sky.jobs.tail_logs(follow=True)` and unconditional `finally`-block cancellation of every job; CI lanes that need exit-code-reflects-worker-success-and-uniform-cleanup pass `--tail` explicitly.

### 4.2 Env-var resolution: launcher → worker

The SkyPilot launcher (`synth_setter.pipeline.skypilot_launch`) needs to forward a small fixed set of secrets and configuration values from the operator's environment into the worker pod's environment. The contract is deliberately narrow — only the keys the worker actually reads — and the resolution is per-key so local dev and CI can share the same launcher code without special cases.

#### The forwarded set

Defined as `_WORKER_ENV_KEYS` in `src/synth_setter/pipeline/skypilot_launch.py`. The tuple is the source of truth for what the launcher forwards to the worker pod via `task.update_envs(...)`; the matching `envs:` block in `configs/compute/runpod-template.yaml` declares the same names with empty defaults so the SkyPilot Task validates as fully-specified before the launcher fills them.

Anything outside the tuple is *not* forwarded to the worker, even if it's set in the launcher's environment. Adding a key requires adding it both to `_WORKER_ENV_KEYS` and to the `envs:` block.

#### Resolution order (per key)

For each key in `_WORKER_ENV_KEYS`, the launcher takes the first value it finds:

1. The `.env` file at `--env-file` (default `<repo_root>/.env`), if the file exists and has the key.
2. The launcher's process env (`os.environ`), if the key is set.
3. Otherwise: skipped — the key keeps the SkyPilot template's default (typically `""`). If the worker actually needs it, rclone fails downstream with an actionable error.

This is per-key, not all-or-nothing — `.env` can resolve some keys and process env can resolve others in the same run.

#### Local dev story

Source of truth: a `.env` file at the repo root.

```bash
cp .env.example .env
$EDITOR .env  # fill in RCLONE_CONFIG_R2_* + WANDB_API_KEY
python -m synth_setter.pipeline.skypilot_launch \
    --experiment generate_dataset/smoke-shard
```

The launcher finds `<repo_root>/.env`, parses it via `python-dotenv`, and resolves all keys from there. Process env is a non-event because `.env` wins per key — useful when you have stale shell exports.

#### CI story

Source of truth: the GitHub-Actions runner's process env, populated from `secrets.*` and passed into the container via `docker run -e ...`. **No `.env` file is ever written to the runner's filesystem.** The launcher's default `--env-file` path doesn't resolve, the `.env` branch is silently skipped, and resolution falls through to the container's process env.

The SkyPilot launch step in `.github/workflows/test-dataset-generation.yml` (only fires on `runpod` / `oci` matrix cells; same-repo PRs run a non-SkyPilot `local` cell — see [github-actions.md](../reference/github-actions.md)):

```yaml
- name: Launch SkyPilot job
  env:
    RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
    RCLONE_CONFIG_R2_ACCESS_KEY_ID: ${{ secrets.RCLONE_CONFIG_R2_ACCESS_KEY_ID }}
    RCLONE_CONFIG_R2_SECRET_ACCESS_KEY: ${{ secrets.RCLONE_CONFIG_R2_SECRET_ACCESS_KEY }}
    RCLONE_CONFIG_R2_ENDPOINT: ${{ secrets.RCLONE_CONFIG_R2_ENDPOINT }}
    R2_ACCOUNT_ID: ${{ secrets.R2_ACCOUNT_ID }}
    WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
  run: |
    docker run --rm \
      -e RUNPOD_API_KEY \
      -e RCLONE_CONFIG_R2_TYPE=s3 \
      -e RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
      -e RCLONE_CONFIG_R2_ACCESS_KEY_ID \
      -e RCLONE_CONFIG_R2_SECRET_ACCESS_KEY \
      -e RCLONE_CONFIG_R2_ENDPOINT \
      -e R2_ACCOUNT_ID \
      -e WANDB_API_KEY \
      "$IMAGE" bash -c '... python -m synth_setter.pipeline.skypilot_launch ...'
```

Notes:

- `RCLONE_CONFIG_R2_TYPE=s3` and `RCLONE_CONFIG_R2_PROVIDER=Cloudflare` are hardcoded literals (constants for Cloudflare R2), not secrets.
- `RUNPOD_API_KEY` is intentionally **not** in `_WORKER_ENV_KEYS` — it's the launcher's own credential for SkyPilot's RunPod-API call, not the worker's. The launch step writes it to `~/.runpod/config.toml` inside the container so SkyPilot can read it (env var alone isn't enough for `sky check runpod`); it never gets forwarded to the worker.

#### Why each key lives where it does

| Where                                  | What                                                                                                                            | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Workflow YAML `env:` block             | `secrets.R2_*`, `secrets.WANDB_API_KEY`                                                                                         | GitHub-side secret materialization. Visible only to same-repo PRs (gated by the `if:` on the `generate` job).                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `docker run -e ...` flags              | Maps host `R2_*` → container `RCLONE_CONFIG_R2_*` (renamed at the boundary because rclone wants the `RCLONE_CONFIG_R2_` prefix) | Container env is the natural place for runtime secrets. No file persists on the runner.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Launcher's `_WORKER_ENV_KEYS`          | rclone-R2, WANDB, worker-spec/git-ref (the keys resolved from `.env` / process env)                                             | Defines the forwarding contract for keys that come *from the operator's environment*. Partition rank/world (`SYNTH_SETTER_WORKER_RANK` / `SYNTH_SETTER_NUM_WORKERS`) are NOT in this tuple — they're synthesized per-rank inside `_run_workers` and injected via `task.update_envs(...)` directly.                                                                                                                                                                                                                                                                                      |
| `runpod-template.yaml` `envs:` block   | Same keys with empty defaults                                                                                                   | Template-side schema declaration. Empty defaults make the Task YAML valid even before the launcher fills them. The interpreter-shutdown defensive workaround for [#735](https://github.com/tinaudio/synth-setter/issues/735) lives in the click `generate_dataset` subcommand of `scripts/docker_entrypoint.py` (force-`os._exit(0)` after `run()`) — applies to every invocation that goes through the click CLI, including the `local` matrix cell of `test-dataset-generation.yml`. The previous "SkyPilot-only consumer" framing predates the entrypoint unification (see PR #828). |
| `~/.runpod/config.toml` (in-container) | `RUNPOD_API_KEY`                                                                                                                | SkyPilot's RunPod backend reads from this file specifically; env var alone is insufficient for `sky check runpod`. Written with `umask 077` so the API key is 600. Skipped entirely when `SKYPILOT_API_SERVER_ENDPOINT` is set — the remote API server holds provider creds and the local SkyPilot client only needs the endpoint URL ([#785](https://github.com/tinaudio/synth-setter/issues/785)).                                                                                                                                                                                    |

#### Failure modes

- **No `.env` and no process env keys:** launcher fails fast with `No worker env vars resolved. Set the rclone-R2 keys in process env (e.g. via 'docker run -e RCLONE_CONFIG_R2_*=...') or populate <path>.`
- **`.env` exists but is empty or has only comments:** treated as no-keys-resolved (same path as above).
- **Some keys resolved, some missing:** the launcher proceeds; rclone fails downstream with its own error (typically "couldn't authenticate") because the worker upload requires all five `RCLONE_CONFIG_R2_*` values. The launcher's narrow contract here is intentional — it's not the launcher's job to enforce rclone's prerequisites.

#### One-line summary

**Local dev: `.env`. CI: `docker run -e ...`. Same launcher code, same resolution function, no special cases.**

### 4.3 Worker adaptation

The existing worker design (`src/synth_setter/pipeline/worker.py`, not yet implemented) stays mostly the same. Changes:

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

### Phase A: Schema changes (DatasetSpec)

**Files to modify:**

- `src/synth_setter/pipeline/schemas/spec.py` — add a `compute_config` field (optional, defaults to `None`) to `DatasetSpec`
- `configs/experiment/generate_dataset/surge-simple-480k-10k.yaml` — add an optional `compute_config` key, or leave it out for local execution
- Tests: `tests/pipeline/test_schemas/` — add test cases for the new field, backward compat

Note: `DatasetConfig`/`DatasetPipelineSpec`/`materialize_spec()` no longer exist as separate types — they unified into `DatasetSpec` ([#887](https://github.com/tinaudio/synth-setter/pull/887)) and the spec is now composed via Hydra (`spec_from_cfg(cfg)` from `configs/dataset.yaml` + an experiment override; the legacy `load_dataset_spec_yaml` bridge was removed in [#917](https://github.com/tinaudio/synth-setter/pull/917)).

### Phase B: SkyPilot compute configs

**Files to create:**

- `configs/compute/vast-spot.yaml`
- `configs/compute/vast-ondemand.yaml`

**Files to modify:**

- `pyproject.toml` — add `skypilot[vast]` as optional dependency

### Phase C: Pipeline CLI + SkyPilot integration

**Files to create/modify:**

- `src/synth_setter/pipeline/worker.py` — worker process with UUID identity, shard range args, idempotent resume
- `src/synth_setter/pipeline/cli.py` / `src/synth_setter/pipeline/__main__.py` — CLI that reconciles, partitions, launches
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
- `src/synth_setter/cli/train.py` — check `cfg.get("compute_config")`, launch via SkyPilot if set
- `src/synth_setter/cli/eval.py` — same

Training/eval SkyPilot integration is architecturally simpler than the pipeline — it's a single job (not N parallel workers). The entrypoint wraps the existing `train(cfg)` call in a SkyPilot task.

### Phase E: Design doc updates

**Files to modify:**

- `docs/design/data-pipeline.md` — §2, §7.3, §7.9, glossary
- `docs/design/data-pipeline-implementation-plan.md` — Phase 4 Task 4.2, Phase 6

## 8. Open Questions

### 8.1 SkyPilot as optional dependency

Should `skypilot` be a required or optional dependency?

**Recommendation:** Optional. `pip install synth-setter[cloud]` or similar extra. Local-only usage shouldn't require SkyPilot. Import lazily in the CLI when `compute_config` is set.

## 9. Verification

### Unit tests

- DatasetSpec accepts/rejects `compute_config` values (None and resolved-dict shapes).
- Existing configs without `compute_config` still validate (backward compat).
- The construction path correctly resolves and embeds compute config content (SkyPilot YAML read from disk before validation).
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
