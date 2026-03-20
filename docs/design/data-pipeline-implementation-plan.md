# Implementation Plan: Distributed Data Pipeline

> **Canonical design:** [data-pipeline.md](data-pipeline.md)
> **Tracking:** [#74](https://github.com/tinaudio/synth-setter/issues/74)
> **Builds on:** Generation infrastructure by benhayes@ (see design doc §1)
> **Last updated:** 2026-03-20

______________________________________________________________________

### Index

| §   | Section                                                       | GitHub issue               |
| --- | ------------------------------------------------------------- | -------------------------- |
| 1   | [Priorities & Conventions](#1-priorities--conventions)        | —                          |
| 2   | [Merge Path](#2-merge-path)                                   | #74                        |
| 3   | [Codebase Inventory](#3-codebase-inventory)                   | —                          |
| 4   | [Pipeline Config Schema](#4-pipeline-config-schema)           | —                          |
| 5   | [Phase 1: Foundation](#5-phase-1-foundation-68)             | #68                        |
| 6   | [Phase 2: Pipeline Core](#6-phase-2-pipeline-core-69)       | #69                        |
| 7   | [Phase 3: Docker Infrastructure](#7-phase-3-docker-infrastructure-70) | #70                        |
| 8   | [Phase 4: Pipeline Engine](#8-phase-4-pipeline-engine-71)   | #71                        |
| 9   | [Phase 5: Pipeline CLI](#9-phase-5-pipeline-cli-72)         | #72                        |
| 10  | [Phase 6: Production & E2E](#10-phase-6-production--e2e-73) | #73                        |
| 11  | [Cross-cutting work](#11-cross-cutting-work)                  | #76, #77, #120, #121, #122 |
| 12  | [Verification Strategy](#12-verification-strategy)            | —                          |
| 13  | [Assumptions](#13-assumptions)                                | —                          |

______________________________________________________________________

## 1. Priorities & Conventions

**Priorities (in order):**

1. Implementation tasks (2.1–2.3, 4.1–6.1) have integration + unit tests, written before implementation (TDD)
2. Infrastructure tasks (1.1–1.5, 3.1) are verified via CI green / Docker builds — not test-first TDD
3. Small PRs (one task = one PR)
4. Always-working pipeline — CI validates every PR

- `pipeline/` at project root (not `src/`) — invoked via `python -m pipeline`
- Tests in `tests/pipeline/` with own `conftest.py`

______________________________________________________________________

## 2. Merge Path

```
main ──●──────────●────────────●──────────●──────────●──────────●──→
       │          │            │          │          │          │
    Phase 1    Phase 2      Phase 3    Phase 4    Phase 5    Phase 6
      #68        #69          #70        #71        #72        #73
```

| Phase                                        | Tasks   | Contents                                 | CI gate                         |
| -------------------------------------------- | ------- | ---------------------------------------- | ------------------------------- |
| **Phase 1: Foundation** #68                  | 1.1–1.5 | Deps, uploader, design doc, CI setup     | `pytest` passes, ruff clean     |
| **Phase 2: Pipeline Core** #69               | 2.1–2.3 | Schemas, storage, validation             | `pytest tests/pipeline/` passes |
| **Phase 3: Docker Infrastructure** #70       | 3.1     | Dockerfile, entrypoint, headless, Make   | Docker build succeeds, BATS     |
| **Phase 4: Pipeline Engine** #71             | 4.1–4.2 | Reconciliation, compute backend + worker | `pytest tests/pipeline/` passes |
| **Phase 5: Pipeline CLI** #72                | 5.1–5.3 | Generate, status, finalize commands      | Full integration tests pass     |
| **Phase 6: Production & E2E** #73           | 6.1     | RunPod backend, Docker updates, E2E      | E2E test + adhoc Docker test    |

**6 phases, 15 tasks (one PR per task)**

______________________________________________________________________

## 3. Codebase Inventory

### On `main` already (no porting needed)

- `src/data/vst/generate_vst_dataset.py` — VST audio generation (worker calls this)
- `scripts/reshard_data.py` — HDF5 virtual dataset resharding
- Basic `Makefile` (help/clean targets only)
- All model/training code, configs, notebooks

### NOT ported (stays on `experiment`)

- `scripts/generate_shards.py`, `finalize_shards.py`, `runpod_launch.py`, `runpod_stop.py`,
  `run_dataset_pipeline.py`, `reshard_data_dynamic_shard.py` + all their tests
- `tests/test_entrypoint.bats`
- `scripts/setup-dev.sh`, `scripts/setup-rclone.sh`
- `.devcontainer/`, `docs/pipeline.md`
- `.github/copilot-instructions.md`, `.github/workflows/data-pipeline.yml`

______________________________________________________________________

## 4. Pipeline Config Schema

Matches design doc §14.5:

```yaml
# configs/pipeline/surge_simple_480k.yaml
experiment_name: surge_simple
param_spec: surge_simple
plugin_path: plugins/Surge XT.vst3    # renderer_version auto-extracted from bundle
output_format: hdf5                   # "hdf5" (local training) or "wds" (multi-GPU streaming)
sample_rate: 16000
shard_size: 10000
num_shards: 48
base_seed: 42

splits:
  train: 44
  val: 2
  test: 2

# Generation params (needed by generate_vst_dataset)
preset_path: presets/surge-base.vstpreset
channels: 2
velocity: 100
signal_duration_seconds: 4.0
min_loudness: -55.0
sample_batch_size: 32
```

CLI (compute/storage are not in config):

```bash
python -m pipeline generate \
  --config configs/pipeline/surge_simple_480k.yaml \
  --workers 10 --backend runpod --image tinaudio/perm:dev-snapshot-abc1234
```

**Renderer version:** Auto-extracted at materialization from VST3 bundle
(`Info.plist` → `CFBundleShortVersionString` on macOS, `moduleinfo.json` or
`SURGE_XT_VERSION` env on Linux). Fallback: `"unknown"` with warning.

**Output format:** `hdf5` produces virtual HDF5 datasets (`train.h5`, `val.h5`, `test.h5`).
`wds` produces WebDataset tar archives (`train-{shard}.tar`, etc.) for multi-GPU streaming.
Generation always produces HDF5 shards regardless of output format — `wds` is a finalize
transcoding step.

______________________________________________________________________

## 5. Phase 1: Foundation ([#68](https://github.com/tinaudio/synth-setter/issues/68))

### Task 1.1: Dependencies & Tooling ([#78](https://github.com/tinaudio/synth-setter/issues/78)) ✅ — Completed in PR [#75](https://github.com/tinaudio/synth-setter/pull/75)

**Goal:** Port build dependencies and code quality tooling from `experiment`.

**Files created/modified:**

- `requirements-app.txt` (new) — pydantic, h5py, click, runpod, wandb, structlog, tenacity, webdataset, numpy, etc.
- `requirements-torch.txt` (new) — torch index URL + packages
- `requirements.txt` (updated) — slimmed to `-r` includes
- `pyproject.toml` — added `pipeline` pytest marker
- `checkmake.ini` (new)

**Verification:** `pip install -r requirements.txt && ruff check . && pytest tests/ -x`

**Design notes:**

- `hdf5plugin` included in deps — required at read time for Blosc2-compressed virtual
  datasets (B6). Task 5.3 finalize and all HDF5 tests must `import hdf5plugin`.
- `pydantic`, `structlog`, `tenacity`, `click`, `pyyaml`, `webdataset` added beyond what
  `experiment` has (R13).
- `mutmut` added to dev dependencies — required for verification strategy (§12).

______________________________________________________________________

### Task 1.2: Core Shared Code ([#79](https://github.com/tinaudio/synth-setter/issues/79))

**Goal:** Port `uploader.py` and minor fixes that the pipeline depends on.

**Files to port from `experiment`:**

- `src/data/uploader.py` (new) — `DatasetUploader` protocol, `RcloneUploader`, `LocalFakeUploader`
- `src/train.py` — minor fixes (resolver registration)
- `src/utils/utils.py` — minor fixes
- `src/data/ksin_datamodule.py` — pin_memory fix
- `src/data/surge_datamodule.py` — fix
- `tests/conftest.py` — register resolvers, lr_monitor fix
- `tests/helpers/package_available.py` — importlib.metadata migration
- `tests/helpers/run_if.py` — fix

**Verification:** `pytest tests/ -x` — all existing tests still pass

______________________________________________________________________

### Task 1.3: Design Doc & Config ([#80](https://github.com/tinaudio/synth-setter/issues/80))

**Goal:** Ensure design doc and environment config are on `main`.

- `docs/design/data-pipeline.md` — ✅ already on main
- `.env.example` — moved to standalone issue [#82](https://github.com/tinaudio/synth-setter/issues/82)

______________________________________________________________________

### Task 1.4: CI Setup ([#81](https://github.com/tinaudio/synth-setter/issues/81))

**Goal:** Ensure every subsequent PR is validated by CI.

**Files to create/modify:**

- `.github/workflows/test.yml` — ✅ already runs `pytest tests/` + `ruff check`
- `.github/workflows/pipeline-ci.yml` (new) — runs `pytest tests/pipeline/ -v`
  on push to dev branch and on PRs to `main`

**Key behaviors:**

- Runs on every push and PR
- Installs dependencies from `requirements.txt`
- Runs ruff lint + pytest (existing tests + pipeline tests as they're added)
- Fails the PR if any test fails

**Verification:** Push to dev branch → CI runs → green

### Task 1.5: .env.example ([#82](https://github.com/tinaudio/synth-setter/issues/82))

**Goal:** Add `.env.example` with R2, W&B, RunPod env var template so new contributors
know which credentials are needed without exposing real values.

**Files to create:**

- `.env.example` — template with placeholder values for `R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`, `WANDB_API_KEY`, `RUNPOD_API_KEY`

**Verification:** File exists, `.env` is in `.gitignore`, no real credentials committed.

______________________________________________________________________

## 6. Phase 2: Pipeline Core ([#69](https://github.com/tinaudio/synth-setter/issues/69))

Sub-issues: [#18](https://github.com/tinaudio/synth-setter/issues/18) (config-driven runs), [#20](https://github.com/tinaudio/synth-setter/issues/20) (schema versioning), [#22](https://github.com/tinaudio/synth-setter/issues/22) (deterministic shard assignment)

### Task 2.1: Pydantic Schemas ([#18](https://github.com/tinaudio/synth-setter/issues/18), [#20](https://github.com/tinaudio/synth-setter/issues/20), [#22](https://github.com/tinaudio/synth-setter/issues/22))

**Goal:** Define the data models that everything else depends on.

**Files to create:**

- `pipeline/__init__.py`
- `pipeline/schemas.py` — `RunConfig`, `PipelineSpec`, `ShardSpec`, `WorkerReport`, `ShardResult`, `DatasetCard`, `ValidationSummary`, `Sample`
- `configs/pipeline/surge_simple_480k.yaml` — sample config
- `tests/pipeline/__init__.py`
- `tests/pipeline/test_schemas.py`

**Key behaviors:**

- `RunConfig` (Pydantic strict): validates raw YAML input. Fields match config schema (§4).
  `output_format` defaults to `"hdf5"` if missing from config.
- `PipelineSpec` (frozen, strict): `run_id`, `created_at`, `code_version`, `is_repo_dirty`,
  `param_spec`, `renderer_version`, `output_format` (`"hdf5"` or `"wds"`), `sample_rate`,
  `shard_size`, `num_shards`, `base_seed`, `splits` (`{"train": N, "val": N, "test": N}`),
  `shards` (list of `ShardSpec`), plus generation params.
  Splits use explicit `{train: N, val: N, test: N}` matching design doc §14.4.
  Validation: `train + val + test == num_shards`.
- `ShardSpec`: `shard_id: int`, `filename: str` (`"shard-000042.h5"`), `seed` (= `base_seed + shard_id`),
  `row_start`, `row_count`, `expected_datasets`, `audio_shape`, `mel_shape`, `param_shape`.
  `shard_id` is int in schema; formatted to string for paths via `shard_dir_name(shard_id) -> str`.
- `Sample` dataclass (frozen, slots): `sample_id: int`, `audio`, `mel_spec`, `params` —
  typed container for HDF5→WDS transcoding (not Pydantic, already-validated data).
  Pydantic for trust boundaries, dataclass for internal typed containers.
- `ShardResult` (inside `WorkerReport`): `shard_id: int`, `filename: str`, `rows: int`,
  `success: bool`, `content_hash: str | None` (SHA-256), `render_time_sec: float`, `error: str | None`.
- `WorkerReport`: includes `cpu_arch`, `os_info`, `attempt_uuid`, `results: list[ShardResult]`.
- `ValidationSummary`: `valid: int`, `quarantined: int`, `quarantined_shards: list[str]`.
- `DatasetCard`: `schema_version`, `run_id`, `finalized_at`, `code_version`, `is_repo_dirty`,
  `param_spec`, `renderer_version`, `output_format`, `sample_rate`, `total_samples`,
  `splits` (sample counts, not shard counts), `stats`, `validation_summary`,
  `worker_architectures` (list of unique CPU archs), `shard_manifest: list[dict]`
  (per-shard `{shard_id, filename, content_hash}`), `input_spec_sha256`, `input_spec_path`.
- Run ID format: `{experiment_name}-{total_samples}-{shard_size}-{YYYYMMDD-HHMMSS}` (human-friendly k/M).
  Uses `total_samples` not `total_train_samples` — design doc §14.6 text is a bug (example is correct).
- `materialize_spec(config: RunConfig, timestamp=None, renderer_version=None) -> PipelineSpec`.
  Optional `renderer_version` override for testing; test fixtures pass `"test-1.0"` explicitly.

**Design doc schema gaps to fix alongside this task:**

- `ValidationSummary` class not defined in design doc (referenced in `DatasetCard` §14.2)
- `base_seed` not in `PipelineSpec` schema §14.1 (referenced in text)
- Generation params (preset_path, channels, etc.) not in `PipelineSpec` schema §14.1
- `shard_manifest` not in `DatasetCard` schema §14.2 (mentioned in §7.6 prose)

**Unit tests (write first):**

- Construction, strict validation, immutability, JSON round-trip
- `materialize_spec` — correct shard count, deterministic seeds, zero-padded IDs
- Row partitioning without gaps/overlaps

**Reference test:**

```python
def test_spec_materialization_end_to_end(tmp_path):
    """Config dict -> materialize -> serialize -> deserialize -> verify integrity."""
    config = {
        "experiment_name": "test_run", "base_seed": 42,
        "num_shards": 10, "shard_size": 1000,
        "splits": {"train": 8, "val": 1, "test": 1},
        "param_spec": "surge_simple", "plugin_path": "plugins/Surge XT.vst3",
        "output_format": "hdf5",
        "preset_path": "presets/surge-base.vstpreset", "sample_rate": 44100,
        "channels": 2, "velocity": 100, "signal_duration_seconds": 4.0,
        "min_loudness": -55.0, "sample_batch_size": 32,
    }
    fixed_ts = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    spec = materialize_spec(RunConfig(**config), timestamp=fixed_ts)
    spec2 = PipelineSpec.model_validate_json(spec.model_dump_json())

    assert len(spec2.shards) == 10
    assert spec2.shards[0].shard_id == 0
    assert spec2.shards[0].filename == "shard-000000.h5"
    assert spec2.shards[0].seed == 42
    assert spec2.shards[5].seed == 47
    assert spec2.output_format == "hdf5"
    assert sum(s.row_count for s in spec2.shards) == 10_000
    assert materialize_spec(RunConfig(**config), timestamp=fixed_ts).model_dump() == spec2.model_dump()
```

______________________________________________________________________

### Task 2.2: Storage Layer ([#102](https://github.com/tinaudio/synth-setter/issues/102))

**Goal:** Abstract R2/local filesystem with design doc's path layout. Wraps `src/data/uploader.py`.

**Files to create:**

- `pipeline/storage.py` — `StorageBackend` protocol, `LocalStorageBackend`, `R2StorageBackend`
- `tests/pipeline/test_storage.py`

**Key behaviors:**

- Path computation matching design doc §6 R2 layout, including helpers for:
  - Shard lifecycle: `write_rendering_marker`, `write_valid_marker`, `write_invalid_marker`
  - Quarantine: `upload_to_quarantine(run_id, shard_id, worker_id, attempt, local_path)`
  - Worker attempts: `upload_report(run_id, worker_id, attempt, report)`,
    `upload_debug_log(run_id, worker_id, attempt, log_path)`
  - Finalize outputs: paths for `data/shards/`, `data/train.h5`, `data/stats.npz`,
    `metadata/dataset.json`, `metadata/dataset.complete`
- `StorageBackend` protocol: `list_shard_markers`, `write_marker`, `upload_file`,
  `download_file`, `list_prefix`, `exists`
- `LocalStorageBackend`: filesystem-based
- `R2StorageBackend`: wraps `RcloneUploader.upload()` for directory uploads, adds
  `rclone copyto` (single file), `rclone lsf` (list), `rclone lsjson` (exists) for
  file-level ops. All rclone operations include `--checksum` (design doc §11.2).

**Unit tests (write first):**

- Path generation matches design doc for all artifact types
- Local: write → exists → list round-trip
- R2: rclone command construction (mock subprocess) — verify `--checksum` in every command
- R2: delegates to `RcloneUploader.upload()` for directory uploads

**Reference test:**

```python
def test_storage_shard_lifecycle(tmp_path):
    """Write complete shard lifecycle, verify directory layout matches design doc."""
    storage = LocalStorageBackend(root=tmp_path)
    run_id, shard_id = "test-10k-1k-20260315-120000", 42
    worker_id, attempt = "pod-abc", "uuid1234"

    storage.write_rendering_marker(run_id, shard_id, worker_id, attempt)
    storage.upload_file(_create_fake_h5(tmp_path / "local.h5"), run_id,
        f"metadata/workers/shards/shard-{shard_id:06d}/{worker_id}-{attempt}.h5")
    storage.write_valid_marker(run_id, shard_id, worker_id, attempt)

    markers = storage.list_shard_markers(run_id, shard_id)
    assert f"{worker_id}-{attempt}.rendering" in markers
    assert f"{worker_id}-{attempt}.h5" in markers
    assert f"{worker_id}-{attempt}.valid" in markers

    shard_dir = tmp_path / run_id / "metadata/workers/shards" / f"shard-{shard_id:06d}"
    assert (shard_dir / f"{worker_id}-{attempt}.valid").exists()
```

______________________________________________________________________

### Task 2.3: Shard Validation ([#103](https://github.com/tinaudio/synth-setter/issues/103))

**Goal:** 3-tier validation from design doc §7.5.

**Files to create:**

- `pipeline/validation.py`
- `tests/pipeline/test_validation.py`
- `tests/pipeline/conftest.py` — shared HDF5 fixture factories (`_make_test_spec`,
  `_make_fixture_shard`)

**Key behaviors:**

- **Full** (4 checks — workers): structural, shape, value, row count
- **Existence** (generate/status): `.h5` + `.valid` marker
- **Structural** (finalize): open h5py, datasets present, shapes match
- Returns `ValidationResult(is_valid, checks: list[CheckResult])`
- Pure functions (functional core)

**`_make_test_spec` helper** (defined in `tests/pipeline/conftest.py`):
Returns a valid `PipelineSpec` with sensible defaults: `renderer_version="test"`,
`code_version="abc1234"`, `run_id` derived from params, `output_format="hdf5"`.
Accepts `num_shards`, `shard_size`, `output_format` overrides.

**`tests/pipeline/conftest.py` also adds project root to `sys.path`** if needed (same
pattern as existing `rootutils.setup_root()`).

**Unit tests (write first):**

- Valid shard passes all 4; corrupt/NaN/wrong-shape/wrong-count each fail the right check
- Existence check: with/without `.valid` marker
- Truncated file fails gracefully

**Reference test:**

```python
def test_tiered_validation_catches_correct_failures(tmp_path):
    """Each tier catches exactly the failures it's responsible for."""
    spec = _make_test_spec(shard_size=100, num_shards=4)
    good = _make_fixture_shard(tmp_path / "good.h5", 100)
    nan = _make_fixture_shard(tmp_path / "nan.h5", 100, inject_nan=True)
    bad_shape = _make_fixture_shard(tmp_path / "shape.h5", 100, wrong_shape=True)
    truncated = tmp_path / "trunc.h5"; truncated.write_bytes(b"not hdf5")

    s = spec.shards[0]
    assert validate_full(good, s).is_valid
    assert not validate_full(nan, s).is_valid
    assert not validate_full(bad_shape, s).is_valid
    assert not validate_full(truncated, s).is_valid

    assert validate_structural(nan, s).is_valid      # NaN not caught at this tier
    assert not validate_structural(bad_shape, s).is_valid
    assert not validate_structural(truncated, s).is_valid
```

______________________________________________________________________

## 7. Phase 3: Docker Infrastructure ([#70](https://github.com/tinaudio/synth-setter/issues/70))

Sub-issue: [#7](https://github.com/tinaudio/synth-setter/issues/7) (buildx TARGET_ARCH)

### Task 3.1: Docker Infrastructure ([#70](https://github.com/tinaudio/synth-setter/issues/70), [#7](https://github.com/tinaudio/synth-setter/issues/7))

**Goal:** Port Docker build system from `experiment`. Needed for worker containers.

**Files to port from `experiment`:**

- `docker/ubuntu22_04/Dockerfile` — multi-stage build with BuildKit secrets
- `scripts/docker_entrypoint.sh` — container dispatch (existing modes only for now)
- `scripts/run-linux-vst-headless.sh` — Xvfb wrapper for headless VST
- `Makefile` additions — `docker-build-dev-snapshot`, `docker-build-dev-live`, etc.

**Verification:**

- `make docker-build-dev-snapshot` builds successfully
- Existing entrypoint modes work (BATS test if ported, otherwise manual check)

______________________________________________________________________

## 8. Phase 4: Pipeline Engine ([#71](https://github.com/tinaudio/synth-setter/issues/71))

Sub-issues: [#3](https://github.com/tinaudio/synth-setter/issues/3) (vst/core.py throughput), [#23](https://github.com/tinaudio/synth-setter/issues/23) (VST generation throughput)

### Task 4.1: Reconciliation Engine ([#104](https://github.com/tinaudio/synth-setter/issues/104))

**Goal:** Compare spec against storage state to determine remaining work.

**Files to create:**

- `pipeline/reconcile.py`
- `tests/pipeline/test_reconcile.py`

**Key behaviors:**

- `reconcile(spec, storage) -> ReconciliationResult` with `missing`, `valid`,
  `rendering`, `invalid` sets (shard IDs as ints)
- Shard state from markers: `.valid` → valid, `.rendering` only → rendering,
  `.invalid` → invalid, nothing → missing
- Multiple attempts: `.valid` wins
- `partition_work(shard_specs: list[ShardSpec], num_workers: int) -> list[list[ShardSpec]]`.
  Caller filters `spec.shards` by missing shard_ids from reconciliation result, then passes
  the filtered list.

**Unit tests (write first):**

- All missing, all valid, mixed states, multiple attempts, partition edge cases

**Reference test:**

```python
def test_reconciliation_mixed_state(tmp_path):
    storage = LocalStorageBackend(root=tmp_path)
    spec = _make_test_spec(num_shards=10, shard_size=100)
    for i in range(7):
        _write_valid_shard(storage, spec.run_id, spec.shards[i].shard_id, tmp_path)
    _write_rendering_only(storage, spec.run_id, spec.shards[7].shard_id)
    _write_invalid_shard(storage, spec.run_id, spec.shards[8].shard_id)
    # shard 9: no markers (missing)

    result = reconcile(spec, storage)
    assert len(result.valid) == 7
    assert result.rendering == {7}      # shard_id is int
    assert result.invalid == {8}
    assert result.missing == {9}
```

______________________________________________________________________

### Task 4.2: ComputeBackend + Worker ([#105](https://github.com/tinaudio/synth-setter/issues/105))

**Goal:** Compute abstraction + worker-side shard generation with lifecycle markers.

**Files to create:**

- `pipeline/backends/__init__.py`, `pipeline/backends/base.py`, `pipeline/backends/local.py`
- `pipeline/worker.py`
- `tests/pipeline/test_backends.py`, `tests/pipeline/test_worker.py`

**Key behaviors — Worker:**

- `run_worker(task_spec, storage, max_workers=None)` — manages concurrent
  shard rendering with per-shard process isolation.
- **Per-shard process isolation:** Each shard renders in a separate OS process via
  `multiprocessing.get_context("spawn").Process(...)`. The parent worker spawns one child
  per shard, catches exit codes (0 = success, -11 = SIGSEGV, -9 = OOM kill), and handles
  failures without losing other in-progress shards. `spawn` starts a fresh Python
  interpreter per child — no inherited VST plugin state, no shared mutable globals.
  See design doc §7.8.1 for full trade-off analysis.
- Per-shard lifecycle: write `.rendering` to **remote** storage FIRST → spawn child process
  → child imports and calls `make_dataset(shard_path, shard_spec)` →
  parent waits with `join(timeout=SHARD_TIMEOUT)` → on success: validate locally →
  upload `.h5` to storage → write `.valid` to storage.
  `.rendering` in remote storage survives worker/child death (crash resilience).
- On validation failure or child crash: upload corrupt shard to `quarantine/`, write
  `.invalid` marker, log failure details including exit code (design doc §7.2).
  `.rendering` marker remains (append-only, never deleted).
- Skip-if-valid optimization: before spawning, check staging directory for an existing
  `.valid` shard. If found, skip and move to next shard. Optimization, not correctness
  requirement (design doc §7.7).
- Produces `WorkerReport` with per-shard results, content hashes (SHA-256), timing,
  exit codes for crashed shards.
- Creates JSONL debug log via structlog file handler to a known local path so the bash
  EXIT trap can upload it on crash (design doc §7.8, Appendix E.1).
- Per-shard timeout via `join(timeout=SHARD_TIMEOUT)` — timed-out children are killed
  with `p.kill()`, shard marked invalid.
- **Xvfb display isolation:** Each child process should use a per-process X11 display
  number (`:N` derived from PID or shard ID) to avoid contention in headless VST rendering.
- **No `generate_fn` argument:** The child process imports `make_dataset` directly
  (`from pipeline.vst import make_dataset`). Under `spawn`, the child is a fresh
  interpreter, so the import is clean. No pickling concerns — only `shard_spec` and
  `shard_path` cross the process boundary. For tests, `LocalBackend` calls
  `run_worker()` in-process (no spawn), so test fixtures can inject a fake function.

**Key behaviors — Backend:**

- `ComputeBackend.submit(image, task_specs) -> list[SubmittedTask]` — fire-and-forget
- `TaskSpec` model defined here: `TaskSpec(run_id, shards, spec)` — a backend concern
- `LocalBackend`: calls `run_worker()` in-process (intentional deviation from design doc
  §7.9 which says Docker; in-process is faster for tests; `test_local_docker.sh` validates
  Docker container behavior)

**Unit tests (write first):**

- Worker lifecycle marker ordering — assert `.rendering` exists in storage before `.valid`
- Quarantine path: validation failure → `.invalid` marker + shard in `quarantine/`
- Process crash isolation: test spawns a child process via `_render_shard` (same
  spawn path as production) with a module-level `crash_generate` that calls
  `os.kill(os.getpid(), signal.SIGSEGV)` → parent sees `exitcode == -11`,
  marks shard invalid, continues to next. Cannot use `LocalBackend` in-process
  mode for this — must exercise the real spawn path.
- Per-shard timeout: test spawns a child via `_render_shard` with a module-level
  `slow_generate` that sleeps forever → child killed after timeout, shard marked invalid
- Failure isolation, report generation, content hashes
- LocalBackend submit + metadata

**Reference test:**

```python
def test_local_backend_generates_shards_with_lifecycle(tmp_path):
    storage = LocalStorageBackend(root=tmp_path / "storage")
    spec = _make_test_spec(num_shards=3, shard_size=100)

    def fake_generate(shard_path, shard_spec):
        _make_fixture_shard(shard_path, n_samples=shard_spec.row_count)

    # LocalBackend runs in-process (no spawn), so closures are fine here.
    # Production worker uses _render_shard with direct import instead.
    backend = LocalBackend(storage=storage, generate_fn=fake_generate)
    backend.submit("unused", [TaskSpec(run_id=spec.run_id, shards=spec.shards, spec=spec)])

    for shard in spec.shards:
        markers = storage.list_shard_markers(spec.run_id, shard.shard_id)
        assert any(m.endswith(".valid") for m in markers)
        assert any(m.endswith(".h5") for m in markers)

    reports = [f for f in storage.list_prefix(spec.run_id, "metadata/workers/attempts/")
               if f.endswith("report.json")]
    assert len(reports) >= 1
```

______________________________________________________________________

## 9. Phase 5: Pipeline CLI ([#72](https://github.com/tinaudio/synth-setter/issues/72))

Sub-issues: [#17](https://github.com/tinaudio/synth-setter/issues/17) (modular CLI), [#19](https://github.com/tinaudio/synth-setter/issues/19) (WebDataset output), [#21](https://github.com/tinaudio/synth-setter/issues/21) (reconciliation status)

### Task 5.1: CLI — `generate` ([#17](https://github.com/tinaudio/synth-setter/issues/17))

**Goal:** Unified CLI entry point via `python -m pipeline` (via `__main__.py` importing
Click group from `cli.py`).

**Files to create:**

- `pipeline/__main__.py`, `pipeline/cli.py`, `pipeline/stages/__init__.py`,
  `pipeline/stages/generate.py`, `pipeline/logging_config.py`, `pipeline/retry.py`
- `tests/pipeline/test_cli_generate.py`

**Key behaviors:**

- `python -m pipeline generate --config <yaml> --workers N --backend local|runpod`
- `--storage-root` (local path for LocalBackend, or env-based for R2)
- `--log-level` (default: `INFO`, options: `DEBUG`, `INFO`, `WARNING`)
- Auth validation: check R2 connectivity + RunPod API key before launching.
  On failure: clear error message, exit 1, no workers launched.
- Early validation: check `plugin_path` exists before materialization — actionable
  error if VST3 bundle not found (avoids unclear renderer_version extraction failure).
- First run: config → validate → extract `renderer_version` → materialize spec →
  upload frozen spec to `metadata/input_spec.json` + source config to
  `metadata/config.yaml` (provenance copy) → if `is_repo_dirty`, upload
  `git diff` to `metadata/run_diff.patch` → reconcile → partition → submit → exit.
  Print `run_id` prominently so user can use it for status/finalize.
- Retry: `--run-id` → load spec → reconcile → submit missing → exit
- `--config` with existing run_id → error (immutable spec)
- `--dry-run` prints plan without submitting

**Unit tests (write first):**

- First run creates spec, retry loads existing, config drift error
- `--dry-run` prints shard assignments, creates no spec, submits no work
- Auth validation failure: missing R2 credentials → clear error, exit 1, no workers
- `plugin_path` validation: nonexistent path → actionable error before materialization

**Reference test:**

```python
def test_generate_cli_end_to_end(tmp_path):
    config_path = _write_test_config(tmp_path, num_shards=4, shard_size=100)
    storage_root = tmp_path / "storage"
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--config", str(config_path),
        "--backend", "local", "--workers", "2", "--storage-root", str(storage_root)])
    assert result.exit_code == 0

    run_id = _extract_run_id(storage_root)
    storage = LocalStorageBackend(root=storage_root)
    for i in range(4):
        assert any(m.endswith(".valid")
            for m in storage.list_shard_markers(run_id, i))  # shard_id is int
    assert storage.exists(run_id, "metadata/input_spec.json")  # frozen PipelineSpec
    assert storage.exists(run_id, "metadata/config.yaml")  # source YAML (provenance)
```

______________________________________________________________________

### Task 5.2: CLI — `status` ([#21](https://github.com/tinaudio/synth-setter/issues/21))

**Goal:** Read-only reconciliation report.

**Files to create:** `tests/pipeline/test_cli_status.py`
**Files to modify:** `pipeline/cli.py`

**Key behaviors:**

- `python -m pipeline status --run-id <id> --storage-root <path>` — prints shard counts, missing IDs
- `--json` flag for machine-readable output (design doc §7.4 shows both table and structured output)
- Overlay recent worker errors from `metadata/workers/attempts/*/report.json` when
  available (design doc §7.4 status output example)
- Exit 0 if all valid, 1 otherwise. No writes to storage.

**Unit tests:** All-valid exit 0, missing exit 1, invalid shows details, `--json` output parseable

**Reference test:**

```python
def test_status_after_partial_generate(tmp_path):
    storage = LocalStorageBackend(root=tmp_path / "storage")
    spec = _make_test_spec(num_shards=5, shard_size=100)
    _upload_spec(storage, spec)
    for i in range(3):
        _write_valid_shard(storage, spec.run_id, spec.shards[i].shard_id, tmp_path)

    result = CliRunner().invoke(cli, ["status", "--run-id", spec.run_id,
        "--storage-root", str(tmp_path / "storage")])
    assert result.exit_code == 1
    assert "shard-000003" in result.output
```

______________________________________________________________________

### Task 5.3: CLI — `finalize` ([#19](https://github.com/tinaudio/synth-setter/issues/19))

**Goal:** Validate staged → promote → download → stats → training outputs → dataset card.

**Files to create:** `pipeline/stages/finalize.py`, `tests/pipeline/test_cli_finalize.py`
**Files to modify:** `pipeline/cli.py`

**Key behaviors:**

- `--output-dir` (local download target), `--skip-wandb`, `--keep-quarantine-days` (default: keep all), `--dry-run`
- Already-finalized → exit 0 (idempotent)
- Stale `dataset.complete` (outputs missing/corrupt) → delete marker, re-run
- Missing shards → exit 1
- Structural-check each staged shard; multiple attempts → pick lexicographically smallest
  `{worker_id}-{attempt_uuid}` filename (deterministic, no clock dependency)
- Promote to `data/shards/`, write `.promoted` markers (staged files NOT deleted)
- Compute stats FIRST, then produce training outputs based on `output_format`:
  - `hdf5`: virtual HDF5 datasets (`train.h5`, `val.h5`, `test.h5`) — implements fresh
    resharding using `VirtualLayout`/`VirtualSource` pattern, reading actual shard dimensions
    from HDF5 metadata. Does NOT call `reshard_data.py` (it hardcodes 10k shard size).
  - `wds`: WebDataset tar archives (`train-{shard}.tar`, etc.) via `Sample` dataclass
- Dataset card includes `output_format`, `worker_architectures` (logs warning if
  heterogeneous), content hashes, shard manifest
- Upload finalized outputs to R2 storage
- `dataset.complete` contains `run_id` + timestamp (written last)
- W&B integration: logs 7 metrics (`pipeline/shards_total`, `pipeline/shards_valid`,
  `pipeline/shards_quarantined`, `pipeline/total_samples`, `pipeline/generation_time_seconds`,
  `pipeline/finalize_time_seconds`, `pipeline/errors_total`) + registers dataset artifact

**Unit tests:** Promotes, rejects missing/corrupt, idempotent, stale marker recovery,
lexicographic shard selection with multiple attempts, `.promoted` markers written,
`dataset.complete` content verified, card contents, both output formats, `--dry-run`,
mock-W&B test verifying all 7 metrics logged

**Reference tests:**

```python
def test_full_generate_then_finalize(tmp_path):
    storage_root, output_dir = tmp_path / "storage", tmp_path / "output"
    config_path = _write_test_config(tmp_path, num_shards=5, shard_size=100,
                                      val_shards=1, test_shards=1)
    runner = CliRunner()
    runner.invoke(cli, ["generate", "--config", str(config_path), "--backend", "local",
        "--workers", "1", "--storage-root", str(storage_root)])
    run_id = _extract_run_id(storage_root)

    result = runner.invoke(cli, ["finalize", "--run-id", run_id,
        "--storage-root", str(storage_root), "--output-dir", str(output_dir), "--skip-wandb"])
    assert result.exit_code == 0

    with h5py.File(output_dir / "train.h5", "r") as f: assert f["audio"].shape[0] == 300
    with h5py.File(output_dir / "val.h5", "r") as f: assert f["audio"].shape[0] == 100
    assert LocalStorageBackend(root=storage_root).exists(run_id, "metadata/dataset.complete")

    # Idempotent
    r2 = runner.invoke(cli, ["finalize", "--run-id", run_id,
        "--storage-root", str(storage_root), "--output-dir", str(output_dir), "--skip-wandb"])
    assert "already finalized" in r2.output.lower()
```

```python
def test_finalize_wds_output_format(tmp_path):
    """WDS output: verify tar archive creation, sample naming, split partitioning."""
    storage_root, output_dir = tmp_path / "storage", tmp_path / "output"
    config_path = _write_test_config(tmp_path, num_shards=4, shard_size=50,
                                      val_shards=1, test_shards=1, output_format="wds")
    runner = CliRunner()
    runner.invoke(cli, ["generate", "--config", str(config_path), "--backend", "local",
        "--workers", "1", "--storage-root", str(storage_root)])
    run_id = _extract_run_id(storage_root)

    result = runner.invoke(cli, ["finalize", "--run-id", run_id,
        "--storage-root", str(storage_root), "--output-dir", str(output_dir), "--skip-wandb"])
    assert result.exit_code == 0

    train_tars = sorted(output_dir.glob("train-*.tar"))
    assert len(train_tars) >= 1
    # Verify sample naming inside tar
    import tarfile
    with tarfile.open(train_tars[0]) as tf:
        names = tf.getnames()
        assert any(n.endswith(".audio.npy") for n in names)
        assert any(n.endswith(".params.npy") for n in names)
        assert any(n.endswith(".mel.npy") for n in names)
```

______________________________________________________________________

## 10. Phase 6: Production & E2E ([#73](https://github.com/tinaudio/synth-setter/issues/73))

### Task 6.1: RunPodBackend + Docker Updates + E2E ([#106](https://github.com/tinaudio/synth-setter/issues/106))

**Goal:** Production backend, Docker integration, full E2E.

**Files to create:**

- `pipeline/backends/runpod.py`, `scripts/test_local_docker.sh`
- `tests/pipeline/test_runpod_backend.py`, `tests/pipeline/test_e2e.py`,
  `tests/pipeline/test_cli_cleanup.py`

**Files to modify:**

- `scripts/docker_entrypoint.sh` — add `MODE=pipeline-worker`
- `Makefile` — `make pipeline-generate`, `pipeline-status`, `pipeline-finalize`

**RunPodBackend:** `runpod.create_pod()` with env vars, auth check, dry-run. Tags all
pods with `run_id` for cleanup.

**`cleanup` CLI command:** `python -m pipeline cleanup --run-id <id>` — queries RunPod API
for pods tagged with `run_id`, terminates them. Safety net for orphaned pods.

**Docker:** `MODE=pipeline-worker` → `python -m pipeline.worker`. Bash EXIT trap uploads
JSONL debug log + fallback `error.json` to `metadata/workers/attempts/{w}-{a}/` on crash.

**Adhoc Docker script:** Builds image, runs container with test config + mounted storage,
verifies shard output. Manual (not pytest).

**Unit tests:** RunPod env vars (mocked API), missing key error, dry-run, pod tagging,
cleanup command, BATS entrypoint, EXIT trap (BATS: kill process, verify log uploaded)

**Reference test — E2E:**

```python
@pytest.mark.slow
def test_e2e_generate_status_finalize(tmp_path):
    storage_root, output_dir = tmp_path / "storage", tmp_path / "output"
    config_path = _write_test_config(tmp_path, num_shards=6, shard_size=50,
                                      val_shards=1, test_shards=1)
    runner = CliRunner()

    gen = runner.invoke(cli, ["generate", "--config", str(config_path), "--backend", "local",
        "--workers", "2", "--storage-root", str(storage_root)])
    assert gen.exit_code == 0
    run_id = _extract_run_id(storage_root)

    status = runner.invoke(cli, ["status", "--run-id", run_id,
        "--storage-root", str(storage_root)])
    assert status.exit_code == 0

    # Retry is no-op
    gen2 = runner.invoke(cli, ["generate", "--run-id", run_id, "--backend", "local",
        "--workers", "1", "--storage-root", str(storage_root)])
    assert "0 missing" in gen2.output.lower() or "nothing to do" in gen2.output.lower()

    fin = runner.invoke(cli, ["finalize", "--run-id", run_id,
        "--storage-root", str(storage_root), "--output-dir", str(output_dir), "--skip-wandb"])
    assert fin.exit_code == 0
    assert (output_dir / "train.h5").exists()
    with h5py.File(output_dir / "train.h5") as f: assert f["audio"].shape[0] == 200
    with h5py.File(output_dir / "val.h5") as f: assert f["audio"].shape[0] == 50
```

______________________________________________________________________

## 11. Cross-cutting Work

### Design Doc Invariant Tests ([#76](https://github.com/tinaudio/synth-setter/issues/76))

Test scenarios from design doc §7 and §11.2 that span multiple PRs:

- `.valid` marker is the LAST write in the shard protocol
- `.rendering` marker is append-only — never deleted
- Workers never write to `data/shards/`
- Missing worker report does not block shard validity

These tests are written incrementally as each PR lands.

### Worker Hard Timeout & RunPod Auto-stop ([#77](https://github.com/tinaudio/synth-setter/issues/77))

- Hard timeout in `scripts/docker_entrypoint.sh` — kill worker process after
  configurable max duration (`WORKER_TIMEOUT_SECONDS`)
- EXIT trap fires on SIGTERM timeout kill (debug log + error.json uploaded).
  **Note:** EXIT traps do NOT fire on SIGKILL (OOM-killer, `kill -9`). For hard kills,
  logs are lost unless an out-of-process shipper is added. Current mitigation: use
  SIGTERM with a grace period before SIGKILL escalation, and rely on reconciliation
  (Layer 3) to detect the missing shards.
- RunPod `auto-stop` configuration in `RunPodBackend.submit()`
- `--timeout` flag on `generate` command
- BATS test: worker killed via SIGTERM after timeout still uploads debug log

### Dataset ID & Run ID Conventions ([#120](https://github.com/tinaudio/synth-setter/issues/120))

Define `dataset_id`, `run_id` naming conventions and W&B artifact provenance chain.

### Align R2 Root Datapaths ([#121](https://github.com/tinaudio/synth-setter/issues/121))

Ensure R2 root datapaths are consistent across data pipeline, eval pipeline, and training pipeline design docs.

### Centralized Provenance & Storage Design Doc ([#122](https://github.com/tinaudio/synth-setter/issues/122))

Write a design doc that unifies provenance and storage conventions across all pipelines.

______________________________________________________________________

## 12. Verification Strategy

1. **Per-PR:** CI runs `pytest` + `ruff` on every push
2. **After all steps:** `pytest tests/pipeline/ -v`, `pytest tests/pipeline/test_e2e.py -v`
3. **Local dry run:** `python -m pipeline generate --config configs/pipeline/surge_simple_480k.yaml --backend local --workers 2`
4. **Docker fidelity:** `bash scripts/test_local_docker.sh`
5. **Mutation testing:** `mutmut run --paths-to-mutate=pipeline/`

______________________________________________________________________

## 13. Assumptions

01. `pipeline/` at project root (not `src/`) — `python -m pipeline`
02. Old scripts NOT ported — stay on `experiment` branch only
03. `LocalBackend` in-process (not Docker) — intentional deviation from design doc §7.9
04. Integration tests use `LocalStorageBackend`, not real R2
05. W&B optional (`--skip-wandb`) — tests skip it; mock test for artifact structure
06. Workers use ThreadPoolExecutor for parallel shard generation
07. Each shard renders in a child process via `multiprocessing.get_context("spawn").Process(...)`.
    Child process imports `make_dataset` directly (`from pipeline.vst import make_dataset`).
    Only `shard_spec` and `shard_path` cross the process boundary — no function objects.
    `LocalBackend` accepts an optional `generate_fn` for tests (runs in-process, no spawn).
    For v1, no seeding (current behavior). Post-launch, dual-RNG seeding
    (`random.seed()` + `np.random.seed()`) for reproducibility (#100, P3). Provides
    OS-level crash isolation (SIGSEGV/OOM kill only one shard), per-shard timeout, and
    clean VST plugin state. See design doc §7.8.1
08. Entrypoint gets `MODE=pipeline-worker`, existing modes untouched
09. Tests in `tests/pipeline/` with own conftest
10. Finalize implements fresh resharding using HDF5 virtual datasets (not calling
    `reshard_data.py` — it hardcodes 10k shard size)
11. `R2StorageBackend` wraps `src/data/uploader.RcloneUploader` (already has `--checksum`)
12. `shard_id` is `int` in schema, formatted to string for paths/filenames
13. Run ID uses `total_samples` not `total_train_samples` — design doc §14.6 text to be fixed
14. Config splits use `{train: N, val: N, test: N}` matching design doc §14.4

______________________________________________________________________

## Appendix: New Gaps Found During Plan Port

All gaps below have been folded into their relevant tasks above. This appendix
serves as a changelog of what was added beyond the original implementation plan.

**GP1. `generate --dry-run` not tested in reference tests.**
Task 5.1 lists `--dry-run` as a behavior but the reference test doesn't exercise it.
Add a unit test: `--dry-run` prints shard assignments, creates no spec, submits no work.

**GP2. `status` command JSON output not specified.**
Issue #21 deliverables include "Output as table (terminal) and JSON (machine-readable)"
but Task 5.2 only describes table output. Add `--json` flag to `status` command.

**GP3. No test for auth validation failure.**
Task 5.1 specifies auth validation before compute but no reference test covers the
failure case. Add unit test: missing R2 credentials → clear error message, exit 1,
no workers launched.

**GP4. `generate` should validate `plugin_path` exists before materialization.**
If the VST3 bundle path in the config doesn't exist, `renderer_version` extraction will
fail with an unclear error. Add early validation: check `plugin_path` exists, error with
actionable message if not.

**GP5. No `--verbose` / log-level flag on CLI.**
Design doc Appendix E.1 shows structured logging config but no CLI flag controls
verbosity. Add `--log-level` flag (default: `INFO`, options: `DEBUG`, `INFO`, `WARNING`).

**GP6. Worker quarantine path not in Task 4.2.**
Design doc §7.2 describes `rendering → invalid`: worker uploads corrupt shard to
`quarantine/` and writes `.invalid` marker. Task 4.2 worker description covers only the
happy path. Add quarantine behavior + reference test for validation-failure shard.

**GP7. Skip-if-valid optimization missing from Task 4.2.**
Design doc §7.7: "Workers check the staging directory for an existing valid shard
before uploading. If one exists, the worker skips the upload." Not in Task 4.2.
Add as optimization (not correctness requirement).

**GP8. Storage layer missing path helpers for quarantine, attempts, and finalize outputs.**
Task 2.2 storage layer should expose path helpers for `quarantine/` subdirectory,
`metadata/workers/attempts/{w}-{a}/` (report.json, debug.log), and `data/` finalize
outputs (train.h5, stats.npz, dataset.json, dataset.complete). Currently only
shard lifecycle paths are described.

**GP9. `status` command should overlay worker errors from reports.**
Design doc §7.4 shows `status` output including "Recent worker errors (from metadata)"
overlaid from worker reports. Task 5.2 only describes shard counts and missing IDs.

**GP10. Design doc schema gaps to fix.**
Several fields in the design doc §14 schemas need updating to match the implementation:

- `ValidationSummary` class not defined in design doc (referenced in `DatasetCard`)
- `base_seed` not in `PipelineSpec` schema (referenced in §14.1 text)
- Generation params (preset_path, channels, etc.) not in `PipelineSpec` schema
- `shard_manifest` not in `DatasetCard` schema (mentioned in §7.6 prose)
