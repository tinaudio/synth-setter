# Experiment UX — Decision and Rollout Plan

> **Status**: Accepted recommendation (maintainer-reviewed; this PR changes documentation only — implementation lands as the PR sequence in §8)
> **Author**: ktinubu@ (agent-drafted)
> **Last Updated**: 2026-07-18
> **Tracking**: [#2122](https://github.com/tinaudio/synth-setter/issues/2122) (one-experiment UX blocker), with [#2118](https://github.com/tinaudio/synth-setter/issues/2118), [#1357](https://github.com/tinaudio/synth-setter/issues/1357), [#1741](https://github.com/tinaudio/synth-setter/issues/1741); checkpoint identity deferred to [#2136](https://github.com/tinaudio/synth-setter/issues/2136)

Design decision for the "select one experiment and go" UX. §4 is the combined
design the maintainers accepted; §8 is its rollout as small independently
shippable PRs; §10 preserves the alternatives considered and why each was
adopted, rejected, or deferred. File references were re-verified against the
tree on the Last-Updated date.

## Index

| §   | Section                                                                                             |
| --- | --------------------------------------------------------------------------------------------------- |
| 1   | [Product goals](#1-product-goals)                                                                   |
| 2   | [Current state](#2-current-state)                                                                   |
| 3   | [Identity primitives](#3-identity-primitives)                                                       |
| 4   | [The decision](#4-the-decision)                                                                     |
| 5   | [Final UX](#5-final-ux)                                                                             |
| 6   | [Cache state machine and failure behavior](#6-cache-state-machine-and-failure-behavior)             |
| 7   | [TorchSynth experiment and test coverage matrix](#7-torchsynth-experiment-and-test-coverage-matrix) |
| 8   | [Rollout plan](#8-rollout-plan)                                                                     |
| 9   | [Success criteria](#9-success-criteria)                                                             |
| 10  | [Alternatives considered](#10-alternatives-considered)                                              |
| 11  | [Risks](#11-risks)                                                                                  |
| 12  | [Non-goals](#12-non-goals)                                                                          |
| 13  | [Open questions](#13-open-questions)                                                                |

______________________________________________________________________

## 1. Product goals

Each goal is testable.

1. **One-selector training.** Normal local and GitHub/SkyPilot training require
   at most `experiment=<name>`:
   `synth-setter-train experiment=surge/flow_simple_440k`, or one `experiment`
   workflow input. The experiment is the runnable recipe; dataset source,
   render profile, and compute sizing are owned by the groups it selects.
   Evaluation keeps its current explicit `ckpt_path` or W&B-backed experiment
   overlay until checkpoint UX is resolved in [#2136].
2. **Dataset caches are validated and offline-first.** A file-backed experiment
   names a predefined **immutable** R2 dataset root. When the local cache for
   that identity carries a valid receipt, the run makes **no R2 contact**. When
   absent, hydration copies the **entire run root** with one resumable
   `rclone copy --immutable --checksum` (§4.3).
3. **Fail before spending.** Dataset completion, ParamSpec agreement across
   `training_data`/datamodule/render, encoded model width, and cloud disk
   capacity are cross-validated **before** provisioning for cloud runs and at
   CLI startup for local runs — never as a tensor-shape crash mid-training.
4. **One scientific definition.** The same experiment runs locally and via
   SkyPilot without copying scientific defaults into launch YAML `cmd:` strings
   or workflow inputs (today `train-runpod-flow-simple-440k.yaml` embeds
   `val_check_interval`, monitor keys, etc.).
5. **Hydra stays open, but optional.** Advanced overrides
   (`model.optimizer.lr=3e-5`) keep working locally, and cloud dispatch keeps an
   expert override escape hatch; ordinary users never need to know which config
   group owns which key or the defaults-list ordering rules.
6. **Every declared runnable is proven runnable.** Config tests compose **all**
   experiments (not a curated allowlist), and behavioral smoke tests exercise
   each architecture family cheaply.
7. **TorchSynth as the host-free proxy for VST architectures.** The AST
   feed-forward path, flow matching with AST conditioning, and (where practical)
   the pretrained AST encoder get TorchSynth-backed coverage that requires no
   live VST host (§7).

**Explicitly out of scope:** checkpoint configuration, caching, and identity.
Current checkpoint behavior is preserved unchanged; the known problems are
deferred to [#2136](https://github.com/tinaudio/synth-setter/issues/2136) (§4.6).

## 2. Current state

Grounded in the tree at the time of writing; file references are load-bearing.

### 2.1 Composition

- `train.yaml` / `eval.yaml` are `# @package _global_` roots with
  `datamodule: ???` / `model: ???` and `experiment: null` late in the defaults
  list. An experiment can only `override /X` for groups declared **before**
  `experiment` (comment at `src/synth_setter/configs/eval.yaml:13-16`) — any new
  group experiments must override has to be threaded into both roots in the
  right position.
- 86 files under `configs/experiment/` (including `generate_dataset/` datagen
  recipes); every training leaf is a `# @package _global_` overlay composing a
  family `base.yaml` plus `override /model`, `override /datamodule`, etc.
- Model output width is now **derived, not hardcoded** (PR #2119):
  `configs/model/vst_flow.yaml:48` sets
  `num_params: ${param_spec_width:${datamodule.param_spec_name}}` via the
  resolver registered in `src/synth_setter/utils/utils.py:33-34`, and the
  datamodule derives width at runtime via
  `resolve_param_spec(self.param_spec_name).encoded_width`
  (`src/synth_setter/data/lance_datamodule.py:402`). Coherence between model and
  datamodule width is therefore structural for the `vst_*` model groups; the
  preflight width check (§4.5) remains as defense in depth for literal overrides
  and archived configs.

### 2.2 Dispatch

- Launch configs (`configs/launch/*.yaml`) are `SkypilotLaunchConfig` payloads
  whose `cmd:` is a shell string:
  `exec synth-setter-train "experiment=${EXPERIMENT:-surge/ffn_simple}" "datamodule.download_dataset_root_uri=${DATASET_ROOT_URI:-r2://…}" …`.
  The 440k launch additionally embeds scientific knobs (`render=surge_simple`,
  `trainer.val_check_interval=2000`, checkpoint monitor) — goal 4 violated today.
- `.github/workflows/train.yml` inputs: `launch_config` (required), `experiment`
  and `dataset_root_uri` (optional, forwarded as `--extra-env` and consumed by
  the `${EXPERIMENT:-…}` shell defaults). `eval.yml` takes **only**
  `launch_config`.
- `src/synth_setter/pipeline/skypilot_launch.py` injects `cmd` into the compute
  template's `run:` and dispatches **SkyPilot managed jobs**: compute is
  provisioned per job and torn down at terminal status. No named-cluster reuse,
  no RunPod network volumes, no persistent disks anywhere in launcher or
  templates.
- The launcher never composes Hydra; it cannot validate the experiment it is
  about to spend money on. All validation happens (or fails to happen) on the
  worker.

### 2.3 Dataset production and hydration

- **Production is fragment-native** (#1776): shard workers write *uncommitted*
  Lance fragment data files directly under the split datasets' `data/`
  directories, and finalize commits manifests over those same files
  (`src/synth_setter/pipeline/data/lance_finalize.py`,
  `src/synth_setter/pipeline/CLAUDE.md`). A finalized fragment-native run root
  therefore contains **no duplicate committed copy** of the data — the root's
  total size ≈ the training payload plus small metadata (sidecars, shard
  markers, worker reports), which is acceptable transfer/disk overhead.
- **The legacy pinned July 6 440k root**
  `r2://experiments/data/surge-simple-lance-440k-20k-20k/surge-simple-lance-440k-20k-20k-20260706T005448315Z/`
  predates that layout: it holds finalized `train/val/test.lance` **and**
  top-level source `shard-*.lance` datasets, measuring ~1.016 TiB, while
  `runpod-training-template.yaml` requests `disk_size: 750` — whole-root
  hydration overflows the disk before training starts ([#2122]). §4.4 resolves
  this.
- `datamodule.dataset_root` defaults to `${paths.output_dir}/data`
  (`configs/datamodule/vst.yaml:2`), i.e. **under Hydra's per-run output dir** —
  the same dataset re-downloads every run ([#1357]). RunPod launches only get a
  stable target because `hydra.run.dir` is pinned in the launch cmd.
- Hydration is `prepare_data()` → `r2_io.download_dir_no_overwrite` → one
  recursive `rclone copy --immutable` (with the shared `--checksum` retry flags)
  of the entire configured root
  (`src/synth_setter/data/vst_datamodule.py:226-231`,
  `src/synth_setter/pipeline/r2_io.py:427-441`). The transfer mechanism is
  already what §4.3 wants; what is missing is a completion gate
  (`dataset.complete` is never checked), a receipt marking a finished hydration,
  and a lock for concurrent runs. Today the closest implicit gate is `stats.npz`
  loading failing later in `setup()`.

### 2.4 Checkpoints (behavior preserved; problems deferred to #2136)

- Eval requires `ckpt_path: ???` — a local path or a `${wandb:…}` resolver
  (`src/synth_setter/utils/utils.py`) that downloads a W&B model artifact's
  `s3://`→`r2://` references into `$PROJECT_ROOT/.cache/checkpoints/<cache_key>`
  and reuses it. Reuse is existence-based, not validated.
- Auto-resume (`training.resume ∈ {off, auto, require}`,
  `src/synth_setter/utils/resume.py`) discovers newest local sibling `last.ckpt`
  first, then the R2 mid-run mirror `r2://{bucket}/checkpoints/{config_id}/…`.
- Identity is weak: the train-end R2 object
  `r2://{bucket}/checkpoints/{config_id}/model.ckpt` is **overwritten per
  config_id** (known limitation, `docs/design/storage-provenance-spec.md` §4),
  and the `experiment/surge/wandb_checkpoint/*.yaml` overlays pin
  `${wandb:…/model-<id>:latest}` — a **floating alias**.

This overhaul **changes none of this**. The mutable-reference and unvalidated-
cache problems are tracked in
[#2136](https://github.com/tinaudio/synth-setter/issues/2136); see §4.6 for the
constraints its eventual solution must satisfy.

### 2.5 Validation

The only early cross-group check is `_validate_probe_spec_match`
(`src/synth_setter/cli/train.py:182-199`):
`render.param_spec_name == datamodule.param_spec_name`, and only when the val
audio probe is enabled. Nothing validates dataset completion or disk capacity
anywhere, and nothing validates spec coherence unconditionally; on cloud runs
any failure surfaces **after** provisioning and hydration.

### 2.6 Tests

- Only a curated subset of experiments is ever composed (`tests/test_configs.py`,
  plus the `DATASET_EXPERIMENTS` allowlist in
  `tests/pipeline/configs/test_experiment_yamls.py`). Group-level schema tests
  glob `configs/model/*.yaml` etc., but **no test auto-discovers and composes
  `configs/experiment/**`** — a dangling group reference in an uncovered
  experiment ships green.
- Launcher logic is well covered with mocked SkyPilot
  (`tests/pipeline/entrypoints/test_skypilot_launch.py`, 1788 lines) — but
  `test_skypilot_launch.py:1686` currently **pins the broken behavior** (it
  asserts `datamodule=surge_lance_map` is present in the 440k launch config).
- TorchSynth coverage is the online datamodule + generic LogMel CNN/MLP only
  (`tests/pipeline/configs/test_torchsynth_experiment_config.py`,
  `tests/data/test_torchsynth_datamodule.py`). The VST AST feed-forward module,
  VST flow modules, and pretrained AST encoder are exercised only via tiny
  contract tests and fake-data surge paths (§7).

### 2.7 Confirmed defects the redesign must not paper over

| Defect                                                                                                                                                                                                                                                         | Evidence                                                                                                                               | Tracking          |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `train-runpod-smoke.yaml:23` and `train-runpod-flow-simple-440k.yaml:13` pass `datamodule=surge_lance_map`, deleted by PR #2075 — worker-side Hydra composition fails                                                                                          | no `configs/datamodule/surge_lance_map.yaml` exists; `tests/pipeline/entrypoints/test_skypilot_launch.py:1686` asserts the stale value | [#2118]           |
| Legacy 440k whole-root hydration (~1.016 TiB incl. top-level `shard-*.lance`) overflows the 750 GB training template disk                                                                                                                                      | `configs/compute/runpod-training-template.yaml`; §2.3                                                                                  | [#2122]           |
| `dataset_root` per-run default defeats cache reuse                                                                                                                                                                                                             | `configs/datamodule/vst.yaml:2`                                                                                                        | [#1357]           |
| Six experiment files reference model groups that do not exist (`ksin_flow`, `ksin_ff`): `experiment/time_weighting.yaml`, `experiment/flow_size/base.yaml` (and its leaves), `experiment/ksin_ood/{flow,mlp_mse,mlp_chamfer,mlp_sort}.yaml` — none can compose | `configs/model/` contains only `ffn/flow*/flowmlp/surge_*/vst_*`; verified by grep                                                     | new — file as Bug |
| No unconditional preflight: spec coherence is probe-gated, dataset completion and disk capacity are never checked                                                                                                                                              | §2.5                                                                                                                                   | this design       |

## 3. Identity primitives

**Dataset identity.** Path-based: `{dataset_config_id}/{dataset_wandb_run_id}`
under `data/` in a bucket (`docs/design/storage-provenance-spec.md` §1–3).
Immutable once `dataset.complete` exists; there is no content hash today and
none is needed (path identity + completion marker + `--immutable` drift
detection suffice). The unit of identity — and, per §4.3, the unit of hydration
— is the **whole run root**.

**Checkpoint identity.** Out of scope here; today's identities are mutable or
floating (§2.4) and the redesign is deferred to
[#2136](https://github.com/tinaudio/synth-setter/issues/2136).

**Caches.** A cache maps identity → validated local directory. Where a cache
can actually pay off:

- **Local dev machines**: the main beneficiary. A stable
  `${SYNTH_SETTER_CACHE_DIR}` keyed by identity gives cross-run and
  cross-worktree reuse.
- **RunPod via SkyPilot managed jobs**: the pod filesystem is **ephemeral and
  per-job** — provisioned with the job, torn down at terminal status. A local
  cache helps only *within* one job (e.g. crash-restart inside the pod).
  Cross-job reuse would require RunPod network volumes or long-lived clusters —
  neither is used, and provisioning them is a non-goal (§12).

## 4. The decision

One combined design: lightweight reusable Hydra groups for data and execution,
selected by runnable experiment configs; whole-root receipted hydration; shared
preflight validation run by both launcher and worker; launcher-derived dispatch.

### 4.1 `training_data`: a reusable group owning dataset identity

A new Hydra group, `configs/training_data/*.yaml` — one small file per pinned
immutable dataset, reused by every experiment that trains on it:

```yaml
# configs/training_data/surge_simple_440k.yaml   (illustrative)
dataset_id: surge-simple-lance-440k-20k-20k/surge-simple-lance-440k-20k-20k-20260706T005448315Z
root_uri: r2://experiments/data/${training_data.dataset_id}/
param_spec_name: surge_simple
dataset_root: ${paths.cache_dir}/datasets/${training_data.dataset_id}
```

- `paths` gains `cache_dir: ${oc.env:SYNTH_SETTER_CACHE_DIR,${paths.root_dir}/.cache}`
  — a stable root, fixing the per-run `dataset_root` default ([#1357]).
- `configs/datamodule/vst.yaml` derives `dataset_root`,
  `download_dataset_root_uri`, and `param_spec_name` from `${training_data.*}`,
  so the spec/URI/cache-path triple is stated **once** per dataset.
- The **experiment is the runnable recipe**: a file-backed experiment selects
  `override /training_data: surge_simple_440k` plus its model/trainer science.
  Scientific settings (cadence, monitors, render profile) live in experiment
  YAML — never in launch YAML.

### 4.2 `execution`: operational compute policy, launcher-derived dispatch

A second lightweight group, `configs/execution/*.yaml`, owns **operational
policy only**. Disk capacity has one authority: the selected compute template's
`resources.disk_size`; the execution group supplies only sizing policy around
that value:

```yaml
# configs/execution/runpod_training.yaml   (illustrative)
compute_template: src/synth_setter/configs/compute/runpod-training-template.yaml
dataset_headroom_fraction: 0.20
workspace_reserve_gb: 50
```

Dispatch inverts today's flow. The train and eval workflows call
`synth-setter-skypilot-launch --entrypoint <train|eval> --experiment <name>`.
The launcher first requires its checkout SHA to equal the `WORKER_GIT_REF` it
will forward, then composes the matching Hydra root (`train.yaml` or
`eval.yaml`) and confirms `experiment_meta.entrypoint` agrees. It runs the
shared preflight (§4.5) **before provisioning**, reads `execution:`, and builds
the matching worker command itself. Consequences:

- Launch YAML (`configs/launch/*.yaml`) shrinks to site/operational policy
  (image tag, env file, tail) or disappears entirely; hand-written `cmd:` shell
  strings and their `${EXPERIMENT:-…}` indirection — the #2118 defect class —
  are deleted. Smoke variants become experiments (`…_smoke.yaml` with
  `max_steps: 10`), not launch files.
- `train.yml` collapses to **one required `experiment` input**, plus one
  optional expert input forwarding extra Hydra overrides. `eval.yml` adopts the
  same experiment-derived dispatch but retains its current explicit
  checkpoint/W&B selection until [#2136]; this overhaul does not promise
  one-input evaluation.
- The launcher refuses to provision unless it composed from the exact
  `WORKER_GIT_REF` the worker will check out. The worker re-runs the same
  preflight as defense in depth.

### 4.3 Whole-root hydration: one resumable copy, receipt written last

Fragment-native finalized roots (§2.3) contain no duplicate committed data, so
the whole root is approximately the training payload — **hydrate all of it**
with the existing single `rclone copy --immutable --checksum` invocation
(`download_dir_no_overwrite`). No consumer prefixes, no selective
include/exclude machinery: metadata is acceptable overhead, and selective
hydration would add a permanent artifact-layout contract to serve what is
fundamentally a legacy-root problem (§4.4).

What hydration adds on top of the existing transfer:

- **Completion gate**: refuse to hydrate a root whose remote `dataset.complete`
  is missing — the identity is not finalized/immutable yet.
- **Per-identity lock**: a flock keyed by `dataset_id` serializes concurrent
  hydrations of the same identity (multiple worktrees, DDP ranks).
- **Receipt written last**: before transfer, capture a remote object manifest
  containing each path, size, modification time, and available checksum. After
  the copy, verify the local object set against that manifest, open the expected
  split manifests, then write `.hydration-receipt.json` (identity, source URI,
  completion marker, object manifest, and tool versions) as the final step.
  Normal startup compares per-file path/size/mtime and keeps the completed cache
  read-only; an explicit full-integrity command rehashes every file when needed.
  This detects ordinary local replacement without rereading hundreds of GB on
  every run; the cache is not a security boundary against an adversary that can
  preserve filesystem metadata. A valid receipt permits **fully offline** use.
  With no receipt, hydration resumes and `--checksum` skips complete files.
- **No atomic directory rename.** Terabyte-scale roots make staged-rename
  patterns impractical; receipt-written-last provides the same "complete or
  clearly incomplete" property without one.

Full state machine and failure table: §6.

### 4.4 The legacy July 6 440k root

The pinned July 6 root is pre-fragment-native: it carries top-level
`shard-*.lance` source datasets beside the finalized splits and measures
~1.016 TiB (§2.3). The resolution is an **ops choice, not permanent
machinery** — one of:

1. **Move the 440k launch to a fragment-native root**: republish the finalized
   artifacts (or regenerate through the current fragment-native pipeline) under
   a new immutable identity, and repoint `training_data/surge_simple_440k` at
   it. Preferred: every pinned root then has the same shape and cost profile.
2. **Provision enough disk**: select or update a compute template whose single
   authoritative `resources.disk_size` covers the measured root, 20% dataset
   headroom, and 50 GiB workspace reserve — at least ~1.27 TiB before any
   additional provider/filesystem margin.

Either closes [#2122]. Building permanent selective-download machinery to
preserve the legacy layout is explicitly rejected.

### 4.5 Shared preflight validation

One plain function on a composed cfg — `validate_experiment(cfg)` — run by the
launcher **before provisioning** and re-run by the worker/CLI at startup:

- **Dataset completion**: remote `dataset.complete` exists (or a valid local
  receipt is present).
- **Spec agreement**: `training_data.param_spec_name` ==
  `datamodule.param_spec_name` == `render.param_spec_name` (where render is
  configured) — unconditional, not probe-gated as today (§2.5).
- **Encoded model width**: composed model width ==
  `resolve_param_spec(…).encoded_width`. Structural for `vst_*` groups since
  PR #2119's derived widths; the check stays as defense in depth for literal
  overrides and archived configs.
- **Cloud disk capacity**: read the selected template's authoritative
  `resources.disk_size` and require it to satisfy
  `ceil(root_gb × (1 + dataset_headroom_fraction) + workspace_reserve_gb)`.
  Measure the root live with `rclone size`; reject insufficient capacity before
  provisioning and re-check usable filesystem space on the worker.

Failure UX: a single-paragraph error naming the two conflicting values and the
file that owns each — emitted by the launcher for cloud runs (nothing
provisioned) and at CLI startup locally.

### 4.6 Checkpoints: unchanged here, deferred to #2136

This overhaul makes **no** checkpoint config, cache, or identity changes;
current `${wandb:…}` resolution, `.cache/checkpoints` reuse, auto-resume, and
per-config_id mirrors (§2.4) all keep working as-is. The known problems —
mutable per-config_id overwrites, floating `:latest` aliases,
existence-only cache reuse — are deferred to
[#2136](https://github.com/tinaudio/synth-setter/issues/2136).

Constraint recorded for that future work: its solution must integrate with
**W&B artifacts as the model discovery, lineage, and promotion surface** while
producing **immutable, reproducible checkpoint identities** — not a parallel
registry that bypasses W&B, and not W&B aliases that float. The deferred
registry sketch is preserved in §10 (Option B, checkpoint half).

## 5. Final UX

```bash
# Local training — hydrates the pinned whole root once into $SYNTH_SETTER_CACHE_DIR,
# then runs offline on every subsequent invocation.
synth-setter-train experiment=surge/flow_simple_440k

# Local eval — checkpoint handling unchanged (ckpt_path / ${wandb:…}, see §4.6).
synth-setter-eval experiment=surge/eval_ffn_4 ckpt_path=…

# Advanced users: plain Hydra overrides still work.
synth-setter-train experiment=surge/flow_simple_440k model.optimizer.lr=3e-5

# Cloud training: identical scientific definition, one input.
gh workflow run train.yml -f experiment=surge/flow_simple_440k

# Evaluation still carries explicit checkpoint selection until #2136.
synth-setter-eval experiment=surge/eval_ffn_4 ckpt_path=/path/to/model.ckpt

# Expert escape hatch (optional workflow input; normal runs never set it).
gh workflow run train.yml -f experiment=… -f extra_overrides="trainer.max_steps=100"
```

## 6. Cache state machine and failure behavior

Applies to the dataset cache; identity = the pinned run root. The
`.hydration-receipt.json` records the source identity, completion marker, tool
versions, and per-object path/size/mtime/checksum manifest; it is written only
after transfer and initial integrity verification complete.

```
ABSENT ──(acquire per-identity flock)──► HYDRATING
  HYDRATING ──(one rclone copy --immutable --checksum of the whole root;
               verify split manifests + dataset.complete; write receipt LAST)──► READY
  crash/partial: no receipt written; next run re-enters HYDRATING and the
  copy resumes idempotently (--checksum skips complete files)

READY ──(receipt + local path/size/mtime manifest agree)──► RUN OFFLINE
READY ──(manifest mismatch)──► INVALID (hard error; never auto-delete;
        error names the receipt, divergent file, and one-line purge command)
```

| Situation                                         | Behavior                                                                                                                       |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Cache READY, no network/creds                     | Compare the local path/size/mtime manifest, then proceed fully offline.                                                        |
| Cache ABSENT, no creds                            | Fail fast: "dataset `<id>` not cached and no R2 credentials resolved".                                                         |
| Concurrent runs, same identity                    | Second run blocks on the flock, then finds READY.                                                                              |
| Crash mid-hydration                               | No receipt → next run resumes the copy (`--checksum` skips complete files).                                                    |
| Remote object drifted under an immutable identity | `rclone --immutable` hard-fails → surface an identity violation; investigate rather than retry.                                |
| `dataset.complete` missing remotely               | Refuse to hydrate because the identity is not finalized.                                                                       |
| Disk insufficient (cloud)                         | Apply dataset headroom and workspace reserve before provisioning; re-check usable worker space before copying.                 |
| Local file replaced normally                      | Path/size/mtime mismatch identifies the file and marks the cache INVALID; full rehash remains available for bit-rot diagnosis. |

## 7. TorchSynth experiment and test coverage matrix

TorchSynth renders in-process (`TorchSynthRenderer`; eval side already renders
predictions without a plugin host via the `"torchsynth"` sentinel in
`src/synth_setter/evaluation/predict_vst_audio.py:135-164`, PR #2112). That
makes it the ideal host-free proxy for the VST architecture paths. Today only
the first row exists.

| Experiment                                     | Datamodule / backing                                                                              | Model module                                                        | Encoder / net                                                                                                     | Covers                                                                                                           | Test tier                                                                                                | Status   |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------- |
| `torchsynth/ffn`                               | `TorchSynthDataModule` (online)                                                                   | `KSinFeedForwardModule`                                             | `LogMelCNNResidualMLP`                                                                                            | online render loop                                                                                               | compose + `fast_dev_run` CPU (exists)                                                                    | exists   |
| `torchsynth/ast_ffn` (new)                     | `LanceVSTDataModule`, tiny TorchSynth-rendered Lance fixture, `param_spec_name=torchsynth_simple` | `VSTFeedForwardModule`                                              | `ASTWithProjectionHead` (scratch, test-shrunk `d_model`/`n_layers`)                                               | **AST feed-forward on the VST code path**, width derivation (`encoded_width`)                                    | compose (fast) + one-step train CPU (fast)                                                               | proposed |
| `torchsynth/flow_ast` (new)                    | same fixture                                                                                      | `VSTFlowMatchingModule`                                             | `encoder=ast` (scratch)                                                                                           | **flow matching with AST conditioning**, preds contract on real rendered data                                    | compose (fast) + one-step train CPU (fast) + preds-contract                                              | proposed |
| `torchsynth/flow_ast_pretrained` (new overlay) | same fixture                                                                                      | `VSTFlowMatchingModule`                                             | `ast_pretrained` (tiny random offline backbone, existing pattern from `build_fake_flow_ast_pretrained_train_cfg`) | **pretrained AST encoder wiring** incl. `d_model` backbone check, offline                                        | fast CPU; optional `@slow` networked leg with the real `MIT/ast-finetuned-audioset-10-10-0.4593` weights | proposed |
| `torchsynth/eval_ast_ffn` (new)                | same fixture + trained tiny ckpt                                                                  | eval root, `render=torchsynth_simple`, `evaluation.render_vst=true` | —                                                                                                                 | full predict → decode → **host-free render** → metrics roundtrip (closes the "TorchSynth eval path is thin" gap) | `@slow` CPU                                                                                              | proposed |

Test scaffolding: one session-scoped fixture generates the tiny Lance fixture (a
few dozen samples) with `TorchSynthRenderer` — no VST host, no X11, CPU-only —
mirroring `fake_surge_smoke_datasets` but with *real* rendered audio and a *real*
registered spec, which the surge fake path cannot provide. All five rows join the
exhaustive experiment-composition test (rollout PR 2) automatically.

What this deliberately does not cover: flow-matching *algorithmic* correctness
(loss target / ODE numerics — a separate `ml-test`-style unit-test effort) and
real-checkpoint AST downloads in default CI (kept behind an opt-in marker).

## 8. Rollout plan

Small, independently shippable PRs; each leaves the repo strictly better even if
the sequence stops. **This document's PR ships recommendations only — no code.**

01. **`fix(training)`: repoint smoke/440k launches to existing datamodule
    groups** — closes [#2118]; also fix the launcher test pinning the stale value
    (`test_skypilot_launch.py:1686`).
02. **`test(configs)`: entrypoint-aware exhaustive experiment composition** —
    add `experiment_meta.entrypoint` to runnable recipes, auto-discover them,
    and compose each only against its owning `train`, `eval`, or `dataset` root.
    Keep an explicit shrinking skip-list for known-broken files; fix or delete
    the `ksin_flow`/`ksin_ff` dangling references (§2.7). This is the tripwire
    that prevents every future #2118.
03. **`internal-feat(training)`: establish shared preflight validation** — wire
    `validate_experiment(cfg)` into worker/CLI startup for invariants available
    on current configs (render/datamodule spec agreement and encoded width).
    PR 4 extends it with `training_data` completion checks; PR 6 adds receipt
    validation; PR 8 runs the complete preflight before provisioning.
04. **`feat(config)`: stable cache root + `training_data` group** —
    `paths.cache_dir` from `SYNTH_SETTER_CACHE_DIR`, `configs/training_data/`
    entries for the pinned roots, datamodule keys derived from
    `${training_data.*}`; closes [#1357].
05. **`refactor(training)`: move scientific knobs out of launch YAML** — new
    `surge/flow_simple_440k.yaml` experiment selects its `training_data` pin and
    owns cadence/monitor knobs; launch `cmd:` shrinks to `experiment=` only.
06. **`feat(data)`: receipted, locked whole-root hydration** (§4.3) — completion
    gate, per-identity flock, receipt-written-last on top of the existing
    `download_dir_no_overwrite` transfer; offline-first cache semantics.
07. **Legacy 440k resolution** (§4.4) — republish/repoint to a fragment-native
    root (preferred) or select a compute template with sufficient
    `resources.disk_size`; whichever is chosen closes [#2122].
08. **`feat(compute)`: launcher-composed dispatch + `execution` group** (§4.2) —
    `--entrypoint`/`--experiment`, pre-provision preflight, one-input training,
    and experiment-derived eval dispatch with existing checkpoint selection;
    delete per-run launch `cmd:`s.
09. **`feat(training)` + `test`: TorchSynth AST/flow experiments and fixture**
    (§7), in 2–3 PRs (fixture + ffn/flow experiments, pretrained overlay, eval
    roundtrip). Independent; can proceed any time after PR 2.
10. **`docs`: update operational references with each implementation** — keep
    `training-pipeline.md` §6.1 and `configuration-reference.md` synchronized as
    the code PRs land.

Minimum useful first step = PRs 1–4. PRs 5–8 deliver the headline UX.

## 9. Success criteria

- `gh workflow run train.yml -f experiment=…` with **no other inputs** launches
  the 440k run successfully. Evaluation remains explicit about checkpoint
  selection until [#2136].
- Second local run of a cached experiment performs **zero R2 operations**
  (assertable in tests by monkeypatching `r2_io` to raise).
- Every recipe marked runnable composes against its declared entrypoint in
  `make test-fast`; abstract family overlays are classified explicitly, and the
  temporary failure skip-list is enforced non-growing.
- A deliberately mismatched width or spec pairing fails at launcher/CLI startup
  in \<60 s with a naming-both-sides error — demonstrated by test, never by a
  training crash; cloud failures occur **before provisioning**.
- `grep -R 'trainer\.\|datamodule\.\|render=' src/synth_setter/configs/launch/`
  returns nothing (no scientific defaults in launch YAML).
- The 440k launch hydrates a root that fits its provisioned disk with recorded
  headroom (via §4.4 option 1 or 2), verified by the launcher's live
  `rclone size` preflight.
- AST-FF, flow+AST, and pretrained-AST wiring each have a green CPU test that
  runs without a VST host.
- Checkpoint behavior is byte-for-byte unchanged by this effort (no config,
  cache, or identity diffs); [#2136] remains the single tracker for that work.

## 10. Alternatives considered

Five genuinely distinct designs were evaluated; the accepted design (§4) is a
combination. Dispositions first, then the original comparison.

**Option A — Experiment-owned pins + shared validated cache root.** Each
experiment pins its dataset identity/URI directly; hydration becomes selective
(consumer-artifact subset) and receipted. *Adopted in refined form*: the cache
root, receipts, and immutable pins survive, but pins moved into the reusable
`training_data` group (one file per dataset, not a literal repeated per
experiment), and **selective consumer-subset hydration was rejected** — with
fragment-native roots there is no duplicate committed copy to skip, so
whole-root hydration is nearly free and avoids a permanent artifact-layout
contract between producer and consumer.

**Option B — Checked-in artifact registries (datasets + checkpoints).**
Pydantic-validated `datasets.yaml`/`checkpoints.yaml` mapping names →
URIs + metadata (bytes, spec), enabling pre-provision disk checks from declared
bytes. *Rejected for datasets*: a second source of truth beside R2 that can
drift (the architecture docs' "R2 as source of truth" principle), and the live
`rclone size` preflight gets the disk check without it. *Checkpoint half
deferred* to [#2136]; any registry there must integrate with W&B artifacts as
the discovery/lineage/promotion surface (§4.6) rather than bypass it.

**Option C — Frozen ExperimentSpec (compile-then-run trust boundary).** A
pydantic spec capturing the compatibility surface, frozen to JSON at dispatch,
re-validated on the worker. *Rejected*: launcher-side compose + shared preflight
(§4.2/§4.5) delivers the pre-provision validation at a fraction of the cost;
the spec schema is a maintenance tax where every new compatibility-relevant
knob must be added or silently escapes validation — classic framework-building
under YAGNI.

**Option D — Launcher composes the experiment.** *Adopted* (§4.2), with the
compute block factored into the reusable `execution` group rather than inlined
per experiment.

**Option E — Mounted/streaming R2 (no hydration).** Read splits directly from
R2; no local copy, tiny disks. *Rejected for this overhaul*: unproven
throughput for this workload (the failure mode moves from "disk full" to "GPU
starved by network"), and local dev still wants a local copy, so it adds a
second read path rather than replacing the first. Remains the independent
[#1800] track with a mandatory throughput gate.

| Criterion                                | A: pins + cache             | B: registry                    | C: frozen spec       | D: launcher-composed       | E: streaming         |
| ---------------------------------------- | --------------------------- | ------------------------------ | -------------------- | -------------------------- | -------------------- |
| One-selector UX (goal 1)                 | ✅                          | ✅                             | ✅                   | ✅ (incl. workflows)       | ✅                   |
| Offline-first validated dataset cache    | ✅                          | ✅                             | (delegates)          | (delegates)                | n/a (no copy)        |
| Validation before provisioning           | ❌ (worker-start)           | ✅ (bytes known)               | ✅✅                 | ✅                         | ✅ + throughput risk |
| Kills scientific-defaults-in-launch-YAML | ✅                          | ✅                             | ✅                   | ✅✅ (deletes launch cmds) | ✅                   |
| New subsystems                           | ~0                          | registry + resolver            | spec schema + 2 CLIs | launcher compose path      | new dataloader       |
| Migration effort                         | S                           | M                              | L                    | M                          | L                    |
| Main failure mode                        | post-provision fail         | registry↔R2 drift              | spec↔Hydra skew      | launcher compose drift     | GPU starvation       |
| Disposition                              | adopted (refined, §4.1/4.3) | rejected / ckpt half → [#2136] | rejected             | adopted (§4.2)             | rejected → [#1800]   |

Also explicitly rejected: directory-existence cache checks (use receipts +
`dataset.complete`), selective include/exclude hydration and consumer-prefix
layouts (§4.3), atomic directory renames for terabyte roots (§4.3), and a
unified runner CLI wrapping the existing entrypoints.

## 11. Risks

- **Hydra interpolation fragility.** Deriving datamodule keys from
  `${training_data.*}` must keep `DatasetSpec.from_hydra_cfg`'s masking and the
  eval `input_spec.json` lineage lookup working; mitigated by PR-2's exhaustive
  composition test landing first.
- **Launcher/worker skew.** The dispatch container mounts the workflow
  checkout; forwarding another ref would validate one tree and run another.
  Mitigate by rejecting `git rev-parse HEAD != WORKER_GIT_REF` before compose or
  provisioning, then revalidate on the worker.
- **Receipt scheme vs. rclone semantics.** `--immutable` interacts subtly with
  resumed partial copies; needs dedicated tests for the crash → resume → READY
  path.
- **Live `rclone size` preflight cost/flakiness.** A listing over a terabyte
  root is cheap but network-dependent; the launcher must fail closed with a
  clear retryable error, not skip the check.
- **Scope creep toward a frozen spec.** The spec-freeze temptation returns with
  every new validation; the guard is that the preflight stays one plain
  function on a composed cfg (§4.5).
- **TorchSynth fixture cost.** Rendering even dozens of 4 s samples at 44.1 kHz
  on CPU must stay within `test-fast` budget; shrink duration/sample-rate in
  the fixture spec if needed (spec identity is test-local, so this is safe).

## 12. Non-goals

- **Any checkpoint config/cache/identity change** — deferred wholesale to
  [#2136](https://github.com/tinaudio/synth-setter/issues/2136) (§4.6).
- A unified runner CLI; the existing `synth-setter-train` / `synth-setter-eval`
  / `synth-setter-skypilot-launch` entrypoints stay.
- A frozen ExperimentSpec / compile-then-run trust boundary (Option C).
- Mounted/streaming R2 training (Option E stays [#1800]).
- Consumer-prefix artifact layouts or selective include/exclude hydration
  (§4.3), and permanent machinery to preserve the legacy July 6 root layout
  (§4.4).
- A generic artifact framework or checked-in dataset registry while the pinned
  dataset count stays in single digits.
- RunPod network volumes, persistent-volume management, or long-lived cluster
  reuse for cross-job caching (§3).
- Content-addressed dataset hashing / a general artifact CAS (path-identity +
  `dataset.complete` + receipts suffice).
- Multi-node/DDP orchestration changes; W&B artifact redesign.
- Backfilling flow-matching algorithmic unit tests (tracked separately from
  this UX effort).

## 13. Open questions

1. **Legacy 440k: republish vs. disk bump** (§4.4) — an ops/maintainer choice
   that gates rollout PR 7. Republishing is preferred; the disk bump is the
   fallback if republication is expensive.
2. **Where does the cache root live on RunPod pods?** It must sit under the
   filesystem represented by the compute template's `resources.disk_size` for
   the capacity check to be truthful; confirm the mount layout during PR 8.
3. **Should `execution` be selected by family bases or leaf experiments?**
   Leaning bases with leaf overrides, to avoid ~80 copies of the same
   selection.
4. **Defaults-list threading**: `training_data` and `execution` must be
   declared before `experiment` in both `train.yaml` and `eval.yaml`
   (`eval.yaml:13-16` ordering rule); confirm no third root needs them
   (`dataset.yaml` is producer-side and out of scope).
5. **Do any other pinned roots predate fragment-native finalize?** The July 6
   440k root is the known case; audit remaining pinned URIs (e.g. the 1k smoke
   root) so PR 7 covers every legacy-layout root at once.

______________________________________________________________________

*Decision recorded in `docs/design/experiment-ux-options.md`.*
