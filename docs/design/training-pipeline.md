# Training Ops — Brain Dump

> **Status**: Brain dump (not a design doc yet)
> **Date**: 2026-03-19

______________________________________________________________________

## What this covers

Making training runs portable (local, Docker, RunPod) with R2-backed checkpointing
and crash resilience. Depends on R2 integration epic (#99) for shared infrastructure.

## How training differs from data generation

| Concern          | Data pipeline                             | Training                            |
| ---------------- | ----------------------------------------- | ----------------------------------- |
| Job shape        | Many short parallel workers               | One long-running job                |
| Duration         | Minutes per worker                        | Hours to days                       |
| Coordination     | Reconciliation (desired vs actual shards) | None — single job                   |
| Failure recovery | Re-run missing shards                     | Resume from last checkpoint         |
| Monitoring       | Storage-based (are shards done?)          | W&B (loss curves) + "is pod alive?" |
| Output           | HDF5 shards → R2                          | Checkpoints → R2, metrics → W&B     |

**Key insight:** The data pipeline's `ComputeBackend` abstraction (submit + check storage)
does not apply. Training on RunPod is just `create_pod(cmd="python src/train.py ...")`.
No reconciliation, no partitioning, no worker lifecycle.

## Current state

### What already works

- `python src/train.py` with Hydra config composition
- W&B logger tracks all training metrics (`log_model: true` uploads ckpts to W&B)
- `ModelCheckpoint` saves every 5000 steps + best + last
- CSV logger as local fallback
- Lightning handles resume via `ckpt_path=`
- `rootutils` sets `PROJECT_ROOT` automatically — paths just work

### What's coupled to the cluster

- `dataset_root` hardcoded in data configs → fixed by eval #94
- No R2 checkpoint persistence → if RunPod pod dies, checkpoints are gone
- No RunPod launch script for training (only for data gen)
- Docker image is data-gen focused, not training focused (no GPU deps baked in)
- `wandb.entity: "benhayes"` hardcoded → should be configurable

## Work items

### 1. R2 checkpoint callback (Option A)

Piggyback on `ModelCheckpoint`'s existing save events. Upload to R2 after each save.

```python
class R2CheckpointUploader(Callback):
    def __init__(self, r2_path: str):
        self.r2_path = r2_path  # r2:synth-data/checkpoints/{experiment}/{run_id}/

    def on_train_epoch_end(self, trainer, pl_module):
        ckpt = trainer.checkpoint_callback
        if ckpt is None:
            return
        for path in [ckpt.best_model_path, ckpt.last_model_path]:
            if path and Path(path).exists():
                rclone_copyto(path, f"{self.r2_path}/{Path(path).name}")
```

Config — **no default, must be explicitly specified**:

```yaml
r2_checkpoint:
  _target_: src.callbacks.r2_checkpoint.R2CheckpointUploader
  r2_path: ???  # required when included
```

Reuses: rclone wrapper from #90, `--checksum` always.

Works with existing `every_n_train_steps: 5000` — uploads happen at the same
cadence as local saves. `--checksum` prevents redundant uploads.

### 2. Resume-from-R2 flow

When a RunPod pod crashes mid-training:

```bash
# Pod 1 dies at step 45000. last.ckpt was uploaded to R2 at step 45000.
# Pod 2 resumes:
python src/train.py ckpt_path=r2:synth-data/checkpoints/flow-simple/run-123/last.ckpt
```

Uses the same `r2:` prefix resolution from eval #92 (`resolve_ckpt_path()`).
Lightning handles all the resume logic (optimizer state, scheduler, epoch counter).

### 3. RunPod training launcher

Simple script — not a backend abstraction:

```python
# scripts/runpod_train.py
def launch_training(
    experiment: str,
    config_overrides: list[str],
    gpu_type: str = "NVIDIA RTX A5000",
    image: str = "ktinubu/synth-perm-train:latest",
):
    """Launch a single training pod on RunPod."""
    cmd = f"python src/train.py experiment={experiment} {' '.join(config_overrides)}"
    pod = runpod.create_pod(
        name=f"train-{experiment}-{timestamp}",
        image_name=image,
        gpu_type_id=gpu_type,
        docker_args=cmd,
        env={
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "RCLONE_CONFIG_R2_TYPE": "s3",
            # ... R2 credentials
        },
    )
    return pod
```

Make target: `make runpod-train EXPERIMENT=surge/flow_simple`

No reconciliation, no batch submission. One pod, one training run.

### 4. Docker training image

Options:

- **Extend data pipeline Docker** — add CUDA, torch, training deps
- **Separate Dockerfile** — `docker/train/Dockerfile`
- **Shared base + stage-specific layers** — base (Python, system deps) → train (CUDA, torch) / pipeline (rclone, headless)

Leaning toward separate. Training needs CUDA + torch + model deps. Data pipeline
needs rclone + VST + headless rendering. Overlap is small (Python, h5py).

### 5. W&B config cleanup

Current issues:

- `entity: "benhayes"` hardcoded → `entity: ${oc.env:WANDB_ENTITY,synth-setter}`
- `project: "synth-permutations"` → maybe rename to match repo direction?
- `log_model: true` uploads ckpts to W&B — redundant if R2 callback exists.
  Keep both? W&B artifacts have nice UI for model registry. R2 is for raw
  checkpoint files. Probably keep `log_model: true` and add R2 as the
  durable/fast option.

### 6. Training CI

Smoke test: train 2 epochs on tiny fixture dataset, verify:

- Checkpoint file exists
- W&B run logged (mock or offline mode)
- CSV metrics file has expected columns
- Model can be loaded from saved checkpoint

Mark `@pytest.mark.slow`. Run on GPU runner or CPU-only (small model).

## Dependency on R2 epic (#99)

| Training work item     | Depends on R2 issue                       |
| ---------------------- | ----------------------------------------- |
| R2 checkpoint callback | #90 (rclone wrapper)                      |
| Resume from R2         | #92 (checkpoint sync)                     |
| RunPod launcher        | #90 (rclone wrapper for credential setup) |
| Docker image           | Independent                               |
| W&B cleanup            | Independent                               |
| Training CI            | Independent                               |

## Open questions

1. **Should RunPod training pods auto-terminate?** Data pipeline workers
   terminate after all shards are done. Training doesn't have a natural
   "done" signal — it runs until `max_epochs` or early stopping. RunPod
   has `max_bid_price` and timeout settings. Use those?

2. **Multi-GPU training on RunPod?** Current models are small enough for
   single GPU. When/if we need multi-GPU, do we use RunPod multi-GPU pods
   or move to a different provider? Not worth designing for now.

3. **Checkpoint garbage collection in R2.** If we upload every 5000 steps
   for a 100-epoch run, that's a lot of checkpoints. Should we mirror
   `save_top_k` logic in the R2 uploader? Or just upload best + last?

4. **W&B artifact vs R2 for checkpoints.** Currently `log_model: true`
   uploads to W&B. R2 callback uploads to R2. Do we want both? W&B is
   nicer for browsing/comparing. R2 is faster for download and cheaper
   at scale. Probably keep both but make R2 the primary for eval download.

5. **Cost model.** Training on RunPod A5000: ~$0.15/hr. A 12-hour training
   run: ~$1.80. With checkpoint callback, worst-case loss on crash is
   5000 steps (the `every_n_train_steps` interval). Acceptable.

## Estimated effort

| Item                   | Effort                     | Priority              |
| ---------------------- | -------------------------- | --------------------- |
| R2 checkpoint callback | Small (1 file + tests)     | P1 — crash resilience |
| Resume-from-R2         | Already covered by #92     | P1                    |
| RunPod launcher        | Small (1 script)           | P2 — nice to have     |
| Docker training image  | Medium (new Dockerfile)    | P2                    |
| W&B cleanup            | Small (config changes)     | P2                    |
| Training CI            | Medium (fixtures, CI yaml) | P2                    |
