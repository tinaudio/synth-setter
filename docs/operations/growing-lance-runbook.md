# Growing Lance runbook

Operate bounded append-only train growth (`synth-setter-growing-lance`) from a
small finalized baseline to an explicit `max_train_shards`, while a training
job adopts each ready snapshot at epoch boundaries. Mechanism and invariants:
[architecture.md](../architecture.md) ("Growing Lance train snapshots").

Every command takes the frozen spec URI
(`r2://<bucket>/<prefix>metadata/input_spec.json`) and a branch name. Set
`SPEC_URI` and pick a branch once:

```bash
export SPEC_URI="r2://intermediate-data/<prefix>/metadata/input_spec.json"
export BRANCH="growing-a"
```

## 1. Initialize the branch (once)

Requires a finalized dataset (`dataset.complete` present). The baseline pins
to the finalized train dataset's current version automatically; pass
`--baseline-version N` only to pin an older manifest.

```bash
synth-setter-growing-lance init "$SPEC_URI" \
  --branch "$BRANCH" --max-train-shards 500 --num-extra-shards 100 \
  --work-dir ~/growing/operator
```

## 2. Start the producers (daemons, run until capacity)

One `grow` driver — it enqueues each bounded range, waits for generators to
stage every position, finalizes the append, and repeats:

```bash
synth-setter-growing-lance grow "$SPEC_URI" \
  --branch "$BRANCH" --work-dir ~/growing/operator --poll-seconds 30
```

N `generate` workers (one per render host; each drains the branch-isolated
claim queue and stages fragments):

```bash
synth-setter-growing-lance generate "$SPEC_URI" \
  --branch "$BRANCH" --work-dir ~/growing/worker --poll-seconds 30
```

All producers exit cleanly on their own once `max_train_shards` is reached.
`--poll-seconds 0` drains whatever is already staged and returns instead of
waiting (useful for cron or debugging). `enqueue` and `finalize` remain
available as manual single-step equivalents of what `grow` drives.

## 3. Start the materializer on the training host (daemon)

Incrementally appends each ready remote version into one shared local
`train.lance` and atomically advances `active.json`:

```bash
synth-setter-growing-lance materialize "$SPEC_URI" \
  --branch "$BRANCH" --local-root ~/growing/local \
  --work-dir ~/growing/materializer --poll-seconds 30
```

An up-to-date tick is cheap (one ready-tag read, no downloads), so short poll
intervals are fine.

## 4. Launch training

`dataset_root` still points at the hydrated baseline (val/test/stats come
from the standard materialization path); only the train split follows the
active record:

```bash
synth-setter-train experiment=<exp> \
  training.growing_active_record=~/growing/local/active.json \
  training.growing_refresh_epoch_interval=1 \
  datamodule.persistent_workers=false
```

- `growing_refresh_epoch_interval` wires the trainer's
  `reload_dataloaders_every_n_epochs`; leaving it `0` with an active record
  set never adopts snapshots (the datamodule warns at `fit` setup).
- `persistent_workers=true` is rejected outright — workers pin the old
  dataset version across reloads.
- Checkpoints record the exact adopted identity; resume validates and
  restores that local version before considering a newer ready snapshot.

## Failure behavior

- **Generator crash mid-render** — the claim lease expires and another worker
  re-claims the position; a crashed attempt leaves only an orphaned
  `.rendering` marker. Duplicate attempts cannot duplicate rows (one winner
  per position at finalize).
- **`grow` crash after the Lance append but before metadata publication** —
  rerun `grow` (or `finalize`); the deterministic pending identity on the
  branch transaction lets the rerun recognize and complete its own append.
- **`grow` raises "another pending growing refresh already exists"** — a
  pending request from a different ready snapshot is durable in
  `metadata/growing/<branch>/pending.json`. Inspect it; if it belongs to a
  crashed epoch of the same contract, rerun `finalize` to complete or clear
  it, then restart `grow`.
- **Materializer crash** — previous `active.json` stays valid; training keeps
  the last adopted snapshot. Restart the daemon; activation is monotonic and
  lock-serialized, so concurrent or stale materializers cannot regress it.
- **Training host loses metadata mid-run** — the datamodule logs a warning
  and retains the prior data whenever the active record fails exact identity
  validation; it never adopts a partially written snapshot.
- **DDP rank divergence** — a snapshot is adopted only when every rank
  validates the same identity; otherwise all ranks retain the prior data.
- **At capacity** — `enqueue`/`generate`/`finalize`/`grow` are safe no-ops
  and the daemons exit; the branch and every prior version stay readable.
