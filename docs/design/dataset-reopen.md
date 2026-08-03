# Dataset reopen: extending a finalized dataset without re-rendering

## Problem

Growing `surge_simple` from 440k to 2M rows must not re-render the existing
440k rows, and must leave the finalized 440k root byte-identical so every
config, checkpoint, and baseline that references it keeps working.

## Why extension is possible

Row parameters are seeded from `(split_master_seed, sample_offset + i, attempt)`
only — shard IDs, shard layout, and dataset size never reach the RNG:

```python
seed=split_seed,                          # spec.py: literal from train_val_test_seeds
sample_offset=split_shard_id * sps,       # spec.py: split-LOCAL index
sample_idx=render_cfg.sample_offset + i,  # writers.py
seed_for_sample(master_seed, sample_idx, attempt)  # seeding.py
```

Because `sample_offset` derives from the **split-local** shard index, enlarging
`train_val_test_sizes[0]` regenerates every original shard position unchanged
and continues the stream:

- train shards `0..175` keep identical `(seed, sample_offset)` in both specs
- shard 176 lands at `(42, 440000)`, exactly continuing shard 175's `(42, 437500)`
- train offsets stay contiguous `0..1_997_500` step `2500`

So a grown spec *is* the extension. No new seeding field is required.

## Approach

Copy the finalized root, reopen the copy as an incomplete dataset, and let the
existing generation + finalize machinery fill in the missing shards. The
resume skip-probe is already keyed on shard ID:

```python
def shard_has_complete_attempt(spec: DatasetSpec, shard_id: int) -> bool:
    """...
    The worker skip-probe: a complete, non-invalidated attempt means the shard
    is already staged and need not be re-rendered (#750 resumability).
```

No new fragment-commit code, no cross-root fragment references, and
restartability is inherited — a half-extended root is just an incomplete
dataset.

## The renumbering hazard

`split_shard_ranges` orders train → val → test, so growing train shifts
val/test:

| spec | num_shards | train      | val          | test         |
| ---- | ---------- | ---------- | ------------ | ------------ |
| 440k | 192        | `(0, 176)` | `(176, 184)` | `(184, 192)` |
| 2M   | 816        | `(0, 800)` | `(800, 808)` | `(808, 816)` |

Shards `176..191` flip from val/test to train. Left in place, the copied
val/test staging would make the skip-probe falsely skip 16 real train shards,
and finalize would search for val fragments under `train.lance/data/`.

Reopen therefore **deletes staged metadata at or above the old train
boundary**. Val/test re-render at their new IDs — 40k rows, ~2.5% of the
delta — and produce identical rows, because their offsets are split-local:

```
orig val  shard 176 -> (43, 0) | grown val  shard 800 -> (43, 0)
orig test shard 184 -> (44, 0) | grown test shard 808 -> (44, 0)
```

## Operation

1. Server-side copy `source_root` → `dest_root`.
2. Rewrite the spec at `dest_root`: `train_val_test_sizes`, `run_id`, and
   `r2.prefix`.
3. Delete `dataset.complete` so finalize does not short-circuit.
4. Delete staged metadata for `shard_id >= old_train_shard_count`.
5. Delete `metadata/shard-claims.lance`; `populate` recreates it over the new
   shard range, and already-staged shards short-circuit on the skip-probe.
6. Run the normal generate + finalize pipeline against `dest_root`.

`dataset.json` is deliberately **kept**: its entries pin the original winning
attempt per shard, preserving exactly which attempt became canonical for
shards `0..175`. Stale entries for the discarded IDs are inert —
`select_checked_winner` sorts by `attempt.name != preferred_name`, a stable
sort that leaves ordering unchanged when nothing matches.

Step 2 is the load-bearing one: `r2.prefix` drives every staging URI, so a
missed rewrite would make workers write into the **source** root — the one
thing this design exists to protect. It is asserted, not assumed.

## API

```python
# synth_setter/pipeline/data/dataset_reopen.py

@dataclass(frozen=True)
class ReopenPlan:
    """Shard-range partition a reopen produces; computed before any write."""
    source_root_uri: str
    dest_root_uri: str
    dest_spec: DatasetSpec
    preserved_shard_ids: range   # staging kept; skip-probe short-circuits these
    discarded_shard_ids: range   # staging deleted; renumbered val/test
    pending_shard_ids: range     # never staged; the delta to render

def validate_reopenable(source_spec: DatasetSpec, new_sizes: tuple[int, int, int]) -> None:
    """Reject specs that cannot be extended coherently.

    Raises when the source has no ``train_val_test_seeds`` (legacy
    ``base_seed + shard_id`` derivation does not survive renumbering), when
    train shrinks, when val/test sizes change, or when any size is not a
    multiple of ``render.samples_per_shard``.
    """

def plan_reopen(source_spec, new_sizes, *, dest_run_id) -> ReopenPlan:
    """Partition the shard space and build the destination spec. No writes."""

def reopen_dataset(source_root_uri, new_sizes, *, dest_run_id=None, dry_run=False) -> ReopenPlan:
    """Copy, reopen, and return the plan; ``dry_run`` skips every write."""
```

CLI — the copy is source-sized, so writing is opt-in behind `--apply`:

```
synth-setter-reopen-dataset --source r2://… --train-size 2000000 [--dest-run-id …] [--apply]
```

## Behaviour changes

`stats.npz` is recomputed by finalize over the full grown train split, so the
extended dataset carries its own normalization rather than inheriting the
source's. The source root is untouched, so existing configs and checkpoints
are unaffected; runs against the extended root are not numerically comparable
to source-root checkpoints without retraining.

Orphaned val/test fragment data files from the source remain under
`<split>.lance/data/` after re-render. Finalize commits an `Overwrite`
transaction, so they are absent from the manifest — dead bytes, not dead
rows.

## Validation

Real-R2, no mocks, PR-presubmit (`test-dataset-reopen.yml`): generate a smoke
dataset with non-zero val/test, reopen it with a larger train size, re-run
generation, and finalize. Pins that the preserved shards were **skipped** and
not re-rendered, that the delta and renumbered val/test rendered, and that the
preserved rows are byte-identical to the source root's.
