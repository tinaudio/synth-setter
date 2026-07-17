# Deterministic Dataset Seeding

This page is the canonical design note for dataset-generation seeding. The goal
is reproducible parameter rows across reruns, worker counts, retry history, and
shard dispatch order. Audio bit identity is only guaranteed under controlled
renderer conditions: same spec, code, Docker image, plugin version, and hardware
class. VST DSP output can still vary across CPU architectures.

## Seed Flow

The frozen `DatasetSpec` is the reproducibility unit. Human-authored Hydra
config supplies distinct `train_val_test_seeds`. Materialization assigns each
shard its split master and split-local starting row:

```text
ShardSpec.seed = train_val_test_seeds[split]
ShardSpec.sample_offset = split_local_shard_id * samples_per_shard
```

The launcher passes both values into `RenderConfig`. Inside the writer loop,
every row receives a `SampleSeed`:

```text
master_seed = RenderConfig.base_seed
sample_idx = RenderConfig.sample_offset + shard_row_index
max_attempts = RenderConfig.attempts_per_sample
```

`generate_sample()` derives each parameter-sampling RNG from:

```text
seed_for_sample(master_seed, sample_idx, attempt)
```

The `attempt` term is the loudness-gate retry number. Attempt 0 is the first
draw; if that render is silent, attempt 1 gets a new deterministic RNG, and so
on until an audible sample is accepted or `max_attempts` is exhausted. This makes
the accepted row deterministic for a fixed loudness threshold while preventing a
silent row from advancing any global RNG stream.

## Implementation Status

The current implementation includes:

- `src/synth_setter/data/vst/seeding.py` owns `seed_for_sample()` and
  `rng_for_sample()`. The encoding is golden-tested because changing it reseeds
  existing datasets.
- VST parameter sampling accepts an explicit `numpy.random.Generator`; it no
  longer depends on process-global `random` or `numpy.random` streams for seeded
  dataset rows.
- The Lance writer projects `base_seed`, `sample_offset`, and
  `attempts_per_sample` through `ShardMetadata`.
- Shard validation rejects seed-provenance mismatches in Lance schema metadata.
  Sidecars without seed fields still validate structurally.
- Tests cover same-seed repeatability, different-seed divergence, worker-count
  independence, shard-size prefix stability, direct row derivation, retry
  attempt semantics, and same-config repeat runs with seed metadata.

## Guarantees

- Same frozen spec plus same renderer environment produces the same parameter
  rows for each split-local row.
- Training rows form a stable prefix as the training split grows; validation
  and test rows stay fixed when training or shard sizes change.
- A row's parameter draw is independent of worker count, shard dispatch order,
  shard boundaries, and previous rows' retry counts.
- Re-rendering the same shard with the same `base_seed`, `sample_offset`,
  `attempts_per_sample`, and loudness threshold gives the same accepted attempt
  and parameter row.
- Validation catches a shard whose stored seed provenance disagrees with the
  spec-derived expected values.

These row-level guarantees apply to `param_sample_cadence="sample"`. Shard
cadence intentionally reuses one patch within each shard, so changing shard
boundaries changes that probe's grouping.

## Legacy Specs

`train_val_test_seeds=None` preserves the original `base_seed + shard_id`
derivation. This keeps materialized specs and shard metadata from older runs
replayable. New Hydra-composed datasets provide explicit split masters.

## Out Of Scope

- Per-row accepted attempt is not stored as a dataset column. The current
  provenance records the split master, sample offset, and attempt budget, not
  each accepted retry.
- Restart/idempotency tests for partially written shards belong to the writer
  resume design, not the seed derivation contract.
- Full audio bit identity for real VST renders is not asserted. The current
  contract pins deterministic parameter rows; audio equality requires renderer
  state and DSP determinism beyond this seeding layer.

## Tracked Work

- [#884](https://github.com/tinaudio/synth-setter/issues/884): deterministic
  per-sample seeding and dataset row reproducibility.
- [#489](https://github.com/tinaudio/synth-setter/issues/489): plugin/render
  state cadence effects that can affect audio even when parameter rows match.
