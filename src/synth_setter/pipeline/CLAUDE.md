# Pipeline-Specific Rules

Auto-loaded by agents when editing files under `src/synth_setter/pipeline/`.
Architectural rationale lives in
[../../../docs/design/data-pipeline.md](../../../docs/design/data-pipeline.md);
this file is the imperative rule sheet.

## Invariants

- **R2 is the source of truth.** Pipeline state is determined by R2 contents,
  not by metadata files, reports, or a coordination database. If R2 disagrees
  with anything else, R2 wins.
- **Worker / finalize write boundary.** Workers only write under
  `metadata/workers/`. Finalize is the only writer of `data/`.
- **Never write to `data/shards/` outside finalize.** Workers stage shards
  under `metadata/workers/<worker-id>/` and finalize moves them.
- **Shard IDs are logical and deterministic.** `shard-000042` is computed
  from the input spec, never from worker ID, hostname, or wall-clock time
  (infrastructure-independent IDs are what make reconciliation work).

## Validation tiers

- **Workers** run the full 4-check shard validation before upload:
  - Structural — opens in the shard's container format (HDF5 file, WebDataset
    tar, or Lance dataset directory).
  - Shape — datasets match the expected shape.
  - Value — values are finite and within bounds.
  - Row count — sample count matches `render.samples_per_shard`.
- **Finalize** runs structural validation only (the workers' full check is the
  trust anchor; finalize must not re-run it on every shard).

## Adding new code under `pipeline/`

- All Pydantic models in `pipeline/schemas/` use `strict=True`. These are
  trust boundaries (R2 JSON, worker reports, Hydra-composed specs).
- `pipeline/data/` utilities are CLI-callable via `python -m`. Add tests under
  `tests/pipeline/data/` mirroring the source layout.

## See also

- [data-pipeline.md](../../../docs/design/data-pipeline.md) — design rationale
- [storage-provenance-spec.md](../../../docs/design/storage-provenance-spec.md)
  — authoritative R2 paths and W&B artifact conventions
