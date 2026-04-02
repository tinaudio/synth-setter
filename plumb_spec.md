# Synth-Setter: Specification

## Cross-Cutting Invariants

The input spec is write-once and immutable. First `generate` materializes the spec and
uploads to R2. No mechanism exists to modify it after creation. All subsequent operations
read from the R2 spec, never from the config file.

R2 storage is the source of truth for pipeline state. Shard completeness is determined by
file existence and validation markers, not metadata, caches, or provider APIs.
Reconciliation derives state from R2 files on every invocation.

Workers write only under `metadata/workers/`. The canonical `data/shards/` directory is
written exclusively by the finalize stage. Staging and canonical paths are separate —
worker uploads cannot reach `data/shards/`.

The `.valid` lifecycle marker is the commit point for staged shards. It is written as the
final step after upload completes. Presence signals the worker completed its full lifecycle:
render, validate, upload, bookkeeping.

Shard identity is deterministic and infrastructure-independent. Shard IDs are logical
(`shard-000042`), seeds are derived as `base_seed + shard_id`, and the same config always
produces the same spec. Infrastructure IDs appear only in staging filenames and metadata.

Pydantic strict mode is used at all trust boundaries. Config parsing, JSON from R2, and
worker reports validate with `strict=True` and no type coercion. Internal typed containers
use frozen dataclasses.

All rclone operations use `--checksum` to verify transfer integrity. Checksum mismatches
trigger failure and re-transfer.

Every W&B run declares inputs via `run.use_artifact()` and outputs via
`run.log_artifact()`. Every run includes `github_sha` in `wandb.config`. Only
`use_artifact()` creates lineage links — `api.artifact()` does not.

The `dataset.complete` marker gates consumption. No pipeline command reads from R2 paths
that lack their completion marker. Once written, canonical data under `data/` is immutable.

Run IDs follow the format `{config_id}-{YYYYMMDDTHHMMSSZ}`. No random UUIDs. IDs are
reconstructible from the config filename stem and a UTC timestamp.
