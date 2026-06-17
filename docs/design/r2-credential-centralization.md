# Object Storage Credential Centralization

> How synth-setter resolves provider-neutral S3-compatible storage settings and
> projects them into the current rclone and Lance dialects.

| Field        | Value      |
| ------------ | ---------- |
| Status       | Draft      |
| Last Updated | 2026-06-17 |
| Tracking     | #138       |

______________________________________________________________________

## Problem

The repo currently uses Cloudflare R2, but R2 is a provider profile, not the
application model. The old credential shape made rclone's
`RCLONE_CONFIG_R2_*` env convention the source of truth, then adapted that shape
for Lance and launcher code. That is the wrong direction for a codebase that
should also work with another S3-compatible store.

The storage contracts in
[object-storage-contracts.md](object-storage-contracts.md) are canonical:
application code loads `SYNTH_SETTER_STORAGE_*` settings, passes a strict
`StorageConfig`, and treats rclone env vars as backend projection output.

## Goals

- Use `StorageSettings` (`pydantic_settings.BaseSettings`) as the only
  environment-reading storage settings type.
- Use strict/frozen `StorageConfig` as the env-free value object passed through
  code.
- Project Lance `storage_options` and rclone env from `StorageConfig`.
- Keep rclone as the current transfer backend while making R2 an implementation
  detail.
- Accept breaking changes in this pre-user migration; do not preserve
  `R2Credentials` or legacy r2 credential promotion as public contracts.

## Current PR Slice

This PR lands the credential slice of the provider-neutral direction:

- `pipeline/schemas/object_storage.py` defines `StorageSettings`,
  `StorageConfig`, `ObjectLocation`, and a small rclone-backed `ObjectStorage`
  facade.
- `r2_io.r2_storage_options()`, `ensure_r2_env_loaded()`, and
  `is_r2_reachable()` now read canonical `SYNTH_SETTER_STORAGE_*` settings and
  project the current rclone env block at the subprocess boundary.
- `skypilot_launch.resolve_worker_env()` reads the same canonical settings and
  forwards the projected rclone env to existing worker templates.
- `R2Credentials` is intentionally removed. New code must not reintroduce it as
  the canonical storage model.

The old `r2_io` module name, `r2://` helper surface, and `R2Location` dataset
layout remain only as existing migration debt. They should move behind
`ObjectStorage` / `DatasetStorageLayout` in the next phase rather than being
expanded.

## Follow-Ups

- Replace `DatasetSpec.r2` / `R2Location` with `DatasetSpec.storage` and
  `DatasetStorageLayout`.
- Persist only `storage.root: {bucket, key}`; remove legacy
  `r2_bucket` / `r2_prefix_root` / `r2_prefix` promotion.
- Migrate public string inputs to accept only `s3://bucket/key`, normalized to
  `ObjectLocation`.
- Move the remaining rclone subprocess helpers behind the `ObjectStorage`
  facade.
- Update compute templates and workflow helpers so rclone remote names are
  backend details, not user-facing configuration.

## See Also

- [object-storage-contracts.md](object-storage-contracts.md) — canonical
  provider-neutral contracts and migration shape
- [skypilot-compute-integration.md](skypilot-compute-integration.md) — worker
  env-var resolution and forwarding
