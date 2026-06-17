# R2 Credential Centralization

> How synth-setter resolves Cloudflare R2 access credentials and projects them
> into the dialects each current consumer needs. This is a transitional slice
> toward the provider-neutral object-storage contracts in
> [object-storage-contracts.md](object-storage-contracts.md), not the final
> storage abstraction.

| Field        | Value      |
| ------------ | ---------- |
| Status       | Draft      |
| Last Updated | 2026-06-17 |
| Tracking     | #138       |

______________________________________________________________________

## 1. Problem

The repo currently talks to Cloudflare R2 through `rclone` and Lance. R2 is an
S3-compatible provider, but the application model leaks the provider everywhere:
`DatasetSpec.r2`, `R2Location`, `r2://` URIs, and `RCLONE_CONFIG_R2_*`
credentials.

The target architecture is provider-neutral object storage, documented in
[object-storage-contracts.md](object-storage-contracts.md). It replaces
provider-shaped public types with `StorageSettings`, `StorageConfig`,
`ObjectLocation`, `DatasetStorageLayout`, and an `ObjectStorage` facade that can
keep using rclone for bulk sync.

This PR is intentionally narrower. It addresses the credential-resolution layer
first while preserving public contracts, so the current R2-shaped code has one
source of truth before the larger rename/facade work starts:

1. **One logical secret, four env-var vocabularies.** The same R2 access
   key / secret / endpoint is expressed as `RCLONE_CONFIG_R2_*` (rclone + Lance),
   `AWS_*` (Lance's Rust object_store, DuckDB httpfs), `WANDB_S3_ENDPOINT_URL`
   (W&B), and `R2_ACCOUNT_ID` (SkyPilot's R2 adaptor). Nothing derives these from
   each other — `.env.example` repeats the same endpoint URL four times under
   four names, kept in sync by hand.

2. **The key list is copy-pasted.** `_SECRET_R2_ENV_KEYS` (`r2_io`),
   `_WORKER_ENV_KEYS` and `_R2_RCLONE_CONSTANTS` (`skypilot_launch`), plus test
   mappings and 10+ GHA workflows each re-spell the `RCLONE_CONFIG_R2_*` names.
   `RCLONE_CONFIG_R2_TYPE=s3` / `PROVIDER=Cloudflare` was defined in two modules.

3. **Resolution choreography is scattered.** `ensure_r2_env_loaded`,
   `resolve_worker_env`, and the test conftest each load and validate env
   independently, with no shared source of truth for *what the names are*.

## 2. Goals / Non-Goals

**Goals**

- One object owns the canonical env-var names, structural constants, and the
  endpoint derivation — every consumer imports from it rather than re-spelling.
- A strict, frozen Pydantic value parsed once at the trust boundary, from which
  each dialect (Lance `storage_options`, the rclone env block) is *projected*.
- Behavior-preserving: public helpers (`r2_io.r2_storage_options`, the launcher
  forwarding) keep their contracts so the migration is invisible to call sites.
- Make the follow-on provider-neutral storage migration easier by collapsing
  today's credential duplication before broad call-site renames.

**Non-Goals**

- No new storage facade in this PR. The facade, `ObjectLocation`, and
  `DatasetStorageLayout` are specified in
  [object-storage-contracts.md](object-storage-contracts.md).
- No change to rclone's role. The object-storage contracts keep rclone as the
  first backend for bulk sync and probes.
- No change to the `R2Location` URI schema in this PR. The follow-on storage
  phase will replace it with `DatasetSpec.storage.root: {bucket, key}` and can
  break old specs.
- The `AWS_*` / `WANDB_*` projections and the GHA-workflow / `.env.example`
  collapse are **out of scope for this PR** (see §5).

## 3. Design

A single `R2Credentials` model (`pipeline/schemas/r2_credentials.py`, alongside
the other strict trust-boundary schemas) is the source of truth.

**Functional core (pure value).** The model holds `access_key_id`,
`secret_access_key`, `endpoint`, and the two rclone structural fields
(`rclone_type`, `rclone_provider`); the secrets are `SecretStr` so a stray
`repr`/log can't leak them. It is `strict=True, frozen=True, extra="forbid"`.
`lance_storage_options()` projects the Lance object-store dialect —
`{access_key_id, secret_access_key, endpoint, region="auto"}` for `lance.dataset`
/ `lance.write_dataset`. `rclone_env()` projects the resolved `RCLONE_CONFIG_R2_*`
block; `ensure_r2_env_loaded` writes it back into `os.environ` so the rclone
subprocess reads the same stripped, blank-free, default-filled, derived values
that were validated — never a raw/blank/padded dotenv value.

**Imperative shell.** `R2Credentials.from_env()` (the `env` argument defaults to
`os.environ`; tests pass a plain mapping) resolves the canonical names, applies
the structural defaults (caller values win), derives the endpoint from
`R2_ACCOUNT_ID` when the explicit endpoint var is absent, and raises
`RuntimeError` listing any missing/blank secret.

**Canonical names live once.** Module constants — `ENV_ACCESS_KEY_ID`,
`ENV_SECRET_ACCESS_KEY`, `ENV_ENDPOINT`, `ENV_TYPE`, `ENV_PROVIDER`,
`ENV_ACCOUNT_ID`, plus the derived tuples `SECRET_ENV_KEYS`, `RCLONE_ENV_KEYS`,
and `STRUCTURAL_DEFAULTS` — are imported by every consumer.

## 4. Migration (this PR)

- `r2_io` drops its private `_SECRET_R2_ENV_KEYS` / `_R2_STRUCTURAL_DEFAULTS`
  duplicates. `r2_storage_options()`, `ensure_r2_env_loaded()`, and
  `is_r2_reachable()` all route validation through `R2Credentials.from_env()`, so
  there is one definition of "present and non-blank" and the account-id endpoint
  derivation applies uniformly. `ensure_r2_env_loaded` then writes
  `creds.rclone_env()` back into `os.environ`, normalizing it to the resolved
  values so the rclone auth ping never inherits a blank/padded var.
- `skypilot_launch._WORKER_ENV_KEYS` is composed from `RCLONE_ENV_KEYS`;
  `_R2_RCLONE_CONSTANTS` is the shared `STRUCTURAL_DEFAULTS`. The duplicated
  literal tuple and the second copy of the type/provider constants are deleted.
  `resolve_worker_env` now treats a blank value as absent, matching `from_env`,
  so a `.env` line `KEY=` never forwards an empty credential to a worker.
- Tests that re-spelled the `RCLONE_CONFIG_R2_*` key list import the canonical
  `SECRET_ENV_KEYS` / `STRUCTURAL_DEFAULTS` / `RCLONE_ENV_KEYS` directly.

Public contracts are unchanged; the existing `r2_io` and launcher test suites
pass as-is, and new tests pin the model and its projections.

## 5. Follow-ups (tracked, not in this PR)

- Implement the provider-neutral storage contracts from
  [object-storage-contracts.md](object-storage-contracts.md) as a phase under the
  existing storage epic. The phase owns the breaking `R2Location` /
  `DatasetSpec.r2` removal, `s3://bucket/key` CLI normalization, and
  rclone-backed `ObjectStorage` facade.
- Project `AWS_*` (Lance-rust / DuckDB) and `WANDB_S3_ENDPOINT_URL` from
  `R2Credentials`; collapse the four endpoint repetitions in `.env.example`.
- Replace the per-workflow inline `RCLONE_CONFIG_R2_*` lists in GHA / compute
  templates with a single documented reference to the canonical set.
- Add a lint/grep gate (sibling of the existing `agent/hooks/`) forbidding
  `os.environ[...]` reads of storage credentials outside this module.

## 6. Alternatives Considered

- **Jump straight to the provider-neutral storage facade.** Deferred. The
  contracts are agreed in
  [object-storage-contracts.md](object-storage-contracts.md), but this PR keeps
  the blast radius to credential centralization. It gives the current R2-shaped
  code one credential source before the breaking type/config rename.
- **Standardize immediately on `fsspec`/`s3fs` or `obstore` as the single
  transport.** Deferred. The storage facade should land first while delegating
  to rclone. A later client swap can be benchmarked behind the facade without
  changing application call sites.
- **A `BaseSettings` that auto-reads env on construction.** Rejected in favor of
  an explicit `from_env` classmethod so the trust boundary (and its validation
  error) is a visible call, and so the pure model stays trivially constructible
  in tests without environment manipulation.

## See also

- [storage-provenance-spec.md](storage-provenance-spec.md) — authoritative R2
  paths and W&B artifact conventions
- [object-storage-contracts.md](object-storage-contracts.md) — target
  provider-neutral storage contracts and migration shape
- [skypilot-compute-integration.md](skypilot-compute-integration.md) — worker
  env-var resolution and forwarding
