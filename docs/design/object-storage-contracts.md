# Design Note: Provider-Neutral Object Storage Contracts

> **Status**: Draft
> **Last Updated**: 2026-07-11

## Context

The current storage surface is named around Cloudflare R2 even though the
runtime requirement is broader: synth-setter needs an S3-compatible object
store. R2 is one provider, not the application model.

The existing code has two useful pieces to preserve:

- rclone remains the right implementation for bulk directory sync, especially
  no-clobber and checksum-verified copies.
- Lance already consumes `s3://` URIs plus object-store `storage_options`.

The target design keeps those backend strengths while removing provider and
tool details from dataset, training, evaluation, and CLI logic.

## Goals

- Application code passes typed object locations, not `r2://`, `s3://`, or
  `r2:` strings.
- Credentials are loaded once from provider-neutral settings and projected into
  backend-specific formats.
- rclone stays available behind the storage facade for bulk file and directory
  transfer.
- Lance and W&B still receive the URI formats they require, but only through
  adapter methods.
- Breaking changes are acceptable. There are no external users to protect yet.

## Non-Goals

- Do not remove rclone from bulk sync in the first migration.
- Do not add a new object-store client until call sites are hidden behind the
  facade.
- Do not support arbitrary generic S3 buckets in places that need the configured
  synth-setter object store. A bare `s3://` URI means "this configured
  S3-compatible store" unless a future API explicitly accepts external stores.

## Contracts

### `StorageSettings`

`StorageSettings` is the only type that reads environment variables, and
`storage_settings_from_sources(env_file)` is the loading entry point: it reads
the dotenv file (when it exists) with non-blank dotenv values taking precedence
over process env, then validates. It uses `pydantic_settings.BaseSettings`
because loading settings is ambient process-boundary behavior.

```python
class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SYNTH_SETTER_STORAGE_",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        str_strip_whitespace=True,
    )

    access_key_id: SecretStr
    secret_access_key: SecretStr
    endpoint_url: str
    provider: ObjectStoreProvider = ObjectStoreProvider.R2
    region: str = "auto"
    default_bucket: str | None = None
    rclone_type: str = "s3"  # current backend type; tests may use "local"

    def to_config(self) -> StorageConfig: ...


def storage_settings_from_sources(env_file: Path | None = None) -> StorageSettings: ...
```

Expected environment shape:

```text
SYNTH_SETTER_STORAGE_PROVIDER=r2
SYNTH_SETTER_STORAGE_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
SYNTH_SETTER_STORAGE_REGION=auto
SYNTH_SETTER_STORAGE_ACCESS_KEY_ID=...
SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY=...
SYNTH_SETTER_STORAGE_DEFAULT_BUCKET=intermediate-data
SYNTH_SETTER_STORAGE_RCLONE_TYPE=s3
```

The rclone remote name is not configurable: every current consumer (worker
templates, workflow env forwarding, `rclone lsd r2:` probes, `r2://` URI
translation) speaks the pinned `r2` remote dialect, so the projection always
emits `RCLONE_CONFIG_R2_*` keys.

### `StorageConfig`

`StorageConfig` is the immutable value object passed through application code.
It should use `pydantic.BaseModel` with strict, frozen validation. It must not
read env.

```python
class StorageConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    provider: ObjectStoreProvider
    endpoint_url: str
    region: str
    access_key_id: SecretStr
    secret_access_key: SecretStr
    default_bucket: str | None
    rclone_type: str

    def lance_storage_options(self) -> dict[str, str]: ...
    def rclone_env(self) -> dict[str, str]: ...
    def storage_env(self) -> dict[str, str]: ...
```

The important invariant: rclone env vars, Lance storage options, and the
canonical `SYNTH_SETTER_STORAGE_*` env block are all projections from this
object. None of the env dialects is canonical.

### `ObjectLocation`

`ObjectLocation` is a provider-neutral pointer to a bucket/key. It should use a
strict, frozen `BaseModel` because it is serialized into specs and config.

```python
class ObjectLocation(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    bucket: str
    key: str

    @property
    def uri(self) -> str: ...
```

The `uri` property is an adapter output for Lance and W&B. Callers should
prefer the storage facade methods over reading it directly.

Slash-safe key joining should live in a private helper used by
`DatasetStorageLayout` and backend adapters, not on the public `ObjectLocation`
contract:

```python
def _join_object_key(prefix: str, name: str) -> str: ...
```

### `DatasetStorageLayout`

`DatasetStorageLayout` owns synth-setter's object naming convention. It replaces
`R2Location`.

```python
class DatasetStorageLayout(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    root: ObjectLocation

    def input_spec(self) -> ObjectLocation: ...
    def config_yaml(self) -> ObjectLocation: ...
    def dataset_card(self) -> ObjectLocation: ...
    def complete_marker(self) -> ObjectLocation: ...
    def shard(self, shard: ShardSpec) -> ObjectLocation: ...
    def split_h5(self, split: Split) -> ObjectLocation: ...
    def split_lance(self, split: Split) -> ObjectLocation: ...
    def split_wds_brace(self, shard_range: tuple[int, int]) -> ObjectLocation: ...
    def stats(self) -> ObjectLocation: ...
    def lance_versions(self, shard: ShardSpec) -> ObjectLocation: ...
```

The persisted `DatasetSpec` field should become:

```python
storage: DatasetStorageLayout
```

Persisted specs should use the nested root shape:

```yaml
storage:
  root:
    bucket: intermediate-data
    key: data/task-name/run-id/
```

The old `r2: R2Location` field and legacy `r2_bucket` / `r2_prefix_root` /
`r2_prefix` promotion can be removed as a breaking change.

### `ObjectStorage`

`ObjectStorage` is the facade that owns backend selection. The first
implementation can delegate all transfer and probes to rclone while hiding
rclone paths and env wiring.

```python
class ObjectStorage:
    def __init__(self, config: StorageConfig) -> None: ...

    def check_auth(self) -> None: ...

    def upload_file(self, local: Path, dest: ObjectLocation) -> None: ...
    def download_file(self, src: ObjectLocation, dest: Path) -> None: ...
    def upload_dir(
        self,
        local: Path,
        dest: ObjectLocation,
        *,
        exclude: str | None = None,
    ) -> None: ...
    def download_dir_no_overwrite(self, src: ObjectLocation, dest: Path) -> None: ...

    def size(self, loc: ObjectLocation) -> int | None: ...
    def exists(self, loc: ObjectLocation) -> bool: ...
    def directory_exists(self, loc: ObjectLocation) -> bool: ...
    def purge_prefix(self, loc: ObjectLocation) -> None: ...

    def lance_uri(self, loc: ObjectLocation) -> str: ...
    def lance_storage_options(self) -> dict[str, str]: ...
    def wandb_reference_uri(self, loc: ObjectLocation) -> str: ...
    def location_from_wandb_reference(self, uri: str) -> ObjectLocation: ...
```

`lance_uri()` and `wandb_reference_uri()` may return `s3://bucket/key`, but that
is an external boundary format, not the application representation.

## Usage Patterns After Migration

### Settings Load

Current pattern:

```python
r2_io.ensure_r2_env_loaded(env_file)
```

Target pattern:

```python
settings = StorageSettings(_env_file=env_file)
storage = ObjectStorage(settings.to_config())
storage.check_auth()
```

### Dataset Layout

Current pattern:

```python
spec.r2.shard_uri(shard)
spec.r2.stats_uri()
spec.r2.dataset_complete_marker_uri()
```

Target pattern:

```python
spec.storage.shard(shard)
spec.storage.stats()
spec.storage.complete_marker()
```

### Single-Object Upload And Download

Current examples:

- `spec_io.upload_spec()` uploads `spec.r2.input_spec_uri()`.
- `finalize_dataset.finalize_hdf5()` downloads each `spec.r2.shard_uri(shard)`.
- `train._upload_best_checkpoint()` uploads a checkpoint URI.

Target pattern:

```python
storage.upload_file(tmp_path, spec.storage.input_spec())
storage.download_file(spec.storage.shard(shard), local_path)
storage.upload_file(best_checkpoint, checkpoint_location)
```

### Bulk Directory Sync

Current examples:

- `SurgeDataModule.prepare_data()` downloads a dataset root with
  `r2_io.download_dir_no_overwrite()`.
- `eval._maybe_upload_output_dir()` mirrors a Hydra output dir to R2.
- Lance shard generation uploads directory datasets and `_versions/`.

Target pattern:

```python
storage.download_dir_no_overwrite(dataset_location, self.dataset_root)
storage.upload_dir(output_dir, eval_output_location)
storage.upload_dir(shard_path, shard_location, exclude="_versions/**")
storage.upload_dir(shard_path / "_versions", spec.storage.lance_versions(shard))
```

The implementation can still be rclone. The caller no longer knows.

### Existence And Size Probes

Current examples:

- Generation checks whether a shard object or Lance `_versions/` exists.
- Finalize checks for `dataset.complete` before doing work.

Target pattern:

```python
already_present = storage.directory_exists(spec.storage.lance_versions(shard))
existing_size = storage.size(spec.storage.shard(shard)) or 0
if storage.exists(spec.storage.complete_marker()):
    return
```

### Lance Read And Write

Current examples:

- `finalize_lance()` calls `r2_io.to_s3_uri()` and `r2_io.r2_storage_options()`.
- `validate_shard` opens each Lance shard from `s3://`.
- `add_embeddings` and `add_mp3_audio` treat `s3://` as this project's R2.

Target pattern:

```python
uri = storage.lance_uri(spec.storage.shard(shard))
options = storage.lance_storage_options()
dataset = lance.dataset(uri, storage_options=options)

write_lance_dataset(
    storage.lance_uri(spec.storage.split_lance(split)),
    schema,
    batches,
    storage_options=options,
)
```

The Lance helper functions can still accept `uri: Path | str` and
`storage_options: dict[str, str] | None`. Those are Lance adapter functions, so
`s3://` remains acceptable there.

### W&B References

Current examples:

- Training logs the model checkpoint as an `s3://` reference.
- Evaluation logs output artifacts as an `s3://` reference.
- The W&B resolver rewrites `s3://` back to `r2://` before download.

Target pattern:

```python
artifact.add_reference(storage.wandb_reference_uri(checkpoint_location), checksum=False)

for ref in wandb_refs:
    loc = storage.location_from_wandb_reference(ref)
    storage.download_file(loc, cache_dir / Path(loc.key).name)
```

W&B can keep storing `s3://` references. The rewrite logic is no longer spread
through training, eval, finalize, and resolver code.

### SkyPilot Worker Env

Current pattern:

```python
worker_env.update(resolve_worker_env(env_file))
os.environ["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] = ...
```

Target pattern:

```python
storage_config = storage_settings_from_sources(env_file).to_config()
worker_env.update(storage_config.rclone_env())
```

Compute templates should use storage-neutral placeholders where possible. If a
debug template is explicitly about rclone, it may still mention rclone env vars.

## Scanned Call-Site Categories

| Category                  | Current examples                                                          | Target surface                                                                 |
| ------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Dataset schema and layout | `DatasetSpec.r2`, `R2Location`, `spec.r2.*_uri()`                         | `DatasetSpec.storage`, `DatasetStorageLayout`, `ObjectLocation`                |
| Credential loading        | `r2_io.ensure_r2_env_loaded()`                                            | `StorageSettings` plus `ObjectStorage.check_auth()`                            |
| Lance credentials         | `r2_io.r2_storage_options()`                                              | `ObjectStorage.lance_storage_options()`                                        |
| Lance URI conversion      | `r2_io.to_s3_uri(spec.r2.shard_uri(...))`                                 | `ObjectStorage.lance_uri(loc)`                                                 |
| rclone transfers          | `r2_io.upload_dir()`, `download_to_path()`, `download_dir_no_overwrite()` | `ObjectStorage.upload_dir()`, `download_file()`, `download_dir_no_overwrite()` |
| Probes                    | `r2_io.object_size()`, `r2_io.r2_directory_exists()`                      | `ObjectStorage.size()`, `exists()`, `directory_exists()`                       |
| W&B references            | `to_s3_uri()` on log, `from_s3_uri()` on resolve                          | `wandb_reference_uri()` and `location_from_wandb_reference()`                  |
| SkyPilot env forwarding   | hard-coded `RCLONE_CONFIG_R2_*`                                           | `StorageConfig.rclone_env()`                                                   |
| CLI/config URI fields     | `r2://` accepted in docs and validation                                   | `s3://bucket/key`, normalized immediately to `ObjectLocation`                  |

## Intentional And Temporary Leaks

Some implementation details remain visible after a first migration:

- `s3://` remains visible at Lance and W&B adapter boundaries. This is required
  by those external APIs.
- rclone env keys remain visible inside the rclone backend and explicitly
  rclone-focused debug templates.
- Existing tests will still assert rclone argv behavior, but under
  `storage.backends.rclone` rather than application modules.

The following leaks should not remain:

- `R2Location` or `DatasetSpec.r2` in persisted specs.
- `r2://` in user-facing config validation.
- `RCLONE_CONFIG_R2_*` as the canonical source of credentials.
- Direct application imports of `to_s3_uri()`, `from_s3_uri()`, or
  `to_rclone_path()`.
- Application code constructing `r2:` paths.

## Migration Shape

1. Add `StorageSettings`, `StorageConfig`, `ObjectLocation`,
   `DatasetStorageLayout`, and `ObjectStorage`.
2. Replace `DatasetSpec.r2` with `DatasetSpec.storage`.
3. Rename Hydra `r2` config group to `storage`.
4. Move rclone subprocess construction into a storage backend module.
5. Migrate single-object upload/download and size/existence probes.
6. Migrate bulk directory sync call sites, keeping rclone behavior.
7. Migrate Lance call sites to `lance_uri()` and `lance_storage_options()`.
8. Migrate W&B reference creation and resolution to storage adapter methods.
9. Delete public `r2_io` translators and provider-shaped env source of truth.

## Risks

| Risk                                                                   | Impact                                                | Mitigation                                                                                                 |
| ---------------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| S3-compatible providers vary in endpoint and region behavior           | Credentials work for R2 but fail for another provider | Keep provider-specific projection logic in `StorageConfig` and test R2 plus a local S3-compatible stand-in |
| rclone remote env names are generated incorrectly                      | Bulk sync fails at runtime                            | Contract-test `rclone_env()` and rclone argv construction                                                  |
| `s3://` accepted as public shorthand is confused with arbitrary AWS S3 | Users expect cross-store reads                        | Document that `s3://` means the configured object store unless an API explicitly accepts external stores   |
| Large rename touches many tests                                        | Migration PR becomes noisy                            | Land in phases, but allow breaking schema changes early                                                    |

## Open Questions

### Decided

1. Persisted specs use `storage.root: {bucket, key}` rather than separate
   `storage.bucket` / `storage.prefix` fields.

2. `ObjectStorage` starts as rclone-only. Small probes (`size`, `exists`,
   `directory_exists`) move behind the facade first, but still delegate to
   rclone until a later benchmark-backed client decision.

3. Public CLI/config string locations accept only `s3://bucket/key`, then
   normalize immediately to `ObjectLocation`. No `bucket/key` shorthand.

   The affected overrides include:

   - `dataset_root_uri` in `finalize_dataset.yaml`
   - `datamodule.download_dataset_root_uri`
   - `evaluation.upload_output_dir_uri`
   - `training.upload_checkpoints_uri`
   - `copy_dataset_root_uri`

### Still Open

None.
