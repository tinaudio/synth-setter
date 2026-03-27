# Image Config: Remove Defaults + Add R2 Fields

## Purpose

Make `ImageConfig` fully explicit (no silent defaults) and add `r2_endpoint`
and `r2_bucket` as config fields. `r2_endpoint` moves from a BuildKit secret
to a committed config value — it's a URL, not a credential.

## Files to modify

### 1. `scripts/image_config.py`

- Remove default values from all static fields (`dockerfile`, `image`,
  `base_image`, `base_image_tag`, `build_mode`, `target_platform`,
  `torch_index_url`). Every field becomes required.
- Add `r2_endpoint: str` and `r2_bucket: str` (no defaults, required).

### 2. `configs/image/dev-snapshot.yaml`

Add:

```yaml
r2_endpoint: "https://efb9275d571811db929e83eb710b74a7.r2.cloudflarestorage.com"
r2_bucket: "intermediate-data"
```

### 3. `Makefile` (lines 90-102)

- Remove `r2_endpoint` from `DOCKER_SECRETS` (no longer a BuildKit secret).
- Add `--build-arg R2_ENDPOINT=$(R2_ENDPOINT)` alongside existing
  `--build-arg R2_BUCKET=$(R2_BUCKET)`.
- Keep `R2_ENDPOINT ?=` as an overridable variable for local use.

### 4. `docker/ubuntu22_04/Dockerfile` (`r2-config-base` stage)

- Add `ARG R2_ENDPOINT`.
- Replace `--mount=type=secret,id=r2_endpoint` with the build-arg in the
  rclone config `printf`.

### 5. `.github/workflows/docker-build-validation.yml`

- Export `r2_endpoint` and `r2_bucket` from the config loader step.
- Pass `R2_ENDPOINT` as a build-arg instead of a secret.
- Pass `R2_BUCKET` as a build-arg.
- Remove `r2_endpoint` from the `secrets:` block.

### 6. `tests/scripts/test_image_config.py`

- Tests using empty/minimal YAML fixtures must now provide all fields
  (no defaults to fall back on). Create a helper that generates a complete
  YAML string, used across all tests.
- Add assertions for `r2_endpoint` and `r2_bucket` in the drift-detection
  test (`test_static_field_values_match_dev_snapshot_yaml`).
- Add validation tests for the new fields (empty string, missing key).

## Verification

- `make test` — all image_config tests pass with updated fixtures.
- `make format` — pre-commit hooks pass.
- Trigger GHA workflow — config loader step exports new fields, build
  passes `R2_ENDPOINT` as build-arg.
