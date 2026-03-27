# rclone Reference

> **Last verified:** 2026-03-26

rclone commands for Cloudflare R2 across all synth-setter workflows.

______________________________________________________________________

## 1. Setup

### Local

Before running high-concurrency transfers (200+), ensure your shell allows
enough open file descriptors:

```bash
# Increase descriptor limit for the current session
ulimit -n 65535

# 1. Source credentials
set -a && source .env && set +a

# 2. Create the "r2" remote (idempotent — overwrites if exists)
rclone config create r2 s3 \
  provider Cloudflare \
  access_key_id "$R2_ACCESS_KEY_ID" \
  secret_access_key "$R2_SECRET_ACCESS_KEY" \
  endpoint "$R2_ENDPOINT" \
  no_check_bucket true

# 3. Verify
rclone lsd r2:$R2_BUCKET
```

Script: `scripts/setup-rclone.sh` wraps the above with env-var validation and
error messages.

### Docker

rclone is installed via apt and configured at build time. The Dockerfile bakes
`/root/.config/rclone/rclone.conf` using BuildKit secrets so credentials never
appear in layer history:

```dockerfile
# docker/ubuntu22_04/Dockerfile (r2-config-base stage)
RUN --mount=type=secret,id=r2_access_key_id \
    --mount=type=secret,id=r2_secret_access_key \
    --mount=type=secret,id=r2_endpoint \
    printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\n' \
        "$(cat /run/secrets/r2_access_key_id)" \
        "$(cat /run/secrets/r2_secret_access_key)" \
        "$(cat /run/secrets/r2_endpoint)" \
        > /root/.config/rclone/rclone.conf
```

Secrets are passed via the Makefile:

```makefile
DOCKER_SECRETS = \
    --secret id=r2_access_key_id,env=R2_ACCESS_KEY_ID \
    --secret id=r2_secret_access_key,env=R2_SECRET_ACCESS_KEY \
    --secret id=r2_endpoint,env=R2_ENDPOINT \
    --build-arg R2_BUCKET=$(R2_BUCKET)
```

> **Security:** R2 credentials are injected at build time via BuildKit secret
> mounts and do not persist in image layers. Still, push ONLY to private
> registries and rotate R2 tokens after each build campaign.

### Required environment variables

| Variable               | Purpose                           | Format                                                                     |
| ---------------------- | --------------------------------- | -------------------------------------------------------------------------- |
| `R2_ACCESS_KEY_ID`     | Cloudflare R2 API token key       | Alphanumeric string                                                        |
| `R2_SECRET_ACCESS_KEY` | Cloudflare R2 API token secret    | Alphanumeric string                                                        |
| `R2_ENDPOINT`          | R2 S3-compatible endpoint URL     | Full URL with scheme, e.g. `https://<account_id>.r2.cloudflarestorage.com` |
| `R2_BUCKET`            | Bucket name (build-arg in Docker) | Lowercase string, no scheme                                                |

______________________________________________________________________

## 2. Project Rules

1. **All rclone transfer operations MUST use `--checksum`** — verifies content
   hashes after transfer. (CLAUDE.md, storage-provenance-spec.md §4)
   Destructive ops (`purge`, `delete`) are exempt — they don't transfer data.

2. **Standard parallelism** for bulk transfers: `--transfers 200 --checkers 200`.
   Ad-hoc commands in Section 4 omit these for readability; add them for large
   transfers.

3. **Use `--fast-list`** for all recursive operations (`ls`, `copy`, `sync`,
   `size`) — batches S3 listing API calls and reduces latency. No downside for
   R2.

4. **Large .h5 files:** use `--s3-chunk-size 64M` to optimize multi-part upload
   throughput. Increase to `128M` if upload speeds stall on very large shards.

5. **`rclone copy` vs `rclone sync`** — prefer `copy` (additive) unless you
   explicitly want to delete files at the destination that don't exist at the
   source. `sync` is used only for eval artifact upload (where re-running an
   eval should fully replace the previous results at the same 6-segment path).

6. **Retries:** rclone defaults to 3 retries and 10 low-level retries. We rely
   on these defaults. Do not reduce them — with 200 concurrent transfers,
   transient failures are expected.

7. **Progress output:** use `--progress` for interactive sessions. Use
   `--stats-one-line --stats 60s` in Docker or any non-TTY context for cleaner
   logs.

______________________________________________________________________

## 3. Codebase Usage

### Data generation — upload shards to R2

**Source:** `scripts/generate_shards.py` (calls `RcloneUploader.upload()`)

```bash
rclone copy <local_dir> r2:<bucket>/<remote_path> \
  --progress --checksum --fast-list \
  --transfers 200 --checkers 200 --s3-chunk-size 64M [--dry-run]
```

`--dry-run` controlled by `DRY_RUN_UPLOAD` env var / `--dry-run-upload` CLI flag.

### Finalize shards — download from R2

**Source:** `scripts/finalize_shards.py`

```bash
rclone copy r2:<bucket>/<remote_path> <local_dir> \
  --progress --checksum --fast-list --transfers 200 --checkers 200
```

Downloads staged shards for resharding into train/val/test splits. After
resharding, uploads the split virtual datasets (`train.h5`, `val.h5`,
`test.h5`) back to the same run directory via `RcloneUploader`.

### Docker entrypoint — train mode download

**Source:** `scripts/docker_entrypoint.sh` (MODE=train)

```bash
rclone copy "r2:${R2_BUCKET}/${R2_PREFIX}" "$OUTPUT_DIR" \
  --stats-one-line --stats 60s --checksum --fast-list \
  --transfers 200 --checkers 200
```

### Docker entrypoint — train mode upload

**Source:** `scripts/docker_entrypoint.sh` (MODE=train, after training)

```bash
rclone copy "$TRAIN_OUTPUT_DIR" "r2:${R2_BUCKET}/${R2_PREFIX}/${UPLOAD_SUFFIX}" \
  --stats-one-line --stats 60s --checksum --fast-list \
  --transfers 200 --checkers 200 --s3-chunk-size 64M \
  [--dry-run]
```

Skipped entirely when `SKIP_UPLOAD=1`.

### Shard reporting

**Source:** `scripts/r2_shard_report.py`

```bash
rclone ls <prefix>
```

Usage:

```bash
python scripts/r2_shard_report.py r2:synth-data/data/<config_id>/<run_id>/shards/ \
  --size-threshold-gib 1.0
```

### Eval artifact upload

**Source:** design doc (`docs/design/eval-pipeline.md`)

```bash
# make upload-eval expands to:
rclone sync <local_eval_dir> \
  r2:synth-data/eval/<dataset_config>/<dataset_run>/<train_config>/<train_run>/<eval_config>/<eval_run>/ \
  --checksum --fast-list
```

Uses `sync` (not `copy`) because re-running an eval at the same 6-segment path
should fully replace the previous results.

### Tests

**Source:** `tests/scripts/test_r2_shard_report.py`

```bash
# Reachability check (skip tests if R2 unreachable)
rclone lsd r2:

# Upload test fixture via stdin
rclone rcat r2:<bucket>/test-prefix/<filename>

# Cleanup test prefix
rclone purge r2:<bucket>/test-prefix/
```

______________________________________________________________________

## 4. Ad-hoc CLI Commands

### Listing and browsing

```bash
# Tree view of bucket structure
rclone tree r2:<bucket>/<prefix>/ --fast-list

# List files with sizes (recursive)
rclone ls r2:<bucket>/<prefix>/ --fast-list

# List immediate directories only
rclone lsd r2:<bucket>/<prefix>/

# Fast flat listing (filenames only, no sizes)
rclone lsf r2:<bucket>/<prefix>/ --fast-list

# JSON listing (useful for scripting / existence checks)
rclone lsjson r2:<bucket>/<prefix>/ --fast-list
```

### Inspecting shard staging state

```bash
# See full attempt history for a shard
rclone ls r2:<bucket>/<run_id>/metadata/workers/shards/shard-000042/
#          0  pod-abc123-a1b2c3d4.rendering   # crashed (no .valid)
#   67108864  pod-def456-e5f6a7b8.h5          # shard data
#          0  pod-def456-e5f6a7b8.rendering   # attempt started
#          0  pod-def456-e5f6a7b8.valid       # committed
#          0  pod-def456-e5f6a7b8.promoted    # promoted by finalize
```

A shard is safe to use when it has all three markers: `.rendering`, `.valid`,
and `.promoted`. A `.rendering` without a `.valid` means the worker crashed
mid-write.

### Reading metadata files

```bash
# Inspect dataset config
rclone cat r2:<bucket>/<run_id>/metadata/config.yaml

# Inspect worker report (pipe to jq for formatting)
rclone cat r2:<bucket>/<run_id>/metadata/workers/attempts/<worker>-<attempt>/report.json | jq .

# Inspect dataset card
rclone cat r2:<bucket>/<run_id>/metadata/dataset.json | jq .
```

### Browsing eval results

```bash
# All evals for a dataset generation run
rclone ls r2:synth-data/eval/<dataset_config_id>/<dataset_run_id>/ --fast-list

# All evals for a specific training run
rclone ls r2:synth-data/eval/<dataset_config>/<dataset_run>/<train_config>/<train_run>/ --fast-list

# A specific eval run (fully qualified 6-segment path)
rclone ls r2:synth-data/eval/<dataset_config>/<dataset_run>/<train_config>/<train_run>/<eval_config>/<eval_run>/ --fast-list
```

### Manual transfers

```bash
# Download a prefix
rclone copy r2:<bucket>/<prefix> <local_dir> --checksum --fast-list --progress

# Upload a directory
rclone copy <local_dir> r2:<bucket>/<prefix> --checksum --fast-list --progress

# Upload large .h5 files
rclone copy <local_dir> r2:<bucket>/<prefix> \
  --checksum --fast-list --progress --s3-chunk-size 64M

# Sync (mirror — deletes extras at destination). Always dry-run first.
rclone sync <local_dir> r2:<bucket>/<prefix> --checksum --fast-list --dry-run
rclone sync <local_dir> r2:<bucket>/<prefix> --checksum --fast-list

# Download a single file
rclone copyto r2:<bucket>/<path>/file.h5 ./file.h5 --checksum
```

### Cleanup

```bash
# Delete an entire prefix and all contents
rclone purge r2:<bucket>/<prefix>/

# Delete a single file
rclone delete r2:<bucket>/<prefix>/file.h5
```

### Bucket size

```bash
rclone size r2:<bucket>/<prefix>/ --fast-list
```

______________________________________________________________________

## 5. R2 Bucket Layout

Full spec: `docs/design/storage-provenance-spec.md`

______________________________________________________________________

## 6. Flags Reference

| Flag                         | Purpose                                | Default | Required?        |
| ---------------------------- | -------------------------------------- | ------- | ---------------- |
| `--checksum`                 | Verify content hashes after transfer   | off     | **Yes (Rule 1)** |
| `--fast-list`                | Batch S3 listing calls, reduce latency | off     | **Yes (Rule 3)** |
| `--s3-chunk-size`            | Multi-part upload chunk size           | `5M`    | `64M` for .h5    |
| `--transfers N`              | Parallel file transfers                | `4`     | `200` for bulk   |
| `--checkers N`               | Parallel hash checkers                 | `8`     | `200` for bulk   |
| `--retries N`                | Retries on transfer failure            | `3`     | Use default      |
| `--low-level-retries N`      | Retries on individual S3 ops           | `10`    | Use default      |
| `--progress`                 | Show real-time transfer progress       | off     | Interactive only |
| `--stats-one-line --stats T` | Compact periodic stats                 | off     | Docker/non-TTY   |
| `--dry-run`                  | Simulate transfer without writing      | off     | Before `sync`    |
| `--no-check-bucket`          | Skip bucket existence check            | off     | Setup only       |

______________________________________________________________________

## 7. Troubleshooting

### Connectivity

```bash
# Check remote is configured
rclone listremotes          # should show "r2:"

# Test connectivity
rclone lsd r2:              # list buckets

# Verbose output for debugging transfer issues
rclone copy <src> <dst> --checksum -v

# Very verbose (shows individual file decisions)
rclone copy <src> <dst> --checksum -vv
```

### Too many open files

With `--transfers 200`, you need enough file descriptors. Check `ulimit -n` on
the host. Fix: `ulimit -n 65535`, or reduce `--transfers` to 64 as a fallback.

### Large .h5 uploads stalling

Increase chunk size: `--s3-chunk-size 128M`. Multi-part uploads with small
chunks create many concurrent requests that can bottleneck on the R2 endpoint.

### Interrupted transfers

`rclone copy` with `--checksum` is safe to re-run after interruption — it skips
files whose remote checksum already matches. Partially uploaded multi-part files
are abandoned on R2 (Cloudflare auto-cleans incomplete multi-part uploads after
24h). There is no manual resume; just re-run the same command.

### Checksum mismatch

If `--checksum` reports a mismatch after transfer, the file will be re-uploaded
on the next run. Persistent mismatches on the same file may indicate disk
corruption on the source — verify locally with `md5sum` before investigating R2.
