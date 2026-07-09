"""Well-known filenames and paths for the data pipeline.

Canonical names from docs/design/data-pipeline.md §7.1 (storage layout).
"""

# Frozen input specification — written once at @hydra.main as a DatasetSpec
# JSON serialization, never modified after launch.
INPUT_SPEC_FILENAME = "input_spec.json"

# Normalization statistics (mel-spec mean / std) written by finalize.
STATS_NPZ_FILENAME = "stats.npz"

# Zero-byte trust-anchor marker; presence under ``r2.prefix`` is the canonical
# "this run is ready to consume" signal. Written strictly last by finalize.
DATASET_COMPLETE_FILENAME = "dataset.complete"

# Dataset card (provenance record of the selected attempts) written by finalize.
DATASET_CARD_FILENAME = "dataset.json"

# Per-attempt staging suffixes under ``metadata/workers/shards/shard-{id}/``.
# Filenames are ``{worker_id}-{attempt_uuid}{suffix}``; a Lance attempt is
# staged-valid only when all of sidecar + stats + ``.valid`` exist (#1776).
LANCE_FRAGMENT_SIDECAR_SUFFIX = ".fragment.json"
LANCE_SHARD_STATS_SUFFIX = ".shard-stats.npz"

# Welford state arrays a `.shard-stats.npz` sidecar must carry; finalize
# reduces the winners' states into dataset-level stats.npz (#1776).
LANCE_SHARD_STATS_KEYS = ("count", "mean", "m2")
ATTEMPT_VALID_SUFFIX = ".valid"
ATTEMPT_RENDERING_SUFFIX = ".rendering"

# Env-var name reserved for the worker to locate the materialized DatasetSpec.
# Today's consumers: the launcher (``dispatch_via_skypilot``) injects the value
# into each rank's ``task.update_envs``; the CI helper (``pipeline.ci.spec_uri``)
# reconstructs the URI for workflow exports. No worker code reads it yet — the
# dispatched cmd rebuilds the spec from Hydra overrides; #1115 / #1116 will
# swap in a worker that reads this env var directly.
WORKER_SPEC_URI_ENV = "WORKER_SPEC_URI"

# Cloudflare R2 remote name used by rclone (``rclone copy <src> r2:bucket/key``).
# Resolution is the caller's responsibility — the standard ``RCLONE_CONFIG_R2_*``
# env vars must be set when any rclone subprocess runs.
RCLONE_REMOTE = "r2"

# Canonical R2 URI scheme. Worker, launcher, and CI validation agree on
# ``r2://bucket/key``; ``r2_io.to_rclone_path`` translates to rclone's
# ``r2:bucket/key`` form at the subprocess boundary.
R2_URI_SCHEME = f"{RCLONE_REMOTE}://"

# RFC 8089 local-file URI scheme — parsed by ``pipeline.file_uri``.
FILE_URI_SCHEME = "file://"
