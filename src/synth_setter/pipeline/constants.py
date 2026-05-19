"""Well-known filenames and paths for the data pipeline.

Canonical names from docs/design/data-pipeline.md §7.1 (storage layout).
"""

# Frozen input specification — written once at @hydra.main as a DatasetSpec
# JSON serialization, never modified after launch.
INPUT_SPEC_FILENAME = "input_spec.json"

# Cloudflare R2 remote name used by rclone (``rclone copy <src> r2:bucket/key``).
# Resolution is the caller's responsibility — the standard ``RCLONE_CONFIG_R2_*``
# env vars must be set when any rclone subprocess runs.
RCLONE_REMOTE = "r2"

# Canonical R2 URI scheme. Worker, launcher, and CI validation agree on
# ``r2://bucket/key``; ``r2_io.to_rclone_path`` translates to rclone's
# ``r2:bucket/key`` form at the subprocess boundary.
R2_URI_SCHEME = f"{RCLONE_REMOTE}://"

# Per-launch R2 key prefix where ``skypilot_launch.upload_spec_to_r2`` writes
# a per-job transport copy of the materialized spec as ``<job_name>.json``
# (workaround for #749 file_mounts — the object key is the job name, not
# ``input_spec.json``). Retired in PR-4 (#1115); see also the launcher's own
# ``upload_spec_to_r2``.
LAUNCHER_SPEC_R2_PREFIX = "skypilot-launcher-specs"
