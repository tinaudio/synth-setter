"""Well-known filenames and paths for the data pipeline.

Canonical names from docs/design/data-pipeline.md §7.1 (storage layout).
"""

# Frozen input specification — written once at @hydra.main as a DatasetSpec
# JSON serialization, never modified after launch.
INPUT_SPEC_FILENAME = "input_spec.json"

# Canonical R2 URI scheme used throughout the pipeline; methods on
# ``R2Location`` build URIs with this prefix so every call site agrees on shape.
R2_URI_SCHEME = "r2://"

# Name of the rclone remote that targets Cloudflare R2 in worker/launcher
# environments. ``R2Location.rclone_prefix`` appends the trailing colon when
# building an rclone-syntax destination.
RCLONE_REMOTE = "r2"

# Per-launch R2 key prefix where ``skypilot_launch.upload_spec_to_r2`` puts the
# materialized spec, keyed by job name. The URI lives outside the run's
# ``r2.prefix`` because the launcher staging area is one-per-job, not one-per-run.
LAUNCHER_SPEC_R2_PREFIX = "skypilot-launcher-specs"
