#!/usr/bin/env bash
# =============================================================================
# Docker entrypoint for synth-permutations images.
#
# Dispatches based on the MODE environment variable (REQUIRED, no default):
#
#   MODE=generate-shards
#     Generate split-agnostic HDF5 shards and upload to R2.
#     Designed for massively parallel cloud execution (e.g. RunPod).
#     Each container generates NUM_SHARDS shards with a unique instance ID
#     derived from RUNPOD_POD_ID (or auto-generated UUID).
#
#     Key env vars:
#       NUM_SHARDS          Number of shards to generate (REQUIRED).
#       SHARD_SIZE          Samples per shard (REQUIRED, Makefile default: 10000).
#       PARAM_SPEC          Param spec to use (REQUIRED, Makefile default: surge_simple).
#       OUTPUT_DIR          Local output directory (REQUIRED, Makefile default: data/surge_simple).
#       R2_PREFIX           R2 path prefix (e.g. runs/20260310-143022-a3f2b1).
#                           REQUIRED when R2_BUCKET is set.
#       R2_BUCKET           R2 bucket name. Read from env var baked at build time.
#       INSTANCE_ID         Worker ID baked into shard filenames.
#                           Auto-derived from RUNPOD_POD_ID if not set.
#       PARALLEL            "1" (default) to run shards concurrently, "0" for sequential.
#       MAX_WORKERS         Cap on concurrent workers (default: auto from CPU count).
#       DRY_RUN_UPLOAD      If "1", passes --dry-run to rclone.
#       IDLE_AFTER          If "1", drop to bash after completion.
#
#   MODE=finalize-shards
#     Download shards from R2, reshard into train/val/test virtual datasets,
#     compute normalization stats, and upload results back to R2.
#     This is the aggregation step after distributed shard generation.
#
#     Key env vars:
#       R2_PREFIX           R2 path prefix where shards live (REQUIRED).
#                           Shards are expected at {R2_PREFIX}/shards/.
#       R2_BUCKET           R2 bucket name (REQUIRED).
#       OUTPUT_DIR          Local output directory (REQUIRED, Makefile default: data/surge_simple).
#       VAL_SHARDS          Number of shards for validation (REQUIRED, Makefile default: 1).
#       TEST_SHARDS         Number of shards for test (REQUIRED, Makefile default: 1).
#       SKIP_UPLOAD         If "1", skip uploading results back to R2.
#       DRY_RUN_UPLOAD      If "1", passes --dry-run to rclone upload.
#       IDLE_AFTER          If "1", drop to bash after completion.
#
#   MODE=train
#     Download a dataset from R2, then run training.
#     Set IDLE_AFTER=1 to drop to a bash shell for inspection instead.
#
#     Key env vars:
#       R2_DATASET_PATH     R2 path to dataset dir (e.g. runs/surge_simple/abc1234).
#                           REQUIRED.
#       PARAM_SPEC          Param spec used when the dataset was generated (REQUIRED,
#                           Makefile default: surge_simple). Valid: surge_simple, surge_xt
#       OUTPUT_DIR          Local path to download dataset into (REQUIRED,
#                           Makefile default: data/surge_simple).
#       TRAIN_ARGS          Args passed to src/train.py (REQUIRED,
#                           Makefile default: experiment=surge/flow_simple).
#       R2_BUCKET           R2 bucket name. Read from env var baked at build time.
#       IDLE_AFTER          If "1", drop to bash after completion (default: 0).
#
#   MODE=shell
#     Drop directly to bash (for debugging or manual workflows).
#
# NOTE: Always pass --init (or install tini) when running this container
#       manually, e.g.:
#         docker run --rm -it --init --gpus all tinaudio/perm:dev
#       This ensures signals are forwarded correctly and zombie processes
#       are reaped. The Makefile docker-run-* targets include --init.
# =============================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/home/build/synth-permutations}"
cd "$APP_DIR"

# ---------------------------------------------------------------------------
# Optional: pull latest code before running the entrypoint.
# ---------------------------------------------------------------------------
if [ "${PULL_LATEST:-0}" = "1" ]; then
  if [ -z "${GIT_PAT:-}" ]; then
    echo "ERROR: PULL_LATEST=1 requires GIT_PAT to be set." >&2
    exit 1
  fi
  echo "[hotpatch] Pulling latest from origin..."
  git -C "$APP_DIR" pull origin "${SYNTH_PERMUTATIONS_GIT_REF:-experiment}"
  pip install -e . --quiet
  export PULL_LATEST=0
  exec "$0" "$@"
fi

MODE="${MODE:?ERROR: MODE is required. Valid: generate-shards, finalize-shards, train, shell}"
R2_BUCKET="${R2_BUCKET:-}"
IDLE_AFTER="${IDLE_AFTER:-0}"

# ---------------------------------------------------------------------------
# Detect code provenance for metadata.json traceability.
#
# git_ref_source:
#   "baked" — source was downloaded as a tarball at image build time
#             (prod or dev-snapshot image via tarball). No .git present.
#             SYNTH_PERMUTATIONS_GIT_REF is authoritative.
#   "local" — source was mounted or git-cloned at runtime (dev-live image,
#             or dev-snapshot via git clone). .git is present; git_sha
#             reflects the actual runtime state of the working tree.
# ---------------------------------------------------------------------------
GIT_SHA="${SYNTH_PERMUTATIONS_GIT_REF:-unknown}"
GIT_REF_SOURCE="unknown"

if git -C "$APP_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  # .git present — dev-live (mounted repo) or dev-snapshot (full git clone)
  GIT_REF_SOURCE="local"
  RUNTIME_SHA="$(git -C "$APP_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
  GIT_SHA="$RUNTIME_SHA"
else
  # No .git — source baked as tarball (prod or dev-snapshot via tarball)
  GIT_REF_SOURCE="baked"
fi

# ---------------------------------------------------------------------------
# Validate PARAM_SPEC and derive the matching Hydra data config name.
# Call as: _set_param_spec_vars <param_spec>
# Sets: DATA_CONFIG
# ---------------------------------------------------------------------------
_set_param_spec_vars() {
  local spec="$1"
  case "$spec" in
    surge_simple) DATA_CONFIG="surge_simple" ;;
    surge_xt)     DATA_CONFIG="surge" ;;
    *)
      echo "ERROR: Unknown PARAM_SPEC='$spec'. Valid values: surge_simple, surge_xt." >&2
      exit 1
      ;;
  esac
}

case "$MODE" in
  # ---------------------------------------------------------------------------
  generate-shards)
    NUM_SHARDS="${NUM_SHARDS:?ERROR: NUM_SHARDS is required for MODE=generate-shards}"
    SHARD_SIZE="${SHARD_SIZE:?ERROR: SHARD_SIZE is required for MODE=generate-shards}"
    PARAM_SPEC="${PARAM_SPEC:?ERROR: PARAM_SPEC is required for MODE=generate-shards}"
    OUTPUT_DIR="${OUTPUT_DIR:?ERROR: OUTPUT_DIR is required for MODE=generate-shards}"
    PLUGIN_PATH="${PLUGIN_PATH:-/usr/lib/vst3/Surge XT.vst3}"
    DRY_RUN_UPLOAD="${DRY_RUN_UPLOAD:-0}"

    # Auto-derive instance ID: prefer RUNPOD_POD_ID, fall back to auto-UUID
    INSTANCE_ID="${INSTANCE_ID:-${RUNPOD_POD_ID:-}}"
    INSTANCE_ID_FLAG=""
    if [ -n "$INSTANCE_ID" ]; then
      INSTANCE_ID_FLAG="--instance-id-prefix $INSTANCE_ID"
    fi

    PARALLEL_FLAG="--parallel"
    if [ "${PARALLEL:-1}" = "0" ]; then
      PARALLEL_FLAG=""
    fi

    MAX_WORKERS_FLAG=""
    if [ -n "${MAX_WORKERS:-}" ]; then
      MAX_WORKERS_FLAG="--max-workers $MAX_WORKERS"
    fi

    R2_ARGS=""
    if [ -n "$R2_BUCKET" ]; then
      R2_PREFIX="${R2_PREFIX:?ERROR: R2_PREFIX is required when R2_BUCKET is set}"
      R2_ARGS="--r2-bucket $R2_BUCKET --r2-prefix $R2_PREFIX"
    else
      R2_ARGS="--local"
    fi

    DRY_RUN_FLAG=""
    if [ "$DRY_RUN_UPLOAD" = "1" ]; then
      DRY_RUN_FLAG="--dry-run-upload"
    fi

    echo "=== synth-permutations: shard generation ==="
    echo "  num_shards    : $NUM_SHARDS"
    echo "  shard_size    : $SHARD_SIZE"
    echo "  param_spec    : $PARAM_SPEC"
    echo "  output_dir    : $OUTPUT_DIR"
    echo "  instance_id   : ${INSTANCE_ID:-<auto>}"
    echo "  parallel      : ${PARALLEL:-1}"
    echo "  max_workers   : ${MAX_WORKERS:-<auto>}"
    echo "  r2_prefix     : ${R2_PREFIX:-<not set>}"
    echo "  r2_bucket     : ${R2_BUCKET:-<not set — local only>}"
    echo "  plugin_path   : $PLUGIN_PATH"
    echo "  dry_run       : $DRY_RUN_UPLOAD"
    echo "  git_sha       : $GIT_SHA"
    echo "  git_ref_source: $GIT_REF_SOURCE"
    echo ""

    chmod +x scripts/run-linux-vst-headless.sh

    # shellcheck disable=SC2086
    python scripts/generate_shards.py \
      --num-shards "$NUM_SHARDS" \
      --shard-size "$SHARD_SIZE" \
      --output-dir "$OUTPUT_DIR" \
      --param-spec "$PARAM_SPEC" \
      --plugin-path "$PLUGIN_PATH" \
      --headless \
      $PARALLEL_FLAG \
      $MAX_WORKERS_FLAG \
      $INSTANCE_ID_FLAG \
      $R2_ARGS \
      $DRY_RUN_FLAG

    echo ""
    echo "=== Shard generation complete. ==="
    if [ "$IDLE_AFTER" = "1" ]; then
      echo "IDLE_AFTER=1: dropping to bash for inspection."
      exec bash
    fi
    ;;

  # ---------------------------------------------------------------------------
  finalize-shards)
    R2_PREFIX="${R2_PREFIX:?ERROR: R2_PREFIX is required for MODE=finalize-shards}"
    OUTPUT_DIR="${OUTPUT_DIR:?ERROR: OUTPUT_DIR is required for MODE=finalize-shards}"
    VAL_SHARDS="${VAL_SHARDS:?ERROR: VAL_SHARDS is required for MODE=finalize-shards}"
    TEST_SHARDS="${TEST_SHARDS:?ERROR: TEST_SHARDS is required for MODE=finalize-shards}"
    DRY_RUN_UPLOAD="${DRY_RUN_UPLOAD:-0}"
    SKIP_UPLOAD="${SKIP_UPLOAD:-0}"

    if [ -z "$R2_BUCKET" ]; then
      echo "ERROR: R2_BUCKET is not set. Cannot download/upload shards." >&2
      exit 1
    fi

    echo "=== synth-permutations: finalize shards ==="
    echo "  r2_prefix     : $R2_PREFIX"
    echo "  r2_bucket     : $R2_BUCKET"
    echo "  output_dir    : $OUTPUT_DIR"
    echo "  val_shards    : $VAL_SHARDS"
    echo "  test_shards   : $TEST_SHARDS"
    echo "  skip_upload   : $SKIP_UPLOAD"
    echo "  dry_run       : $DRY_RUN_UPLOAD"
    echo ""

    DRY_RUN_FLAG=""
    if [ "$DRY_RUN_UPLOAD" = "1" ]; then
      DRY_RUN_FLAG="--dry-run-upload"
    fi

    SKIP_UPLOAD_FLAG=""
    if [ "$SKIP_UPLOAD" = "1" ]; then
      SKIP_UPLOAD_FLAG="--skip-upload"
    fi

    # shellcheck disable=SC2086
    python scripts/finalize_shards.py \
      --r2-prefix "$R2_PREFIX" \
      --r2-bucket "$R2_BUCKET" \
      --output-dir "$OUTPUT_DIR" \
      --val-shards "$VAL_SHARDS" \
      --test-shards "$TEST_SHARDS" \
      $DRY_RUN_FLAG \
      $SKIP_UPLOAD_FLAG

    echo ""
    echo "=== Finalize shards complete. ==="
    if [ "$IDLE_AFTER" = "1" ]; then
      echo "IDLE_AFTER=1: dropping to bash for inspection."
      exec bash
    fi
    ;;

  # ---------------------------------------------------------------------------
  train)
    PARAM_SPEC="${PARAM_SPEC:?ERROR: PARAM_SPEC is required for MODE=train}"
    _set_param_spec_vars "$PARAM_SPEC"

    R2_DATASET_PATH="${R2_DATASET_PATH:?ERROR: R2_DATASET_PATH is required for MODE=train}"
    OUTPUT_DIR="${OUTPUT_DIR:?ERROR: OUTPUT_DIR is required for MODE=train}"
    TRAIN_ARGS="${TRAIN_ARGS:?ERROR: TRAIN_ARGS is required for MODE=train}"

    if [ -z "$R2_BUCKET" ]; then
      echo "ERROR: R2_BUCKET is not set. Cannot download dataset from R2." >&2
      exit 1
    fi

    echo "=== synth-permutations: download dataset + train ==="
    echo "  param_spec      : $PARAM_SPEC"
    echo "  data_config     : $DATA_CONFIG"
    echo "  r2_dataset_path : $R2_DATASET_PATH"
    echo "  output_dir      : $OUTPUT_DIR"
    echo "  train_args      : $TRAIN_ARGS"
    echo ""

    echo "[download] rclone copy r2:${R2_BUCKET}/${R2_DATASET_PATH} ${OUTPUT_DIR}"
    mkdir -p "$OUTPUT_DIR"
    rclone copy "r2:${R2_BUCKET}/${R2_DATASET_PATH}" "$OUTPUT_DIR" --progress --checksum --transfers 200 --checkers 200

    echo ""
    echo "[train] python src/train.py data=${DATA_CONFIG} data.dataset_root=${OUTPUT_DIR} ${TRAIN_ARGS}"
    # shellcheck disable=SC2086
    python src/train.py \
      "data=${DATA_CONFIG}" \
      "data.dataset_root=${OUTPUT_DIR}" \
      "data.predict_file=${OUTPUT_DIR}/val.h5" \
      $TRAIN_ARGS

    echo ""
    echo "=== Training complete. ==="
    if [ "$IDLE_AFTER" = "1" ]; then
      echo "IDLE_AFTER=1: dropping to bash for inspection."
      exec bash
    fi
    ;;

  # ---------------------------------------------------------------------------
  shell)
    echo "=== synth-permutations: shell mode ==="
    exec bash
    ;;

  # ---------------------------------------------------------------------------
  *)
    echo "ERROR: Unknown MODE='$MODE'. Valid values: generate-shards, finalize-shards, train, shell." >&2
    exit 1
    ;;
esac
