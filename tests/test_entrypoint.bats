#!/usr/bin/env bats
# shellcheck disable=SC2030,SC2031
# SC2030/SC2031: "Modification is local to subshell" — intentional in BATS;
# each @test runs in its own subshell by design.
# =============================================================================
# BATS tests for scripts/docker_entrypoint.sh
#
# Tests the entrypoint's env var validation and mode dispatch WITHOUT Docker.
# A fake `python` and `git` on PATH prevent real subprocess execution.
#
# Run:  bats tests/test_entrypoint.bats
# =============================================================================

ENTRYPOINT="$BATS_TEST_DIRNAME/../scripts/docker_entrypoint.sh"

# Helper: run entrypoint with stderr merged into stdout so BATS captures
# error messages from bash's ${VAR:?} and explicit >&2 writes.
# Uses /bin/bash explicitly twice: the outer shell avoids the stub on PATH,
# and the inner shell runs the entrypoint with the shebang bypassed.
# Inside the entrypoint, `exec bash` still finds the stub (via PATH) and exits cleanly.
run_entrypoint() {
  run /bin/bash -c 'exec /bin/bash "$1" 2>&1' _ "$ENTRYPOINT"
}

setup() {
  # Create a temp bin dir with stubs for python, git, pip, chmod, rclone
  export FAKE_BIN="$BATS_TEST_TMPDIR/bin"
  mkdir -p "$FAKE_BIN"

  # Stub python: exits 0, prints args (so entrypoint doesn't actually run scripts)
  cat > "$FAKE_BIN/python" << 'STUB'
#!/bin/bash
echo "[stub] python $*"
exit 0
STUB
  chmod +x "$FAKE_BIN/python"

  # Stub git: pretend no .git directory (baked mode)
  cat > "$FAKE_BIN/git" << 'STUB'
#!/bin/bash
# rev-parse --git-dir should fail (no .git)
if [[ "$*" == *"rev-parse --git-dir"* ]]; then
  exit 1
fi
echo "[stub] git $*"
exit 0
STUB
  chmod +x "$FAKE_BIN/git"

  # Stub pip
  cat > "$FAKE_BIN/pip" << 'STUB'
#!/bin/bash
exit 0
STUB
  chmod +x "$FAKE_BIN/pip"

  # Stub rclone
  cat > "$FAKE_BIN/rclone" << 'STUB'
#!/bin/bash
echo "[stub] rclone $*"
exit 0
STUB
  chmod +x "$FAKE_BIN/rclone"

  # Stub chmod (entrypoint runs chmod +x on headless wrapper)
  cat > "$FAKE_BIN/chmod" << 'STUB'
#!/bin/bash
exit 0
STUB
  chmod +x "$FAKE_BIN/chmod"

  # Stub bash for exec bash in shell/idle modes — just exit cleanly.
  # Uses /bin/bash shebang (not /usr/bin/env bash) to avoid infinite
  # shebang loop when this stub is first on PATH.
  cat > "$FAKE_BIN/bash" << 'STUB'
#!/bin/bash
exit 0
STUB
  chmod +x "$FAKE_BIN/bash"

  # Put stubs first on PATH
  export PATH="$FAKE_BIN:$PATH"

  # Set APP_DIR to a temp dir so cd doesn't fail
  export APP_DIR="$BATS_TEST_TMPDIR/app"
  mkdir -p "$APP_DIR"

  # Sanitize env: prevent host variables (e.g. from .env) from leaking into tests.
  unset R2_BUCKET R2_PREFIX R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY
  unset NUM_SHARDS INSTANCE_ID RUNPOD_POD_ID
  unset MODE PULL_LATEST GIT_PAT IDLE_AFTER SKIP_UPLOAD DRY_RUN_UPLOAD

  # Defaults for required config vars (entrypoint no longer has defaults;
  # Makefile is the single source of truth). Tests override/unset as needed.
  export SHARD_SIZE=100
  export PARAM_SPEC="surge_simple"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export VAL_SHARDS=1
  export TEST_SHARDS=1
  export TRAIN_ARGS="experiment=surge/flow_simple"
}


# ---------------------------------------------------------------------------
# MODE validation
# ---------------------------------------------------------------------------

@test "missing MODE exits with error" {
  unset MODE
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"MODE"* ]]
  [[ "$output" == *"required"* ]]
}

@test "invalid MODE exits with error" {
  export MODE="bogus"
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"Unknown MODE"* ]]
  [[ "$output" == *"bogus"* ]]
}

@test "MODE=shell exits cleanly" {
  export MODE="shell"
  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"shell mode"* ]]
}


# ---------------------------------------------------------------------------
# MODE=generate-shards validation
# ---------------------------------------------------------------------------

@test "generate-shards: missing NUM_SHARDS exits with error" {
  export MODE="generate-shards"
  unset NUM_SHARDS
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"NUM_SHARDS"* ]]
}

@test "generate-shards: missing SHARD_SIZE exits with error" {
  export MODE="generate-shards"
  export NUM_SHARDS=1
  unset SHARD_SIZE
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"SHARD_SIZE"* ]]
}

@test "generate-shards: missing PARAM_SPEC exits with error" {
  export MODE="generate-shards"
  export NUM_SHARDS=1
  unset PARAM_SPEC
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"PARAM_SPEC"* ]]
}

@test "generate-shards: missing OUTPUT_DIR exits with error" {
  export MODE="generate-shards"
  export NUM_SHARDS=1
  unset OUTPUT_DIR
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"OUTPUT_DIR"* ]]
}

@test "generate-shards: with required vars succeeds" {
  export MODE="generate-shards"
  export NUM_SHARDS=2
  export PARAM_SPEC="surge_simple"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"shard generation"* ]]
}

@test "generate-shards: missing R2_BUCKET uses --local" {
  export MODE="generate-shards"
  export NUM_SHARDS=1
  export PARAM_SPEC="surge_simple"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  unset R2_BUCKET
  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"--local"* ]]
}

@test "generate-shards: R2_BUCKET set requires R2_PREFIX" {
  export MODE="generate-shards"
  export NUM_SHARDS=1
  export PARAM_SPEC="surge_simple"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export R2_BUCKET="my-bucket"
  unset R2_PREFIX
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"R2_PREFIX"* ]]
}


# ---------------------------------------------------------------------------
# MODE=finalize-shards validation
# ---------------------------------------------------------------------------

@test "finalize-shards: missing R2_PREFIX exits with error" {
  export MODE="finalize-shards"
  unset R2_PREFIX
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"R2_PREFIX"* ]]
}

@test "finalize-shards: missing OUTPUT_DIR exits with error" {
  export MODE="finalize-shards"
  export R2_PREFIX="runs/batch42"
  unset OUTPUT_DIR
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"OUTPUT_DIR"* ]]
}

@test "finalize-shards: missing R2_BUCKET exits with error" {
  export MODE="finalize-shards"
  export R2_PREFIX="runs/batch42"
  unset R2_BUCKET
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"R2_BUCKET"* ]]
}


# ---------------------------------------------------------------------------
# MODE=train validation
# ---------------------------------------------------------------------------

@test "train: missing PARAM_SPEC exits with error" {
  export MODE="train"
  unset PARAM_SPEC
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"PARAM_SPEC"* ]]
}

@test "train: missing R2_PREFIX exits with error" {
  export MODE="train"
  unset R2_PREFIX
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"R2_PREFIX"* ]]
}

@test "train: missing OUTPUT_DIR exits with error" {
  export MODE="train"
  export R2_PREFIX="runs/surge_simple/abc123"
  unset OUTPUT_DIR
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"OUTPUT_DIR"* ]]
}

@test "train: missing TRAIN_ARGS exits with error" {
  export MODE="train"
  export R2_PREFIX="runs/surge_simple/abc123"
  unset TRAIN_ARGS
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"TRAIN_ARGS"* ]]
}

@test "train: missing R2_BUCKET exits with error" {
  export MODE="train"
  export R2_PREFIX="runs/surge_simple/abc123"
  export R2_BUCKET=""
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"R2_BUCKET"* ]]
}

@test "train: with required vars succeeds" {
  export MODE="train"
  export PARAM_SPEC="surge_simple"
  export R2_PREFIX="runs/surge_simple/abc123"
  export R2_BUCKET="my-bucket"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export TRAIN_ARGS="experiment=surge/flow_simple"
  mkdir -p "$BATS_TEST_TMPDIR/data"
  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"download dataset"* ]]
}

@test "train: passes hydra.run.dir override to train.py" {
  export MODE="train"
  export PARAM_SPEC="surge_simple"
  export R2_PREFIX="runs/surge_simple/abc123"
  export R2_BUCKET="my-bucket"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export TRAIN_ARGS="experiment=surge/flow_simple"
  mkdir -p "$BATS_TEST_TMPDIR/data"
  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"hydra.run.dir=logs/train/runs/surge_simple/abc123"* ]]
}

@test "train: uploads training output to R2 with wandb run ID" {
  export MODE="train"
  export PARAM_SPEC="surge_simple"
  export R2_PREFIX="runs/surge_simple/abc123"
  export R2_BUCKET="my-bucket"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export TRAIN_ARGS="experiment=surge/flow_simple"
  mkdir -p "$BATS_TEST_TMPDIR/data"

  # Override python stub to write a fake wandb_run_id file
  cat > "$FAKE_BIN/python" << 'STUB'
#!/bin/bash
echo "[stub] python $*"
if [[ "$*" == *"src/train.py"* ]]; then
  for arg in "$@"; do
    if [[ "$arg" == hydra.run.dir=* ]]; then
      OUTDIR="${arg#hydra.run.dir=}"
      mkdir -p "$OUTDIR"
      echo "fakeid99" > "$OUTDIR/wandb_run_id"
    fi
  done
fi
exit 0
STUB
  chmod +x "$FAKE_BIN/python"

  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"rclone copy logs/train/runs/surge_simple/abc123 r2:my-bucket/runs/surge_simple/abc123/training/fakeid99/"* ]]
}

@test "train: uploads without run ID subdirectory when wandb_run_id file missing" {
  export MODE="train"
  export PARAM_SPEC="surge_simple"
  export R2_PREFIX="runs/surge_simple/abc123"
  export R2_BUCKET="my-bucket"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export TRAIN_ARGS="experiment=surge/flow_simple"
  mkdir -p "$BATS_TEST_TMPDIR/data"
  run_entrypoint
  [ "$status" -eq 0 ]
  # Upload happens to training/ without a run ID subdirectory
  [[ "$output" == *"rclone copy logs/train/runs/surge_simple/abc123 r2:my-bucket/runs/surge_simple/abc123/training/"* ]]
}

@test "train: SKIP_UPLOAD=1 skips checkpoint upload" {
  export MODE="train"
  export PARAM_SPEC="surge_simple"
  export R2_PREFIX="runs/surge_simple/abc123"
  export R2_BUCKET="my-bucket"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export TRAIN_ARGS="experiment=surge/flow_simple"
  export SKIP_UPLOAD=1
  mkdir -p "$BATS_TEST_TMPDIR/data"
  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"skipping checkpoint upload"* ]]
  # No rclone upload call (only the download call should appear)
  [[ "$output" != *"rclone copy logs/train/"* ]]
}

@test "train: DRY_RUN_UPLOAD=1 passes --dry-run to rclone upload" {
  export MODE="train"
  export PARAM_SPEC="surge_simple"
  export R2_PREFIX="runs/surge_simple/abc123"
  export R2_BUCKET="my-bucket"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export TRAIN_ARGS="experiment=surge/flow_simple"
  export DRY_RUN_UPLOAD=1
  mkdir -p "$BATS_TEST_TMPDIR/data"
  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"--dry-run"* ]]
}

@test "train: banner shows upload config" {
  export MODE="train"
  export PARAM_SPEC="surge_simple"
  export R2_PREFIX="runs/surge_simple/abc123"
  export R2_BUCKET="my-bucket"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export TRAIN_ARGS="experiment=surge/flow_simple"
  mkdir -p "$BATS_TEST_TMPDIR/data"
  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"train_output_dir"* ]]
  [[ "$output" == *"skip_upload"* ]]
  [[ "$output" == *"dry_run"* ]]
}

@test "train: banner shows wandb_auth when netrc has wandb entry" {
  export MODE="train"
  export PARAM_SPEC="surge_simple"
  export R2_PREFIX="runs/surge_simple/abc123"
  export R2_BUCKET="my-bucket"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export TRAIN_ARGS="experiment=surge/flow_simple"
  mkdir -p "$BATS_TEST_TMPDIR/data"

  # Create a fake ~/.netrc with wandb entry
  export HOME="$BATS_TEST_TMPDIR/home"
  mkdir -p "$HOME"
  printf 'machine api.wandb.ai\n  login user\n  password testkey123\n' > "$HOME/.netrc"
  chmod 600 "$HOME/.netrc"

  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"wandb_auth"* ]]
  [[ "$output" == *"netrc"* ]]
}

@test "train: banner shows wandb_auth not configured when netrc missing" {
  export MODE="train"
  export PARAM_SPEC="surge_simple"
  export R2_PREFIX="runs/surge_simple/abc123"
  export R2_BUCKET="my-bucket"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  export TRAIN_ARGS="experiment=surge/flow_simple"
  mkdir -p "$BATS_TEST_TMPDIR/data"

  # Ensure no ~/.netrc exists
  export HOME="$BATS_TEST_TMPDIR/home"
  mkdir -p "$HOME"
  rm -f "$HOME/.netrc"

  run_entrypoint
  [ "$status" -eq 0 ]
  [[ "$output" == *"wandb_auth"* ]]
  [[ "$output" == *"not configured"* ]]
}


# ---------------------------------------------------------------------------
# PULL_LATEST validation
# ---------------------------------------------------------------------------

@test "PULL_LATEST=main without GIT_PAT exits with error" {
  export PULL_LATEST=main
  export MODE="shell"
  unset GIT_PAT
  run_entrypoint
  [ "$status" -ne 0 ]
  [[ "$output" == *"GIT_PAT"* ]]
}


# ---------------------------------------------------------------------------
# IDLE_AFTER default
# ---------------------------------------------------------------------------

@test "IDLE_AFTER defaults to 0 (no idle)" {
  export MODE="generate-shards"
  export NUM_SHARDS=1
  export PARAM_SPEC="surge_simple"
  export OUTPUT_DIR="$BATS_TEST_TMPDIR/data"
  unset IDLE_AFTER
  run_entrypoint
  [ "$status" -eq 0 ]
  # Should NOT contain "IDLE_AFTER=1" message
  [[ "$output" != *"dropping to bash"* ]]
}
