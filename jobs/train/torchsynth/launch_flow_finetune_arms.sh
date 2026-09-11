#!/bin/bash
# Submit the simulator-feedback ablation grid as managed RunPod jobs.
#
# The grid is arms x seeds. The null arm is the comparator: gradient-vs-null and
# learned-vs-null separate what the simulator signal buys from what the extra
# control capacity buys, so dropping it makes any improvement unattributable.
#
# Seeds are part of the grid, not an afterthought. A regression probe on this
# path could not separate a 2% param_mse shift from run-to-run variance at one
# run per arm, and the effect under test is smaller than that.
#
# Dispatch goes through the Hydra-native launcher (#2762): compute selects the
# pool, skypilot_launch.cmd carries the worker shell command. The step budget and
# validation cadence live in the arm's experiment YAML so `experiment=` alone
# reproduces a run (#2118, #2196).

set -euo pipefail

readonly COMPUTE_OPTION="runpod/torchsynth"
readonly WORKER_IMAGE_TAG="dev-snapshot"
readonly -a ARMS=(
  flow_finetune
  flow_finetune_learned
  flow_finetune_null
)
readonly -a SEEDS=(1 2 3)

usage() {
  cat >&2 <<'EOF'
Usage: launch_flow_finetune_arms.sh --base-checkpoint URI [options]

  --base-checkpoint URI   rclone URI of the pretrained flow every arm starts
                          from, e.g. r2:bucket/path/last.ckpt (required).
  --seeds "1 2 3"         Seeds to run (default: 1 2 3).
  --arms "a b"            Arms to run (default: all three).
  --execute               Actually submit. Without it, nothing is launched.

Dry run by default: submitting spends money, so it must be asked for.
EOF
}

main() {
  local base_checkpoint="" execute=false
  local -a arms=("${ARMS[@]}") seeds=("${SEEDS[@]}")
  local repo_root status=0

  while (($# > 0)); do
    case "$1" in
      --base-checkpoint) base_checkpoint="${2:-}"; shift 2 ;;
      --seeds) read -r -a seeds <<<"${2:-}"; shift 2 ;;
      --arms) read -r -a arms <<<"${2:-}"; shift 2 ;;
      --execute) execute=true; shift ;;
      -h|--help) usage; return 0 ;;
      *) echo "unknown argument: $1" >&2; usage; return 2 ;;
    esac
  done

  if [[ -z "${base_checkpoint}" ]]; then
    echo "--base-checkpoint is required: arms started from different bases are not comparable" >&2
    usage
    return 2
  fi

  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  cd "${repo_root}"

  # The shared server's endpoint var is malformed in the image; unset so the
  # local client is used.
  unset SKYPILOT_API_SERVER_ENDPOINT SKYPILOT_API_SERVER_KEY SKYPILOT_API_SERVER_TOKEN

  if [[ "${execute}" == true ]]; then
    # Exhausted balance surfaces as jobs stuck in STARTING with no visible cause.
    uv run python -c \
      "from synth_setter.pipeline.skypilot_launch import _check_runpod_balance; _check_runpod_balance()"
    echo "balance preflight passed"
  fi

  local arm seed experiment run_name worker_cmd
  for arm in "${arms[@]}"; do
    for seed in "${seeds[@]}"; do
      experiment="torchsynth/${arm}"
      run_name="${arm}_s${seed}"

      # The module loads a local path and a fresh pod has none; copyto pins a
      # fixed name so every arm reads the same file whatever the remote layout.
      printf -v worker_cmd "%s" \
        "rclone copyto --checksum ${base_checkpoint} /home/build/base.ckpt && " \
        "exec synth-setter-train " \
        "experiment=${experiment} " \
        "run_name=${run_name} " \
        "model.base_checkpoint=/home/build/base.ckpt " \
        "seed=${seed} " \
        "training.upload_checkpoints_during_training=true " \
        "hydra.run.dir=/home/build/synth-setter/train-run"

      if [[ "${execute}" == false ]]; then
        echo "DRY RUN: ${experiment} seed=${seed} run_name=${run_name}"
        continue
      fi

      if ! uv run synth-setter-skypilot-launch \
        "skypilot_launch/compute=${COMPUTE_OPTION}" \
        skypilot_launch.tail=false \
        "skypilot_launch.worker_image_tag=${WORKER_IMAGE_TAG}" \
        "skypilot_launch.cmd=\"${worker_cmd}\"" \
        hydra.run.dir=/tmp/synth-setter-skypilot-launch \
        hydra.output_subdir=null; then
        echo "Failed to submit ${experiment} seed=${seed}" >&2
        status=1
      fi
    done
  done

  if [[ "${execute}" == false ]]; then
    echo
    echo "Nothing submitted. Re-run with --execute to launch ${#arms[@]}x${#seeds[@]} jobs."
  fi
  return "${status}"
}

main "$@"
