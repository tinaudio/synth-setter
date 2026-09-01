#!/bin/bash
# Submit the flow-sketch content/sketch CFG grid as managed RunPod jobs.

set -euo pipefail
readonly COMPUTE_OPTION="runpod/torchsynth"
WORKER_IMAGE_TAG="dev-snapshot-$(git rev-parse HEAD)"
readonly WORKER_IMAGE_TAG
readonly EXPERIMENT="surge/flow_sketch_cfg_ablation"
readonly CHECKPOINT_URI="r2://intermediate-data/checkpoints/flow_sketch_prelim/model.ckpt"
readonly CHECKPOINT_PATH="/home/build/flow_sketch_prelim.ckpt"
readonly R2_BASE="r2://experiments/eval/flow_sketch_prelim/flow_sketch_prelim-20260901T163655048Z/cfg-ablation"
readonly -a CONTENT_STRENGTHS=(0 1 2)
readonly -a SKETCH_STRENGTHS=(0 1 2)
# Validate routing and launch one job per CFG pair.
main() {
  local execute=false
  local ablation_id
  ablation_id="cfg-$(date -u +%Y%m%dT%H%M%SZ)"
  while (($# > 0)); do
    case "$1" in
      --ablation-id)
        if (($# < 2)); then
          echo "--ablation-id requires a value" >&2
          return 2
        fi
        ablation_id="$2"
        shift 2
        ;;
      --execute)
        execute=true
        shift
        ;;
      *)
        echo "Usage: $0 [--ablation-id ID] [--execute]" >&2
        return 2
        ;;
    esac
  done
  if [[ ! "${ablation_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ablation ID must contain only letters, numbers, dot, underscore, or hyphen" >&2
    return 2
  fi

  unset SKYPILOT_API_SERVER_ENDPOINT SKYPILOT_API_SERVER_KEY SKYPILOT_API_SERVER_TOKEN
  if [[ "${execute}" == true ]]; then
    uv run python -c \
      "from synth_setter.pipeline.skypilot_launch import _check_runpod_balance; _check_runpod_balance()"
    echo "balance preflight passed"
  fi

  local content sketch arm run_root dataset_root validation_dir audio_dir
  local validation_uri audio_uri worker_cmd status=0
  for content in "${CONTENT_STRENGTHS[@]}"; do
    for sketch in "${SKETCH_STRENGTHS[@]}"; do
      arm="c${content}-s${sketch}"
      run_root="/home/build/synth-setter/cfg-ablation/${ablation_id}/${arm}"
      dataset_root="${run_root}/data"
      validation_dir="${run_root}/validation"
      audio_dir="${run_root}/audio"
      validation_uri="${R2_BASE}/${ablation_id}/${arm}/validation"
      audio_uri="${R2_BASE}/${ablation_id}/${arm}/audio"

      printf -v worker_cmd '%s' \
        "rclone copyto --checksum ${CHECKPOINT_URI} ${CHECKPOINT_PATH} && " \
        "synth-setter-eval experiment=${EXPERIMENT} ckpt_path=${CHECKPOINT_PATH} " \
        "mode=validate run_name=cfg-${arm}-validation " \
        "datamodule.dataset_root=${dataset_root} " \
        "model.validation_cfg_strength=${content} " \
        "model.validation_sketch_cfg_strength=${sketch} " \
        "trainer.limit_val_batches=20 paths.output_dir=${validation_dir} " \
        "hydra.run.dir=${validation_dir} evaluation.upload_output_dir_uri=${validation_uri} && " \
        "exec synth-setter-eval experiment=${EXPERIMENT} ckpt_path=${CHECKPOINT_PATH} " \
        "mode=predict run_name=cfg-${arm}-audio " \
        "datamodule.dataset_root=${dataset_root} datamodule.batch_size=32 " \
        "model.test_cfg_strength=${content} model.test_sketch_cfg_strength=${sketch} " \
        "trainer.limit_predict_batches=1 paths.output_dir=${audio_dir} " \
        "hydra.run.dir=${audio_dir} evaluation.upload_output_dir_uri=${audio_uri}"

      if [[ "${execute}" == false ]]; then
        echo "DRY RUN: content_cfg=${content} sketch_cfg=${sketch} ${worker_cmd}"
      elif ! uv run synth-setter-skypilot-launch \
        "skypilot_launch/compute=${COMPUTE_OPTION}" \
        skypilot_launch.tail=false \
        "skypilot_launch.worker_image_tag=${WORKER_IMAGE_TAG}" \
        "skypilot_launch.cmd=\"${worker_cmd}\"" \
        hydra.run.dir=/tmp/synth-setter-skypilot-launch \
        hydra.output_subdir=null; then
        echo "Failed to submit content_cfg=${content} sketch_cfg=${sketch}" >&2
        status=1
      fi
    done
  done

  if [[ "${execute}" == false ]]; then
    echo "Nothing submitted. Re-run with --execute to launch the 3x3 grid."
  fi
  return "${status}"
}

main "$@"
