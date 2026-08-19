#!/bin/bash
# Evaluate the canonical Simple 440k conditioning checkpoints in every mode.

set -euo pipefail

readonly -a ARMS=(mel clap m2l same_s)
readonly -a MODES=(test validate val predict)
readonly DATASET_TXIDS='{'\
'train:84e41d46-61fe-4475-a4d3-894108480502,'\
'val:9d4d2fea-23c1-4e9d-af69-82c6b1fe533d,'\
'test:4d4e1ba2-2458-4472-94e4-968a99dcf02c}'
readonly OUTPUT_ROOT='/home/build/synth-setter/eval-flow-simple-440k'
readonly R2_OUTPUT_ROOT='r2://experiments/eval/flow-simple-440k-100k'
readonly WANDB_PROJECT='khaledtinubu-n-a/synth-setter'

usage() {
  echo 'Usage: run-flow-simple-440k-all-modes.sh [--arm=NAME] [--mode=NAME] [--execute]' >&2
}

# Build and optionally execute one checkpoint/mode evaluation.
# Arguments: $1 arm, $2 mode, $3 true to execute.
# Outputs: command preview and evaluator output to stdout.
# Returns: evaluator status, or 0 for a dry run.
run_cell() {
  local arm="$1"
  local mode="$2"
  local execute="$3"
  local train_config_id='flow_simple_440k_100k'
  [[ "${arm}" == 'mel' ]] || train_config_id="flow_simple_440k_${arm}_100k"

  local output_dir="${OUTPUT_ROOT}/${arm}/${mode}"
  local upload_uri="${R2_OUTPUT_ROOT}/${arm}/${mode}"
  local -a command=(
    synth-setter-eval experiment=surge/flow_simple_440k
    "ckpt_path=\${wandb:${WANDB_PROJECT}/model-${train_config_id}:v0}"
    "consumed_train_config_id=${train_config_id}" "mode=${mode}"
    "run_name=eval_flow_simple_440k_100k_${arm}_${mode}"
    logger=wandb callbacks=eval_vst
    "datamodule.download_dataset_txids=${DATASET_TXIDS}"
    datamodule.download_dataset_row_limit=8 datamodule.batch_size=4
    datamodule.num_workers=0 datamodule.persistent_workers=false
    trainer.limit_val_batches=2 +trainer.limit_test_batches=2
    +trainer.limit_predict_batches=2
    "evaluation.upload_output_dir_uri=${upload_uri}"
    "hydra.run.dir=${output_dir}"
  )
  [[ "${arm}" == 'mel' ]] || command+=("conditioning=${arm}")
  if [[ "${mode}" == 'predict' ]]; then
    command+=(evaluation.render_vst=true evaluation.compute_metrics=true)
    command+=(evaluation.rerender_target=true)
  fi

  local label='DRY RUN'
  [[ "${execute}" == true ]] && label='RUN'
  printf '%s %s/%s\t' "${label}" "${arm}" "${mode}"
  printf '%q ' "${command[@]}"
  printf '\n'
  [[ "${execute}" == false ]] || "${command[@]}"
}

# Parse filters and run matching evaluation cells.
# Globals: ARMS and MODES are read.
# Arguments: --arm=NAME, --mode=NAME, --execute, or --help.
# Outputs: command plans to stdout and diagnostics to stderr.
# Returns: 0 on success, 2 for invalid input, or an evaluator status.
main() {
  local selected_arm=''
  local selected_mode=''
  local execute=false

  while (( $# > 0 )); do
    case "$1" in
      --arm=mel|--arm=clap|--arm=m2l|--arm=same_s) selected_arm="${1#*=}" ;;
      --mode=test|--mode=validate|--mode=val|--mode=predict) selected_mode="${1#*=}" ;;
      --arm=*) echo "unsupported arm: ${1#*=}" >&2; return 2 ;;
      --mode=*) echo "unsupported mode: ${1#*=}" >&2; return 2 ;;
      --execute) execute=true ;;
      -h|--help) usage; return 0 ;;
      *) echo "unknown argument: $1" >&2; usage; return 2 ;;
    esac
    shift
  done

  local arm mode
  for arm in "${ARMS[@]}"; do
    [[ -n "${selected_arm}" && "${arm}" != "${selected_arm}" ]] && continue
    for mode in "${MODES[@]}"; do
      [[ -n "${selected_mode}" && "${mode}" != "${selected_mode}" ]] && continue
      run_cell "${arm}" "${mode}" "${execute}"
    done
  done
}

main "$@"
