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
  cat >&2 <<'EOF'
Usage: run-flow-simple-440k-all-modes.sh [options]

  --arm NAME     Run only mel, clap, m2l, or same_s.
  --mode NAME    Run only test, validate, val, or predict.
  --execute      Execute evaluation. The default prints commands only.
EOF
}

# Build and optionally run one checkpoint/mode cell.
# Globals:
#   DATASET_TXIDS, OUTPUT_ROOT, R2_OUTPUT_ROOT, WANDB_PROJECT, execute.
# Arguments:
#   Conditioning arm and evaluation mode.
# Outputs:
#   The cell label and shell-escaped command.
# Returns:
#   2 for an unsupported arm; otherwise the eval command status.
run_cell() {
  local arm="$1"
  local mode="$2"
  local artifact_name train_config_id

  case "${arm}" in
    mel)
      artifact_name='model-flow_simple_440k_100k:v0'
      train_config_id='flow_simple_440k_100k'
      ;;
    clap|m2l|same_s)
      artifact_name="model-flow_simple_440k_${arm}_100k:v0"
      train_config_id="flow_simple_440k_${arm}_100k"
      ;;
    *)
      echo "unsupported arm: ${arm}" >&2
      return 2
      ;;
  esac

  local output_dir="${OUTPUT_ROOT}/${arm}/${mode}"
  local upload_uri="${R2_OUTPUT_ROOT}/${arm}/${mode}"
  local -a command=(
    synth-setter-eval
    experiment=surge/flow_simple_440k
    "ckpt_path=\${wandb:${WANDB_PROJECT}/${artifact_name}}"
    "consumed_train_config_id=${train_config_id}"
    "mode=${mode}"
    "run_name=eval_flow_simple_440k_100k_${arm}_${mode}"
    logger=wandb
    callbacks=eval_vst
    "datamodule.download_dataset_txids=${DATASET_TXIDS}"
    datamodule.download_dataset_row_limit=8
    datamodule.batch_size=4
    datamodule.num_workers=0
    datamodule.persistent_workers=false
    trainer.limit_val_batches=2
    +trainer.limit_test_batches=2
    +trainer.limit_predict_batches=2
    "evaluation.upload_output_dir_uri=${upload_uri}"
    "hydra.run.dir=${output_dir}"
  )
  if [[ "${arm}" != 'mel' ]]; then
    command+=("conditioning=${arm}")
  fi
  if [[ "${mode}" == 'predict' ]]; then
    command+=(
      evaluation.render_vst=true
      evaluation.compute_metrics=true
      evaluation.rerender_target=true
    )
  fi

  if [[ "${execute}" == true ]]; then
    printf 'RUN %s/%s\t' "${arm}" "${mode}"
  else
    printf 'DRY RUN %s/%s\t' "${arm}" "${mode}"
  fi
  printf '%q ' "${command[@]}"
  printf '\n'

  if [[ "${execute}" == true ]]; then
    "${command[@]}"
  fi
}

# Parse filters and run the requested matrix cells.
# Globals:
#   ARMS, MODES, execute.
# Arguments:
#   Command-line options.
# Returns:
#   2 for invalid options or filters; otherwise the first failed cell status.
main() {
  local selected_arm=''
  local selected_mode=''
  execute=false

  while (( $# > 0 )); do
    case "$1" in
      --arm)
        if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
          echo "--arm requires a value" >&2
          return 2
        fi
        selected_arm="$2"
        shift 2
        ;;
      --mode)
        if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
          echo "--mode requires a value" >&2
          return 2
        fi
        selected_mode="$2"
        shift 2
        ;;
      --execute) execute=true; shift ;;
      -h|--help) usage; return 0 ;;
      *) echo "unknown argument: $1" >&2; usage; return 2 ;;
    esac
  done

  if [[ -n "${selected_arm}" && " ${ARMS[*]} " != *" ${selected_arm} "* ]]; then
    echo "unsupported arm: ${selected_arm}" >&2
    return 2
  fi
  if [[ -n "${selected_mode}" && " ${MODES[*]} " != *" ${selected_mode} "* ]]; then
    echo "unsupported mode: ${selected_mode}" >&2
    return 2
  fi

  local arm mode
  for arm in "${ARMS[@]}"; do
    [[ -n "${selected_arm}" && "${arm}" != "${selected_arm}" ]] && continue
    for mode in "${MODES[@]}"; do
      [[ -n "${selected_mode}" && "${mode}" != "${selected_mode}" ]] && continue
      run_cell "${arm}" "${mode}"
    done
  done
}

main "$@"
