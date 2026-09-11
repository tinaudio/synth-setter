#!/usr/bin/env bash
# Validate shell-interpolated inputs shared by train and eval dispatch workflows.
set -euo pipefail

if (( $# != 2 && $# != 4 )); then
  echo "Usage: $0 EXPERIMENT COMPUTE_OPTION [CHECKPOINT_REF DATASET_ROOT_URI]" >&2
  exit 2
fi

readonly EXPERIMENT=$1
readonly COMPUTE_OPTION=$2

if [[ ! "$EXPERIMENT" =~ ^[A-Za-z0-9_./-]+$ ]]; then
  echo "::error::experiment contains unsupported shell characters"
  exit 1
fi
if [[ ! "$COMPUTE_OPTION" =~ ^[A-Za-z0-9_./-]+$ ]]; then
  echo "::error::compute contains unsupported shell characters"
  exit 1
fi
if (( $# == 2 )); then
  exit 0
fi

readonly CHECKPOINT_REF=$3
readonly DATASET_ROOT_URI=$4
if [[ ! "$CHECKPOINT_REF" =~ ^[A-Za-z0-9_./+-]+:v[0-9]+$ ]]; then
  echo "::error::checkpoint_ref must name an explicit W&B artifact version"
  exit 1
fi
if [[ ! "$DATASET_ROOT_URI" =~ ^r2://[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*/?$ ]]; then
  echo "::error::dataset_root_uri must name an r2:// bucket and safe object path"
  exit 1
fi
