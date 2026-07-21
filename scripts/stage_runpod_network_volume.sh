#!/bin/bash
# Stage an immutable R2 dataset on an attached RunPod network volume.

set -euo pipefail

readonly COMPLETION_MARKER=".synth-setter-stage-complete"

main() {
  if (( $# != 2 )); then
    echo "Usage: $0 <r2://source/> <destination-directory>" >&2
    return 2
  fi

  local source_uri="$1"
  local destination="$2"
  if [[ "${source_uri}" != r2://* ]]; then
    echo "Dataset source must use r2://: ${source_uri}" >&2
    return 2
  fi

  local source_path="r2:${source_uri#r2://}"
  mkdir -p "${destination}"
  rm -f "${destination}/${COMPLETION_MARKER}"
  rclone copy --immutable --checksum "${source_path}" "${destination}"
  rclone check --one-way --checksum "${source_path}" "${destination}"
  printf '%s\n' "${source_uri}" > "${destination}/${COMPLETION_MARKER}"
}

main "$@"
