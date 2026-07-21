#!/bin/bash
# Stage an immutable R2 dataset on an attached RunPod network volume.

set -euo pipefail

readonly COMPLETION_MARKER=".synth-setter-stage-complete"

#######################################
# Copy and validate one R2 dataset prefix before publishing its marker.
# Arguments:
#   R2 dataset URI containing a bucket and key.
#   Destination directory on the mounted network volume.
# Returns:
#   2 when the arguments do not identify one dataset prefix.
#######################################
main() {
  if (( $# != 2 )); then
    echo "Usage: $0 <r2://source/> <destination-directory>" >&2
    return 2
  fi

  local source_uri="$1"
  local destination="$2"
  if [[ ! "${source_uri}" =~ ^r2://[^/]+/.+ ]]; then
    echo "Dataset source must include an r2:// bucket and key: ${source_uri}" >&2
    return 2
  fi

  local source_path="r2:${source_uri#r2://}"
  mkdir -p "${destination}"
  rm -f "${destination}/${COMPLETION_MARKER}"
  # Reliability flags mirror r2_io.py's shared argv; --transfers/--multi-thread-streams
  # parallelize the ~10 GiB objects; --stats keeps the long copy observable in job logs.
  rclone copy --immutable --checksum -v \
    --contimeout=30s --timeout=300s --retries=3 \
    --transfers=8 --multi-thread-streams=8 \
    --stats 60s --stats-one-line \
    "${source_path}" "${destination}"
  rclone check --checksum "${source_path}" "${destination}"
  : > "${destination}/${COMPLETION_MARKER}"
}

main "$@"
