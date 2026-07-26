#!/bin/bash
# Preserve a remote training run's W&B recovery state before worker teardown.

set -euo pipefail

readonly DEFAULT_RECOVERY_PREFIX="r2://intermediate-data/diagnostics/wandb/training"
readonly RECOVERY_RETENTION="30d"

wandb_root=""
prior_run=""

resolve_latest_run() {
  local latest="${wandb_root}/latest-run"
  [[ -L "${latest}" ]] || return 1

  local resolved
  resolved="$(readlink -f "${latest}")" || return 1
  [[ -d "${resolved}" && "${resolved}" == "${wandb_root}/"* ]] || return 1
  printf '%s\n' "${resolved}"
}

#######################################
# Upload the new canonical run and expire old recovery bundles.
# Outputs:
#   Recovery URI to stdout; failures to stderr.
# Returns:
#   Nonzero when discovery, archival, upload, or retention fails.
#######################################
archive_wandb_run() {
  local run_dir
  run_dir="$(resolve_latest_run)" || {
    echo "W&B recovery failed: no canonical latest-run directory" >&2
    return 1
  }
  if [[ -n "${prior_run}" && "${run_dir}" == "${prior_run}" ]]; then
    echo "W&B recovery failed: training created no new run directory" >&2
    return 1
  fi

  local -a datastore_files
  readarray -t datastore_files < <(find "${run_dir}" -maxdepth 1 -type f -name 'run-*.wandb')
  if (( ${#datastore_files[@]} != 1 )); then
    echo "W&B recovery failed: expected one run-*.wandb datastore" >&2
    return 1
  fi

  local datastore_name="${datastore_files[0]##*/}"
  local run_id="${datastore_name#run-}"
  run_id="${run_id%.wandb}"
  if [[ ! "${run_id}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "W&B recovery failed: unsafe run ID" >&2
    return 1
  fi

  local attempt_id
  attempt_id="$(cat /proc/sys/kernel/random/uuid)"
  local recovery_uri="${DEFAULT_RECOVERY_PREFIX}/${run_id}/${attempt_id}/wandb-run.tar.gz"

  local archive
  archive="$(mktemp --suffix=.tar.gz)"
  if ! tar -C "${run_dir%/*}" -czf "${archive}" "${run_dir##*/}"; then
    rm -f "${archive}"
    echo "W&B recovery failed: archive creation failed" >&2
    return 1
  fi

  local remote_path="r2:${recovery_uri#r2://}"
  if ! rclone copyto --immutable --checksum -v \
    --contimeout=30s --timeout=3h --retries=3 \
    "${archive}" "${remote_path}"; then
    rm -f "${archive}"
    echo "W&B recovery failed: R2 upload failed" >&2
    return 1
  fi
  rm -f "${archive}"
  printf 'WANDB_RECOVERY_URI=%s\n' "${recovery_uri}"

  if ! rclone delete --checksum --min-age="${RECOVERY_RETENTION}" \
    --contimeout=30s --timeout=3h --retries=3 \
    "r2:${DEFAULT_RECOVERY_PREFIX#r2://}"; then
    echo "W&B recovery failed: retention cleanup failed" >&2
    return 1
  fi
}

#######################################
# Archive on EXIT without replacing a failed training command's status.
# Returns:
#   Exits with the child status, or the archive status after child success.
#######################################
preserve_status_and_archive() {
  local child_status=$?
  trap - EXIT
  set +e
  archive_wandb_run
  local archive_status=$?

  if (( child_status != 0 )); then
    exit "${child_status}"
  fi
  exit "${archive_status}"
}

main() {
  if (( $# < 3 )) || [[ "$2" != "--" ]]; then
    echo "Usage: $0 <wandb-root> -- <training-command> [args...]" >&2
    return 2
  fi

  wandb_root="$(realpath -m "${1%/}")"

  prior_run="$(resolve_latest_run || true)"
  shift 2
  trap preserve_status_and_archive EXIT
  "$@"
}

main "$@"
