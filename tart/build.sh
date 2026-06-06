#!/usr/bin/env bash
#
# Validate — and optionally build — the Tart macOS dev-VM Packer template.
#
# Runs the Packer validate/build steps for macos.pkr.hcl — the same gates CI
# runs on the template — so contributors can catch regressions locally before
# pushing:
#
#   - validate (default; fast, host-agnostic): packer fmt -check, init, validate.
#   - build (--build): the full `packer build` — pulls the base image and runs
#     every provisioner including the in-VM pytest gate. Needs an Apple Silicon
#     host with Virtualization.framework; takes ~30 min.
#
# The build pins the cloned synth-setter ref to the current git HEAD (override
# with SYNTH_SETTER_GIT_REF) and uses a throwaway VM name (override with
# VM_NAME) so it never clobbers a running `synth-setter-macos` VM.
set -euo pipefail

readonly TEMPLATE="macos.pkr.hcl"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly HERE

usage() {
  cat <<'EOF'
Usage: tart/build.sh [--build] [-h|--help]

  (no args)  Run packer fmt -check, init, and validate (fast presubmit).
  --build    Also run the full packer build (Apple Silicon host, ~30 min).

Environment overrides (only consulted with --build):
  SYNTH_SETTER_GIT_REF  Git ref baked into the VM (default: current HEAD).
  VM_NAME               Built VM name (default: synth-setter-macos-build).
EOF
}

err() {
  echo "[tart/build.sh] $*" >&2
}

main() {
  # The script takes at most one argument; reject extras so unknown flags
  # after a valid one (e.g. `--build --nope`) error rather than being ignored.
  if [[ $# -gt 1 ]]; then
    err "too many arguments — expected at most one"
    usage >&2
    return 2
  fi

  local do_build="false"
  case "${1:-}" in
    --build) do_build="true" ;;
    -h | --help) usage; return 0 ;;
    "") ;;
    *)
      err "unknown argument: ${1}"
      usage >&2
      return 2
      ;;
  esac

  if ! command -v packer >/dev/null; then
    err "packer not found on PATH — install with: brew install packer"
    return 127
  fi

  cd "${HERE}"
  packer fmt -check "${TEMPLATE}"
  packer init "${TEMPLATE}"
  packer validate "${TEMPLATE}"

  if [[ "${do_build}" == "false" ]]; then
    return 0
  fi

  # --build drives Tart's Virtualization.framework — Apple Silicon macOS only.
  # Fail fast here rather than leaving Packer/Tart to emit a cryptic error.
  if [[ "$(uname -sm)" != "Darwin arm64" ]]; then
    err "--build needs an Apple Silicon macOS host (got: $(uname -sm))"
    return 1
  fi
  if ! command -v tart >/dev/null; then
    err "tart not found on PATH — install: brew install cirruslabs/cli/tart"
    return 127
  fi

  local git_ref vm_name
  git_ref="${SYNTH_SETTER_GIT_REF:-$(git -C "${HERE}" rev-parse HEAD)}"
  vm_name="${VM_NAME:-synth-setter-macos-build}"
  err "building VM '${vm_name}' at ref ${git_ref} (this takes ~30 min)…"
  packer build \
    -var "synth_setter_git_ref=${git_ref}" \
    -var "vm_name=${vm_name}" \
    "${TEMPLATE}"
}

main "$@"
