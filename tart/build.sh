#!/usr/bin/env bash
#
# Validate — and optionally build — the Tart macOS dev-VM Packer template,
# mirroring the gates CI runs on macos.pkr.hcl so contributors catch
# regressions locally before pushing:
#   - validate (default; fast, host-agnostic): packer fmt -check, init, validate.
#   - build (--build): full `packer build` — pulls the base image, runs every
#     provisioner incl. the in-VM pytest gate. Apple Silicon host; ~30 min.
#
# --build pins the cloned ref to the current git HEAD (override:
# SYNTH_SETTER_GIT_REF) and uses a throwaway VM name (override: VM_NAME) so it
# never clobbers a running `synth-setter-macos` VM.
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
    err "packer not found on PATH — see https://developer.hashicorp.com/packer/install (macOS: brew install packer)"
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
  # `hw.optional.arm64` reports the hardware, so it stays 1 even under Rosetta
  # (where `uname -m` would misreport x86_64).
  if [[ "$(uname -s)" != "Darwin" ]] ||
    [[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" != "1" ]]; then
    err "--build needs an Apple Silicon macOS host"
    return 1
  fi
  if ! command -v tart >/dev/null; then
    err "tart not found on PATH — install: brew install cirruslabs/cli/tart"
    return 127
  fi

  local git_ref vm_name
  if [[ -n "${SYNTH_SETTER_GIT_REF:-}" ]]; then
    git_ref="${SYNTH_SETTER_GIT_REF}"
  elif ! git_ref="$(git -C "${HERE}" rev-parse HEAD 2>/dev/null)"; then
    err "could not resolve git HEAD — set SYNTH_SETTER_GIT_REF to the ref to build"
    return 1
  fi
  vm_name="${VM_NAME:-synth-setter-macos-build}"
  err "building VM '${vm_name}' at ref ${git_ref} (this takes ~30 min)…"
  packer build \
    -var "synth_setter_git_ref=${git_ref}" \
    -var "vm_name=${vm_name}" \
    "${TEMPLATE}"
}

main "$@"
