#!/usr/bin/env bash
# Opt-in launcher for continuous CPU profiling: Grafana Alloy -> Grafana Cloud Pyroscope.
#
# Inert unless SYNTH_SETTER_PROFILING_ENABLED is truthy. `pyroscope.ebpf` fails by collecting
# nothing rather than by erroring, so every precondition is checked up front and refused loudly.
# Setup, required container flags, and the RunPod limitation: docs/reference/profiling.md.

set -euo pipefail

ALLOY_BIN="${SYNTH_SETTER_ALLOY_BIN:-alloy}"
ALLOY_CONFIG="${SYNTH_SETTER_ALLOY_CONFIG:-/etc/alloy/profiling.alloy}"
# Path is fixed by the embedded profiler, which writes its symbol cache here unconditionally.
SYMBOL_CACHE_DIR=/tmp/symb-cache
# Alloy's --storage.path defaults to data-alloy/ under the cwd, which in a devcontainer is the
# bind-mounted checkout; keep its state out of the working tree.
STORAGE_DIR=/tmp/alloy-data
TRACEFS_DIR=/sys/kernel/tracing

REQUIRED_CREDENTIAL_VARS=(
  GRAFANA_CLOUD_PYROSCOPE_ENDPOINT
  GRAFANA_CLOUD_PYROSCOPE_USER
  GRAFANA_CLOUD_PYROSCOPE_API_KEY
)

log() {
  echo "start-alloy-profiling: $*" >&2
}

profiling_is_enabled() {
  case "${SYNTH_SETTER_PROFILING_ENABLED:-0}" in
    1 | true | TRUE | yes | YES) return 0 ;;
    *) return 1 ;;
  esac
}

# Decide whether pyroscope.ebpf can actually collect. Host state arrives as arguments so the
# decision is exercisable without root. `nspid_fields` is the field count of the NSpid line in
# /proc/<pid>/status: more than one means a nested PID namespace, where the profiler's host-side
# PIDs never match the namespaced /proc entries and every target silently drops.
alloy_profiling_preflight() {
  local euid="$1" nspid_fields="$2" tracefs_dir="$3"

  local missing=() var
  for var in "${REQUIRED_CREDENTIAL_VARS[@]}"; do
    [[ -n "${!var:-}" ]] || missing+=("$var")
  done
  if ((${#missing[@]} > 0)); then
    log "profiling enabled but these variables are unset: ${missing[*]}"
    return 1
  fi

  if [[ "$euid" -ne 0 ]]; then
    log "pyroscope.ebpf must run as root (effective uid ${euid})"
    return 1
  fi

  if [[ "$nspid_fields" -ne 1 ]]; then
    log "pyroscope.ebpf needs the host PID namespace; relaunch the container with --pid=host"
    return 1
  fi

  if [[ ! -d "$tracefs_dir" ]]; then
    log "tracefs is not mounted at ${tracefs_dir}; mount it read-only from the host"
    return 1
  fi

  return 0
}

# Replace this process with Alloy so a supervisor (or `... &`) owns the collector directly.
alloy_profiling_start() {
  local config="$1"
  mkdir -p "$SYMBOL_CACHE_DIR" "$STORAGE_DIR"
  log "starting Alloy: service_name=${SYNTH_SETTER_PROFILING_SERVICE_NAME:-synth-setter} config=${config}"
  exec "$ALLOY_BIN" run --disable-reporting "--storage.path=${STORAGE_DIR}" "$config"
}

main() {
  if ! profiling_is_enabled; then
    log "profiling disabled; set SYNTH_SETTER_PROFILING_ENABLED=1 to enable"
    return 0
  fi

  local nspid_fields
  nspid_fields="$(awk '/^NSpid:/ {print NF - 1; exit}' /proc/self/status)"
  # Kernels before 4.1 omit NSpid entirely; assume the host namespace rather than refusing on a
  # field that cannot be read.
  nspid_fields="${nspid_fields:-1}"

  alloy_profiling_preflight "$(id -u)" "$nspid_fields" "$TRACEFS_DIR"

  if [[ ! -r "$ALLOY_CONFIG" ]]; then
    log "Alloy config is not readable at ${ALLOY_CONFIG}"
    return 1
  fi

  export SYNTH_SETTER_PROFILING_SERVICE_NAME="${SYNTH_SETTER_PROFILING_SERVICE_NAME:-synth-setter}"
  alloy_profiling_start "$ALLOY_CONFIG"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
