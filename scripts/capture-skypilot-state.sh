#!/bin/bash
# Capture kind/k8s controller-side state for the SkyPilot managed-jobs
# controller and worker pods before `sky local down` reaps the cluster.
#
# Why this exists: SkyPilot's on-disk sky_logs don't include controller pod
# status, scheduler events, or worker-pod RBAC errors — all of which live
# in the kube apiserver and vanish when `sky local down` tears down the
# kind cluster. This script captures that state BEFORE teardown so
# failure diagnostics survive the run.
#
# In particular, the controller pod's on-pod `~/sky_logs/sky-*/provision.log`
# carries the per-retry "Insufficient cpu / memory" + `terminate_instances`
# loop that pinpoints scheduling-side hangs. The CI surface symptom is
# silent dead-air after `streaming logs for job N`; only provision.log
# reveals why. See PR #876.
#
# Usage:
#   scripts/capture-skypilot-state.sh
#
# Required env:
#   RUN_METADATA_DIR   Output root. Captures land in <RUN_METADATA_DIR>/k8s_state/.
#
# Output files (under <RUN_METADATA_DIR>/k8s_state/):
#   pods.txt, events.txt, nodes.txt, pods-yaml.txt         — cluster overview
#   describe-worker-<safe-name>.txt                        — per worker pod
#   describe-worker-none.txt                               — sentinel if no workers
#   describe-controller.txt                                — controller pod (or sentinel)
#   logs-controller.txt, logs-controller-previous.txt      — controller stdout/err
#   controller-provision-log.txt                           — on-pod ~/sky_logs provision.log
#
# Behavior: best-effort. Every kubectl/exec call is tolerated with `|| true`
# so a missing pod, rename, or unprivileged exec never fails the script.
# Exits non-zero only if RUN_METADATA_DIR is unset/empty.

# `-e` is intentionally omitted: best-effort calls deliberately tolerate
# failure via `|| true`. `-u` and `-o pipefail` still catch real bugs
# (typo'd variable, broken pipe before the `|| true`).
set -uo pipefail

if [[ -z "${RUN_METADATA_DIR:-}" ]]; then
  echo "error: RUN_METADATA_DIR is required (unset or empty)" >&2
  exit 2
fi

readonly OUT_DIR="${RUN_METADATA_DIR}/k8s_state"

# Worker-pod filename safety: pod names contain only [a-z0-9-] per RFC 1123,
# but we sanitize defensively in case a future kubectl format change leaks
# a `/` or `.` and turns the describe-file path into a traversal.
_safe_basename() {
  printf %s "$1" | tr -c 'A-Za-z0-9_-' '_'
}

_capture_cluster_overview() {
  mkdir -p "${OUT_DIR}"
  kubectl get pods -A -o wide > "${OUT_DIR}/pods.txt" 2>&1 || true
  kubectl get events -A --sort-by=.lastTimestamp \
    > "${OUT_DIR}/events.txt" 2>&1 || true
  kubectl describe nodes > "${OUT_DIR}/nodes.txt" 2>&1 || true
  kubectl get pods -A -o yaml > "${OUT_DIR}/pods-yaml.txt" 2>&1 || true
}

# Filters workers in the `default` namespace; sky-jobs-controller may live in
# the `skypilot` namespace and is captured separately by _capture_controller_pod.
_capture_worker_pods() {
  local worker_pods
  readarray -t worker_pods < <(kubectl get pods -n default --no-headers 2>/dev/null \
    | awk '$1 !~ /^sky-jobs-controller/ {print $1}')
  if [[ ${#worker_pods[@]} -eq 0 ]]; then
    echo "no non-controller pods in default namespace at capture time" \
      > "${OUT_DIR}/describe-worker-none.txt"
    return 0
  fi
  local pod safe
  for pod in "${worker_pods[@]}"; do
    safe="$(_safe_basename "${pod}")"
    kubectl describe pod -n default "${pod}" \
      > "${OUT_DIR}/describe-worker-${safe}.txt" 2>&1 || true
  done
}

# `ray-node` is the SkyPilot 0.12 managed-jobs-controller pod's container name.
# The `bash -lc` payload tails the latest sky_logs/sky-* directory's provision.log
# — that's where SkyPilot writes the per-retry scheduling decisions.
_capture_controller_pod() {
  local controller_line controller_ns controller_pod
  controller_line="$(kubectl get pods -A --no-headers 2>/dev/null \
    | grep sky-jobs-controller | head -n1)"
  if [[ -z "${controller_line}" ]]; then
    echo "no sky-jobs-controller pod found" \
      > "${OUT_DIR}/describe-controller.txt"
    return 0
  fi
  controller_ns="$(awk '{print $1}' <<<"${controller_line}")"
  controller_pod="$(awk '{print $2}' <<<"${controller_line}")"
  kubectl describe pod -n "${controller_ns}" "${controller_pod}" \
    > "${OUT_DIR}/describe-controller.txt" 2>&1 || true
  kubectl logs -n "${controller_ns}" "${controller_pod}" \
    > "${OUT_DIR}/logs-controller.txt" 2>&1 || true
  kubectl logs -n "${controller_ns}" "${controller_pod}" --previous \
    > "${OUT_DIR}/logs-controller-previous.txt" 2>&1 || true
  # shellcheck disable=SC2016
  # ${HOME} and ${latest} must expand inside the controller pod, not the runner.
  kubectl exec -n "${controller_ns}" "${controller_pod}" -c ray-node -- \
    bash -lc 'set -euo pipefail; latest="$(ls -td "${HOME}/sky_logs/sky-"*/ 2>/dev/null | head -n1)"; if [[ -n "${latest}" ]]; then echo "=== ${latest}provision.log ==="; tail -400 "${latest}provision.log" 2>/dev/null; fi' \
    > "${OUT_DIR}/controller-provision-log.txt" 2>&1 || true
}

_print_capture_size() {
  echo "=== k8s_state sizes ==="
  du -sh "${OUT_DIR}" 2>/dev/null || true
}

main() {
  _capture_cluster_overview
  _capture_worker_pods
  _capture_controller_pod
  _print_capture_size
}

main "$@"
