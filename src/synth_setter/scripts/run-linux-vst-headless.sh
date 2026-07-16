#!/usr/bin/env bash
#
# Headless VST3 bootstrap for Linux CI
#
# Some VST3 plugins query X11 desktop
# settings during editor initialization. In minimal headless
# environments, no XSettings manager owns the _XSETTINGS_S0 selection.
# This can lead to XGetProperty() being called on window ID 0,
# resulting in a BadWindow error and process termination.
#
# This script provides the minimal desktop components required for
# stable initialization:
#   - Xvfb (headless X server)
#   - D-Bus session
#   - xsettingsd (XSettings manager)
#
# It creates a lightweight, CI-friendly X session sufficient for
# plugin initialization without requiring a full desktop environment.

set -euo pipefail

TMP_DIR=$(mktemp -d)
export XAUTHORITY="${XAUTHORITY:-$TMP_DIR/.Xauthority}"

export LIBGL_ALWAYS_SOFTWARE=1
export NO_AT_BRIDGE=1
export JUCE_USE_XINPUT2=0

DISPLAY_FILE="$TMP_DIR/display_num"

cleanup() {
  # `kill PID` is async — bash returns while the child is still draining.
  # On RunPod, SkyPilot's job-status reporter watches the SSH session's
  # process tree, so any straggler child keeps the job in RUNNING forever
  # even after the wrapped command and the wrapper bash have logically
  # finished (#735). `wait` after the SIGTERMs reaps the X-stack daemons
  # we tracked; `pkill -P $$` then sweeps any grandchildren openbox or
  # dbus-launch may have spawned that we didn't track. Every step echoes
  # to stderr so `tail_logs` evidence can pinpoint where cleanup stalls
  # if it ever stalls again.
  echo "[wrapper] cleanup: starting (pid=$$)" >&2
  echo "[wrapper] cleanup: child PIDs OPENBOX=${OPENBOX_PID-} XSETTINGS=${XSETTINGS_PID-} XVFB=${XVFB_PID-}" >&2
  echo "[wrapper] cleanup: pre-kill process tree:" >&2
  ps -eo pid,ppid,pgid,stat,comm --no-headers \
    | awk -v me=$$ '$2==me || $3==me' >&2 || true
  if [ -n "${OPENBOX_PID-}" ]; then
    echo "[wrapper] cleanup: kill openbox pid=${OPENBOX_PID}" >&2
    kill "${OPENBOX_PID}" 2>/dev/null || true
  fi
  if [ -n "${XSETTINGS_PID-}" ]; then
    echo "[wrapper] cleanup: kill xsettingsd pid=${XSETTINGS_PID}" >&2
    kill "${XSETTINGS_PID}" 2>/dev/null || true
  fi
  if [ -n "${XVFB_PID-}" ]; then
    echo "[wrapper] cleanup: kill xvfb pid=${XVFB_PID}" >&2
    kill "${XVFB_PID}" 2>/dev/null || true
  fi
  echo "[wrapper] cleanup: waiting for tracked children to reap" >&2
  wait 2>/dev/null || true
  echo "[wrapper] cleanup: sweeping any orphan grandchildren" >&2
  pkill -P $$ 2>/dev/null || true
  echo "[wrapper] cleanup: post-kill process tree:" >&2
  ps -eo pid,ppid,pgid,stat,comm --no-headers \
    | awk -v me=$$ '$2==me || $3==me' >&2 || true
  if [ -n "${TMP_DIR-}" ]; then
    echo "[wrapper] cleanup: removing TMP_DIR=${TMP_DIR}" >&2
    rm -rf "${TMP_DIR}" || true
  fi
  echo "[wrapper] cleanup: done" >&2
}
trap cleanup EXIT

# Concurrent shard renders each bootstrap their own Xvfb; under startup
# contention an instance can lose the display-lock race or miss the readiness
# window, so the bootstrap retries instead of failing the renderer (#2035).
XVFB_BOOTSTRAP_ATTEMPTS="${XVFB_BOOTSTRAP_ATTEMPTS:-3}"
# Readiness probes per attempt, 0.1 s apart; widen on congested hosts.
XVFB_READY_PROBES="${XVFB_READY_PROBES:-50}"
# Max retry jitter in deciseconds; 0 disables (deterministic tests).
XVFB_RETRY_JITTER_MAX="${XVFB_RETRY_JITTER_MAX:-9}"

# Start Xvfb and wait until it accepts connections; returns non-zero on any
# startup fault, leaving XVFB_PID for reap_failed_xvfb to collect.
start_xvfb_attempt() {
  : > "$DISPLAY_FILE"
  # -displayfd 3 writes the chosen display number to file descriptor 3;
  # 3>"$DISPLAY_FILE" redirects FD 3 to our temp file so we can read it later
  Xvfb -displayfd 3 -screen 0 1920x1080x24 -nolisten tcp \
    </dev/null >/dev/null 3>"$DISPLAY_FILE" 2>"$TMP_DIR/xvfb.log" &
  XVFB_PID=$!

  # Check for death before sleeping: a lost display-lock race kills Xvfb
  # within milliseconds, and instant detection keeps failed attempts cheap.
  local count=0
  while [ ! -s "$DISPLAY_FILE" ]; do
    if ! kill -0 "$XVFB_PID" 2>/dev/null; then
      echo "Xvfb died unexpectedly" >&2
      return 1
    fi
    count=$((count+1))
    if [ "$count" -gt 100 ]; then
      echo "Timeout waiting for Xvfb to start" >&2
      return 1
    fi
    sleep 0.1
  done

  DISPLAY=":$(cat "$DISPLAY_FILE")"
  export DISPLAY

  for _ in $(seq "$XVFB_READY_PROBES"); do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  echo "Xvfb did not become ready on $DISPLAY" >&2
  return 1
}

# Kill and reap a failed attempt's Xvfb (a readiness timeout leaves it
# running) so retries never stack servers, then surface its log.
reap_failed_xvfb() {
  kill "$XVFB_PID" 2>/dev/null || true
  wait "$XVFB_PID" 2>/dev/null || true
  XVFB_PID=""
  cat "$TMP_DIR/xvfb.log" >&2 || true
}

attempt=1
until start_xvfb_attempt; do
  reap_failed_xvfb
  if [ "$attempt" -ge "$XVFB_BOOTSTRAP_ATTEMPTS" ]; then
    echo "Xvfb bootstrap failed after ${attempt} attempt(s)" >&2
    exit 1
  fi
  attempt=$((attempt+1))
  # Sub-second jitter decorrelates concurrent bootstraps racing the same
  # display locks.
  if [ "$XVFB_RETRY_JITTER_MAX" -gt 0 ]; then
    sleep "0.$((RANDOM % XVFB_RETRY_JITTER_MAX + 1))"
  fi
  echo "[wrapper] retrying Xvfb bootstrap (attempt ${attempt}/${XVFB_BOOTSTRAP_ATTEMPTS})" >&2
done

# Start an XSettings manager.
# `</dev/null` detaches stdin so backgrounded daemons (and any grandchildren
# they fork) don't keep the parent SSH session's stdin pipe open. Without
# this, SkyPilot's RunPod backend never observes EOF on the SSH command's
# pipes and the job stays in RUNNING forever even after the wrapped
# command exits (#735).
xsettingsd --config /dev/null </dev/null >"$TMP_DIR/xsettingsd.log" 2>&1 &
XSETTINGS_PID=$!

# Start a lightweight window manager (important for many plugins)
openbox-session </dev/null >"$TMP_DIR/openbox.log" 2>&1 &
OPENBOX_PID=$!

# Run the actual command inside a D-Bus session.
#
# Do NOT use `exec` here. With `exec`, the wrapper bash is replaced by
# dbus-run-session, the `trap cleanup EXIT` above never fires, and
# Xvfb / xsettingsd / openbox keep running after the wrapped command
# exits. On RunPod, SkyPilot's job-status reporter never sees the SSH
# session's process tree go quiet — the job stays in RUNNING forever
# even after the worker uploads its artifacts. Bisected via the
# test-skypilot-debug matrix; see #735 for the full evidence.
dbus-run-session -- "$@"
