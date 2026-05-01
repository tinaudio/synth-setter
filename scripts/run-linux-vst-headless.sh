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
  # we tracked (XVFB / XSETTINGS / OPENBOX). Every step echoes to stderr
  # so `tail_logs` evidence can pinpoint where cleanup stalls if it ever
  # stalls again.
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
  if [ -n "${TMP_DIR-}" ]; then
    echo "[wrapper] cleanup: removing TMP_DIR=${TMP_DIR}" >&2
    rm -rf "${TMP_DIR}" || true
  fi
  echo "[wrapper] cleanup: done" >&2
}
trap cleanup EXIT

# Start Xvfb (let it pick a display)
# -displayfd 3 writes the chosen display number to file descriptor 3
# 3>"$DISPLAY_FILE" redirects FD 3 to our temp file so we can read it later
Xvfb -displayfd 3 -screen 0 1920x1080x24 -nolisten tcp \
  </dev/null >/dev/null 3>"$DISPLAY_FILE" 2>"$TMP_DIR/xvfb.log" &
XVFB_PID=$!

# Wait for Xvfb to output the display number
count=0
while [ ! -s "$DISPLAY_FILE" ]; do
  sleep 0.1
  count=$((count+1))
  if [ "$count" -ge 100 ]; then
     echo "Timeout waiting for Xvfb to start" >&2
     cat "$TMP_DIR/xvfb.log" >&2
     exit 1
  fi
  # Check if Xvfb died
  if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "Xvfb died unexpectedly" >&2
    cat "$TMP_DIR/xvfb.log" >&2
    exit 1
  fi
done

DISPLAY_NUM=$(cat "$DISPLAY_FILE")
export DISPLAY=:${DISPLAY_NUM}

# Wait for X to be ready (avoid races)
for _ in {1..50}; do
  if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 || { echo "Xvfb did not start"; cat "$TMP_DIR/xvfb.log" || true; exit 1; }

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
