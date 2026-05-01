#!/usr/bin/env bash
#
# DEBUG variant — bisect canary for the stdin-detach piece of the wrapper fix.
# Identical to scripts/run-linux-vst-headless.sh except the backgrounded
# daemons (Xvfb, xsettingsd, openbox-session) keep their stdin attached to the
# wrapper bash's stdin (i.e. NO `</dev/null`). Cleanup, exec-removal, wait,
# and pkill -P $$ are all kept.
#
# Used by the test-skypilot-debug matrix's `headless-no-stdin-detach` variant
# to prove that the stdin detach is independently necessary on RunPod: any
# daemon (or grandchild reparented to PID 1) that inherits the SSH command's
# stdin keeps that pipe open after the foreground command exits, and
# SkyPilot's RunPod backend never observes EOF on the SSH pipes — the job
# stays in RUNNING forever. See scripts/run-linux-vst-headless.sh for the
# narrative.
#
# Expected matrix outcome: FAIL (job times out at $JOB_DEADLINE_SECONDS).

set -euo pipefail

TMP_DIR=$(mktemp -d)
export XAUTHORITY="${XAUTHORITY:-$TMP_DIR/.Xauthority}"

export LIBGL_ALWAYS_SOFTWARE=1
export NO_AT_BRIDGE=1
export JUCE_USE_XINPUT2=0

DISPLAY_FILE="$TMP_DIR/display_num"

cleanup() {
  echo "[wrapper-no-stdin-detach] cleanup: starting (pid=$$)" >&2
  if [ -n "${OPENBOX_PID-}" ]; then kill "${OPENBOX_PID}" 2>/dev/null || true; fi
  if [ -n "${XSETTINGS_PID-}" ]; then kill "${XSETTINGS_PID}" 2>/dev/null || true; fi
  if [ -n "${XVFB_PID-}" ]; then kill "${XVFB_PID}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  pkill -P $$ 2>/dev/null || true
  if [ -n "${TMP_DIR-}" ]; then rm -rf "${TMP_DIR}" || true; fi
  echo "[wrapper-no-stdin-detach] cleanup: done" >&2
}
trap cleanup EXIT

# NOTE: no `</dev/null` on Xvfb's stdin (this is the reverted piece).
# Xvfb still gets stdout redirected because `-displayfd` and stdout share
# behavior in Xvfb's CLI; without that, the displayfd handshake breaks.
Xvfb -displayfd 3 -screen 0 1920x1080x24 -nolisten tcp \
  >/dev/null 3>"$DISPLAY_FILE" 2>"$TMP_DIR/xvfb.log" &
XVFB_PID=$!

count=0
while [ ! -s "$DISPLAY_FILE" ]; do
  sleep 0.1
  count=$((count+1))
  if [ "$count" -ge 100 ]; then
     echo "Timeout waiting for Xvfb to start" >&2
     cat "$TMP_DIR/xvfb.log" >&2
     exit 1
  fi
  if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "Xvfb died unexpectedly" >&2
    cat "$TMP_DIR/xvfb.log" >&2
    exit 1
  fi
done

DISPLAY_NUM=$(cat "$DISPLAY_FILE")
export DISPLAY=:${DISPLAY_NUM}

for _ in {1..50}; do
  if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 || { echo "Xvfb did not start"; cat "$TMP_DIR/xvfb.log" || true; exit 1; }

# NOTE: no `</dev/null` on these daemons (this is the reverted piece).
xsettingsd --config /dev/null >"$TMP_DIR/xsettingsd.log" 2>&1 &
XSETTINGS_PID=$!

openbox-session >"$TMP_DIR/openbox.log" 2>&1 &
OPENBOX_PID=$!

dbus-run-session -- "$@"
