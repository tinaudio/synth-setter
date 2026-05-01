#!/usr/bin/env bash
#
# DEBUG variant — bisect canary for the `pkill -P $$` orphan-grandchild
# sweep in the wrapper cleanup. Identical to scripts/run-linux-vst-headless.sh
# except the cleanup trap does NOT run `pkill -P $$` after `wait`. Stdin
# detach, exec removal, and `wait` are all kept.
#
# Used by the test-skypilot-debug matrix's `headless-no-pkill` variant.
# `wait` only reaps children the wrapper *tracked* (XVFB/XSETTINGS/OPENBOX);
# any grandchild forked by openbox or by dbus-launch (XDG autostart, e.g.)
# is reparented to PID 1 if its parent dies first, escaping `wait`. On
# RunPod those grandchildren can hold the SSH command's stdin/stdout open.
# `pkill -P $$` walks bash's children-of-self set and SIGTERMs anything
# still alive there. Together with `wait`, this guarantees no descendant
# the wrapper spawned is left holding the SSH pipes open after cleanup.
#
# Expected matrix outcome: PASS most of the time (the daemons we kill are
# already the dominant offenders), but FAIL whenever a grandchild
# reparented to init holds the pipe. Catches a future regression where a
# new daemon added to the wrapper forks a child without redirecting stdio.

set -euo pipefail

TMP_DIR=$(mktemp -d)
export XAUTHORITY="${XAUTHORITY:-$TMP_DIR/.Xauthority}"

export LIBGL_ALWAYS_SOFTWARE=1
export NO_AT_BRIDGE=1
export JUCE_USE_XINPUT2=0

DISPLAY_FILE="$TMP_DIR/display_num"

cleanup() {
  echo "[wrapper-no-pkill] cleanup: starting (pid=$$)" >&2
  if [ -n "${OPENBOX_PID-}" ]; then kill "${OPENBOX_PID}" 2>/dev/null || true; fi
  if [ -n "${XSETTINGS_PID-}" ]; then kill "${XSETTINGS_PID}" 2>/dev/null || true; fi
  if [ -n "${XVFB_PID-}" ]; then kill "${XVFB_PID}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  # NOTE: no `pkill -P $$` here (this is the reverted piece).
  if [ -n "${TMP_DIR-}" ]; then rm -rf "${TMP_DIR}" || true; fi
  echo "[wrapper-no-pkill] cleanup: done" >&2
}
trap cleanup EXIT

Xvfb -displayfd 3 -screen 0 1920x1080x24 -nolisten tcp \
  </dev/null >/dev/null 3>"$DISPLAY_FILE" 2>"$TMP_DIR/xvfb.log" &
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

xsettingsd --config /dev/null </dev/null >"$TMP_DIR/xsettingsd.log" 2>&1 &
XSETTINGS_PID=$!

openbox-session </dev/null >"$TMP_DIR/openbox.log" 2>&1 &
OPENBOX_PID=$!

dbus-run-session -- "$@"
