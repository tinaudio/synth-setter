#!/usr/bin/env bash
#
# DEBUG variant — bisect canary for the `wait` piece of the wrapper cleanup.
# Identical to scripts/run-linux-vst-headless.sh except the cleanup trap
# does NOT call `wait` after the kills (and skips `pkill -P $$` too — once
# you skip wait, pkill races on PIDs that may not yet be reaped, so this
# variant is testing "fire-and-forget kill, no synchronous reap").
# Stdin-detach and exec-removal are kept.
#
# Used by the test-skypilot-debug matrix's `headless-no-wait` variant to
# prove that synchronous reaping is independently necessary: `kill PID` is
# async — bash returns from `kill` before the kernel finishes draining the
# child's exit. Without `wait`, the wrapper bash returns and the SSH
# command's pipes can be inherited by a not-yet-reaped child for a brief
# window; on RunPod that's enough to keep the SSH session's process tree
# nominally alive past the wrapper bash's exit, and SkyPilot doesn't see
# EOF until the kernel finishes the cleanup. With `wait`, the wrapper bash
# blocks until the children are fully reaped before returning.
#
# Expected matrix outcome: probably FAIL on a slow-reap day. If this passes
# consistently across many dispatches, `wait` was overkill and we can
# simplify the production wrapper.

set -euo pipefail

TMP_DIR=$(mktemp -d)
export XAUTHORITY="${XAUTHORITY:-$TMP_DIR/.Xauthority}"

export LIBGL_ALWAYS_SOFTWARE=1
export NO_AT_BRIDGE=1
export JUCE_USE_XINPUT2=0

DISPLAY_FILE="$TMP_DIR/display_num"

cleanup() {
  echo "[wrapper-no-wait] cleanup: starting (pid=$$)" >&2
  if [ -n "${OPENBOX_PID-}" ]; then kill "${OPENBOX_PID}" 2>/dev/null || true; fi
  if [ -n "${XSETTINGS_PID-}" ]; then kill "${XSETTINGS_PID}" 2>/dev/null || true; fi
  if [ -n "${XVFB_PID-}" ]; then kill "${XVFB_PID}" 2>/dev/null || true; fi
  # NOTE: no `wait` here (this is the reverted piece). Also no
  # `pkill -P $$` because once the synchronous reap is gone, pkill races
  # on PIDs that may already be in the kernel's reap queue.
  if [ -n "${TMP_DIR-}" ]; then rm -rf "${TMP_DIR}" || true; fi
  echo "[wrapper-no-wait] cleanup: done" >&2
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
