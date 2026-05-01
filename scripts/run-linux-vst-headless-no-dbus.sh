#!/usr/bin/env bash
#
# Diagnostic variant 4 of run-linux-vst-headless.sh: identical to the
# original wrapper, except the final invocation is `exec "$@"` (no
# dbus-run-session). The X server and XSettings/openbox children are
# still spawned, but no dbus session bus is created.
#
# Hypothesis: dbus-run-session itself spawns a daemon that outlives the
# wrapped command and prevents SkyPilot from seeing the worker exit on
# RunPod. If variant 3 (no-exec) still hangs but this one succeeds,
# dbus-run-session is the culprit and the trap question is a red
# herring.
#
# Caveat: VST3 plugins that rely on D-Bus session APIs may misbehave
# without a session bus; this variant is for diagnostic bisection only.

set -euo pipefail

TMP_DIR=$(mktemp -d)
export XAUTHORITY="${XAUTHORITY:-$TMP_DIR/.Xauthority}"

export LIBGL_ALWAYS_SOFTWARE=1
export NO_AT_BRIDGE=1
export JUCE_USE_XINPUT2=0

DISPLAY_FILE="$TMP_DIR/display_num"

cleanup() {
  if [ -n "${OPENBOX_PID-}" ]; then
    kill "${OPENBOX_PID}" 2>/dev/null || true
  fi
  if [ -n "${XSETTINGS_PID-}" ]; then
    kill "${XSETTINGS_PID}" 2>/dev/null || true
  fi
  if [ -n "${XVFB_PID-}" ]; then
    kill "${XVFB_PID}" 2>/dev/null || true
  fi
  if [ -n "${TMP_DIR-}" ]; then
    rm -rf "${TMP_DIR}" || true
  fi
}
trap cleanup EXIT

Xvfb -displayfd 3 -screen 0 1920x1080x24 -nolisten tcp 3>"$DISPLAY_FILE" 2>"$TMP_DIR/xvfb.log" &
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

xsettingsd --config /dev/null >"$TMP_DIR/xsettingsd.log" 2>&1 &
XSETTINGS_PID=$!

openbox-session >"$TMP_DIR/openbox.log" 2>&1 &
OPENBOX_PID=$!

# Variant-4 difference: no dbus-run-session.
exec "$@"
