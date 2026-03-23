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

export XAUTHORITY=${XAUTHORITY:-/tmp/.Xauthority}

export LIBGL_ALWAYS_SOFTWARE=1
export NO_AT_BRIDGE=1
export JUCE_USE_XINPUT2=0

# Create temp dir for display number coordination
TMP_DIR=$(mktemp -d)
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

# Start Xvfb (let it pick a display)
# -displayfd 3 writes the chosen display number to file descriptor 3
# 3>"$DISPLAY_FILE" redirects FD 3 to our temp file so we can read it later
Xvfb -displayfd 3 -screen 0 1920x1080x24 -nolisten tcp 3>"$DISPLAY_FILE" 2>/tmp/xvfb.log &
XVFB_PID=$!

# Wait for Xvfb to output the display number
count=0
while [ ! -s "$DISPLAY_FILE" ]; do
  sleep 0.1
  count=$((count+1))
  if [ "$count" -ge 100 ]; then
     echo "Timeout waiting for Xvfb to start" >&2
     cat /tmp/xvfb.log >&2
     exit 1
  fi
  # Check if Xvfb died
  if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "Xvfb died unexpectedly" >&2
    cat /tmp/xvfb.log >&2
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
xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 || { echo "Xvfb did not start"; cat /tmp/xvfb.log || true; exit 1; }

# Start an XSettings manager
xsettingsd --config /dev/null >/tmp/xsettingsd.log 2>&1 &
XSETTINGS_PID=$!

# Start a lightweight window manager (important for many plugins)
openbox-session >/tmp/openbox.log 2>&1 &
OPENBOX_PID=$!

# Run the actual command inside a D-Bus session
exec dbus-run-session -- "$@"
