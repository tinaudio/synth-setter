#!/usr/bin/env bash
#
# Headless VST3 bootstrap for Linux CI
#
# Some VST3 pluginsn query X11 desktop
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

export DISPLAY=${DISPLAY:-:99}
export XAUTHORITY=${XAUTHORITY:-/tmp/.Xauthority}

# Common envs for headless plugin GUIs
export LIBGL_ALWAYS_SOFTWARE=1
export NO_AT_BRIDGE=1
export JUCE_USE_XINPUT2=0

# Start Xvfb
Xvfb "$DISPLAY" -screen 0 1920x1080x24 -nolisten tcp -ac >/tmp/xvfb.log 2>&1 &
XVFB_PID=$!

# Start an XSettings manager (this is the key fix)
xsettingsd --config /dev/null >/tmp/xsettingsd.log 2>&1 &
XSETTINGS_PID=$!

cleanup() {
  kill "$XSETTINGS_PID" "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Run the actual command inside a D-Bus session
exec dbus-run-session -- "$@"
