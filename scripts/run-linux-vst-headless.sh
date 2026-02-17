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

#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=${DISPLAY:-:99}
export XAUTHORITY=${XAUTHORITY:-/tmp/.Xauthority}

export LIBGL_ALWAYS_SOFTWARE=1
export NO_AT_BRIDGE=1
export JUCE_USE_XINPUT2=0

# Start Xvfb
Xvfb "$DISPLAY" -screen 0 1920x1080x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
XVFB_PID=$!

cleanup() {
  kill "$OPENBOX_PID" "$XSETTINGS_PID" "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for X to be ready (avoid races)
for i in {1..50}; do
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
