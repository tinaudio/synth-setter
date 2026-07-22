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

TMP_DIR=""
DISPLAY_FILE=""
readonly XVFB_STARTUP_PROBES=100

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

# Concurrent bootstraps can lose the display-lock race or miss the readiness
# window, so retry instead of failing the renderer (#2035).
# Globals:
#   Reads the environment variable named by the first argument.
# Arguments:
#   Variable name, default value, and inclusive maximum.
# Outputs:
#   Normalized decimal value to stdout; clamp diagnostics to stderr.
normalize_decimal() {
  local name=$1
  local default=$2
  local maximum=$3
  local value=${!name:-$default}

  case "$value" in
    *[!0-9]*)
      printf '%s\n' "$default"
      return
      ;;
  esac
  while [[ "$value" == 0* ]]; do
    value=${value#0}
  done
  value=${value:-0}
  if (( ${#value} > ${#maximum} )) ||
    [[ ${#value} -eq ${#maximum} && "$value" > "$maximum" ]]; then
    echo "[wrapper] clamping ${name} to ${maximum}" >&2
    printf '%s\n' "$maximum"
    return
  fi
  printf '%s\n' "$value"
}

# Normalize a positive decimal, falling back when the override represents zero.
# Arguments:
#   Variable name, default value, and inclusive maximum.
# Outputs:
#   Normalized positive decimal value to stdout; clamp diagnostics to stderr.
normalize_positive_decimal() {
  local default=$2
  local value
  value=$(normalize_decimal "$1" "$default" "$3")

  if (( value == 0 )); then
    printf '%s\n' "$default"
    return
  fi
  printf '%s\n' "$value"
}

# Start Xvfb and confirm the display it published via -displayfd.
# Globals:
#   DISPLAY_FILE, DISPLAY, TMP_DIR, XVFB_PID, XVFB_READY_PROBES,
#   XVFB_STARTUP_PROBES.
# Outputs:
#   Startup diagnostics to stderr.
# Returns:
#   0 once Xvfb has published a display; 1 only when Xvfb fails to start or dies.
start_xvfb_attempt() {
  : > "$DISPLAY_FILE"
  # -displayfd 3 writes the chosen display number to file descriptor 3;
  # 3>"$DISPLAY_FILE" redirects FD 3 to our temp file so we can read it later
  Xvfb -displayfd 3 -screen 0 1920x1080x24 -nolisten tcp \
    </dev/null >/dev/null 3>"$DISPLAY_FILE" 2>"$TMP_DIR/xvfb.log" &
  XVFB_PID=$!

  # The display-number write has its own fixed limit before readiness probes.
  local count=0
  while [[ ! -s "$DISPLAY_FILE" ]]; do
    if ! kill -0 "$XVFB_PID" 2>/dev/null; then
      echo "Xvfb died unexpectedly" >&2
      return 1
    fi
    count=$((count+1))
    if (( count >= XVFB_STARTUP_PROBES )); then
      echo "Timeout waiting for Xvfb to start" >&2
      return 1
    fi
    sleep 0.1
  done

  DISPLAY=":$(cat "$DISPLAY_FILE")"
  export DISPLAY

  # -displayfd publishes the display number only once the X server is listening
  # and ready to accept connections, so DISPLAY is already usable here. xdpyinfo
  # (from x11-utils) is a best-effort early confirmation: the bare CI runner
  # ships no x11-utils, and concurrent bootstraps can transiently refuse the
  # probe, so a missing or unconfirmed probe must not fail a display that
  # -displayfd already declared ready (#2320). A dead server is still caught.
  if ! command -v xdpyinfo >/dev/null 2>&1; then
    return 0
  fi
  local probe
  for ((probe = 0; probe < XVFB_READY_PROBES; probe += 1)); do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$XVFB_PID" 2>/dev/null; then
      echo "Xvfb died before $DISPLAY became ready" >&2
      return 1
    fi
    sleep 0.1
  done
  echo "[wrapper] xdpyinfo did not confirm $DISPLAY within" \
    "${XVFB_READY_PROBES} probes; trusting -displayfd readiness" >&2
  return 0
}

# Reap a failed Xvfb so retries do not stack servers.
# Globals:
#   TMP_DIR, XVFB_PID.
# Outputs:
#   The failed Xvfb log to stderr.
reap_failed_xvfb() {
  kill "$XVFB_PID" 2>/dev/null || true
  wait "$XVFB_PID" 2>/dev/null || true
  XVFB_PID=""
  cat "$TMP_DIR/xvfb.log" >&2 || true
}

# Bootstrap X11 and run the requested command in its D-Bus session.
# Globals:
#   Modifies DISPLAY_FILE, TMP_DIR, OPENBOX_PID, XSETTINGS_PID, and Xvfb
#   retry settings; exports X11 compatibility variables.
# Arguments:
#   Command and arguments to execute.
# Outputs:
#   Bootstrap and cleanup diagnostics to stderr; command output unchanged.
# Returns:
#   The wrapped command's exit status, or non-zero when Xvfb bootstrap fails.
main() {
  TMP_DIR=$(mktemp -d)
  DISPLAY_FILE="$TMP_DIR/display_num"
  export XAUTHORITY="${XAUTHORITY:-$TMP_DIR/.Xauthority}"
  export LIBGL_ALWAYS_SOFTWARE=1
  export NO_AT_BRIDGE=1
  export JUCE_USE_XINPUT2=0
  trap cleanup EXIT

  XVFB_BOOTSTRAP_ATTEMPTS=$(normalize_positive_decimal \
    XVFB_BOOTSTRAP_ATTEMPTS 3 9)
  XVFB_READY_PROBES=$(normalize_positive_decimal XVFB_READY_PROBES 50 100)
  XVFB_RETRY_JITTER_MAX=$(normalize_decimal XVFB_RETRY_JITTER_MAX 9 9)
  readonly XVFB_BOOTSTRAP_ATTEMPTS XVFB_READY_PROBES XVFB_RETRY_JITTER_MAX

  local attempt=1
  until start_xvfb_attempt; do
    reap_failed_xvfb
    if (( attempt >= XVFB_BOOTSTRAP_ATTEMPTS )); then
      echo "Xvfb bootstrap failed after ${attempt} attempt(s)" >&2
      return 1
    fi
    attempt=$((attempt+1))
    echo "[wrapper] retrying Xvfb bootstrap" \
      "(attempt ${attempt}/${XVFB_BOOTSTRAP_ATTEMPTS})" >&2
    # Jitter decorrelates concurrent bootstraps racing the same display locks.
    if (( XVFB_RETRY_JITTER_MAX > 0 )); then
      sleep "0.$((RANDOM % XVFB_RETRY_JITTER_MAX + 1))"
    fi
  done

  # Detached stdin prevents daemons from keeping SkyPilot's SSH pipe open (#735).
  xsettingsd --config /dev/null </dev/null >"$TMP_DIR/xsettingsd.log" 2>&1 &
  XSETTINGS_PID=$!

  openbox-session </dev/null >"$TMP_DIR/openbox.log" 2>&1 &
  OPENBOX_PID=$!

  # exec would replace bash and bypass the cleanup trap (#735).
  dbus-run-session -- "$@"
}

main "$@"
