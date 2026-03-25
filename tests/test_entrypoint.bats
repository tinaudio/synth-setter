#!/usr/bin/env bats

# BATS tests for scripts/docker_entrypoint.sh
# Tests the MODE dispatch: idle, passthrough, unset, and unknown modes.

EP="$BATS_TEST_DIRNAME/../scripts/docker_entrypoint.sh"

# Wait for entrypoint to be ready. The echo proves the trap is installed
# (trap is set before echo in the entrypoint). No sleeps needed.
wait_for_ready() {
  local log="$1"
  while ! grep -q "Idle" "$log" 2>/dev/null; do :; done
}

# ---------------------------------------------------------------------------
# Idle mode (MODE=idle)
# ---------------------------------------------------------------------------

@test "test_idle_stays_alive" {
  local log
  log="$(mktemp)"
  MODE=idle "$EP" >"$log" 2>&1 &
  pid=$!
  wait_for_ready "$log"
  kill -0 "$pid"
  kill "$pid"
  wait "$pid" 2>/dev/null || true
  rm -f "$log"
}

@test "test_idle_prints_informational_message" {
  local log
  log="$(mktemp)"
  MODE=idle "$EP" >"$log" 2>&1 &
  pid=$!
  wait_for_ready "$log"
  grep -qi "idle" "$log"
  kill "$pid"
  wait "$pid" 2>/dev/null || true
  rm -f "$log"
}

@test "test_idle_ignores_command_args" {
  local log
  log="$(mktemp)"
  MODE=idle "$EP" echo SHOULD_BE_IGNORED >"$log" 2>&1 &
  pid=$!
  wait_for_ready "$log"
  run grep -q "SHOULD_BE_IGNORED" "$log"
  [ "$status" -ne 0 ]
  kill "$pid"
  wait "$pid" 2>/dev/null || true
  rm -f "$log"
}

@test "test_idle_exits_cleanly_on_sigterm" {
  local log
  log="$(mktemp)"
  MODE=idle "$EP" >"$log" 2>&1 &
  pid=$!
  wait_for_ready "$log"
  kill "$pid"
  wait "$pid"
  rm -f "$log"
}

# ---------------------------------------------------------------------------
# Passthrough mode (MODE=passthrough)
# ---------------------------------------------------------------------------

@test "test_passthrough_with_args_executes_command" {
  run env MODE=passthrough "$EP" echo hello
  [ "$status" -eq 0 ]
  [ "$output" = "hello" ]
}

@test "test_passthrough_with_args_forwards_exit_code" {
  run env MODE=passthrough "$EP" false
  [ "$status" -eq 1 ]
}

@test "test_passthrough_with_args_preserves_spaces" {
  run env MODE=passthrough "$EP" echo "hello world"
  [ "$status" -eq 0 ]
  [ "$output" = "hello world" ]
}

@test "test_passthrough_no_args_exits_zero" {
  run env MODE=passthrough "$EP"
  [ "$status" -eq 0 ]
}

@test "test_passthrough_no_args_prints_message" {
  run env MODE=passthrough "$EP"
  [ "$status" -eq 0 ]
  [ -n "$output" ]
}

# ---------------------------------------------------------------------------
# No MODE (unset or empty)
# ---------------------------------------------------------------------------

@test "test_no_mode_exits_nonzero" {
  run env -u MODE "$EP"
  [ "$status" -ne 0 ]
}

@test "test_no_mode_prints_error_to_stderr" {
  run env -u MODE "$EP"
  [[ "$output" =~ MODE ]]
}

@test "test_no_mode_does_not_execute_args" {
  run env -u MODE "$EP" echo SHOULD_NOT_RUN
  [[ ! "$output" =~ "SHOULD_NOT_RUN" ]]
}

@test "test_empty_mode_behaves_same_as_unset" {
  run env MODE="" "$EP"
  [ "$status" -ne 0 ]
}

# ---------------------------------------------------------------------------
# Unknown MODE
# ---------------------------------------------------------------------------

@test "test_unknown_mode_exits_nonzero" {
  run env MODE=bogus "$EP"
  [ "$status" -ne 0 ]
}

@test "test_unknown_mode_prints_error_with_mode_name" {
  run env MODE=bogus "$EP"
  [[ "$output" =~ "bogus" ]]
}

@test "test_unknown_mode_does_not_execute_args" {
  run env MODE=bogus "$EP" echo SHOULD_NOT_RUN
  [[ ! "$output" =~ "SHOULD_NOT_RUN" ]]
}
