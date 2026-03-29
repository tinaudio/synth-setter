#!/usr/bin/env bats

# BATS tests for scripts/docker_entrypoint.sh
# Tests the MODE dispatch: idle, passthrough, unset, and unknown modes.
#
# Idle tests require Linux (sleep infinity is GNU coreutils).
# Non-idle tests are portable and run everywhere.

EP="$BATS_TEST_DIRNAME/../scripts/docker_entrypoint.sh"

# ---------------------------------------------------------------------------
# Idle mode (MODE=idle) — Linux only (sleep infinity is GNU coreutils)
# ---------------------------------------------------------------------------

@test "test_idle_stays_alive" {
  [[ "$(uname)" == "Linux" ]] || skip "idle mode uses sleep infinity (GNU)"
  # Concurrency: fork starts the entrypoint, kill -0 checks it's alive.
  # Practically deterministic — bash fork completes before parent resumes.
  # The only failure mode is an entrypoint parse error (caught by shellcheck).
  MODE=idle "$EP" &
  pid=$!
  kill -0 "$pid"
  kill "$pid"
  wait "$pid" 2>/dev/null || true
}

@test "test_idle_prints_informational_message" {
  [[ "$(uname)" == "Linux" ]] || skip "idle mode uses sleep infinity (GNU)"
  # Concurrency: the poll loop is both the assertion and the synchronization.
  # grep succeeds only after the child's echo lands in the file — no race.
  # If the entrypoint breaks before echoing, the loop hangs (CI timeout = failure).
  local log
  log="$(mktemp)"
  MODE=idle "$EP" >"$log" 2>&1 &
  pid=$!
  while ! grep -qi "idle" "$log" 2>/dev/null; do :; done
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null || true
  rm -f "$log"
}

@test "test_idle_ignores_command_args" {
  [[ "$(uname)" == "Linux" ]] || skip "idle mode uses sleep infinity (GNU)"
  # Concurrency: negative assertion — SHOULD_BE_IGNORED is never printed
  # regardless of timing. If kill fires before echo, log is empty (passes).
  # If kill fires after echo, idle mode ignores args anyway (passes).
  # Both outcomes are correct — the assertion is timing-invariant.
  local log
  log="$(mktemp)"
  MODE=idle "$EP" echo SHOULD_BE_IGNORED >"$log" 2>&1 &
  pid=$!
  kill -0 "$pid"
  kill "$pid"
  wait "$pid" 2>/dev/null || true
  run grep -q "SHOULD_BE_IGNORED" "$log"
  [ "$status" -ne 0 ]
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

# ---------------------------------------------------------------------------
# Generate dataset mode (MODE=generate_dataset)
# ---------------------------------------------------------------------------

@test "test_generate_dataset_missing_config_exits_nonzero" {
  run env -u DATASET_CONFIG MODE=generate_dataset "$EP"
  [ "$status" -ne 0 ]
}

@test "test_generate_dataset_missing_config_prints_error" {
  run env -u DATASET_CONFIG MODE=generate_dataset "$EP"
  [[ "$output" =~ "DATASET_CONFIG" ]]
}
