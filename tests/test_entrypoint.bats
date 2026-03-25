#!/usr/bin/env bats

# BATS tests for scripts/docker_entrypoint.sh
# Tests the MODE dispatch: idle, passthrough, unset, and unknown modes.

EP="$BATS_TEST_DIRNAME/../scripts/docker_entrypoint.sh"

# ---------------------------------------------------------------------------
# Idle mode (MODE=idle)
# ---------------------------------------------------------------------------

@test "test_idle_starts_sleep_process" {
  # Create a mock sleep that records it was called then exits.
  local mock_dir
  mock_dir="$(mktemp -d)"
  cat > "$mock_dir/sleep" <<'MOCK'
#!/usr/bin/env bash
echo "sleep called with: $*" > "$MOCK_LOG"
exit 0
MOCK
  chmod +x "$mock_dir/sleep"

  local mock_log
  mock_log="$(mktemp)"
  MOCK_LOG="$mock_log" PATH="$mock_dir:$PATH" MODE=idle "$EP" || true
  grep -q "sleep" "$mock_log"
  rm -rf "$mock_dir" "$mock_log"
}

@test "test_idle_prints_informational_message" {
  # exec replaces the process so we can't capture its stdout directly.
  # Instead, run in a subshell that captures the echo before exec.
  local out
  out=$(MODE=idle "$EP" 2>&1 &
    pid=$!
    sleep 0.5
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  )
  [[ "$out" =~ [Ii]dle ]]
}

@test "test_idle_ignores_command_args" {
  local out
  out=$(MODE=idle "$EP" echo hello 2>&1 &
    pid=$!
    sleep 0.5
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  )
  [[ ! "$out" =~ "hello" ]]
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
  run env -u MODE "$EP" echo hello
  [[ ! "$output" =~ "hello" ]]
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
  run env MODE=bogus "$EP" echo hello
  [[ ! "$output" =~ "hello" ]]
}
