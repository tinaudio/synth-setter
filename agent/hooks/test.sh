#!/usr/bin/env bash
# test.sh — unit harness for agent/hooks/*. Runs from repo root:
#   bash agent/hooks/test.sh
# Each test is a function (`T_*`) registered with `it`. The runner executes
# each in a subshell so env-var sets, cwd changes, and `set -e` semantics
# don't leak between tests. The PATH-stubbed `claude`/`gh` live in $STUBS;
# `git` is real (the sandbox is a fresh init repo).
#
# edit-write.sh — credential-protect coverage lives here; the matching
# `format` and `test` modes are best-effort hooks that aren't easily
# stubbed in a sandbox (they shell out to ruff/pytest), so only the
# dispatch contract is tested here. The Python suite at
# `tests/claude_hooks/test_settings_hooks.py` covers credential-protect
# from the settings.json wiring perspective.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

TEST_DIR=$(mktemp -d)
cleanup() { rm -rf "$TEST_DIR"; }
trap cleanup EXIT

SANDBOX="$TEST_DIR/repo"
mkdir -p "$SANDBOX/agent/hooks"
cp agent/hooks/_lib.sh \
   agent/hooks/doc-drift.sh \
   agent/hooks/pr-review-resolver.sh \
   agent/hooks/pre-pr-review-gate.sh \
   agent/hooks/edit-write.sh \
   agent/hooks/verify-gh-taxonomy.sh \
   "$SANDBOX/agent/hooks/"
cd "$SANDBOX"
git init -q
git config user.email test@test
git config user.name test
git commit -q --allow-empty -m init

STUBS="$TEST_DIR/stubs"
mkdir -p "$STUBS"
cat > "$STUBS/claude" <<'EOF'
#!/usr/bin/env bash
if [[ "${AGENT_STUB_FAIL:-0}" == "1" ]]; then
  echo "simulated headless agent auth failure" >&2
  exit 3
fi
echo "# STUB headless agent output"
echo "# prompt was: $*"
EOF
cat > "$STUBS/gh" <<'EOF'
#!/usr/bin/env bash
# Minimal stub. $GH_STUB_PR governs `gh pr view`'s number output.
if [[ "$1" == "pr" && "$2" == "view" ]]; then
  if [[ -n "${GH_STUB_PR:-}" ]]; then
    echo "${GH_STUB_PR}"
    exit 0
  fi
  echo "no PR" >&2
  exit 1
fi
exit 0
EOF
chmod +x "$STUBS/claude" "$STUBS/gh"
export PATH="$STUBS:$PATH"

# ---------------------------------------------------------------------------
# Test registry + runner
# ---------------------------------------------------------------------------
TESTS=()

it() {
  # Usage: it "<description>" <T_function_name>
  TESTS+=("$1|$2")
}

reset_sandbox() {
  rm -rf .agent-reviews
}

run_tests() {
  local pass=0 fail=0 entry desc fn out exit_code
  for entry in "${TESTS[@]}"; do
    desc="${entry%%|*}"
    fn="${entry#*|}"
    # Subshell isolates env, cwd, and exported var changes between tests.
    # Disable the outer `set -e` around the capture so a failing test doesn't
    # terminate the runner — we want to print FAIL and keep going.
    set +e
    out=$( ( set -e; reset_sandbox; "$fn" ) 2>&1 )
    exit_code=$?
    set -e
    if [[ "$exit_code" == "0" ]]; then
      printf '  PASS  %s\n' "$desc"
      pass=$((pass + 1))
    else
      printf '  FAIL  %s (exit %s)\n' "$desc" "$exit_code"
      printf '%s\n' "$out" | sed 's/^/         /'
      fail=$((fail + 1))
    fi
  done
  echo
  echo "== Summary =="
  echo "PASS: $pass"
  echo "FAIL: $fail"
  [[ "$fail" -eq 0 ]]
}

assert() {
  # Usage: assert "<msg>" <test-expr...>
  # Example: assert "expected EXIT:0" [[ "$out" == *"EXIT:0"* ]]
  # Caller passes the test as a positional expr that's eval'd via `if`.
  local msg="$1"; shift
  if "$@"; then return 0; fi
  echo "ASSERT FAILED: ${msg}" >&2
  return 1
}

# `last_exit_line` extracts the trailing `EXIT:N` marker the test bodies
# append to captured `out` so we can match on exact equality rather than
# substring (which would false-pass if stderr contained `EXIT:0`).
last_exit_line() {
  printf '%s\n' "$1" | tail -1
}

# ===========================================================================
# doc-drift.sh
# ===========================================================================

T_doc_drift_no_match() {
  local out
  out=$(echo '{"tool_input":{"command":"echo hello"}}' | bash agent/hooks/doc-drift.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ ! -d .agent-reviews ]] || { echo ".agent-reviews should not exist"; return 1; }
}
it "doc-drift: non-matching command exits 0 silently" T_doc_drift_no_match

T_doc_drift_quoted_substring() {
  local out
  export GH_STUB_PR=42
  out=$(echo '{"tool_input":{"command":"echo testing the gh pr create matcher"}}' | bash agent/hooks/doc-drift.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ ! -d .agent-reviews ]] || { echo ".agent-reviews should not exist"; return 1; }
}
it "doc-drift: quoted 'gh pr create' substring inside echo does NOT trigger" T_doc_drift_quoted_substring

T_doc_drift_no_pr() {
  local out
  unset GH_STUB_PR
  out=$(echo '{"tool_input":{"command":"gh pr create --title x"}}' | bash agent/hooks/doc-drift.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  if compgen -G ".agent-reviews/doc-drift-*.md" >/dev/null; then
    echo "no report should exist"
    return 1
  fi
}
it "doc-drift: gh pr create with no PR found → exit 0, no report" T_doc_drift_no_pr

T_doc_drift_with_pr() {
  local out stderr_file report
  export GH_STUB_PR=42 DOC_DRIFT_DRY_RUN=1
  stderr_file="$TEST_DIR/stderr.txt"
  out=$(echo '{"tool_input":{"command":"gh pr create --title x"}}' | bash agent/hooks/doc-drift.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  report=$(find .agent-reviews -maxdepth 1 -name 'doc-drift-*.md' 2>/dev/null | head -1)
  [[ -n "$report" ]] || { echo "report file missing"; return 1; }
  grep -q "PR #42" "$stderr_file" || { echo "stderr should mention PR #42: $(cat "$stderr_file")"; return 1; }
  grep -q "docs/doc-map.yaml" "$report" || { echo "report should reference doc-map.yaml: $(cat "$report")"; return 1; }
}
it "doc-drift: gh pr create with PR → exit 2, report written, stderr points to it" T_doc_drift_with_pr

T_doc_drift_agent_failure() {
  local out report
  export GH_STUB_PR=42 AGENT_STUB_FAIL=1
  out=$(echo '{"tool_input":{"command":"gh pr create --title x"}}' | bash agent/hooks/doc-drift.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  report=$(find .agent-reviews -maxdepth 1 -name 'doc-drift-*.md' 2>/dev/null | head -1)
  [[ -n "$report" ]] || { echo "report missing"; return 1; }
  grep -q "FAILED" "$report" || { echo "report should say FAILED"; return 1; }
  grep -q "simulated headless agent auth failure" "$report" || { echo "report should include stderr tail"; return 1; }
  # Regression: the captured exit code MUST be the stub's real exit (3), not 0.
  # An earlier refactor of run_review used `$?` outside an else branch, which
  # bash resets to 0 after a falsy `if; then; fi`, silently corrupting the
  # failure report. Catch any regression by asserting the exact exit code.
  grep -qE '^## headless agent exit code$' "$report" || { echo "report missing exit-code header"; return 1; }
  awk '/^## headless agent exit code$/{getline; print; exit}' "$report" | grep -qE '^3$' || {
    echo "expected captured exit code 3 in report, got:"
    awk '/^## headless agent exit code$/{getline; print; exit}' "$report"
    return 1
  }
}
it "doc-drift: headless agent failure → verbose FAILED report (exit code + stderr captured)" T_doc_drift_agent_failure

# ===========================================================================
# pr-review-resolver.sh
# ===========================================================================

T_resolver_no_match() {
  local out
  export RESOLVER_SLEEP_SECS=1 RESOLVER_DRY_RUN=1
  out=$(echo '{"tool_input":{"command":"git commit -m x"}}' | bash agent/hooks/pr-review-resolver.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ ! -d .agent-reviews ]] || { echo ".agent-reviews should not exist"; return 1; }
}
it "pr-review-resolver: non-matching command exits 0 silently" T_resolver_no_match

T_resolver_quoted_substring() {
  local out
  export RESOLVER_SLEEP_SECS=1 RESOLVER_DRY_RUN=1
  out=$(echo '{"tool_input":{"command":"git commit -m \"fix git push bug\""}}' | bash agent/hooks/pr-review-resolver.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ ! -d .agent-reviews ]] || { echo ".agent-reviews should not exist"; return 1; }
}
it "pr-review-resolver: quoted 'git push' substring inside commit message does NOT trigger" T_resolver_quoted_substring

T_resolver_main_branch() {
  local out start elapsed
  export RESOLVER_SLEEP_SECS=5 RESOLVER_DRY_RUN=1
  git checkout -q -b main 2>/dev/null || git checkout -q main
  start=$(date +%s)
  out=$(echo '{"tool_input":{"command":"git push origin main"}}' | bash agent/hooks/pr-review-resolver.sh 2>&1; echo "EXIT:$?")
  elapsed=$(($(date +%s) - start))
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ "$elapsed" -lt 2 ]] || { echo "main-branch push should not have slept (elapsed=${elapsed}s)"; return 1; }
  [[ ! -d .agent-reviews ]] || { echo ".agent-reviews should not exist"; return 1; }
}
it "pr-review-resolver: git push on main → exits 0 before sleeping" T_resolver_main_branch

T_resolver_no_pr() {
  local out
  export RESOLVER_SLEEP_SECS=1 RESOLVER_DRY_RUN=1
  unset GH_STUB_PR
  git checkout -q -b feature-x 2>/dev/null || git checkout -q feature-x
  out=$(echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  if compgen -G ".agent-reviews/pr-review-resolver-*.md" >/dev/null; then
    echo "no report should exist"
    return 1
  fi
}
it "pr-review-resolver: feature branch push with no PR → exits 0, no report" T_resolver_no_pr

T_resolver_with_pr() {
  local out stderr_file report
  export RESOLVER_SLEEP_SECS=1 RESOLVER_DRY_RUN=1 GH_STUB_PR=99
  git checkout -q -b feature-x 2>/dev/null || git checkout -q feature-x
  stderr_file="$TEST_DIR/stderr.txt"
  out=$(echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  report=$(find .agent-reviews -maxdepth 1 -name 'pr-review-resolver-*.md' 2>/dev/null | head -1)
  [[ -n "$report" ]] || { echo "report missing"; return 1; }
  grep -q "PR #99" "$stderr_file" || { echo "stderr should mention PR #99"; return 1; }
  grep -q "gh api repos" "$report" || { echo "fallback report should mention gh api repos"; return 1; }
}
it "pr-review-resolver: feature branch push with PR → exit 2, report written, fallback prompt content" T_resolver_with_pr

T_resolver_lockfile_dedupe() {
  local bg_pid bg_exit
  export RESOLVER_SLEEP_SECS=3 RESOLVER_DRY_RUN=1 GH_STUB_PR=99
  git checkout -q -b feature-x 2>/dev/null || git checkout -q feature-x
  ( echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh >/dev/null 2>&1; echo $? > "$TEST_DIR/bg_exit" ) &
  bg_pid=$!
  sleep 1
  mkdir -p .agent-reviews
  echo "intruder-token" > .agent-reviews/.resolver-feature-x.lock
  wait "$bg_pid"
  bg_exit=$(cat "$TEST_DIR/bg_exit")
  [[ "$bg_exit" == "0" ]] || { echo "superseded run should exit 0, got $bg_exit"; return 1; }
  if compgen -G ".agent-reviews/pr-review-resolver-*.md" >/dev/null; then
    echo "superseded run should not write a report"
    return 1
  fi
}
it "pr-review-resolver: lockfile dedupe — superseded run exits 0 with no report" T_resolver_lockfile_dedupe

T_resolver_invalid_sleep_secs() {
  local out
  export RESOLVER_SLEEP_SECS=abc RESOLVER_DRY_RUN=1 GH_STUB_PR=99
  git checkout -q -b feature-x 2>/dev/null || git checkout -q feature-x
  # An invalid RESOLVER_SLEEP_SECS used to abort at `sleep` under set -e.
  # The hook must fall back to the default; we keep a tiny override via
  # the existence of GH_STUB_PR to avoid waiting the full 360s — instead
  # we only assert that the hook starts and gets past the validation.
  # Run with a 4s timeout; if it took >4s the validation didn't engage.
  timeout 4 bash -c 'echo "{\"tool_input\":{\"command\":\"git push\"}}" | bash agent/hooks/pr-review-resolver.sh' >/dev/null 2>"$TEST_DIR/stderr.txt" &
  local hook_pid=$!
  sleep 2
  # The hook should still be running (sleeping its 360s default) after 2s
  # but NOT exited from a `sleep abc` set -e abort.
  if kill -0 "$hook_pid" 2>/dev/null; then
    kill "$hook_pid" 2>/dev/null || true
    wait "$hook_pid" 2>/dev/null || true
    # Check the log captured the validation message.
    grep -q "invalid RESOLVER_SLEEP_SECS=abc" .agent-reviews/.hook.log 2>/dev/null || {
      echo "expected validation log; .hook.log: $(cat .agent-reviews/.hook.log 2>/dev/null || echo missing)"
      return 1
    }
  else
    wait "$hook_pid" 2>/dev/null || true
    echo "hook exited early — RESOLVER_SLEEP_SECS=abc validation did not engage"; return 1
  fi
}
it "pr-review-resolver: invalid RESOLVER_SLEEP_SECS falls back to default (not set -e abort)" T_resolver_invalid_sleep_secs

# ===========================================================================
# pre-pr-review-gate.sh
# ===========================================================================

T_gate_non_pr_create_falls_through() {
  local out
  out=$(echo '{"tool_input":{"command":"git rev-parse HEAD"}}' | bash agent/hooks/pre-pr-review-gate.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ "$out" != *"BLOCKED"* ]] || { echo "should not have BLOCKED a non-gh-pr-create command"; return 1; }
}
it "pre-pr-review-gate: non-gh-pr-create command falls through with exit 0" T_gate_non_pr_create_falls_through

T_gate_quoted_substring() {
  local out
  out=$(echo '{"tool_input":{"command":"echo testing the gh pr create matcher"}}' | bash agent/hooks/pre-pr-review-gate.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ "$out" != *"BLOCKED"* ]] || { echo "quoted substring should not trigger gate"; return 1; }
}
it "pre-pr-review-gate: quoted 'gh pr create' substring inside echo does NOT trigger the gate" T_gate_quoted_substring

T_gate_no_token_blocks() {
  local out stderr_file
  stderr_file="$TEST_DIR/gate_stderr.txt"
  out=$(echo '{"tool_input":{"command":"gh pr create --title x --body y"}}' | bash agent/hooks/pre-pr-review-gate.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  grep -q "BLOCKED" "$stderr_file" || { echo "stderr should contain BLOCKED"; return 1; }
  grep -q "REVIEW_FULL_DONE=1" "$stderr_file" || { echo "stderr should reference the token name"; return 1; }
}
it "pre-pr-review-gate: gh pr create without token → exit 2 with BLOCKED + token name on stderr" T_gate_no_token_blocks

T_gate_with_token_passes() {
  local out
  out=$(echo '{"tool_input":{"command":"gh pr create --title x --body y  # REVIEW_FULL_DONE=1"}}' | bash agent/hooks/pre-pr-review-gate.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ "$out" != *"BLOCKED"* ]] || { echo "should not have BLOCKED"; return 1; }
}
it "pre-pr-review-gate: gh pr create with REVIEW_FULL_DONE=1 trailing comment → exit 0" T_gate_with_token_passes

T_gate_leading_whitespace_still_blocks() {
  local out stderr_file
  stderr_file="$TEST_DIR/gate_stderr.txt"
  out=$(echo '{"tool_input":{"command":"  gh pr create --title x --body y"}}' | bash agent/hooks/pre-pr-review-gate.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  grep -q "BLOCKED" "$stderr_file" || { echo "leading-whitespace path should still BLOCK"; return 1; }
}
it "pre-pr-review-gate: leading-whitespace gh pr create without token → still blocked" T_gate_leading_whitespace_still_blocks

# ===========================================================================
# edit-write.sh — credential-protect + Unknown mode dispatch
# ===========================================================================

T_edit_write_credential_blocks_env() {
  local stderr_file rc
  stderr_file="$TEST_DIR/ew_stderr.txt"
  set +e
  echo '{"tool_input":{"file_path":".env.local"}}' \
    | bash agent/hooks/edit-write.sh credential-protect 2>"$stderr_file"
  rc=$?
  set -e
  [[ "$rc" == "1" ]] || { echo "expected exit 1, got $rc"; return 1; }
  grep -q "BLOCKED" "$stderr_file" || { echo "stderr should contain BLOCKED"; return 1; }
}
it "edit-write: credential-protect blocks .env.local with exit 1 + BLOCKED stderr" T_edit_write_credential_blocks_env

T_edit_write_credential_allows_py() {
  local rc
  set +e
  echo '{"tool_input":{"file_path":"src/foo.py"}}' \
    | bash agent/hooks/edit-write.sh credential-protect >/dev/null 2>&1
  rc=$?
  set -e
  [[ "$rc" == "0" ]] || { echo "expected exit 0 for src/foo.py, got $rc"; return 1; }
}
it "edit-write: credential-protect allows non-secret paths" T_edit_write_credential_allows_py

T_edit_write_unknown_mode_exits_2() {
  local stderr_file rc
  stderr_file="$TEST_DIR/ew_stderr.txt"
  set +e
  echo '{}' | bash agent/hooks/edit-write.sh bogus-mode 2>"$stderr_file"
  rc=$?
  set -e
  [[ "$rc" == "2" ]] || { echo "expected exit 2 for unknown mode, got $rc"; return 1; }
  grep -q "Unknown" "$stderr_file" || { echo "stderr should mention Unknown mode"; return 1; }
}
it "edit-write: unknown mode → exit 2 with 'Unknown' on stderr" T_edit_write_unknown_mode_exits_2

T_edit_write_format_non_fatal_on_missing_tool() {
  local rc
  # Add a no-op pytest/ruff to the stubs so neither is missing; the assertion
  # is that the hook exits 0 either way, not that any tool ran.
  set +e
  echo '{"tool_input":{"file_path":"src/foo.py"}}' \
    | bash agent/hooks/edit-write.sh format >/dev/null 2>&1
  rc=$?
  set -e
  [[ "$rc" == "0" ]] || { echo "format mode should exit 0, got $rc"; return 1; }
}
it "edit-write: format mode exits 0 (best-effort under set -e)" T_edit_write_format_non_fatal_on_missing_tool

T_edit_write_test_mode_finds_mirrored_layout() {
  # Sandbox: create both layouts and confirm the script picks the mirrored
  # one for a src/<pkg>/sub/file.py edit. Use a per-test scratch dir so test
  # state doesn't leak between cases.
  local scratch
  scratch=$(mktemp -d "$TEST_DIR/scratch-XXXX")
  mkdir -p "$scratch/src/synth_setter/pipeline/data" "$scratch/tests/pipeline/data"
  : > "$scratch/src/synth_setter/pipeline/data/stats.py"
  : > "$scratch/tests/pipeline/data/test_stats.py"
  # Stub pytest so the hook can `Running ...` then run something fast.
  cat > "$scratch/pytest" <<'PYEOF'
#!/usr/bin/env bash
exit 0
PYEOF
  chmod +x "$scratch/pytest"
  local out
  out=$(cd "$scratch" && echo '{"tool_input":{"file_path":"src/synth_setter/pipeline/data/stats.py"}}' \
    | env PATH="$scratch:$PATH" bash "$REPO_ROOT/agent/hooks/edit-write.sh" test 2>&1)
  rm -rf "$scratch"
  [[ "$out" == *"Running tests/pipeline/data/test_stats.py"* ]] || {
    echo "expected mirrored-layout test, got: $out"
    return 1
  }
}
it "edit-write: test mode finds mirrored layout tests/<pkg>/test_<base>.py first" T_edit_write_test_mode_finds_mirrored_layout

T_edit_write_test_mode_falls_back_to_flat_layout() {
  local scratch
  scratch=$(mktemp -d "$TEST_DIR/scratch-XXXX")
  mkdir -p "$scratch/src/synth_setter" "$scratch/tests"
  : > "$scratch/src/synth_setter/foo.py"
  : > "$scratch/tests/test_foo.py"
  cat > "$scratch/pytest" <<'PYEOF'
#!/usr/bin/env bash
exit 0
PYEOF
  chmod +x "$scratch/pytest"
  local out
  out=$(cd "$scratch" && echo '{"tool_input":{"file_path":"src/synth_setter/foo.py"}}' \
    | env PATH="$scratch:$PATH" bash "$REPO_ROOT/agent/hooks/edit-write.sh" test 2>&1)
  rm -rf "$scratch"
  [[ "$out" == *"Running tests/test_foo.py"* ]] || {
    echo "expected flat-layout fallback, got: $out"
    return 1
  }
}
it "edit-write: test mode falls back to flat layout tests/test_<base>.py when mirror missing" T_edit_write_test_mode_falls_back_to_flat_layout

# ===========================================================================
# verify-gh-taxonomy.sh — smoke tests for early-exit paths
# (mode_pr / mode_issue / mode_hierarchy require full gh-API stubbing; those
# go in a follow-up.)
# ===========================================================================

T_taxonomy_non_matching_command_exits_silently() {
  local out
  out=$(echo '{"tool_input":{"command":"echo hello"},"tool_response":""}' \
    | bash agent/hooks/verify-gh-taxonomy.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  # Should not emit JSON on stdout for a no-op.
  local without_exit
  without_exit="${out%EXIT:*}"
  [[ -z "${without_exit//[[:space:]]/}" ]] || { echo "expected silent stdout, got: $without_exit"; return 1; }
}
it "verify-gh-taxonomy: non-matching command exits 0 silently with no JSON output" T_taxonomy_non_matching_command_exits_silently

T_taxonomy_non_synth_setter_url_exits_silently() {
  local out
  out=$(echo '{"tool_input":{"command":"gh pr create --title x"},"tool_response":"https://github.com/other-owner/other-repo/pull/42"}' \
    | bash agent/hooks/verify-gh-taxonomy.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  local without_exit
  without_exit="${out%EXIT:*}"
  [[ -z "${without_exit//[[:space:]]/}" ]] || { echo "expected silent stdout for non-synth-setter PR, got: $without_exit"; return 1; }
}
it "verify-gh-taxonomy: gh pr create against non-synth-setter URL exits 0 silently" T_taxonomy_non_synth_setter_url_exits_silently

T_taxonomy_pr_no_url_exits_silently() {
  local out
  out=$(echo '{"tool_input":{"command":"gh pr create --title x"},"tool_response":"no URL here"}' \
    | bash agent/hooks/verify-gh-taxonomy.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
}
it "verify-gh-taxonomy: gh pr create with no URL in response exits 0 silently" T_taxonomy_pr_no_url_exits_silently

T_taxonomy_hierarchy_non_synth_setter_exits_silently() {
  local out
  out=$(echo '{"tool_input":{"command":"mutation { addSubIssue(input: {issueId: \"I_kwABC\", subIssueId: \"I_kwXYZ\"}) { issue { number } } }"},"tool_response":""}' \
    | bash agent/hooks/verify-gh-taxonomy.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0 (no synth-setter in command), got: $out"; return 1; }
}
it "verify-gh-taxonomy: addSubIssue not targeting synth-setter exits 0 silently" T_taxonomy_hierarchy_non_synth_setter_exits_silently

T_taxonomy_check_ci_minimum_joins_with_comma_space() {
  # Unit test for the comma-space join in check_ci_minimum + check_project_fields.
  # Regression for the IFS/`${arr[*]}` quirk Copilot caught on PR #1119: setting
  # `IFS=', '` and expanding `"${arr[*]}"` joins on only the FIRST char of IFS
  # (the comma), producing `issue-type,domain-label` without the space.
  local script_src result
  script_src=$(awk '/^check_ci_minimum\(\) {/,/^}$/' "$REPO_ROOT/agent/hooks/verify-gh-taxonomy.sh")
  result=$(bash -c "$script_src; check_ci_minimum '' 'false' ''")
  [[ "$result" == "issue-type, domain-label, milestone" ]] || {
    echo "expected 'issue-type, domain-label, milestone', got: '$result'"
    return 1
  }
  result=$(bash -c "$script_src; check_ci_minimum 'Task' 'true' 'v1.0'")
  [[ -z "$result" ]] || { echo "expected empty for fully-populated, got: '$result'"; return 1; }
  result=$(bash -c "$script_src; check_ci_minimum 'Task' 'false' 'v1.0'")
  [[ "$result" == "domain-label" ]] || { echo "expected 'domain-label', got: '$result'"; return 1; }
}
it "verify-gh-taxonomy: check_ci_minimum joins missing fields with ', ' (comma+space)" T_taxonomy_check_ci_minimum_joins_with_comma_space

# ===========================================================================
# Run
# ===========================================================================
run_tests
