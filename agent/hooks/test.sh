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
mkdir -p "$SANDBOX/agent/hooks" "$SANDBOX/agent/_shared"
cp agent/hooks/_lib.sh \
   agent/hooks/doc-drift.sh \
   agent/hooks/pr-review-resolver.sh \
   agent/hooks/pre-pr-review-gate.sh \
   agent/hooks/edit-write.sh \
   agent/hooks/verify-gh-taxonomy.sh \
   agent/hooks/worktree-guard.sh \
   agent/hooks/session-start-cwd-banner.sh \
   "$SANDBOX/agent/hooks/"
cp agent/_shared/review_sentinel.py "$SANDBOX/agent/_shared/"
cd "$SANDBOX"
git init -q
git config user.email test@test
git config user.name test
git commit -q --allow-empty -m init
# `INIT_SHA` lets reset_sandbox roll HEAD back between gate tests that create
# commits to set up lag/ancestry scenarios. Exported so the subshell each test
# runs in inherits it.
INIT_SHA=$(git rev-parse HEAD)
export INIT_SHA

STUBS="$TEST_DIR/stubs"
mkdir -p "$STUBS"
# `claude` stub: records $PWD and `git rev-parse HEAD` to caller-supplied files
# so worktree-isolation tests can assert the headless agent ran inside a
# detached worktree pinned to the captured HEAD. CLAUDE_STUB_SLEEP_SECS forces
# a sleep so timeout-wrapping tests can exercise the timeout path.
cat > "$STUBS/claude" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${CLAUDE_STUB_PWD_FILE:-}" ]]; then
  pwd > "$CLAUDE_STUB_PWD_FILE"
fi
if [[ -n "${CLAUDE_STUB_HEAD_FILE:-}" ]]; then
  git rev-parse HEAD > "$CLAUDE_STUB_HEAD_FILE" 2>/dev/null || true
fi
if [[ -n "${CLAUDE_STUB_SLEEP_SECS:-}" ]]; then
  sleep "$CLAUDE_STUB_SLEEP_SECS"
fi
if [[ "${AGENT_STUB_FAIL:-0}" == "1" ]]; then
  echo "simulated headless agent auth failure" >&2
  exit 3
fi
echo "# STUB headless agent output"
echo "# prompt was: $*"
EOF
# `gh` stub: $GH_STUB_PR governs `gh pr view`'s number output. When
# $GH_STUB_LOG is set, every invocation appends its argv there so tests can
# assert explicit-branch arg passing (`gh pr view <branch>`).
cat > "$STUBS/gh" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${GH_STUB_LOG:-}" ]]; then
  printf '%s\n' "$*" >> "$GH_STUB_LOG"
fi
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
  # Drop linked worktrees from prior tests so the next `git worktree add` is
  # free to claim a fresh path; skip the main worktree explicitly.
  local self wt
  self=$(pwd)
  while IFS= read -r wt; do
    [[ -z "$wt" || "$wt" == "$self" ]] && continue
    git worktree remove --force "$wt" >/dev/null 2>&1 || true
  done < <(git worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}')
  git worktree prune >/dev/null 2>&1 || true
  rm -rf .agent-reviews
  # Reset HEAD and refs to the init commit so gate tests that build up
  # commit history don't leak state between runs. Also clean up any side
  # branches the previous test may have created (for first-parent tests).
  if [[ -n "${INIT_SHA:-}" ]]; then
    git reset --hard "$INIT_SHA" >/dev/null 2>&1 || true
    # Drop any non-master/main branches that earlier tests created. Avoid
    # `xargs -r` (GNU extension; absent on macOS/BSD) — read line by line.
    while IFS= read -r branch; do
      [[ -n "$branch" ]] || continue
      git branch -D "$branch" >/dev/null 2>&1 || true
    done < <(git for-each-ref --format='%(refname:short)' refs/heads \
      | grep -vE '^(main|master)$' || true)
  fi
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

# ---------------------------------------------------------------------------
# pr-review-resolver — worktree isolation + cross-session safety
# ---------------------------------------------------------------------------
# Pin the cascade-bug fixes: (1) the headless agent's cwd must NOT be the
# user's main worktree, so a prompt that asks it to switch branches can't
# `git checkout` over uncommitted work; (2) the report file must land in the
# main repo's $REVIEWS_DIR (absolute path), not the temp worktree's, so the
# receiving session can read it after the EXIT trap cleans the worktree;
# (3) the rewake stderr stamps origin HEAD/branch/PR so a session that
# receives a cross-session leak can detect it.

resolver_setup_feature_branch() {
  # Usage: resolver_setup_feature_branch <branch>
  # `--allow-empty` keeps each branch's HEAD distinct from init without
  # depending on a tracked marker file (which would clash across tests that
  # reuse the sandbox repo).
  local branch="$1"
  git checkout -q -b "$branch" 2>/dev/null || git checkout -q "$branch"
  git commit -q --allow-empty -m "feature commit on $branch"
}

T_resolver_headless_agent_runs_in_isolated_worktree() {
  local pwd_file head_file pwd_seen head_seen sandbox_root sandbox_head
  pwd_file="$TEST_DIR/claude-pwd.txt"
  head_file="$TEST_DIR/claude-head.txt"
  rm -f "$pwd_file" "$head_file"
  export RESOLVER_SLEEP_SECS=1 GH_STUB_PR=99
  export CLAUDE_STUB_PWD_FILE="$pwd_file" CLAUDE_STUB_HEAD_FILE="$head_file"
  resolver_setup_feature_branch "feature-iso"
  sandbox_root=$(pwd)
  sandbox_head=$(git rev-parse HEAD)
  echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh >/dev/null 2>&1 || true
  [[ -f "$pwd_file" ]] || { echo "claude stub did not record PWD — headless agent not invoked"; return 1; }
  pwd_seen=$(cat "$pwd_file")
  head_seen=$(cat "$head_file" 2>/dev/null || echo "")
  [[ "$pwd_seen" != "$sandbox_root" ]] || {
    echo "headless agent ran from sandbox repo root ($sandbox_root) — worktree isolation failed"
    return 1
  }
  [[ "$pwd_seen" == *"/worktrees/"* ]] || {
    echo "expected pwd under .agent-reviews/worktrees/, got: $pwd_seen"
    return 1
  }
  [[ "$head_seen" == "$sandbox_head" ]] || {
    echo "worktree HEAD ($head_seen) != captured sandbox HEAD ($sandbox_head)"
    return 1
  }
}
it "pr-review-resolver: headless agent runs inside isolated worktree pinned to captured HEAD" T_resolver_headless_agent_runs_in_isolated_worktree

T_resolver_report_lands_in_main_repo() {
  local report
  export RESOLVER_SLEEP_SECS=1 GH_STUB_PR=99
  unset CLAUDE_STUB_PWD_FILE CLAUDE_STUB_HEAD_FILE
  resolver_setup_feature_branch "feature-report-path"
  echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh >/dev/null 2>&1 || true
  report=$(find .agent-reviews -maxdepth 1 -name 'pr-review-resolver-*.md' 2>/dev/null | head -1)
  [[ -n "$report" ]] || {
    echo "report missing from main repo .agent-reviews/ — anything under worktrees/?"
    find .agent-reviews -name 'pr-review-resolver-*.md' 2>/dev/null
    return 1
  }
}
it "pr-review-resolver: report file lands in main repo .agent-reviews/, not the worktree" T_resolver_report_lands_in_main_repo

T_resolver_worktree_cleaned_up_after_exit() {
  export RESOLVER_SLEEP_SECS=1 GH_STUB_PR=99
  resolver_setup_feature_branch "feature-cleanup"
  echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh >/dev/null 2>&1 || true
  if compgen -G ".agent-reviews/worktrees/resolver-*" >/dev/null; then
    echo "worktree directory still present after hook exit:"
    ls -la .agent-reviews/worktrees/
    return 1
  fi
}
it "pr-review-resolver: EXIT trap removes the worktree after hook completes" T_resolver_worktree_cleaned_up_after_exit

T_resolver_stderr_stamps_origin_head_branch_pr() {
  local stderr_file head_full head_short
  stderr_file="$TEST_DIR/stderr.txt"
  export RESOLVER_SLEEP_SECS=1 RESOLVER_DRY_RUN=1 GH_STUB_PR=99
  resolver_setup_feature_branch "feature-stamp"
  head_full=$(git rev-parse HEAD)
  head_short="${head_full:0:7}"
  echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh 2>"$stderr_file" || true
  grep -q "PR #99" "$stderr_file" || { echo "stderr missing PR number; got: $(cat "$stderr_file")"; return 1; }
  grep -q "branch feature-stamp" "$stderr_file" || { echo "stderr missing branch name; got: $(cat "$stderr_file")"; return 1; }
  grep -qE "origin HEAD ${head_short}" "$stderr_file" || {
    echo "stderr missing origin HEAD short sha ($head_short); got: $(cat "$stderr_file")"
    return 1
  }
  grep -q "crossed sessions" "$stderr_file" || {
    echo "stderr missing cross-session caveat; got: $(cat "$stderr_file")"
    return 1
  }
}
it "pr-review-resolver: rewake stderr stamps PR/branch/origin-HEAD so receivers can detect cross-session leaks" T_resolver_stderr_stamps_origin_head_branch_pr

T_resolver_gh_pr_view_uses_explicit_branch() {
  local gh_log
  gh_log="$TEST_DIR/gh-log.txt"
  rm -f "$gh_log"
  export RESOLVER_SLEEP_SECS=1 RESOLVER_DRY_RUN=1 GH_STUB_PR=99 GH_STUB_LOG="$gh_log"
  resolver_setup_feature_branch "feature-gh-arg"
  echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh >/dev/null 2>&1 || true
  grep -qE '^pr view feature-gh-arg ' "$gh_log" || {
    echo "expected 'gh pr view feature-gh-arg ...' invocation, log was:"
    cat "$gh_log"
    return 1
  }
}
it "pr-review-resolver: gh pr view is called with the captured branch as an explicit arg" T_resolver_gh_pr_view_uses_explicit_branch

T_resolver_bails_skip_worktree_creation() {
  export RESOLVER_SLEEP_SECS=1 RESOLVER_DRY_RUN=1
  # (a) main branch: no PR lookup, no worktree
  git checkout -q -b main 2>/dev/null || git checkout -q main
  export GH_STUB_PR=99
  echo '{"tool_input":{"command":"git push origin main"}}' | bash agent/hooks/pr-review-resolver.sh >/dev/null 2>&1 || true
  if compgen -G ".agent-reviews/worktrees/resolver-*" >/dev/null; then
    echo "main-branch bail leaked a worktree"; return 1
  fi
  # (b) no PR: gh stub returns nothing, no worktree
  resolver_setup_feature_branch "feature-bail-no-pr"
  unset GH_STUB_PR
  echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh >/dev/null 2>&1 || true
  if compgen -G ".agent-reviews/worktrees/resolver-*" >/dev/null; then
    echo "no-PR bail leaked a worktree"; return 1
  fi
}
it "pr-review-resolver: main-branch and no-PR bails skip worktree creation" T_resolver_bails_skip_worktree_creation

T_resolver_sweeps_stale_worktrees_on_start() {
  local stale
  stale=".agent-reviews/worktrees/resolver-stale-abcdef1234"
  export RESOLVER_SLEEP_SECS=1 RESOLVER_DRY_RUN=1 GH_STUB_PR=99
  resolver_setup_feature_branch "feature-sweep"
  mkdir -p "$stale"
  perl -e '$t = time - 3600; utime $t, $t, @ARGV or die "utime failed: $!"' "$stale"
  echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh >/dev/null 2>&1 || true
  [[ ! -d "$stale" ]] || {
    echo "stale worktree directory survived the sweep: $stale"
    return 1
  }
}
it "pr-review-resolver: sweeps leaked resolver-* worktrees older than 30 minutes on hook start" T_resolver_sweeps_stale_worktrees_on_start

T_resolver_headless_timeout_yields_failed_report() {
  local report
  export RESOLVER_SLEEP_SECS=1 GH_STUB_PR=99
  export CLAUDE_STUB_SLEEP_SECS=5 AGENT_TIMEOUT_SECS=1
  resolver_setup_feature_branch "feature-timeout"
  echo '{"tool_input":{"command":"git push"}}' | bash agent/hooks/pr-review-resolver.sh >/dev/null 2>&1 || true
  report=$(find .agent-reviews -maxdepth 1 -name 'pr-review-resolver-*.md' 2>/dev/null | head -1)
  [[ -n "$report" ]] || { echo "report missing after timeout"; return 1; }
  grep -q "FAILED" "$report" || {
    echo "expected FAILED marker in timeout report; got:"
    cat "$report"
    return 1
  }
  # GNU timeout exits 124 on TERM; the report header records that code.
  awk '/^## headless agent exit code$/{getline; print; exit}' "$report" | grep -qE '^124$' || {
    echo "expected exit code 124 (timeout) in report, got:"
    awk '/^## headless agent exit code$/{getline; print; exit}' "$report"
    return 1
  }
}
it "pr-review-resolver: AGENT_TIMEOUT_SECS bounds the headless agent and surfaces exit 124 in the report" T_resolver_headless_timeout_yields_failed_report

# ---------------------------------------------------------------------------
# doc-drift — worktree isolation + cross-session safety
# ---------------------------------------------------------------------------

T_doc_drift_headless_agent_runs_in_isolated_worktree() {
  local pwd_file head_file pwd_seen head_seen sandbox_root sandbox_head
  pwd_file="$TEST_DIR/claude-pwd.txt"
  head_file="$TEST_DIR/claude-head.txt"
  rm -f "$pwd_file" "$head_file"
  export GH_STUB_PR=42
  export CLAUDE_STUB_PWD_FILE="$pwd_file" CLAUDE_STUB_HEAD_FILE="$head_file"
  resolver_setup_feature_branch "drift-iso"
  sandbox_root=$(pwd)
  sandbox_head=$(git rev-parse HEAD)
  echo '{"tool_input":{"command":"gh pr create --title x"}}' | bash agent/hooks/doc-drift.sh >/dev/null 2>&1 || true
  [[ -f "$pwd_file" ]] || { echo "claude stub did not record PWD — headless agent not invoked"; return 1; }
  pwd_seen=$(cat "$pwd_file")
  head_seen=$(cat "$head_file" 2>/dev/null || echo "")
  [[ "$pwd_seen" != "$sandbox_root" ]] || {
    echo "doc-drift headless agent ran from sandbox repo root — worktree isolation failed"
    return 1
  }
  [[ "$pwd_seen" == *"/worktrees/"* ]] || {
    echo "expected pwd under .agent-reviews/worktrees/, got: $pwd_seen"
    return 1
  }
  [[ "$head_seen" == "$sandbox_head" ]] || {
    echo "doc-drift worktree HEAD ($head_seen) != captured sandbox HEAD ($sandbox_head)"
    return 1
  }
}
it "doc-drift: headless agent runs inside isolated worktree pinned to captured HEAD" T_doc_drift_headless_agent_runs_in_isolated_worktree

T_doc_drift_stderr_stamps_origin_head_branch_pr() {
  local stderr_file head_short
  stderr_file="$TEST_DIR/stderr.txt"
  export GH_STUB_PR=42 DOC_DRIFT_DRY_RUN=1
  resolver_setup_feature_branch "drift-stamp"
  head_short=$(git rev-parse HEAD); head_short="${head_short:0:7}"
  echo '{"tool_input":{"command":"gh pr create --title x"}}' | bash agent/hooks/doc-drift.sh 2>"$stderr_file" || true
  grep -q "PR #42" "$stderr_file" || { echo "stderr missing PR; got: $(cat "$stderr_file")"; return 1; }
  grep -q "branch drift-stamp" "$stderr_file" || { echo "stderr missing branch; got: $(cat "$stderr_file")"; return 1; }
  grep -qE "origin HEAD ${head_short}" "$stderr_file" || {
    echo "stderr missing origin HEAD; got: $(cat "$stderr_file")"
    return 1
  }
  grep -q "crossed sessions" "$stderr_file" || {
    echo "stderr missing cross-session caveat; got: $(cat "$stderr_file")"
    return 1
  }
}
it "doc-drift: rewake stderr stamps PR/branch/origin-HEAD so receivers can detect cross-session leaks" T_doc_drift_stderr_stamps_origin_head_branch_pr

T_doc_drift_gh_pr_view_uses_explicit_branch() {
  local gh_log
  gh_log="$TEST_DIR/gh-log.txt"
  rm -f "$gh_log"
  export GH_STUB_PR=42 DOC_DRIFT_DRY_RUN=1 GH_STUB_LOG="$gh_log"
  resolver_setup_feature_branch "drift-gh-arg"
  echo '{"tool_input":{"command":"gh pr create --title x"}}' | bash agent/hooks/doc-drift.sh >/dev/null 2>&1 || true
  grep -qE '^pr view drift-gh-arg ' "$gh_log" || {
    echo "expected 'gh pr view drift-gh-arg ...' invocation, log was:"
    cat "$gh_log"
    return 1
  }
}
it "doc-drift: gh pr view is called with the captured branch as an explicit arg" T_doc_drift_gh_pr_view_uses_explicit_branch

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

# --- helpers for gate sentinel-validation tests ---
# `gate_sentinel_path` echoes the canonical sentinel path for a SHA via the
# shared Python helper, so the test contract drifts with the helper rather
# than hardcoding the format string in bash. Caller is expected to have the
# helper at agent/_shared/review_sentinel.py (sandbox copies it in).
gate_sentinel_path() {
  # Usage: gate_sentinel_path <sha>
  python3 agent/_shared/review_sentinel.py path "$1"
}

# `gate_make_sentinel_review` creates a review file at the canonical sentinel
# path for a SHA, padded past the 200-byte minimum so the size guard doesn't
# false-trigger.
gate_make_sentinel_review() {
  # Usage: gate_make_sentinel_review <sha>
  local sha="$1" path
  path=$(gate_sentinel_path "$sha")
  mkdir -p "$(dirname "$path")"
  {
    printf '# repo-review-full-no-comments — review @ %s\n\n' "$sha"
    printf '## Summary\n\n0 BLOCK, 0 WARN across N skills.\n\n'
    printf 'finding line %d\n' {1..20}
  } > "$path"
  printf '%s\n' "$path"
}

T_gate_no_path_blocks() {
  local out stderr_file
  stderr_file="$TEST_DIR/gate_stderr.txt"
  out=$(echo '{"tool_input":{"command":"gh pr create --title x --body y"}}' | bash agent/hooks/pre-pr-review-gate.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  grep -q "BLOCKED" "$stderr_file" || { echo "stderr should contain BLOCKED"; return 1; }
  grep -q "REVIEW_FULL=" "$stderr_file" || { echo "stderr should reference REVIEW_FULL=<path> contract"; return 1; }
}
it "pre-pr-review-gate: gh pr create without REVIEW_FULL=<path> → exit 2 with BLOCKED + contract on stderr" T_gate_no_path_blocks

T_gate_missing_file_blocks() {
  local out stderr_file
  stderr_file="$TEST_DIR/gate_stderr.txt"
  out=$(echo '{"tool_input":{"command":"gh pr create --title x --body y  # REVIEW_FULL=.agent-reviews/does-not-exist.md"}}' | bash agent/hooks/pre-pr-review-gate.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  grep -q "does not point at a file" "$stderr_file" || { echo "stderr should mention the missing-file reason"; return 1; }
}
it "pre-pr-review-gate: REVIEW_FULL=<nonexistent> → exit 2 with file-not-found reason" T_gate_missing_file_blocks

T_gate_small_file_blocks() {
  local out stderr_file path head_sha
  stderr_file="$TEST_DIR/gate_stderr.txt"
  head_sha=$(git rev-parse HEAD)
  path=$(gate_sentinel_path "$head_sha")
  mkdir -p "$(dirname "$path")"
  # Sentinel-named, but content is a trivial touch-bypass (well under 200B).
  printf 'x\n' > "$path"
  out=$(echo "{\"tool_input\":{\"command\":\"gh pr create --title x --body y  # REVIEW_FULL=$path\"}}" | bash agent/hooks/pre-pr-review-gate.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  grep -q "suspiciously small" "$stderr_file" || { echo "stderr should mention size guard: $(cat "$stderr_file")"; return 1; }
}
it "pre-pr-review-gate: REVIEW_FULL=<sentinel-named but <200B> → exit 2 (touch-bypass guard)" T_gate_small_file_blocks

T_gate_size_boundary_199_blocks() {
  # Exactly one byte below the 200-byte guard — must block.
  local out stderr_file path head_sha actual_size
  stderr_file="$TEST_DIR/gate_stderr.txt"
  head_sha=$(git rev-parse HEAD)
  path=$(gate_sentinel_path "$head_sha")
  mkdir -p "$(dirname "$path")"
  printf 'x%.0s' {1..199} > "$path"
  actual_size=$(stat -c %s "$path" 2>/dev/null || stat -f %z "$path")
  [[ "$actual_size" == "199" ]] || { echo "fixture size wrong: $actual_size"; return 1; }
  out=$(echo "{\"tool_input\":{\"command\":\"gh pr create --title x --body y  # REVIEW_FULL=$path\"}}" | bash agent/hooks/pre-pr-review-gate.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2 at 199B, got: $out"; return 1; }
  grep -q "suspiciously small" "$stderr_file" || { echo "stderr should mention size guard: $(cat "$stderr_file")"; return 1; }
}
it "pre-pr-review-gate: REVIEW_FULL file at exactly 199 bytes → exit 2 (lower bound)" T_gate_size_boundary_199_blocks

T_gate_size_boundary_200_passes() {
  # Exactly at the 200-byte floor — must pass the size guard (lag=0 since
  # the sentinel encodes HEAD).
  local out path head_sha actual_size
  head_sha=$(git rev-parse HEAD)
  path=$(gate_sentinel_path "$head_sha")
  mkdir -p "$(dirname "$path")"
  printf 'x%.0s' {1..200} > "$path"
  actual_size=$(stat -c %s "$path" 2>/dev/null || stat -f %z "$path")
  [[ "$actual_size" == "200" ]] || { echo "fixture size wrong: $actual_size"; return 1; }
  out=$(echo "{\"tool_input\":{\"command\":\"gh pr create --title x --body y  # REVIEW_FULL=$path\"}}" | bash agent/hooks/pre-pr-review-gate.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0 at 200B, got: $out"; return 1; }
}
it "pre-pr-review-gate: REVIEW_FULL file at exactly 200 bytes → exit 0 (lower bound inclusive)" T_gate_size_boundary_200_passes

T_gate_bad_filename_blocks() {
  local out stderr_file path
  stderr_file="$TEST_DIR/gate_stderr.txt"
  path=".agent-reviews/not-a-sentinel.md"
  mkdir -p .agent-reviews
  # Plenty of bytes, just wrong filename shape.
  printf 'review body, %s bytes of padding %s' "$(printf 'x%.0s' {1..200})" "tail" > "$path"
  out=$(echo "{\"tool_input\":{\"command\":\"gh pr create --title x --body y  # REVIEW_FULL=$path\"}}" | bash agent/hooks/pre-pr-review-gate.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  grep -q "does not match the sentinel pattern" "$stderr_file" || { echo "stderr should mention sentinel-mismatch reason: $(cat "$stderr_file")"; return 1; }
}
it "pre-pr-review-gate: REVIEW_FULL=<non-sentinel filename> → exit 2 with sentinel-mismatch reason" T_gate_bad_filename_blocks

T_gate_non_ancestor_blocks() {
  local out stderr_file fake_sha path
  stderr_file="$TEST_DIR/gate_stderr.txt"
  # Well-shaped 40-char hex SHA that the repo has never seen.
  fake_sha="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
  path=$(gate_make_sentinel_review "$fake_sha")
  out=$(echo "{\"tool_input\":{\"command\":\"gh pr create --title x --body y  # REVIEW_FULL=$path\"}}" | bash agent/hooks/pre-pr-review-gate.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  grep -q "is not an ancestor of HEAD" "$stderr_file" || { echo "stderr should mention non-ancestor reason: $(cat "$stderr_file")"; return 1; }
}
it "pre-pr-review-gate: REVIEW_FULL=<SHA not on this branch> → exit 2 with non-ancestor reason" T_gate_non_ancestor_blocks

T_gate_head_sha_passes() {
  local out path head_sha
  head_sha=$(git rev-parse HEAD)
  path=$(gate_make_sentinel_review "$head_sha")
  out=$(echo "{\"tool_input\":{\"command\":\"gh pr create --title x --body y  # REVIEW_FULL=$path\"}}" | bash agent/hooks/pre-pr-review-gate.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ "$out" != *"BLOCKED"* ]] || { echo "HEAD-SHA review should not BLOCK"; return 1; }
}
it "pre-pr-review-gate: REVIEW_FULL=<HEAD-SHA sentinel> → exit 0 (lag=0)" T_gate_head_sha_passes

T_gate_lag_within_default_passes() {
  # Build two more commits on top of the reviewed SHA. lag = 2 = default max.
  local out path review_sha
  review_sha=$(git rev-parse HEAD)
  path=$(gate_make_sentinel_review "$review_sha")
  git commit -q --allow-empty -m "follow-up 1"
  git commit -q --allow-empty -m "follow-up 2"
  out=$(echo "{\"tool_input\":{\"command\":\"gh pr create --title x --body y  # REVIEW_FULL=$path\"}}" | bash agent/hooks/pre-pr-review-gate.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0 at lag=2, got: $out"; return 1; }
}
it "pre-pr-review-gate: review-SHA + 2 first-parent commits → exit 0 (lag=max default)" T_gate_lag_within_default_passes

T_gate_lag_over_default_blocks() {
  # Three commits past the reviewed SHA: lag=3, default max=2.
  local out stderr_file path review_sha
  stderr_file="$TEST_DIR/gate_stderr.txt"
  review_sha=$(git rev-parse HEAD)
  path=$(gate_make_sentinel_review "$review_sha")
  git commit -q --allow-empty -m "follow-up 1"
  git commit -q --allow-empty -m "follow-up 2"
  git commit -q --allow-empty -m "follow-up 3"
  out=$(echo "{\"tool_input\":{\"command\":\"gh pr create --title x --body y  # REVIEW_FULL=$path\"}}" | bash agent/hooks/pre-pr-review-gate.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2 at lag=3, got: $out"; return 1; }
  grep -q "first-parent commits behind HEAD" "$stderr_file" || { echo "stderr should mention lag overflow: $(cat "$stderr_file")"; return 1; }
}
it "pre-pr-review-gate: review-SHA + 3 first-parent commits → exit 2 (lag>default)" T_gate_lag_over_default_blocks

T_gate_max_lag_override_passes() {
  # Same lag=3 scenario, but with REVIEW_MAX_LAG=5 to widen the window.
  local out path review_sha
  review_sha=$(git rev-parse HEAD)
  path=$(gate_make_sentinel_review "$review_sha")
  git commit -q --allow-empty -m "follow-up 1"
  git commit -q --allow-empty -m "follow-up 2"
  git commit -q --allow-empty -m "follow-up 3"
  out=$(REVIEW_MAX_LAG=5 bash -c "echo '{\"tool_input\":{\"command\":\"gh pr create --title x --body y  # REVIEW_FULL=$path\"}}' | bash agent/hooks/pre-pr-review-gate.sh" 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0 with REVIEW_MAX_LAG=5, got: $out"; return 1; }
}
it "pre-pr-review-gate: REVIEW_MAX_LAG=5 widens the lag window → exit 0" T_gate_max_lag_override_passes

T_gate_first_parent_merge_counts_as_one() {
  # Side branch with 5 commits merged with --no-ff: first-parent lag = 1, not 6.
  # Exercises the lag-counting contract: merges from main count as 1 commit.
  local out path review_sha side_branch main_branch
  main_branch=$(git rev-parse --abbrev-ref HEAD)
  review_sha=$(git rev-parse HEAD)
  path=$(gate_make_sentinel_review "$review_sha")
  side_branch="side-$$"
  git checkout -q -b "$side_branch"
  for n in 1 2 3 4 5; do
    git commit -q --allow-empty -m "side $n"
  done
  git checkout -q "$main_branch"
  git merge -q --no-ff --no-edit "$side_branch" >/dev/null
  out=$(echo "{\"tool_input\":{\"command\":\"gh pr create --title x --body y  # REVIEW_FULL=$path\"}}" | bash agent/hooks/pre-pr-review-gate.sh 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "first-parent merge of 5 commits should be lag=1, got: $out"; return 1; }
  # Pin the actual lag against .agent-reviews/.hook.log (where the gate's
  # `log` helper writes). Guards against --first-parent → plain rev-list
  # regressions (which would report lag=6).
  grep -q "lag=1/" .agent-reviews/.hook.log || {
    echo "expected 'lag=1/' in .agent-reviews/.hook.log to confirm --first-parent semantics"
    cat .agent-reviews/.hook.log 2>/dev/null || echo "(log missing)"
    return 1
  }
}
it "pre-pr-review-gate: --first-parent — merging a 5-commit side branch counts as lag=1" T_gate_first_parent_merge_counts_as_one

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

# Helper: extract strip_markdown_issue_links from verify-gh-taxonomy.sh and
# invoke it on $1, printing the sanitized output. Slices the function with
# awk anchored on `^}$` — the function's docstring documents this coupling.
run_strip_markdown_issue_links() {
  local script_src
  script_src=$(awk '/^strip_markdown_issue_links\(\) {/,/^}$/' \
    "$REPO_ROOT/agent/hooks/verify-gh-taxonomy.sh")
  bash -c "$script_src"$'\n''strip_markdown_issue_links "$1"' _ "$1"
}

T_taxonomy_strip_markdown_issue_links_drops_review_comment_refs() {
  # Regression for PR #1163 run 26126477593: Copilot review-comment
  # `[#3269588963](.../discussion_r…)` markdown links were matched by
  # `#[0-9]+` and fetched as bogus issue numbers, 404'ing the gate.
  local body result
  body=$'Closes #1157.\n- [#3269588963](https://github.com/x/y/pull/1#discussion_r3269588963)\n- [#3269589023](https://github.com/x/y/pull/1#discussion_r3269589023)\nRefs #42'
  result=$(run_strip_markdown_issue_links "$body" | grep -oE '#[0-9]+' | sort -u | tr '\n' ' ')
  [[ "$result" == "#1157 #42 " ]] || {
    echo "expected '#1157 #42 ' (markdown-linked review-comment IDs stripped), got: '$result'"
    return 1
  }
}
it "verify-gh-taxonomy: strip_markdown_issue_links removes [#N](url) review-comment refs (PR #1163 regression)" T_taxonomy_strip_markdown_issue_links_drops_review_comment_refs

T_taxonomy_strip_markdown_issue_links_preserves_bare_refs() {
  # Counter-test: bare `#N` and `Closes #N` / `Refs #N` patterns must survive
  # sanitization so the gate keeps enforcing linked issues.
  local body result
  body=$'## Summary\n\nFollow-up to PR #1157.\nCloses #42\nRefs #99'
  result=$(run_strip_markdown_issue_links "$body" | grep -oE '#[0-9]+' | sort -u | tr '\n' ' ')
  [[ "$result" == "#1157 #42 #99 " ]] || {
    echo "expected '#1157 #42 #99 ' (all bare refs preserved), got: '$result'"
    return 1
  }
}
it "verify-gh-taxonomy: strip_markdown_issue_links preserves bare #N references" T_taxonomy_strip_markdown_issue_links_preserves_bare_refs

T_taxonomy_strip_markdown_issue_links_drops_code_span_refs() {
  # Regression for PR #1171 run 26127785920: the PR body cited comment IDs
  # inside backticked code spans (e.g. `#3269588963` in prose, `#1157, #1165,
  # #3269588963, ...` in an enum), not in `[#N](url)` markdown links. The
  # markdown-only strip from #1163 missed those, so the gate re-failed.
  # Inline code spans are prose-as-text, never the way a real issue ref is
  # written — strip the whole span before extraction.
  local body result
  body=$'## Summary\n\nCloses #42. Comment IDs like `#3269588963` are not issues.\n`enum: #1157, #3269588963, #3269589002`\nRefs #99'
  result=$(run_strip_markdown_issue_links "$body" | grep -oE '#[0-9]+' | sort -u | tr '\n' ' ')
  [[ "$result" == "#42 #99 " ]] || {
    echo "expected '#42 #99 ' (refs in code spans stripped), got: '$result'"
    return 1
  }
}
it "verify-gh-taxonomy: strip_markdown_issue_links removes #N refs inside backticked code spans (PR #1171 regression)" T_taxonomy_strip_markdown_issue_links_drops_code_span_refs

T_taxonomy_workflow_inlines_same_sanitize_regex_as_hook() {
  # Drift guard: the bash hook and pr-metadata-gate.yaml each carry their own
  # copy of the sanitization sed pipeline. Pin the workflow to have the exact
  # expression once per job; a unilateral edit to either side fails here.
  local workflow="$REPO_ROOT/.github/workflows/pr-metadata-gate.yaml"
  local expected count
  expected="sed -E -e 's/\`[^\`]*\`//g' -e 's/\\[#[0-9]+\\]\\([^)]*\\)//g'"
  count=$(grep -cF "$expected" "$workflow") || count=0
  [[ "$count" -eq 2 ]] || {
    echo "expected pr-metadata-gate.yaml to contain the sanitize sed pipeline exactly 2x (once per job), found ${count}."
    echo "If you intentionally changed the workflow regex, mirror the change in strip_markdown_issue_links() in verify-gh-taxonomy.sh and update this test."
    return 1
  }
}
it "verify-gh-taxonomy: pr-metadata-gate.yaml inlines the same sanitize sed pipeline (drift guard)" T_taxonomy_workflow_inlines_same_sanitize_regex_as_hook

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
# worktree-guard.sh — primary-checkout detection + mode dispatch
# ===========================================================================
#
# Sandbox is its own primary checkout at $SANDBOX; tests that need a linked
# worktree do `git worktree add` inside the sandbox (reset_sandbox tears them
# down between cases). Empty stdin is intentional — the guard ignores
# tool_input.file_path, so the test harness need not synthesize realistic JSON.

T_worktree_guard_warns_in_primary() {
  local out stderr_file
  stderr_file="$TEST_DIR/wg_stderr.txt"
  out=$(echo '' | bash agent/hooks/worktree-guard.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  grep -q "WARNING" "$stderr_file" || { echo "stderr should contain WARNING; got: $(cat "$stderr_file")"; return 1; }
  grep -q "git worktree add" "$stderr_file" || { echo "stderr should suggest 'git worktree add'"; return 1; }
  grep -q "make link-plugins" "$stderr_file" || { echo "remediation should chain 'make link-plugins' so the new worktree gets the VST symlink; got: $(cat "$stderr_file")"; return 1; }
}
it "worktree-guard: edit in primary (warn mode default) → exit 0 with WARNING + remediation on stderr" T_worktree_guard_warns_in_primary

T_worktree_guard_blocks_in_primary_when_mode_block() {
  local out stderr_file
  stderr_file="$TEST_DIR/wg_stderr.txt"
  out=$(WORKTREE_GUARD_MODE=block bash -c 'echo "" | bash agent/hooks/worktree-guard.sh' 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:2" ]] || { echo "expected EXIT:2, got: $out"; return 1; }
  grep -q "BLOCKED" "$stderr_file" || { echo "stderr should contain BLOCKED; got: $(cat "$stderr_file")"; return 1; }
}
it "worktree-guard: WORKTREE_GUARD_MODE=block in primary → exit 2 with BLOCKED on stderr" T_worktree_guard_blocks_in_primary_when_mode_block

T_worktree_guard_off_mode_silent_in_primary() {
  local out stderr_file
  stderr_file="$TEST_DIR/wg_stderr.txt"
  out=$(WORKTREE_GUARD_MODE=off bash -c 'echo "" | bash agent/hooks/worktree-guard.sh' 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ -z "$(cat "$stderr_file")" ]] || { echo "stderr should be empty in off mode; got: $(cat "$stderr_file")"; return 1; }
}
it "worktree-guard: WORKTREE_GUARD_MODE=off → exit 0 with no output (escape hatch)" T_worktree_guard_off_mode_silent_in_primary

T_worktree_guard_silent_in_linked_worktree() {
  local out stderr_file wt
  stderr_file="$TEST_DIR/wg_stderr.txt"
  wt="$TEST_DIR/linked-wt"
  git worktree add -q -b wg-test-branch "$wt" >/dev/null 2>&1
  out=$(cd "$wt" && echo '' | bash "$SANDBOX/agent/hooks/worktree-guard.sh" 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ -z "$(cat "$stderr_file")" ]] || { echo "stderr should be empty in linked worktree; got: $(cat "$stderr_file")"; return 1; }
}
it "worktree-guard: edit inside a linked worktree → exit 0, no warning" T_worktree_guard_silent_in_linked_worktree

T_worktree_guard_silent_outside_repo() {
  local out stderr_file scratch
  stderr_file="$TEST_DIR/wg_stderr.txt"
  scratch=$(mktemp -d "$TEST_DIR/no-git-XXXX")
  out=$(cd "$scratch" && echo '' | bash "$SANDBOX/agent/hooks/worktree-guard.sh" 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ -z "$(cat "$stderr_file")" ]] || { echo "stderr should be empty outside any git repo; got: $(cat "$stderr_file")"; return 1; }
}
it "worktree-guard: edit outside any git repo → exit 0, no warning" T_worktree_guard_silent_outside_repo

T_worktree_guard_unknown_mode_falls_back_to_warn() {
  local out stderr_file
  stderr_file="$TEST_DIR/wg_stderr.txt"
  out=$(WORKTREE_GUARD_MODE=bogus bash -c 'echo "" | bash agent/hooks/worktree-guard.sh' 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0 even on unknown mode, got: $out"; return 1; }
  grep -q "ignoring unknown WORKTREE_GUARD_MODE=bogus" "$stderr_file" || {
    echo "stderr should log the unknown-mode fallback; got: $(cat "$stderr_file")"
    return 1
  }
}
it "worktree-guard: unknown WORKTREE_GUARD_MODE value logs + exits 0 (never blocks via typo)" T_worktree_guard_unknown_mode_falls_back_to_warn

T_worktree_guard_off_and_unknown_drain_large_stdin() {
  # Regression: before draining-before-dispatch, `off` and unknown-mode
  # exited without reading stdin, leaving a >64KB-pipe-buffer write blocked
  # in the harness writer; the read-end close then surfaced as SIGPIPE
  # (exit 141 with pipefail). Both early-exit paths must drain.
  #
  # `head -c N /dev/zero` is a *finite* source: it emits exactly N bytes
  # then exits 0, so the pipeline only flags SIGPIPE if the *hook* (not an
  # infinite generator like `yes`) failed to read.
  local exit_code
  set +e
  head -c 131072 /dev/zero | WORKTREE_GUARD_MODE=off bash agent/hooks/worktree-guard.sh >/dev/null 2>&1
  exit_code=$?
  set -e
  [[ "$exit_code" == "0" ]] || { echo "off-mode pipeline exit was $exit_code (likely SIGPIPE in writer)"; return 1; }
  set +e
  head -c 131072 /dev/zero | WORKTREE_GUARD_MODE=bogus bash agent/hooks/worktree-guard.sh >/dev/null 2>&1
  exit_code=$?
  set -e
  [[ "$exit_code" == "0" ]] || { echo "unknown-mode pipeline exit was $exit_code (likely SIGPIPE in writer)"; return 1; }
}
it "worktree-guard: off + unknown modes drain stdin before exiting (no SIGPIPE risk)" T_worktree_guard_off_and_unknown_drain_large_stdin

T_worktree_guard_detached_head_in_primary() {
  # Regression: --abbrev-ref HEAD returns the literal "HEAD" when detached,
  # so the warning + remediation must not show 'HEAD' as a branch name, and
  # `git worktree add --detach` must not be passed a positional commit-ish
  # of "HEAD" derived from the bad detection.
  local out stderr_file
  stderr_file="$TEST_DIR/wg_stderr.txt"
  git checkout -q --detach HEAD
  out=$(echo '' | bash agent/hooks/worktree-guard.sh 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  grep -q "WARNING" "$stderr_file" || { echo "stderr should still WARN in detached primary; got: $(cat "$stderr_file")"; return 1; }
  grep -q "detached HEAD" "$stderr_file" || {
    echo "stderr should report detached HEAD, not literal branch name; got: $(cat "$stderr_file")"
    return 1
  }
  # Remediation must use --detach (always works); no branch arg leaked from the bad probe.
  grep -q "git worktree add --detach" "$stderr_file" || {
    echo "stderr should recommend 'git worktree add --detach'; got: $(cat "$stderr_file")"
    return 1
  }
  grep -qE "git worktree add .* HEAD$" "$stderr_file" && {
    echo "stderr should NOT pass 'HEAD' as a branch arg; got: $(cat "$stderr_file")"
    return 1
  }
  return 0
}
it "worktree-guard: detached HEAD in primary → 'detached HEAD' in message, '--detach' remediation, no 'HEAD' branch arg" T_worktree_guard_detached_head_in_primary

T_worktree_guard_remediation_path_is_absolute_from_subdir() {
  # Regression: original remediation used relative `.claude/worktrees/<slug>`, so
  # running from `<primary>/src/foo/` produced `git worktree add --detach
  # .claude/worktrees/<slug>` — which `cd`'d the agent into a path that didn't
  # exist (subdir-relative). Remediation must anchor at $primary_root.
  local out stderr_file subdir
  stderr_file="$TEST_DIR/wg_stderr.txt"
  subdir="$SANDBOX/src/nested"
  mkdir -p "$subdir"
  out=$(cd "$subdir" && echo '' | bash "$SANDBOX/agent/hooks/worktree-guard.sh" 2>"$stderr_file"; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  grep -q "WARNING" "$stderr_file" || { echo "subdir cwd should still WARN; got: $(cat "$stderr_file")"; return 1; }
  grep -qE "git worktree add --detach ${SANDBOX}/\\.claude/worktrees/" "$stderr_file" || {
    echo "remediation must anchor at \$primary_root (${SANDBOX}); got: $(cat "$stderr_file")"
    return 1
  }
  grep -qE "git worktree add --detach \\.claude/worktrees/" "$stderr_file" && {
    echo "remediation must NOT use a bare relative path; got: $(cat "$stderr_file")"
    return 1
  }
  return 0
}
it "worktree-guard: remediation path anchored to primary_root (works from any subdir)" T_worktree_guard_remediation_path_is_absolute_from_subdir

# ===========================================================================
# session-start-cwd-banner.sh — banner content per cwd
# ===========================================================================

T_session_start_banner_in_primary() {
  local out
  out=$(bash agent/hooks/session-start-cwd-banner.sh </dev/null 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ "$out" == *"PRIMARY CHECKOUT"* ]] || { echo "banner should flag PRIMARY CHECKOUT; got: $out"; return 1; }
  [[ "$out" == *"git worktree add"* ]] || { echo "banner should suggest 'git worktree add'; got: $out"; return 1; }
  [[ "$out" == *"make link-plugins"* ]] || { echo "spawn command should chain 'make link-plugins' so the new worktree gets the VST symlink; got: $out"; return 1; }
}
it "session-start-banner: in primary → stdout flags PRIMARY CHECKOUT and shows remediation" T_session_start_banner_in_primary

T_session_start_banner_in_linked_worktree() {
  local out wt
  wt="$TEST_DIR/banner-wt"
  git worktree add -q -b banner-test-branch "$wt" >/dev/null 2>&1
  out=$(cd "$wt" && bash "$SANDBOX/agent/hooks/session-start-cwd-banner.sh" </dev/null 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ "$out" == *"isolated worktree"* ]] || { echo "banner should say 'isolated worktree'; got: $out"; return 1; }
  [[ "$out" != *"PRIMARY CHECKOUT"* ]] || { echo "banner should NOT flag PRIMARY in a linked worktree; got: $out"; return 1; }
}
it "session-start-banner: in linked worktree → stdout reports 'isolated worktree (OK)'" T_session_start_banner_in_linked_worktree

T_session_start_banner_silent_outside_repo() {
  local out scratch
  scratch=$(mktemp -d "$TEST_DIR/banner-no-git-XXXX")
  out=$(cd "$scratch" && bash "$SANDBOX/agent/hooks/session-start-cwd-banner.sh" </dev/null 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  local without_exit="${out%EXIT:*}"
  [[ -z "${without_exit//[[:space:]]/}" ]] || {
    echo "banner should be silent outside any git repo; got: $without_exit"
    return 1
  }
}
it "session-start-banner: outside any git repo → silent, exit 0" T_session_start_banner_silent_outside_repo

T_session_start_banner_detached_head_in_primary() {
  # Regression: --abbrev-ref HEAD returns literal "HEAD" when detached, so the
  # banner used to render `branch: HEAD` (misleading) and bake `HEAD` into the
  # remediation. Switch to --show-current + a labeled fallback.
  local out
  git checkout -q --detach HEAD
  out=$(bash agent/hooks/session-start-cwd-banner.sh </dev/null 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ "$out" == *"PRIMARY CHECKOUT"* ]] || { echo "detached primary should still flag PRIMARY CHECKOUT; got: $out"; return 1; }
  [[ "$out" == *"detached HEAD"* ]] || {
    echo "banner should label detached HEAD instead of printing 'branch: HEAD'; got: $out"
    return 1
  }
  [[ "$out" != *"branch   : HEAD"* ]] || {
    echo "banner should NOT show 'branch: HEAD' for detached HEAD; got: $out"
    return 1
  }
  # Remediation must use --detach and must not pass HEAD as a branch arg.
  [[ "$out" == *"git worktree add --detach"* ]] || {
    echo "banner should recommend 'git worktree add --detach'; got: $out"
    return 1
  }
}
it "session-start-banner: detached HEAD in primary → 'detached HEAD' label, '--detach' remediation, no 'branch: HEAD'" T_session_start_banner_detached_head_in_primary

T_session_start_banner_remediation_path_is_absolute_from_subdir() {
  # Same regression as worktree-guard's subdir-anchored-path test:
  # the SessionStart hook's PWD is whatever the operator launched `claude`
  # from. If it's a subdir of primary, a bare relative `.claude/worktrees/...`
  # in the remediation `cd`'s into a path that doesn't exist.
  local out subdir
  subdir="$SANDBOX/src/nested-banner"
  mkdir -p "$subdir"
  out=$(cd "$subdir" && bash "$SANDBOX/agent/hooks/session-start-cwd-banner.sh" </dev/null 2>&1; echo "EXIT:$?")
  [[ "$(last_exit_line "$out")" == "EXIT:0" ]] || { echo "expected EXIT:0, got: $out"; return 1; }
  [[ "$out" == *"PRIMARY CHECKOUT"* ]] || { echo "subdir cwd should still flag PRIMARY; got: $out"; return 1; }
  echo "$out" | grep -qE "git worktree add --detach '?${SANDBOX}/\\.claude/worktrees/" || {
    echo "remediation must anchor at \$primary_root (${SANDBOX}); got: $out"
    return 1
  }
  return 0
}
it "session-start-banner: remediation path anchored to primary_root (works from any subdir)" T_session_start_banner_remediation_path_is_absolute_from_subdir

# ===========================================================================
# Run
# ===========================================================================
run_tests
