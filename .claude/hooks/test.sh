#!/usr/bin/env bash
# =============================================================================
# test.sh — unit harness for .claude/hooks/
# =============================================================================
#
# Tests run each hook script with canned stdin and PATH-stubbed `claude`/`gh`/`git`,
# asserting exit codes, stderr pointers, report files, and logged decisions.
#
# Run from repo root:   bash .claude/hooks/test.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Isolate test artifacts in a temp dir we cd into, so .agent-reviews/ paths
# used by the hooks land in a sandbox and don't pollute the real worktree.
TEST_DIR=$(mktemp -d)
cleanup() { rm -rf "$TEST_DIR"; }
trap cleanup EXIT

# Copy hook scripts into the sandbox so paths resolve; point GIT_DIR / repo at
# the sandbox so hooks calling `git` don't see the real repo state.
SANDBOX="$TEST_DIR/repo"
mkdir -p "$SANDBOX/.claude/hooks"
cp .claude/hooks/_lib.sh .claude/hooks/doc-drift.sh .claude/hooks/pr-review-resolver.sh \
  "$SANDBOX/.claude/hooks/"
cd "$SANDBOX"
git init -q
git config user.email test@test
git config user.name test
git commit -q --allow-empty -m init

# Stub PATH with fake claude/gh (git is real — we need it for branch detection).
STUBS="$TEST_DIR/stubs"
mkdir -p "$STUBS"
cat > "$STUBS/claude" <<'EOF'
#!/usr/bin/env bash
echo "# STUB claude -p output"
echo "# prompt was: $*"
EOF
cat > "$STUBS/gh" <<'EOF'
#!/usr/bin/env bash
# $GH_STUB_PR governs what `gh pr view` returns. Empty = no PR.
if [ "$1" = "pr" ] && [ "$2" = "view" ]; then
  if [ -n "${GH_STUB_PR:-}" ]; then
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

PASS=0
FAIL=0
pass() { printf '  PASS  %s\n' "$1"; PASS=$((PASS+1)); }
fail() { printf '  FAIL  %s — %s\n' "$1" "$2"; FAIL=$((FAIL+1)); }

reset_sandbox() {
  rm -rf .agent-reviews
}

# -----------------------------------------------------------------------------
# doc-drift.sh
# -----------------------------------------------------------------------------
echo "== doc-drift.sh =="

# 1. Non-matching command → exit 0, no report.
reset_sandbox
out=$(echo '{"tool_input":{"command":"echo hello"}}' | bash .claude/hooks/doc-drift.sh 2>&1; echo "EXIT:$?")
if [[ "$out" == *"EXIT:0"* ]] && [ ! -d .agent-reviews ]; then
  pass "non-matching command exits 0 silently"
else
  fail "non-matching command exits 0 silently" "$out"
fi

# 1b. Substring-only match (e.g. 'gh pr create' inside echo text) → no match.
reset_sandbox
export GH_STUB_PR=42
out=$(echo '{"tool_input":{"command":"echo testing the gh pr create matcher"}}' | bash .claude/hooks/doc-drift.sh 2>&1; echo "EXIT:$?")
if [[ "$out" == *"EXIT:0"* ]] && [ ! -d .agent-reviews ]; then
  pass "quoted 'gh pr create' substring inside echo does NOT trigger (word-boundary match)"
else
  fail "quoted substring should not trigger" "$out"
fi
unset GH_STUB_PR

# 2. gh pr create with no PR found → exit 0 silently.
reset_sandbox
unset GH_STUB_PR
out=$(echo '{"tool_input":{"command":"gh pr create --title x"}}' | bash .claude/hooks/doc-drift.sh 2>&1; echo "EXIT:$?")
if [[ "$out" == *"EXIT:0"* ]] && ! compgen -G ".agent-reviews/doc-drift-*.md" >/dev/null; then
  pass "gh pr create with no PR → exits 0, no report"
else
  fail "gh pr create with no PR → exits 0, no report" "$out"
fi

# 3. gh pr create with PR → exit 2, report file written, stderr has pointer.
reset_sandbox
export GH_STUB_PR=42 DOC_DRIFT_DRY_RUN=1
stderr_file="$TEST_DIR/stderr.txt"
out=$(echo '{"tool_input":{"command":"gh pr create --title x"}}' | bash .claude/hooks/doc-drift.sh 2>"$stderr_file"; echo "EXIT:$?")
report=$(find .agent-reviews -maxdepth 1 -name 'doc-drift-*.md' 2>/dev/null | head -1)
if [[ "$out" == *"EXIT:2"* ]] && [ -n "$report" ] && grep -q "PR #42" "$stderr_file"; then
  pass "gh pr create with PR → exit 2, report written, stderr points to it"
else
  fail "gh pr create with PR" "exit=$out report=$report stderr=$(cat "$stderr_file")"
fi

# 4. Fallback prompt when skill is missing.
if grep -q "docs/doc-map.yaml" "$report"; then
  pass "doc-drift fallback references docs/doc-map.yaml"
else
  fail "doc-drift fallback references docs/doc-map.yaml" "report: $(cat "$report")"
fi

# -----------------------------------------------------------------------------
# pr-review-resolver.sh
# -----------------------------------------------------------------------------
echo "== pr-review-resolver.sh =="
unset DOC_DRIFT_DRY_RUN
export RESOLVER_SLEEP_SECS=1 RESOLVER_DRY_RUN=1

# 5. Non-matching command → exit 0.
reset_sandbox
out=$(echo '{"tool_input":{"command":"git commit -m x"}}' | bash .claude/hooks/pr-review-resolver.sh 2>&1; echo "EXIT:$?")
if [[ "$out" == *"EXIT:0"* ]] && [ ! -d .agent-reviews ]; then
  pass "non-matching command exits 0 silently"
else
  fail "non-matching command exits 0 silently" "$out"
fi

# 5b. Commit message containing 'git push' substring → no match.
reset_sandbox
out=$(echo '{"tool_input":{"command":"git commit -m \"fix git push bug\""}}' | bash .claude/hooks/pr-review-resolver.sh 2>&1; echo "EXIT:$?")
if [[ "$out" == *"EXIT:0"* ]] && [ ! -d .agent-reviews ]; then
  pass "quoted 'git push' substring inside commit message does NOT trigger"
else
  fail "quoted substring should not trigger resolver" "$out"
fi

# 6. Push while on main → exit 0 before the sleep.
reset_sandbox
git checkout -q -b main 2>/dev/null || git checkout -q main
start=$(date +%s)
out=$(echo '{"tool_input":{"command":"git push origin main"}}' | bash .claude/hooks/pr-review-resolver.sh 2>&1; echo "EXIT:$?")
elapsed=$(($(date +%s) - start))
if [[ "$out" == *"EXIT:0"* ]] && [ "$elapsed" -lt 1 ] && [ ! -d .agent-reviews ]; then
  pass "git push on main → exits 0 before sleeping"
else
  fail "git push on main" "exit=$out elapsed=${elapsed}s"
fi

# 7. Push on feature branch with no PR → exits 0 after sleep, no report.
git checkout -q -b feature-x
reset_sandbox
unset GH_STUB_PR
out=$(echo '{"tool_input":{"command":"git push"}}' | bash .claude/hooks/pr-review-resolver.sh 2>&1; echo "EXIT:$?")
if [[ "$out" == *"EXIT:0"* ]] && ! compgen -G ".agent-reviews/pr-review-resolver-*.md" >/dev/null; then
  pass "feature branch push with no PR → exits 0, no report"
else
  fail "feature branch push with no PR" "$out"
fi

# 8. Push on feature branch with PR → exit 2, report written.
reset_sandbox
export GH_STUB_PR=99
stderr_file="$TEST_DIR/stderr.txt"
out=$(echo '{"tool_input":{"command":"git push"}}' | bash .claude/hooks/pr-review-resolver.sh 2>"$stderr_file"; echo "EXIT:$?")
report=$(find .agent-reviews -maxdepth 1 -name 'pr-review-resolver-*.md' 2>/dev/null | head -1)
if [[ "$out" == *"EXIT:2"* ]] && [ -n "$report" ] && grep -q "PR #99" "$stderr_file"; then
  pass "feature branch push with PR → exit 2, report written"
else
  fail "feature branch push with PR" "exit=$out report=$report"
fi

# 9. Fallback prompt content.
if grep -q "gh api repos" "$report"; then
  pass "resolver fallback references gh api repos"
else
  fail "resolver fallback references gh api repos" "report: $(cat "$report")"
fi

# 10. Lockfile dedupe: a newer run overwrites the lock, older run exits silently.
reset_sandbox
export GH_STUB_PR=99 RESOLVER_SLEEP_SECS=3
# Launch first hook in background; overwrite its lock mid-sleep; observe no report.
( echo '{"tool_input":{"command":"git push"}}' | bash .claude/hooks/pr-review-resolver.sh >/dev/null 2>&1; echo $? > "$TEST_DIR/bg_exit" ) &
bg_pid=$!
sleep 1
# Overwrite the lock with a different token.
mkdir -p .agent-reviews
echo "intruder-token" > .agent-reviews/.resolver-feature-x.lock
wait "$bg_pid"
bg_exit=$(cat "$TEST_DIR/bg_exit")
if [ "$bg_exit" = "0" ] && ! compgen -G ".agent-reviews/pr-review-resolver-*.md" >/dev/null; then
  pass "lockfile dedupe: superseded run exits 0 with no report"
else
  report_glob=$(compgen -G ".agent-reviews/pr-review-resolver-*.md" 2>/dev/null || true)
  fail "lockfile dedupe" "bg_exit=$bg_exit, report=$report_glob"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "== Summary =="
echo "PASS: $PASS"
echo "FAIL: $FAIL"
[ "$FAIL" -eq 0 ]
