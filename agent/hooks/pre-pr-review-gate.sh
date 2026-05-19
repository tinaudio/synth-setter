#!/usr/bin/env bash
# =============================================================================
# pre-pr-review-gate.sh — PreToolUse hook, gates `gh pr create`
# =============================================================================
#
# PURPOSE
# -------
# Block `gh pr create` unless the command carries `REVIEW_FULL=<path>` and that
# path points at a real `/repo-review-full-no-comments` report freshly written
# against the current HEAD. Forces the agent to actually produce and save the
# review body, not just append an acknowledgment flag. The gate is honor-system
# — the agent can still write a fake file — but it raises the floor from "type a
# token" to "produce text that looks like a review and is newer than HEAD."
#
# CONTRACT — what the command line must carry
#   REVIEW_FULL=<path>
#     - <path> resolves to an existing, non-empty file (relative to cwd).
#     - File mtime ≥ HEAD's commit time, so a review predating the latest
#       commit is rejected. This is the part the old token-only gate could not
#       enforce.
#     - File starts with either `# repo-review-full-no-comments` (the skill's
#       Step-7 header) or a `PASS` line (the skill's empty-findings short
#       form). Header-mode reports must be ≥200 bytes to block trivial stubs.
#
# INPUT (stdin)
#   JSON with .tool_input.command — the Bash command the agent is about to run.
#
# INVOCATION
#   Registered in the agent's settings (e.g. .claude/settings.json) as a
#   PreToolUse hook on Bash. The handler-level `if: "Bash(gh pr create *)"`
#   guard is unreliable for PreToolUse hooks (see PR #1090 history), so this
#   script re-validates the command itself — same defensive pattern
#   doc-drift.sh / pr-review-resolver.sh use. Non-matching commands exit 0
#   silently.
# =============================================================================
set -euo pipefail

export HOOK_NAME="pre-pr-review-gate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent/hooks/_lib.sh
# shellcheck disable=SC1091  # _lib.sh resolved at runtime; not staged together
source "${SCRIPT_DIR}/_lib.sh"

INPUT=$(cat)
COMMAND=$(jq -r '.tool_input.command // empty' 2>/dev/null <<<"$INPUT" || true)

# Here-string (not `echo |`) so SIGPIPE under pipefail can't fail-open. Allow
# leading whitespace at line start so `  gh pr create` is still gated.
if ! grep -qE '(^[[:space:]]*|[;|&`(][[:space:]]*)gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$)' <<<"$COMMAND"; then
  exit 0
fi

# shellcheck disable=SC2016  # intentional: no expansion wanted in the help block
BLOCK_HELP='Run /repo-review-full-no-comments, save the rendered report to a file under
.agent-reviews/ (e.g. `.agent-reviews/pre-pr-<slug>.md`) AFTER your latest
commit, then re-run with the path in the command — recommended as a trailing
comment so other hooks still fire:

  gh pr create --title "..." --body "..."  # REVIEW_FULL=.agent-reviews/pre-pr-<slug>.md

The file must look like a /repo-review-full-no-comments report (start with
`# repo-review-full-no-comments` or `PASS`) and have mtime ≥ HEAD commit time.'

block() {
  local reason="$1"
  log "blocking: $reason"
  printf 'BLOCKED: %s\n\n%s\n' "$reason" "$BLOCK_HELP" >&2
  exit 2
}

# Match REVIEW_FULL=<path>; stop at shell metacharacters that would never
# appear in a sensible path. Backtick excluded intentionally — paths with
# backticks are not worth supporting and would invite command-injection
# surprises if the path is later re-quoted.
# `grep` returns 1 on no match; under `pipefail` that would abort the script
# before the empty-path branch runs, so trap the no-match case explicitly.
REVIEW_PATH=$(grep -oE 'REVIEW_FULL=[^[:space:]#;|&`()<>"'"'"']+' <<<"$COMMAND" | head -1 | cut -d= -f2- || true)

if [[ -z "$REVIEW_PATH" ]]; then
  block "gh pr create is missing REVIEW_FULL=<path-to-review-file>"
fi
if [[ ! -f "$REVIEW_PATH" ]]; then
  block "REVIEW_FULL path does not point at a file: $REVIEW_PATH"
fi
if [[ ! -s "$REVIEW_PATH" ]]; then
  block "REVIEW_FULL file is empty: $REVIEW_PATH"
fi

# Portable mtime: GNU stat first (Linux), BSD stat fallback (macOS).
review_mtime=$(stat -c %Y "$REVIEW_PATH" 2>/dev/null || stat -f %m "$REVIEW_PATH" 2>/dev/null || echo 0)
head_ctime=$(git log -1 --format=%ct HEAD 2>/dev/null || echo 0)
if [[ "$review_mtime" -lt "$head_ctime" ]]; then
  block "review file is older than HEAD commit (mtime=$review_mtime, HEAD=$head_ctime) — rerun /repo-review-full-no-comments and save the new output"
fi

if grep -qE '^PASS\b' "$REVIEW_PATH"; then
  log "PASS report accepted: $REVIEW_PATH"
elif grep -qE '^# repo-review-full-no-comments\b' "$REVIEW_PATH"; then
  size=$(wc -c <"$REVIEW_PATH")
  if [[ "$size" -lt 200 ]]; then
    block "REVIEW_FULL file has the header but is suspiciously short (<200 bytes): $REVIEW_PATH"
  fi
  log "header report accepted: $REVIEW_PATH ($size bytes)"
else
  block "REVIEW_FULL file does not look like a /repo-review-full-no-comments report (missing '# repo-review-full-no-comments' header or 'PASS' marker): $REVIEW_PATH"
fi

exit 0
