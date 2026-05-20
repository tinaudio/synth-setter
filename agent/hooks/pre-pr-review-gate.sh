#!/usr/bin/env bash
# =============================================================================
# pre-pr-review-gate.sh — PreToolUse hook, gates `gh pr create`
# =============================================================================
#
# PURPOSE
# -------
# Block `gh pr create` unless the command carries `REVIEW_FULL=<path>` and that
# path points at a sentinel review file whose name encodes a commit SHA that
# is reachable from HEAD within `REVIEW_MAX_LAG` first-parent commits.
#
# The sentinel format (basename) is::
#
#     repo-review-full-no-comments.<40-char-sha>.md
#
# Filename format is owned by `agent/_shared/review_sentinel.py`; this script
# shells out to it for parsing so the format has exactly one source of truth
# (shared with the `/repo-review-full-no-comments` skill).
#
# CONTRACT — what the command line must carry
#   REVIEW_FULL=<path>
#     - <path> resolves to an existing file (relative to cwd) that is at least
#       200 bytes (cheap stub-bypass guard — `touch` produces 0 bytes).
#     - The file's basename matches the sentinel pattern and encodes a SHA.
#     - That SHA is an ancestor of HEAD AND is at most `REVIEW_MAX_LAG`
#       first-parent commits behind HEAD (default 2; override via env). The
#       `--first-parent` mode means merging `main` into the branch counts as
#       one commit, not the hundreds it brings in.
#
# The gate stays honor-system — a determined agent can still write a file
# named after a real recent SHA — but the filename-encoded-SHA approach
# raises the floor materially: an agent must know a valid recent SHA to
# fabricate one, the freshness check is no longer mtime-based (so cross-
# worktree / cross-clone timestamp games don't help), and we no longer
# inspect file contents at all (Copilot's PR-#1175 finding about line-
# anchored vs file-anchored grep is moot — there is no grep).
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

SENTINEL_PY="${SCRIPT_DIR}/../_shared/review_sentinel.py"
MIN_REVIEW_BYTES=200
REVIEW_MAX_LAG="${REVIEW_MAX_LAG:-2}"

INPUT=$(cat)
COMMAND=$(jq -r '.tool_input.command // empty' 2>/dev/null <<<"$INPUT" || true)

# Here-string (not `echo |`) so SIGPIPE under pipefail can't fail-open. Allow
# leading whitespace at line start so `  gh pr create` is still gated.
if ! grep -qE '(^[[:space:]]*|[;|&`(][[:space:]]*)gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$)' <<<"$COMMAND"; then
  exit 0
fi

# shellcheck disable=SC2016  # intentional: no expansion wanted in the help block
BLOCK_HELP='Run /repo-review-full-no-comments — it writes the rendered report to
.agent-reviews/repo-review-full-no-comments.<HEAD-sha>.md (filename owned by
agent/_shared/review_sentinel.py). Then re-run with the path in the command
— recommended as a trailing comment so other hooks still fire:

  gh pr create --title "..." --body "..."  # REVIEW_FULL=.agent-reviews/repo-review-full-no-comments.<sha>.md

The encoded SHA must be an ancestor of HEAD and within REVIEW_MAX_LAG
first-parent commits behind it (default 2; override via the REVIEW_MAX_LAG
env var if you have a justified larger gap).'

block() {
  local reason="$1"
  log "blocking: $reason"
  printf 'BLOCKED: %s\n\n%s\n' "$reason" "$BLOCK_HELP" >&2
  exit 2
}

# `grep` returns 1 on no match; under `pipefail` that would abort the script
# before the empty-path branch runs, so trap the no-match case explicitly.
REVIEW_PATH=$(grep -oE 'REVIEW_FULL=[^[:space:]#;|&`()<>"'"'"']+' <<<"$COMMAND" | head -1 | cut -d= -f2- || true)

if [[ -z "$REVIEW_PATH" ]]; then
  block "gh pr create is missing REVIEW_FULL=<path-to-review-file>"
fi
if [[ ! -f "$REVIEW_PATH" ]]; then
  block "REVIEW_FULL path does not point at a file: $REVIEW_PATH"
fi

review_size=$(wc -c <"$REVIEW_PATH")
if [[ "$review_size" -lt "$MIN_REVIEW_BYTES" ]]; then
  block "REVIEW_FULL file is suspiciously small (${review_size} < ${MIN_REVIEW_BYTES} bytes — likely a touch-bypass): $REVIEW_PATH"
fi

# Filename -> SHA via the shared Python helper (single source of truth).
if ! review_sha=$(python3 "$SENTINEL_PY" parse "$REVIEW_PATH" 2>/dev/null); then
  block "REVIEW_FULL filename does not match the sentinel pattern 'repo-review-full-no-comments.<40-char-sha>.md': $REVIEW_PATH"
fi

# Ancestry: a SHA on a sibling branch (or one rewritten by rebase/amend) is
# rejected outright — the review covers a different line of history.
if ! git merge-base --is-ancestor "$review_sha" HEAD 2>/dev/null; then
  block "review SHA ${review_sha} is not an ancestor of HEAD (rebase/amend rewrote history? run /repo-review-full-no-comments again)"
fi

# Lag in first-parent space: merging origin/main into the branch counts as
# one commit, not the dozens it brings in. That's the behavior we want.
lag=$(git rev-list "${review_sha}..HEAD" --first-parent --count 2>/dev/null || echo "")
if [[ -z "$lag" ]]; then
  block "could not compute first-parent lag between review SHA ${review_sha} and HEAD"
fi
if [[ "$lag" -gt "$REVIEW_MAX_LAG" ]]; then
  block "review is ${lag} first-parent commits behind HEAD (max ${REVIEW_MAX_LAG}; set REVIEW_MAX_LAG=N to widen)"
fi

log "review accepted: ${REVIEW_PATH} (sha=${review_sha}, lag=${lag}/${REVIEW_MAX_LAG}, size=${review_size}B)"
exit 0
