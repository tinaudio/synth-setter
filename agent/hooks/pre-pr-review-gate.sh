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
# Honor-system gate: an agent could fabricate a sentinel named after a real
# recent SHA, but it must already know a valid recent SHA on this branch.
# Filename-SHA + ancestry + first-parent lag are the floor; the gate does not
# inspect file contents or mtime.
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

readonly SENTINEL_PY="${SCRIPT_DIR}/../_shared/review_sentinel.py"
readonly MIN_REVIEW_BYTES=200
REVIEW_MAX_LAG="${REVIEW_MAX_LAG:-2}"

INPUT=$(cat)
COMMAND=$(jq -r '.tool_input.command // empty' 2>/dev/null <<<"$INPUT" || true)

# Here-string (not `echo |`) so SIGPIPE under pipefail can't fail-open. Allow
# leading whitespace at line start so `  gh pr create` is still gated.
if ! grep -qE '(^[[:space:]]*|[;|&`(][[:space:]]*)gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$)' <<<"$COMMAND"; then
  exit 0
fi

# shellcheck disable=SC2016  # intentional: no expansion wanted in the help block
readonly BLOCK_HELP='Run /repo-review-full-no-comments — it writes the rendered report to
.agent-reviews/repo-review-full-no-comments.<HEAD-sha>.md (filename owned by
agent/_shared/review_sentinel.py). Then re-run with the path in the command
— recommended as a trailing comment so other hooks still fire:

  gh pr create --title "..." --body "..."  # REVIEW_FULL=.agent-reviews/repo-review-full-no-comments.<sha>.md

REVIEW_FULL=<value> runs to the next whitespace or shell metachar. Quoted
paths are NOT recognized — the value-character class excludes both single
and double quotes entirely, so REVIEW_FULL="..." reads as missing-REVIEW_FULL.
Pass the path bare.

The encoded SHA must be an ancestor of HEAD and within REVIEW_MAX_LAG
first-parent commits behind it (default 2; override via the REVIEW_MAX_LAG
env var, must be a non-negative integer).'

block() {
  local reason="$1"
  log "blocking: $reason"
  printf 'BLOCKED: %s\n\n%s\n' "$reason" "$BLOCK_HELP" >&2
  exit 2
}

if [[ ! "$REVIEW_MAX_LAG" =~ ^[0-9]+$ ]]; then
  block "REVIEW_MAX_LAG must be a non-negative integer, got: ${REVIEW_MAX_LAG}"
fi

if [[ ! -f "$SENTINEL_PY" ]]; then
  block "missing companion helper at ${SENTINEL_PY} (gate cannot parse sentinel filenames without it)"
fi

# Allow only one REVIEW_FULL= occurrence — multiple would silently pick the
# first and an agent juggling two reviews could pass against the wrong one.
# `|| true` traps grep's no-match exit-1; pipefail would otherwise abort the
# script before the empty-path branch below can emit its own BLOCKED message.
review_full_matches=$(grep -oE 'REVIEW_FULL=[^[:space:]#;|&`()<>"'"'"']+' <<<"$COMMAND" || true)
review_full_count=$(printf '%s' "$review_full_matches" | grep -c . || true)
if [[ "$review_full_count" -gt 1 ]]; then
  block "multiple REVIEW_FULL= tokens in command (${review_full_count}); pick exactly one"
fi

# `grep` returns 1 on no match (under pipefail that aborts the script) and
# `head -1`'s SIGPIPE on closing the pipe early would also surface — `|| true`
# traps both so the empty-path branch below can emit a clean BLOCKED message.
REVIEW_PATH=$(grep -oE 'REVIEW_FULL=[^[:space:]#;|&`()<>"'"'"']+' <<<"$COMMAND" | head -1 | cut -d= -f2- || true)

if [[ -z "$REVIEW_PATH" ]]; then
  block "gh pr create is missing REVIEW_FULL=<path-to-review-file>"
fi
if [[ ! -f "$REVIEW_PATH" ]]; then
  block "REVIEW_FULL path does not point at a file: $REVIEW_PATH"
fi

# Inode-metadata size read (no file contents touched). GNU `stat -c %s`
# vs BSD/macOS `stat -f %z` — try the GNU form first and fall back; both
# return a bare integer suitable for `-lt`.
if ! review_size=$(stat -c %s "$REVIEW_PATH" 2>/dev/null); then
  review_size=$(stat -f %z "$REVIEW_PATH" 2>/dev/null || echo "")
fi
if [[ -z "$review_size" ]]; then
  block "could not stat REVIEW_FULL path: $REVIEW_PATH"
fi
if [[ "$review_size" -lt "$MIN_REVIEW_BYTES" ]]; then
  block "REVIEW_FULL file is suspiciously small (${review_size} < ${MIN_REVIEW_BYTES} bytes — likely a touch-bypass): $REVIEW_PATH"
fi

# Filename -> SHA via the shared Python helper (single source of truth). The
# helper exits 1 when the basename doesn't match the sentinel pattern, exit 2
# on usage error or ValueError — surface those separately so a missing
# python3 / broken helper isn't misreported as "bad filename".
helper_stderr=$(mktemp)
trap 'rm -f "$helper_stderr"' EXIT
if review_sha=$(python3 "$SENTINEL_PY" parse "$REVIEW_PATH" 2>"$helper_stderr"); then
  :
else
  helper_rc=$?
  helper_msg=$(cat "$helper_stderr")
  case "$helper_rc" in
    1)
      block "REVIEW_FULL filename does not match the sentinel pattern 'repo-review-full-no-comments.<40-char-sha>.md': $REVIEW_PATH"
      ;;
    *)
      block "internal helper error (exit ${helper_rc}) parsing ${REVIEW_PATH}: ${helper_msg:-no stderr captured}"
      ;;
  esac
fi

# Ancestry: a SHA on a sibling branch (or one rewritten by rebase/amend) is
# rejected outright — the review covers a different line of history.
if ! git merge-base --is-ancestor "$review_sha" HEAD 2>/dev/null; then
  block "review SHA ${review_sha} is not an ancestor of HEAD (rebase/amend rewrote history? run /repo-review-full-no-comments again)"
fi

# First-parent lag — merging origin/main counts as one commit, not the
# dozens it brings in.
lag=$(git rev-list "${review_sha}..HEAD" --first-parent --count 2>/dev/null || echo "")
if [[ -z "$lag" ]]; then
  block "could not compute first-parent lag between review SHA ${review_sha} and HEAD"
fi
if [[ "$lag" -gt "$REVIEW_MAX_LAG" ]]; then
  block "review is ${lag} first-parent commits behind HEAD (max ${REVIEW_MAX_LAG}; set REVIEW_MAX_LAG=N to widen)"
fi

log "review accepted: ${REVIEW_PATH} (sha=${review_sha}, lag=${lag}/${REVIEW_MAX_LAG}, size=${review_size}B)"
exit 0
