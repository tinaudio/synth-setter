#!/usr/bin/env bash
# worktree-guard.sh — PreToolUse advisory for AGENTS.md's "Always work in an
# isolated git worktree" rule. Compares git's per-worktree git dir against the
# common git dir; equality means the current cwd is the primary checkout.
#
# Modes (WORKTREE_GUARD_MODE):
#   warn   default — loud stderr, exit 0 (never blocks)
#   block  exit 2 with the same message
#   off    no-op
set -euo pipefail

# shellcheck disable=SC2034  # read by log() in _lib.sh via ${HOOK_NAME:-unknown}
readonly HOOK_NAME="worktree-guard"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

[[ -f "${SCRIPT_DIR}/_lib.sh" ]] && {
  # shellcheck source=agent/hooks/_lib.sh
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/_lib.sh"
}

main() {
  local mode="${WORKTREE_GUARD_MODE:-warn}"
  case "$mode" in
    off) exit 0 ;;
    warn|block) ;;
    *)
      printf 'worktree-guard: ignoring unknown WORKTREE_GUARD_MODE=%s (use warn|block|off)\n' "$mode" >&2
      exit 0
      ;;
  esac

  # Drain stdin so the calling tool harness doesn't see SIGPIPE if it pipes JSON.
  cat >/dev/null 2>&1 || true

  local git_dir common_dir
  git_dir=$(git rev-parse --git-dir 2>/dev/null) || exit 0
  common_dir=$(git rev-parse --git-common-dir 2>/dev/null) || exit 0

  # Primary's --git-dir is relative (`.git`); linked worktrees' is absolute. Normalize before comparing.
  local abs_git_dir abs_common_dir
  abs_git_dir=$(cd "$git_dir" 2>/dev/null && pwd) || exit 0
  abs_common_dir=$(cd "$common_dir" 2>/dev/null && pwd) || exit 0

  [[ "$abs_git_dir" == "$abs_common_dir" ]] || exit 0

  local primary_root branch slug
  primary_root=$(dirname "$abs_common_dir")
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "<detached>")
  slug=${branch//\//-}
  [[ -z "$slug" || "$slug" == "<detached>" ]] && slug="scratch"

  command -v log >/dev/null 2>&1 && log "primary-checkout edit detected (mode=${mode}, branch=${branch})"

  local prefix override_hint
  if [[ "$mode" == "block" ]]; then
    prefix="BLOCKED"
    override_hint="Override: WORKTREE_GUARD_MODE=off (one-off edits) or =warn (advisory only)."
  else
    prefix="WARNING"
    override_hint="Override: WORKTREE_GUARD_MODE=off (one-off edits) or =block (fail-fast)."
  fi

  cat >&2 <<EOF
${prefix}: editing inside the primary checkout (${primary_root}).
AGENTS.md "Always" rule: the primary checkout is read-only; switch to an
isolated worktree before editing.

Recommended for branch '${branch}':
  git worktree add .claude/worktrees/${slug} ${branch}
  cd .claude/worktrees/${slug}

${override_hint}
EOF

  [[ "$mode" == "block" ]] && exit 2
  exit 0
}

main "$@"
