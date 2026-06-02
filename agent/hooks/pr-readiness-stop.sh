#!/usr/bin/env bash
# pr-readiness-stop.sh — Stop hook enforcing docs/pr-readiness-loop.md. When a
# turn tries to end while an open PR for the current branch has unmet readiness
# gates, blocks the Stop (exit 2) with a message so the agent keeps driving the
# loop instead of treating "I pushed the fix" as "the PR is ready".
#
# Modes (PR_READINESS_GATE):
#   block  default — exit 2 with the failing-gate message
#   warn   loud stderr, exit 0 (never blocks)
#   off    no-op
#
# Deterministic vs delegated: gate 1 (CI green) and gate 2
# (mergeable=MERGEABLE) are checked here against `gh pr checks` / `gh pr view`.
# Gates 3 (every review comment answered) and 4 (no fresh Copilot findings) are
# not robustly decidable in bash — the block message points the agent at
# /pr-readiness and docs/pr-readiness-loop.md for those.
set -euo pipefail

# shellcheck disable=SC2034  # read by log() in _lib.sh via ${HOOK_NAME:-unknown}
readonly HOOK_NAME="pr-readiness-stop"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

[[ -f "${SCRIPT_DIR}/_lib.sh" ]] && {
  # shellcheck source=agent/hooks/_lib.sh
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/_lib.sh"
}

# `in_headless_context` reports whether this turn belongs to a headless runner
# whose Stop must never block. Three independent signals, any one suffices:
#   - PR_READINESS_HEADLESS — exported by _lib.sh's run_agent_prompt, the single
#     chokepoint the pr-review-resolver/doc-drift hooks spawn agents through.
#   - CI — set by GitHub Actions and most CI runners.
#   - cwd under .agent-reviews/worktrees/ — the detached worktree those headless
#     agents run in, a fallback for runners that bypass run_agent_prompt.
in_headless_context() {
  [[ -n "${PR_READINESS_HEADLESS:-}" ]] && return 0
  [[ -n "${CI:-}" ]] && return 0
  case "$PWD" in
    */.agent-reviews/worktrees/*) return 0 ;;
  esac
  return 1
}

# `in_primary_checkout` mirrors worktree-guard's detection: git's per-worktree
# git dir equals the common git dir only in the primary checkout. Returns 1
# (not primary) on any resolution failure so a probe error can't force a block.
in_primary_checkout() {
  local git_dir common_dir abs_git_dir abs_common_dir
  git_dir=$(git rev-parse --git-dir 2>/dev/null) || return 1
  common_dir=$(git rev-parse --git-common-dir 2>/dev/null) || return 1
  abs_git_dir=$(cd "$git_dir" 2>/dev/null && pwd) || return 1
  abs_common_dir=$(cd "$common_dir" 2>/dev/null && pwd) || return 1
  [[ "$abs_git_dir" == "$abs_common_dir" ]]
}

# `ci_is_green` returns 0 only when every check has concluded successfully.
# `gh pr checks` exits non-zero while checks are pending (8) or failing (1);
# both count as "gate not yet holding" per docs/pr-readiness-loop.md — the hook
# does not sleep, it tells the agent to watch/loop.
ci_is_green() {
  gh pr checks "$1" >/dev/null 2>&1
}

main() {
  # Drain stdin before any dispatch so off/headless early-exits don't leave the
  # harness's JSON write blocked on a full pipe (SIGPIPE under pipefail).
  cat >/dev/null 2>&1 || true

  local mode="${PR_READINESS_GATE:-block}"
  case "$mode" in
    off) exit 0 ;;
    block|warn) ;;
    *)
      printf 'pr-readiness-stop: ignoring unknown PR_READINESS_GATE=%s (use block|warn|off)\n' "$mode" >&2
      exit 0
      ;;
  esac

  in_headless_context && exit 0
  command -v gh >/dev/null 2>&1 || exit 0
  in_primary_checkout && exit 0

  local branch pr
  branch=$(git branch --show-current 2>/dev/null || true)
  [[ -n "$branch" ]] || exit 0
  pr=$(gh pr view "$branch" --json number -q .number 2>/dev/null || true)
  [[ -n "$pr" ]] || exit 0

  local mergeable failed_gate=""
  if ! ci_is_green "$pr"; then
    failed_gate="Gate 1 (CI not fully green — checks failing or still pending)"
  else
    mergeable=$(gh pr view "$pr" --json mergeable -q .mergeable 2>/dev/null || true)
    if [[ "$mergeable" != "MERGEABLE" ]]; then
      failed_gate="Gate 2 (mergeable=${mergeable:-UNKNOWN}, want MERGEABLE)"
    fi
  fi

  [[ -z "$failed_gate" ]] && exit 0

  declare -F ensure_reviews_dir >/dev/null 2>&1 && ensure_reviews_dir
  declare -F log >/dev/null 2>&1 && log "blocking Stop for PR #${pr} (branch ${branch}): ${failed_gate}"

  local prefix override_hint
  if [[ "$mode" == "block" ]]; then
    prefix="BLOCKED"
    override_hint="Override: PR_READINESS_GATE=warn (advisory only) or =off (no-op)."
  else
    prefix="WARNING"
    override_hint="Override: PR_READINESS_GATE=off (no-op) or =block (fail-fast)."
  fi

  cat >&2 <<EOF
${prefix}: PR #${pr} (branch ${branch}) is not ready — ${failed_gate}.
AGENTS.md "After every push, drive the readiness loop until all four gates
hold." Do not end the turn yet.

Run /pr-readiness to drive the loop (watch CI, check mergeable, reply inline to
every open review comment, wait for Copilot). This hook checks gates 1-2 only;
also confirm gate 3 (every open review comment has an inline reply) and gate 4
(no fresh Copilot findings since the last push) — see docs/pr-readiness-loop.md.

${override_hint}
EOF

  [[ "$mode" == "block" ]] && exit 2
  exit 0
}

main "$@"
