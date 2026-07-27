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
# Gates 1-3 come from agent/_shared/pr_readiness_probe.sh; gate 4 is advisory.
# ACTION_REQUIRED (1) and WAIT (8) block, while probe errors fail open.
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
#     chokepoint the doc-drift hook spawns its agent through.
#   - CI — set by GitHub Actions and most CI runners.
#   - cwd under .agent-reviews/worktrees/ — the detached worktree that headless
#     agent runs in, a fallback for runners that bypass run_agent_prompt.
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
# (not primary) on any git resolution failure so it can't force a block.
in_primary_checkout() {
  local git_dir common_dir abs_git_dir abs_common_dir
  git_dir=$(git rev-parse --git-dir 2>/dev/null) || return 1
  common_dir=$(git rev-parse --git-common-dir 2>/dev/null) || return 1
  abs_git_dir=$(cd "$git_dir" 2>/dev/null && pwd) || return 1
  abs_common_dir=$(cd "$common_dir" 2>/dev/null && pwd) || return 1
  [[ "$abs_git_dir" == "$abs_common_dir" ]]
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

  local probe probe_out probe_rc=0
  probe="${SCRIPT_DIR}/../_shared/pr_readiness_probe.sh"
  if [[ ! -f "$probe" ]]; then
    { declare -F log >/dev/null 2>&1 && log "probe missing at ${probe}; skipping readiness gates (fail-open)"; } || true
    exit 0
  fi
  # --gates-only: the hook only consumes the exit code on the passing path, so
  # skip the probe's advisory Copilot lookup (two paginated REST calls).
  probe_out=$(bash "$probe" --gates-only "$pr" 2>&1) || probe_rc=$?
  [[ "$probe_rc" -eq 0 ]] && exit 0
  if [[ "$probe_rc" -ne 1 && "$probe_rc" -ne 8 ]]; then
    { declare -F log >/dev/null 2>&1 && log "probe exited ${probe_rc} for PR #${pr}; fail-open: ${probe_out}"; } || true
    exit 0
  fi

  local next_step
  if [[ "$probe_rc" -eq 8 ]]; then
    next_step="Only transient work remains; continue monitoring with the "
    next_step+="action-aware command in /pr-readiness."
  else
    next_step="Do not keep polling: remediate each ACTION gate above, push "
    next_step+="if needed, then re-probe."
  fi

  declare -F ensure_reviews_dir >/dev/null 2>&1 && ensure_reviews_dir
  declare -F log >/dev/null 2>&1 && log "blocking Stop for PR #${pr} (branch ${branch}): readiness gates failing"

  local prefix override_hint
  if [[ "$mode" == "block" ]]; then
    prefix="BLOCKED"
    override_hint="Override: PR_READINESS_GATE=warn (advisory only) or =off (no-op)."
  else
    prefix="WARNING"
    override_hint="Override: PR_READINESS_GATE=off (no-op) or =block (fail-fast)."
  fi

  cat >&2 <<EOF
${prefix}: PR #${pr} (branch ${branch}) is not ready — readiness probe report:

${probe_out}

${next_step}

AGENTS.md "After every push, drive the readiness loop until all four gates
hold." Do not end the turn yet. Run /pr-readiness for the full procedure and
traps in docs/pr-readiness-loop.md.

${override_hint}
EOF

  [[ "$mode" == "block" ]] && exit 2
  exit 0
}

main "$@"
