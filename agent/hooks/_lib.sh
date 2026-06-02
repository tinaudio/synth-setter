#!/usr/bin/env bash
# _lib.sh — shared helpers for agent/hooks/*. See each function's own block
# comment for its contract. Sourced by every hook; intentionally omits
# `set -euo pipefail` so each caller chooses its own error-handling regime.

# Absolute paths so a hook that `cd`s into a worktree still writes reports
# where the receiving session reads them. Falls back to cwd outside a git tree.
_repo_root_for_hooks() {
  local common_dir parent
  common_dir=$(git rev-parse --git-common-dir 2>/dev/null) || return 1
  case "$common_dir" in
    /*) parent=$(dirname "$common_dir") ;;
    *)  parent=$(cd "$(dirname "$common_dir")" 2>/dev/null && pwd) ;;
  esac
  [[ -n "$parent" ]] || return 1
  printf '%s\n' "$parent"
}

_REPO_ROOT_FOR_HOOKS=$(_repo_root_for_hooks 2>/dev/null || pwd)
REVIEWS_DIR="${_REPO_ROOT_FOR_HOOKS}/.agent-reviews"
HOOK_LOG="${REVIEWS_DIR}/.hook.log"
WORKTREES_DIR="${REVIEWS_DIR}/worktrees"

ensure_reviews_dir() {
  mkdir -p "$REVIEWS_DIR"
}

log() {
  # Usage: log "message"
  # Appends "<iso8601> [<hook_name>] <msg>" to .agent-reviews/.hook.log.
  # Silent on failure — log path may not exist early in a run.
  local ts
  ts=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  printf '%s [%s] %s\n' "$ts" "${HOOK_NAME:-unknown}" "$*" >> "$HOOK_LOG" 2>/dev/null || true
}

gen_id() {
  # Portable unique ID for report filenames and lock tokens.
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen
  elif [[ -r /proc/sys/kernel/random/uuid ]]; then
    cat /proc/sys/kernel/random/uuid
  else
    printf '%s-%s' "$(date +%s)" "$$"
  fi
}

default_branch() {
  # Resolve the repo's default branch name. Order:
  #   1. origin/HEAD symbolic-ref (best: tracks the real remote default)
  #   2. local 'main' if it exists
  #   3. local 'master' if it exists
  #   4. 'main' (final fallback)
  local ref b
  ref=$(git symbolic-ref --quiet refs/remotes/origin/HEAD 2>/dev/null || true)
  if [[ -n "$ref" ]]; then
    echo "${ref##*/}"
    return 0
  fi
  # `b` is declared `local` above so the loop variable doesn't leak into
  # the caller's scope (`for` does not auto-localize).
  for b in main master; do
    git show-ref --verify --quiet "refs/heads/$b" 2>/dev/null && { echo "$b"; return 0; }
    git show-ref --verify --quiet "refs/remotes/origin/$b" 2>/dev/null && { echo "$b"; return 0; }
  done
  echo "main"
}

has_skill() {
  # Usage: has_skill <skill-name>
  # Returns 0 if <name>/SKILL.md exists in any known skill location.
  # Worktree-aware: also checks the main repo (via $_REPO_ROOT_FOR_HOOKS) so
  # hooks running from a linked worktree still find skills installed only in
  # the primary checkout. Unmatched plugin globs expand to their literal
  # pattern, which the `-f` check safely rejects without spurious warnings.
  local name="$1" home="${HOME:-}" p
  local paths=(
    "agent/skills/${name}/SKILL.md"
    ".claude/skills/${name}/SKILL.md"
  )
  if [[ -n "$home" ]]; then
    paths+=(
      "${home}/.claude/skills/${name}/SKILL.md"
      "${home}/.codex/skills/${name}/SKILL.md"
      "${home}"/.claude/plugins/*/skills/"${name}"/SKILL.md
      "${home}"/.codex/plugins/*/skills/"${name}"/SKILL.md
    )
  fi
  if [[ -n "${_REPO_ROOT_FOR_HOOKS:-}" ]]; then
    paths+=(
      "${_REPO_ROOT_FOR_HOOKS}/agent/skills/${name}/SKILL.md"
      "${_REPO_ROOT_FOR_HOOKS}/.claude/skills/${name}/SKILL.md"
    )
  fi
  for p in "${paths[@]}"; do
    [[ -f "$p" ]] && return 0
  done
  return 1
}

run_agent_prompt() {
  # Usage: run_agent_prompt <prompt>
  # Wraps the headless CLI in `timeout` so a hung agent surfaces as exit 124.
  # Default 800s leaves headroom inside both harness ceilings: doc-drift's
  # 900s (800 + 10 kill-after = 810 < 900) and pr-review-resolver's
  # 1200s after a 360s sleep (360 + 800 + 10 = 1170 < 1200). Operators
  # raising AGENT_TIMEOUT_SECS must check the matching .claude/settings.json
  # `timeout` field or the harness will SIGKILL.
  local prompt="$1" cli="${AGENT_HEADLESS:-}" candidate
  local timeout_secs="${AGENT_TIMEOUT_SECS:-800}"
  if [[ -z "$cli" ]]; then
    for candidate in claude codex; do
      if command -v "$candidate" >/dev/null 2>&1; then
        cli="$candidate"
        break
      fi
    done
  fi
  # GNU coreutils ships as `timeout` on Linux and `gtimeout` on macOS
  # (Homebrew). With neither on PATH, fall back to an unwrapped run and log —
  # an operator can install coreutils to re-enable the hung-agent SIGKILL.
  # `--kill-after=10` SIGKILLs a child that ignores the SIGTERM the soft
  # timeout sends — grandchildren the agent CLI spawned otherwise survive.
  local timeout_bin=""
  if command -v timeout >/dev/null 2>&1; then
    timeout_bin="timeout"
  elif command -v gtimeout >/dev/null 2>&1; then
    timeout_bin="gtimeout"
  else
    log "no GNU timeout/gtimeout on PATH; running agent without timeout enforcement"
  fi
  local -a cmd
  case "$cli" in
    claude) cmd=(claude -p "$prompt") ;;
    codex)  cmd=(codex exec "$prompt") ;;
    *)
      printf 'No supported headless agent CLI found; install claude or codex (AGENT_HEADLESS=%s).\n' \
        "${AGENT_HEADLESS:-}" >&2
      return 127
      ;;
  esac
  # Mark the spawned agent's session headless so its Stop hook
  # (pr-readiness-stop.sh) never blocks — the resolver/doc-drift runners must
  # finish and rewake the parent, not deadlock on their own readiness gates.
  export PR_READINESS_HEADLESS=1
  if [[ -n "$timeout_bin" ]]; then
    "$timeout_bin" --kill-after=10 "$timeout_secs" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

emit_rewake_stamp() {
  # Usage: emit_rewake_stamp <slug> <pr> <branch> <head_sha> <review_file> [tail]
  # Prints the metadata-stamped advisory pointer on stderr so a session that
  # receives a cross-session leak can compare the origin HEAD to its own and
  # discard. AGENTS.md documents the receiving-side contract.
  local slug="$1" pr="$2" branch="$3" head_sha="$4" review_file="$5" tail="${6:-}" short_head_sha
  short_head_sha=$(git rev-parse --short=7 "$head_sha" 2>/dev/null || printf '%s' "${head_sha:0:7}")
  printf '%s report for PR #%s (branch %s, origin HEAD %s) at %s.%s\n' \
    "$slug" "$pr" "$branch" "$short_head_sha" "$review_file" \
    "${tail:+ $tail}" >&2
  printf 'If your current HEAD is not %s, this advisory crossed sessions — verify before acting.\n' \
    "$short_head_sha" >&2
}

make_isolated_worktree() {
  # Usage: make_isolated_worktree <slug> <head_sha>
  # Echoes the new worktree path on stdout; returns non-zero (no output) on
  # failure. Caller registers cleanup via `trap '...' EXIT` + remove_worktree.
  local slug="$1" sha="$2" wt
  ensure_reviews_dir
  mkdir -p "$WORKTREES_DIR"
  wt="${WORKTREES_DIR}/${slug}-${sha:0:7}-$(gen_id)"
  if git worktree add --detach "$wt" "$sha" >/dev/null 2>&1; then
    printf '%s\n' "$wt"
    return 0
  fi
  return 1
}

remove_worktree() {
  # Usage: remove_worktree <path>
  # Best-effort cleanup; safe to call on a missing path. Prunes the
  # registration so `git worktree list` doesn't accrete dangling entries.
  local wt="$1"
  [[ -n "$wt" ]] || return 0
  if [[ -d "$wt" ]]; then
    git worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
  fi
  git worktree prune >/dev/null 2>&1 || true
}

sweep_stale_worktrees() {
  # Usage: sweep_stale_worktrees <slug> [max_age_minutes]
  # Removes leaked <slug>-* worktrees the EXIT trap missed (e.g. when the
  # harness SIGKILL'd the hook at its timeout ceiling). max_age default 30.
  local slug="$1" max_age="${2:-30}" wt
  [[ -d "$WORKTREES_DIR" ]] || return 0
  while IFS= read -r -d '' wt; do
    log "sweeping stale worktree: $wt"
    remove_worktree "$wt"
  done < <(find "$WORKTREES_DIR" -maxdepth 1 -type d -name "${slug}-*" -mmin "+${max_age}" -print0 2>/dev/null)
}

run_review() {
  # Usage: run_review <slug> <meta> <prompt> [<dry_run_env_var>]
  #
  # Owns the DRY_RUN-vs-headless-agent dispatch shared by doc-drift.sh and
  # pr-review-resolver.sh. Generates the report path (`.agent-reviews/<slug>-<uuid>.md`),
  # invokes the headless agent (or writes a stub under dry-run), and on
  # failure writes a verbose report with the exit code, prompt, and stderr
  # tail. Echoes the resulting report path on stdout so the caller can
  # surface it. Always returns 0 — the caller decides the hook's own exit.
  #
  # `<slug>`             short name used in the report header and filename
  #                      (e.g. "doc-drift", "pr-review-resolver").
  # `<meta>`             printf-ready block of "Key: value" lines.
  # `<prompt>`           text passed to run_agent_prompt; stubbed under dry-run.
  # `<dry_run_env_var>`  optional name of the env var that gates dry-run mode
  #                      (e.g. "DOC_DRIFT_DRY_RUN"); if its value is "1" the
  #                      stub branch fires. Omit for non-dry-run-capable hooks.
  local slug="$1" meta="$2" prompt="$3" dry_run_var="${4:-}"
  local review_id review_file stderr_file dry_run exit_code
  review_id=$(gen_id)
  review_file="${REVIEWS_DIR}/${slug}-${review_id}.md"
  stderr_file="${REVIEWS_DIR}/${slug}-${review_id}.stderr"
  dry_run=0
  if [[ -n "$dry_run_var" ]]; then
    # Bash indirect expansion to read the named env var; defaults to 0.
    dry_run="${!dry_run_var:-0}"
  fi
  if [[ "$dry_run" == "1" ]]; then
    log "DRY_RUN: writing stub report"
    printf '# %s (dry-run)\n%s\n## Prompt\n%s\n' "$slug" "$meta" "$prompt" > "$review_file"
    printf '%s\n' "$review_file"
    return 0
  fi
  log "invoking headless agent"
  if run_agent_prompt "$prompt" > "$review_file" 2>"$stderr_file"; then
    rm -f "$stderr_file"
    printf '%s\n' "$review_file"
    return 0
  else
    # `$?` inside the else branch is the failed command's exit. Outside the
    # `if`/`else` block, bash resets it to 0 (the if-statement's own exit
    # status when the condition was false), so the capture must live here.
    exit_code=$?
  fi
  log "headless agent failed (exit ${exit_code})"
  {
    printf '# %s (FAILED)\n\n%s\n' "$slug" "$meta"
    printf '## headless agent exit code\n%s\n\n' "$exit_code"
    # shellcheck disable=SC2016  # backticks here are literal markdown fences
    printf '## Prompt\n```\n%s\n```\n\n' "$prompt"
    # shellcheck disable=SC2016
    printf '## Stderr (tail)\n```\n'
    tail -40 "$stderr_file" 2>/dev/null || true
    printf '\n```\n'
  } > "$review_file"
  rm -f "$stderr_file"
  printf '%s\n' "$review_file"
  return 0
}
