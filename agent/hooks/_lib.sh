#!/usr/bin/env bash
# _lib.sh — shared helpers for agent/hooks/*. See each function's own block
# comment for its contract. Sourced by every hook; intentionally omits
# `set -euo pipefail` so each caller chooses its own error-handling regime.

REVIEWS_DIR=".agent-reviews"
HOOK_LOG="${REVIEWS_DIR}/.hook.log"

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
  # Worktree-aware: also checks the main repo via the git common dir, since
  # agent tool directories may live only in the primary checkout. Unmatched
  # plugin globs expand to their literal pattern, which the `-f` check safely
  # rejects without spurious warnings.
  local name="$1" home="${HOME:-}" common_dir repo_root p
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
  if common_dir=$(git rev-parse --git-common-dir 2>/dev/null) && [[ -n "$common_dir" ]]; then
    repo_root=$(cd "$common_dir/.." 2>/dev/null && pwd)
    if [[ -n "$repo_root" ]]; then
      paths+=(
        "${repo_root}/agent/skills/${name}/SKILL.md"
        "${repo_root}/.claude/skills/${name}/SKILL.md"
      )
    fi
  fi
  for p in "${paths[@]}"; do
    [[ -f "$p" ]] && return 0
  done
  return 1
}

run_agent_prompt() {
  # Usage: run_agent_prompt <prompt>
  # Routes the prompt through the available headless agent CLI. AGENT_HEADLESS
  # (`claude` or `codex`) overrides auto-detection; otherwise prefer claude,
  # fall back to codex, fail loud (exit 127) if neither is installed.
  local prompt="$1" cli="${AGENT_HEADLESS:-}" candidate
  if [[ -z "$cli" ]]; then
    for candidate in claude codex; do
      if command -v "$candidate" >/dev/null 2>&1; then
        cli="$candidate"
        break
      fi
    done
  fi
  case "$cli" in
    claude) claude -p "$prompt" ;;
    codex)  codex exec "$prompt" ;;
    *)
      printf 'No supported headless agent CLI found; install claude or codex (AGENT_HEADLESS=%s).\n' \
        "${AGENT_HEADLESS:-}" >&2
      return 127
      ;;
  esac
}

run_review() {
  # Usage: run_review <slug> <meta> <prompt> <review_file> <stderr_file> <dry_run>
  #
  # Orchestrates the DRY_RUN-vs-headless-agent dispatch shared by doc-drift.sh
  # and pr-review-resolver.sh. On success: writes the agent's stdout to
  # `<review_file>` and removes `<stderr_file>`. On failure: writes a verbose
  # report (slug, meta block, exit code, prompt, stderr tail). Returns 0 in
  # both cases — the caller decides the hook's own exit code.
  #
  # `<slug>`        short name used in the report header (e.g. "doc-drift").
  # `<meta>`        printf-ready block of "Key: value" lines (one per line).
  # `<prompt>`      text passed to run_agent_prompt; stubbed under dry-run.
  # `<review_file>` final report path (overwritten).
  # `<stderr_file>` scratch file for captured stderr; deleted at end.
  # `<dry_run>`     "1" skips the agent and writes a stub report.
  local slug="$1" meta="$2" prompt="$3" review_file="$4" stderr_file="$5" dry_run="$6"
  local exit_code
  if [[ "$dry_run" == "1" ]]; then
    log "DRY_RUN: writing stub report"
    printf '# %s (dry-run)\n%s\n## Prompt\n%s\n' "$slug" "$meta" "$prompt" > "$review_file"
    return 0
  fi
  log "invoking headless agent"
  if run_agent_prompt "$prompt" > "$review_file" 2>"$stderr_file"; then
    rm -f "$stderr_file"
    return 0
  fi
  exit_code=$?
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
  return 0
}
