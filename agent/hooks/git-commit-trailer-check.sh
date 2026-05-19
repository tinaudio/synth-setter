#!/usr/bin/env bash
# PreToolUse Bash(git commit *) gate. Exits 0 (clean / not git commit) or 2
# (forbidden --no-verify/-n flag or Co-Authored-By / agent-attribution trailer).
set -euo pipefail

readonly HOOK_NAME="git-commit-trailer-check"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

[[ -f "${SCRIPT_DIR}/_lib.sh" ]] || {
  echo "BLOCKED: ${HOOK_NAME} could not find _lib.sh next to itself." >&2
  exit 2
}
# shellcheck source=agent/hooks/_lib.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib.sh"

# Re-scope regex: matches `git ... commit` where the slot between `git` and
# `commit` is empty or holds only `git`-level option tokens (e.g. `-c key=val`,
# `--git-dir=.git`). Required because the handler-level `if: "Bash(git commit *)"`
# is a literal-prefix permission rule that does NOT match `git -c X commit ...`;
# this hook is broadened to `Bash(git*)` and the wrapper re-scopes here.
readonly GIT_COMMIT_RE='(^[[:space:]]*|[;|&`(][[:space:]]*)git([[:space:]]+(-[a-zA-Z]|--[a-zA-Z][a-zA-Z0-9_-]*)(=[^[:space:]]*)?([[:space:]]+[^-[:space:]][^[:space:]]*)?)*[[:space:]]+commit([[:space:]]|$)'

main() {
  # Any unexpected failure (Python crash, jq parse, etc.) must block — never
  # leak a non-2 exit that bypasses the contract documented in the header.
  trap 'log "internal failure on line $LINENO; blocking"; echo "BLOCKED: ${HOOK_NAME} hit an internal error (line $LINENO); fix the hook or report it." >&2; exit 2' ERR

  # stdin is bounded by Claude Code (single tool-call JSON payload), so reading
  # it all into memory is safe.
  local input cmd findings
  input=$(cat)

  # Fail closed: blocking malformed input is the whole point of this gate.
  if ! cmd=$(jq -r '.tool_input.command // empty' <<<"$input" 2>/dev/null); then
    log "jq parse failed; blocking conservatively"
    echo "BLOCKED: ${HOOK_NAME} could not parse tool-call JSON." >&2
    exit 2
  fi

  if ! grep -qE "$GIT_COMMIT_RE" <<<"$cmd"; then
    exit 0
  fi

  # Pipe the command on stdin to the Python scanner so argv slicing can
  # distinguish `git commit -n` from a downstream `grep -n`.
  findings=$(printf '%s' "$cmd" | python3 "${SCRIPT_DIR}/git_commit_trailer_check.py")

  if [[ -n "$findings" ]]; then
    log "blocking forbidden flag/trailer"
    {
      echo "BLOCKED: \`git commit\` invocation is non-compliant."
      echo
      echo "Findings:"
      printf '  %s\n' "${findings//$'\n'/$'\n  '}"
      echo
      echo "Rules (AGENTS.md):"
      echo "  - Never use --no-verify / -n; hooks work inside worktrees and must run."
      echo "  - Never add Co-Authored-By trailers."
      echo "  - Never add agent-attribution footers (\"Generated with ...\", \"Claude ...\", etc.)."
      echo
      echo "Fix the underlying hook failure (or rewrite the commit message) and re-run."
    } >&2
    exit 2
  fi

  exit 0
}

main "$@"
