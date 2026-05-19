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

  # The Python scanner is the authoritative parser — it uses shlex, so it
  # tracks quoted-value forms like `git -c user.name="A B" commit` that an
  # ERE pre-filter cannot. Non-commit invocations produce empty stdout (no
  # commit slice found), so this also serves as the fast-path for
  # `git status`, `git log`, etc.
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
