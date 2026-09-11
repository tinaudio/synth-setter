#!/usr/bin/env bash
# PreToolUse Bash(gh*) gate. Exits 0 (clean / not a gated gh pr command) or 2
# (off-vocabulary title, or a release-triggering type without RELEASE_INTENT=1).
# PR_TITLE_GUARD: block (default) / warn (stderr only, exit 0) / off (no-op).
set -euo pipefail

readonly HOOK_NAME="pr-title-guard"
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
  local mode="${PR_TITLE_GUARD:-block}"
  case "$mode" in
    off) exit 0 ;;
    block | warn) ;;
    *)
      echo "BLOCKED: PR_TITLE_GUARD must be one of block|warn|off, got: ${mode}" >&2
      exit 2
      ;;
  esac

  # Any unexpected failure (Python crash, jq parse, etc.) must block — never
  # leak a non-2 exit that bypasses the contract documented in the header.
  trap 'log "internal failure on line $LINENO; blocking"; echo "BLOCKED: ${HOOK_NAME} hit an internal error (line $LINENO); fix the hook or report it." >&2; exit 2' ERR

  local input cmd findings
  input=$(cat)

  # Fail closed: blocking malformed input is the whole point of this gate.
  if ! cmd=$(jq -r '.tool_input.command // empty' <<<"$input" 2>/dev/null); then
    log "jq parse failed; blocking conservatively"
    echo "BLOCKED: ${HOOK_NAME} could not parse tool-call JSON." >&2
    exit 2
  fi

  # Fast path: skip the python spawn for commands that cannot be gated.
  if [[ "$cmd" != *gh* || "$cmd" != *pr* ]]; then
    exit 0
  fi

  findings=$(printf '%s' "$cmd" | python3 "${SCRIPT_DIR}/pr_title_guard.py" --command)

  if [[ -n "$findings" ]]; then
    if [[ "$mode" == "warn" ]]; then
      log "warn: ${findings}"
      printf 'WARNING: PR title check failed:\n  %s\n' "${findings//$'\n'/$'\n  '}" >&2
      exit 0
    fi
    log "blocking: ${findings}"
    {
      echo "BLOCKED: PR title check failed."
      echo
      echo "Findings:"
      printf '  %s\n' "${findings//$'\n'/$'\n  '}"
      echo
      echo "Rules (AGENTS.md):"
      echo "  - PR titles are squash-merge subjects; the type must be in .gitlint's vocabulary."
      echo "  - feat/fix/perf/revert cut a release on merge — reserved for deliberate releases."
      echo "  - Use internal-feat: / internal-fix: for logic PRs."
      echo "  - Deliberate release: prefix the same command with RELEASE_INTENT=1."
    } >&2
    exit 2
  fi

  exit 0
}

main "$@"
