#!/usr/bin/env bash
# no-yaml-run-comments.sh — PreToolUse gate blocking `#`-comments inside a
# `run: |` / `setup: |` block scalar in workflow / compute YAML. Reads
# tool-call JSON on stdin; exits 0 (out of scope / clean) or 2 (offender found).
set -euo pipefail

# shellcheck disable=SC2034  # read by log() in _lib.sh via ${HOOK_NAME:-unknown}
readonly HOOK_NAME="no-yaml-run-comments"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

if [[ ! -f "${SCRIPT_DIR}/_lib.sh" ]]; then
  echo "BLOCKED: no-yaml-run-comments cannot locate _lib.sh at ${SCRIPT_DIR}." >&2
  exit 2
fi
# shellcheck source=agent/hooks/_lib.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib.sh"

# Any unexpected failure (Python crash, jq parse, etc.) must block — never
# leak a non-2 exit that bypasses the contract documented in the header.
trap 'log "internal failure on line $LINENO; blocking"; echo "BLOCKED: no-yaml-run-comments hit an internal error (line $LINENO); fix the hook or report it." >&2; exit 2' ERR

in_scope() {
  # Usage: in_scope <file_path>
  # 0 if the path is a workflow or compute YAML this hook gates, else 1.
  case "$1" in
    .github/workflows/*.yaml|.github/workflows/*.yml) return 0 ;;
    */.github/workflows/*.yaml|*/.github/workflows/*.yml) return 0 ;;
    configs/compute/*.yaml|configs/compute/*.yml) return 0 ;;
    */configs/compute/*.yaml|*/configs/compute/*.yml) return 0 ;;
  esac
  return 1
}

emit_block_message() {
  # Usage: emit_block_message <file_path> <violations>
  # Renders the user-facing BLOCKED guidance on stderr.
  local file_path="$1" violations="$2"
  {
    echo "BLOCKED: comments inside a YAML \`run: |\` / \`setup: |\` block scalar."
    echo "File: ${file_path}"
    echo
    echo "Block-scalar bodies are bash. Stray ', \`, \$, or \\ inside a comment"
    echo "has caused unintended shell expansion. Move the comment ABOVE the step:"
    echo
    echo "  # Pin the template's image_id from its default to the dispatched tag."
    echo "  - name: Pin image tag"
    echo "    run: |"
    echo "      sed -i \"s|...|...|\" configs/compute/runpod-template.yaml"
    echo
    echo "Offenders (tab-delimited: line<TAB>block<TAB>header_line<TAB>text):"
    printf '  %s\n' "${violations//$'\n'/$'\n  '}"
  } >&2
}

main() {
  local input file_path violations stderr_file scanner_stderr_line
  input=$(cat)

  if ! file_path=$(jq -r '.tool_input.file_path // empty' <<<"$input" 2>/dev/null); then
    log "jq parse failed; blocking conservatively"
    echo "BLOCKED: no-yaml-run-comments could not parse tool-call JSON." >&2
    exit 2
  fi

  if ! in_scope "$file_path"; then
    exit 0
  fi

  stderr_file=$(mktemp -t no-yaml-run-comments.XXXXXX)
  # shellcheck disable=SC2064  # expand stderr_file at trap-install time so the cleanup path is fixed.
  trap "rm -f '$stderr_file'" EXIT

  violations=$(python3 "${SCRIPT_DIR}/no_yaml_run_comments.py" <<<"$input" 2>"$stderr_file")

  # Replay the scanner's log channel through the hook log. Non-LOG: stderr
  # (real crashes) re-emerges as-is so the ERR trap can surface it.
  while IFS= read -r scanner_stderr_line; do
    if [[ "$scanner_stderr_line" == LOG:* ]]; then
      log "${scanner_stderr_line#LOG:}"
    else
      printf '%s\n' "$scanner_stderr_line" >&2
    fi
  done < "$stderr_file"

  if [[ -n "$violations" ]]; then
    log "blocking yaml-run-comment in $file_path"
    emit_block_message "$file_path" "$violations"
    exit 2
  fi

  exit 0
}

main "$@"
