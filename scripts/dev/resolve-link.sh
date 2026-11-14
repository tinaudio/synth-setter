#!/usr/bin/env bash
# Print a symlink's target as a physical path.
#
# `readlink` yields the link text verbatim while `git rev-parse` resolves
# symlinks, so comparing the two raw strings misreports a correct link whenever
# the checkout is reached through a symlinked path (#3297). Resolving the
# target's parent directory puts both sides in the same namespace. The target
# itself may be absent — that is the dangling case callers must still detect —
# so only the parent is resolved.
set -euo pipefail

main() {
  local link="${1:?usage: resolve-link.sh <symlink>}" target parent
  target=$(readlink "$link")
  parent=$(cd "$(dirname "$target")" 2>/dev/null && pwd -P) || parent=""
  if [[ -z "$parent" ]]; then
    printf '%s\n' "$target"
    return 0
  fi
  printf '%s/%s\n' "$parent" "$(basename "$target")"
}

main "$@"
