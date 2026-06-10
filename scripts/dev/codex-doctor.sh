#!/usr/bin/env bash
set -euo pipefail

main() {
  local repo_root failures=0 missing_skills=0 skill
  repo_root=$(git rev-parse --show-toplevel)
  # has_skill probes repo-relative paths (agent/skills, .agents/skills), so run
  # the checks from the repo root regardless of where the doctor was invoked.
  cd "$repo_root"

  # shellcheck source=agent/hooks/_lib.sh
  # shellcheck disable=SC1091
  source "$repo_root/agent/hooks/_lib.sh"

  if command -v codex >/dev/null 2>&1; then
    printf 'OK: codex CLI found\n'
  else
    printf 'MISSING: codex CLI not found on PATH\n'
    failures=$((failures + 1))
  fi

  if [[ -d "$repo_root/.agents/skills" ]]; then
    printf 'OK: .agents/skills resolves to a directory\n'
  else
    printf 'MISSING: .agents/skills does not resolve to a directory\n'
    failures=$((failures + 1))
  fi

  if grep -q 'tinaudio/skills' "$repo_root/.agents/plugins/marketplace.json" 2>/dev/null; then
    printf 'OK: repo marketplace advertises tinaudio/skills\n'
  else
    printf 'MISSING: repo marketplace does not advertise tinaudio/skills\n'
    failures=$((failures + 1))
  fi

  for skill in \
    code-health \
    github-taxonomy \
    pr-checkbox \
    pr-readiness \
    pr-review-resolver \
    simplify \
    tdd-implementation
  do
    if has_skill "$skill"; then
      printf 'OK: skill %s is discoverable\n' "$skill"
    else
      printf 'MISSING: skill %s is not discoverable\n' "$skill"
      missing_skills=$((missing_skills + 1))
    fi
  done

  if [[ "$missing_skills" -gt 0 ]]; then
    failures=$((failures + 1))
    printf '\nOne or more required skills are missing.\n'
    printf 'If tinaudio plugin skills are missing, install/enable the plugin, then start a new Codex thread:\n'
    printf '  codex plugin marketplace add tinaudio/skills\n'
    printf '  /plugins -> tinaudio Skills -> tinaudio-synth-setter-skills\n'

  if [[ "$failures" -gt 0 ]]; then
    printf '\nCodex setup has %s issue(s).\n' "$failures"
    return 1
  fi

  printf '\nCodex setup looks ready.\n'
}

main "$@"
