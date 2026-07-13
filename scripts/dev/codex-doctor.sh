#!/usr/bin/env bash
set -euo pipefail

main() {
  local repo_root failures=0 missing_skills=0 skill
  local codex_path codex_real codex_version
  local codex_report=""
  local version_failures=0
  local codex_reals=$'\n' codex_versions=$'\n'
  local codex_real_count=0 codex_version_count=0
  repo_root=$(git rev-parse --show-toplevel)
  # has_skill probes repo-relative paths (agent/skills, .agents/skills), so run
  # the checks from the repo root regardless of where the doctor was invoked.
  cd "$repo_root"

  # shellcheck source=agent/hooks/_lib.sh
  # shellcheck disable=SC1091
  source "$repo_root/agent/hooks/_lib.sh"

  "$repo_root/scripts/dev/link-skills.sh"

  if command -v codex >/dev/null 2>&1; then
    printf 'OK: codex CLI found\n'
    while IFS= read -r codex_path; do
      codex_real=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$codex_path")
      if codex_version=$("$codex_path" --version 2>/dev/null); then
        :
      else
        codex_version="<version failed>"
        version_failures=$((version_failures + 1))
      fi
      printf -v codex_report '%s  %s -> %s (%s)\n' \
        "$codex_report" "$codex_path" "$codex_real" "$codex_version"
      if ! contains_line "$codex_real" "$codex_reals"; then
        codex_reals="${codex_reals}${codex_real}"$'\n'
        codex_real_count=$((codex_real_count + 1))
      fi
      if ! contains_line "$codex_version" "$codex_versions"; then
        codex_versions="${codex_versions}${codex_version}"$'\n'
        codex_version_count=$((codex_version_count + 1))
      fi
    done < <(type -P -a codex | awk '!seen[$0]++')

    if [[ "$version_failures" -gt 0 ]]; then
      printf 'MISSING: %s Codex launcher(s) failed to report a version\n' "$version_failures" >&2
      printf '%s' "$codex_report" >&2
      failures=$((failures + 1))
    elif [[ "$codex_real_count" -gt 1 || "$codex_version_count" -gt 1 ]]; then
      printf 'MISMATCH: Multiple Codex launchers resolve to different installs or versions\n' >&2
      printf '%s' "$codex_report" >&2
      failures=$((failures + 1))
    else
      printf 'OK: codex launchers resolve to one install\n'
    fi
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
    printf 'Repo-local skills (e.g. pr-readiness) come from the repo skill projection, not the plugin.\n'
  fi

  if [[ "$failures" -gt 0 ]]; then
    printf '\nCodex setup has %s issue(s).\n' "$failures"
    return 1
  fi

  printf '\nCodex setup looks ready.\n'
}

contains_line() {
  local needle=$1 haystack=$2
  [[ "$haystack" == *$'\n'"$needle"$'\n'* ]]
}

main "$@"
