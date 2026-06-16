#!/usr/bin/env bash
# Project Claude Code's installed tinaudio/skills marketplace into
# ~/.agents/skills so Agent-Skills-standard CLIs see the full pack. See #1561.
set -euo pipefail
shopt -s nullglob

marketplace="${HOME}/.claude/plugins/marketplaces/tinaudio-skills"
codex_src="${marketplace}/codex/synth-setter-skills"
dest="${HOME}/.agents/skills"

if [[ ! -d "${marketplace}" ]]; then
  # The cache only exists once Claude has installed the plugin, so a fresh box
  # legitimately has nothing to project — skip rather than fail.
  echo "link-skills: no skills marketplace at ${marketplace} — install the tinaudio/skills plugin first; skipping." >&2
  exit 0
fi

repo_root="${SYNTH_SETTER_REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
repo_dest="${repo_root}/agent/skills"

mkdir -p "${dest}"
linked=0
workspace_linked=0
for src in "${marketplace}" "${codex_src}"; do
  [[ -d "${src}" ]] || continue
  for skill_dir in "${src}"/*/; do
    [[ -f "${skill_dir}SKILL.md" ]] || continue
    skill_name=$(basename "${skill_dir}")

    # 1. Project into ~/.agents/skills for Codex
    ln -sfn "${skill_dir%/}" "${dest}/${skill_name}"
    linked=$((linked + 1))

    # 2. Project into workspace agent/skills/ for Gemini/Antigravity
    if [[ -d "${repo_dest}" ]]; then
      if [[ ! -d "${repo_dest}/${skill_name}" || -L "${repo_dest}/${skill_name}" ]]; then
        ln -sfn "${skill_dir%/}" "${repo_dest}/${skill_name}"
        workspace_linked=$((workspace_linked + 1))

        # Add to .git/info/exclude to avoid dirtying git status
        exclude_file=""
        if [[ -d "${repo_root}/.git" ]]; then
          exclude_file=$(git -C "${repo_root}" rev-parse --git-path info/exclude 2>/dev/null || echo "${repo_root}/.git/info/exclude")
        else
          exclude_file="${repo_root}/.git/info/exclude"
        fi

        if [[ -n "${exclude_file}" ]]; then
          mkdir -p "$(dirname "${exclude_file}")"
          touch "${exclude_file}"
          if ! grep -q "^agent/skills/${skill_name}$" "${exclude_file}"; then
            echo "agent/skills/${skill_name}" >> "${exclude_file}"
          fi
        fi
      fi
    fi
  done
done

echo "link-skills: projected ${linked} skill link(s) from ${marketplace} into ${dest}."
if [[ "${workspace_linked}" -gt 0 ]]; then
  echo "link-skills: projected ${workspace_linked} skill link(s) into workspace ${repo_dest}."
fi
