#!/usr/bin/env bash
# Project the installed tinaudio/skills marketplace into ~/.agents/skills so
# Agent-Skills-standard CLIs (Gemini, Antigravity) see the full pack. See #1561.
set -euo pipefail
shopt -s nullglob

src="${HOME}/.claude/plugins/marketplaces/tinaudio-skills"
dest="${HOME}/.agents/skills"

if [[ ! -d "${src}" ]]; then
  # The cache only exists once Claude has installed the plugin, so a fresh box
  # legitimately has nothing to project — skip rather than fail.
  echo "link-skills: no skills marketplace at ${src} — install the tinaudio/skills plugin first; skipping." >&2
  exit 0
fi

mkdir -p "${dest}"
linked=0
for skill_dir in "${src}"/*/; do
  [[ -f "${skill_dir}SKILL.md" ]] || continue
  ln -sfn "${skill_dir%/}" "${dest}/$(basename "${skill_dir}")"
  linked=$((linked + 1))
done

echo "link-skills: projected ${linked} skill(s) from ${src} into ${dest}."
