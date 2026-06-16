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

mkdir -p "${dest}"
linked=0
for src in "${marketplace}" "${codex_src}"; do
  [[ -d "${src}" ]] || continue
  for skill_dir in "${src}"/*/; do
    [[ -f "${skill_dir}SKILL.md" ]] || continue
    ln -sfn "${skill_dir%/}" "${dest}/$(basename "${skill_dir}")"
    linked=$((linked + 1))
  done
done

echo "link-skills: projected ${linked} skill link(s) from ${marketplace} into ${dest}."
