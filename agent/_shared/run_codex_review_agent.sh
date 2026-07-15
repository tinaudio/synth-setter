#!/bin/bash
# Launch a project-pinned Codex PR-review role.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
launcher="${repo_root}/agent/_shared/run_codex_review_agent.py"
resolved="$(uv run --no-sync python "${launcher}" "$@")"
dry_run_filter='if (.dry_run | type) == "boolean" then
  (.dry_run | tostring)
else
  error("invalid dry_run")
end'
dry_run="$(jq -er "${dry_run_filter}" <<<"${resolved}")"

if [[ ${dry_run} == true ]]; then
  printf '%s\n' "${resolved}"
  exit 0
fi

command_filter='.command | if type == "array"
  and length > 0
  and all(.[]; type == "string")
then
  .[]
else
  error("invalid command")
end'
command_lines="$(jq -er "${command_filter}" <<<"${resolved}")"
mapfile -t command <<<"${command_lines}"
prompt_filter='if (.prompt | type) == "string" then
  .prompt
else
  error("invalid prompt")
end'
prompt="$(jq -er "${prompt_filter}" <<<"${resolved}")"
event_file="$(mktemp)"
log_file="$(mktemp)"
trap 'rm -f "${event_file}" "${log_file}"' EXIT
# codex exec appends piped stdin to the prompt and blocks until EOF; the prompt is already complete.
if ! "${command[@]}" --json "${prompt}" </dev/null >"${event_file}" 2>"${log_file}"; then
  cat "${log_file}" >&2
  exit 1
fi
report_filter='[
  .[]
  | select(.type == "item.completed")
  | select(.item.type == "agent_message")
  | .item.text
] | last // error("missing final agent message")'
if ! report="$(jq -ers "${report_filter}" "${event_file}")"; then
  cat "${log_file}" >&2
  exit 1
fi
printf '%s' "${report}"
