#!/bin/bash
# Launch a project-pinned OpenCode PR-review pass.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
launcher="${repo_root}/agent/_shared/run_opencode_review_agent.py"
resolved="$(uv run --no-project --script "${launcher}" "$@")"
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

# Exit 3 is the caller's degrade-and-note signal for a missing CLI.
if ! command -v opencode >/dev/null; then
  echo "opencode CLI not found on PATH" >&2
  exit 3
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
# `read` (not `mapfile`) so macOS bash 3.2 works. Each line is one argv element.
command=()
while IFS= read -r line; do
  command+=("${line}")
done <<<"${command_lines}"
prompt_filter='if (.prompt | type) == "string" then
  .prompt
else
  error("invalid prompt")
end'
prompt="$(jq -er "${prompt_filter}" <<<"${resolved}")"
event_file="$(mktemp)"
log_file="$(mktemp)"
trap 'rm -f "${event_file}" "${log_file}"' EXIT
timeout_s="${OPENCODE_REVIEW_TIMEOUT:-600}"
# Stdin is withheld so a caller-held pipe can never stall the review gate.
"${command[@]}" "${prompt}" </dev/null >"${event_file}" 2>"${log_file}" &
run_pid=$!
# macOS ships no coreutils `timeout`; a watchdog subshell bounds hung runs.
(sleep "${timeout_s}" && kill "${run_pid}") >/dev/null 2>&1 &
watchdog_pid=$!
status=0
wait "${run_pid}" || status=$?
kill "${watchdog_pid}" >/dev/null 2>&1 || true
# Reap the watchdog so bash doesn't print job-control noise to stderr.
wait "${watchdog_pid}" 2>/dev/null || true
if [[ ${status} -ne 0 ]]; then
  cat "${log_file}" >&2
  if [[ ${status} -eq 143 ]]; then
    echo "opencode run timed out after ${timeout_s}s" >&2
  fi
  exit 1
fi
# shellcheck disable=SC2016  # $final is a jq variable, not a shell expansion
report_filter='[
  .[]
  | select(.type == "text")
  | .part
  | select(.type == "text")
]
| if length == 0 then error("missing assistant text") else . end
| (last | .messageID) as $final
| map(select(.messageID == $final))
| group_by(.id)
| map(last | .text)
| join("\n\n")'
if ! report="$(jq -ers "${report_filter}" "${event_file}")"; then
  cat "${log_file}" >&2
  exit 1
fi
printf '%s' "${report}"
