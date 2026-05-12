#!/usr/bin/env bash
# Local CI triage agent launcher. Fetches a failing run's context from
# GitHub, writes /tmp/triage/context.json, and pipes the prompt template
# into `claude -p`. See .github/triage/README.md.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <failing-run-id>" >&2
  echo "  e.g.: $0 25687139710" >&2
  exit 2
fi

for cmd in gh jq claude git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$0: required command not found on PATH: $cmd" >&2
    echo "  see .github/triage/README.md for setup" >&2
    exit 3
  fi
done

RUN_ID="$1"
REPO="${TRIAGE_REPO:-tinaudio/synth-setter}"

# /tmp/triage is hard-coded in .github/triage/triage-prompt.md (the agent reads
# /tmp/triage/context.json directly); don't expose an override that would
# silently desync the launcher and the agent.
CONTEXT_DIR="/tmp/triage"
PROMPT_FILE="${TRIAGE_PROMPT:-$(git rev-parse --show-toplevel)/.github/triage/triage-prompt.md}"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "$0: prompt template missing: $PROMPT_FILE" >&2
  exit 4
fi

mkdir -p "$CONTEXT_DIR"

echo "Fetching run ${RUN_ID} metadata from ${REPO}..." >&2
gh api "repos/${REPO}/actions/runs/${RUN_ID}" \
  --jq '{run_id: .id, name, head_branch, head_sha, conclusion, html_url, event, workflow_id}' \
  > "${CONTEXT_DIR}/run.json"

jq -n \
  --arg repo "${REPO}" \
  --arg branch "ci-triage/run-${RUN_ID}" \
  --slurpfile run "${CONTEXT_DIR}/run.json" \
  '{repo: $repo, triage_branch: $branch, run: $run[0]}' \
  > "${CONTEXT_DIR}/context.json"

echo "Context written to ${CONTEXT_DIR}/context.json:" >&2
jq . "${CONTEXT_DIR}/context.json" >&2

echo "Invoking claude -p — agent transcript follows..." >&2
exec claude -p "$(cat "$PROMPT_FILE")" \
  --max-turns 30 \
  --permission-mode acceptEdits \
  --add-dir "$(git rev-parse --show-toplevel)" \
  --add-dir "$CONTEXT_DIR"
