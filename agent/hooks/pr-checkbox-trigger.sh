#!/usr/bin/env bash
# pr-checkbox-trigger.sh — PostToolUse Bash hook.
#
# Detects real ``gh pr create`` invocations and prompts the agent to invoke
# the ``/pr-checkbox`` skill. Routes classification through
# ``agent/_shared/pr_command_classifier.py`` (the same contract the
# pre-PR review gate uses) so quoted prose that merely mentions the recipe —
# a ``gh issue create`` body quoting it, an ``echo`` printing it, a commit
# message — does NOT trip the reminder. The classifier returns ``direct`` /
# ``wrapped`` / ``unparsable`` for an actual invocation (in any form) and
# ``""`` for prose; this hook fires only on the non-empty modes.
#
# Regression: [#1942](https://github.com/tinaudio/synth-setter/issues/1942) —
# the old inline ``grep -q 'gh pr create'`` matched any command whose quoted
# arguments contained the phrase.
set -euo pipefail

export HOOK_NAME="pr-checkbox-trigger"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CLASSIFIER_PY="${SCRIPT_DIR}/../_shared/pr_command_classifier.py"

# Trust boundary: stdin is the Claude Code hook JSON payload.
COMMAND=$(jq -r '.tool_input.command // ""' 2>/dev/null || true)

# Fast substring pre-check skips the python3 spawn for the overwhelming
# majority of Bash commands; kept loose (``*gh* && *pr* && *create*``) so the
# classifier remains the authority on adjacency.
if [[ "$COMMAND" != *gh* || "$COMMAND" != *pr* || "$COMMAND" != *create* ]]; then
  exit 0
fi

# ``direct`` / ``wrapped`` / ``unparsable`` (non-empty) ⟶ a real ``gh pr create``
# invocation in some form: emit the reminder. ``""`` ⟶ quoted prose: silent.
# A missing python3/module fails OPEN here — this is advisory, not a gate.
MODE=$(python3 "$CLASSIFIER_PY" "$COMMAND" 2>/dev/null || true)
if [[ -n "$MODE" ]]; then
  jq -n --arg ctx \
    'A PR was just created. You MUST now invoke the /pr-checkbox skill to add verification checkboxes to this PR.' \
    '{hookSpecificOutput:{hookEventName:"PostToolUse", additionalContext:$ctx}}'
fi
exit 0
