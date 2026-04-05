#!/usr/bin/env bash
# =============================================================================
# verify-gh-taxonomy.sh — Claude Code PostToolUse hook
# =============================================================================
#
# PURPOSE
# -------
# Enforces the project's GitHub taxonomy conventions (defined in
# docs/design/github-taxonomy.md) every time Claude creates an issue, a PR,
# or modifies issue hierarchy. Without this hook, the github-taxonomy skill
# can silently skip steps (project board, priority, hierarchy) even though
# the skill text requires them.
#
# HOW IT WORKS
# ------------
# This script is registered as a PostToolUse hook on the Bash tool in
# .claude/settings.json:
#
#   {
#     "hooks": {
#       "PostToolUse": [{
#         "matcher": "Bash",
#         "hooks": [{
#           "type": "command",
#           "command": "bash .claude/hooks/verify-gh-taxonomy.sh",
#           "timeout": 30
#         }]
#       }]
#     }
#   }
#
# After every Bash tool call, Claude Code pipes a JSON payload to this
# script's stdin containing:
#   - tool_input.command: the shell command that was executed
#   - tool_response: the stdout/stderr from that command
#
# The script pattern-matches the command to detect three triggers:
#   1. `gh issue create`  → MODE=issue
#   2. `gh pr create`     → MODE=pr
#   3. `addSubIssue`      → MODE=hierarchy
#
# For any other command, the script exits 0 immediately (no-op).
#
# RESPONSE FORMAT
# ---------------
# The script communicates back to Claude via JSON on stdout:
#
#   BLOCK (hard stop — Claude must fix before continuing):
#     {"decision":"block","reason":"..."}
#
#   WARN (advisory — Claude sees the message and should act on it):
#     {"hookSpecificOutput":{"hookEventName":"PostToolUse",
#       "additionalContext":"WARNING: ..."}}
#
#   PASS (informational — confirms compliance):
#     {"hookSpecificOutput":{"hookEventName":"PostToolUse",
#       "additionalContext":"Taxonomy check passed: ..."}}
#
# WHAT EACH MODE CHECKS
# ---------------------
#
# MODE=issue (after `gh issue create`):
#   Checks the newly created issue for the "CI minimum three":
#     - Issue type (Bug, Task, Feature, Phase, Epic)
#     - Domain label (data-pipeline, ci-automation, etc.)
#     - Milestone
#   Result: BLOCK if missing. Also BLOCKs with a reminder to add the issue
#   to the sub-issue hierarchy, project board, and set priority — ensuring
#   these steps are never skipped.
#
# MODE=pr (after `gh pr create`):
#   1. Checks the PR body for issue references (Fixes/Closes/Refs #N).
#      Result: BLOCK if no linked issue — the pr-metadata-gate CI will fail.
#   2. For each linked issue, checks the "CI minimum three" (type, label, milestone).
#      Result: BLOCK if any are missing.
#   3. For each linked issue, checks epic lineage (walks parent chain for Epic).
#      Result: BLOCK if no Epic ancestor — the pr-metadata-gate will fail.
#   4. For each linked issue, checks project board membership and priority.
#      Result: WARN if missing — these don't fail CI but are required by the
#      taxonomy lifecycle.
#
# MODE=hierarchy (after `addSubIssue` GraphQL mutation):
#   Validates that Tasks/Bugs/Features are attached to Phases, not directly
#   to Epics. The taxonomy requires: Epic → Phase → Task/Bug/Feature.
#   Result: BLOCK if a leaf issue is attached directly to an Epic.
#
# =============================================================================
set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# ---------------------------------------------------------------------------
# Route: match the command to a mode, or exit early for unrelated commands.
# ---------------------------------------------------------------------------
case "$COMMAND" in
  *"gh pr create"*) MODE="pr" ;;
  *"gh issue create"*) MODE="issue" ;;
  *"addSubIssue"*) MODE="hierarchy" ;;
  *) exit 0 ;;
esac

TOOL_RESPONSE=$(echo "$INPUT" | jq -r '.tool_response // empty')

# ---------------------------------------------------------------------------
# Scope: only run for commands that targeted synth-setter.
# The hook lives in the synth-setter project, but commands may target other
# repos (e.g. `gh pr create` run from a skills repo clone). We extract the
# actual GitHub URL from the response and check its repo path. We can't just
# grep the whole response for "synth-setter" because Bash tool output always
# includes "Shell cwd was reset to .../synth-setter" when cwd changes.
# For addSubIssue mutations, check the GraphQL query in the command string.
# ---------------------------------------------------------------------------
if [ "$MODE" = "hierarchy" ]; then
  if ! echo "$COMMAND" | grep -qE 'owner.*synth-setter|repo.*synth-setter'; then
    exit 0
  fi
else
  # Extract the GitHub URL (PR or issue) and check if it's synth-setter
  GITHUB_URL=$(echo "$TOOL_RESPONSE" | grep -oE 'https://github.com/[^/]+/[^/]+/(pull|issues)/[0-9]+' | head -1)
  if [ -z "$GITHUB_URL" ]; then exit 0; fi
  if ! echo "$GITHUB_URL" | grep -q 'synth-setter'; then
    exit 0
  fi
fi

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
OWNER="tinaudio"
REPO="synth-setter"
DOMAIN_LABELS="data-pipeline ci-automation code-health documentation evaluation experiments monitoring storage testing training"

# ---------------------------------------------------------------------------
# Helper: query an issue's taxonomy metadata (type, labels, milestone).
# Usage: query_issue_metadata <issue_number>
# Sets: ISSUE_TYPE, LABELS, MILESTONE, HAS_DOMAIN
# ---------------------------------------------------------------------------
query_issue_metadata() {
  local issue_num="$1"

  local result
  # shellcheck disable=SC2016
  result=$(gh api graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) {
          issueType { name }
          labels(first: 10) { nodes { name } }
          milestone { title }
        }
      }
    }
  ' -f owner="$OWNER" -f repo="$REPO" -F number="$issue_num" 2>/dev/null || echo "{}")

  ISSUE_TYPE=$(echo "$result" | jq -r '.data.repository.issue.issueType.name // empty')
  LABELS=$(echo "$result" | jq -r '[.data.repository.issue.labels.nodes[].name] | join(", ")' 2>/dev/null || echo "")
  MILESTONE=$(echo "$result" | jq -r '.data.repository.issue.milestone.title // empty')

  HAS_DOMAIN=false
  for dl in $DOMAIN_LABELS; do
    if echo "$LABELS" | grep -q "$dl"; then HAS_DOMAIN=true; break; fi
  done
}

# ---------------------------------------------------------------------------
# Helper: check the "CI minimum three" — issue type, domain label, milestone.
# Usage: check_ci_minimum <issue_number>
# Returns: space-separated list of missing fields, or empty string if all set.
# Requires query_issue_metadata to have been called first.
# ---------------------------------------------------------------------------
check_ci_minimum() {
  local missing=""
  [ -z "$ISSUE_TYPE" ] && missing="${missing} issue-type"
  [ "$HAS_DOMAIN" = "false" ] && missing="${missing} domain-label"
  [ -z "$MILESTONE" ] && missing="${missing} milestone"
  echo "$missing"
}

# ---------------------------------------------------------------------------
# Helper: check if an issue is on the project board and has priority set.
# Usage: check_project_fields <issue_number>
# Returns: space-separated list of missing fields, or empty string if all set.
# ---------------------------------------------------------------------------
check_project_fields() {
  local issue_num="$1"

  # Get the issue's node ID, then check project items on it.
  local node_id
  # shellcheck disable=SC2016
  node_id=$(gh api graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) { id }
      }
    }
  ' -f owner="$OWNER" -f repo="$REPO" -F number="$issue_num" \
    --jq '.data.repository.issue.id' 2>/dev/null || echo "")

  if [ -z "$node_id" ]; then
    echo " project-board priority"
    return
  fi

  # Query project items on this issue. Check if any project item exists
  # and whether it has a Priority field set.
  local project_result
  # shellcheck disable=SC2016
  project_result=$(gh api graphql -f query='
    query($id: ID!) {
      node(id: $id) {
        ... on Issue {
          projectItems(first: 5) {
            nodes {
              project { title }
              fieldValues(first: 10) {
                nodes {
                  ... on ProjectV2ItemFieldSingleSelectValue {
                    field { ... on ProjectV2SingleSelectField { name } }
                    name
                  }
                }
              }
            }
          }
        }
      }
    }
  ' -f id="$node_id" 2>/dev/null || echo "{}")

  local on_board
  on_board=$(echo "$project_result" | jq -r '.data.node.projectItems.nodes | length' 2>/dev/null || echo "0")

  local missing=""
  if [ "$on_board" = "0" ]; then
    missing="${missing} project-board priority"
  else
    # Check if Priority field is set on any project item
    local has_priority
    has_priority=$(echo "$project_result" | jq -r '
      [.data.node.projectItems.nodes[].fieldValues.nodes[]
        | select(.field.name == "Priority")
        | .name] | length' 2>/dev/null || echo "0")
    if [ "$has_priority" = "0" ]; then
      missing="${missing} priority"
    fi
  fi

  echo "$missing"
}


# ---------------------------------------------------------------------------
# Helper: walk the sub-issue parent chain to find an Epic ancestor.
# Usage: check_epic_lineage <issue_number>
# Returns: "ok" if the issue traces to an Epic, "no-epic" otherwise.
# ---------------------------------------------------------------------------
check_epic_lineage() {
  local issue_num="$1"
  local current="$issue_num"
  local depth=0

  while [ "$depth" -lt 4 ]; do
    local result
    # shellcheck disable=SC2016
    result=$(gh api graphql -f query='
      query($owner: String!, $repo: String!, $number: Int!) {
        repository(owner: $owner, name: $repo) {
          issue(number: $number) {
            issueType { name }
            parent { number issueType { name } }
          }
        }
      }
    ' -f owner="$OWNER" -f repo="$REPO" -F number="$current" \
      --jq '.data.repository.issue' 2>/dev/null) || {
      echo "no-epic"
      return
    }

    local type
    type=$(echo "$result" | jq -r '.issueType.name // empty' 2>/dev/null) || type=""
    if [ "$type" = "Epic" ]; then
      echo "ok"
      return
    fi

    local parent_num
    parent_num=$(echo "$result" | jq -r '.parent.number // empty' 2>/dev/null) || parent_num=""
    if [ -z "$parent_num" ] || [ "$parent_num" = "null" ]; then
      break
    fi
    current="$parent_num"
    depth=$((depth + 1))
  done

  echo "no-epic"
}


# ===========================================================================
# MODE: pr — runs after `gh pr create`
# ===========================================================================
if [ "$MODE" = "pr" ]; then
  # Step 1: Extract PR number from the tool response URL.
  PR_URL=$(echo "$TOOL_RESPONSE" | grep -oE 'https://github.com/[^/]+/[^/]+/pull/[0-9]+' | head -1)
  if [ -z "$PR_URL" ]; then exit 0; fi
  PR_NUM=$(echo "$PR_URL" | grep -oE '[0-9]+$')

  # Step 2: Check the PR body for issue references (Fixes/Closes/Refs #N).
  # Without a linked issue, the pr-metadata-gate CI workflow will fail.
  PR_BODY=$(gh pr view "$PR_NUM" --repo "${OWNER}/${REPO}" --json body --jq '.body' 2>/dev/null || echo "")
  ISSUE_NUMS=$(echo "$PR_BODY" | grep -oE '(Fixes|Closes|Refs|fixes|closes|refs) #[0-9]+' | grep -oE '[0-9]+' | sort -un | grep -v "^${PR_NUM}$" || true)

  # Fallback: check for any #N reference in the body (matches bare #N and
  # markdown hyperlinks like [#399](url)), same as pr-metadata-gate.yaml.
  if [ -z "$ISSUE_NUMS" ]; then
    ISSUE_NUMS=$(echo "$PR_BODY" | grep -oE '#[0-9]+' | tr -d '#' | sort -un | grep -v "^${PR_NUM}$" || true)
  fi

  # Fallback: check for any #N reference in the body (matches bare #N and
  # markdown hyperlinks like [#399](url)), same as pr-metadata-gate.yaml.
  if [ -z "$ISSUE_NUMS" ]; then
    ISSUE_NUMS=$(echo "$PR_BODY" | grep -oE '#[0-9]+' | tr -d '#' | sort -un | grep -v "^${PR_NUM}$" || true)
  fi

  if [ -z "$ISSUE_NUMS" ]; then
    echo '{"decision":"block","reason":"PR has no linked issue reference. Acceptable patterns include Fixes #N / Closes #N / Refs #N or any bare #N reference in the PR body. The pr-metadata-gate CI check will fail."}'
    exit 0
  fi

  # Step 3: For each linked issue, verify the "CI minimum three" (BLOCK if missing)
  # and project board + priority (WARN if missing).
  WARNINGS=""
  for ISSUE_NUM in $ISSUE_NUMS; do
    query_issue_metadata "$ISSUE_NUM"
    CI_MISSING=$(check_ci_minimum)

    if [ -n "$CI_MISSING" ]; then
      echo "{\"decision\":\"block\",\"reason\":\"Issue #${ISSUE_NUM} is missing taxonomy metadata:${CI_MISSING}. Fix before the pr-metadata-gate CI check fails.\"}"
      exit 0
    fi

    # Check epic lineage — the pr-metadata-gate CI workflow requires it.
    LINEAGE=$(check_epic_lineage "$ISSUE_NUM")
    if [ "$LINEAGE" != "ok" ]; then
      echo "{\"decision\":\"block\",\"reason\":\"Issue #${ISSUE_NUM} does not trace to any Epic. Add it as a sub-issue of the appropriate Phase/Epic — the pr-metadata-gate epic lineage check will fail.\"}"
      exit 0
    fi

    # CI gate will pass, but check the full taxonomy lifecycle too.
    PROJECT_MISSING=$(check_project_fields "$ISSUE_NUM")
    if [ -n "$PROJECT_MISSING" ]; then
      WARNINGS="${WARNINGS}Issue #${ISSUE_NUM} is missing:${PROJECT_MISSING}. "
    fi
  done

  if [ -n "$WARNINGS" ]; then
    echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PostToolUse\",\"additionalContext\":\"Taxonomy check passed (CI gate OK), but full lifecycle incomplete: ${WARNINGS}Add to project board and set priority before merging.\"}}"
  else
    echo '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"Taxonomy check passed: linked issue(s) have issue type, domain label, and milestone."}}'
  fi


# ===========================================================================
# MODE: issue — runs after `gh issue create`
# ===========================================================================
elif [ "$MODE" = "issue" ]; then
  # Extract the issue number from the tool response URL.
  ISSUE_URL=$(echo "$TOOL_RESPONSE" | grep -oE 'https://github.com/[^/]+/[^/]+/issues/[0-9]+' | head -1)
  if [ -z "$ISSUE_URL" ]; then exit 0; fi
  ISSUE_NUM=$(echo "$ISSUE_URL" | grep -oE '[0-9]+$')

  # Check the CI minimum three. At this point, issue type is often not yet
  # set because `gh issue create` can't set it — it requires a separate
  # GraphQL mutation. BLOCK either way to force completing all taxonomy steps.
  query_issue_metadata "$ISSUE_NUM"
  CI_MISSING=$(check_ci_minimum)

  if [ -n "$CI_MISSING" ]; then
    echo "{\"decision\":\"block\",\"reason\":\"Issue #${ISSUE_NUM} is missing taxonomy metadata:${CI_MISSING}. Set these now before proceeding. Then add to sub-issue hierarchy, project board, and set priority.\"}"
    exit 0
  fi

  echo "{\"decision\":\"block\",\"reason\":\"Issue #${ISSUE_NUM} created with type, label, and milestone. You MUST now add it to the sub-issue hierarchy (addSubIssue to the appropriate Phase/Epic), add to project board, and set priority before proceeding.\"}"


# ===========================================================================
# MODE: hierarchy — runs after `addSubIssue` GraphQL mutation
# ===========================================================================
elif [ "$MODE" = "hierarchy" ]; then
  # Validate that Tasks/Bugs/Features are attached to Phases, not directly
  # to Epics. The taxonomy hierarchy is: Epic → Phase → Task/Bug/Feature.
  #
  # Extract issue node IDs (format: I_kw...) from the GraphQL mutation command.
  NODE_IDS=$(echo "$COMMAND" | grep -oE 'I_kw[A-Za-z0-9_/+=.-]+' || true)
  if [ -z "$NODE_IDS" ]; then exit 0; fi

  PARENT_TYPE=""
  PARENT_NUM=""
  CHILD_TYPE=""
  CHILD_NUM=""

  for NODE_ID in $NODE_IDS; do
    # shellcheck disable=SC2016
    RESULT=$(gh api graphql -f query='
      query($id: ID!) {
        node(id: $id) {
          ... on Issue { number issueType { name } }
        }
      }
    ' -f id="$NODE_ID" 2>/dev/null || echo "{}")

    TYPE=$(echo "$RESULT" | jq -r '.data.node.issueType.name // empty')
    NUM=$(echo "$RESULT" | jq -r '.data.node.number // empty')

    # The addSubIssue mutation takes (parentId, childId). We infer which is
    # which from the issue type: Epic/Phase are parents, Task/Bug/Feature
    # are children.
    case "$TYPE" in
      Epic|Phase)
        PARENT_TYPE="$TYPE"
        PARENT_NUM="$NUM"
        ;;
      Task|Bug|Feature)
        CHILD_TYPE="$TYPE"
        CHILD_NUM="$NUM"
        ;;
    esac
  done

  # Block if a leaf issue is attached directly to an Epic (must go under a Phase).
  if [ "$PARENT_TYPE" = "Epic" ] && [ -n "$CHILD_TYPE" ]; then
    echo "{\"decision\":\"block\",\"reason\":\"${CHILD_TYPE} #${CHILD_NUM} cannot be a direct child of Epic #${PARENT_NUM}. Task/Bug/Feature issues must be sub-issues of a Phase, not an Epic. Add it under the appropriate Phase instead.\"}"
    exit 0
  fi

  echo '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"Hierarchy check passed."}}'
fi
