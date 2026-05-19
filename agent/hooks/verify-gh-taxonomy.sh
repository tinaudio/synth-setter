#!/usr/bin/env bash
# verify-gh-taxonomy.sh — agent PostToolUse Bash hook.
#
# Enforces docs/design/github-taxonomy.md on `gh issue create`, `gh pr create`,
# and `addSubIssue` GraphQL mutations. Emits Claude hook JSON on stdout:
# {"decision":"block",...} or {"hookSpecificOutput":{...}}. Per-mode rules
# live below in the mode_* functions.
set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OWNER="tinaudio"
REPO="synth-setter"
DOMAIN_LABELS=(
  data-pipeline ci-automation code-health documentation
  evaluation experiments monitoring storage testing training
)

# ---------------------------------------------------------------------------
# JSON emitters — `jq -n --arg` quotes the payload safely so a `"`, `\`, or
# newline in the reason/context string can't produce invalid JSON.
# ---------------------------------------------------------------------------
emit_block() {
  jq -n --arg reason "$1" '{decision:"block", reason:$reason}'
}

emit_advisory() {
  # Used for both WARN and PASS contexts — schema is identical, only the
  # additionalContext wording differentiates them.
  jq -n --arg ctx "$1" \
    '{hookSpecificOutput:{hookEventName:"PostToolUse", additionalContext:$ctx}}'
}

# ---------------------------------------------------------------------------
# Metadata queries
# ---------------------------------------------------------------------------
query_issue_metadata() {
  # Usage: query_issue_metadata <issue_number>
  # Emits one JSON object on stdout: `{type, labels, milestone, has_domain}`.
  # Callers extract fields via `jq -r '.type'` etc. — explicit named access
  # avoids the positional-order coupling of a TSV contract.
  local issue_num="$1" result type labels milestone has_domain=false dl
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
  type=$(echo "$result" | jq -r '.data.repository.issue.issueType.name // empty' 2>/dev/null || echo "")
  labels=$(echo "$result" | jq -r '[.data.repository.issue.labels.nodes[].name] | join(", ")' 2>/dev/null || echo "")
  milestone=$(echo "$result" | jq -r '.data.repository.issue.milestone.title // empty' 2>/dev/null || echo "")
  for dl in "${DOMAIN_LABELS[@]}"; do
    if echo "$labels" | grep -q "$dl"; then has_domain=true; break; fi
  done
  jq -n \
    --arg type "$type" \
    --arg labels "$labels" \
    --arg milestone "$milestone" \
    --argjson has_domain "$has_domain" \
    '{type:$type, labels:$labels, milestone:$milestone, has_domain:$has_domain}'
}

check_ci_minimum() {
  # Usage: check_ci_minimum <type> <has_domain> <milestone>
  # Echoes a comma-and-space-separated list of missing fields (e.g.
  # "issue-type, milestone"), or empty string if all three are set. Callers
  # interpolate directly into user-facing error strings.
  local type="$1" has_domain="$2" milestone="$3"
  local missing=() joined
  [[ -z "$type" ]] && missing+=("issue-type")
  [[ "$has_domain" == "false" ]] && missing+=("domain-label")
  [[ -z "$milestone" ]] && missing+=("milestone")
  if [[ "${#missing[@]}" -gt 0 ]]; then
    # `printf -v + strip trailing sep` — `IFS=', '` with `"${arr[*]}"` would
    # join on only the FIRST char of IFS (a bash quirk; see Copilot review on
    # PR #1119), producing `issue-type,domain-label` without the space.
    printf -v joined '%s, ' "${missing[@]}"
    echo "${joined%, }"
  fi
}

check_project_fields() {
  # Usage: check_project_fields <issue_number>
  # Echoes a comma-separated list of missing fields ("project-board, priority"
  # or "priority"), or empty if both are set. Returns the bare list with no
  # leading space so callers can interpolate cleanly into error strings.
  local issue_num="$1" node_id project_result on_board has_priority
  local missing=()
  # shellcheck disable=SC2016
  node_id=$(gh api graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) { id }
      }
    }
  ' -f owner="$OWNER" -f repo="$REPO" -F number="$issue_num" \
    --jq '.data.repository.issue.id' 2>/dev/null || echo "")
  if [[ -z "$node_id" ]]; then
    echo "project-board, priority"
    return
  fi
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
  on_board=$(echo "$project_result" | jq -r '.data.node.projectItems.nodes | length' 2>/dev/null || echo "0")
  if [[ "$on_board" == "0" ]]; then
    missing+=("project-board" "priority")
  else
    has_priority=$(echo "$project_result" | jq -r '
      [.data.node.projectItems.nodes[].fieldValues.nodes[]
        | select(.field.name == "Priority")
        | .name] | length' 2>/dev/null || echo "0")
    if [[ "$has_priority" == "0" ]]; then
      missing+=("priority")
    fi
  fi
  if [[ "${#missing[@]}" -gt 0 ]]; then
    # See check_ci_minimum for the IFS-vs-printf-v rationale.
    local joined
    printf -v joined '%s, ' "${missing[@]}"
    echo "${joined%, }"
  fi
}

check_epic_lineage() {
  # Usage: check_epic_lineage <issue_number>
  # Walks the parent chain (max depth 4) looking for an Epic ancestor.
  # Echoes "ok" if found, "no-epic" otherwise.
  local issue_num="$1"
  local current="$issue_num"
  local depth=0 result type parent_num
  while [[ "$depth" -lt 4 ]]; do
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
      --jq '.data.repository.issue' 2>/dev/null) || { echo "no-epic"; return; }
    type=$(echo "$result" | jq -r '.issueType.name // empty' 2>/dev/null || echo "")
    if [[ "$type" == "Epic" ]]; then
      echo "ok"
      return
    fi
    parent_num=$(echo "$result" | jq -r '.parent.number // empty' 2>/dev/null || echo "")
    if [[ -z "$parent_num" || "$parent_num" == "null" ]]; then
      break
    fi
    current="$parent_num"
    depth=$((depth + 1))
  done
  echo "no-epic"
}

# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------
mode_pr() {
  # Usage: mode_pr <tool_response>
  local tool_response="$1" pr_url pr_num pr_body issue_nums issue_num metadata
  local type milestone has_domain ci_missing lineage project_missing warnings=""
  pr_url=$(echo "$tool_response" | grep -oE 'https://github.com/[^/]+/[^/]+/pull/[0-9]+' | head -1 || true)
  [[ -z "$pr_url" ]] && return 0
  pr_num=$(echo "$pr_url" | grep -oE '[0-9]+$' || true)
  # Defensive: if the trailing-number extraction failed, treat as no-op rather
  # than calling `gh pr view ""` and racing the `|| echo ""` swallow.
  [[ -z "$pr_num" ]] && return 0

  # Check the PR body for issue references (Fixes/Closes/Refs #N). Without a
  # linked issue, the pr-metadata-gate CI workflow will fail.
  pr_body=$(gh pr view "$pr_num" --repo "${OWNER}/${REPO}" --json body --jq '.body' 2>/dev/null || echo "")
  issue_nums=$(echo "$pr_body" | grep -oE '(Fixes|Closes|Refs|fixes|closes|refs) #[0-9]+' \
    | grep -oE '[0-9]+' | sort -un | grep -v "^${pr_num}$" || true)
  # Fallback: any #N reference (matches bare #N and markdown hyperlinks like
  # [#399](url)), mirroring pr-metadata-gate.yaml.
  if [[ -z "$issue_nums" ]]; then
    issue_nums=$(echo "$pr_body" | grep -oE '#[0-9]+' | tr -d '#' | sort -un | grep -v "^${pr_num}$" || true)
  fi
  if [[ -z "$issue_nums" ]]; then
    emit_block "PR has no linked issue reference. Acceptable patterns include Fixes #N / Closes #N / Refs #N or any bare #N reference in the PR body. The pr-metadata-gate CI check will fail."
    return 0
  fi

  # For each linked issue: verify CI minimum (BLOCK if missing), epic lineage
  # (BLOCK if missing), then project board + priority (WARN if missing).
  while IFS= read -r issue_num; do
    [[ -z "$issue_num" ]] && continue
    metadata=$(query_issue_metadata "$issue_num")
    type=$(echo "$metadata" | jq -r '.type')
    milestone=$(echo "$metadata" | jq -r '.milestone')
    has_domain=$(echo "$metadata" | jq -r '.has_domain')
    ci_missing=$(check_ci_minimum "$type" "$has_domain" "$milestone")
    if [[ -n "$ci_missing" ]]; then
      emit_block "Issue #${issue_num} is missing taxonomy metadata: ${ci_missing}. Fix before the pr-metadata-gate CI check fails."
      return 0
    fi
    lineage=$(check_epic_lineage "$issue_num")
    if [[ "$lineage" != "ok" ]]; then
      emit_block "Issue #${issue_num} does not trace to any Epic. Add it as a sub-issue of the appropriate Phase/Epic — the pr-metadata-gate epic lineage check will fail."
      return 0
    fi
    project_missing=$(check_project_fields "$issue_num")
    if [[ -n "$project_missing" ]]; then
      warnings="${warnings}Issue #${issue_num} is missing: ${project_missing}. "
    fi
  done <<< "$issue_nums"

  if [[ -n "$warnings" ]]; then
    emit_advisory "Taxonomy check passed (CI gate OK), but full lifecycle incomplete: ${warnings}Add to project board and set priority before merging."
  else
    emit_advisory "Taxonomy check passed: linked issue(s) have issue type, domain label, and milestone."
  fi
}

mode_issue() {
  # Usage: mode_issue <tool_response>
  local tool_response="$1" issue_url issue_num metadata type milestone has_domain ci_missing
  issue_url=$(echo "$tool_response" | grep -oE 'https://github.com/[^/]+/[^/]+/issues/[0-9]+' | head -1 || true)
  [[ -z "$issue_url" ]] && return 0
  issue_num=$(echo "$issue_url" | grep -oE '[0-9]+$' || true)
  [[ -z "$issue_num" ]] && return 0

  metadata=$(query_issue_metadata "$issue_num")
  type=$(echo "$metadata" | jq -r '.type')
  milestone=$(echo "$metadata" | jq -r '.milestone')
  has_domain=$(echo "$metadata" | jq -r '.has_domain')
  ci_missing=$(check_ci_minimum "$type" "$has_domain" "$milestone")
  if [[ -n "$ci_missing" ]]; then
    emit_block "Issue #${issue_num} is missing taxonomy metadata: ${ci_missing}. Set these now before proceeding. Then add to sub-issue hierarchy, project board, and set priority."
    return 0
  fi
  # CI minimum is satisfied, but `gh issue create` can't set issue type —
  # always BLOCK to force completing the full lifecycle (hierarchy + project).
  emit_block "Issue #${issue_num} created with type, label, and milestone. You MUST now add it to the sub-issue hierarchy (addSubIssue to the appropriate Phase/Epic), add to project board, and set priority before proceeding."
}

mode_hierarchy() {
  # Usage: mode_hierarchy <command>
  # The addSubIssue mutation takes (parentId, childId) as opaque node IDs. We
  # GraphQL-resolve each ID to (number, type) and infer which is parent vs
  # child from the type — Epic/Phase are parents, Task/Bug/Feature are
  # children. BLOCK if a leaf is attached directly to an Epic.
  local command="$1" node_ids node_id result type num
  local parent_type="" parent_num="" child_type="" child_num=""
  node_ids=$(echo "$command" | grep -oE 'I_kw[A-Za-z0-9_/+=.-]+' || true)
  [[ -z "$node_ids" ]] && return 0

  while IFS= read -r node_id; do
    [[ -z "$node_id" ]] && continue
    # shellcheck disable=SC2016
    result=$(gh api graphql -f query='
      query($id: ID!) {
        node(id: $id) {
          ... on Issue { number issueType { name } }
        }
      }
    ' -f id="$node_id" 2>/dev/null || echo "{}")
    type=$(echo "$result" | jq -r '.data.node.issueType.name // empty' 2>/dev/null || echo "")
    num=$(echo "$result" | jq -r '.data.node.number // empty' 2>/dev/null || echo "")
    case "$type" in
      Epic|Phase)
        parent_type="$type"
        parent_num="$num"
        ;;
      Task|Bug|Feature)
        child_type="$type"
        child_num="$num"
        ;;
    esac
  done <<< "$node_ids"

  if [[ "$parent_type" == "Epic" && -n "$child_type" ]]; then
    emit_block "${child_type} #${child_num} cannot be a direct child of Epic #${parent_num}. Task/Bug/Feature issues must be sub-issues of a Phase, not an Epic. Add it under the appropriate Phase instead."
    return 0
  fi
  emit_advisory "Hierarchy check passed."
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
is_synth_setter() {
  # Usage: is_synth_setter <mode> <command> <tool_response>
  # Returns 0 if the command targets the synth-setter repo, 1 otherwise.
  #
  # The hook lives in this repo but agents may run `gh pr create` from a
  # sibling clone; we extract the actual GitHub URL from the response and
  # check the repo path. Plain `grep` on the response is unsafe — Bash tool
  # output includes "Shell cwd was reset to .../synth-setter" on every cwd
  # change. For addSubIssue the URL isn't in the response, so we inspect the
  # GraphQL query in the command string.
  local mode="$1" command="$2" tool_response="$3" github_url
  if [[ "$mode" == "hierarchy" ]]; then
    echo "$command" | grep -qE 'owner.*synth-setter|repo.*synth-setter'
    return $?
  fi
  github_url=$(echo "$tool_response" | grep -oE 'https://github.com/[^/]+/[^/]+/(pull|issues)/[0-9]+' | head -1 || true)
  [[ -z "$github_url" ]] && return 1
  echo "$github_url" | grep -q 'synth-setter'
}

main() {
  local input command mode tool_response
  input=$(cat)
  command=$(echo "$input" | jq -r '.tool_input.command // empty' 2>/dev/null || echo "")

  case "$command" in
    *"gh pr create"*)    mode="pr" ;;
    *"gh issue create"*) mode="issue" ;;
    *"addSubIssue"*)     mode="hierarchy" ;;
    *) return 0 ;;
  esac

  tool_response=$(echo "$input" | jq -r '.tool_response // empty' 2>/dev/null || echo "")

  is_synth_setter "$mode" "$command" "$tool_response" || return 0

  case "$mode" in
    pr)        mode_pr "$tool_response" ;;
    issue)     mode_issue "$tool_response" ;;
    hierarchy) mode_hierarchy "$command" ;;
  esac
}

main "$@"
