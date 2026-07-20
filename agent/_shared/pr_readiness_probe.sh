#!/usr/bin/env bash
# pr_readiness_probe.sh — four-gate PR readiness probe shared by the Stop hook
# and the /pr-readiness loop; semantics + exit contract: docs/pr-readiness-loop.md § The probe.
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") [--gates-only] [--loop] <pr-number>" >&2
}

fail_usage() {
  usage
  [[ "${loop_mode:-0}" == "1" ]] && exit 0
  exit 2
}

fail_env() {
  printf 'ERROR: probe could not evaluate readiness: %s\n' "$*" >&2
  [[ "${loop_mode:-0}" == "1" ]] && exit 0
  exit 2
}

# Gate-3 listing truncates each finding body to one locator-sized line.
readonly BODY_PREVIEW_CHARS=100

gates_only=0
loop_mode=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --gates-only) gates_only=1 ;;
    --loop) loop_mode=1 ;;
    *) fail_usage ;;
  esac
  shift
done
[[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]] || fail_usage
readonly PR="$1"

command -v gh >/dev/null 2>&1 || fail_env "gh not on PATH"
command -v jq >/dev/null 2>&1 || fail_env "jq not on PATH"

meta=$(gh pr view "$PR" --json state,mergeable,headRefOid,author,url 2>&1) \
  || fail_env "gh pr view ${PR} failed: ${meta}"
# The 2>&1 capture (for the error message above) can prepend gh notices to the
# JSON on success — validate before parsing so a jq failure can't exit 1 and
# masquerade as a gate verdict.
jq -e . >/dev/null 2>&1 <<<"$meta" \
  || fail_env "gh pr view ${PR} returned non-JSON: ${meta}"
pr_state=$(jq -r '.state // ""' <<<"$meta")
mergeable=$(jq -r '.mergeable // ""' <<<"$meta")
head_sha=$(jq -r '.headRefOid // ""' <<<"$meta")
pr_author=$(jq -r '.author.login // ""' <<<"$meta")
# Owner/repo parsed from the PR url (https://github.com/<owner>/<repo>/pull/N)
# — saves a `gh repo view` round-trip and stays correct across forks.
owner=$(jq -r '.url | split("/")[3]' <<<"$meta")
name=$(jq -r '.url | split("/")[4]' <<<"$meta")

# mergeable=UNKNOWN is expected on merged/closed PRs; gates are moot there.
if [[ "$pr_state" == "MERGED" || "$pr_state" == "CLOSED" ]]; then
  echo "READY: PR #${PR} is ${pr_state} — readiness gates no longer apply."
  exit 0
fi

action_required=0
wait_required=0

checks_rc=0
checks=$(gh pr checks "$PR" --json name,state,bucket,link 2>&1) \
  || checks_rc=$?
if [[ "$checks_rc" -ne 0 && "$checks_rc" -ne 1 && "$checks_rc" -ne 8 ]]; then
  fail_env "gh pr checks ${PR} failed (${checks_rc}): ${checks}"
fi
if ! jq -e 'type == "array"' >/dev/null 2>&1 <<<"$checks"; then
  if [[ "$checks_rc" -eq 1 && "$checks" == no\ checks\ reported* ]]; then
    echo "Gate 1 (CI): WAIT — ${checks}"
    wait_required=1
  else
    fail_env "gh pr checks ${PR} returned non-JSON: ${checks}"
  fi
else
  failed_checks=$(jq -r '
    .[] | select(.bucket == "fail" or .bucket == "cancel")
    | "  - \(.name) (\(.state)) — " + (.link // "no URL")
  ' <<<"$checks")
  pending_count=$(jq \
    '[.[] | select(.bucket == "pending")] | length' <<<"$checks")
  if [[ -n "$failed_checks" ]]; then
    count=$(grep -c . <<<"$failed_checks")
    echo "Gate 1 (CI): ACTION — ${count} terminal check failure(s):"
    printf '%s\n' "$failed_checks"
    action_required=1
  elif [[ "$pending_count" -gt 0 ]]; then
    echo "Gate 1 (CI): WAIT — ${pending_count} check(s) still pending"
    wait_required=1
  else
    echo "Gate 1 (CI): PASS — all checks green"
  fi
fi

case "$mergeable" in
  MERGEABLE)
    echo "Gate 2 (mergeable): PASS"
    ;;
  CONFLICTING)
    echo "Gate 2 (mergeable): ACTION — CONFLICTING;" \
      "update from the base branch"
    action_required=1
    ;;
  *)
    echo "Gate 2 (mergeable): WAIT — ${mergeable:-UNKNOWN}"
    wait_required=1
    ;;
esac

# Awaiting = unresolved && (single comment || last comment not the PR
# author's) — so a finding self-posted by the author still counts until replied.
# shellcheck disable=SC2016  # $-names in the query are GraphQL variables, not shell
threads=$(gh api graphql --paginate \
  -f owner="$owner" -f name="$name" -F pr="$PR" \
  -f query='query($owner: String!, $name: String!, $pr: Int!, $endCursor: String) {
    repository(owner: $owner, name: $name) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100, after: $endCursor) {
          pageInfo { hasNextPage endCursor }
          nodes {
            isResolved
            first: comments(first: 1) {
              totalCount
              nodes { author { login } path line originalLine body }
            }
            last: comments(last: 1) { nodes { author { login } } }
          }
        }
      }
    }
  }' 2>&1) || fail_env "reviewThreads query failed: ${threads}"

awaiting=$(jq -r --arg author "$pr_author" --argjson chars "$BODY_PREVIEW_CHARS" '
  .data.repository.pullRequest.reviewThreads.nodes[]
  | select(.isResolved | not)
  | select((.first.totalCount == 1)
           or ((.last.nodes[0].author.login // "") != $author))
  | .first.nodes[0]
  | "  - \(.path):\(.line // .originalLine // "?") @\(.author.login // "?"): "
    + (.body | split("\n")[0] | .[0:$chars])
' <<<"$threads") || fail_env "reviewThreads parse failed"

if [[ -z "$awaiting" ]]; then
  echo "Gate 3 (review threads): PASS — no unresolved threads awaiting a reply"
else
  count=$(grep -c . <<<"$awaiting")
  echo "Gate 3 (review threads): FAIL — ${count} unresolved thread(s)" \
    "awaiting a reply:"
  printf '%s\n' "$awaiting"
  echo "  Drive /pr-review-resolver: reply inline with a fix-commit SHA" \
    "or justification."
  action_required=1
fi

if [[ "$action_required" == "0" && "$wait_required" == "0" \
  && "$gates_only" == "0" ]]; then
  short_head=${head_sha:0:7}
  copilot_err=0
  comments_json=$(gh api "repos/${owner}/${name}/pulls/${PR}/comments" \
    --paginate 2>/dev/null) || copilot_err=1
  reviews_json=$(gh api "repos/${owner}/${name}/pulls/${PR}/reviews" \
    --paginate 2>/dev/null) || copilot_err=1
  copilot_hits=$(printf '%s\n%s\n' "$comments_json" "$reviews_json" \
    | jq -r --arg head "$head_sha" '[.[]
        | select(((.user.login // "") | test("copilot"; "i"))
                 and .commit_id == $head)] | length' \
    | awk '{ total += $1 } END { print total + 0 }') || copilot_err=1
  if [[ "$copilot_err" == "1" ]]; then
    echo "Gate 4 (Copilot, advisory): could not query Copilot activity" \
      "— check manually"
  elif [[ "$copilot_hits" -gt 0 ]]; then
    echo "Gate 4 (Copilot, advisory): Copilot has reviewed head" \
      "${short_head} — fresh findings appear as Gate 3 threads"
  else
    echo "Gate 4 (Copilot, advisory): no Copilot activity on head" \
      "${short_head} yet — wait ~60s (up to 15 min), then" \
      "docs/pr-readiness-loop.md step 6a"
  fi
fi

if [[ "$action_required" == "1" ]]; then
  echo "ACTION_REQUIRED: stop polling and remediate the actionable" \
    "gate(s) above."
  [[ "$loop_mode" == "1" ]] && exit 0
  exit 1
fi

if [[ "$wait_required" == "1" ]]; then
  echo "WAIT: only transient readiness work remains; continue monitoring."
  [[ "$loop_mode" == "1" ]] && exit 1
  exit 8
fi

echo "READY: gates 1-3 hold."
exit 0
