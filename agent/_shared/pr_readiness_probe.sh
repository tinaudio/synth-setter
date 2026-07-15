#!/usr/bin/env bash
# pr_readiness_probe.sh — one-shot readiness report for a PR, the single gate
# source shared by the /pr-readiness polling loop and
# agent/hooks/pr-readiness-stop.sh (docs/pr-readiness-loop.md is canonical).
#
# Usage: pr_readiness_probe.sh [--gates-only] <pr-number>
#
# Prints one line per gate: Gate 1 (CI), Gate 2 (mergeable), Gate 3 (review
# threads awaiting a reply) are enforced; Gate 4 (Copilot re-review) is
# advisory-only and printed when gates 1-3 hold. --gates-only skips the
# gate-4 lookup (two paginated REST calls) for callers that only consume the
# exit code, like the Stop hook. Merged/closed PRs short-circuit to READY.
# Exit: 0 = gates 1-3 hold, 1 = a gate failed, 2 = usage/environment error
# (never a silent pass).
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") [--gates-only] <pr-number>" >&2
}

fail_env() {
  printf 'probe error: %s\n' "$*" >&2
  exit 2
}

gates_only=0
if [[ "${1:-}" == "--gates-only" ]]; then
  gates_only=1
  shift
fi
[[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]] || { usage; exit 2; }
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

gates_hold=1

if gh pr checks "$PR" >/dev/null 2>&1; then
  echo "Gate 1 (CI): PASS — all checks green"
else
  echo "Gate 1 (CI): FAIL — checks failing or still pending" \
    "(gh pr checks ${PR})"
  gates_hold=0
fi

if [[ "$mergeable" == "MERGEABLE" ]]; then
  echo "Gate 2 (mergeable): PASS"
else
  echo "Gate 2 (mergeable): FAIL — ${mergeable:-UNKNOWN} (want MERGEABLE)"
  gates_hold=0
fi

# A thread "awaits a reply" when it is unresolved and either has a single
# comment (nobody replied — including self-authored sentinel findings, where
# author == PR author) or its most recent comment is not the PR author's.
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

awaiting=$(jq -r --arg author "$pr_author" '
  .data.repository.pullRequest.reviewThreads.nodes[]
  | select(.isResolved | not)
  | select((.first.totalCount == 1)
           or ((.last.nodes[0].author.login // "") != $author))
  | .first.nodes[0]
  | "  - \(.path):\(.line // .originalLine // "?") @\(.author.login // "?"): "
    + (.body | split("\n")[0] | .[0:100])
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
  gates_hold=0
fi

if [[ "$gates_hold" == "1" && "$gates_only" == "0" ]]; then
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

if [[ "$gates_hold" == "1" ]]; then
  echo "READY: gates 1-3 hold."
  exit 0
fi

echo "NOT READY: fix the failing gate(s) above, push, and re-probe." \
  "See docs/pr-readiness-loop.md."
exit 1
