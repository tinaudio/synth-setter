#!/bin/bash
# Recover a stale Semantic Release checkout without rewriting published history.
set -euo pipefail

readonly GIT_NO_MATCH_EXIT_STATUS=2

fail() {
  printf '::error::%s\n' "$1" >&2
  exit 1
}

: "${GITHUB_OUTPUT:?missing GITHUB_OUTPUT}"
: "${GITHUB_WORKSPACE:?missing GITHUB_WORKSPACE}"
: "${RELEASE_BASE:?missing RELEASE_BASE}"
if [[ "${GITHUB_ACTIONS:-false}" != "true" ]]; then
  fail "Recovery is restricted to GitHub Actions."
fi
if [[ "${RELEASE_RECOVERY_ALLOWED:-false}" != "true" ]]; then
  fail "Release recovery requires explicit opt-in."
fi

repository_root="$(git rev-parse --show-toplevel)"
workspace_root="$(realpath "${GITHUB_WORKSPACE}")"
if [[ "${repository_root}" != "${workspace_root}" ]]; then
  fail "Recovery must run at GITHUB_WORKSPACE."
fi

git fetch --force --tags origin main:refs/remotes/origin/main
remote_tip="$(git rev-parse refs/remotes/origin/main)"
local_tip="$(git rev-parse HEAD)"

if [[ "${local_tip}" != "${RELEASE_BASE}" ]] &&
  git merge-base --is-ancestor "${local_tip}" "${remote_tip}"; then
  fail "The failed release commit may already be published; refusing to retry."
fi
if [[ "${remote_tip}" == "${RELEASE_BASE}" ]]; then
  fail "Main did not advance; refusing an unrelated release retry."
fi
if ! git merge-base --is-ancestor "${RELEASE_BASE}" "${remote_tip}"; then
  fail "Main diverged from the release base; refusing to rewrite history."
fi

if [[ "${local_tip}" != "${RELEASE_BASE}" ]]; then
  if ! release_tags="$(git tag --points-at "${local_tip}")"; then
    fail "Unable to inspect local release tags."
  fi
  if [[ -n "${release_tags}" ]]; then
    while IFS= read -r tag; do
      if git ls-remote --exit-code --tags origin \
        "refs/tags/${tag}" >/dev/null 2>&1; then
        fail "Release tag ${tag} is already published; refusing to retry."
      else
        tag_lookup_status=$?
        if [[ "${tag_lookup_status}" -ne "${GIT_NO_MATCH_EXIT_STATUS}" ]]; then
          fail "Unable to verify whether release tag ${tag} is published."
        fi
      fi
      git tag --delete "${tag}"
    done <<<"${release_tags}"
  fi
fi

git clean -ffdx
git reset --hard "${remote_tip}"
printf 'base=%s\n' "${remote_tip}" >>"${GITHUB_OUTPUT}"

if [[ "${FAIL_AFTER_RECOVERY:-false}" == "true" ]]; then
  fail "Main advanced twice; leaving the release workflow failed."
fi
