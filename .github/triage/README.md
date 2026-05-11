# CI Triage Agent — Local Operator Notes

A locally-invoked Claude Code agent that triages a failing GitHub Actions
run. Design and rationale: [#923](https://github.com/tinaudio/synth-setter/issues/923).

> **Local-only.** Claude Code agents must run on the developer's machine —
> there is no CI-side invocation. The helper script + prompt template are
> designed for direct local use against `claude -p`.

## What it does

Given a failing workflow run ID, the agent:

1. Fetches the run's failed-job logs via `gh run view --log-failed`.
2. Classifies the failure: `flake`, `resource_starvation`, `auth`, or
   `real_bug`.
3. Optionally reproduces locally with `nektos/act` (for the deterministic
   buckets).
4. Routes to one of:
   - **Path A** — opens a draft PR with telemetry additions (mirroring
     [PR #876](https://github.com/tinaudio/synth-setter/pull/876)'s
     `Snapshot sky logs` step).
   - **Path B** — files a taxonomy-compliant tracking issue.
   - **Path C** — re-runs a single flake.

The full agent playbook lives in `triage-prompt.md`.

## Prerequisites

One-time, on your local machine:

```bash
# Claude Code CLI (local-only)
npm install -g @anthropic-ai/claude-code
claude --version

# gh authenticated with repo write scope
gh auth login --scopes repo

# nektos/act for local reproduction (optional but recommended)
curl -fsSL https://raw.githubusercontent.com/nektos/act/master/install.sh \
  | sudo bash -s -- -b /usr/local/bin
act --version
```

## Usage

Given a failing run ID — copy it from the GitHub Actions UI or
`gh run list --status failure`:

```bash
./scripts/triage-ci.sh 25687139710
```

The helper script writes `/tmp/triage/context.json`, then pipes the prompt
template into `claude -p`. Watch the agent's output; the final response is
the contents of `/tmp/triage/REPORT.md` summarizing what it did.

To run the agent by hand without the wrapper:

```bash
# Write the context sidecar yourself:
RUN_ID=25687139710
REPO=tinaudio/synth-setter
mkdir -p /tmp/triage
gh api "repos/${REPO}/actions/runs/${RUN_ID}" \
  --jq '{run_id: .id, name, head_branch, head_sha, conclusion, html_url, event, workflow_id}' \
  > /tmp/triage/run.json
jq -n \
  --arg repo "${REPO}" \
  --arg branch "ci-triage/run-${RUN_ID}" \
  --slurpfile run /tmp/triage/run.json \
  '{repo: $repo, triage_branch: $branch, run: $run[0]}' \
  > /tmp/triage/context.json

# Then invoke the agent:
claude -p "$(cat .github/triage/triage-prompt.md)" \
  --max-turns 30 \
  --permission-mode acceptEdits
```

## Reviewing what the agent produced

The agent only opens **draft** PRs. To merge:

1. Open the PR and read the body — the agent records its classification,
   reproduction outcome (did `act` repro?), and a link to the failing run.
2. Skim `/tmp/triage/REPORT.md` from your local triage session.
3. If the agent's reasoning is sound, mark the PR ready for review and
   follow the normal review flow.
4. If the agent's reasoning is wrong, close the PR (and the branch) and
   file a bug against the prompt template with a link to what the agent
   actually did vs. what was correct.

The agent will never auto-merge. The `Auto-approve PR` workflow does not
approve `ci-triage/*` PRs because they're drafts.

## What the agent will and will not do

It **will**:

- Read failing logs, classify into `flake` / `resource_starvation` / `auth` /
  `real_bug`.
- Run `act` to attempt local reproduction (for `real_bug` and
  `resource_starvation` only).
- Open a draft PR with telemetry additions (PR #876 pattern) when repro
  succeeds and the fix is mechanical.
- File a taxonomy-compliant tracking issue (default route).
- Re-run a single flake.

It **will not**:

- Run in GitHub Actions or any CI environment (Claude Code is local-only).
- Push to `main`, `release/*`, or `dev`.
- Edit application code, test code, or production configs in a PR. Only
  telemetry/logging changes go in via the agent — anything else gets an
  issue.
- Modify dependency versions.
- Rotate or read secrets.
- Mark its own PRs ready for review or apply auto-merge labels.
