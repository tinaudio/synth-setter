# CI Triage Agent — Operator Notes

Autonomous Claude Code agent that triages failed workflow runs. Design and
rationale: [#923](https://github.com/tinaudio/synth-setter/issues/923).

## How it fires

`.github/workflows/ci-triage.yaml` listens for `workflow_run` completion on
the workflows in its `on.workflow_run.workflows` list (see the workflow
file for the authoritative list). On `conclusion: failure`, the job:

1. Skips if the failing run was on a fork (head_repository != this repo) —
   write perms must not run against untrusted code.
2. Skips if `CLAUDE_CODE_OAUTH_TOKEN` is unset.
3. Skips if a `ci-triage/run-<id>` branch already exists (cooldown).
4. Installs `act` + Claude Code CLI, writes `/tmp/triage/context.json`, then
   invokes `claude -p` with `.github/triage/triage-prompt.md`.

The agent's transcript is uploaded as the
`ci-triage-transcript-<run-id>` artifact with 14-day retention.

## One-time setup

The workflow is a no-op until you add the OAuth secret:

```bash
gh secret set CLAUDE_CODE_OAUTH_TOKEN --repo tinaudio/synth-setter
# paste the token when prompted
```

Generate the token from a local Claude Code session:

```bash
claude setup-token
```

(One token per repo. Rotate by re-running `setup-token` and overwriting the
secret — old tokens stop working after rotation.)

## Approving a fix PR

The agent only opens **draft** PRs. To merge:

1. Open the PR and read the body — the agent records its classification,
   reproduction outcome (did `act` repro?), and links to the transcript
   artifact.
2. Download the transcript artifact and skim `REPORT.md`. If the agent's
   reasoning is sound, mark the PR ready for review and follow the normal
   review flow.
3. If the agent's reasoning is wrong, close the PR and `gh issue create` a
   bug against this triage workflow with a link to the transcript.

The agent will never auto-merge. The `Auto-approve PR` workflow does not
approve `ci-triage/*` PRs because they're drafts.

## Disabling the agent

Disable the workflow without removing files:

```bash
gh workflow disable "CI Triage Agent" --repo tinaudio/synth-setter
```

Or delete the `CLAUDE_CODE_OAUTH_TOKEN` secret — same effect, but the job
will still spin up and bail out in the token-check step.

## Manual re-trigger

If a failure slipped through (or the agent crashed mid-run), re-trigger
against a specific run ID:

```bash
gh workflow run ci-triage.yaml --repo tinaudio/synth-setter \
  -f run_id=<failing-run-id>
```

Delete the existing `ci-triage/run-<id>` branch first if you want to bypass
the cooldown.

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

- Push to `main`, `release/*`, or `dev`.
- Edit application code, test code, or production configs in a PR. Only
  telemetry/logging changes go in via the agent — anything else gets an
  issue.
- Modify dependency versions.
- Rotate or read secrets.
- Mark its own PRs ready for review or apply auto-merge labels.

## Cost notes

Every failed run on the watched workflows fires this job. The job is
~5 minutes (mostly the agent), so cost ~= (failure rate) × (token cost per
session, capped by `--max-turns` in `.github/workflows/ci-triage.yaml`).
Watch the `Actions` minutes budget; if it gets noisy, trim the `workflows:`
list in `ci-triage.yaml` to the high-value ones and let small flakes go
untriaged.
