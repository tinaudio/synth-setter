# Checklist: Migrating to a GitHub Organization

> **Status**: COMPLETED — Migration to `tinaudio` org finished. This document
> is retained for historical reference only.
>
> **Author**: ktinubu@
> **Last Updated**: 2026-03-31

______________________________________________________________________

## Motivation

Moving `synth-permutations` from the personal `ktinubu` account to a GitHub
organization unlocks features that are only available at the org level:

- **Issue Types** — native Epic, Phase, Step, Bug types (no more label hacks)
- **Issue Fields** — typed metadata columns on issues (public preview)
- **Team management** — role-based access, code ownership by team
- **Native blocking relationships** — replace the current `blocked` label + body
  text convention with first-class issue dependencies

This checklist covers everything required before, during, and after the transfer.

______________________________________________________________________

## Pre-Migration Checklist

Complete all items before initiating the repository transfer.

### Open Pull Requests

Merge or close all open PRs. GitHub preserves PRs during transfer, but
reviewers lose access if they aren't added to the new org.

- [ ] PR #84
- [ ] PR #101
- [ ] PR #108

### Repository Secrets

Document all secrets that need re-creation in the new org. Values are stored
in 1Password — only names are listed here.

| Secret                     | Used by                        |
| -------------------------- | ------------------------------ |
| `ANTHROPIC_API_KEY`        | CI workflows (Claude agent)    |
| `APPROVAL_BOT_APP_ID`      | Approval bot GitHub App        |
| `APPROVAL_BOT_PRIVATE_KEY` | Approval bot GitHub App        |
| `RUNPOD_API_KEY`           | Pipeline orchestration scripts |

### Projects V2

All five projects are currently owned by the `ktinubu` user account, not the
repo. They will not transfer automatically.

| Project         | Purpose                            |
| --------------- | ---------------------------------- |
| CI & Automation | Workflow and bot tracking          |
| Data Pipeline   | Shard generation and finalize work |
| Code Health     | Lint cleanup, tech debt            |
| Evaluation      | Model evaluation tasks             |
| Training        | Training loop and experiment work  |

- [ ] Export each project's configuration (fields, views, item assignments)
- [ ] Screenshot or document custom views and filters
- [ ] Note which issues/PRs are assigned to each project

### CI and Integrations

- [ ] Verify all CI workflows pass on `main`
- [ ] List all GitHub Apps installed on the repo (e.g., Approval Bot)
- [ ] List all webhooks configured on the repo
- [ ] Document any branch protection rules

### Cleanup

- [ ] Delete stale branches (merged PRs, abandoned work)
- [ ] Remove leftover git worktrees
- [ ] Verify no draft releases or pending deployments

______________________________________________________________________

## Migration Steps

### 1. Create the GitHub Organization

- [ ] Create the org on GitHub (Settings > Organizations > New)
- [ ] Configure org-level settings (member permissions, default repo visibility)
- [ ] Invite any collaborators who need access

### 2. Transfer the Repository

- [ ] Go to repo Settings > Danger Zone > Transfer ownership
- [ ] Transfer `ktinubu/synth-permutations` to the new org
- [ ] Verify GitHub sets up the `ktinubu/synth-permutations` redirect

### 3. Re-create Secrets

Secrets do not transfer. Re-create each one in the new org's repo settings.

- [ ] `ANTHROPIC_API_KEY`
- [ ] `APPROVAL_BOT_APP_ID`
- [ ] `APPROVAL_BOT_PRIVATE_KEY`
- [ ] `RUNPOD_API_KEY`

### 4. Update Hardcoded URLs

Only one tracked file contains a hardcoded repo URL:

- [ ] `.github/agents/lint-cleanup.md` — update
  `https://github.com/ktinubu/synth-permutations/issues/25` to the new org URL

### 5. Re-link Projects V2

User-owned projects do not transfer with the repo.

- [ ] Create org-level projects (or re-link user projects)
- [ ] Recreate fields, views, and filters from the exported configuration
- [ ] Re-assign issues and PRs to the new projects

### 6. Re-install GitHub Apps

- [ ] Install the Approval Bot app on the new org
- [ ] Reconfigure any webhooks that pointed to the old repo URL

### 7. Verify CI

- [ ] Trigger a CI run on `main` (push an empty commit or re-run)
- [ ] Confirm all workflows pass with the new secrets

______________________________________________________________________

## Post-Migration Setup

New org-level features to enable after the transfer is complete.

### Issue Types

- [ ] Enable Issue Types in the org settings
- [ ] Create types: **Epic**, **Phase**, **Step**, **Bug**
- [ ] Re-type existing issues to use the new types instead of labels

### Native Blocking Relationships

- [ ] Enable native blocking/blocked-by relationships
- [ ] Migrate existing `blocked` label + body text conventions to native blocks
- [ ] Remove the `blocked` label once migration is complete

### Issue Fields (Public Preview)

- [ ] Enable Issue Fields if available
- [ ] Define typed metadata fields (e.g., priority, component, effort)

### Org-Level Settings

- [ ] Configure default labels for the org
- [ ] Set up team-based code ownership (CODEOWNERS)
- [ ] Configure member permission levels

______________________________________________________________________

## Post-Migration Verification

Run through every item to confirm nothing broke.

### CI and Workflows

- [ ] All CI workflows green on `main`
- [ ] Scheduled workflows (if any) trigger on schedule
- [ ] GitHub Actions minutes are tracked under the org

### Project Tracking

- [ ] All Projects V2 views show correct data
- [ ] Issue hierarchy (sub-issues / parent-child) intact
- [ ] Issue type assignments display correctly

### Access and Redirects

- [ ] Old `ktinubu/synth-permutations` URLs redirect to the new org
- [ ] PR cross-references in issue bodies still resolve
- [ ] Collaborators have correct permissions in the new org

### Secrets and Integrations

- [ ] Secrets accessible in CI workflows (validated by green CI)
- [ ] GitHub Apps functioning (Approval Bot responding)
- [ ] Webhooks firing correctly

### Local Development

- [ ] Update local git remotes: `git remote set-url origin <new-url>`
- [ ] Verify `git push` and `git pull` work against the new remote
- [ ] Update any local scripts or aliases that reference the old URL
