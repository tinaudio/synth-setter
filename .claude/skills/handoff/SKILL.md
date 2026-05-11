---
name: handoff
description: |
  Generate the two artifacts a fresh agent needs to continue a multi-PR chain:
  (1) a handoff-update comment posted to the tracking issue, and (2) a startup
  prompt printed inline (and saved to .claude/handoffs/) for the next session.
  Triggered by `/handoff [--issue N] [--dry-run] [--comment-only] [--prompt-only]`.
  Derives state from chain.yaml + live `gh` queries (prior handoffs, merged PRs,
  open PRs, Phase task numbering, worktree inventory). Use whenever the user
  asks to "hand off the chain", "write the next-agent prompt", "post a handoff
  update", or after a session has driven part of a PR plan toward merge and
  needs to transfer to a fresh agent.
---

# handoff — chain handoff artifact generator

You MUST complete every step in order.

## What this skill does

Two artifacts are produced from session state:

1. **Comment** — a `## Handoff update` comment posted to the chain's tracking issue (e.g. #882). Mirrors the structure of [this reference handoff](https://github.com/tinaudio/synth-setter/issues/882#issuecomment-4425353839): done-since-prior, in-flight PRs with explicit next-actions, remaining-chain table (collapsed), Phase task-numbering, recommended task list, worktree inventory, surprises, and ongoing-conventions reminder.
2. **Prompt** — a markdown block the human pastes into the next fresh agent's first message. Includes a START-HERE link to the just-posted comment, three first-commands-to-run, the dependency graph, the default-action narrative, mandatory-skills cycle, anti-patterns list, and PR-readiness checklist.

The chain itself is encoded in `.claude/skills/handoff/chain.yaml`. The skill reads it, reconciles status against live PR state, writes back only when something changed, and renders the artifacts.

## Step 1: Determine the tracking issue

Default: use the `tracking_issue` from `.claude/skills/handoff/chain.yaml`.

User overrides:

- `/handoff --issue 874` — use issue #874 (e.g. the wds-tail chain).
- `/handoff --chain .claude/skills/handoff/chain.wds.yaml` — use a different manifest file.

If `chain.yaml` is missing or has no `tracking_issue`, ask the user via `AskUserQuestion` which issue to use, and offer to persist their answer back into the manifest so future invocations don't re-ask.

## Step 2: Collect Surprises and Anti-patterns from the session

The "Surprises" comment section and the "Anti-patterns" prompt section keep the lessons-learned curated by the user, not auto-fabricated. Unless `--no-prompt-questions` was passed:

1. Use `AskUserQuestion` to prompt for **Surprises** with `multiSelect: true` and categories: `Copilot race`, `Maintainer-self-push`, `Infra/dep surprise`, `Tool-quirk`, `Process-race`, `Other`. Skip the question if `multiSelect` is unavailable in the environment — fall back to a plain prompt.
2. For each selected category, ask the user for a one-line note describing the surprise.
3. Separately ask for any **Anti-patterns** (things the prior agent tried that didn't work — short list, one-liners).
4. Pass them to the writer via repeated `--surprise "<category>: <note>"` and `--anti-pattern "<line>"` flags.

If you cannot prompt (non-interactive environment, or user passed `--no-prompt-questions`), render with the sections empty. Better to ship a working handoff than to block on missing lessons.

## Step 3: Invoke the writer

The writer composes both artifacts, applies the idempotency guard, and (unless gated) posts the comment plus saves the prompt locally.

```bash
python3 .claude/skills/handoff/helpers/write_handoff.py \
  [--issue N] \
  [--repo OWNER/NAME] \
  [--chain PATH] \
  [--dry-run] \
  [--comment-only | --prompt-only] \
  [--surprise "<category>: <note>" ...] \
  [--anti-pattern "<line>" ...] \
  [--no-prompt-questions] \
  [--no-save-prompt] \
  [--force]
```

What it does in order:

1. Reads `chain.yaml`.
2. Calls `discover_state.derive_state(...)` — queries the tracking issue's comments (filters those whose body starts with `## Handoff update`), merged PRs since the most-recent handoff timestamp (`gh pr list --search "#<N> merged:>YYYY-MM-DDTHH:MM:SSZ"`), open PRs referencing the tracking issue, per-in-flight-PR health (mergeable, review decision, status checks, unresolved review threads via GraphQL, new Copilot comments since head-commit committer date), Phase task-number ceiling (sub-issues of the parent phase, regex-matched against the chain's `task_prefix`), and `git worktree list --porcelain` cross-referenced with `gh pr view <branch>`.
3. Reconciles `chain.yaml`'s `status` / `pr_number` columns against the live data. Merged PRs become `status: merged`; open PRs matching a chain title become `status: in_flight`. Writes the file back only when something actually changed (preserves the file's leading header comment block via `CHAIN_HEADER`).
4. **Idempotency guard:** if the most recent prior handoff is younger than 30 minutes, refuses to post unless `--force` is passed. Override only when you have a concrete reason (e.g. a major event happened immediately after the last handoff).
5. Renders `templates/comment.md.j2` and `templates/prompt.md.j2` (Jinja2, `StrictUndefined`) against the derived state.
6. Posts the comment via `gh issue comment <N> --body-file <tmp>` and captures its `html_url`.
7. Re-renders the prompt with the just-posted comment URL embedded as `START HERE`.
8. Prints the prompt to stdout (unless `--comment-only`).
9. Saves the prompt to `.claude/handoffs/handoff-YYYY-MM-DD-HHMM.md` (unless `--no-save-prompt`). The `.claude/` tree is `.gitignore`d, so these files stay session-local.

## Step 4: Report back to the user

After a successful run:

- Quote the posted comment's URL.
- Quote the local path the prompt was saved to.
- Tell the user the prompt was also printed inline above — they paste it into the next agent's first message.

If the writer exited non-zero, surface the stderr verbatim and stop. Common failure modes: missing `chain.yaml`, `gh` not authenticated, the idempotency guard refused without `--force`.

## Argument surface

| Flag | Effect |
|------|--------|
| (none) | Full flow: render, post comment, print + save prompt |
| `--issue N` | Override `chain.yaml`'s `tracking_issue` |
| `--repo OWNER/NAME` | Override `chain.yaml`'s `repo` |
| `--chain PATH` | Use a different manifest (e.g. `chain.wds.yaml`) |
| `--dry-run` | Render both, post nothing, print everything, leave `chain.yaml` unchanged |
| `--comment-only` | Post comment, suppress prompt printing |
| `--prompt-only` | Print prompt, suppress comment posting |
| `--no-prompt-questions` | Skip the AskUserQuestion step; render with empty Surprises / Anti-patterns sections |
| `--force` | Bypass the 30-min idempotency guard |
| `--surprise "<cat>: <note>"` | Add one Surprise row (repeatable) |
| `--anti-pattern "<line>"` | Add one Anti-pattern line (repeatable) |
| `--no-save-prompt` | Don't save the prompt to `.claude/handoffs/` |

## Constraints (hard rules)

- **Never auto-merge a PR.** The chain ends at `gh pr merge` — the user runs that. Handoffs are reversible (delete the comment); merges are not.
- **No new env vars.** The skill must run with whatever the repo already requires (`gh` auth, `git`).
- **No absolute paths in templates or imports.** `SKILL_DIR = Path(__file__).resolve().parent.parent` anchors everything.
- **No new dependencies.** `jinja2` and `pyyaml` are already in the environment (jinja2 transitively via pytorch-lightning; pyyaml in `requirements-app.txt`). Don't add `ruamel.yaml` or other.
- **One tracking issue per invocation.** Multi-issue chains are out of scope — copy `chain.yaml` to `chain.wds.yaml` and pass `--chain` for a parallel chain.
- **Skill files (`.claude/handoffs/handoff-*.md`) MUST NOT be committed.** The `.claude/` tree is already `.gitignore`d; the explicit override line in `.gitignore` documents the intent for any future contributor who adds a `!.claude/` allow-list.

## When the `chain.yaml` doesn't fit the situation

If the user is starting a brand-new chain (no existing manifest), don't try to back-fill from PR history. Instead:

1. Ask the user for the tracking issue, the parent Phase, the task prefix, and the rough list of upcoming PRs.
2. Write a minimal `chain.yaml` with one plan_pr per upcoming PR (`status: pending`, `pr_number: null`).
3. Run `/handoff` against the new manifest — its first invocation will have an empty Done-since table and no in-flight PRs, which is correct.

## How the skill is tested

The unit tests under `.claude/skills/handoff/tests/` cover the pure-function surface:

- `test_discover_state.py` — chain.yaml round-trip, worktree porcelain parsing, prior-handoff filtering, Phase task-number extraction, Copilot-comment timestamp filtering, status-check rollup summarization, unresolved-thread counting, chain status reconciliation.
- `test_write_handoff.py` — template renders (comment + prompt), per-in-flight-PR next-action wording for each state (failing / conflicting / unresolved-threads / approved-clean), prior-handoff breadcrumb rendering, safe-to-remove worktree annotation, idempotency guard window, surprise parsing, ASCII dependency graph for linear-chain and fan-out shapes.

Run with:

```bash
python3 -m pytest .claude/skills/handoff/tests/ -v
```

For the integration smoke test, run `/handoff --dry-run` from a live session and spot-check the rendered artifacts against the prior handoff comment they're meant to supersede. The diff between hand-written and skill-generated should be structurally minor (formatting, predecessor breadcrumbs, dependency graph) — no missing rows, no fabricated rows.
