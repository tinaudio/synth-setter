---
name: repo-review-full
description: |
  Full multi-skill PR review. Fans out parallel agents (code-health, shell-style,
  gha-workflow-validator, synth-setter-project-standards, python-style,
  tdd-refactor, plus tdd-implementation/ml-data-pipeline/ml-test when the diff
  warrants) and posts every BLOCK/WARN as an individual unresolved inline PR
  review comment. Reproduces the workflow demoed on PR #777. Requires the
  tinaudio-synth-setter-skills plugin.
---

# repo-review-full — Multi-Skill Parallel PR Review

You MUST complete every step in order.

## Step 1: Resolve the PR

Determine the PR number:

- If invoked `/repo-review-full <N>`, use `<N>`.
- Otherwise resolve via `gh pr view --json number`.

Fetch metadata once:

```bash
gh pr view <N> --repo "$(gh repo view --json nameWithOwner -q .nameWithOwner)" \
  --json number,headRefOid,baseRefOid,files,title,headRefName
```

If no PR exists for the current branch, stop and tell the user to push and open a PR first.

## Step 2: Pick which skills to fan out to

Read the file list from Step 1. Map file types → relevant skills:

| File pattern | Skills that always run |
|---|---|
| Always | `code-health`, `synth-setter-project-standards` |
| `*.py` | `python-style`, `tdd-implementation` |
| `*.sh` or bash inside YAML `run:` blocks | `shell-style` |
| `.github/workflows/*.{yml,yaml}` | `gha-workflow-validator` |
| ML model / pipeline / training code under `src/` or `pipeline/` | `ml-data-pipeline`, `ml-test` |
| Diff renames or moves files (anything with `R` in `git diff --name-status`) | `tdd-refactor` |

Always run `code-health` and `synth-setter-project-standards`. Other skills opt in based on file extensions in the diff. Note which skills you selected; you'll launch one parallel agent per skill.

## Step 3: Launch parallel review agents

Launch one `general-purpose` Agent per selected skill. **All agents in a single message** so they run concurrently (Claude Code supports this — one message with N tool calls = N parallel agents).

Each agent's prompt MUST include:

- The PR number, repo, base SHA, head SHA.
- The full file list (with per-file line counts is helpful but optional).
- The exact skill to invoke: `Invoke the tinaudio-synth-setter-skills:<skill-name> skill via the Skill tool and apply its checklist to this PR's diff.`
- The expected output shape (see below).

Each agent returns a Markdown report with `BLOCK` and `WARN` sections. Each finding cites `<path>:<line>`. Agents work independently — they should not coordinate.

### Per-agent output contract

Each agent returns a Markdown block:

````
## <skill-name> review — PR #<N>

### BLOCK findings
1. **<path>:<line>** — <description>

### WARN findings
1. **<path>:<line>** — <description>

### What looks good
- ...
````

Aim each agent at a 1500-word ceiling so reports stay scannable. The orchestrator (you) can ask for tighter output if a skill's domain is small.

## Step 4: Aggregate findings

Once every parallel agent returns, parse each report's BLOCK and WARN findings. For each finding, build one entry in the post-review JSON. Prefix each comment body with `[<skill>:<severity>]` so reviewers can see which checklist surfaced it.

Severity → severity tag:

- `BLOCK` → `block`
- `WARN` → `warn`

Skill → tag (short form for comment body):

| Plugin skill | Comment-body tag |
|---|---|
| `code-health` | `code-health` |
| `shell-style` | `shell-style` |
| `python-style` | `python-style` |
| `gha-workflow-validator` | `gha` |
| `synth-setter-project-standards` | `synth-setter` |
| `tdd-implementation` | `tdd-impl` |
| `tdd-refactor` | `tdd-refactor` |
| `ml-data-pipeline` | `ml-pipeline` |
| `ml-test` | `ml-test` |

A finding becomes:

```json
{
  "path": "<path>",
  "line": <line>,
  "body": "**[<short-tag>:<severity>]** <description>"
}
```

Do NOT dedupe near-duplicate findings across skills (e.g. shell-style and synth-setter-project-standards both flagging a `[ ]` vs `[[ ]]` issue) — keep each skill's signal independent. Acceptable noise per the plan's out-of-scope list.

## Step 5: Build the findings JSON

Same shape `post_review.py` consumes:

```json
{
  "pr_number": <N>,
  "repo": "<owner>/<repo>",
  "review_body": "Multi-skill review of PR #<N> — <K> parallel passes (<list of skills>). Every BLOCK/WARN posted below as an individual unresolved thread. Findings on files outside the diff are anchored to the line in the diff that *causes* the staleness or rolled into the review body.",
  "findings": [ ... ]
}
```

Write the JSON to a temp file:

```bash
cat > /tmp/repo-review-full-findings.json <<'JSON'
... payload ...
JSON
```

## Step 6: Submit the review

```bash
python3 .claude/skills/_shared/post_review.py < /tmp/repo-review-full-findings.json
```

Helper behavior matches `repo-review`:

- Anchors each finding to its target line if that line falls inside a diff hunk.
- Falls back to the nearest in-hunk line on the same file with a cross-ref note prepended to the body.
- Rolls orphan findings (file outside the diff) into the review body under `## Findings on files outside the diff`.
- Submits as `event=COMMENT`. Threads stay unresolved.

Report the helper's `html_url` back to the user along with a one-line summary (`Posted N findings: B BLOCK + W WARN across K skills`).

## Notes

- This skill depends on the `tinaudio-synth-setter-skills` plugin being enabled. If a sub-skill invocation fails, surface the error — don't silently skip. Falling back to `repo-review` (MVP) is the user's call, not the skill's.
- Each parallel agent is a *general-purpose* sub-agent that itself invokes a plugin skill via the Skill tool. The two-level structure is intentional: the parallel fan-out is the orchestrator's contribution; each plugin skill's authoritative checklist is the source of truth for its domain.
- For the concrete invocation pattern (parallel Agent tool calls in a single message, expected per-agent prompt shape), see the example trace recorded in PR #777's review history — that's the workflow this skill packages.
