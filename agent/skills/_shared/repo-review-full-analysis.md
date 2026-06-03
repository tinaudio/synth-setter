# repo-review-full analysis steps

Shared analysis pipeline for `repo-review-full` and `repo-review-full-no-comments`.
Both skills run the same Steps 1–6 below; only the final delivery step (post inline
comments vs. print the report to the user) differs and lives in each skill's
SKILL.md.

You MUST complete every step below in order, then return to the calling skill's
SKILL.md for the final delivery step.

## Step 1: Resolve the PR

Determine the PR number:

- If the caller invoked the slash command with an explicit `<N>`, use `<N>`.
- Otherwise resolve via `gh pr view --json number`.

Fetch metadata once:

```bash
gh pr view <N> --repo "$(gh repo view --json nameWithOwner -q .nameWithOwner)" \
  --json number,headRefOid,baseRefOid,files,title,headRefName,mergeable,mergeStateStatus,statusCheckRollup
```

If no PR exists for the current branch, stop and tell the user to push and open a PR first.

## Step 2: Inspect PR health (merge conflicts + failing checks)

Reviewers need to know up front if the PR can't merge or has failing CI — both are independent of the diff and the fan-out sub-agents would never look for them. The orchestrator handles them; sub-agents don't need to know.

**Merge conflicts.** From the JSON in Step 1:

- `mergeable == "CONFLICTING"` → record one BLOCK line:
  ```
  BLOCK: <PR> — [pr-health] Merge conflict with base branch (mergeStateStatus=<value>). Rebase or merge base before review.
  ```
- `mergeable == "UNKNOWN"` → GitHub hasn't computed mergeability yet; skip (no finding).
- `mergeable == "MERGEABLE"` → no finding.

**Failing checks.** Parse `statusCheckRollup` from Step 1. Each entry is either a check run (has `conclusion`) or a legacy commit status (has `state`). A check is *failing* if any of these hold:

- `conclusion` ∈ {`FAILURE`, `TIMED_OUT`, `STARTUP_FAILURE`, `ACTION_REQUIRED`}.
- `state` ∈ {`FAILURE`, `ERROR`}.

Skip `SUCCESS`, `SKIPPED`, `NEUTRAL`, `CANCELLED`, and anything still pending/in-progress. For each failing entry record one BLOCK line:

```
BLOCK: <PR> — [pr-health] Failing check: <name> (<conclusion-or-state>) — <detailsUrl-or-targetUrl>
```

If `gh pr checks <N>` is easier than parsing the JSON, use it — but capture the same fields (name, fail reason, link). Hold these PR-health BLOCK lines aside; Step 6 folds them into the review body (they aren't anchored to diff lines).

## Step 3: Pick which skills to fan out to

Read the file list from Step 1. Map file types → relevant skills:

| File pattern                                                                | Skills to run                                           |
| --------------------------------------------------------------------------- | ------------------------------------------------------- |
| Always                                                                      | `code-health`, `synth-setter-project-standards`         |
| `*.py`                                                                      | `python-style`, `tdd-implementation`, `comment-hygiene` |
| `*.sh` or bash inside YAML `run:` blocks                                    | `shell-style`                                           |
| `.github/workflows/*.{yml,yaml}`                                            | `gha-workflow-validator`, `comment-hygiene`             |
| Any `*.{yml,yaml}` under `configs/`                                         | `comment-hygiene`                                       |
| `docs/doc-map.yaml`                                                         | `comment-hygiene`                                       |
| ML model / pipeline / training code under `src/synth_setter/`               | `ml-data-pipeline`, `ml-test`                           |
| Diff renames or moves files (anything with `R` in `git diff --name-status`) | `tdd-refactor`                                          |

Always run `code-health` and `synth-setter-project-standards`. Other skills opt in based on file extensions in the diff. `comment-hygiene` deduplicates: even if multiple rows above select it, fan out only one parallel agent per skill. Note which skills you selected; you'll launch one parallel agent per skill.

## Step 4: Launch parallel review agents

Launch one `general-purpose` Agent per selected skill. **All agents in a single message** so they run concurrently (an agent supports this — one message with N tool calls = N parallel agents).

Each agent's prompt MUST include:

- The PR number, repo, base SHA, head SHA.
- The full file list (with per-file line counts is helpful but optional).
- The exact skill to invoke: `Invoke the tinaudio-synth-setter-skills:<skill-name> skill via the Skill tool and apply its checklist to this PR's diff.`
- The expected output shape (see below).

Each agent returns a Markdown report with `BLOCK` and `WARN` sections. Each finding cites `<path>:<line>`. Agents work independently — they should not coordinate.

### Per-agent output contract

Each agent returns a Markdown block:

```
## <skill-name> review — PR #<N>

### BLOCK findings
1. **<path>:<line>** — <description>

### WARN findings
1. **<path>:<line>** — <description>

### What looks good
- ...
```

Aim each agent at a 1500-word ceiling so reports stay scannable. The orchestrator (you) can ask for tighter output if a skill's domain is small.

## Step 5: Aggregate findings

Once every parallel agent returns, parse each report's BLOCK and WARN findings. For each finding, build one entry in the findings JSON. Prefix each comment body with `[<skill>:<severity>]` so reviewers can see which checklist surfaced it.

Severity → severity tag:

- `BLOCK` → `block`
- `WARN` → `warn`

Skill → tag (short form for comment body):

| Plugin skill                     | Comment-body tag  |
| -------------------------------- | ----------------- |
| `code-health`                    | `code-health`     |
| `comment-hygiene`                | `comment-hygiene` |
| `shell-style`                    | `shell-style`     |
| `python-style`                   | `python-style`    |
| `gha-workflow-validator`         | `gha`             |
| `synth-setter-project-standards` | `synth-setter`    |
| `tdd-implementation`             | `tdd-impl`        |
| `tdd-refactor`                   | `tdd-refactor`    |
| `ml-data-pipeline`               | `ml-pipeline`     |
| `ml-test`                        | `ml-test`         |

A finding becomes:

```json
{
  "path": "<path>",
  "line": <line>,
  "body": "**[<short-tag>:<severity>]** <description>"
}
```

Do NOT dedupe near-duplicate findings across skills (e.g. shell-style and synth-setter-project-standards both flagging a `[ ]` vs `[[ ]]` issue) — keep each skill's signal independent. Acceptable noise per the plan's out-of-scope list.

## Step 6: Build the findings JSON

Same shape `post_review.py` consumes. **Fold the Step 2 PR-health BLOCKs into `review_body`** (they aren't anchored to diff lines, so they can't be inline comments). Prepend a `## PR health` section listing every PR-health BLOCK; if Step 2 produced nothing, omit the section entirely.

Transform each Step 2 BLOCK line into one bullet under `## PR health`: strip the `BLOCK: <PR> — ` prefix and prepend `- **[<calling-skill>:block]** `, leaving the `[pr-health] …` body unchanged. Substitute `<calling-skill>` with the calling skill's name (`repo-review-full` or `repo-review-full-no-comments`). For example, `BLOCK: 897 — [pr-health] Failing check: ci/test (FAILURE) — https://…` becomes `- **[repo-review-full:block]** [pr-health] Failing check: ci/test (FAILURE) — https://…` when called from `repo-review-full`.

```json
{
  "pr_number": <N>,
  "repo": "<owner>/<repo>",
  "review_body": "Multi-skill review of PR #<N> — <K> parallel passes (<list of skills>). Every BLOCK/WARN posted below as an individual unresolved thread. Findings on files outside the diff are anchored to the line in the diff that *causes* the staleness or rolled into the review body.\n\n## PR health\n\n- **[<calling-skill>:block]** [pr-health] Merge conflict with base branch (mergeStateStatus=DIRTY). Rebase or merge base before review.\n- **[<calling-skill>:block]** [pr-health] Failing check: ci/test (FAILURE) — https://github.com/.../runs/123",
  "findings": [ ... ]
}
```

The exact wording of `review_body` is up to the calling skill — `repo-review-full` writes the "posted below as an individual unresolved thread" phrasing; `repo-review-full-no-comments` writes a variant that says nothing was posted. Both reuse the `## PR health` bullet format.

When the calling skill submits via `post_review.py` (i.e. `repo-review-full`), add a top-level `"event"`: `REQUEST_CHANGES` if any finding is a BLOCK (any `[*:block]`, including the folded PR-health BLOCKs), else `COMMENT` if any WARN exists, else `APPROVE`. `repo-review-full-no-comments` renders to chat and never posts, so it omits `"event"`.

Write the JSON to a temp file:

```bash
cat > /tmp/<calling-skill>-findings.json <<'JSON'
... payload ...
JSON
```

Return to the calling skill's SKILL.md for the final delivery step.

## Notes

- This pipeline depends on the `tinaudio-synth-setter-skills` plugin being enabled. If a sub-skill invocation fails, surface the error — don't silently skip. Falling back to `repo-review` (MVP) is the user's call, not the skill's.
- Each parallel agent is a *general-purpose* sub-agent that itself invokes a plugin skill via the Skill tool. The two-level structure is intentional: the parallel fan-out is the orchestrator's contribution; each plugin skill's authoritative checklist is the source of truth for its domain.
- For the concrete invocation pattern (parallel Agent tool calls in a single message, expected per-agent prompt shape), see the example trace recorded in PR #777's review history — that's the workflow this pipeline packages.
