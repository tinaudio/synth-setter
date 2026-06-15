# repo-review-full analysis steps

Shared analysis pipeline for `repo-review-full` and `repo-review-full-no-comments`.
`repo-review-full` runs all of Steps 1–6 below. `repo-review-full-no-comments`
owns its own Steps 1–2 (so it can also review a local branch with no PR open) and
delegates only Steps 3–6 here. The final delivery step (Step 7 — post inline
comments vs. print the report to the user) differs and lives in each skill's
orchestrator brief.

"You" below is the **orchestrator agent** the calling skill spawned to run this
whole pipeline — not the main agent, which only launches you and relays your
result.

You MUST complete every step below in order, then return to your orchestrator
brief's Step 7 for the final delivery step.

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
| `*.py` that imports or calls Lance (content-detected — see below)           | `lance-review`                                          |
| Diff renames or moves files (anything with `R` in `git diff --name-status`) | `tdd-refactor`                                          |

Always run `code-health` and `synth-setter-project-standards`. Other skills opt in based on file extensions in the diff. `comment-hygiene` deduplicates: even if multiple rows above select it, fan out only one parallel agent per skill. Note which skills you selected; you'll launch one parallel agent per skill.

`lance-review` opts in by **content, not extension** — a `*.py` file selects it
only when the diff touches the Lance API. Grep the changed Python files:

```bash
grep -lE 'import lance|lancedb|lance\.[a-z]|Lance[A-Z]|FragmentMetadata|write_dataset|\.scanner\(|\.to_batches\(|\.take\(|add_columns|merge_columns' <changed *.py>
```

This is the **same pattern** the skill itself uses to enumerate touch-points
(`agent/skills/lance-review/SKILL.md` Step 2) — they are kept identical so the
router never skips a file the skill would have findings for. If nothing matches,
skip `lance-review`.

## Step 4: Launch parallel review agents

Launch one `general-purpose` Agent per selected skill. **All agents in a single message** so they run concurrently (one message with N tool calls = N parallel agents). You are an orchestrator agent yourself, so these review agents are your sub-agents.

If your harness does not let a sub-agent spawn its own sub-agents, fall back to running each selected skill sequentially in your own context — invoke each `tinaudio-synth-setter-skills:<skill-name>` via the Skill tool one at a time. The fallback must produce the *same* per-skill result as the parallel path: feed each skill the same inputs the sub-agent prompt would carry (PR number, repo, base/head SHA, file list) and capture its findings in the per-agent output contract below (BLOCK/WARN sections, each finding citing `<path>:<line>`), so Step 5 aggregation parses sequential and parallel output identically. Parallel fan-out is preferred; the sequential fallback preserves correctness when nesting is unavailable.

Each agent's prompt MUST include:

- The PR number, repo, base SHA, head SHA.
- The full file list (with per-file line counts is helpful but optional).
- The exact skill to invoke: `Invoke the tinaudio-synth-setter-skills:<skill-name> skill via the Skill tool and apply its checklist to this PR's diff.`
- The expected output shape (see below).

`lance-review` is **repo-local**, not a plugin skill: its agent attempts the bare
`lance-review` skill via the Skill tool first, and only if that call errors (the
harness has not registered it) falls back to reading and applying
`agent/skills/lance-review/SKILL.md` directly. The
per-agent output contract below is unchanged. **Either path** requires web access
(`WebFetch`/`WebSearch`): the skill's grounding rule only holds if the agent can
fetch the live Lance docs, so spawn its sub-agent as `general-purpose` (which has
both tools) regardless of which invocation path it takes.

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

Once every parallel agent returns, parse each report's BLOCK and WARN findings. **BLOCK and WARN are routed differently** so a flood of low-severity WARNs can't bury the BLOCKs:

- **BLOCK findings** become entries in the `findings` JSON array (Step 6) — each posts as its own inline unresolved thread.
- **WARN findings** are *not* added to `findings`. They collapse into a single `## Advisory (WARN) findings` section appended to `review_body` (Step 6), one bullet each. This keeps the inline-thread list short enough that BLOCKs stay visible.

Prefix each finding body (BLOCK comment or WARN bullet) with the `[<skill>:<severity>]` scheme so reviewers can see which checklist surfaced it — using the short-tag form from the table below (`[<short-tag>:block]` / `[<short-tag>:warn]`), not the full skill name.

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
| `lance-review`                   | `lance`           |

A BLOCK finding becomes one entry in the `findings` array:

```json
{
  "path": "<path>",
  "line": <line>,
  "body": "**[<short-tag>:block]** <description>"
}
```

A WARN finding becomes one bullet in the `## Advisory (WARN) findings` section of `review_body` (built in Step 6), grouped by skill then by `path:line`:

```
- **[<short-tag>:warn]** <path>:<line> — <description>
```

Do NOT dedupe BLOCKs across skills (e.g. shell-style and synth-setter-project-standards both flagging a `[ ]` vs `[[ ]]` issue) — keep each BLOCK's signal independent, as its own inline thread.

**Dedupe WARNs only.** WARNs are collapsed into one advisory list, so near-duplicates are pure noise there. If the same `path:line` plus a near-identical description is flagged by multiple skills, keep one bullet and append `(also: <other-tags>)` listing the other skills' short tags. For example, if both `shell-style` and `synth-setter` flag the same `[ ]` vs `[[ ]]` at `scripts/run.sh:42`, emit a single bullet `- **[shell-style:warn]** scripts/run.sh:42 — Use [[ ]] for the test (also: synth-setter)`. This dedupe applies to WARNs only; never collapse BLOCKs.

## Step 6: Build the findings JSON

Same shape `post_review.py` consumes. The `findings` array holds **only BLOCK findings** (from Step 5) plus, indirectly, the PR-health BLOCKs folded into `review_body` below. **WARN findings do not appear in `findings`** — they live in the `## Advisory (WARN) findings` section of `review_body`.

`review_body` carries up to two appended sections, in this order: `## PR health` (Step 2 BLOCKs), then `## Advisory (WARN) findings` (the collapsed WARNs from Step 5). Omit either section if it would be empty.

**Fold the Step 2 PR-health BLOCKs into `review_body`** (they aren't anchored to diff lines, so they can't be inline comments). Prepend a `## PR health` section listing every PR-health BLOCK; if Step 2 produced nothing, omit the section entirely.

Transform each Step 2 BLOCK line into one bullet under `## PR health`: strip the `BLOCK: <PR> — ` prefix and prepend `- **[<calling-skill>:block]** `, leaving the `[pr-health] …` body unchanged. Substitute `<calling-skill>` with the calling skill's name (`repo-review-full` or `repo-review-full-no-comments`). For example, `BLOCK: 897 — [pr-health] Failing check: ci/test (FAILURE) — https://…` becomes `- **[repo-review-full:block]** [pr-health] Failing check: ci/test (FAILURE) — https://…` when called from `repo-review-full`.

**Append the collapsed WARNs under `## Advisory (WARN) findings`.** Emit one bullet per (deduped) WARN from Step 5, grouped by skill then by `path:line`, using the `- **[<short-tag>:warn]** <path>:<line> — <description>` shape. If Step 5 produced no WARNs, omit the section entirely.

```json
{
  "pr_number": <N>,
  "repo": "<owner>/<repo>",
  "review_body": "Multi-skill review of PR #<N> — <K> parallel passes (<list of skills>). Each BLOCK posted below as an individual unresolved thread; WARNs are collapsed into the `## Advisory (WARN) findings` section to keep the inline-thread list short. Findings on files outside the diff are anchored to the line in the diff that *causes* the staleness or rolled into the review body.\n\n## PR health\n\n- **[<calling-skill>:block]** [pr-health] Merge conflict with base branch (mergeStateStatus=DIRTY). Rebase or merge base before review.\n- **[<calling-skill>:block]** [pr-health] Failing check: ci/test (FAILURE) — https://github.com/.../runs/123\n\n## Advisory (WARN) findings\n\n- **[code-health:warn]** src/foo.py:42 — Function exceeds the length budget; consider extracting a helper.\n- **[comment-hygiene:warn]** src/foo.py:7 — Docstring restates the signature; tighten to the contract.",
  "findings": [ ... ]
}
```

The `findings` array now holds only BLOCK items (each posts as its own inline unresolved thread); WARNs live in `review_body`. The exact wording of `review_body` is up to the calling skill — `repo-review-full` writes the "each BLOCK posted below as an individual unresolved thread" phrasing; `repo-review-full-no-comments` writes a variant that says nothing was posted. Both reuse the `## PR health` and `## Advisory (WARN) findings` section formats.

When the calling skill submits via `post_review.py` (i.e. `repo-review-full`), add a top-level `"event"`: `REQUEST_CHANGES` if any finding is a BLOCK (any `[*:block]`, including the folded PR-health BLOCKs), else `COMMENT` if any WARN exists, else `APPROVE`. `repo-review-full-no-comments` renders to chat and never posts, so it omits `"event"`.

Write the JSON to a temp file:

```bash
cat > /tmp/<calling-skill>-findings.json <<'JSON'
... payload ...
JSON
```

Return to your orchestrator brief's Step 7 for the final delivery step.

## Notes

- Collapsing WARNs into a single advisory section (instead of one inline thread each) is intentional signal-preservation: posting every finding as its own thread trains reviewers to ignore the whole list, which buries the rare BLOCK that actually gates the merge.
- This pipeline depends on the `tinaudio-synth-setter-skills` plugin being enabled. If a sub-skill invocation fails, surface the error — don't silently skip. Falling back to `repo-review` (MVP) is the user's call, not the skill's.
- The structure is three-level and intentional: the main agent spawns one orchestrator agent (you), which fans out one *general-purpose* review sub-agent per skill; each review sub-agent invokes its plugin skill via the Skill tool. The orchestration is your contribution; each plugin skill's authoritative checklist is the source of truth for its domain.
- For the concrete invocation pattern (parallel Agent tool calls in a single message, expected per-agent prompt shape), see the example trace recorded in PR #777's review history — that's the workflow this pipeline packages.
