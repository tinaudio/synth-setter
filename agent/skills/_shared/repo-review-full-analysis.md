# repo-review-full analysis steps

Shared analysis pipeline for `repo-review-full` and `repo-review-full-no-comments`.
`repo-review-full` runs all of Steps 1–6 below. `repo-review-full-no-comments`
owns its own Steps 1–2 (so it can also review a local branch with no PR open) and
delegates only Steps 3–6 here. The final delivery step (Step 7 — post inline
comments vs. print the report to the user) differs and lives in each skill's
orchestrator brief.

"You" below is the review orchestrator. Claude Code, Codex, and OpenCode spawn
a dedicated orchestrator agent. Under Pi, the main agent is the orchestrator
because Tintin intentionally removes its `Agent` tool from spawned subagents;
Pi therefore uses the flat fan-out in Step 4 instead of unsupported nesting.

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

| File pattern                                                                | Skills to run                                                         |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Always                                                                      | `code-health`, `synth-setter-project-standards`, `correctness-review` |
| `*.py`                                                                      | `python-style`, `tdd-implementation`, `comment-hygiene`               |
| `*.sh` or bash inside YAML `run:` blocks                                    | `shell-style`                                                         |
| `.github/workflows/*.{yml,yaml}`                                            | `gha-workflow-validator`, `comment-hygiene`                           |
| Any `*.{yml,yaml}` under `configs/`                                         | `comment-hygiene`                                                     |
| `docs/doc-map.yaml`                                                         | `comment-hygiene`                                                     |
| ML model / pipeline / training code under `src/synth_setter/`               | `ml-data-pipeline`, `ml-test`                                         |
| `*.py` that imports or calls Lance (content-detected — see below)           | `lance-review`                                                        |
| Diff renames or moves files (anything with `R` in `git diff --name-status`) | `tdd-refactor`                                                        |

Always run `code-health`, `synth-setter-project-standards`, and `correctness-review` — correctness is checked on every diff regardless of file type. Other skills opt in based on file extensions in the diff. `comment-hygiene` deduplicates: even if multiple rows above select it, fan out only one parallel agent per skill. Note which skills you selected; you'll launch one parallel agent per skill.

`lance-review` opts in by **content, not extension** — a `*.py` file selects it
only when the diff touches the Lance API. Grep the changed Python files:

```bash
# `read` (not `mapfile`) so macOS bash 3.2 works. Each line is one changed path.
changed_py=()
while IFS= read -r f; do changed_py+=("$f"); done < <(gh pr view <N> --repo "$(gh repo view --json nameWithOwner -q .nameWithOwner)" --json files -q '.files[].path' | grep '\.py$')
# PR file lists include deleted paths; keep only files present in the checkout
present=(); for f in "${changed_py[@]}"; do [[ -f $f ]] && present+=("$f"); done
[[ ${#present[@]} -gt 0 ]] && grep -lE 'import lance|lancedb|lance\.[a-z]|Lance[A-Z]|FragmentMetadata|write_dataset|\.scanner\(|\.to_batches\(|\.take\(|add_columns|merge_columns' -- "${present[@]}"
```

This is the **same pattern** the skill itself uses to enumerate touch-points
(`agent/skills/lance-review/SKILL.md` Step 2) — they are kept identical so the
router never skips a file the skill would have findings for. If nothing matches,
skip `lance-review`.

## Step 4: Launch parallel review agents

### Pi + Tintin

Pi uses a flat fan-out: the main agent runs Steps 1–7 and launches every pass
with Tintin's `Agent` tool using `subagent_type: "pr-review-worker"` and
`run_in_background: true`. Tintin removes `Agent` from subagents, so do not
spawn a Pi orchestrator and then ask it to nest workers. Launch all independent
passes in one message, capture each start result's `Agent ID` and `Output file`,
then collect them with `get_subagent_result(wait: true)`. Every pass receives
the same complete worker prompt and must not modify the checkout or GitHub
state.

Run two model passes for every selected skill, then merge their reports using
the existing union and near-duplicate rules in **Cross-model opencode pass**
below. For Pi, replace `native` provenance with `codex` and `opencode`
provenance with `openrouter`; do not run the opencode launcher. Record each
attempt's skill, pass, exact model, thinking level, Tintin agent id, status, and
the exact transcript path from the result's `Output file:` field in a
`## Pi review audit` section of `review_body`. This audit section does not
change the findings JSON shape or inline-comment contract.

Before fan-out, run `pi --list-models` once and parse its provider/model rows.
If either `openai-codex` or `openrouter` has no available models, stop with an
actionable authentication prerequisite (`/login <provider>` or set its API key
before starting Pi); missing authentication is not quota exhaustion. Remove an
unlisted candidate from its row before launch and record it as `unavailable`
with no agent id or transcript. If a row has no candidate left, stop. This lets
a retired free model fall through without hiding a provider setup error. If a
model disappears after preflight and `Agent` returns `Model not found`, apply
the same unavailable-candidate rule rather than classifying it as quota.

Choose the model candidates and base thinking level from this table. Start with
`Initial`; on a qualifying quota/capacity failure try `Fallback 1`, then
`Fallback 2`, each at most once with the same prompt and thinking level.

| Pass                | Skills                               | Initial                                             | Fallback 1                   | Fallback 2                                          | Base thinking |
| ------------------- | ------------------------------------ | --------------------------------------------------- | ---------------------------- | --------------------------------------------------- | ------------- |
| Codex deep          | `correctness-review`, `lance-review` | `openai-codex/gpt-5.6-sol`                          | `openrouter/openrouter/free` | `openrouter/nvidia/nemotron-3-super-120b-a12b:free` | `high`        |
| OpenRouter deep     | `correctness-review`, `lance-review` | `openrouter/nvidia/nemotron-3-super-120b-a12b:free` | `openai-codex/gpt-5.6-sol`   | `openrouter/openrouter/free`                        | `high`        |
| Codex standard      | Every other skill                    | `openai-codex/gpt-5.6-terra`                        | `openrouter/openrouter/free` | `openrouter/qwen/qwen3-coder:free`                  | `medium`      |
| OpenRouter standard | Every other skill                    | `openrouter/qwen/qwen3-coder:free`                  | `openai-codex/gpt-5.6-terra` | `openrouter/openrouter/free`                        | `medium`      |

Downgrade `comment-hygiene`, `python-style`, and `shell-style` to `low` on a
diff under 200 changed lines. Promote a pass one level, capped at `high`, when
the diff exceeds 800 changed lines or touches file moves, concurrency,
persistence, authentication, workflow permissions, or numeric/dtype/shape
logic. State the applicable condition in the audit row so the allocation is
reproducible.

If an `Agent` result fails with an out-of-quota or transient capacity signal —
HTTP `429`, `quota`, `rate limit`, `resource exhausted`, `insufficient credits`,
`no endpoints available`, or `provider unavailable` — follow that row's
fallbacks. The allocation is per pass, so mechanical checks stay cheap while
correctness-sensitive checks retain deep reasoning.

Never resume the failed session under another model. Add every failed and
successful attempt to the audit section. If all candidates fail, stop and
surface the quota/capacity error; silently dropping a checklist would violate
the review gate. Do not retry authentication, malformed-output, tool, or
checklist errors as quota failures.

### Claude Code, Codex, and OpenCode

Launch one named review Agent per selected skill. `correctness-review` uses
`pr-review-worker-deep`; all other selected skills use
`pr-review-worker-fast`. Under Claude Code, select those exact values with the
Agent tool's `subagent_type`. Under Codex, write each complete worker prompt to
a unique temporary file, then invoke
`agent/_shared/run_codex_review_agent.sh <role> --prompt-file <path>`.
That launcher resolves the role's project-pinned model and reasoning effort
before starting `codex exec`. **Launch all agents in a single message** so they
run concurrently (one message with N tool calls = N parallel agents). Do not
add any other per-invocation model override: each project agent file owns its
provider-specific model and effort. You are an orchestrator agent yourself, so
these review agents are your sub-agents.

If either named worker is unavailable or nested spawning is disabled, stop and
surface the configuration error. Do not fall back to an inherited, anonymous,
or sequential worker because that would bypass the gate's model policy.

Each agent's prompt MUST include:

- The PR number, repo, base SHA, head SHA.
- The full file list (with per-file line counts is helpful but optional).
- The exact skill to invoke: `Invoke the tinaudio-synth-setter-skills:<skill-name> skill via the Skill tool and apply its checklist to this PR's diff.` **Exception:** `lance-review` and `correctness-review` are repo-local, not plugin skills — instruct each agent to invoke the bare skill name (no `tinaudio-synth-setter-skills:` prefix), per the note just below. Do not emit the plugin-prefixed string for either.
- The expected output shape (see below).

`lance-review` and `correctness-review` are **repo-local**, not plugin skills:
each agent attempts the bare skill name via the Skill tool first, and only if
that call errors (the harness has not registered it) falls back to reading and
applying `agent/skills/<skill-name>/SKILL.md` directly. The per-agent output
contract below is unchanged for both.

`lance-review` additionally requires web access (`WebFetch`/`WebSearch`): its
grounding rule only holds if the agent can fetch the live Lance docs. The
`pr-review-worker-fast` role inherits the parent agent's tools, including web
access. `correctness-review` needs no web access and runs on the deep worker.

### Cross-model opencode pass (non-Pi workers)

Every non-Pi worker prompt MUST additionally instruct the worker to run a
second, parallel opencode pass of the same checklist and merge the two:

1. **Before anything else**, write an opencode prompt to a unique temp file
   (`mktemp`). It contains: the skill name and its checklist (the SKILL.md
   checklist text or a faithful summary), the PR/branch metadata (repo, base
   SHA, head SHA, file list), an instruction to inspect the diff via read-only
   git/gh commands, and the exact per-agent output contract below.
2. **Immediately launch the pass in the background** so it runs while the
   native pass works:
   `agent/_shared/run_opencode_review_agent.sh <your-role> --prompt-file <tmp>`
   (Claude Code: the Bash tool's background mode; otherwise append `&`).
   `<your-role>` is the worker role you are running as:
   `pr-review-worker-deep` for `correctness-review`, `pr-review-worker-fast`
   for everything else. The launcher owns model pinning and a hard timeout; do
   not pass model flags.
3. Run your native skill pass as specified above.
4. Collect the background result. If the launcher exited zero and produced a
   parseable report, **merge**: take the union of findings; collapse
   near-duplicates (same `<path>:<line>` and the same defect) into one
   finding. End every merged finding's description with its provenance:
   `(flagged by: native)`, `(flagged by: opencode)`, or `(flagged by: both)`.
   The merged report keeps the single-report contract and the 1500-word cap.
5. **Degrade + note.** If the launcher exits non-zero (opencode missing,
   timed out, errored) or returns nothing parseable, the native findings
   stand unchanged;
   mark every finding `(flagged by: native)` and append one final line to the
   report: `_opencode pass skipped/failed: <one-line reason>._` Never block,
   retry more than once, or drop native findings because the second pass
   failed. CI has no opencode CLI, so this path is normal there.

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

Each finding ends with cross-model provenance. Non-Pi workers use `native`,
`opencode`, or `both`; a degraded opencode pass tags findings `native` and adds
the required failure line. Pi uses `codex`, `openrouter`, or `both`.

## Step 5: Aggregate findings

Once every parallel agent returns, parse each report's BLOCK and WARN findings. **Both severities become entries in the `findings` JSON array** (Step 6) — each posts as its own inline unresolved thread. Posting WARNs inline (rather than collapsing them into a body bullet list) is deliberate: a bullet inside a long review body is easy to scroll past, while an unresolved inline thread forces an explicit reply or resolution before the PR ships. The severity tag on the comment body lets reviewers filter or batch-resolve, and `post_review.py` already keeps every thread unresolved.

Prefix each finding body with the `[<skill>:<severity>]` scheme so reviewers can see which checklist surfaced it — using the short-tag form from the table below (`[<short-tag>:block]` / `[<short-tag>:warn]`), not the full skill name.

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
| `correctness-review`             | `correctness`     |

Each finding (BLOCK or WARN) becomes one entry in the `findings` array, with `<severity>` set to `block` or `warn`:

```json
{
  "path": "<path>",
  "line": <line>,
  "body": "**[<short-tag>:<severity>]** <description>"
}
```

The shape is identical for both severities — only the tag changes. `post_review.py` anchors the entry, posts it as an inline review comment on the GitHub review, and leaves the thread unresolved.

Do NOT dedupe findings across skills (e.g. `shell-style` and `synth-setter-project-standards` both flagging the same `[ ]` vs `[[ ]]` issue at `scripts/run.sh:42`) — keep each finding's signal independent, as its own inline thread. Two short threads from two checklists tell the reviewer more than one merged thread that hides which checklist surfaced it; the cost of a near-duplicate inline comment is much smaller than the cost of misattributing a finding.

## Step 6: Build the findings JSON

Same shape `post_review.py` consumes. The `findings` array holds **every BLOCK and every WARN** from Step 5; the PR-health BLOCKs from Step 2 are folded into `review_body` separately because they aren't anchored to diff lines.

`review_body` carries one optional appended section, `## PR health` (Step 2 BLOCKs). Omit it if Step 2 produced nothing.

**Fold the Step 2 PR-health BLOCKs into `review_body`** (they aren't anchored to diff lines, so they can't be inline comments). Prepend a `## PR health` section listing every PR-health BLOCK; if Step 2 produced nothing, omit the section entirely.

Transform each Step 2 BLOCK line into one bullet under `## PR health`: strip the `BLOCK: <PR> — ` prefix and prepend `- **[<calling-skill>:block]** `, leaving the `[pr-health] …` body unchanged. Substitute `<calling-skill>` with the calling skill's name (`repo-review-full` or `repo-review-full-no-comments`). For example, `BLOCK: 897 — [pr-health] Failing check: ci/test (FAILURE) — https://…` becomes `- **[repo-review-full:block]** [pr-health] Failing check: ci/test (FAILURE) — https://…` when called from `repo-review-full`.

```json
{
  "pr_number": <N>,
  "repo": "<owner>/<repo>",
  "review_body": "Multi-skill review of PR #<N> — <K> parallel passes (<list of skills>). Each finding below (BLOCK or WARN) is posted as an individual unresolved inline thread so it can't be scrolled past without an explicit reply or resolution. Findings on files outside the diff are anchored to the line in the diff that *causes* the staleness or rolled into the review body.\n\n## PR health\n\n- **[<calling-skill>:block]** [pr-health] Merge conflict with base branch (mergeStateStatus=DIRTY). Rebase or merge base before review.\n- **[<calling-skill>:block]** [pr-health] Failing check: ci/test (FAILURE) — https://github.com/.../runs/123",
  "findings": [
    {"path": "src/foo.py", "line": 42, "body": "**[code-health:warn]** Function exceeds the length budget; consider extracting a helper."},
    {"path": "src/foo.py", "line": 7,  "body": "**[comment-hygiene:warn]** Docstring restates the signature; tighten to the contract."}
  ]
}
```

The `findings` array carries every BLOCK and every WARN (each posts as its own inline unresolved thread). The exact wording of `review_body` is up to the calling skill — `repo-review-full` writes the "each finding posted below as an individual unresolved inline thread" phrasing; `repo-review-full-no-comments` writes a variant that says nothing was posted. Both reuse the same `## PR health` section format.

When the calling skill submits via `post_review.py` (i.e. `repo-review-full`), add a top-level `"event"`: `REQUEST_CHANGES` if any finding is a BLOCK (any `[*:block]`, including the folded PR-health BLOCKs), else `COMMENT` if any WARN exists, else `APPROVE`. `repo-review-full-no-comments` renders to chat and never posts, so it omits `"event"`.

Write the JSON to a temp file:

```bash
cat > /tmp/<calling-skill>-findings.json <<'JSON'
... payload ...
JSON
```

Return to your orchestrator brief's Step 7 for the final delivery step.

## Notes

- WARN findings are posted inline (as their own unresolved threads) rather than collapsed into a body bullet list. The earlier collapse design optimized for keeping BLOCKs visible, but in practice body bullets were silently ignored — every review converged on `event=COMMENT` with zero inline threads, and the WARNs never got addressed. The inline form forces an explicit reply or resolution before merge under "Conversations must be resolved" branch protection, and the `[<short-tag>:<severity>]` prefix lets reviewers filter or batch-resolve.
- Most of this pipeline depends on the `tinaudio-synth-setter-skills` plugin being enabled; the repo-local `lance-review` and `correctness-review` skills are the standing exceptions (they run from `agent/skills/<name>/SKILL.md` even with the plugin absent, and `correctness-review` runs on every diff). If a sub-skill invocation fails, surface the error — don't silently skip. Falling back to `repo-review` (MVP) is the user's call, not the skill's.
- Claude Code, Codex, and OpenCode use the intentional three-level structure:
  main agent → orchestrator → named review workers. Pi uses main agent → flat
  Tintin workers because Tintin excludes nested `Agent` calls. Both paths use
  each plugin skill's authoritative checklist as the source of truth.
- Non-Pi workers internally run native + opencode; Pi launches codex +
  OpenRouter passes directly. Both paths return one merged report per skill,
  preserve the aggregation/tagging/sentinel contract, and dedupe only
  within-skill cross-model duplicates.
- For the concrete invocation pattern (parallel Agent tool calls in a single message, expected per-agent prompt shape), see the example trace recorded in PR #777's review history — that's the workflow this pipeline packages.
