# repo-review-full analysis steps

Shared analysis pipeline for `repo-review-full` and `repo-review-full-no-comments`.
`repo-review-full` runs all of Steps 1–6 below. `repo-review-full-no-comments`
owns its own Steps 1–2 (so it can also review a local branch with no PR open) and
delegates only Steps 3–6 here. The final delivery step (Step 7 — post inline
comments vs. print the report to the user) differs and lives in each skill's
orchestrator brief.

"You" below is the Pi review orchestrator. Claude Code and Codex enter through
`agent/_shared/run_pi_review.sh`, so every harness uses the same flat Tintin
fan-out. Tintin intentionally removes its `Agent` tool from spawned subagents.

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
with Tintin's `Agent` tool using `subagent_type: "pr-review-worker"`,
`run_in_background: true`, and `max_turns: <plan.max_turns>` from that pass's
helper output. Tintin removes `Agent` from
subagents, so do not spawn a Pi orchestrator and then ask it to nest workers.
Launch all independent passes in one message, capture each start result's
`Agent ID` and `Output file`, then collect them with
`get_subagent_result(wait: true)`. Every pass receives the same complete worker
prompt and must not modify the checkout or GitHub state.

Give every worker the exact base SHA, head SHA, and changed paths. Require it to
inspect only `git diff <base>..<head> -- <changed-paths>` and explicit checklist
paths. It must never recursively discover files or checklists, search above the
current worktree, or inspect `.venv`, caches, dependencies, or sibling worktrees. An explicitly
assigned `tdd-refactor` pass may search tracked files with `git grep` and
`git ls-files`, as its exhaustive-reference contract requires. Every Bash tool
call has a 60-second timeout. A command timeout or
turn budget exhausted result is a failed attempt, never a partial success; add
its exact diagnostic to the audit and retry through the candidate sequence.

Run two model passes for every selected skill, then merge their reports using
the provenance and near-duplicate verification rules below. Pi does not run the
opencode launcher; it derives provenance from each successful effective model
as specified below. Record each attempt's skill,
pass, exact model, thinking level, Tintin agent id, status, and
the exact transcript path from the result's `Output file:` field in a
`## Pi review audit` section of `review_body`. This audit section does not change the findings JSON shape or inline-comment
contract. Render it as this table so the sentinel caller never has to inspect a
process tree or transcript to understand execution:

| Skill | Pass | Model | Thinking | Max turns | Status | Elapsed | Turns | Cumulative tokens | Agent ID | Transcript | Detail |
| ----- | ---- | ----- | -------- | --------: | ------ | ------: | ----: | ----------------: | -------- | ---------- | ------ |

Use explicit statuses: `success`, `unavailable`, `quota/capacity`,
`authentication`, `tool/checklist error`, `command timeout`, `turn budget exhausted`, `malformed report`, `verified`, or `rejected`. `Detail` contains the
exact failure diagnostic or allocation reason,
not a generic summary. Include one row per attempt, including retries and
verification passes. Precede the table with one sentence counting successful,
failed, retried, and rejected attempts. If the pipeline must fail closed, print the audit table before stopping even though no sentinel is written.

Use the tested routing helper as the single source of model candidates,
availability filtering, thinking levels, and allocation reasons. Count changed
lines and inspect the diff for these risk signals: file moves, concurrency,
persistence, authentication, workflow permissions, and numeric/dtype/shape
logic. Pass each detected signal with `--risk` and one `--skill` argument per
selected checklist:

```bash
python3 agent/_shared/pi_review_routing.py plan \
  --skill correctness-review --skill code-health \
  --changed-lines "$changed_lines" --risk concurrency
```

The command runs `pi --list-models` and returns JSON with two logical passes per
skill, the ordered available primary `candidates`, skipped `unavailable` models,
OpenRouter's same-provider `secondary_fallback_candidates`, cross-provider
Codex `fallback_candidates`, `thinking`, and `reason`. Codex is required.
OpenRouter is optional because its pass can degrade through the secondary
free-model tier and then the returned Codex fallback. Record skipped candidates
in the audit with no agent id or transcript.

Start each pass with its first candidate. If `Agent` reports HTTP `429`,
`quota`, `rate limit`, `resource exhausted`, `insufficient credits`,
`no endpoints available`, `provider unavailable`, or `Model not found`, record
the failure and launch a fresh worker with the next same-provider candidate.
Codex-pass candidates are always `openai-codex/*`; OpenRouter-pass candidates
are always `openrouter/*`. Exhaust the primary OpenRouter `candidates` first,
then continue through `secondary_fallback_candidates` before attempting any
Codex fallback. If an OpenRouter pass has no free candidates, or every primary
and secondary candidate exhausts quota/capacity, move the successful Codex pass's effective
model to the end of `fallback_candidates`, then launch a fresh worker
with the first model. This prefers a distinct fallback even when the Codex pass
reached its own fallback. Continue through that bounded Codex sequence only for
the same availability failures. Record each launch as `Codex fallback` in the
audit detail. Never resume a failed session under a different model.
Authentication, tool/checklist, malformed-report, timeout, and turn-budget
failures do not trigger the cross-provider fallback.

A completed worker is not successful until its final assistant Markdown passes
the report contract. The `Output file` is Tintin JSONL audit data, not Markdown;
extract its final assistant text deterministically, then validate that file:

```bash
python3 agent/_shared/pi_review_routing.py extract-report \
  <output-file> --output <report-path>
python3 agent/_shared/pi_review_routing.py validate-report \
  <report-path> --skill <skill-name> --target <PR-or-branch-label>
python3 agent/_shared/pi_review_routing.py transcript-stats <output-file>
```

Use `transcript-stats` for the audit's elapsed, turns, and cumulative-token
columns. The token number is explicitly cumulative processed context across
turns, not generated output, so label it exactly as the table does.

Do not copy the `get_subagent_result` envelope or feed the JSONL transcript
directly to `validate-report`. If extraction or validation fails, record
`malformed report` and try the next candidate. This
is a bounded report-quality retry, not a quota classification. Authentication,
tool, and checklist errors stop immediately. If a Codex pass exhausts its
candidates, stop before aggregation or delivery; never write a PASS sentinel
after silently dropping the required Codex pass. If an OpenRouter pass exhausts
its primary `candidates`, `secondary_fallback_candidates`, and bounded Codex
`fallback_candidates`, continue the review with Codex-only findings, record the
failed OpenRouter attempt chain in the audit, and add the exact sentence
`OpenRouter failed; only Codex ran.` to `review_body` so the posted review or
rendered report is explicit and truthful about the degraded coverage.

CI cannot exercise authenticated Tintin providers. Before opening a PR that
changes this flow, run both host harnesses against the PR from the worktree:

```bash
claude -p --dangerously-skip-permissions --no-session-persistence --model haiku \
  --effort low --output-format text \
  'Invoke repo-review-full-no-comments <PR> and wait for its foreground Pi launcher.'
codex exec --dangerously-bypass-approvals-and-sandbox \
  'Invoke repo-review-full-no-comments <PR> and wait for its foreground Pi launcher.'
```

After each command, verify the sentinel audit contains every planned
Codex/OpenRouter pass, bounded turn/runtime/token columns, an existing
transcript path for each launched attempt, and the current full HEAD from
`review_sentinel.py parse`. This live L1 smoke is mandatory in addition to
helper CLI tests.

- [ ] Record both authenticated host commands, their exit status, parsed
  sentinel HEAD, and fallback audit result in the PR verification comment.

Attribute findings from each successful report to the provider that actually
produced it, including after same-provider fallback:

```bash
python3 agent/_shared/pi_review_routing.py provenance <effective-model>
```

Merge duplicate findings using effective provenance. Findings independently
reported by Codex and OpenRouter are `both`; Codex-only findings are `codex`.
OpenRouter-only findings never enter aggregation directly. A logical
OpenRouter pass produced by a Codex fallback has Codex provenance and needs no
additional verification. For each skill that has any, launch one additional
`pr-review-worker` with the successful Codex
pass's effective model, `high` thinking, and the successful Codex pass's
`max_turns`, supplying the exact candidate findings and asking it to return only
those it can reproduce from the diff. This model has already passed availability
preflight; if the original Codex pass used a fallback, verification uses that
same effective fallback rather than a hard-coded selector.
Extract and validate that verification report through the same helper commands.
A confirmed candidate is tagged `openrouter; verified by: codex`; a rejected
candidate is omitted and recorded in the audit. If verification fails or is
malformed, stop rather than posting unverified free-model output. Add every
unavailable, failed, malformed, verified, rejected, and successful attempt to
the audit section.

Each worker's prompt MUST include:

- The PR number, repo, base SHA, head SHA.
- The full file list (with per-file line counts is helpful but optional).
- The exact skill to invoke: `Invoke the tinaudio-synth-setter-skills:<skill-name> skill via the Skill tool and apply its checklist to this PR's diff.` **Exception:** `lance-review` and `correctness-review` are repo-local, not plugin skills — instruct each agent to invoke the bare skill name (no `tinaudio-synth-setter-skills:` prefix), per the note just below. Do not emit the plugin-prefixed string for either.
- The expected output shape (see below).

`lance-review` and `correctness-review` are **repo-local**, not plugin skills:
each agent attempts the bare skill name via the Skill tool first, and only if
that call errors (the harness has not registered it) falls back to reading and
applying `agent/skills/<skill-name>/SKILL.md` directly. The per-agent output
contract below is unchanged for both.

`lance-review` additionally requires live documentation access. Its Pi worker
must fetch the required upstream pages through read-only Bash commands within
the same 60-second command deadline. `correctness-review` needs no web access.

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

Each finding ends with `codex`, `openrouter`, or `both` provenance.

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

The `findings` array carries every BLOCK and every WARN (each posts as its own inline unresolved thread). The exact wording of `review_body` is up to the calling skill — `repo-review-full` writes the "each finding posted below as an individual unresolved inline thread" phrasing; `repo-review-full-no-comments` writes a variant that says nothing was posted. Both reuse the same `## PR health` section format. When every OpenRouter path failed and only Codex-origin reports survived, prepend `OpenRouter failed; only Codex ran.` to the non-health portion of `review_body`.

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
- Claude Code and Codex both invoke the Pi-native main agent → flat Tintin
  worker structure. Every harness therefore uses the same checklist,
  aggregation, audit, and fallback contracts.
- For the concrete invocation pattern (parallel Agent tool calls in a single message, expected per-agent prompt shape), see the example trace recorded in PR #777's review history — that's the workflow this pipeline packages.
