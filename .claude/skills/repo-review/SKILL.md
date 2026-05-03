---
name: repo-review
description: |
  Quick PR review using the repo's core checklist (CLAUDE.md hard rules).
  Use when the tinaudio-synth-setter-skills plugin isn't available, or for a
  fast sanity-check before opening for full review. Single agent, no plugin
  dependency. Posts findings as individual unresolved inline PR review
  comments via .claude/skills/_shared/post_review.py.
---

# repo-review — MVP PR Review

You MUST complete every step in order.

## Step 1: Resolve the PR

Determine the PR number:

- If the user invoked `/repo-review <N>`, use `<N>`.
- Otherwise resolve via `gh pr view --json number` — runs against the current branch's PR.

Fetch the PR's metadata once and remember it:

```bash
gh pr view <N> --repo "$(gh repo view --json nameWithOwner -q .nameWithOwner)" \
  --json number,headRefOid,baseRefOid,files,title,headRefName
```

If there is no PR for the current branch, stop and tell the user to push and open a PR first.

## Step 2: Read the diff

Get the full unified diff:

```bash
gh pr diff <N> --repo <owner>/<repo>
```

Read every changed file at the head SHA (use the `Read` tool, not `cat`). Skim the PR description for context on intent.

## Step 3: Apply the core checklist

Evaluate ONLY the changed code. Skip items that don't apply to the diff (e.g. don't flag missing type annotations on a YAML-only PR).

For each finding emit one line:

```
BLOCK: <path>:<line> — [<category>] <description>
WARN:  <path>:<line> — [<category>] <description>
```

Categories: `comment-hygiene`, `yaml-bash`, `python`, `shell`, `pipeline`, `security`, `commit-style`, `pr-link`, `stale-ref`, `secret-doc`.

### The core checklist (sourced from CLAUDE.md)

**Comment hygiene (CLAUDE.md "Comment Hygiene" + "No Comments Inside YAML run: Block-Scalars")**

- [comment-hygiene] No comment restates a constant value, count, or list contents (`# 6 samples`, `# three things: a, b, c`).
- [comment-hygiene] No multi-paragraph essay-comments. If a comment runs more than ~2 lines, replace with a one-line pointer to a GitHub issue.
- [yaml-bash] **No `#`-comments inside `run: |` or `setup: |` block-scalars** in `.github/workflows/*.{yml,yaml}` or `configs/compute/*.yaml`. Comments belong ABOVE the `run:` key, not inside the bash. This is a HARD project rule — flag every occurrence as BLOCK.

**Python (CLAUDE.md "Writing Code" + plugin `python-style` BLOCK items)**

- [python] All function signatures are type-annotated. No `Any` — use `Union`, `Optional`, or specific types. (PY8)
- [python] No bare `except:`. Always catch a specific exception class. (PY2)
- [python] No mutable default arguments (`def f(x: list = [])` is a footgun — defaults are shared across calls). Use `None` + initialize inside. (PY3)
- [python] No `assert` for input validation or invariants — `python -O` strips them. Raise an explicit exception. (PY4)
- [python] Use `with` statements for every file / socket / resource open. No bare `open(...)` without `with`. (PY13)
- [python] Pydantic `BaseModel` with `strict=True` at trust boundaries (config parsing, JSON from R2, worker reports).
- [python] `structlog` for logging in pipeline code; Python's `logging` module elsewhere.
- [python] No `print()` statements in production code. CLI helpers + tests excepted (and exempted in `pyproject.toml` per-file-ignores). (P29)

**Shell (plugin `shell-style` BLOCK items — applies to `.sh` files AND bash inside YAML `run:` / `setup:` block-scalars in `.github/workflows/*.{yml,yaml}` and `configs/compute/*.yaml`)**

- [shell] `set -euo pipefail` at the top of every shell script and every YAML `run: |` / `setup: |` block-scalar. Inner `bash -c '...'` shells get their own `set -euo pipefail`. (SH1)
- [shell] All variable expansions are double-quoted: `"${VAR}"` not `$VAR`. Exceptions: integers and `$?`. (SH2)
- [shell] `[[ ]]` not `[ ]` for tests. Single-bracket is BLOCK. (SH3)
- [shell] Return values of every command are checked — failure of `var=$(cmd)` or `mkdir`, `printf`, `cd` does not silently continue. With `set -e` this is automatic; without it the assignment-substitution variant must be split or wrapped. (SH8)
- [shell] No `eval`. Refactor whatever needed `eval` into an array invocation or a function. (SH11)

**Pipeline (CLAUDE.md "Pipeline-Specific Rules" + plugin invariants)**

- [pipeline] All `rclone` operations include `--checksum`. (P13)
- [pipeline] No writes to `data/shards/` outside `finalize` stage. (P12)
- [pipeline] Workers only write under `metadata/workers/`. Finalize only writes to `data/`. (P11)
- [pipeline] Shard IDs are logical (`shard-000042`), deterministic, infrastructure-independent. No `pod_id` / `host` / `runner_id` in shard names. (P14)
- [pipeline] Specs are immutable after creation — no code path mutates a frozen spec in place. (P15)
- [pipeline] `.valid` marker is written as the last step of shard lifecycle (commit point). Earlier writes leave shards in a partial state with no marker. (P16)
- [pipeline] Array shapes match spec (sample rate, spectrogram bins, parameter count). dtypes are explicit — `float32` where expected, not `float64`. (P23, P24)

**Security (plugin `synth-setter-project-standards` security block)**

- [security] No credential leaks. API keys, tokens, OCIDs do not appear in code, logs, or error messages. Tracing-back-from-an-error to the secret value is also a leak. (P19)
- [security] No command injection via subprocess. User-controlled input never gets interpolated into a shell command — pass argv arrays, never `shell=True` with concatenated strings. (P20)
- [security] No unsafe deserialization. `pickle.loads` from untrusted sources is forbidden — use JSON / msgpack / protobuf for cross-trust-boundary data. (P21)

**Commit style (CLAUDE.md "Commit Messages")**

- [commit-style] Every commit on this branch uses a conventional-commit prefix (`feat:`, `fix:`, `internal-feat:`, `internal-fix:`, `docs:`, `ci:`, `chore:`, `refactor:`, `test:`, `style:`, `build:`, `monitoring:`, `perf:`, `revert:`).
- [commit-style] No commit carries a `Co-Authored-By:` trailer.
- [commit-style] No commit / PR body contains a "Generated with Claude Code" attribution.
- [commit-style] PR-level prefix matches the user-facing nature of the change (`feat:` for user-visible, `internal-feat:` for groundwork).

**PR link (CLAUDE.md "PR & Issue References")**

- [pr-link] PR body includes `Fixes #N`, `Closes #N`, `Refs #N`, or `Part of #N` linking to a taxonomy-compliant issue.
- [pr-link] Use `Refs #N` (not `Fixes`/`Closes`) for partial fixes / workarounds.

**Stale-reference audit (CLAUDE.md "Refactoring")**

- [stale-ref] After any rename / move, references in *all* file types are updated: `.py`, `.yaml`, `.yml`, `.toml`, `.json`, `.md`, `.sh`, Dockerfile. Run a grep to verify; flag any survivors.
- [stale-ref] If the PR renames a workflow artifact, every doc that names that artifact in a copy-paste command (`gh run download -n <name>`) is updated.

**Secret/input documentation**

- [secret-doc] Every secret the workflow reads (`secrets.X`) is enumerated in the workflow header comment AND in any operator-facing setup doc.
- [secret-doc] Every `inputs.X` referenced in steps is declared in `workflow_dispatch.inputs`.

End the listing with:

```
Summary: X BLOCK, Y WARN
```

If there are zero findings, output `PASS` and stop — do not post an empty review.

## Step 4: Build the findings JSON

Convert your BLOCK/WARN list to the JSON shape `post_review.py` consumes. Each finding becomes one inline comment with a `[repo-review:<severity>]` prefix.

```json
{
  "pr_number": <N>,
  "repo": "<owner>/<repo>",
  "review_body": "Repo-review (MVP): <X> BLOCK, <Y> WARN. Inline core checklist from CLAUDE.md.",
  "findings": [
    {
      "path": "<path>",
      "line": <line>,
      "body": "**[repo-review:block]** [<category>] <description>"
    }
  ]
}
```

Write the JSON to a temp file (do NOT echo it inline — keep it readable):

```bash
cat > /tmp/repo-review-findings.json <<'JSON'
... payload ...
JSON
```

## Step 5: Submit the review

```bash
python3 .claude/skills/_shared/post_review.py < /tmp/repo-review-findings.json
```

The helper:

- Fetches the PR diff and parses hunks.
- Anchors each finding to its target line if that line falls inside a diff hunk.
- Falls back to the nearest in-hunk line on the same file with a `*(anchored at line X; finding is on line Y, outside the diff hunks)*` cross-ref prepended to the body — preserves the original line number for the human reader.
- Rolls findings on files entirely outside the diff into a `## Findings on files outside the diff` section in the review body.
- Submits as `event=COMMENT` so threads stay unresolved without approving or rejecting.

The helper prints the review's `html_url` on success. Report it back to the user.

## Notes

- Severity threshold: post everything (every BLOCK and every WARN). Tuning to top-N or by-category is a follow-up — see issue #778's "Out of scope" list.
- Idempotency: re-running the skill on the same PR posts a fresh review with fresh comment threads (duplicates by design — easy to delete a whole review, fiddly to dedupe).
- Drift: this checklist is sourced from CLAUDE.md verbatim. When CLAUDE.md changes, this SKILL.md should change in the same PR — that's the contract.
