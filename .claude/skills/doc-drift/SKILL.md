______________________________________________________________________

## name: doc-drift description: > Detect when documentation has drifted out of sync with the codebase after code changes. Use this skill whenever the user asks to check if docs are up to date, review doc freshness, find stale documentation, audit reference docs, or asks "did this change break any docs?" Also trigger when the user mentions doc drift, doc rot, doc sync, doc consistency, or says things like "my docs might be out of date" or "check the docs after that refactor." Trigger even on indirect cues like "I just merged a big change" or "we renamed a bunch of files" — these are doc drift events. If the user asks to create or update a doc-map.yaml, that also triggers this skill.

# Doc Drift — Documentation Consistency Checker

Detect and fix documentation that has gone stale after code changes. Combines
two complementary strategies: **grep-based detection** (automatic, finds textual
references) and **structural mapping** (manual, catches drift by omission).

## When to use

- After merging code changes — "did this break any docs?"
- Periodic audits — "are my docs still accurate?"
- After refactors — "we renamed X to Y, what docs need updating?"
- When docs feel wrong — "something in the docker doc doesn't match reality"
- To bootstrap or update the structural mapping — "create/update doc-map.yaml"

______________________________________________________________________

## Core concepts

### Two detection strategies

**Grep-based detection** scans documentation files for references to source
paths, function names, CLI flags, env vars, and other identifiers that appear
in the code diff. This is automatic and zero-config, but only catches drift
when the doc already mentions the thing that changed.

**Structural mapping** (`doc-map.yaml`) encodes which source files *should* be
documented where — regardless of whether the doc currently references them.
This catches drift by omission: a new entrypoint mode that no doc mentions,
a config field that was added but never documented. The mapping file lives in
the user's repo and is itself subject to drift review.

Both strategies run together. Grep catches the common case (doc says X, code
changed X). The mapping catches the dangerous case (code added Y, no doc
mentions Y at all).

### Drift categories

1. **Stale reference** — doc describes behavior that the code no longer
   implements. Example: doc says MODE accepts `idle` and `passthrough`, but
   code now also supports `train`.

1. **Broken path** — doc references a file, function, class, or CLI flag that
   was renamed, moved, or deleted.

1. **Inconsistent value** — doc states a default, threshold, URL, or env var
   name that differs from the code.

1. **Omission** — source file is in the structural mapping's coverage set for
   a doc, but the doc doesn't mention new features/modes/flags added in the
   diff. This is only detectable via `doc-map.yaml`.

1. **Cross-doc inconsistency** — two docs describe the same thing differently.
   Example: rclone.md says MODE=train exists, docker.md doesn't mention it.

1. **Mapping staleness** — `doc-map.yaml` references files that no longer exist,
   or the repo has source files in mapped directories that aren't covered.

______________________________________________________________________

## Workflow

### Step 1: Determine scope

Ask: what changed? The answer determines the diff source.

| Trigger | Diff source |
|---|---|
| "Check after last merge" | `git diff HEAD~1 HEAD` |
| "Check after last N commits" | `git diff HEAD~N HEAD` |
| "Check after branch merged" | `git diff main..HEAD` or `git log --merges -1` |
| "Full audit" | No diff — review all mappings and all docs |
| "These specific files changed" | User-provided file list |

Run the diff and extract:

- Changed file paths
- Changed function/class/method names (from the diff hunks)
- Changed env var names, CLI flags, config keys (grep the diff for patterns)
- Renamed/moved/deleted files (`git diff --name-status` for R/D entries)

### Step 2: Find candidate docs

**Grep-based scan:**

```bash
# For each changed file, find docs that reference it
for f in $CHANGED_FILES; do
  basename=$(basename "$f")
  grep -rl "$basename" docs/ --include="*.md"
  # Also search by path fragments
  grep -rl "$f" docs/ --include="*.md"
done

# For renamed/deleted files, search for the OLD name
for old_name in $DELETED_OR_RENAMED; do
  grep -rl "$old_name" docs/ --include="*.md"
done

# For changed identifiers (env vars, flags, function names)
for ident in $CHANGED_IDENTIFIERS; do
  grep -rl "$ident" docs/ --include="*.md"
done
```

Deduplicate the results. These are docs with textual references to things
that changed — high probability of drift.

**Structural mapping scan:**

If `doc-map.yaml` exists, load it (see schema in `references/doc-map-schema.md`).
For each changed file, check whether it falls under any mapping entry's
`sources` patterns. If so, add the mapped `doc` to the candidate set — even
if the grep found nothing. This is how omission drift is detected.

Also check for **mapping staleness**: do any `sources` patterns in the YAML
match zero files in the current repo? If so, flag the mapping entry as
potentially stale.

### Step 3: Analyze each candidate doc

For each candidate doc, read it fully. Then for each changed file that
triggered it, read the relevant code (the current version, not the diff).

Check for each drift category:

1. **Stale references**: Does the doc describe behavior (modes, flags, defaults,
   workflows) that doesn't match the current code?

1. **Broken paths**: Does the doc reference files, functions, or identifiers
   that were renamed or deleted in the diff?

1. **Inconsistent values**: Does the doc state specific values (defaults, URLs,
   env var names, flag names) that differ from the code?

1. **Omissions** (mapping-triggered only): The structural mapping says this
   source file should be covered in this doc. Does the doc mention the
   features/modes/flags that the source file implements? If the diff added
   something new and the doc doesn't cover it, that's omission drift.

1. **Cross-doc inconsistency**: If multiple docs are in the candidate set and
   they describe overlapping topics, do they agree?

### Step 4: Report and propose fixes

For each drift instance found, report:

```
## Drift report

### <doc_path>

#### <drift_category>: <short_description>

**Source:** <source_file>:<line_range_or_function>
**Doc location:** <doc_file>, § <section_name>
**Issue:** <what's wrong>
**Suggested fix:** <concrete corrected text or action>
**Confidence:** high | medium | low
```

Confidence levels:

- **High**: The doc makes a specific factual claim that directly contradicts
  the code (e.g., lists 2 modes, code has 3).
- **Medium**: The doc is vague or incomplete in a way that *might* mislead
  (e.g., doesn't mention a new flag, but doesn't claim to be exhaustive).
- **Low**: Structural mapping suggests coverage, but the doc may intentionally
  omit the topic (e.g., an advanced feature documented elsewhere).

### Step 5: Act on findings

Depending on context:

**In Claude Code:** Apply fixes directly via `str_replace` on the doc files,
then commit to a `doc-drift/<date>` branch and open a PR. Group all fixes in
one PR, one commit per doc file.

**In Claude.ai:** Present the drift report and proposed fixes in conversation.
Offer to produce the corrected doc files.

**In a GitHub Action:** Output the drift report as a PR comment or open a
follow-up PR with fixes (see `references/github-action-integration.md`).

______________________________________________________________________

## Bootstrapping doc-map.yaml

If the repo doesn't have a `doc-map.yaml` yet, offer to create one.

### Auto-generation approach

1. Find all docs: `find docs/ -name "*.md" -type f`
1. For each doc, extract source file references (paths, script names, module
   names) via grep
1. For each referenced source file, find its parent directory
1. Generate mapping entries with the source directory as a glob pattern

This produces a first draft that captures existing textual cross-references.
The user should review and add structural relationships that aren't yet
referenced in the docs (this is the whole point — the mapping should be a
*superset* of what the docs currently reference).

### Placement

`doc-map.yaml` lives at the repo root (or `docs/doc-map.yaml`). It should be
version-controlled — changes to the mapping are reviewable like any other code.

______________________________________________________________________

## Edge cases and judgment calls

**Intentional omission vs. drift:** Some docs intentionally don't cover
everything. A "quickstart" doc that skips advanced modes isn't drifted — it's
scoped. Use confidence levels to distinguish. If the doc says "for the full
spec, see X," and X covers the topic, it's not an omission.

**Generated docs:** If a doc is auto-generated (e.g., from docstrings or a
build script), flag it but don't propose inline fixes — propose fixing the
generator input instead.

**Doc freshness markers:** If a doc has a "Last verified" date (like the rclone
and docker references in this project), update it when applying fixes.

**Large diffs:** If the diff touches 50+ files, don't try to analyze every
possible doc. Focus on high-confidence matches: structural mapping hits and
grep hits where the doc makes specific claims about the changed code. Report
that a full audit may be needed.

**Monorepo considerations:** If docs/ isn't the only doc location (READMEs in
subdirectories, inline doc comments, SKILL.md files), the grep scan should
cover the full tree, filtered to markdown files. The structural mapping can
specify non-standard doc locations.

______________________________________________________________________

## Reference files

- `references/doc-map-schema.md` — YAML schema for the structural mapping file,
  with annotated examples.
- `references/github-action-integration.md` — How to wire this into a
  post-merge GitHub Action (workflow YAML, prompt template, PR creation).

Read these when needed — they're supplementary to the core workflow above.
