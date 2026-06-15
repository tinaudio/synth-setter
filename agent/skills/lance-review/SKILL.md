---
name: lance-review
description: |-
  Deep-scan the Lance documentation and review this repo's Lance (`pylance`)
  usage to confirm it follows upstream best practices and does not hand-roll
  code that Lance already provides natively. Every suggestion must be grounded
  in a verbatim doc quote and a deep link to the section it came from. Use when
  reviewing or writing code that imports `lance`/`lancedb`, calls
  `lance.write_dataset`, `lance.dataset`, `LanceFragment`, `LanceDataset`,
  `LanceOperation`, `.scanner(...)`, `.to_batches(...)`, `.take(...)`,
  `add_columns`/`merge_columns`, or builds/commits Lance fragments. Used by the
  `/repo-review-full` and `/repo-review-full-no-comments` fan-out.
---

# lance-review — Lance Best-Practices Review

Confirm Lance is used the way upstream documents it. Lance ships a large,
fast-moving native API; the failure mode this skill catches is **hand-rolling a
helper that duplicates a documented Lance primitive** — manual fragment-id
bookkeeping, custom row reordering, bespoke object-store plumbing, a
reimplemented scanner/filter, ad-hoc schema evolution, or a homegrown
versioning scheme. Each one is slower, less correct, and drifts from the format
guarantees Lance maintains.

**The grounding rule is hard: no finding ships without a verbatim quote from the
Lance docs and a deep link to the section it came from.** If you cannot fetch a
doc page that supports the suggestion, or cannot find the exact passage, you
**drop the finding** — do not emit an ungrounded "Lance probably has an API for
this." A suggestion the reviewer can't trace to a doc section is noise.

## Step 1: Pin the version and doc roots

Read the installed version before quoting anything — the API drifts release to
release.

```bash
grep -n 'pylance' pyproject.toml          # declared floor, e.g. pylance>=7.0.0
grep -nA2 'name = "pylance"' uv.lock      # resolved version actually installed
```

Canonical doc roots (fetch live; verify each page resolves before quoting):

| Source               | URL                                           | Use for                                         |
| -------------------- | --------------------------------------------- | ----------------------------------------------- |
| Lance guides         | `https://lancedb.github.io/lance/`            | Concepts, quickstart, blob/versioning/IO guides |
| Python API reference | `https://lancedb.github.io/lance-python-doc/` | Exact `lance.*` / `LanceDataset.*` signatures   |
| Format spec          | `https://lancedb.github.io/lance/format/`     | On-disk fragment/manifest guarantees            |

This repo uses **`pylance`** (the Lance columnar format), **not** `lancedb` the
vector database — quote `lance`/`LanceDataset` APIs, not `lancedb` table APIs,
unless the diff actually imports `lancedb`. Note the version you reviewed
against in your report so a future reviewer can re-check against a newer release.

## Step 2: Enumerate the Lance touch-points in the diff

Restrict to files in the PR/diff. Find every Lance interaction. `$BASE` and
`$HEAD` are the PR's base- and head-commit SHAs (from PR metadata, e.g.
`gh pr view <N> --json baseRefOid,headRefOid`, or otherwise set by the harness);
export them before running the snippet:

```bash
# --diff-filter=d drops deleted paths; the '*.py' pathspec keeps the scan on
# code so prose mentions of Lance APIs in docs/markdown don't false-positive
mapfile -t changed_py < <(git diff --name-only --diff-filter=d "$BASE"..."$HEAD" -- '*.py')
[[ ${#changed_py[@]} -gt 0 ]] && grep -nE 'import lance|lancedb|lance\.[a-z]|Lance[A-Z]|FragmentMetadata|write_dataset|\.scanner\(|\.to_batches\(|\.take\(|add_columns|merge_columns' -- "${changed_py[@]}"
```

This is the **same pattern** the fan-out router uses to decide whether to run
this skill at all (`agent/skills/_shared/repo-review-full-analysis.md` Step 3) —
keep the two in sync. It deliberately avoids bare `fragment`/`commit` tokens (they match
`fragment_id`, `git commit`, and other non-Lance code); `Lance[A-Z]` already
covers `LanceOperation` / `LanceDataset.commit` / `LanceFragment`.

For each touch-point, ask: **is there a native Lance API that already does
this?** The repo's existing, correct usage lives in
`src/synth_setter/pipeline/data/lance_shard.py`,
`src/synth_setter/cli/finalize_dataset.py`, and
`src/synth_setter/data/lance_datamodule.py` — use those as the in-repo baseline
for what "native" looks like (`lance.write_dataset`, `LanceFragment.create`,
`lance.LanceOperation.Overwrite` + `LanceDataset.commit`, `dataset.scanner(...)`,
`dataset.to_batches(...)`, `LanceDataset.take`, `storage_options=` for R2).

## Step 3: Hand-roll vs. native — what to look for

These are the recurring reinventions. For each, fetch the cited doc area, find
the primitive, and quote it in the finding.

| Smell in the diff                                                | Native Lance primitive to check the docs for                         |
| ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| Manual Arrow/Parquet file writing + directory layout             | `lance.write_dataset` (mode=`create`/`append`/`overwrite`)           |
| Hand-managed `fragment_id` sequencing / manual manifest assembly | `LanceFragment.create` + `LanceOperation` + `LanceDataset.commit`    |
| Custom row-reordering after a gather/index lookup                | `LanceDataset.take(indices)` (preserves requested order)             |
| Python-side filtering after reading all rows                     | `dataset.scanner(filter=..., columns=...)` predicate/column pushdown |
| Bespoke S3/GCS/R2 client wiring for dataset IO                   | `storage_options=` on `lance.dataset` / `write_dataset` / `commit`   |
| Reading whole columns to compute stats / iterate                 | `dataset.to_batches(columns=...)` streaming                          |
| Adding a column by rewriting the dataset                         | `add_columns` / `merge_columns` (schema evolution in place)          |
| Home-grown "latest version" / snapshot / rollback scheme         | dataset versioning: `dataset.version`, `checkout_version`, `tags`    |
| Manual large-binary side files next to the dataset               | blob columns / `take_blobs` (see the blob guide)                     |
| Custom random-sample / shuffle index over rows                   | `dataset.sample` / scanner sampling options                          |

A finding is only valid if (a) the diff genuinely reimplements the primitive and
(b) the primitive exists in the version under review. Confirm both before
writing it.

## Step 4: Output — grounded findings only

Return the standard fan-out report. **Severity:**

- **BLOCK** — the diff hand-rolls a primitive whose native form is a documented
  one-liner *and* the hand-roll risks correctness (wrong fragment order,
  silently dropped rows, version races, object-store retries Lance handles).
- **WARN** — a native API would be simpler/faster but the hand-roll is correct,
  or the call uses a deprecated/superseded API the docs flag.

Every BLOCK and WARN body MUST contain, in this order:

1. The `<path>:<line>` anchor.
2. One sentence naming the hand-roll and the native primitive that replaces it.
3. A **verbatim quote** (≤2 sentences) from the Lance docs, in `>` blockquote or
   inline quotes.
4. The **deep link** — the exact doc URL including the section anchor you pulled
   the quote from.

Example finding body:

> Builds a per-shard "latest" pointer file by hand at `path/to/shard_io.py:NN`;
> Lance tracks this natively via dataset versions. Docs: "Lance supports
> versioning of data. Each write operation creates a new version of the dataset."
> — https://lancedb.github.io/lance/format/#dataset-versioning

If you fetched the docs and the native API does **not** cover the case (e.g. the
repo's use is genuinely outside Lance's scope), say so under "What looks good"
rather than forcing a finding.

Report shape (same contract the fan-out expects):

```
## lance-review review — PR #<N>  (reviewed against pylance <version>)

### BLOCK findings
1. **<path>:<line>** — <hand-roll → native primitive>. Docs: "<verbatim quote>" — <deep link>

### WARN findings
1. **<path>:<line>** — <…>. Docs: "<verbatim quote>" — <deep link>

### What looks good
- <native-API usage worth keeping; cite the file>
```

## Review checklist

| #   | Check                    | Looks for                                                                 | Severity |
| --- | ------------------------ | ------------------------------------------------------------------------- | -------- |
| 1   | **Native writer**        | Datasets created via `write_dataset`/fragment+commit, not hand-laid files | BLOCK    |
| 2   | **Fragment integrity**   | Fragment ids/commit go through `LanceOperation`, not manual manifests     | BLOCK    |
| 3   | **Ordered take**         | Row gather uses `LanceDataset.take`, not Python-side reordering           | WARN     |
| 4   | **Pushdown**             | Filter/column selection pushed into `scanner`, not done after full read   | WARN     |
| 5   | **storage_options**      | Object-store IO uses `storage_options=`, not a bespoke client             | BLOCK    |
| 6   | **Streaming reads**      | Bulk reads use `to_batches`/scanner, not load-all-then-iterate            | WARN     |
| 7   | **Schema evolution**     | New columns via `add_columns`/`merge_columns`, not full rewrite           | WARN     |
| 8   | **Versioning**           | Snapshots/rollback use Lance versions/tags, not a homegrown pointer file  | WARN     |
| 9   | **Blob columns**         | Large binaries are blob columns / read via `take_blobs`, not side files   | WARN     |
| 10  | **Native sampling**      | Random sample/shuffle uses `dataset.sample`/scanner, not a custom index   | WARN     |
| 11  | **Version-matched docs** | Quoted API exists in the resolved `pylance` version                       | BLOCK    |
| 12  | **Grounded findings**    | Every finding carries a verbatim doc quote + deep link                    | BLOCK    |

BLOCK = must fix before merge · WARN = advisory.

## Notes

- The Lance API moves fast. A pattern that was hand-rolled correctly last year
  may now have a native API — and an API quoted from `latest` docs may not exist
  in the pinned version. Always reconcile the quote against the resolved version
  from Step 1.
- This skill reviews *how Lance is called*; it does not re-derive pipeline
  structure or types — `ml-data-pipeline`, `synth-setter-project-standards`, and
  `python-style` own those. Keep findings scoped to Lance-native vs. hand-rolled.
